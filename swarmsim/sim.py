"""Детерминированный цикл симуляции (docs/02, раздел 4).

Порядок шага фиксирован:
  1) инъекция отказов → освобождение задач отказавших
  2) доставка сообщений (CommsBus)
  3) глобальное перераспределение — по событию (Allocator.should_trigger)
  4) локальный цикл агентов + обнаружение целей
  5) события мира, жёсткие ограничения (столкновения, геозоны), метрики

Детерминизм: фиксированный dt, время t = k·dt (без накопления ошибки), независимые
потоки RNG из одного SeedSequence, ничто не зависит от настенного времени.
"""
from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import alloc, comms, faults, missions  # noqa: F401  (регистрация встроенных плагинов)
from .agents import safety as _safety  # noqa: F401
from .agents.base import Agent
from .agents.energy import RotaryWingEnergy
from .agents.kinematics import DoubleIntegratorMotion, KinematicMotion
from .agents.safety import PassthroughSafety
from .datatypes import Assignment, Mode, Task, TaskSet, TaskStatus, WorldSnapshot
from .interfaces import ALLOCATORS, COMMS, FAULTS, MISSIONS, SAFETY, Allocator, lookup
from .decentral import AGENT_REPORTED
from .metrics import MetricsRecorder, recovery_metrics
from .planning.objective import mission_utility, sector_values
from .planning.validator import PlanValidator
from .scenario import Scenario
from .sensing import SensorModel
from .world import World

# Порядок потоков фиксирован; новые потоки добавлять ТОЛЬКО в конец —
# иначе изменятся миры для старых зёрен.
RNG_STREAMS = ("world", "sensing", "comms", "faults", "alloc")


class CollisionError(RuntimeError):
    pass


class AssignmentError(ValueError):
    pass


class EventLog:
    """Журнал событий («чёрный ящик», SR-4 / FR-10)."""

    def __init__(self):
        self.events: list[dict] = []

    def log(self, t: float, kind: str, **data) -> None:
        self.events.append({"t": round(float(t), 6), "type": kind, **data})

    def __len__(self) -> int:
        return len(self.events)

    def since(self, idx: int) -> list[dict]:
        return self.events[idx:]


@dataclass
class RunResult:
    summary: dict
    timeseries: list[dict]
    trajectories: list[dict]
    events: list[dict]
    manifest: dict = field(default_factory=dict)
    world: dict = field(default_factory=dict)      # статическая геометрия для replay (world.json)


def make_rngs(seed: int) -> dict[str, np.random.Generator]:
    children = np.random.SeedSequence(seed).spawn(len(RNG_STREAMS))
    return {name: np.random.default_rng(s) for name, s in zip(RNG_STREAMS, children)}


class SimLoop:
    def __init__(self, scenario: Scenario, seed: int, allocator: Allocator | None = None):
        self.scenario = scenario
        self.seed = int(seed)
        rngs = make_rngs(self.seed)
        self.events = EventLog()
        log = self.events.log

        self.world = World(scenario, rngs["world"], rngs["sensing"], log=log)
        self.sensor = SensorModel(scenario.sensor)
        self.energy = RotaryWingEnergy(scenario.energy)
        self.safety = lookup(SAFETY, scenario.safety.filter, "фильтр безопасности")(scenario)
        self.agents = self._build_agents()
        self._by_id = {a.id: a for a in self.agents}

        self.mission = lookup(MISSIONS, scenario.mission.kind, "тип миссии")(scenario)
        self.tasks = self.mission.decompose(self.world)
        if scenario.planning.value == "prior":         # Pd-планирование: ценность = вероятность целей
            for tid, v in sector_values(self.world, self.tasks).items():
                self.tasks[tid].priority = v

        ac = scenario.allocator
        self.allocator = allocator or lookup(ALLOCATORS, ac.kind, "распределитель")(
            scenario, dict(ac.model_extra or {}), rngs["alloc"])
        cc = scenario.comms
        self.comms = lookup(COMMS, "ideal" if cc.kind == "network" else cc.kind, "модель связи")(
            scenario, dict(cc.model_extra or {}), rngs["comms"])
        self.faults = lookup(FAULTS, scenario.faults.kind, "модель отказов")(scenario, rngs["faults"])
        self.metrics = MetricsRecorder(scenario.output.record_every, scenario.output.trajectories)
        self.n_allocations = 0
        if hasattr(self.allocator, "bind"):
            self.allocator.bind(self.world, self.energy)
        self.validator = PlanValidator(scenario, self.world, self.energy)
        self.alloc_wall_ms: list[float] = []
        self.plan_violations: dict[str, int] = {}
        self._pending_detect: list[tuple[float, int]] = []          # (когда заметят, агент) — тихие отказы
        self._pending_tasks = sorted(scenario.world.new_tasks, key=lambda e: e.t)
        # P4: реальное знание (сеть, реплики, CBBA/гибрид) — см. swarmsim/decentral.py.
        # Свой поток RNG (6-й потомок зерна): старые потоки и прогоны не меняются.
        self.runtime = None
        if (cc.kind == "network" or getattr(self.allocator, "decentralized", False)
                or getattr(self.allocator, "hybrid", False)):
            from .decentral import NetworkRuntime
            net_rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(len(RNG_STREAMS) + 1)[-1])
            self.runtime = NetworkRuntime(self, net_rng)
        # возмущения (ветер, порывы, разброс скорости): свой поток RNG (7-й потомок зерна),
        # включаются только realism.enabled — прежние прогоны не меняются
        self._wind = None
        rz = scenario.realism
        if rz.enabled:
            rrng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(len(RNG_STREAMS) + 2)[-1])
            ang = rrng.uniform(0, 2 * np.pi)
            self._wind = rrng.uniform(0, rz.wind_max) * np.array([np.cos(ang), np.sin(ang)])
            self._gust = np.zeros((len(self.agents), 2))
            self._rrng = rrng
            for a in self.agents:
                f = float(np.clip(rrng.normal(1.0, rz.speed_spread), 0.7, 1.3))
                a.v_cruise = a.v_nominal = a.v_cruise * f

    # ------------------------------------------------------------ построение
    def _build_agents(self) -> list[Agent]:
        c = self.scenario.agents
        e = self.scenario.energy
        _, _, w, h = self.world.bounds
        agents = []
        for i in range(c.count):
            home = np.array([c.home[0] + (i - (c.count - 1) / 2) * c.home_spacing, c.home[1]])
            home = np.clip(home, [0.0, 0.0], [w, h])
            alt = c.altitude + (i % c.altitude_layers) * c.altitude_step
            if c.motion == "kinematic":
                motion = KinematicMotion(home, alt, c.v_max)
            else:
                motion = DoubleIntegratorMotion(home, alt, c.v_max, c.a_max, np.radians(c.yaw_rate_max_deg))
            agents.append(Agent(
                agent_id=i, motion=motion, energy=self.energy, capacity_j=c.battery_wh * 3600.0,
                reserve=c.reserve, home=home, v_cruise=c.v_cruise, arrive_radius=c.arrive_radius,
                e_takeoff=e.E_takeoff, e_land=e.E_land, log=self.events.log,
                safety=None if isinstance(self.safety, PassthroughSafety) else self.safety))
        return agents

    # ------------------------------------------------------------ распределение
    def snapshot(self, t: float) -> WorldSnapshot:
        """Что видит распределитель. Тихо отказавший агент, пока отказ не замечен, выглядит
        живым: центр знает его последнее состояние, но не знает, что он молчит навсегда."""
        states = []
        for a in self.agents:
            s = a.snapshot()
            if a.undetected_failure:
                s = replace(s, healthy=True, mode=Mode.MISSION)
            states.append(s)
        return WorldSnapshot(t=t, agents=tuple(states), bounds=self.world.bounds, nofly=self.world.nofly)

    def _allocate(self, t: float, reason: str) -> None:
        snap = self.snapshot(t)
        w0 = time.perf_counter()
        assignment = self.allocator.allocate(snap, self.tasks)
        self.alloc_wall_ms.append(1000 * (time.perf_counter() - w0))
        self._check_assignment(assignment)
        # SR-6: план допустим только после проверки валидатором
        mode = self.scenario.safety.plan_validation
        violations = self.validator.check(snap, assignment, self.tasks) if mode != "off" else []
        for v in violations:
            self.plan_violations[v.kind] = self.plan_violations.get(v.kind, 0) + 1
            if v.kind != "layer":
                self.events.log(t, "plan_violation", **v.as_event())
        if mode == "repair" and any(v.kind == "energy" for v in violations):
            assignment, dropped = self.validator.repair(snap, assignment, self.tasks)
            if dropped:
                self.events.log(t, "plan_repaired", dropped=dropped)
        for a in self.agents:
            a.set_plan(list(assignment.get(a.id, [])), self.tasks, t)
        self.n_allocations += 1
        explain = {k: v for k, v in (getattr(self.allocator, "explain", None) or {}).items()
                   if k not in ("wall_ms", "wall_s")}
        if isinstance(explain.get("cpsat"), dict):
            explain["cpsat"] = {k: v for k, v in explain["cpsat"].items() if k != "wall_s"}
        self.events.log(t, "allocation", reason=reason, allocator=self.allocator.name,
                        assignment={str(k): list(v) for k, v in sorted(assignment.items())},
                        **({"explain": explain} if explain else {}))

    def _allocate_gcs(self, t: float, reason: str, view: TaskSet) -> None:
        """Централизованное распределение по знанию станции; план уходит агентам по сети."""
        snap = self.runtime.gcs_snapshot(t)
        w0 = time.perf_counter()
        assignment = self.allocator.allocate(snap, view)
        self.alloc_wall_ms.append(1000 * (time.perf_counter() - w0))
        mode = self.scenario.safety.plan_validation
        violations = self.validator.check(snap, assignment, view) if mode != "off" else []
        for v in violations:
            self.plan_violations[v.kind] = self.plan_violations.get(v.kind, 0) + 1
            if v.kind != "layer":
                self.events.log(t, "plan_violation", **v.as_event())
        if mode == "repair" and any(v.kind == "energy" for v in violations):
            assignment, dropped = self.validator.repair(snap, assignment, view)
            if dropped:
                self.events.log(t, "plan_repaired", dropped=dropped)
        self.n_allocations += 1
        explain = {k: v for k, v in (getattr(self.allocator, "explain", None) or {}).items()
                   if k not in ("wall_ms", "wall_s")}
        self.events.log(t, "allocation", reason=reason, allocator=self.allocator.name,
                        assignment={str(k): list(v) for k, v in sorted(assignment.items())},
                        **({"explain": explain} if explain else {}))
        self.runtime.dispatch(t, assignment)

    def _drift(self, a, dt: float) -> None:
        """Снос ветром и порывом (порыв — процесс Орнштейна–Уленбека); внутри границ поля."""
        rz = self.scenario.realism
        g = self._gust[a.id]
        g += -g * dt / rz.gust_tau + rz.gust_sigma * np.sqrt(2 * dt / rz.gust_tau) * self._rrng.normal(size=2)
        s = a.motion.state
        _, _, w, h = self.world.bounds
        s.p = np.clip(s.p + (self._wind + g) * dt, [0.0, 0.0], [w, h])

    def _sensed_neighbors(self) -> dict[int, list]:
        """Соседи для тактического избегания: в радиусе обзора и на том же эшелоне (снимки до шага)."""
        sf = self.scenario.safety
        air = [a for a in self.agents if a.airborne and a.healthy]
        out = {a.id: [] for a in self.agents}
        if len(air) < 2:
            return out
        P = np.array([a.p for a in air])
        Z = np.array([a.alt for a in air])
        close = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1) <= sf.sense_range ** 2
        close &= np.abs(Z[:, None] - Z[None, :]) < sf.vertical_separation
        np.fill_diagonal(close, False)
        idx = np.argwhere(close)
        if len(idx):
            snaps = {}
            for i, j in idx:
                s = snaps.get(j)
                if s is None:
                    s = snaps[j] = air[j].snapshot()
                out[air[i].id].append(s)
        return out

    def _spawn_task(self, t: float, ev) -> None:
        """Новая цель → задача осмотра точки: маленькая «змейка» вокруг неё (FR-9)."""
        from .missions.search import lawnmower
        h = ev.size / 2
        wps = lawnmower(ev.x - h, ev.y - h, ev.x + h, ev.y + h, ev.size / 3)
        tid = max((tk.id for tk in self.tasks), default=-1) + 1
        self.tasks.add(Task(id=tid, waypoints=wps, area=ev.size ** 2, priority=ev.priority,
                            t_appear=t, kind="point"))
        self.events.log(t, "new_task", task=tid, priority=ev.priority, x=ev.x, y=ev.y)

    def _check_assignment(self, assignment: Assignment) -> None:
        seen: set[int] = set()
        for aid, tids in assignment.items():
            agent = self._by_id.get(aid)
            if agent is None:
                raise AssignmentError(f"нет агента {aid}")
            if tids and not agent.healthy and not agent.undetected_failure:
                raise AssignmentError(f"задачи назначены отказавшему агенту {aid}")
            for tid in tids:
                if tid not in self.tasks:
                    raise AssignmentError(f"нет задачи {tid}")
                if tid in seen:
                    raise AssignmentError(f"задача {tid} назначена дважды")
                if self.tasks[tid].status == TaskStatus.DONE:
                    raise AssignmentError(f"задача {tid} уже выполнена")
                seen.add(tid)

    # ------------------------------------------------------------ цикл
    def run(self) -> RunResult:
        scn = self.scenario
        dt = scn.dt
        n_steps = int(round(scn.t_end / dt))
        wall0 = time.perf_counter()

        if self.runtime is not None:
            self.runtime.pre_mission()
        else:
            self._allocate(0.0, "initial")
        for a in self.agents:
            a.takeoff(0.0)
        self.metrics.record(0.0, self.world, self.agents, self.tasks)
        ev_mark = len(self.events)
        use_neighbors = not isinstance(self.safety, PassthroughSafety)
        repair_every = max(1, int(round(scn.agents.repair_period / dt)))
        dev_every = max(1, int(round(1.0 / dt)))          # SR-5: отставание проверяем раз в секунду

        k = 0
        while k < n_steps:
            t = k * dt

            # 1) отказы и деградации
            for item in self.faults.inject_due(t, self.agents):
                aid, reason = item[0], item[1]
                spec = item[2] if len(item) > 2 else None
                a = self._by_id.get(aid)
                if a is None:
                    continue
                if spec is not None and spec.type == "degrade":
                    a.degrade(t, spec.factor)
                    continue
                detect = spec.detect if spec is not None else "reported"
                a.fail(t, reason, self.tasks, detect)
                if detect == "silent" and self.runtime is None:     # в сети — по heartbeat (decentral)
                    self._pending_detect.append((t + scn.faults.detect_timeout, aid))
            # 1б) тихие отказы замечают по пропаже heartbeat
            for td, aid in [p for p in self._pending_detect if p[0] <= t + 1e-9]:
                self._pending_detect.remove((td, aid))
                a = self._by_id[aid]
                a.undetected_failure = False
                released = a.release_all(self.tasks, t, "failure_detected")
                self.events.log(t, "failure_detected", agent=aid, observer="gcs",
                                delay=round(scn.faults.detect_timeout, 3), released=released)
            # 1в) новые задачи (FR-9)
            while self._pending_tasks and self._pending_tasks[0].t <= t + 1e-9:
                self._spawn_task(t, self._pending_tasks.pop(0))

            # 2) связь
            self.comms.step(t, self.agents)
            new_events = self.events.since(ev_mark)
            ev_mark = len(self.events)

            # 3) перераспределение по событию
            if self.runtime is None:
                if self.allocator.should_trigger(t, self.tasks, new_events) and self.comms.gcs_reachable():
                    self._allocate(t, "trigger")
            else:
                gcs_events = self.runtime.step_begin(t)
                if self.runtime.kind in ("central", "hybrid"):
                    known = gcs_events + [e for e in new_events if e["type"] == "new_task" or (
                        e["type"] in AGENT_REPORTED and e.get("agent") is not None
                        and e["agent"] < len(self.agents) and self.agents[e["agent"]].gcs_ok)]
                    view = self.runtime.gcs_tasks()
                    if self.allocator.should_trigger(t, view, known):
                        self._allocate_gcs(t, "trigger", view)

            # 4) агенты + обнаружение
            near = self._sensed_neighbors() if use_neighbors else None
            for a in self.agents:
                nb = near[a.id] if near else []
                a.step(t, dt, nb, self.tasks)
                if self._wind is not None and a.airborne and a.healthy:
                    self._drift(a, dt)
                self.world.observe(a, self.sensor)
                if not a.healthy and not a.undetected_failure and (a.current is not None or a.plan):
                    a.release_all(self.tasks, t + dt, "battery_depleted")
                if a.healthy and a.airborne and k % dev_every == 0:
                    a.check_deviation(t + dt, self.tasks, scn.agents.deviation_threshold)
                    if scn.agents.local_repair and k % repair_every == 0:
                        a.local_repair(t + dt, self.tasks, scn.planning.discount)
                a.update_mode(t + dt)
            if self.runtime is not None:
                self.runtime.after_agents(t + dt)

            # 5) мир, жёсткие ограничения, метрики
            self.world.step(dt)
            new_col, _ = self.world.check_separation(self.agents)
            self.world.check_geofence(self.agents)
            if new_col and scn.safety.strict:
                raise CollisionError(f"t={self.world.t:.2f}: столкновение {new_col}")

            k += 1
            self.metrics.record(k * dt, self.world, self.agents, self.tasks)
            if not any(a.airborne for a in self.agents):
                break

        t_final = k * dt
        wall = time.perf_counter() - wall0
        extra = {
            "scenario": scn.name, "seed": self.seed, "allocator": self.allocator.name,
            "motion": scn.agents.motion, "n_agents": len(self.agents),
            "config_hash": scn.config_hash()[:16], "steps": k, "allocations": self.n_allocations,
            **self.comms.stats(),
        }
        summary = self.metrics.finalize(t_final, self.world, self.agents, self.tasks, extra)
        summary.update(mission_utility(self.tasks, summary["E_total_Wh"], scn.objective))
        summary.update(recovery_metrics(self.events.events))
        if self.runtime is not None:
            summary.update(self.runtime.summary(t_final))
        summary["N_conf"] = int(getattr(self.safety, "n_conf", 0))
        summary["N_plan_energy"] = self.plan_violations.get("energy", 0)
        summary["N_plan_geo"] = self.plan_violations.get("geofence", 0)
        summary["alloc_ms_mean"] = round(float(np.mean(self.alloc_wall_ms)), 3) if self.alloc_wall_ms else 0.0
        summary["alloc_ms_max"] = round(float(np.max(self.alloc_wall_ms)), 3) if self.alloc_wall_ms else 0.0
        summary["wall_time_s"] = round(wall, 4)
        summary["realtime_factor"] = round(t_final / wall, 2) if wall > 0 else None
        return RunResult(summary=summary, timeseries=self.metrics.timeseries,
                         trajectories=self.metrics.trajectories, events=self.events.events,
                         manifest=self._manifest(), world=self._world_dump())

    def _world_dump(self) -> dict:
        """Геометрия прогона, которой нет в CSV: цели, маршруты задач, дома, зоны."""
        w = self.world
        return {
            "bounds": list(w.bounds),
            "nofly": [p.tolist() for p in w.nofly],
            "homes": {str(a.id): [float(a.home[0]), float(a.home[1])] for a in self.agents},
            "tasks": [{"id": t.id, "waypoints": np.round(t.waypoints, 3).tolist(),
                       "t_start": t.t_start, "t_done": t.t_done, "owner": t.owner}
                      for t in self.tasks],
            "targets": [{"id": i, "x": round(float(p[0]), 3), "y": round(float(p[1]), 3),
                         "t_appear": float(w.target_appear[i]),
                         "t_detected": None if np.isnan(w.target_t_detected[i])
                         else round(float(w.target_t_detected[i]), 6)}
                        for i, p in enumerate(w.target_pos)],
        }

    def _manifest(self) -> dict:
        import pydantic

        from . import __version__
        return {
            "scenario": self.scenario.name,
            "seed": self.seed,
            "allocator": self.allocator.name,
            "config_hash": self.scenario.config_hash(),
            "rng_streams": list(RNG_STREAMS),
            "swarmcore_version": __version__,
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pydantic": pydantic.VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": self.scenario.model_dump(mode="json"),
        }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run(scenario: Scenario, seed: int, allocator: Allocator | None = None) -> RunResult:
    """run(scenario, seed) -> RunResult — точка входа ядра (docs/02, SimLoop)."""
    return SimLoop(scenario, seed, allocator=allocator).run()
