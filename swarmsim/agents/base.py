"""Агент и абстракция источника движения (docs/02, разделы 2.1 и 5.1).

MotionSource — то, что двигает аппарат. В симуляции это KinematicMotion /
DoubleIntegratorMotion, позже — MavsdkMotion (PX4 SITL / реальный борт) с тем же
интерфейсом. Логика агента от этого не меняется.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..datatypes import AgentState, Mode, TaskSet, TaskStatus, VehicleState

if TYPE_CHECKING:
    from ..interfaces import SafetyFilter
    from .energy import RotaryWingEnergy


@dataclass(frozen=True)
class MotionCommand:
    """Уставка для MotionSource: либо «лети в точку», либо «держи скорость», либо «стоп»."""

    target: np.ndarray | None = None
    speed: float = 0.0
    velocity: np.ndarray | None = None

    @staticmethod
    def goto(target, speed: float) -> "MotionCommand":
        return MotionCommand(target=np.asarray(target, dtype=float), speed=speed)

    @staticmethod
    def set_velocity(v) -> "MotionCommand":
        return MotionCommand(velocity=np.asarray(v, dtype=float))

    @staticmethod
    def hold() -> "MotionCommand":
        return MotionCommand()


class MotionSource(ABC):
    _state: VehicleState

    @property
    def state(self) -> VehicleState:
        return self._state

    @abstractmethod
    def step(self, dt: float, cmd: MotionCommand) -> VehicleState:
        """Продвинуть аппарат на dt под командой cmd; вернуть новое состояние."""

    def halt(self) -> None:
        self._state.v = np.zeros(2)


EventSink = Callable[..., None]


class Agent:
    """Локальный цикл агента: следование плану, контроль энергии, возврат домой.

    План — упорядоченный список task_id; каждая задача — маршрут из путевых точек.
    """

    def __init__(self, agent_id: int, motion: MotionSource, energy: "RotaryWingEnergy",
                 capacity_j: float, reserve: float, home, v_cruise: float,
                 arrive_radius: float, e_takeoff: float, e_land: float,
                 log: EventSink | None = None, safety: "SafetyFilter | None" = None):
        self.id = agent_id
        self.motion = motion
        self.energy = energy
        self.capacity_j = capacity_j
        self.e_j = capacity_j
        self.reserve = reserve
        self.home = np.asarray(home, dtype=float)
        self.v_cruise = v_cruise
        self.arrive_radius = arrive_radius
        self.e_takeoff = e_takeoff
        self.e_land = e_land
        self._log = log or (lambda *a, **k: None)
        self.safety = safety

        self.mode = Mode.IDLE
        self.plan: list[int] = []
        self.current: int | None = None
        self.wp_idx = 0
        self.prev_p = self.p.copy()
        self.energy_used_j = 0.0
        self.distance_m = 0.0
        self.low_battery = False
        # P3: деградация, отставание от плана (SR-5), тихий отказ, режимы NORMAL/DEGRADED/NO_GCS/RTL
        self.v_nominal = v_cruise
        self.degraded = False
        self.undetected_failure = False      # отказал «тихо»: задачи висят, пока отказ не заметят
        self.gcs_ok = True
        self.ops_mode = "NORMAL"
        self._task_t0: float | None = None   # когда начата текущая задача
        self._task_len0 = 0.0                # её оставшаяся длина в момент начала
        self._deviation_reported = False
        # P4: в сетевом режиме решения «брать ли задачу» — по знанию агента (swarmsim.decentral);
        # None — по истинному состоянию (идеальная связь, как раньше)
        self.can_take = None

    # ------------------------------------------------------------ свойства
    @property
    def state(self) -> VehicleState:
        return self.motion.state

    @property
    def p(self) -> np.ndarray:
        return self.motion.state.p

    @property
    def alt(self) -> float:
        return self.motion.state.alt

    @property
    def airborne(self) -> bool:
        return self.mode in (Mode.MISSION, Mode.RTL)

    @property
    def healthy(self) -> bool:
        return self.mode != Mode.FAILED

    def snapshot(self) -> AgentState:
        s = self.motion.state
        return AgentState(id=self.id, p=s.p.copy(), v=s.v.copy(), psi=s.psi, alt=s.alt,
                          energy_j=self.e_j, capacity_j=self.capacity_j, healthy=self.healthy,
                          mode=self.mode, home=self.home.copy(), v_cruise=self.v_cruise,
                          plan=tuple(self.plan), current_task=self.current, wp_idx=self.wp_idx,
                          low_battery=self.low_battery)

    # ------------------------------------------------------------ управление
    def takeoff(self, t: float) -> None:
        if self.mode != Mode.IDLE:
            return
        self._consume(self.e_takeoff, t)
        if self.healthy:
            self.mode = Mode.MISSION
            self._log(t, "takeoff", agent=self.id)

    def set_plan(self, task_ids: list[int], tasks: TaskSet, t: float) -> None:
        """Принять новый план от распределителя. Задачи, выпавшие из плана, освобождаются."""
        if not self.healthy or self.mode == Mode.LANDED:
            return
        if self.can_take is None:
            new = [tid for tid in task_ids if tasks[tid].status not in (TaskStatus.DONE, TaskStatus.FAILED)]
        else:
            new = [tid for tid in task_ids if tid == self.current or self.can_take(tid, tasks)]
        if self.current is not None and self.current in new and new[0] != self.current:
            # приоритетное прерывание (FR-9): текущая задача уходит в очередь, прогресс сохранён
            tasks[self.current].status = TaskStatus.ASSIGNED
            self._log(t, "task_preempted", agent=self.id, task=self.current, by=new[0])
            self.current = None
        if self.current is not None and self.current not in new:
            if tasks[self.current].owner == self.id:
                self._release(self.current, tasks, t, "replanned")
            self.current = None
        for tid in self.plan:
            if tid not in new and tasks[tid].owner == self.id:
                self._release(tid, tasks, t, "replanned")
        if self.current is not None:
            new.remove(self.current)
        for tid in new:
            task = tasks[tid]
            task.owner = self.id
            if task.status not in (TaskStatus.IN_PROGRESS, TaskStatus.DONE):
                task.status = TaskStatus.ASSIGNED
        self.plan = new
        if self.mode == Mode.RTL and new and not self.low_battery:
            self.mode = Mode.MISSION
        self._log(t, "plan", agent=self.id, tasks=list(new),
                  current=self.current)

    def fail(self, t: float, reason: str, tasks: TaskSet, detect: str = "reported") -> list[int]:
        """Отказ агента.

        reported — бортовая диагностика успела сообщить: задачи освобождаются сразу;
        silent   — аппарат просто замолчал: его задачи «висят», пока отказ не заметят
                   по пропаже heartbeat (SimLoop вызовет release_all с причиной failure_detected).
        """
        if not self.healthy:
            return []
        released = self.release_all(tasks, t, reason) if detect == "reported" else []
        self.undetected_failure = detect == "silent"
        self.mode = Mode.FAILED
        self.motion.halt()
        extra = {"detect": detect} if detect != "reported" else {}
        self._log(t, "agent_failed", agent=self.id, reason=reason, released=released, **extra)
        return released

    def degrade(self, t: float, factor: float) -> None:
        """Деградация (двигатель, встречный ветер): крейсерская скорость падает в factor раз."""
        if not self.healthy:
            return
        self.v_cruise = self.v_nominal * factor
        self.degraded = True
        self._log(t, "degraded", agent=self.id, factor=factor, v_cruise=round(self.v_cruise, 3))
        self.update_mode(t)

    def update_mode(self, t: float) -> None:
        """Режимы docs/04 п. 2.5: NORMAL / DEGRADED / NO_GCS / RTL (+ FAILED, LANDED)."""
        if self.mode in (Mode.FAILED, Mode.LANDED):
            new = self.mode.value.upper()
        elif self.mode == Mode.RTL:
            new = "RTL"
        elif not self.gcs_ok:
            new = "NO_GCS"
        elif self.degraded:
            new = "DEGRADED"
        else:
            new = "NORMAL"
        if new != self.ops_mode:
            self._log(t, "mode_change", agent=self.id, frm=self.ops_mode, to=new)
            self.ops_mode = new

    def check_deviation(self, t: float, tasks: TaskSet, threshold: float) -> bool:
        """SR-5: факт отстаёт от плана больше порога → запросить перераспределение (один раз на задачу)."""
        if self.current is None or self._task_t0 is None or self._deviation_reported:
            return False
        task = tasks[self.current]
        w = task.waypoints[self.wp_idx:]
        rest = float(np.linalg.norm(np.diff(np.vstack([self.p, w]), axis=0), axis=1).sum()) if len(w) else 0.0
        done = max(self._task_len0 - rest, 0.0)
        lag = (t - self._task_t0) - done / self.v_nominal
        if lag > threshold:
            self._deviation_reported = True
            self._log(t, "plan_deviation", agent=self.id, task=self.current, lag_s=round(lag, 1))
            return True
        return False

    def local_repair(self, t: float, tasks: TaskSet, discount: float) -> list[int]:
        """Локальный ремонт (docs/05 P3): если на весь план не хватает энергии — отдать хвост.

        Проверка той же моделью маршрута, что у распределителей. Освобождаются задачи с конца
        очереди, пока остаток не станет допустимым; текущую задачу не трогаем (её охраняет
        _check_battery). Освобождённые задачи — повод для перераспределения.
        """
        if self.mode != Mode.MISSION or not self.plan:
            return []
        from ..planning.routing import RouteModel, TaskGeom, agent_context
        ctx = agent_context(self.snapshot(), tasks, self.energy, self.reserve, self.e_land)
        released = []
        while self.plan:
            geom = TaskGeom.build(tasks, self.plan)
            if RouteModel(geom, discount).evaluate(ctx, list(range(len(self.plan)))).feasible:
                break
            tid = self.plan.pop()
            if tasks[tid].owner == self.id and tasks[tid].status != TaskStatus.DONE:
                self._release(tid, tasks, t, "energy_repair")
                released.append(tid)
        return released

    def release_all(self, tasks: TaskSet, t: float, reason: str) -> list[int]:
        released = []
        for tid in ([self.current] if self.current is not None else []) + self.plan:
            if tasks[tid].owner == self.id and tasks[tid].status != TaskStatus.DONE:
                self._release(tid, tasks, t, reason)
                released.append(tid)
        self.current = None
        self.plan = []
        return released

    def _release(self, tid: int, tasks: TaskSet, t: float, reason: str) -> None:
        task = tasks[tid]
        if task.status == TaskStatus.DONE:          # выполненное не «развыполняется» (FR-8)
            return
        task.status = TaskStatus.UNASSIGNED
        task.owner = None
        self._log(t, "task_released", agent=self.id, task=tid, reason=reason)

    # ------------------------------------------------------------ шаг
    def step(self, t: float, dt: float, neighbors: list[AgentState], tasks: TaskSet) -> None:
        self.prev_p = self.p.copy()
        if not self.airborne:
            return

        self._check_battery(t, tasks)
        cmd = self._command(t, tasks)
        if self.safety is not None:
            cmd = self.safety.filter(self.snapshot(), cmd, neighbors)

        state = self.motion.step(dt, cmd)
        self.distance_m += float(np.linalg.norm(state.p - self.prev_p))
        self._consume(self.energy.power(state.speed) * dt, t + dt)
        if self.healthy:
            self._progress(t + dt, tasks)

    def energy_to_home(self) -> float:
        d = float(np.linalg.norm(self.home - self.p))
        return self.energy.energy_for_distance(d, self.v_cruise) + self.e_land

    def _check_battery(self, t: float, tasks: TaskSet) -> None:
        if self.mode != Mode.MISSION:
            return
        if self.e_j - self.energy_to_home() < self.reserve * self.capacity_j:
            self.low_battery = True
            released = self.release_all(tasks, t, "low_battery")
            self.mode = Mode.RTL
            self._log(t, "low_battery_rtl", agent=self.id, energy_j=round(self.e_j, 3),
                      released=released)

    def _command(self, t: float, tasks: TaskSet) -> MotionCommand:
        if self.mode == Mode.MISSION:
            while self.current is None and self.plan:
                tid = self.plan.pop(0)
                task = tasks[tid]
                if self.can_take is None:
                    if task.owner != self.id or task.status in (TaskStatus.DONE, TaskStatus.FAILED):
                        continue
                elif not self.can_take(tid, tasks):
                    continue
                if task.status != TaskStatus.DONE:          # истину «выполнено» не откатываем
                    task.status = TaskStatus.IN_PROGRESS
                task.owner = self.id
                task.t_start = t
                self.current, self.wp_idx = tid, min(task.progress, len(task.waypoints) - 1)
                w = task.waypoints[self.wp_idx:]
                self._task_t0 = t
                self._task_len0 = float(np.linalg.norm(np.diff(np.vstack([self.p, w]), axis=0), axis=1).sum())
                self._deviation_reported = False
                self._log(t, "task_started", agent=self.id, task=tid)
            if self.current is None:
                self.mode = Mode.RTL
                self._log(t, "plan_complete_rtl", agent=self.id)
            else:
                return MotionCommand.goto(tasks[self.current].waypoints[self.wp_idx], self.v_cruise)
        return MotionCommand.goto(self.home, self.v_cruise)

    def _progress(self, t: float, tasks: TaskSet) -> None:
        if self.mode == Mode.MISSION and self.current is not None:
            task = tasks[self.current]
            if self._reached(task.waypoints[self.wp_idx]):
                self.wp_idx += 1
                task.progress = max(task.progress, self.wp_idx)
                if self.wp_idx >= len(task.waypoints):
                    task.status = TaskStatus.DONE
                    task.times_done += 1
                    task.t_done = t
                    self._log(t, "task_done", agent=self.id, task=task.id)
                    self.current = None
        elif self.mode == Mode.RTL and self._reached(self.home):
            self.mode = Mode.LANDED
            self.motion.halt()
            self._consume(self.e_land, t)
            if self.healthy:
                self._log(t, "landed", agent=self.id, energy_j=round(self.e_j, 3))

    def _reached(self, target: np.ndarray) -> bool:
        d = self.p - target
        return math.hypot(d[0], d[1]) <= self.arrive_radius

    def _consume(self, joules: float, t: float) -> None:
        self.e_j -= joules
        self.energy_used_j += joules
        if self.e_j <= 0.0 and self.healthy:
            self.e_j = 0.0
            self.mode = Mode.FAILED
            self.motion.halt()
            self._log(t, "battery_depleted", agent=self.id)
