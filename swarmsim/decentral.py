"""Среда исполнения с реальным (не мгновенным) знанием — docs/05 P4.

Включается, когда связь моделируется как сеть (`comms.kind: network`) или распределитель
децентрализованный (`cbba`) / гибридный (`hybrid`). Тогда:

  • каждый агент и станция знают только то, что до них дошло (swarmsim.knowledge);
  • отказ замечают по пропаже heartbeat (тихий) или по сообщению об отказе (с сообщением),
    если оно дошло; ушедшего из зоны связи тоже «теряют» — это реальный источник дублей;
  • централизованный распределитель (B0–B3, центр B5) видит снимок станции, его планы идут
    агентам сообщениями с задержкой/потерями и повторяются раз в секунду до подтверждения;
  • CBBA (B4) — узлы на агентах обмениваются ставками по сети, центра нет;
  • гибрид (B5): при связи со станцией — её план, без связи — локальный CBBA по «сиротам».

Истинное состояние задач (для метрик и физики) остаётся в TaskSet; решения агентов и
станции принимаются по их знанию. Разрешение конфликтов исполнения: если два агента делают
одну задачу, уступает больший номер (то же правило, что у ничьей заявок в CBBA).
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from .alloc.cbba import CBBANode
from .datatypes import AgentState, Mode, Task, TaskSet, TaskStatus, WorldSnapshot
from .knowledge import DONE, FREE, RUNNING, KnowledgeBase
from .network import HB_BASE, Network
from .planning.routing import agent_context

if TYPE_CHECKING:
    from .sim import SimLoop

MODE_CODE = {Mode.IDLE: 0, Mode.MISSION: 1, Mode.RTL: 2, Mode.LANDED: 3, Mode.FAILED: 4}
CODE_MODE = {v: k for k, v in MODE_CODE.items()}
STATUS_OF = {FREE: TaskStatus.UNASSIGNED, 1: TaskStatus.ASSIGNED, RUNNING: TaskStatus.IN_PROGRESS,
             DONE: TaskStatus.DONE}
# события, которые станция узнаёт только от агента — если агент слышит станцию
AGENT_REPORTED = {"task_released", "low_battery_rtl", "plan_complete_rtl", "plan_deviation", "battery_depleted"}


class NetworkRuntime:
    def __init__(self, sim: "SimLoop", rng: np.random.Generator):
        self.sim = sim
        scn = sim.scenario
        self.n = len(sim.agents)
        params = dict(scn.comms.model_extra or {})
        homes = np.array([a.home for a in sim.agents])
        gcs_pos = np.asarray(params.get("gcs_pos", homes.mean(axis=0)), float)
        self.net = Network(self.n, gcs_pos, params, rng, ideal=scn.comms.kind == "ideal")
        self.kb = KnowledgeBase(self.n, len(sim.tasks), scn.faults.detect_timeout)
        self.gcs_timeout = float(params.get("gcs_timeout", 3.0))
        # обнаружение с учётом дальности: молчание вне зоны связи — не отказ (до hard_timeout)
        self.range_aware = scn.comms.kind != "ideal" and bool(params.get("range_aware_detect", True))
        self.hard_timeout = float(params.get("lost_hard_timeout", 60.0))
        # вести через соседей (gossip раз в секунду) стареют на ~1 с за переход
        self.relay_timeout = float(params.get("relay_timeout", scn.faults.detect_timeout + 2.5))
        self.expect_margin = float(params.get("expect_margin", 0.9))
        self.alloc = sim.allocator
        self.kind = "cbba" if getattr(self.alloc, "decentralized", False) else (
            "hybrid" if getattr(self.alloc, "hybrid", False) else "central")
        self.nodes: dict[int, CBBANode] = {}
        if self.kind in ("cbba", "hybrid"):
            L = self.alloc.bundle_L(len(sim.tasks), self.n)
            hyst = float(getattr(self.alloc, "params", {}).get("hysteresis", 0.05))
            self.nodes = {a.id: CBBANode(a.id, self.n, len(sim.tasks), L, hyst) for a in sim.agents}
        self.last_sent = np.full(self.n, -1e9)
        self.plan_version = 0
        self.agent_plan_version = np.zeros(self.n, dtype=int)
        self.pending_plans: dict[int, tuple[int, list[int], float]] = {}   # агент → (версия, план, последняя отправка)
        self.next_hb = 0.0
        self.gossip = bool(params.get("gossip", True))      # gossip-обнаружение отказов
        self.next_gossip = 0.0
        self.detected: set[int] = set()             # отказы, о которых кто-то уже узнал (для метрик)
        self.gcs_ok_prev = np.ones(self.n, dtype=bool)
        self.retired: set[int] = set()
        self.node_ms: list[float] = []              # настенное время одного пересчёта пакета на борту              # узлы, уже отказавшиеся от заявок (посадка, заряд)
        self.outage_done: list[float] = []
        self.ev_mark = 0
        for a in sim.agents:
            a.can_take = self._can_take_factory(a.id)

    # ---------------------------------------------------------------- решения агента по его знанию
    def _can_take_factory(self, aid: int):
        def can_take(tid: int, tasks: TaskSet) -> bool:
            st = self.kb.status[aid, tid]
            if st == DONE:
                return False
            ow = self.kb.owner[aid, tid]
            if st == RUNNING and ow not in (-1, aid) and self.kb.alive[aid, ow]:
                return False
            return True
        return can_take

    # ---------------------------------------------------------------- до вылета
    def pre_mission(self) -> None:
        sim = self.sim
        if self.kind == "cbba":
            self._preflight_cbba()
        else:
            sim._allocate(0.0, "initial")                     # на земле все на связи: план сразу
            self.plan_version += 1
            self.agent_plan_version[:] = self.plan_version

    def _preflight_cbba(self) -> None:
        """Торги на земле до взлёта: все рядом, связь полная — синхронные раунды до согласия."""
        sim = self.sim
        tasks = sim.tasks
        snap = sim.snapshot(0.0)
        cand = [t.id for t in tasks]
        rounds, stable, msgs = 0, 0, 0
        for rounds in range(1, 201):
            for a in snap.agents:
                ctx = agent_context(a, tasks, sim.energy, sim.scenario.agents.reserve, sim.scenario.energy.E_land)
                self.nodes[a.id].build(ctx, tasks, cand, sim.scenario.planning.discount)
            outbox = {i: nd.message() for i, nd in self.nodes.items()}
            before = {i: (nd.y.copy(), nd.z.copy()) for i, nd in self.nodes.items()}
            for i, nd in self.nodes.items():
                for k, (yk, zk, sk) in outbox.items():
                    if k != i:
                        nd.merge(k, float(rounds), yk, zk, sk)
                        msgs += 1
            changed = any(not (np.array_equal(before[i][0], nd.y) and np.array_equal(before[i][1], nd.z))
                          for i, nd in self.nodes.items())
            stable = 0 if changed else stable + 1
            if stable >= 2:
                break
        for a in sim.agents:
            nd = self.nodes[a.id]
            nd.dirty = True
            a.set_plan(list(nd.path), tasks, 0.0)
        assignment = {a.id: list(self.nodes[a.id].path) for a in sim.agents}
        sim.n_allocations += 1
        sim.events.log(0.0, "allocation", reason="initial", allocator="cbba",
                       assignment={str(k): v for k, v in sorted(assignment.items())},
                       explain={"rounds": rounds, "messages": msgs, "preflight": True})
        for nd in self.nodes.values():
            nd.s[:] = 0.0

    # ---------------------------------------------------------------- начало шага
    def step_begin(self, t: float) -> list[dict]:
        """Heartbeat, доставка сообщений, обнаружение. Вернуть события для центра (что узнала станция)."""
        sim = self.sim
        self._grow()
        pos = np.array([a.p for a in sim.agents])
        alive = np.array([a.healthy for a in sim.agents])
        self.net.update_positions(pos, alive)
        link = self.net.link_matrix(t)
        self._link = link
        if t + 1e-9 >= self.next_hb:
            self.next_hb = t + self.net.hb_period
            gossip = self.gossip and (t + 1e-9 >= self.next_gossip)
            if gossip:
                self.next_gossip = t + 1.0
            hb = [self._heartbeat(a, gossip) if a.healthy else None for a in sim.agents]
            self.kb.own_cur[:] = [-1 if (a.current is None or not a.healthy) else a.current for a in sim.agents]
            sizes = np.array([HB_BASE + 2 * len(a.plan) + (len(sim.tasks) + 7) // 8 + (4 * self.n if gossip else 0)
                              for a in sim.agents] + [8])
            got = self.net.heartbeat_receivers(t, link, sizes)
            self.kb.on_heartbeats(t, got, hb)
        for m in self.net.deliver(t):
            self._on_message(t, m)
        return self._detect(t)

    def _heartbeat(self, a, gossip: bool = False) -> dict:
        kb = self.kb
        done = np.flatnonzero(kb.status[a.id] == DONE)
        hb = {"p": a.p.copy(), "v": a.state.v.copy(), "e": a.e_j, "mode": MODE_CODE[a.mode],
              "cur": -1 if a.current is None else a.current, "wp": a.wp_idx, "lowbat": a.low_battery,
              "vc": a.v_cruise, "pv": int(self.agent_plan_version[a.id]), "plan": list(a.plan), "done": done}
        if gossip:
            hb["lh"] = kb.last_heard[a.id].copy()
        return hb

    def _on_message(self, t: float, m) -> None:
        sim = self.sim
        dst = m.dst
        if m.kind == "bids" and dst in self.nodes:
            yk, zk, sk, t_sent = m.payload
            self.nodes[dst].merge(m.src, t_sent, yk, zk, sk)
        elif m.kind == "plan" and dst < self.n:
            ver, plan = m.payload
            a = sim.agents[dst]
            if ver > self.agent_plan_version[dst] and a.healthy:
                self.agent_plan_version[dst] = ver
                keep = [j for j in plan if j == a.current or a.can_take(j, sim.tasks)]
                if self.kind == "hybrid" and dst in self.nodes:
                    self.nodes[dst].reset_claims()
                a.set_plan(keep, sim.tasks, t)
        elif m.kind == "event" and m.payload[0] == "failure":
            self.kb.last_heard[dst, m.payload[1]] = -1e9      # сообщение об отказе дошло

    def _timeouts(self, t: float) -> np.ndarray:
        """Сколько наблюдатель o ждёт вестей от a, прежде чем счесть его отказавшим:
        a (по последней известной позиции) в зоне связи самого o — detect_timeout;
        в зоне кого-то, кого o слышит, — relay_timeout (вести идут через gossip);
        вне зоны всех — hard_timeout: молчание объяснимо дальностью, это не отказ."""
        kb, net, n = self.kb, self.net, self.n
        direct = np.zeros((kb.o, n), dtype=bool)
        ex = np.zeros((kb.o, n), dtype=bool)
        r_ag = self.expect_margin * net.range
        r_g = self.expect_margin * net.range_gcs
        own = np.array([a.p for a in self.sim.agents])
        # неопределённость позиции: сколько агент мог пролететь с тех пор, как его слышали
        unc_all = np.linalg.norm(kb.vel, axis=-1) * np.maximum(t - kb.pos_t, 0.0)
        for o in range(kb.o):
            pos = kb.pos[o]
            unc = unc_all[o]
            if o < n:
                pos = pos.copy()
                pos[o] = own[o]
                unc = unc.copy()
                unc[o] = 0.0
            members = np.flatnonzero(kb.alive[o])
            if o < n:
                members = np.union1d(members, [o])
            if len(members):
                d = np.sqrt(((pos[members][:, None, :] - pos[None, :, :]) ** 2).sum(-1))
                ex[o] = (d + unc[members][:, None] + unc[None, :] <= r_ag).any(0)
            if o == kb.gcs:
                if not net.gcs_up(t):
                    ex[o] = False                    # станция знает, что её радио молчит
                    continue
                direct[o] = np.sqrt(((pos - net.gcs_pos) ** 2).sum(-1)) + unc <= r_g
            else:
                direct[o] = np.sqrt(((pos - pos[o]) ** 2).sum(-1)) + unc <= r_ag
        tmo = np.full((kb.o, n), self.hard_timeout)
        tmo[ex] = self.relay_timeout
        tmo[direct] = kb.timeout
        if not net.gcs_up(t):
            tmo[kb.gcs] = np.inf              # станция знает, что молчит её радио, а не агенты
        return tmo

    def _detect(self, t: float) -> list[dict]:
        sim = self.sim
        out = []
        tmo = self._timeouts(t) if self.range_aware else None
        for o, a, kind in self.kb.detect(t, timeouts=tmo):
            if kind == "lost":
                if a not in self.detected and not sim.agents[a].healthy:
                    self.detected.add(a)
                    ag = sim.agents[a]
                    released = ag.release_all(sim.tasks, t, "failure_detected") if ag.undetected_failure else []
                    ag.undetected_failure = False
                    sim.events.log(t, "failure_detected", agent=a, observer="gcs" if o == self.kb.gcs else o,
                                   released=released)
                if o < self.n and o in self.nodes:
                    self.nodes[o].release_peer(a)
                if o == self.kb.gcs:
                    if sim.agents[a].healthy:
                        sim.events.log(t, "peer_lost", agent=a, observer="gcs")   # ложная потеря (вне связи)
                    out.append({"t": t, "type": "failure_detected", "agent": a, "observer": "gcs"})
            else:
                if o < self.n and o in self.nodes:
                    self.nodes[o].peer_back(a)
                if o == self.kb.gcs:
                    sim.events.log(t, "peer_recovered", agent=a, observer="gcs")
                    out.append({"t": t, "type": "gcs_reconnect", "agent": a})
        # режим NO_GCS и возвращение связи
        for a in sim.agents:
            ok = self.kb.gcs_link(a.id, t, self.gcs_timeout)
            if a.healthy and ok and not self.gcs_ok_prev[a.id]:
                sim.events.log(t, "gcs_reconnect", agent=a.id)
                out.append({"t": t, "type": "gcs_reconnect", "agent": a.id})
            self.gcs_ok_prev[a.id] = ok
            a.gcs_ok = ok
        return out

    # ---------------------------------------------------------------- центр: снимок станции и доставка плана
    def gcs_snapshot(self, t: float) -> WorldSnapshot:
        self._grow()
        kb, g = self.kb, self.kb.gcs
        states = []
        for a in self.sim.agents:
            i = a.id
            mode = CODE_MODE[int(kb.mode[g, i])]
            cur = int(kb.cur[g, i])
            states.append(AgentState(
                id=i, p=kb.pos[g, i].copy(), v=kb.vel[g, i].copy(), psi=0.0, alt=a.alt,
                energy_j=float(kb.energy[g, i]), capacity_j=a.capacity_j,
                healthy=bool(kb.alive[g, i]) and mode != Mode.FAILED, mode=mode, home=a.home.copy(),
                v_cruise=float(kb.vcruise[g, i]) or a.v_cruise, plan=tuple(kb.plan_gcs[i]),
                current_task=None if cur < 0 else cur, wp_idx=int(kb.wp[g, i]), low_battery=bool(kb.lowbat[g, i])))
        return WorldSnapshot(t=t, agents=tuple(states), bounds=self.sim.world.bounds, nofly=self.sim.world.nofly)

    def gcs_tasks(self) -> TaskSet:
        self._grow()
        kb, g = self.kb, self.kb.gcs
        out = TaskSet()
        for tk in self.sim.tasks:
            st = int(kb.status[g, tk.id])
            ow = int(kb.owner[g, tk.id])
            out.add(replace(tk, status=STATUS_OF[st], owner=None if ow < 0 else ow,
                            progress=max(int(kb.progress[g, tk.id]), 0)))
        return out

    def dispatch(self, t: float, assignment: dict[int, list[int]]) -> None:
        """План станции → сообщения агентам (с повтором раз в секунду до подтверждения)."""
        self.plan_version += 1
        for aid, plan in assignment.items():
            self.pending_plans[aid] = (self.plan_version, list(plan), -1e9)
        self._resend(t)

    def _resend(self, t: float) -> None:
        link = self._link
        for aid, (ver, plan, last) in list(self.pending_plans.items()):
            if self.kb.plan_version[self.kb.gcs, aid] >= ver or not self.sim.agents[aid].healthy:
                del self.pending_plans[aid]
                continue
            if t - last >= 1.0:
                self.net.send(t, self.kb.gcs, [aid], "plan", (ver, plan), 8 + 2 * len(plan), link)
                self.pending_plans[aid] = (ver, plan, t)

    # ---------------------------------------------------------------- после шага агентов
    def _grow(self) -> None:
        """Появились новые задачи (FR-9) — расширить реплики и узлы до их числа."""
        m = len(self.sim.tasks)
        self.kb.grow(m)
        for nd in self.nodes.values():
            nd.grow(m)

    def after_agents(self, t: float) -> None:
        sim = self.sim
        tasks = sim.tasks
        self._grow()
        # свои действия агентов → своё знание (и узел CBBA)
        new = sim.events.since(self.ev_mark)
        self.ev_mark = len(sim.events)
        for e in new:
            k, aid = e["type"], e.get("agent")
            if k == "task_started":
                self.kb.local(aid, e["task"], RUNNING, aid)
                if aid in self.nodes:
                    self.nodes[aid].set_current(e["task"])
            elif k == "task_done":
                self.kb.local(aid, e["task"], DONE, aid)
                if self.net.gcs_up(t) is False:
                    self.outage_done.append(t)
                if aid in self.nodes:
                    self.nodes[aid].set_done(e["task"], aid)
            elif k == "task_released" and aid is not None and aid < self.n:
                self.kb.local(aid, e["task"], FREE, -1)
                if aid in self.nodes:
                    nd = self.nodes[aid]
                    j = e["task"]
                    if nd.z[j] == aid and nd.y[j] < 2e9:
                        nd.y[j], nd.z[j] = 0.0, -1
                    if nd.current == j:
                        nd.current = None
                    for lst in (nd.bundle, nd.path):
                        if j in lst:
                            lst.remove(j)
                    nd.changed = nd.dirty = True
            elif k == "agent_failed" and e.get("detect", "reported") == "reported":
                self.net.send(t, aid, "all", "event", ("failure", aid), 8, self._link)
            elif k == "low_battery_rtl" and aid in self.nodes:
                self.nodes[aid].reset_claims()
        for a in sim.agents:
            if a.healthy and a.current is not None:
                self.kb.local(a.id, a.current, RUNNING, a.id, a.wp_idx)
        # узнал, что мою текущую задачу уже выполнил другой (был вне связи) — бросить её
        for a in sim.agents:
            j = a.current
            if a.healthy and j is not None and self.kb.status[a.id, j] == DONE:
                self._yield(t, a, j, int(self.kb.owner[a.id, j]), reason="already_done")
        self._conflicts(t)
        if self.kind == "cbba":
            self._run_nodes(t, orphans_only=False)
        elif self.kind == "hybrid":
            self._run_nodes(t, orphans_only=True)
        if self.pending_plans:
            self._resend(t)

    def _yield(self, t: float, a, j: int, winner: int, reason: str = "conflict") -> None:
        """Уступить задачу j агенту winner: убрать из работы и из плана, запомнить победителя."""
        self.sim.events.log(t, "conflict_yield", agent=a.id, task=j, to=winner, reason=reason)
        if a.current == j:
            a.current = None
        a.plan = [x for x in a.plan if x != j]
        if self.kb.status[a.id, j] != DONE:
            self.kb.status[a.id, j], self.kb.owner[a.id, j] = RUNNING, winner
        nd = self.nodes.get(a.id)
        if nd is not None:
            if nd.current == j:
                nd.current = None
            for lst in (nd.bundle, nd.path):
                if j in lst:
                    lst.remove(j)
            nd.dirty = True

    def _conflicts(self, t: float) -> None:
        """Два агента выполняют одну задачу (узнали из heartbeat): уступает больший номер.

        Правило одно на всю систему и совпадает с ничьей заявок «выполняю» в CBBA — поэтому оба
        агента, даже с устаревшими сведениями друг о друге, приходят к одному выводу и не уступают
        одновременно (иначе задача остаётся ничьей). Прогресс не теряется: он общий у задачи."""
        sim = self.sim
        seen = set()
        for me, j, other, other_wp in self.kb.conflicts:
            a = sim.agents[me]
            if (me, j) in seen or a.current != j or not a.healthy:
                continue
            seen.add((me, j))
            if other < a.id:                 # то же правило, что у ничьей заявок RUN в CBBA
                self._yield(t, a, j, other)
        self.kb.conflicts.clear()

    def _run_nodes(self, t: float, orphans_only: bool) -> None:
        sim = self.sim
        tasks = sim.tasks
        scn = sim.scenario
        link = self._link
        m_bytes = 5 * len(tasks) + 4 * self.n
        for a in sim.agents:
            nd = self.nodes[a.id]
            active = a.healthy and a.mode in (Mode.IDLE, Mode.MISSION, Mode.RTL) and not a.low_battery
            if a.healthy and a.id not in self.retired and (a.low_battery or a.mode == Mode.LANDED):
                # ухожу из торгов (домой по заряду или сел): отказаться от заявок и сказать об этом
                self.retired.add(a.id)
                nd.reset_claims()
                yk, zk, sk = nd.message()
                sk[a.id] = t
                self.net.send(t, a.id, "all", "bids", (yk, zk, sk, t), m_bytes, link)
                self.last_sent[a.id] = t
                continue
            if orphans_only:
                active = active and not a.gcs_ok
            if not active:
                continue
            kb = self.kb
            for j in np.flatnonzero((kb.status[a.id, : len(nd.y)] == DONE) & (nd.y < 2e9)):
                nd.set_done(int(j), int(kb.owner[a.id, j]))
            nd.set_current(a.current)
            if nd.lost_current():
                self._yield(t, a, a.current, int(nd.z[a.current]))
            if nd.dirty:                     # пересчёт пакета — только когда что-то изменилось
                snap = a.snapshot()
                ctx = agent_context(snap, tasks, sim.energy, scn.agents.reserve, scn.energy.E_land)
                ctx.t0 += t      # абсолютное время миссии: ставки разных агентов и моментов сравнимы
                st = kb.status[a.id]
                ow = kb.owner[a.id]
                if orphans_only:
                    nd.base = [j for j in a.plan if j not in nd.bundle]
                    if not nd.path:
                        nd.path = list(a.plan)
                    cand = np.flatnonzero(st == FREE).tolist()
                else:
                    busy = (st == RUNNING) & (ow >= 0) & (ow != a.id)
                    busy &= kb.alive[a.id, np.where(ow >= 0, ow, 0)]
                    cand = np.flatnonzero((st != DONE) & ~busy).tolist()
                t_w = time.perf_counter()
                nd.build(ctx, tasks, cand, scn.planning.discount)
                self.node_ms.append(1000 * (time.perf_counter() - t_w))
                for j in [j for j in nd.path if j != a.current and j not in nd.base and not a.can_take(j, tasks)]:
                    nd.drop(j)                  # узел и агент должны видеть одно и то же
                path = [j for j in nd.path if j != a.current]
                if path != list(a.plan):
                    a.set_plan(([a.current] if a.current is not None else []) + path, tasks, t)
            due = t - self.last_sent[a.id] >= self.alloc.period
            if (nd.changed and t - self.last_sent[a.id] >= self.alloc.min_gap) or due:
                yk, zk, sk = nd.message()
                sk[a.id] = t
                self.net.send(t, a.id, "all", "bids", (yk, zk, sk, t), m_bytes, link)
                self.last_sent[a.id] = t

    # ---------------------------------------------------------------- итог
    def summary(self, t_final: float) -> dict:
        sim = self.sim
        s = self.net.stats(t_final)
        # N_lost: невыполненные задачи, которых в конце нет ни в чьём плане живого агента
        held = set()
        for a in sim.agents:
            if a.healthy:
                held |= set(a.plan) | ({a.current} if a.current is not None else set())
        s["N_lost"] = sum(1 for tk in sim.tasks if tk.status != TaskStatus.DONE and tk.id not in held)
        s["node_ms_mean"] = round(float(np.mean(self.node_ms)), 3) if self.node_ms else 0.0
        s["node_ms_max"] = round(float(np.max(self.node_ms)), 3) if self.node_ms else 0.0
        s["N_conflict_yield"] = sum(1 for e in sim.events.events if e["type"] == "conflict_yield")
        s["N_false_loss"] = sum(1 for e in sim.events.events if e["type"] == "peer_lost")
        # A_gcs: темп выполнения задач во время разрыва связи со станцией / вне разрыва
        outs = self.net.outages
        if outs:
            dur_in = sum(min(b, t_final) - min(a, t_final) for a, b in outs)
            done_t = [e["t"] for e in sim.events.events if e["type"] == "task_done"]
            n_in = sum(any(a <= x < b for a, b in outs) for x in done_t)
            t_last = max(done_t) if done_t else t_final
            dur_out = max(min(t_last, t_final) - dur_in, 1e-9)
            rate_in = n_in / dur_in if dur_in > 0 else float("nan")
            rate_out = (len(done_t) - n_in) / dur_out
            s["A_gcs"] = round(rate_in / rate_out, 6) if rate_out > 0 else None
            s["done_in_outage"] = n_in
        return s
