"""Знание о миссии у каждого наблюдателя (docs/01 §1: «агент располагает копией состояния миссии»).

Наблюдатели: агенты 0..N−1 и наземная станция N. Каждый знает только то, что до него дошло
по каналу (swarmsim.network):

  • кто жив — по времени последнего heartbeat: молчит дольше timeout → «потерян».
    Gossip-обнаружение: heartbeat раз в секунду несёт «когда я последний раз слышал каждого»,
    поэтому агент, слышимый хоть кем-то в связной сети, жив для всех; по-настоящему
    отказавшего не слышит никто. Ложно «теряют» только при реальном разрыве сети на части;
  • последнее состояние каждого агента: позиция, заряд, режим, текущая задача, план;
  • реплика статусов задач: 0 — никто (по моим данным), 1 — назначена, 2 — выполняется,
    3 — выполнена; владелец; прогресс (номер путевой точки).

Слияние идемпотентно и монотонно там, где это важно для FR-8: «выполнена» не отменяется
никогда, прогресс только растёт; остальное — «последний услышанный владелец прав».
Heartbeat отправителя a переносит: его текущую задачу и прогресс, его план (намерение)
и множество задач, про которые a знает, что они выполнены, — так знание о выполненном
распространяется через соседей (транзитом), даже без прямой связи.
"""
from __future__ import annotations

import numpy as np

FREE, ASSIGNED, RUNNING, DONE = 0, 1, 2, 3


class KnowledgeBase:
    def __init__(self, n_agents: int, n_tasks: int, timeout: float):
        self.n = n_agents
        self.o = n_agents + 1                      # наблюдатели: агенты + GCS
        self.gcs = n_agents
        self.timeout = timeout
        self.last_heard = np.zeros((self.o, n_agents))
        self.alive = np.ones((self.o, n_agents), dtype=bool)
        self.gcs_heard = np.zeros(n_agents)         # когда агент последний раз слышал станцию
        # последнее известное состояние агентов (наблюдатель × агент)
        self.pos = np.zeros((self.o, n_agents, 2))
        self.pos_t = np.zeros((self.o, n_agents))       # когда позиция получена (gossip её не обновляет)
        self.vel = np.zeros((self.o, n_agents, 2))
        self.energy = np.zeros((self.o, n_agents))
        self.mode = np.zeros((self.o, n_agents), dtype=np.int8)
        self.cur = -np.ones((self.o, n_agents), dtype=int)
        self.wp = np.zeros((self.o, n_agents), dtype=int)
        self.lowbat = np.zeros((self.o, n_agents), dtype=bool)
        self.vcruise = np.zeros((self.o, n_agents))
        self.plan_version = np.zeros((self.o, n_agents), dtype=int)
        self.plan_gcs: list[tuple[int, ...]] = [()] * n_agents     # планы — только для станции
        # реплика задач
        self.status = np.zeros((self.o, n_tasks), dtype=np.int8)
        self.owner = -np.ones((self.o, n_tasks), dtype=int)
        self.progress = np.zeros((self.o, n_tasks), dtype=int)
        self.appeared = np.ones(n_tasks, dtype=bool)
        self.own_cur = -np.ones(n_agents, dtype=int)   # что сейчас выполняет каждый агент (его знание о себе)
        self.conflicts: list[tuple[int, int, int, int]] = []   # (я, задача, другой, его прогресс)

    # ---------------------------------------------------------------- рост набора задач
    def grow(self, n_tasks: int) -> None:
        m = self.status.shape[1]
        if n_tasks <= m:
            return
        add = n_tasks - m
        self.status = np.hstack([self.status, np.zeros((self.o, add), np.int8)])
        self.owner = np.hstack([self.owner, -np.ones((self.o, add), int)])
        self.progress = np.hstack([self.progress, np.zeros((self.o, add), int)])
        self.appeared = np.concatenate([self.appeared, np.ones(add, bool)])

    # ---------------------------------------------------------------- свои действия агента
    def local(self, a: int, task: int, status: int, owner: int, progress: int = 0) -> None:
        if self.status[a, task] == DONE and status != DONE:
            return
        self.status[a, task] = status
        self.owner[a, task] = owner
        self.progress[a, task] = max(self.progress[a, task], progress)

    # ---------------------------------------------------------------- heartbeat
    def on_heartbeats(self, t: float, got: np.ndarray, hb: list[dict | None]) -> None:
        """got[src, dst] — кто что услышал; hb[a] — содержимое heartbeat агента a (None — молчит)."""
        for a, h in enumerate(hb):
            if h is None:
                continue
            rcv = np.flatnonzero(got[a])
            if len(rcv) == 0:
                continue
            self.last_heard[rcv, a] = t
            self.pos[rcv, a] = h["p"]; self.vel[rcv, a] = h["v"]; self.pos_t[rcv, a] = t
            self.energy[rcv, a] = h["e"]; self.mode[rcv, a] = h["mode"]
            self.cur[rcv, a] = h["cur"]; self.wp[rcv, a] = h["wp"]
            self.lowbat[rcv, a] = h["lowbat"]; self.vcruise[rcv, a] = h["vc"]
            self.plan_version[rcv, a] = h["pv"]
            if self.gcs in rcv:
                self.plan_gcs[a] = tuple(h["plan"])
            self._merge_tasks(rcv, a, h)
            lh = h.get("lh")
            if lh is not None:                                   # gossip: сведения о живости
                self.last_heard[rcv] = np.maximum(self.last_heard[rcv], lh[None, :])
        # станция тоже шлёт heartbeat (1 из 5 тиков хватает, но проще — каждый): агенты знают о связи
        g = np.flatnonzero(got[self.gcs, : self.n])
        self.gcs_heard[g] = t

    def _merge_tasks(self, rcv: np.ndarray, a: int, h: dict) -> None:
        st = self.status[rcv]
        own = self.owner[rcv]
        done = h["done"]
        if len(done):
            st[:, done] = DONE
        intent = np.zeros(st.shape[1], dtype=bool)
        plan = np.asarray(h["plan"], dtype=int)
        if len(plan):
            intent[plan] = True
            sub = st[:, plan]
            free = sub < RUNNING
            sub[free] = ASSIGNED
            st[:, plan] = sub
            o = own[:, plan]
            o[free] = a
            own[:, plan] = o
        j = h["cur"]
        if j >= 0:
            # конфликт исполнения: получатель сам сейчас выполняет ту же задачу
            for r in rcv[rcv < self.n]:
                if r != a and self.own_cur[r] == j:
                    self.conflicts.append((int(r), int(j), int(a), int(h["wp"])))
            intent[j] = True
            col = st[:, j] != DONE
            # если задачу уже выполняет другой живой агент, выигрывающий конфликт (больше прогресс,
            # при равенстве — меньший номер), владелец остаётся прежним: иначе проигравший, уступив,
            # «освободит» чужую задачу у всех, кто его слышал
            ow = own[:, j]
            for k in np.flatnonzero(col & (st[:, j] == RUNNING) & (ow >= 0) & (ow != a)):
                r, o = rcv[k], ow[k]
                if self.alive[r, o] and self.cur[r, o] == j and (self.wp[r, o], -o) > (h["wp"], -a):
                    col[k] = False
            st[col, j] = RUNNING
            own[col, j] = a
            self.progress[rcv[col], j] = np.maximum(self.progress[rcv[col], j], h["wp"])
        # a больше не держит задачи, которых нет в его намерении → они свободны (по данным a)
        stale = (own == a) & (st < DONE) & ~intent[None, :]
        st[stale] = FREE
        own[stale] = -1
        self.status[rcv] = st
        self.owner[rcv] = own

    def forget_owner(self, observer: int, a: int) -> list[int]:
        """Наблюдатель считает a потерянным: его незавершённые задачи — свободны (по моим данным)."""
        mask = (self.owner[observer] == a) & (self.status[observer] < DONE)
        idx = np.flatnonzero(mask)
        self.status[observer, idx] = FREE
        self.owner[observer, idx] = -1
        return idx.tolist()

    # ---------------------------------------------------------------- обнаружение
    def detect(self, t: float, reported_dead: set[int] | None = None,
               timeouts: np.ndarray | None = None) -> list[tuple[int, int, str]]:
        """Пересчитать «кто жив» у всех наблюдателей: список (наблюдатель, агент, lost|recovered).

        timeouts[o, a] — сколько o готов ждать вестей от a (зависит от того, должен ли o слышать a
        напрямую, через соседей или вообще не должен — см. NetworkRuntime._timeouts).
        Без него (идеальная связь) — единый timeout."""
        silent = t - self.last_heard
        now = silent <= (self.timeout if timeouts is None else timeouts) + 1e-9
        for a in reported_dead or ():
            now[:, a] = False
        idx = np.arange(self.n)
        now[idx, idx] = True                                     # себя всегда «слышу»
        lost = np.argwhere(self.alive & ~now)
        back = np.argwhere(~self.alive & now)
        self.alive = now
        out = [(int(o), int(a), "lost") for o, a in lost] + [(int(o), int(a), "recovered") for o, a in back]
        for o, a, kind in out:
            if kind == "lost":
                self.forget_owner(o, a)
        return out

    def gcs_link(self, a: int, t: float, timeout: float) -> bool:
        return (t - self.gcs_heard[a]) <= timeout + 1e-9
