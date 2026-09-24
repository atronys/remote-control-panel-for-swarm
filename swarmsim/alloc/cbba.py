"""B4 `cbba` и B5 `hybrid` (docs/03, раздел 2; docs/04 п. 1.2; docs/05 P4).

CBBA — Consensus-Based Bundle Algorithm (Choi, Brunet, How, IEEE T-RO 2009), здесь в
асинхронном виде: у каждого агента свой узел `CBBANode`, ставки уходят сообщениями по каналу
(swarmsim.network) — с задержками, потерями и ограниченной дальностью. Центра нет.

Узел агента i хранит:
  y[j] — лучшая известная ставка за задачу j;  z[j] — кто её сделал (−1 — никто);
  s[k] — время последних сведений об агенте k;  bundle — выигранные задачи в порядке добавления;
  path — те же задачи в порядке выполнения (плюс «основа» base у гибрида).
Специальные ставки: RUN — «выполняю» (заявка исполнителя, её не перебить; при двух заявках
побеждает меньший номер — проигравший уступает задачу), DONE — «выполнена» (не отменяется).

Фаза 1 (build): пока пакет меньше L, добавлять задачу с наибольшим приростом целевой функции
S = Σ value·λ^C (та же RouteModel, что у B2/B3 — одинаковая цель), если он перебивает y[j].
Фаза 2 (merge): правила update / reset / leave из табл. 1 статьи — векторно по задачам;
проигравший задачу теряет её и всё, что добавил в пакет после неё (каскад).
Потеря соседа (молчит дольше таймаута) — его ставки сбрасываются, задачи разыгрываются снова.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import Allocator, register_allocator
from ..planning.routing import RouteModel, TaskGeom, agent_context

RUN = 1e9
DONE = 2e9
EPS = 1e-12


def cbba_actions(i: int, k: int, yk, zk, yi, zi, sk, si) -> np.ndarray:
    """Правила консенсуса (Choi et al. 2009, табл. 1), векторно по задачам.

    Возвращает код действия на каждую задачу: 0 — оставить, 1 — принять (y, z) отправителя,
    2 — сбросить в «никто». i — получатель, k — отправитель, s — метки свежести (по агентам).
    """
    m = len(yk)
    act = np.zeros(m, dtype=np.int8)
    nk = np.where(zk >= 0, zk, 0)
    ni = np.where(zi >= 0, zi, 0)
    newer_zk = (zk >= 0) & (sk[nk] > si[nk])
    newer_zi = (zi >= 0) & (sk[ni] > si[ni])
    beats = (yk > yi + EPS) | ((np.abs(yk - yi) <= EPS) & (zk >= 0) & ((zi < 0) | (zk < zi)))
    none_i = zi < 0
    zi_i, zi_k = zi == i, zi == k
    zi_m = ~none_i & ~zi_i & ~zi_k
    # A: отправитель считает победителем себя
    A = zk == k
    act[A & zi_i & beats] = 1
    act[A & (zi_k | none_i)] = 1
    act[A & zi_m & (newer_zi | beats)] = 1
    # B: отправитель считает победителем меня
    B = zk == i
    act[B & zi_k] = 2
    act[B & zi_m & newer_zi] = 2
    # C: отправитель считает победителем третьего m
    C = (zk >= 0) & (zk != k) & (zk != i)
    same = zi == zk
    other = zi_m & ~same
    act[C & zi_i & newer_zk & beats] = 1
    act[C & zi_k & newer_zk] = 1
    act[C & zi_k & ~newer_zk] = 2
    act[C & same & newer_zk] = 1
    act[C & none_i & newer_zk] = 1
    c_other = C & other
    upd = c_other & ((newer_zk & newer_zi) | (newer_zk & beats))
    act[upd] = 1
    si_m_gt = si[nk] > sk[nk]
    act[c_other & ~upd & newer_zi & si_m_gt] = 2
    # D: отправитель считает, что никто
    D = zk < 0
    act[D & zi_k] = 1
    act[D & zi_m & newer_zi] = 1
    # «выполнено» всегда распространяется и никогда не отменяется
    act[yk >= DONE] = 1
    act[yi >= DONE] = 0
    return act


class CBBANode:
    """Узел CBBA на борту агента i."""

    def __init__(self, i: int, n_agents: int, n_tasks: int, bundle_size: int, hysteresis: float = 0.05):
        self.i = i
        self.hysteresis = hysteresis          # перебить чужую заявку — только на столько лучше
        self.n = n_agents
        self.L = bundle_size
        self.y = np.zeros(n_tasks)
        self.z = -np.ones(n_tasks, dtype=int)
        self.s = np.zeros(n_agents)
        self.bundle: list[int] = []
        self.path: list[int] = []
        self.base: list[int] = []            # гибрид: задачи от станции (не разыгрываются)
        self.current: int | None = None
        self.dirty = True
        self.changed = True
        self.lost: set[int] = set()

    def grow(self, n_tasks: int) -> None:
        m = len(self.y)
        if n_tasks > m:
            self.y = np.concatenate([self.y, np.zeros(n_tasks - m)])
            self.z = np.concatenate([self.z, -np.ones(n_tasks - m, dtype=int)])
            self.dirty = True

    # ---------------------------------------------------------------- состояние задач
    def set_current(self, j: int | None) -> None:
        if j == self.current:
            return
        if j is not None:
            self.y[j], self.z[j] = RUN, self.i
            if j in self.bundle:
                self.bundle.remove(j)
            if j in self.path:
                self.path.remove(j)
        self.current = j
        self.changed = self.dirty = True

    def set_done(self, j: int, owner: int) -> None:
        if self.y[j] >= DONE:
            return
        self.y[j], self.z[j] = DONE, owner
        if j in self.bundle:
            self.bundle.remove(j)
        if j in self.path:
            self.path.remove(j)
        if j == self.current:
            self.current = None
        self.changed = self.dirty = True

    def release_peer(self, k: int) -> None:
        """Сосед потерян: его заявки (кроме «выполнено») сброшены."""
        mask = (self.z == k) & (self.y < DONE)
        if mask.any():
            self.y[mask], self.z[mask] = 0.0, -1
            self.changed = self.dirty = True
        self.lost.add(k)

    def peer_back(self, k: int) -> None:
        self.lost.discard(k)

    # ---------------------------------------------------------------- фаза 2: консенсус
    def merge(self, k: int, t_msg: float, yk, zk, sk) -> None:
        if k in self.lost:
            return
        m = min(len(self.y), len(yk))
        sk = sk.copy()
        sk[k] = t_msg
        if self.lost:
            # заявки от имени агента, которого я считаю потерянным, — «ничьи», от кого бы ни пришли:
            # иначе соседи, ещё не заметившие отказ, вернут мне заявки мёртвого агента
            dead = np.isin(zk, list(self.lost)) & (yk < DONE)
            if dead.any():
                yk, zk = yk.copy(), zk.copy()
                yk[dead], zk[dead] = 0.0, -1
        diff = np.flatnonzero((yk[:m] != self.y[:m]) | (zk[:m] != self.z[:m]))
        if len(diff):
            si = self.s.copy()
            act = cbba_actions(self.i, k, yk[diff], zk[diff], self.y[diff], self.z[diff], sk, si)
            upd, rst = diff[act == 1], diff[act == 2]
            if len(upd) or len(rst):
                self.y[upd], self.z[upd] = yk[upd], zk[upd]
                self.y[rst], self.z[rst] = 0.0, -1
                self.changed = self.dirty = True
        np.maximum(self.s, sk, out=self.s)
        self._cascade()

    def _cascade(self) -> None:
        lost_at = next((q for q, j in enumerate(self.bundle) if self.z[j] != self.i), None)
        if lost_at is None:
            return
        for j in self.bundle[lost_at + 1:]:
            if self.z[j] == self.i and self.y[j] < RUN:
                self.y[j], self.z[j] = 0.0, -1
        drop = set(self.bundle[lost_at:])
        self.bundle = self.bundle[:lost_at]
        self.path = [j for j in self.path if j not in drop]
        self.changed = self.dirty = True

    def drop(self, j: int) -> None:
        """Снять свою заявку на j (агент не может её выполнить) — задача снова разыгрывается."""
        if self.z[j] == self.i and self.y[j] < RUN:
            self.y[j], self.z[j] = 0.0, -1
        for lst in (self.bundle, self.path):
            if j in lst:
                lst.remove(j)
        self.changed = self.dirty = True

    def reset_claims(self) -> None:
        """Отказаться от всех своих заявок, кроме выполняемой и выполненных (новый план / уход домой)."""
        mask = (self.z == self.i) & (self.y < RUN)
        if mask.any():
            self.y[mask], self.z[mask] = 0.0, -1
            self.changed = True
        self.bundle, self.path, self.base = [], [], []
        self.dirty = True

    def lost_current(self) -> bool:
        """Мою текущую задачу по консенсусу выполняет другой (конфликт исполнения)?"""
        j = self.current
        return j is not None and self.z[j] != self.i and self.y[j] < DONE

    # ---------------------------------------------------------------- фаза 1: пакет
    def build(self, ctx, tasks, candidates: list[int], discount: float) -> None:
        """Добрать пакет до L задачами с наибольшим перебивающим приростом S."""
        if not self.dirty:
            return
        self.dirty = False
        cand = [j for j in candidates if j not in self.bundle and j not in self.base and self.y[j] < RUN]
        route_ids = [j for j in self.path]
        ids = list(dict.fromkeys(route_ids + cand))
        if not ids:
            return
        geom = TaskGeom.build(tasks, ids)
        model = RouteModel(geom, discount)
        row = {tid: r for r, tid in enumerate(ids)}
        route = [row[j] for j in route_ids]
        pool = [row[j] for j in cand]
        while len(self.bundle) < self.L and pool:
            ev = model.evaluate(ctx, route)
            gains, pos = model.insertion(ctx, route, np.asarray(pool), ev)
            yv = self.y[[int(geom.ids[r]) for r in pool]]
            zv = self.z[[int(geom.ids[r]) for r in pool]]
            margin = np.where((zv >= 0) & (zv != self.i), yv * self.hysteresis, 0.0)
            outbid = (gains > yv + margin + EPS) | ((np.abs(gains - yv) <= EPS) & (gains > 0) & (zv < 0))
            gains = np.where(outbid & np.isfinite(gains) & (gains > 0), gains, -np.inf)
            c = int(np.argmax(gains))
            if not np.isfinite(gains[c]):
                break
            j = int(geom.ids[pool[c]])
            route.insert(int(pos[c]), pool[c])
            del pool[c]
            self.bundle.append(j)
            self.y[j], self.z[j] = float(gains[c]), self.i
            self.changed = True
        self.path = [int(geom.ids[r]) for r in route]

    def message(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.changed = False
        return self.y.copy(), self.z.copy(), self.s.copy()


@register_allocator("cbba")
class CBBAAllocator(Allocator):
    """B4: децентрализованный CBBA. Сам ничего не распределяет — узлы на агентах (SimLoop)."""

    decentralized = True

    def __init__(self, scenario, params: dict[str, Any], rng):
        super().__init__(scenario, params, rng)
        self.bundle_size = params.get("bundle_size", "auto")
        self.period = float(params.get("period", 1.0))           # периодическая рассылка ставок, с
        self.min_gap = float(params.get("min_gap", 0.1))         # не чаще раза за шаг

    def allocate(self, snapshot, tasks):
        return {}

    def bundle_L(self, n_tasks: int, n_agents: int) -> int:
        L = self.bundle_size
        if not isinstance(L, (int, float)) or L <= 0:
            L = int(np.ceil(n_tasks / max(n_agents, 1))) + 1
        return int(L)


@register_allocator("hybrid")
class HybridAllocator(Allocator):
    """B5: B2 (центральный жадный) при связи со станцией + локальный ремонт CBBA без неё.

    Пока агент слышит станцию — выполняет её план (как B2). Потеряв станцию (режим NO_GCS),
    агент продолжает свой план и вместе с соседями разыгрывает «сирот» — задачи, которые по его
    реплике никому не принадлежат (включая задачи потерянных агентов), вставляя их в свой маршрут.
    Когда связь возвращается, станция видит новые намерения в heartbeat и пересчитывает план
    с политикой repair (не ломая уже взятое).
    """

    decentralized = False
    hybrid = True

    def __init__(self, scenario, params, rng):
        super().__init__(scenario, params, rng)
        from .central import CentralGreedy
        self.central = CentralGreedy(scenario, {**params, "kind": "central_greedy"}, rng)
        self.central.name = "hybrid"
        self.bundle_size = params.get("bundle_size", 2)
        self.period = float(params.get("period", 1.0))
        self.min_gap = float(params.get("min_gap", 0.1))
        self.explain = {}

    def bind(self, world, energy):
        self.central.bind(world, energy)

    def allocate(self, snapshot, tasks):
        out = self.central.allocate(snapshot, tasks)
        self.explain = self.central.explain
        return out

    def should_trigger(self, t, tasks, events):
        return self.central.should_trigger(t, tasks, events)

    def bundle_L(self, n_tasks, n_agents):
        return int(self.bundle_size)
