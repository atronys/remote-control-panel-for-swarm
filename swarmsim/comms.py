"""CommsBus. Владелец — R3 (дальность/задержка/потери/полоса — MVP-2).

Здесь базовая модель `ideal`: доставка на следующем шаге всем в радиусе `range`,
без потерь. Этого достаточно для MVP-0/1.
"""
from __future__ import annotations

import numpy as np

from .interfaces import CommsBus, register_comms


@register_comms("ideal")
class IdealComms(CommsBus):
    def __init__(self, scenario, params, rng):
        super().__init__(scenario, params, rng)
        self.range = float(params.get("range", 1e9))
        self._pending: list[tuple[int, int, dict]] = []
        self._inbox: dict[int, list[dict]] = {}
        self._neighbors: dict[int, list[int]] = {}
        self._pos: dict[int, np.ndarray] = {}
        self.n_msg = 0

    def step(self, t, agents):
        live = [a for a in agents if a.healthy]
        self._pos = {a.id: a.p for a in live}
        ids = [a.id for a in live]
        self._neighbors = {i: [] for i in ids}
        if len(live) >= 2:
            P = np.array([a.p for a in live])
            d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
            for i, j in zip(*np.nonzero(np.triu(d <= self.range, 1))):
                self._neighbors[ids[i]].append(ids[j])
                self._neighbors[ids[j]].append(ids[i])
        self._inbox = {i: [] for i in ids}
        for src, dst, msg in self._pending:
            if dst in self._inbox and src in self._pos and dst in self._neighbors.get(src, ()):
                self._inbox[dst].append({"src": src, **msg})
        self._pending = []

    def send(self, src, dst, msg):
        self.n_msg += 1
        self._pending.append((src, dst, msg))

    def broadcast(self, src, msg):
        for dst in self._neighbors.get(src, ()):
            self.send(src, dst, msg)

    def neighbors_of(self, agent_id):
        return list(self._neighbors.get(agent_id, ()))

    def inbox(self, agent_id):
        return self._inbox.get(agent_id, [])

    def stats(self):
        return {"N_msg": self.n_msg}
