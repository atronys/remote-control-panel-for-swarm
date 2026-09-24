"""Модель канала связи (docs/02 п. 3.4, docs/05 P4): дальность, задержка, потери, полоса, GCS.

Узлы: агенты 0..N−1 и наземная станция (GCS) — узел N, стоит у точки взлёта.

  • связь агент↔агент возможна, если расстояние ≤ range; агент↔GCS — если ≤ range_gcs и
    станция не в «окне разрыва» (gcs_outages: список [t0, t1]);
  • каждое доставляемое сообщение теряется независимо с вероятностью loss (loss_gcs для GCS);
  • задержка = latency + U(0, jitter) + размер/полоса (очередь отправителя, байт/с);
  • широковещательная передача (broadcast) — одна передача в эфир: в трафик считается один раз,
    доставляется каждому соседу в радиусе с независимой потерей.

Heartbeat (5 Гц, NFR-2) моделируется пакетно: на каждом «тике» для всех пар разом (numpy),
без объектов-сообщений — иначе N=50 не уложится в бюджет симуляции. Остальные сообщения
(ставки CBBA, планы от GCS, события) — поштучно, с задержкой.

Все случайности — из потока RNG «comms»: при одном зерне канал ведёт себя одинаково для всех
распределителей (парный дизайн docs/03).
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# размеры сообщений, байт (оценка для бюджета трафика NFR-6: ≤ 5 кбит/с на агента)
HEADER = 16
HB_BASE = 40            # позиция, скорость, заряд, режим, текущая задача, путевая точка


@dataclass(order=True)
class _Pending:
    t_deliver: float
    seq: int
    dst: int = field(compare=False)
    src: int = field(compare=False)
    kind: str = field(compare=False)
    payload: Any = field(compare=False)


class Network:
    def __init__(self, n_agents: int, gcs_pos: np.ndarray, params: dict, rng: np.random.Generator,
                 ideal: bool = False):
        self.n = n_agents
        self.gcs = n_agents
        self.gcs_pos = np.asarray(gcs_pos, float)
        self.rng = rng
        self.ideal = ideal
        p = dict(params)
        big = 1e12
        self.range = float(p.get("range", big))
        self.range_gcs = float(p.get("range_gcs", p.get("range", big)))
        self.latency = 0.0 if ideal else float(p.get("latency", 0.02))
        self.jitter = 0.0 if ideal else float(p.get("jitter", 0.01))
        self.loss = 0.0 if ideal else float(p.get("loss", 0.0))
        self.loss_gcs = 0.0 if ideal else float(p.get("loss_gcs", self.loss))
        self.bandwidth = float(p.get("bandwidth", 32_000))          # байт/с на отправителя
        self.hb_period = float(p.get("hb_period", 0.2))
        self.outages = [tuple(map(float, w)) for w in p.get("gcs_outages", [])]
        self._q: list[_Pending] = []
        self._seq = itertools.count()
        self._busy_until = np.zeros(n_agents + 1)
        self.bytes = {"state": 0, "bids": 0, "plan": 0, "event": 0}
        self.msgs = {"state": 0, "bids": 0, "plan": 0, "event": 0}
        self.pos = np.zeros((n_agents + 1, 2))
        self.pos[self.gcs] = self.gcs_pos
        self.alive = np.ones(n_agents + 1, dtype=bool)

    # ---------------------------------------------------------------- геометрия связи
    def gcs_up(self, t: float) -> bool:
        return not any(a <= t < b for a, b in self.outages)

    def update_positions(self, pos: np.ndarray, alive: np.ndarray) -> None:
        self.pos[: self.n] = pos
        self.alive[: self.n] = alive

    def link_matrix(self, t: float) -> np.ndarray:
        """(N+1)×(N+1): есть ли радиосвязь между узлами сейчас (без учёта потерь)."""
        d = np.sqrt(((self.pos[:, None, :] - self.pos[None, :, :]) ** 2).sum(-1))
        ok = d <= self.range
        ok[self.gcs, :] = d[self.gcs, :] <= self.range_gcs
        ok[:, self.gcs] = d[:, self.gcs] <= self.range_gcs
        if not self.gcs_up(t):
            ok[self.gcs, :] = False
            ok[:, self.gcs] = False
        ok &= self.alive[:, None] & self.alive[None, :]
        np.fill_diagonal(ok, False)
        return ok

    def _loss_for(self, src: int, dst: int) -> float:
        return self.loss_gcs if self.gcs in (src, dst) else self.loss

    # ---------------------------------------------------------------- heartbeat (пакетно)
    def heartbeat_receivers(self, t: float, link: np.ndarray, hb_bytes: np.ndarray) -> np.ndarray:
        """Кто услышал heartbeat каждого отправителя на этом тике: (N+1)×(N+1), [src, dst]."""
        loss = np.full(link.shape, self.loss)
        loss[self.gcs, :] = self.loss_gcs
        loss[:, self.gcs] = self.loss_gcs
        got = link & (self.rng.random(link.shape) >= loss) if not self.ideal else link.copy()
        senders = self.alive.copy()
        self.bytes["state"] += int(hb_bytes[senders].sum())
        self.msgs["state"] += int(senders.sum())
        return got & senders[:, None]

    # ---------------------------------------------------------------- поштучные сообщения
    def send(self, t: float, src: int, dsts, kind: str, payload: Any, size: int, link: np.ndarray) -> None:
        """Отправить src → dsts (список или 'all' — широковещательно по соседям)."""
        if not self.alive[src]:
            return
        if isinstance(dsts, str):
            dsts = np.flatnonzero(link[src]).tolist()
        tx = size + HEADER
        start = max(t, self._busy_until[src])
        self._busy_until[src] = start + tx / self.bandwidth
        self.bytes[kind] += tx
        self.msgs[kind] += 1
        for d in dsts:
            if d == src or not link[src, d]:
                continue
            if not self.ideal and self.rng.random() < self._loss_for(src, d):
                continue
            delay = self.latency + (self.rng.random() * self.jitter if self.jitter else 0.0)
            heapq.heappush(self._q, _Pending(start + tx / self.bandwidth + delay if not self.ideal else t,
                                             next(self._seq), d, src, kind, payload))

    def deliver(self, t: float) -> list[_Pending]:
        out = []
        while self._q and self._q[0].t_deliver <= t + 1e-9:
            m = heapq.heappop(self._q)
            if self.alive[m.dst]:
                out.append(m)
        return out

    def stats(self, t_mission: float) -> dict:
        total = sum(self.bytes.values())
        return {"N_msg": int(sum(self.msgs.values())),
                "B_per_agent": round(total / max(self.n * max(t_mission, 1e-9), 1e-9), 3),
                **{f"B_{k}": round(v / max(self.n * max(t_mission, 1e-9), 1e-9), 3) for k, v in self.bytes.items()},
                **{f"N_msg_{k}": v for k, v in self.msgs.items()}}
