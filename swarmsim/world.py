"""Среда (docs/02, раздел 2.1): границы, бесполётные зоны, карта вероятности целей, цели.

Мир 2D (высота есть у агентов — для GSD и эшелонирования). Все случайные величины мира
берутся из собственного потока RNG, поэтому при одном зерне мир одинаков для любых
распределителей и любого числа агентов (парный дизайн, docs/03).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from .geometry import point_in_polygon, polygon_bbox, segment_intersects_polygon
from .scenario import Scenario
from .datatypes import WorldSnapshot

if TYPE_CHECKING:
    from .agents.base import Agent
    from .sensing import SensorModel


class World:
    def __init__(self, scenario: Scenario, rng: np.random.Generator, sensing_rng: np.random.Generator,
                 log: Callable[..., None] | None = None):
        cfg = scenario.world
        self.cfg = cfg
        self.safety = scenario.safety
        self.rng = rng
        self.sensing_rng = sensing_rng
        self._log = log or (lambda *a, **k: None)
        self.t = 0.0

        w, h = cfg.size
        self.bounds = (0.0, 0.0, float(w), float(h))
        self.nofly = tuple(np.asarray(p, dtype=float) for p in cfg.nofly)
        self._nofly_bbox = [polygon_bbox(p) for p in self.nofly]

        self.cell = cfg.cell_size
        self.nx = max(1, math.ceil(w / self.cell))
        self.ny = max(1, math.ceil(h / self.cell))
        self.cell_area = self.cell * self.cell
        self.prior = self._build_prior()

        # Все цели (начальные и будущие) генерируются сразу — порядок потребления RNG
        # не зависит от хода симуляции.
        n0 = cfg.targets
        schedule = sorted(cfg.new_targets, key=lambda e: e.t)
        counts = [n0] + [e.count for e in schedule]
        total = sum(counts)
        self.target_pos = self._sample_targets(total)
        self.target_appear = np.concatenate(
            [np.zeros(n0)] + [np.full(e.count, e.t) for e in schedule]) if total else np.zeros(0)
        self.target_cell = np.array([self.cell_index(p) for p in self.target_pos], dtype=int)
        self.target_active = self.target_appear <= 0.0
        self.target_detected = np.zeros(total, dtype=bool)
        self.target_t_detected = np.full(total, np.nan)
        self.target_miss = np.ones(total)              # Π(1 − Pd_k) по проходам над целью
        self.coverage_miss = np.ones(self.ny * self.nx)  # то же по ячейкам (карта покрытия)
        self._by_cell: dict[int, list[int]] = {}
        for i in np.flatnonzero(self.target_active):
            self._by_cell.setdefault(int(self.target_cell[i]), []).append(int(i))

        # Жёсткие ограничения
        self._pairs_col: set[tuple[int, int]] = set()
        self._pairs_nm: set[tuple[int, int]] = set()
        self._geo_violators: set[int] = set()
        self.n_collisions = 0
        self.n_near_miss = 0
        self.n_geofence = 0

    # ------------------------------------------------------------ построение
    def _cell_center(self, iy: int, ix: int) -> np.ndarray:
        return np.array([(ix + 0.5) * self.cell, (iy + 0.5) * self.cell])

    def _build_prior(self) -> np.ndarray:
        pc = self.cfg.prior
        ys = (np.arange(self.ny) + 0.5) * self.cell
        xs = (np.arange(self.nx) + 0.5) * self.cell
        X, Y = np.meshgrid(xs, ys)
        if pc.kind == "uniform":
            prior = np.ones((self.ny, self.nx))
        else:
            _, _, w, h = self.bounds
            centers = self.rng.uniform([0, 0], [w, h], size=(pc.n, 2))
            dens = np.zeros_like(X)
            for cx, cy in centers:
                dens += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * pc.sigma ** 2))
            dens /= dens.sum()
            prior = (1 - pc.background) * dens + pc.background / dens.size
        for iy in range(self.ny):
            for ix in range(self.nx):
                if self.in_nofly(self._cell_center(iy, ix)):
                    prior[iy, ix] = 0.0
        s = prior.sum()
        if s <= 0:
            raise ValueError("вся территория закрыта бесполётными зонами")
        return prior / s

    def _sample_targets(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2))
        flat = self.prior.ravel()
        for k in range(n):
            for _ in range(1000):
                c = int(self.rng.choice(flat.size, p=flat))
                iy, ix = divmod(c, self.nx)
                p = (np.array([ix, iy]) + self.rng.uniform(0, 1, 2)) * self.cell
                p = np.minimum(p, [self.bounds[2], self.bounds[3]])
                if not self.in_nofly(p):
                    break
            out[k] = p
        return out

    # ------------------------------------------------------------ геометрия
    def cell_index(self, p) -> int:
        ix = min(max(int(p[0] // self.cell), 0), self.nx - 1)
        iy = min(max(int(p[1] // self.cell), 0), self.ny - 1)
        return iy * self.nx + ix

    def in_bounds(self, p) -> bool:
        x0, y0, x1, y1 = self.bounds
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def in_nofly(self, p) -> bool:
        for poly, (bx0, by0, bx1, by1) in zip(self.nofly, self._nofly_bbox):
            if bx0 <= p[0] <= bx1 and by0 <= p[1] <= by1 and point_in_polygon(p, poly):
                return True
        return False

    def segment_is_free(self, a, b) -> bool:
        """Отрезок не выходит за границы и не пересекает бесполётные зоны (для валидатора R2)."""
        if not (self.in_bounds(a) and self.in_bounds(b)):
            return False
        for poly, (bx0, by0, bx1, by1) in zip(self.nofly, self._nofly_bbox):
            if max(a[0], b[0]) < bx0 or min(a[0], b[0]) > bx1 or max(a[1], b[1]) < by0 or min(a[1], b[1]) > by1:
                continue
            if segment_intersects_polygon(a, b, poly):
                return False
        return True

    # ------------------------------------------------------------ интерфейс
    def step(self, dt: float) -> list[int]:
        """Продвинуть время мира; вернуть индексы целей, появившихся на этом шаге."""
        self.t += dt
        appeared = np.flatnonzero(~self.target_active & (self.target_appear <= self.t + 1e-9))
        for i in appeared:
            self.target_active[i] = True
            self._by_cell.setdefault(int(self.target_cell[i]), []).append(int(i))
        if len(appeared):
            self._log(self.t, "new_targets", targets=[int(i) for i in appeared])
        return [int(i) for i in appeared]

    def observe(self, agent: "Agent", sensor: "SensorModel") -> list[int]:
        """Обнаружение за последний шаг агента (отрезок prev_p → p). Вернуть новые цели."""
        if not agent.airborne:
            return []
        a, b = agent.prev_p, agent.p
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length <= 0.0:
            return []
        width = sensor.swath_width(agent.alt)
        if width <= 0.0:
            return []

        n = max(1, math.ceil(length / (0.5 * self.cell)))
        per_cell: dict[int, float] = {}
        for k in range(n):
            mid = a + (b - a) * ((k + 0.5) / n)
            c = self.cell_index(mid)
            per_cell[c] = per_cell.get(c, 0.0) + length / n

        found = []
        for c, l in per_cell.items():
            pd = sensor.pd_pass(l, width, self.cell_area)
            self.coverage_miss[c] *= 1.0 - pd
            for i in self._by_cell.get(c, ()):
                self.target_miss[i] *= 1.0 - pd
                if not self.target_detected[i] and self.sensing_rng.random() < pd:
                    self.target_detected[i] = True
                    self.target_t_detected[i] = self.t
                    found.append(i)
        for i in found:
            self._log(self.t, "target_detected", agent=agent.id, target=int(i))
        return found

    def check_separation(self, agents: list["Agent"]) -> tuple[list, list]:
        """Новые столкновения и опасные сближения (события на входе пары в зону)."""
        air = [a for a in agents if a.airborne]
        col_now: set[tuple[int, int]] = set()
        nm_now: set[tuple[int, int]] = set()
        if len(air) >= 2:
            P = np.array([a.p for a in air])
            Z = np.array([a.alt for a in air])
            dh = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            dz = np.abs(Z[:, None] - Z[None, :])
            sep = self.safety.min_separation
            close = np.triu((dz < self.safety.vertical_separation) & (dh < 2 * sep), 1)
            for i, j in zip(*np.nonzero(close)):
                pair = (air[i].id, air[j].id)
                nm_now.add(pair)
                if dh[i, j] < sep:
                    col_now.add(pair)
        new_col = sorted(col_now - self._pairs_col)
        new_nm = sorted(nm_now - self._pairs_nm)
        self._pairs_col, self._pairs_nm = col_now, nm_now
        self.n_collisions += len(new_col)
        self.n_near_miss += len(new_nm)
        for pair in new_col:
            self._log(self.t, "collision", agents=list(pair))
        for pair in new_nm:
            self._log(self.t, "near_miss", agents=list(pair))
        return new_col, new_nm

    def check_geofence(self, agents: list["Agent"]) -> list[int]:
        """Агенты, вошедшие в бесполётную зону или вышедшие за границы на этом шаге."""
        now = {a.id for a in agents
               if a.airborne and not self.segment_is_free(a.prev_p, a.p)}
        new = sorted(now - self._geo_violators)
        self._geo_violators = now
        self.n_geofence += len(new)
        for aid in new:
            self._log(self.t, "geofence_violation", agent=aid)
        return new

    def snapshot(self, agents: list["Agent"]) -> WorldSnapshot:
        return WorldSnapshot(t=self.t, agents=tuple(a.snapshot() for a in agents),
                             bounds=self.bounds, nofly=self.nofly)

    # ------------------------------------------------------------ метрики мира
    @property
    def n_active_targets(self) -> int:
        return int(self.target_active.sum())

    def detected_fraction(self) -> float:
        n = self.n_active_targets
        return float(self.target_detected[self.target_active].sum() / n) if n else 1.0

    def expected_detected_fraction(self) -> float:
        """Pd-взвешенная доля: среднее по активным целям накопленной Pd (docs/03, T_detect)."""
        n = self.n_active_targets
        return float((1.0 - self.target_miss[self.target_active]).mean()) if n else 1.0

    def coverage_map(self) -> np.ndarray:
        return (1.0 - self.coverage_miss).reshape(self.ny, self.nx)
