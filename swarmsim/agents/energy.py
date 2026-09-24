"""Энергетическая модель мультиротора (Zeng, Xu, Zhang, 2019), docs/02 раздел 3.2.

P(v) = P0·(1 + 3v²/U_tip²)                                   профиль лопастей
     + Pi·sqrt( sqrt(1 + v⁴/(4·v0⁴)) − v²/(2·v0²) )          индукционная
     + 0.5·d0·ρ·s·A·v³                                       паразитная
"""
from __future__ import annotations

import numpy as np

from ..scenario import EnergyConfig


class RotaryWingEnergy:
    def __init__(self, cfg: EnergyConfig):
        self.cfg = cfg
        self._grid = np.linspace(0.0, 35.0, 3501)
        self._pgrid = self.power(self._grid)

    def power(self, v):
        """Потребляемая мощность, Вт, при воздушной скорости v, м/с (скаляр или массив)."""
        c = self.cfg
        v = np.asarray(v, dtype=float)
        v2 = v * v
        profile = c.P0 * (1.0 + 3.0 * v2 / c.U_tip ** 2)
        induced = c.Pi * np.sqrt(np.sqrt(1.0 + v2 * v2 / (4.0 * c.v0 ** 4)) - v2 / (2.0 * c.v0 ** 2))
        parasite = 0.5 * c.d0 * c.rho * c.s * c.A * v2 * v
        p = profile + induced + parasite
        return float(p) if p.ndim == 0 else p

    @property
    def hover_power(self) -> float:
        return self.cfg.P0 + self.cfg.Pi

    def energy_per_meter(self, v):
        """Дж/м. Бесконечность в висении."""
        v = np.asarray(v, dtype=float)
        with np.errstate(divide="ignore"):
            return self.power(v) / v

    def v_min_power(self) -> float:
        """Скорость максимальной продолжительности полёта (минимум P)."""
        return float(self._grid[np.argmin(self._pgrid)])

    def v_max_range(self) -> float:
        """Скорость максимальной дальности (минимум Дж/м)."""
        g = self._grid[1:]
        return float(g[np.argmin(self._pgrid[1:] / g)])

    def energy_for_distance(self, distance: float, v: float) -> float:
        """Энергия на перелёт distance с постоянной скоростью v, Дж."""
        if distance <= 0:
            return 0.0
        return self.power(v) * distance / v
