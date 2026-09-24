"""Модель обнаружения (docs/02, раздел 3.3). Владелец — R3; здесь базовая версия по формулам документа.

GSD = altitude · pixel_pitch / focal_length
W   = W_footprint · min(1, GSD_target / GSD),  если GSD ≤ GSD_max, иначе 0
Pd  = 1 − exp(−W · L / A_cell)                 (один проход длиной L над ячейкой)
"""
from __future__ import annotations

import math

from .scenario import SensorConfig


class SensorModel:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        self._cache: dict[float, float] = {}

    def gsd(self, alt: float) -> float:
        return alt * self.cfg.pixel_pitch_m / self.cfg.focal_length_m

    def footprint(self, alt: float) -> float:
        return 2.0 * alt * math.tan(math.radians(self.cfg.fov_deg) / 2.0)

    def swath_width(self, alt: float) -> float:
        w = self._cache.get(alt)
        if w is None:
            gsd = self.gsd(alt)
            w = 0.0 if gsd > self.cfg.gsd_max_m else self.footprint(alt) * min(1.0, self.cfg.gsd_target_m / gsd)
            self._cache[alt] = w
        return w

    @staticmethod
    def pd_pass(length: float, width: float, cell_area: float) -> float:
        return 1.0 - math.exp(-width * length / cell_area)
