"""Миссия «поиск»: разбиение территории на секторы, в каждом — «змейка» (lawnmower).

Одна задача = один сектор. Путевые точки внутри бесполётных зон выбрасываются;
обход зон на переходах — ответственность планировщика/валидатора (R2).
"""
from __future__ import annotations

import math

import numpy as np

from ..interfaces import MissionPlugin, register_mission
from ..datatypes import Task, TaskSet


def lawnmower(x0: float, y0: float, x1: float, y1: float, spacing: float) -> np.ndarray:
    """Галсы вдоль x с шагом spacing, отступ spacing/2 от краёв сектора (место на разворот)."""
    ys = np.arange(y0 + spacing / 2, y1, spacing)
    if len(ys) == 0:
        ys = np.array([(y0 + y1) / 2])
    if x1 - x0 > spacing:
        x0, x1 = x0 + spacing / 2, x1 - spacing / 2
    pts = []
    for k, y in enumerate(ys):
        xa, xb = (x0, x1) if k % 2 == 0 else (x1, x0)
        pts += [(xa, y), (xb, y)]
    return np.array(pts, dtype=float)


@register_mission("lawnmower_grid")
class LawnmowerGridMission(MissionPlugin):
    def decompose(self, world) -> TaskSet:
        m = self.scenario.mission
        _, _, w, h = world.bounds
        nx, ny = math.ceil(w / m.sector_size), math.ceil(h / m.sector_size)
        tasks, tid = TaskSet(), 0
        for iy in range(ny):
            for ix in range(nx):
                x0, y0 = ix * m.sector_size, iy * m.sector_size
                x1, y1 = min(x0 + m.sector_size, w), min(y0 + m.sector_size, h)
                wps = lawnmower(x0, y0, x1, y1, m.track_spacing)
                wps = np.array([p for p in wps if not world.in_nofly(p)])
                if len(wps) < 2:
                    continue
                tasks.add(Task(id=tid, waypoints=wps, area=(x1 - x0) * (y1 - y0)))
                tid += 1
        return tasks
