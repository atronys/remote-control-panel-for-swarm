"""Ценность задач и целевая функция миссии U(M) (docs/01 п. 2.2, docs/03 раздел 4)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..datatypes import TaskSet, TaskStatus

if TYPE_CHECKING:
    from ..scenario import ObjectiveConfig
    from ..world import World


def sector_values(world: "World", tasks: TaskSet) -> dict[int, float]:
    """Доля априорной вероятности целей, приходящаяся на сектор, нормированная к среднему 1.

    Это «Pd-планирование» (docs/03 §5): распределитель знает карту вероятности, но не сами цели.
    """
    ys = (np.arange(world.ny) + 0.5) * world.cell
    xs = (np.arange(world.nx) + 0.5) * world.cell
    X, Y = np.meshgrid(xs, ys)
    vals = {}
    for t in tasks:
        w = t.waypoints
        pad = 10.0 + world.cell / 2
        inside = ((X >= w[:, 0].min() - pad) & (X <= w[:, 0].max() + pad)
                  & (Y >= w[:, 1].min() - pad) & (Y <= w[:, 1].max() + pad))
        vals[t.id] = float(world.prior[inside].sum())
    mean = float(np.mean(list(vals.values()))) if vals else 0.0
    if mean <= 0:
        return {k: 1.0 for k in vals}
    return {k: v / mean for k, v in vals.items()}


def true_values(world: "World", tasks: TaskSet, eps: float = 0.05) -> dict[int, float]:
    """Ценность по настоящим целям — только для оракула B0 «с полным знанием мира» (docs/03, B0).

    eps — малая ценность пустых секторов: оракул всё равно обязан их обыскать (CR),
    но делает это в последнюю очередь.
    """
    counts = {t.id: 0.0 for t in tasks}
    for p in world.target_pos:
        for t in tasks:
            w = t.waypoints
            if w[:, 0].min() - 10 <= p[0] <= w[:, 0].max() + 10 and w[:, 1].min() - 10 <= p[1] <= w[:, 1].max() + 10:
                counts[t.id] += 1.0
                break
    mean = float(np.mean(list(counts.values()))) if counts else 0.0
    if mean <= 0:
        return {k: 1.0 for k in counts}
    return {k: eps + v / mean for k, v in counts.items()}


def mission_utility(tasks: TaskSet, energy_wh: float, cfg: "ObjectiveConfig") -> dict[str, float]:
    """U(M) = Σ_j priority_j·1[выполнена до deadline] − λ_E·ΣE − λ_T·makespan.

    makespan — момент завершения последней выполненной задачи. U_norm = U / Σ priority —
    доля от идеальной миссии (всё сделано мгновенно и бесплатно), удобна для сравнения
    сценариев разного размера.
    """
    done = [t for t in tasks if t.status == TaskStatus.DONE]
    on_time = sum(t.priority for t in done if t.t_done is not None and t.t_done <= cfg.deadline)
    makespan = max((t.t_done for t in done), default=0.0)
    total = sum(t.priority for t in tasks) or 1.0
    u = on_time - cfg.lambda_E * energy_wh - cfg.lambda_T * makespan
    return {"U": round(u, 6), "U_norm": round(u / total, 6), "U_ontime": round(on_time, 6)}
