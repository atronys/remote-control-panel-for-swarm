"""Модель маршрута агента по его задачам (docs/01, п. 2.2): время, энергия, ценность плана.

Одна модель для всех распределителей (B0–B5) — это правило честности docs/03 §8: у всех
одинаковая целевая функция планирования, различается только способ её оптимизации и то,
какую информацию распределитель видит.

Целевая функция планирования (максимизируется):

    S(маршрут) = Σ_j  value_j · λ^{C_j}

  C_j   — через сколько секунд (от момента планирования) задача j будет закончена;
  λ     — дисконт по времени (planning.discount): раньше найдено — ценнее;
  value — ценность задачи (priority): доля априорной вероятности целей в секторе
          или 1 (planning.value = uniform — «планирование по покрытию», для ablation).

Ограничение (docs/01, 2.2): энергия маршрута + возврат домой ≤ доступной с резервом.
Движение — по прямой с крейсерской скоростью; «змейка» задачи — её длина от текущего
прогресса (task.progress) до конца.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..datatypes import AgentState, Task, TaskSet


@dataclass(frozen=True)
class TaskGeom:
    """Геометрия задач в виде массивов (строка r ↔ id задачи ids[r])."""

    ids: np.ndarray        # (M,) int
    entry: np.ndarray      # (M, 2) первая оставшаяся точка
    exit: np.ndarray       # (M, 2) последняя точка
    length: np.ndarray     # (M,) оставшаяся длина «змейки», м
    value: np.ndarray      # (M,) ценность
    d_ee: np.ndarray       # (M, M) расстояние exit_i → entry_j

    @property
    def m(self) -> int:
        return len(self.ids)

    def row(self, task_id: int) -> int:
        return int(np.flatnonzero(self.ids == task_id)[0])

    @staticmethod
    def remaining(task: Task, progress: int | None = None) -> tuple[np.ndarray, float]:
        """Точка входа и оставшаяся длина задачи с учётом прогресса."""
        k = task.progress if progress is None else progress
        k = min(max(int(k), 0), len(task.waypoints) - 1)
        w = task.waypoints[k:]
        length = float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) if len(w) > 1 else 0.0
        return w[0], length

    @classmethod
    def build(cls, tasks: TaskSet, ids: list[int]) -> "TaskGeom":
        m = len(ids)
        entry = np.zeros((m, 2)); exit_ = np.zeros((m, 2)); length = np.zeros(m); value = np.zeros(m)
        for r, tid in enumerate(ids):
            t = tasks[tid]
            entry[r], length[r] = cls.remaining(t)
            exit_[r] = t.waypoints[-1]
            value[r] = t.priority
        d_ee = np.sqrt(((exit_[:, None, :] - entry[None, :, :]) ** 2).sum(-1)) if m else np.zeros((0, 0))
        return cls(np.asarray(ids, dtype=int), entry, exit_, length, value, d_ee)


@dataclass
class AgentCtx:
    """С какого места, момента и с каким запасом энергии агент начинает свободную часть плана."""

    id: int
    start: np.ndarray      # (2,) позиция или выход из текущей задачи
    t0: float              # через сколько секунд свободная часть начнётся (остаток текущей задачи)
    e_avail: float         # Дж на свободную часть + возврат домой
    home: np.ndarray
    v: float
    e_per_m: float         # Дж/м при крейсерской скорости
    locked: int | None = None   # текущая задача (закреплена, в торгах не участвует)


def agent_context(a: AgentState, tasks: TaskSet, energy, reserve: float, e_land: float) -> AgentCtx:
    """Контекст агента из снимка: остаток текущей задачи вычитается из времени и энергии."""
    e_per_m = float(energy.power(a.v_cruise)) / a.v_cruise
    e_avail = a.energy_j - reserve * a.capacity_j - e_land
    start, t0 = np.asarray(a.p, dtype=float), 0.0
    if a.current_task is not None and a.current_task in tasks:
        task = tasks[a.current_task]
        w = task.waypoints[a.wp_idx:] if a.wp_idx < len(task.waypoints) else task.waypoints[-1:]
        pts = np.vstack([start, w])
        rest = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        t0 = rest / a.v_cruise
        e_avail -= rest * e_per_m
        start = task.waypoints[-1].astype(float)
    return AgentCtx(a.id, start, t0, e_avail, np.asarray(a.home, float), a.v_cruise, e_per_m,
                    a.current_task)


@dataclass
class RouteEval:
    score: float
    completion: np.ndarray    # (L,) C_j, с от момента планирования
    energy: float             # Дж: маршрут + возврат домой
    feasible: bool
    t_return: float           # когда будет дома


class RouteModel:
    """Оценка маршрутов и вставок для всех распределителей."""

    def __init__(self, geom: TaskGeom, discount: float):
        self.g = geom
        self.lam = discount
        self.ln_lam = math.log(discount) if discount < 1 else 0.0

    def disc(self, t):
        return np.exp(self.ln_lam * np.asarray(t, dtype=float))

    # ---------------------------------------------------------------- маршрут целиком
    def legs(self, ctx: AgentCtx, route: list[int]) -> np.ndarray:
        """Расстояния подлёта к каждой задаче маршрута (строки geom)."""
        g = self.g
        if not route:
            return np.zeros(0)
        r = np.asarray(route)
        d = np.empty(len(r))
        d[0] = np.hypot(*(g.entry[r[0]] - ctx.start))
        if len(r) > 1:
            d[1:] = g.d_ee[r[:-1], r[1:]]
        return d

    def evaluate(self, ctx: AgentCtx, route: list[int]) -> RouteEval:
        g = self.g
        if not route:
            back = float(np.hypot(*(ctx.home - ctx.start)))
            return RouteEval(0.0, np.zeros(0), back * ctx.e_per_m,
                             back * ctx.e_per_m <= ctx.e_avail + 1e-6, ctx.t0 + back / ctx.v)
        r = np.asarray(route)
        seg = self.legs(ctx, route) + g.length[r]
        completion = ctx.t0 + np.cumsum(seg) / ctx.v
        back = float(np.hypot(*(ctx.home - g.exit[r[-1]])))
        dist = float(seg.sum()) + back
        energy = dist * ctx.e_per_m
        score = float((g.value[r] * self.disc(completion)).sum())
        return RouteEval(score, completion, energy, energy <= ctx.e_avail + 1e-6,
                         float(completion[-1]) + back / ctx.v)

    # ---------------------------------------------------------------- вставка
    def insertion(self, ctx: AgentCtx, route: list[int], cands: np.ndarray,
                  ev: RouteEval | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Лучшая допустимая вставка каждого кандидата: (прирост S, позиция); −inf — нельзя.

        Для позиции p: задача j встаёт между «предыдущей точкой» (старт или выход r[p−1]) и
        «следующей» (вход r[p] или дом). Время последующих задач сдвигается на Δt, их вклад
        умножается на λ^{Δt} — поэтому прирост считается за O(1) через суффиксные суммы.
        """
        g = self.g
        cands = np.asarray(cands, dtype=int)
        k = len(cands)
        if k == 0:
            return np.zeros(0), np.zeros(0, dtype=int)
        ev = ev or self.evaluate(ctx, route)
        L = len(route)
        r = np.asarray(route, dtype=int)
        prev_pt = np.vstack([ctx.start[None, :], g.exit[r]]) if L else ctx.start[None, :]      # (L+1, 2)
        prev_t = np.concatenate([[ctx.t0], ev.completion]) if L else np.array([ctx.t0])         # (L+1,)
        contrib = g.value[r] * self.disc(ev.completion) if L else np.zeros(0)
        suffix = np.concatenate([np.cumsum(contrib[::-1])[::-1], [0.0]])                         # (L+1,)
        # следующая точка: вход r[p] для p < L, дом для p = L
        next_pt = np.vstack([g.entry[r], ctx.home[None, :]]) if L else ctx.home[None, :]
        # a: prev → entry_j, b: exit_j → next, c: prev → next
        a = np.sqrt(((prev_pt[None, :, :] - g.entry[cands][:, None, :]) ** 2).sum(-1))           # (k, L+1)
        b = np.sqrt(((g.exit[cands][:, None, :] - next_pt[None, :, :]) ** 2).sum(-1))
        c = np.sqrt(((prev_pt - next_pt) ** 2).sum(-1))[None, :]
        detour = a + g.length[cands][:, None] + b - c
        cj = prev_t[None, :] + (a + g.length[cands][:, None]) / ctx.v
        shift = detour / ctx.v
        shift[:, L] = 0.0                                   # после последней задачи сдвигать нечего
        gain = g.value[cands][:, None] * self.disc(cj) - (1.0 - self.disc(shift)) * suffix[None, :]
        ok = ev.energy + detour * ctx.e_per_m <= ctx.e_avail + 1e-6
        gain = np.where(ok, gain, -np.inf)
        pos = np.argmax(gain, axis=1)
        return gain[np.arange(k), pos], pos

    # ---------------------------------------------------------------- локальное улучшение
    def improve(self, ctx: AgentCtx, route: list[int], max_passes: int = 3) -> list[int]:
        """Перестановка одной задачи на другое место (relocate), пока растёт S."""
        best = list(route)
        best_ev = self.evaluate(ctx, best)
        for _ in range(max_passes):
            improved = False
            for i in range(len(best)):
                for j in range(len(best)):
                    if i == j:
                        continue
                    cand = best[:i] + best[i + 1:]
                    cand.insert(j, best[i])
                    ev = self.evaluate(ctx, cand)
                    if ev.feasible and ev.score > best_ev.score + 1e-12:
                        best, best_ev, improved = cand, ev, True
            if not improved:
                break
        return best
