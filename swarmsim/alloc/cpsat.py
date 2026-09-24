"""Точные распределители на OR-Tools CP-SAT (docs/03, раздел 2):

  B3 `central_optimal` — оптимум (или лучшее найденное с доказанной границей) для малых N,
     та же целевая функция S, что у B2/B4, пересчёт по событиям;
  B0 `static_oracle`   — офлайн-разбиение с ПОЛНЫМ знанием мира: ценность сектора считается
     по настоящим целям, распределение один раз на старте, без перераспределения. Честный
     ориентир: рой обязан обыгрывать его только в режимах с нарушениями.

Модель (для каждого агента k — свой контур, AddCircuit):
  узлы: 0 — старт/дом агента, 1..M — задачи; дуга x[k,i,j] — «после i сразу j»;
  петля x[k,j,j] — задача j агентом k не выполняется; петля x[k,0,0] — агент без задач.
  Σ_k visit[k,j] ≤ 1                                 — задача не более одному агенту
  T_j ≥ t0_k + τ(старт_k → j)       если x[k,0,j]      — время завершения (секунды)
  T_j ≥ T_i + τ(i → j)               если x[k,i,j]
  Σ_дуги x·E ≤ e_avail_k                              — энергия, включая возврат домой
  цель: max Σ_j value_j·λ^{T_j}·[j выполнена]  (λ^T — через AddElement по таблице)
Время округляется до секунды, энергия — до джоуля; итоговый маршрут переоценивается точно.
Решение детерминировано: 1 поток и ограничение по «детерминированному времени».
"""
from __future__ import annotations

import math

import numpy as np
from ortools.sat.python import cp_model

from ..interfaces import register_allocator
from ..planning.objective import true_values
from .central import CentralAllocator, greedy_insert

SCALE = 10_000


def solve_cpsat(model, ctxs, routes_hint, pool_and_kept: list[int], time_limit: float,
                workers: int = 1) -> tuple[list[list[int]], dict]:
    """Оптимальные маршруты по строкам geom `rows` (подсказка — routes_hint)."""
    g = model.g
    rows = list(pool_and_kept)
    m, n = len(rows), len(ctxs)
    info = {"status": "empty", "objective": 0.0, "bound": 0.0}
    if m == 0 or n == 0:
        return [[] for _ in ctxs], info
    idx = {r: i + 1 for i, r in enumerate(rows)}          # строка geom → узел 1..m
    horizon = 1
    for c in ctxs:
        horizon = max(horizon, int(math.ceil(c.t0 + max(c.e_avail, 0) / c.e_per_m / c.v)) + 1)
    horizon = min(horizon, 20_000)
    disc = np.exp(model.ln_lam * np.arange(horizon + 1))
    tables = [[int(round(SCALE * g.value[r] * d)) for d in disc] for r in rows]

    cp = cp_model.CpModel()
    T = [None] + [cp.NewIntVar(0, horizon, f"T{j}") for j in range(1, m + 1)]
    rew = [None] + [cp.NewIntVar(0, tables[j - 1][0], f"R{j}") for j in range(1, m + 1)]
    for j in range(1, m + 1):
        cp.AddElement(T[j], tables[j - 1], rew[j])
    visit = [[None] * (m + 1) for _ in range(n)]
    arcs_lit = []
    for k, c in enumerate(ctxs):
        arcs = []
        lit = {}
        e_terms = []
        s_to = np.hypot(g.entry[rows, 0] - c.start[0], g.entry[rows, 1] - c.start[1])
        to_home = np.hypot(g.exit[rows, 0] - c.home[0], g.exit[rows, 1] - c.home[1])
        empty = cp.NewBoolVar(f"empty{k}")
        arcs.append((0, 0, empty))
        e_terms.append((empty, int(round(np.hypot(*(c.home - c.start)) * c.e_per_m))))
        for j in range(1, m + 1):
            rj = rows[j - 1]
            skip = cp.NewBoolVar(f"skip{k}_{j}")
            arcs.append((j, j, skip))
            visit[k][j] = skip.Not()
            # старт → j
            x0 = cp.NewBoolVar(f"x{k}_0_{j}")
            arcs.append((0, j, x0)); lit[(0, j)] = x0
            cp.Add(T[j] >= int(round(c.t0 + (s_to[j - 1] + g.length[rj]) / c.v))).OnlyEnforceIf(x0)
            e_terms.append((x0, int(round((s_to[j - 1] + g.length[rj]) * c.e_per_m))))
            # j → дом
            xh = cp.NewBoolVar(f"x{k}_{j}_0")
            arcs.append((j, 0, xh)); lit[(j, 0)] = xh
            e_terms.append((xh, int(round(to_home[j - 1] * c.e_per_m))))
            for i in range(1, m + 1):
                if i == j:
                    continue
                ri = rows[i - 1]
                x = cp.NewBoolVar(f"x{k}_{i}_{j}")
                arcs.append((i, j, x)); lit[(i, j)] = x
                d = g.d_ee[ri, rj] + g.length[rj]
                cp.Add(T[j] >= T[i] + int(round(d / c.v))).OnlyEnforceIf(x)
                e_terms.append((x, int(round(d * c.e_per_m))))
        cp.AddCircuit(arcs)
        cp.Add(sum(l * e for l, e in e_terms) <= int(math.floor(max(c.e_avail, 0))))
        arcs_lit.append(lit)
    obj = []
    for j in range(1, m + 1):
        any_v = cp.NewBoolVar(f"v{j}")
        cp.Add(sum(visit[k][j] for k in range(n)) == any_v)
        o = cp.NewIntVar(0, tables[j - 1][0], f"o{j}")
        cp.Add(o <= rew[j])
        cp.Add(o <= tables[j - 1][0] * any_v)
        obj.append(o)
    cp.Maximize(sum(obj))

    # подсказка — жадное решение (CP-SAT не может оказаться хуже неё)
    for k, route in enumerate(routes_hint):
        nodes = [0] + [idx[r] for r in route if r in idx] + [0]
        used = set(zip(nodes[:-1], nodes[1:]))
        for (i, j), l in arcs_lit[k].items():
            cp.AddHint(l, (i, j) in used)
        ev = model.evaluate(ctxs[k], [r for r in route if r in idx])
        for r, cj in zip([r for r in route if r in idx], ev.completion):
            cp.AddHint(T[idx[r]], min(horizon, int(round(cj))))

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = int(workers)
    solver.parameters.max_deterministic_time = float(time_limit)
    solver.parameters.random_seed = 0
    status = solver.Solve(cp)
    info = {"status": solver.StatusName(status), "objective": solver.ObjectiveValue() / SCALE,
            "bound": solver.BestObjectiveBound() / SCALE, "wall_s": round(solver.WallTime(), 3)}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [list(r) for r in routes_hint], info
    out = []
    for k in range(n):
        route, cur, guard = [], 0, 0
        while guard <= m:
            nxt = next((j for (i, j), l in arcs_lit[k].items() if i == cur and solver.BooleanValue(l)), 0)
            if nxt == 0:
                break
            route.append(rows[nxt - 1]); cur = nxt; guard += 1
        out.append(route)
    return out, info


@register_allocator("central_optimal")
class CentralOptimal(CentralAllocator):
    """B3: эталон качества централизованного назначения.

    1) жадное решение B2 (подсказка);
    2) CP-SAT, если задача мала (≤ cp_max_tasks свободных задач): доказанный оптимум или граница;
    3) Ruin & Recreate LNS по той же целевой функции — «лучшее известное» решение.
    Итог — лучшее из трёх по точной оценке RouteModel; в explain — статус CP-SAT и граница.
    """

    def __init__(self, scenario, params, rng):
        super().__init__(scenario, params, rng)
        self.time_limit = float(params.get("time_limit", 10.0))
        self.workers = int(params.get("workers", 1))
        self.cp_max_tasks = int(params.get("cp_max_tasks", 12))
        self.lns_iters = int(params.get("lns_iters", 400))
        self.last_info: dict = {}

    def solve(self, model, ctxs, routes, pool):
        from .central import ruin_recreate, total_score
        hint = greedy_insert(model, ctxs, routes, pool)
        hint = [model.improve(c, r) for c, r in zip(ctxs, hint)]
        g_score = total_score(model, ctxs, hint)
        cands = [("greedy", hint)]
        rows = sorted({r for rt in routes for r in rt} | set(pool))
        info = {"status": "skipped", "bound": None}
        if len(rows) <= self.cp_max_tasks and self.time_limit > 0:
            sol, info = solve_cpsat(model, ctxs, hint, rows, self.time_limit, self.workers)
            if all(model.evaluate(c, r).feasible for c, r in zip(ctxs, sol)):
                cands.append(("cpsat", sol))
        if self.lns_iters > 0:
            rng = np.random.default_rng(int(self.rng.integers(1 << 31))) if self.rng is not None                 else np.random.default_rng(0)
            best_so_far = max(cands, key=lambda kv: total_score(model, ctxs, kv[1]))[1]
            cands.append(("lns", ruin_recreate(model, ctxs, best_so_far, rng, self.lns_iters)))
        name, best = max(cands, key=lambda kv: total_score(model, ctxs, kv[1]))
        b_score = total_score(model, ctxs, best)
        self.last_info = {"source": name, "status": info.get("status"),
                          "bound": None if info.get("bound") is None else round(info["bound"], 6),
                          "greedy_score": round(g_score, 6), "score": round(b_score, 6)}
        return best

    def allocate(self, snapshot, tasks):
        out = super().allocate(snapshot, tasks)
        self.explain.update({"cpsat": self.last_info})
        return out


@register_allocator("static_oracle")
class StaticOracle(CentralOptimal):
    """B0: оракул с полным знанием целей, один раз на старте, без перераспределения."""

    def __init__(self, scenario, params, rng):
        params = {**params, "dynamic": False}
        super().__init__(scenario, params, rng)
        self.time_limit = float(params.get("time_limit", 10.0))

    def values_override(self, geom, tasks):
        if self.world is None:
            return
        tv = true_values(self.world, tasks)
        for r, tid in enumerate(geom.ids):
            geom.value[r] = tv[int(tid)]
