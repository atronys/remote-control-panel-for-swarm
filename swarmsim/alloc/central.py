"""Централизованные распределители (docs/03, раздел 2): общая база и B2 `central_greedy`.

Централизованный распределитель видит снимок роя (что знает наземная станция), строит
маршруты всем и пересчитывает их по событиям. Общая часть:

  • участники торгов — исправные агенты в воздухе или на старте, не ушедшие домой по заряду;
  • текущая задача агента закреплена за ним (в торгах не участвует), маршрут продолжается
    от её выхода — время и энергия остатка учтены (planning.routing.agent_context);
  • политика «кого переназначать» (param replan):
        full   — все невыполненные незакреплённые задачи распределяются заново;
        repair — очереди агентов сохраняются, вставляются только осиротевшие задачи
                 (меньше перестроений — «минимальное вмешательство»);
  • повод пересчитать — события (should_trigger): отказ, низкий заряд, освобождение задачи,
    появление новой задачи, отклонение от плана, агент свободен при наличии бесхозных задач.

B2 — последовательный аукцион «один лот за раз»: на каждом шаге среди всех пар
(агент, свободная задача) выбирается максимальный прирост целевой функции S при лучшей
позиции вставки (с проверкой энергии), задача вставляется, прирост пересчитывается только
у получившего её агента. Затем — локальное улучшение каждого маршрута (relocate).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..datatypes import Assignment, Mode, TaskSet, TaskStatus, WorldSnapshot
from ..interfaces import Allocator, register_allocator
from ..planning.routing import AgentCtx, RouteModel, TaskGeom, agent_context

TRIGGER_EVENTS = {"agent_failed", "failure_detected", "low_battery_rtl", "battery_depleted",
                  "new_task", "plan_deviation", "gcs_reconnect"}


class CentralAllocator(Allocator):
    """База: участники, контексты, пул задач, сборка назначения, триггеры."""

    def __init__(self, scenario, params: dict[str, Any], rng):
        super().__init__(scenario, params, rng)
        self.replan = params.get("replan", "full")
        self.interrupt = params.get("interrupt", 3.0)     # FR-9: приоритет, прерывающий текущую задачу
        self.do_improve = bool(params.get("improve", True))
        self.dynamic = bool(params.get("dynamic", True))
        self.world = None
        self.energy = None
        self.explain: dict = {}
        self.n_calls = 0

    def bind(self, world, energy) -> None:
        self.world, self.energy = world, energy

    # ---------------------------------------------------------------- когда пересчитывать
    def should_trigger(self, t: float, tasks: TaskSet, events: list[dict]) -> bool:
        if not self.dynamic:
            return False
        orphans = any(tk.status == TaskStatus.UNASSIGNED for tk in tasks if tk.t_appear <= t + 1e-9)
        for e in events:
            k = e["type"]
            if k in TRIGGER_EVENTS:
                if k == "agent_failed" and e.get("detect", "reported") == "silent":
                    continue                        # тихий отказ центр узнаёт только по таймауту
                return True
            if k == "task_released" and e.get("reason") not in ("replanned",):
                return True
            if k == "plan_complete_rtl" and orphans:
                return True
        return False

    # ---------------------------------------------------------------- общий каркас
    def values(self, tasks: TaskSet) -> None:
        """Ценности задач уже в task.priority; оракул B0 их переопределяет."""

    def allocate(self, snapshot: WorldSnapshot, tasks: TaskSet) -> Assignment:
        t_wall = time.perf_counter()
        self.n_calls += 1
        agents = list(snapshot.agents)
        reserve, e_land = self.scenario.agents.reserve, self.scenario.energy.E_land
        bidders = [a for a in agents if a.healthy and not a.low_battery
                   and a.mode in (Mode.IDLE, Mode.MISSION, Mode.RTL)]
        urgent_head, bidders = self._urgent(snapshot, tasks, bidders)
        preempted = {a.current_task for a in bidders if a.id in urgent_head and a.current_task is not None}
        locked = {a.current_task for a in agents if a.healthy and a.current_task is not None} - preempted
        locked |= set(urgent_head.values())
        visible = [tk for tk in tasks if tk.t_appear <= snapshot.t + 1e-9
                   and tk.status not in (TaskStatus.DONE, TaskStatus.FAILED) and tk.id not in locked]
        keep: dict[int, list[int]] = {a.id: [] for a in bidders}
        if self.replan == "repair":
            live_ids = {a.id for a in bidders}
            for a in bidders:
                keep[a.id] = [tid for tid in a.plan
                              if tid in tasks and tasks[tid].status not in (TaskStatus.DONE, TaskStatus.FAILED)
                              and tid not in locked and tasks[tid].owner in live_ids]
        kept = {tid for v in keep.values() for tid in v}
        pool_ids = [tk.id for tk in visible if tk.id not in kept]
        all_ids = sorted(kept) + pool_ids
        geom = TaskGeom.build(tasks, all_ids)
        model = RouteModel(geom, self.scenario.planning.discount)
        self.values_override(geom, tasks)
        ctxs = [self._context(a, tasks, urgent_head.get(a.id), reserve, e_land) for a in bidders]
        row = {tid: r for r, tid in enumerate(all_ids)}
        routes = [[row[tid] for tid in keep[a.id]] for a in bidders]
        pool = [row[tid] for tid in pool_ids]
        routes = self.solve(model, ctxs, routes, pool)

        assignment: Assignment = {}
        for a in agents:
            if not a.healthy:
                continue
            if a.id in urgent_head:
                head = [urgent_head[a.id]]
            else:
                head = [a.current_task] if a.current_task is not None else []
            assignment[a.id] = head
        for ctx, route in zip(ctxs, routes):
            assignment[ctx.id] = assignment.get(ctx.id, []) + [int(geom.ids[r]) for r in route]
        assigned = {tid for v in assignment.values() for tid in v}
        score = sum(model.evaluate(c, r).score for c, r in zip(ctxs, routes))
        self.explain = {"method": self.name, "replan": self.replan, "bidders": [c.id for c in ctxs],
                        **({"urgent": {str(k): v for k, v in urgent_head.items()}} if urgent_head else {}),
                        "pool": len(pool), "score": round(score, 6),
                        "unassigned": [t for t in pool_ids if t not in assigned],
                        "wall_ms": round(1000 * (time.perf_counter() - t_wall), 3)}
        return assignment

    def _urgent(self, snapshot, tasks, bidders):
        """FR-9: срочные задачи — ближайшему по времени подлёта агенту, с прерыванием текущей."""
        from dataclasses import replace
        heads: dict[int, int] = {}
        if not self.interrupt:
            return heads, bidders
        busy_tasks = {a.current_task for a in snapshot.agents if a.current_task is not None}
        urgent = sorted((tk for tk in tasks if tk.t_appear <= snapshot.t + 1e-9 and tk.kind == "point"
                         and tk.priority >= float(self.interrupt) and tk.id not in busy_tasks
                         and tk.status not in (TaskStatus.DONE, TaskStatus.FAILED)),
                        key=lambda tk: -tk.priority)
        free = {a.id: a for a in bidders if a.mode != Mode.IDLE or True}
        for tk in urgent:
            if not free:
                break
            entry, _ = TaskGeom.remaining(tk)
            best = min(free.values(), key=lambda a: (np.hypot(*(entry - a.p)) / a.v_cruise, a.id))
            heads[best.id] = tk.id
            del free[best.id]
        out = [replace(a, current_task=None, wp_idx=0) if a.id in heads else a for a in bidders]
        return heads, out

    def _context(self, a, tasks, urgent_tid, reserve, e_land):
        """Контекст агента; со срочной задачей — маршрут начинается после неё."""
        ctx = agent_context(a, tasks, self.energy, reserve, e_land)
        if urgent_tid is None:
            return ctx
        entry, length = TaskGeom.remaining(tasks[urgent_tid])
        d = float(np.hypot(*(entry - ctx.start))) + length
        ctx.t0 += d / ctx.v
        ctx.e_avail -= d * ctx.e_per_m
        ctx.start = tasks[urgent_tid].waypoints[-1].astype(float)
        ctx.locked = urgent_tid
        return ctx

    def values_override(self, geom: TaskGeom, tasks: TaskSet) -> None:
        """Хук для оракула: заменить ценности (по умолчанию — priority задач)."""

    def solve(self, model: RouteModel, ctxs: list[AgentCtx], routes: list[list[int]],
              pool: list[int]) -> list[list[int]]:
        raise NotImplementedError


def greedy_insert(model: RouteModel, ctxs: list[AgentCtx], routes: list[list[int]],
                  pool: list[int]) -> list[list[int]]:
    """Последовательный аукцион «один лот за раз» (B2) — используется и как подсказка для CP-SAT."""
    routes = [list(r) for r in routes]
    pool = list(pool)
    if not ctxs or not pool:
        return routes
    n = len(ctxs)
    evs = [model.evaluate(c, r) for c, r in zip(ctxs, routes)]
    gains = np.full((n, len(pool)), -np.inf)
    poss = np.zeros((n, len(pool)), dtype=int)
    for k in range(n):
        gains[k], poss[k] = model.insertion(ctxs[k], routes[k], np.asarray(pool), evs[k])
    while pool:
        flat = int(np.argmax(gains))
        k, c = divmod(flat, len(pool))
        if not np.isfinite(gains[k, c]) or gains[k, c] <= 0:
            break
        routes[k].insert(int(poss[k, c]), pool[c])
        del pool[c]
        gains = np.delete(gains, c, axis=1)
        poss = np.delete(poss, c, axis=1)
        if pool:
            evs[k] = model.evaluate(ctxs[k], routes[k])
            gains[k], poss[k] = model.insertion(ctxs[k], routes[k], np.asarray(pool), evs[k])
    return routes


@register_allocator("central_greedy")
class CentralGreedy(CentralAllocator):
    """B2: центральный жадный аукцион с пулом задач, пересчёт по событиям."""

    def solve(self, model, ctxs, routes, pool):
        routes = greedy_insert(model, ctxs, routes, pool)
        if self.do_improve:
            routes = [model.improve(c, r) for c, r in zip(ctxs, routes)]
        return routes


def total_score(model: RouteModel, ctxs: list[AgentCtx], routes: list[list[int]]) -> float:
    return float(sum(model.evaluate(c, r).score for c, r in zip(ctxs, routes)))


def ruin_recreate(model: RouteModel, ctxs: list[AgentCtx], routes: list[list[int]],
                  rng: np.random.Generator, iters: int = 400, frac: float = 0.3,
                  temp0: float = 0.01) -> list[list[int]]:
    """Ruin & Recreate LNS (Schrimpf et al., 2000): вынуть часть задач, вставить жадно заново.

    Разрушение: случайные задачи или «соседние» (ближайшие к случайной). Принятие —
    имитация отжига по относительному изменению S; хранится лучшее найденное.
    Используется как эталон «лучшее известное» для B3 там, где CP-SAT не успевает доказать оптимум.
    """
    g = model.g
    cur = [list(r) for r in routes]
    cur_s = best_s = total_score(model, ctxs, cur)
    best = [list(r) for r in cur]
    assigned = [r for rt in cur for r in rt]
    if len(assigned) < 2 or not ctxs:
        return best
    for it in range(iters):
        flat = [r for rt in cur for r in rt]
        if len(flat) < 2:
            break
        q = max(1, int(round(frac * len(flat) * rng.uniform(0.3, 1.0))))
        if rng.random() < 0.5:
            removed = set(rng.choice(flat, size=min(q, len(flat)), replace=False).tolist())
        else:
            seed = flat[int(rng.integers(len(flat)))]
            d = np.hypot(*(g.entry[flat] - g.entry[seed]).T)
            removed = {flat[i] for i in np.argsort(d)[:q]}
        cand = [[r for r in rt if r not in removed] for rt in cur]
        pool = sorted(removed, key=lambda _: rng.random())
        cand = greedy_insert(model, ctxs, cand, pool)
        s = total_score(model, ctxs, cand)
        temp = temp0 * max(cur_s, 1e-9) * (1 - it / iters)
        if s > cur_s or (temp > 0 and rng.random() < np.exp((s - cur_s) / temp)):
            cur, cur_s = cand, s
            if s > best_s + 1e-12:
                best, best_s = [list(r) for r in cand], s
    return best
