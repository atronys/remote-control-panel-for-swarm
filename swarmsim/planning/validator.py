"""PlanValidator (SR-6, docs/02 §2.1): план допустим только после проверки.

Проверки:
  energy     — назначенный маршрут + возврат домой укладываются в заряд с резервом (docs/01 2.2);
  energy_current — не хватает даже на остаток текущей задачи (не решение распределителя —
               срабатывает бортовая защита по заряду; в счётчик нарушений плана не входит);
  geofence   — перелёты между задачами не пересекают бесполётные зоны и не выходят за поле;
  duplicate  — задача назначена не более чем одному агенту;
  layer      — стратегическая деконфликтуация: у летящих агентов разные эшелоны (docs/00, S5).

`repair` отбрасывает хвост маршрута, пока энергия не станет допустимой — так распределитель,
ошибившийся с энергией, не может отправить дрон в полёт без возврата. Каждое срабатывание
записывается (P2: «ограничение энергии ни разу не нарушено» проверяется по этому счётчику).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..datatypes import Assignment, TaskSet, WorldSnapshot
from .routing import RouteModel, TaskGeom, agent_context

if TYPE_CHECKING:
    from ..scenario import Scenario
    from ..world import World


@dataclass
class Violation:
    kind: str
    agent: int
    detail: dict

    def as_event(self) -> dict:
        return {"violation": self.kind, "agent": self.agent, **self.detail}


class PlanValidator:
    def __init__(self, scenario: "Scenario", world: "World", energy):
        self.scn = scenario
        self.world = world
        self.energy = energy

    def check(self, snap: WorldSnapshot, assignment: Assignment, tasks: TaskSet) -> list[Violation]:
        out: list[Violation] = []
        seen: dict[int, int] = {}
        for aid, tids in assignment.items():
            for tid in tids:
                if tid in seen and seen[tid] != aid:
                    out.append(Violation("duplicate", aid, {"task": tid, "other": seen[tid]}))
                seen[tid] = aid
        by_id = {a.id: a for a in snap.agents}
        for aid, tids in assignment.items():
            a = by_id.get(aid)
            if a is None or not tids:
                continue
            ctx = agent_context(a, tasks, self.energy, self.scn.agents.reserve, self.scn.energy.E_land)
            free = [t for t in tids if t != a.current_task]
            geom = TaskGeom.build(tasks, free)
            model = RouteModel(geom, self.scn.planning.discount)
            ev = model.evaluate(ctx, list(range(len(free))))
            if not model.evaluate(ctx, []).feasible:
                # не хватает даже на остаток ТЕКУЩЕЙ задачи и возврат — это не решение распределителя,
                # а зона бортовой защиты по заряду (agent._check_battery уведёт домой)
                out.append(Violation("energy_current", aid, {"task": a.current_task,
                                                             "excess_j": round(ev.energy - ctx.e_avail, 1)}))
            elif not ev.feasible:
                out.append(Violation("energy", aid, {"excess_j": round(ev.energy - ctx.e_avail, 1),
                                                     "tasks": list(free)}))
            # перелёты: старт → вход первой, выход → вход следующей, выход последней → дом
            pts = [ctx.start]
            for r in range(len(free)):
                pts += [geom.entry[r], geom.exit[r]]
            pts.append(ctx.home)
            for k in range(0, len(pts) - 1, 2):
                if not self.world.segment_is_free(pts[k], pts[k + 1]):
                    out.append(Violation("geofence", aid, {"from": np.round(pts[k], 1).tolist(),
                                                           "to": np.round(pts[k + 1], 1).tolist()}))
        flying = [a for a in snap.agents if a.healthy and assignment.get(a.id)]
        vs = self.scn.safety.vertical_separation
        for i in range(len(flying)):
            for j in range(i + 1, len(flying)):
                if abs(flying[i].alt - flying[j].alt) < vs:
                    out.append(Violation("layer", flying[i].id, {"other": flying[j].id}))
        return out

    def repair(self, snap: WorldSnapshot, assignment: Assignment, tasks: TaskSet) -> tuple[Assignment, list[int]]:
        """Отбросить хвосты энергетически недопустимых маршрутов. Вернуть (план, снятые задачи)."""
        by_id = {a.id: a for a in snap.agents}
        dropped: list[int] = []
        fixed: Assignment = {}
        for aid, tids in assignment.items():
            a = by_id.get(aid)
            tids = list(tids)
            if a is None:
                fixed[aid] = tids
                continue
            ctx = agent_context(a, tasks, self.energy, self.scn.agents.reserve, self.scn.energy.E_land)
            head = [t for t in tids if t == a.current_task]
            free = [t for t in tids if t != a.current_task]
            while free:
                geom = TaskGeom.build(tasks, free)
                ev = RouteModel(geom, self.scn.planning.discount).evaluate(ctx, list(range(len(free))))
                if ev.feasible:
                    break
                dropped.append(free.pop())
            fixed[aid] = head + free
        return fixed, dropped
