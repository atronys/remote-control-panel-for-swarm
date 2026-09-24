"""static_roundrobin — наивный базовый метод B0: задачи по кругу, один раз, без перераспределения.

Владелец — R2. Нужен ядру как эталонная реализация интерфейса Allocator.
"""
from __future__ import annotations

from ..interfaces import Allocator, register_allocator
from ..datatypes import Assignment, TaskSet, TaskStatus, WorldSnapshot


@register_allocator("static_roundrobin")
class StaticRoundRobin(Allocator):
    def allocate(self, snapshot: WorldSnapshot, tasks: TaskSet) -> Assignment:
        agents = sorted(a.id for a in snapshot.healthy_agents())
        plan: Assignment = {aid: [] for aid in agents}
        if not agents:
            return plan
        todo = [t.id for t in tasks if t.status not in (TaskStatus.DONE, TaskStatus.FAILED)]
        for k, tid in enumerate(todo):
            plan[agents[k % len(agents)]].append(tid)
        return plan
