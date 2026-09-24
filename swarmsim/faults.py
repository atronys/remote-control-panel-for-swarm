"""FaultManager (docs/02 §2.1, docs/05 P3).

  scheduled — отказы/деградации по списку сценария (тесты, демо);
  random    — λ·N случайных агентов отказывают в случайные моменты из окна t_range
              (λ = faults.rate, «доля агентов за миссию» — фактор эксперимента docs/03).
Кто и когда — из собственного потока RNG «faults»: при одном зерне отказы одинаковы
для всех распределителей (парный дизайн).

inject_due возвращает (agent_id, причина, FaultEvent).
"""
from __future__ import annotations

from .interfaces import FaultManager, register_faults
from .scenario import FaultEvent


@register_faults("scheduled")
class ScheduledFaults(FaultManager):
    def __init__(self, scenario, rng):
        super().__init__(scenario, rng)
        self._queue = sorted(scenario.faults.scheduled, key=lambda e: (e.t, e.agent))

    def inject_due(self, t, agents):
        due = []
        while self._queue and self._queue[0].t <= t + 1e-9:
            ev = self._queue.pop(0)
            due.append((ev.agent, "injected", ev))
        return due


@register_faults("random")
class RandomFaults(ScheduledFaults):
    def __init__(self, scenario, rng):
        super().__init__(scenario, rng)
        cfg = scenario.faults
        n = scenario.agents.count
        k = int(round(cfg.rate * n))
        who = rng.permutation(n)[:k]
        when = rng.uniform(cfg.t_range[0], cfg.t_range[1], size=k)
        dt = scenario.dt
        extra = [FaultEvent(t=round(float(tw) / dt) * dt, agent=int(a), detect=cfg.detect)
                 for a, tw in zip(who, when)]
        self._queue = sorted(self._queue + extra, key=lambda e: (e.t, e.agent))
        self.planned = [(e.t, e.agent) for e in extra]
