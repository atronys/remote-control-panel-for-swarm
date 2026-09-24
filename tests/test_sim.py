import pytest

from swarmsim import apply_overrides, run
from swarmsim.datatypes import TaskStatus
from swarmsim.interfaces import Allocator
from swarmsim.metrics import WALLCLOCK_FIELDS
from swarmsim.sim import AssignmentError, SimLoop


def _det(summary):
    return {k: v for k, v in summary.items() if k not in WALLCLOCK_FIELDS}


@pytest.mark.parametrize("motion", ["kinematic", "double_integrator"])
def test_determinism_same_seed_same_numbers(small, motion):
    scn = apply_overrides(small, {"agents.motion": motion})
    r1, r2 = run(scn, 3), run(scn, 3)
    assert _det(r1.summary) == _det(r2.summary)
    assert r1.events == r2.events
    assert r1.timeseries == r2.timeseries
    assert r1.trajectories == r2.trajectories


def test_different_seed_different_world(small):
    assert run(small, 1).events != run(small, 2).events


def test_baseline_completes_all_tasks_safely(small):
    s = run(small, 1).summary
    assert s["CR"] == 1.0
    assert s["valid"] and s["N_col"] == 0 and s["N_dup"] == 0 and s["N_geo"] == 0
    assert s["agents_landed"] == 4
    assert s["T_detect_50"] is not None and s["T_detect_50"] <= s["t_final"]


def test_failure_releases_tasks_and_static_baseline_loses_them(small):
    scn = apply_overrides(small, {"faults.scheduled": [{"t": 30, "agent": 1}]})
    res = run(scn, 1)
    s = res.summary
    assert s["agents_failed"] == 1
    assert s["CR"] < 1.0 and s["tasks_orphaned"] >= 1
    released = [e for e in res.events if e["type"] == "task_released" and e["agent"] == 1]
    assert released and all(e["reason"] == "injected" for e in released)


def test_low_battery_triggers_rtl(small):
    # валидатор плана только записывает нарушение — срабатывает второй рубеж: бортовой резерв
    scn = apply_overrides(small, {"agents.battery_wh": 6, "safety.plan_validation": "log"})
    res = run(scn, 1)
    kinds = {e["type"] for e in res.events}
    assert "plan_violation" in kinds
    assert "low_battery_rtl" in kinds
    assert "battery_depleted" not in kinds          # резерв сработал до разряда
    assert res.summary["agents_landed"] == 4


class _Reallocating(Allocator):
    """Тестовый распределитель: при освобождении задач раздаёт их живым агентам."""
    name = "test_realloc"

    def allocate(self, snapshot, tasks):
        live = sorted(a.id for a in snapshot.healthy_agents())
        plan = {a.id: ([a.current_task] if a.current_task is not None else []) + list(a.plan)
                for a in snapshot.agents if a.healthy}
        free = [t.id for t in tasks if t.status == TaskStatus.UNASSIGNED]
        for k, tid in enumerate(free):
            plan[live[k % len(live)]].append(tid)
        return plan

    def should_trigger(self, t, tasks, events):
        return any(e["type"] == "task_released" for e in events)


def test_core_supports_dynamic_reallocation(small):
    scn = apply_overrides(small, {"faults.scheduled": [{"t": 30, "agent": 1}]})
    s = run(scn, 1, allocator=_Reallocating(scn, {}, None)).summary
    assert s["CR"] == 1.0 and s["N_dup"] == 0 and s["allocations"] >= 2


class _Broken(Allocator):
    name = "broken"

    def allocate(self, snapshot, tasks):
        return {0: [0], 1: [0]}


def test_duplicate_assignment_is_rejected(small):
    with pytest.raises(AssignmentError):
        SimLoop(small, 1, allocator=_Broken(small, {}, None)).run()


def test_collisions_invalidate_run(small):
    scn = apply_overrides(small, {"agents.altitude_step": 0, "agents.home_spacing": 1})
    s = run(scn, 1).summary
    assert s["N_col"] > 0 and s["valid"] is False
