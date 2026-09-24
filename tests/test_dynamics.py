"""P3: отказы (с сообщением и тихие), деградация (SR-5), новые задачи (FR-9), локальный ремонт,
режимы, метрики восстановления."""
import pytest

from swarmsim import apply_overrides, run
from swarmsim.metrics import recovery_metrics


def _greedy(scn, **extra):
    return apply_overrides(scn, {"allocator": {"kind": "central_greedy"}, **extra})


def test_reported_failure_recovers_immediately(small):
    s = run(_greedy(small, **{"faults.scheduled": [{"t": 60, "agent": 1}]}), 1).summary
    assert s["recovered_frac"] == 1.0 and s["CR"] == 1.0
    assert s["T_recovery_max"] <= 0.2                     # следующий шаг симуляции
    assert s["N_dup"] == 0


def test_silent_failure_waits_for_heartbeat_timeout(small):
    scn = _greedy(small, **{"faults.scheduled": [{"t": 60, "agent": 1, "detect": "silent"}],
                            "faults.detect_timeout": 1.5})
    res = run(scn, 1)
    det = [e for e in res.events if e["type"] == "failure_detected"]
    assert len(det) == 1 and det[0]["t"] == pytest.approx(61.5)
    # до обнаружения задачи отказавшего не освобождались
    early = [e for e in res.events if e["type"] == "task_released" and e["agent"] == 1 and e["t"] < 61.5]
    assert not early
    s = res.summary
    assert s["recovered_frac"] == 1.0 and 1.5 <= s["T_recovery_max"] <= 1.7


def test_static_baseline_does_not_recover(small):
    s = run(apply_overrides(small, {"faults.scheduled": [{"t": 60, "agent": 1}]}), 1).summary
    assert s["recovered_frac"] == 0.0 and s["T_recovery_max"] is None and s["CR"] < 1.0


def test_random_faults_rate_and_pairing(small):
    """λ = 0.5 при N = 4 → ровно 2 отказа; при одном зерне — одинаковые для любых распределителей."""
    cfg = {"faults.kind": "random", "faults.rate": 0.5, "faults.t_range": [30, 90]}
    a = run(apply_overrides(small, cfg), 7)
    b = run(_greedy(small, **cfg), 7)
    fa = [(e["t"], e["agent"]) for e in a.events if e["type"] == "agent_failed"]
    fb = [(e["t"], e["agent"]) for e in b.events if e["type"] == "agent_failed"]
    assert len(fa) == 2 and fa == fb


def test_degradation_triggers_deviation_and_replan(small):
    scn = _greedy(small, **{"faults.scheduled": [{"t": 20, "agent": 0, "type": "degrade", "factor": 0.4}],
                            "agents.deviation_threshold": 15})
    res = run(scn, 1)
    kinds = [e["type"] for e in res.events]
    assert "degraded" in kinds and "plan_deviation" in kinds
    i = kinds.index("plan_deviation")
    assert any(e["type"] == "allocation" and e["reason"] == "trigger" for e in res.events[i:])
    modes = [e for e in res.events if e["type"] == "mode_change" and e["agent"] == 0]
    assert any(m["to"] == "DEGRADED" for m in modes)
    assert res.summary["CR"] == 1.0


def test_new_task_is_served_with_preemption(small):
    scn = _greedy(small, **{"world.new_tasks": [{"t": 100, "x": 300, "y": 300, "priority": 5}]})
    res = run(scn, 1)
    s = res.summary
    assert s["tasks_total"] == 5 and s["CR"] == 1.0
    assert s["T_response_new"] is not None and s["T_response_new"] <= 0.2
    assert s["T_newtask_done"] is not None


def test_static_ignores_new_task(small):
    scn = apply_overrides(small, {"world.new_tasks": [{"t": 100, "x": 300, "y": 300, "priority": 5}]})
    s = run(scn, 1).summary
    assert s["tasks_total"] == 5 and s["tasks_orphaned"] == 1 and s["T_response_new"] is None


def test_local_repair_releases_tail_instead_of_everything(small):
    """При нехватке энергии дрон отдаёт хвост плана и продолжает, а не бросает всё сразу."""
    scn = apply_overrides(small, {"agents.count": 1, "agents.battery_wh": 12,
                                  "safety.plan_validation": "log", "agents.local_repair": True})
    res = run(scn, 1)
    reasons = [e["reason"] for e in res.events if e["type"] == "task_released"]
    assert "energy_repair" in reasons
    assert res.summary["tasks_done"] >= 1
    assert "battery_depleted" not in {e["type"] for e in res.events}


def test_recovery_metrics_on_synthetic_log():
    ev = [
        {"t": 10.0, "type": "agent_failed", "agent": 1, "released": [5, 6]},
        {"t": 10.0, "type": "task_released", "agent": 1, "task": 5, "reason": "injected"},
        {"t": 10.0, "type": "task_released", "agent": 1, "task": 6, "reason": "injected"},
        {"t": 10.1, "type": "allocation"},
        {"t": 10.1, "type": "plan", "agent": 0, "tasks": [5], "current": None},
        {"t": 10.3, "type": "plan", "agent": 2, "tasks": [7, 6], "current": None},
        {"t": 50.0, "type": "task_started", "agent": 0, "task": 5},
        {"t": 90.0, "type": "task_started", "agent": 2, "task": 6},
    ]
    m = recovery_metrics(ev)
    assert m["recovered_frac"] == 1.0
    assert m["T_recovery_max"] == pytest.approx(0.3)
    assert m["T_resume_max"] == pytest.approx(80.0)
    assert m["T_realloc_max"] == pytest.approx(0.1)
