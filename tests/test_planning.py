"""P2: модель маршрута, валидатор плана, распределители B0/B2/B3, U(M)."""
import itertools

import numpy as np
import pytest

from swarmsim import apply_overrides, run
from swarmsim.alloc.central import greedy_insert, ruin_recreate, total_score
from swarmsim.alloc.cpsat import solve_cpsat
from swarmsim.datatypes import TaskStatus
from swarmsim.planning.routing import RouteModel, TaskGeom, agent_context
from swarmsim.sim import SimLoop


def _setup(small, n=2, prior="hotspots"):
    scn = apply_overrides(small, {"agents.count": n, "world.prior.kind": prior})
    sim = SimLoop(scn, 1)
    snap = sim.snapshot(0.0)
    ids = [t.id for t in sim.tasks]
    geom = TaskGeom.build(sim.tasks, ids)
    model = RouteModel(geom, scn.planning.discount)
    ctxs = [agent_context(a, sim.tasks, sim.energy, scn.agents.reserve, scn.energy.E_land) for a in snap.agents]
    return sim, model, ctxs, ids


def test_insertion_gain_matches_full_evaluation(small):
    """Быстрая формула прироста (суффиксные суммы) = разность полных оценок."""
    _, model, ctxs, ids = _setup(small)
    route = [0, 2]
    base = model.evaluate(ctxs[0], route)
    gains, pos = model.insertion(ctxs[0], route, np.array([1, 3]))
    for g, p, j in zip(gains, pos, [1, 3]):
        cand = route[:p] + [j] + route[p:]
        assert g == pytest.approx(model.evaluate(ctxs[0], cand).score - base.score, abs=1e-9)
        # и это действительно лучшая позиция
        best = max(model.evaluate(ctxs[0], route[:q] + [j] + route[q:]).score for q in range(3))
        assert base.score + g == pytest.approx(best, abs=1e-9)


def test_greedy_is_feasible_and_complete(small):
    _, model, ctxs, ids = _setup(small, n=2)
    routes = greedy_insert(model, ctxs, [[] for _ in ctxs], list(range(len(ids))))
    flat = [r for rt in routes for r in rt]
    assert sorted(flat) == list(range(len(ids)))                 # все задачи, без дублей
    assert all(model.evaluate(c, r).feasible for c, r in zip(ctxs, routes))


def test_energy_limit_leaves_tasks_unassigned(small):
    """При малой батарее жадный аукцион не берёт задачи, на которые не хватит энергии."""
    scn = apply_overrides(small, {"agents.count": 1, "agents.battery_wh": 8})
    sim = SimLoop(scn, 1)
    snap = sim.snapshot(0.0)
    ids = [t.id for t in sim.tasks]
    model = RouteModel(TaskGeom.build(sim.tasks, ids), scn.planning.discount)
    ctx = agent_context(snap.agents[0], sim.tasks, sim.energy, scn.agents.reserve, scn.energy.E_land)
    routes = greedy_insert(model, [ctx], [[]], list(range(len(ids))))
    assert model.evaluate(ctx, routes[0]).feasible
    assert len(routes[0]) < len(ids)


def test_cpsat_matches_brute_force_on_tiny_instance(small):
    """Оптимальность CP-SAT: 1 агент, 4 задачи — полный перебор 4! порядков."""
    _, model, ctxs, ids = _setup(small, n=1)
    ctx = ctxs[0]
    best = max(model.evaluate(ctx, list(p)).score for p in itertools.permutations(range(len(ids)))
               if model.evaluate(ctx, list(p)).feasible)
    sol, info = solve_cpsat(model, [ctx], [[]], list(range(len(ids))), time_limit=20, workers=1)
    assert info["status"] == "OPTIMAL"
    assert model.evaluate(ctx, sol[0]).score == pytest.approx(best, rel=2e-3)   # округление времени до 1 с


def test_lns_never_worse_than_start(small):
    _, model, ctxs, ids = _setup(small, n=2)
    g = greedy_insert(model, ctxs, [[] for _ in ctxs], list(range(len(ids))))
    lns = ruin_recreate(model, ctxs, g, np.random.default_rng(0), iters=200)
    assert total_score(model, ctxs, lns) >= total_score(model, ctxs, g) - 1e-12
    assert all(model.evaluate(c, r).feasible for c, r in zip(ctxs, lns))


@pytest.mark.parametrize("kind", ["central_greedy", "central_optimal", "static_oracle"])
def test_new_allocators_run_valid(small, kind):
    scn = apply_overrides(small, {"allocator": {"kind": kind, "time_limit": 2, "lns_iters": 50}})
    s = run(scn, 1).summary
    assert s["CR"] == 1.0 and s["valid"] and s["N_dup"] == 0
    assert s["N_plan_energy"] == 0                  # энергетически недопустимых планов не было


def test_central_reallocates_after_failure(small):
    """B2 пересчитывает план по событию отказа — в отличие от B1."""
    scn = apply_overrides(small, {"allocator": {"kind": "central_greedy"},
                                  "faults.scheduled": [{"t": 30, "agent": 1}]})
    res = run(scn, 1)
    assert res.summary["CR"] == 1.0 and res.summary["allocations"] >= 2
    assert res.summary["tasks_orphaned"] == 0


def test_validator_repairs_infeasible_plan(small):
    """Валидатор отрезает хвост плана, на который не хватает энергии (SR-6)."""
    from swarmsim.interfaces import Allocator

    class Greedy(Allocator):
        name = "all_to_one"

        def allocate(self, snapshot, tasks):
            return {0: [t.id for t in tasks]}

    scn = apply_overrides(small, {"agents.battery_wh": 8})
    res = run(scn, 1, allocator=Greedy(scn, {}, None))
    kinds = {e["type"] for e in res.events}
    assert "plan_violation" in kinds and "plan_repaired" in kinds
    assert res.summary["N_plan_energy"] >= 1
    assert "battery_depleted" not in kinds


def test_mission_utility_formula(small):
    res = run(small, 1)
    s = res.summary
    o = small.objective
    manual = s["U_ontime"] - o.lambda_E * s["E_total_Wh"] - o.lambda_T * s["makespan"]
    assert s["U"] == pytest.approx(manual, abs=1e-5)
    assert 0 < s["U_norm"] <= 1


def test_task_resumes_from_progress(small):
    """Переназначенная задача продолжается с места остановки (task.progress)."""
    scn = apply_overrides(small, {"allocator": {"kind": "central_greedy"},
                                  "faults.scheduled": [{"t": 60, "agent": 0}]})
    res = run(scn, 1)
    released = [e for e in res.events if e["type"] == "task_released" and e["agent"] == 0]
    assert released
    tid = released[0]["task"]
    started = [e for e in res.events if e["type"] == "task_started" and e["task"] == tid]
    assert len(started) == 2                                   # начали, отказ, продолжил другой
    sim_tasks = SimLoop(scn, 1)
    assert res.summary["CR"] == 1.0
    assert all(e["agent"] != 0 for e in started[1:])
