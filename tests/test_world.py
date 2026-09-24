import math

import numpy as np
import pytest

from swarmsim.geometry import point_in_polygon, segment_intersects_polygon
from swarmsim.scenario import Scenario, SensorConfig
from swarmsim.sensing import SensorModel
from swarmsim.sim import SimLoop, make_rngs
from swarmsim.world import World

SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)


def test_geometry():
    assert point_in_polygon([5, 5], SQUARE)
    assert not point_in_polygon([15, 5], SQUARE)
    assert segment_intersects_polygon([-5, 5], [15, 5], SQUARE)     # проходит насквозь
    assert not segment_intersects_polygon([-5, -5], [-5, 15], SQUARE)


def _world(**world):
    scn = Scenario.model_validate({"world": world})
    r = make_rngs(1)
    return World(scn, r["world"], r["sensing"])


def test_nofly_and_bounds():
    w = _world(size=[100, 100], nofly=[[[40, 40], [60, 40], [60, 60], [40, 60]]], targets=50)
    assert w.in_nofly([50, 50]) and not w.in_nofly([10, 10])
    assert not w.segment_is_free([10, 50], [90, 50])
    assert w.segment_is_free([10, 10], [90, 10])
    assert not w.segment_is_free([10, 10], [110, 10])               # выход за границы
    assert not any(w.in_nofly(p) for p in w.target_pos)             # цели не в бесполётной зоне


def test_targets_inside_bounds_and_prior_normalized():
    w = _world(size=[500, 300], prior={"kind": "hotspots", "n": 2, "sigma": 50}, targets=200)
    assert w.prior.sum() == pytest.approx(1.0)
    assert (w.target_pos >= 0).all()
    assert (w.target_pos[:, 0] <= 500).all() and (w.target_pos[:, 1] <= 300).all()


def test_world_is_paired_across_agent_count():
    """Парный дизайн: то же зерно → тот же мир при любом N."""
    a = SimLoop(Scenario.model_validate({"agents": {"count": 3}}), seed=7).world
    b = SimLoop(Scenario.model_validate({"agents": {"count": 12}}), seed=7).world
    c = SimLoop(Scenario.model_validate({"agents": {"count": 3}}), seed=8).world
    assert np.array_equal(a.target_pos, b.target_pos)
    assert not np.array_equal(a.target_pos, c.target_pos)


def test_new_targets_appear_on_schedule():
    w = _world(targets=5, new_targets=[{"t": 1.0, "count": 3}])
    assert w.n_active_targets == 5
    for _ in range(9):
        w.step(0.1)
    assert w.n_active_targets == 5
    w.step(0.1)
    assert w.n_active_targets == 8


def test_swath_from_gsd():
    s = SensorModel(SensorConfig(fov_deg=60, pixel_pitch_m=3e-6, focal_length_m=8e-3,
                                 gsd_target_m=0.02, gsd_max_m=0.05))
    assert s.gsd(50) == pytest.approx(0.01875)
    assert s.swath_width(50) == pytest.approx(2 * 50 * math.tan(math.radians(30)))  # GSD лучше целевого
    assert 0 < s.swath_width(100) < s.footprint(100)                                # хуже — полоса сужается
    assert s.swath_width(200) == 0.0                                                 # GSD > GSD_max


class _FakeAgent:
    id, airborne, alt = 0, True, 50.0

    def __init__(self, a, b):
        self.prev_p, self.p = np.array(a, float), np.array(b, float)


def test_observe_accumulates_pd_as_in_formula():
    """Pd_cum после нескольких проходов = 1 − exp(−W·ΣL/A_cell)."""
    w = _world(size=[100, 100], cell_size=100, targets=5)
    sensor = SensorModel(SensorConfig())
    W = sensor.swath_width(50)
    total = 0.0
    for x in range(0, 100, 10):
        w.observe(_FakeAgent([x, 50], [x + 10, 50]), sensor)
        total += 10
    expected = 1 - math.exp(-W * total / w.cell_area)
    assert w.coverage_map()[0, 0] == pytest.approx(expected)
    assert w.expected_detected_fraction() == pytest.approx(expected)


class _Pos:
    def __init__(self, i, p, alt, airborne=True):
        self.id, self.p, self.alt, self.airborne = i, np.array(p, float), alt, airborne


def test_separation_counts_entries_not_steps():
    w = _world()
    close = [_Pos(0, [0, 0], 50), _Pos(1, [3, 0], 50), _Pos(2, [3, 0], 56)]  # 2 — на другом эшелоне
    for _ in range(5):
        w.check_separation(close)
    assert w.n_collisions == 1 and w.n_near_miss == 1
    w.check_separation([_Pos(0, [0, 0], 50), _Pos(1, [100, 0], 50)])
    w.check_separation(close)
    assert w.n_collisions == 2
