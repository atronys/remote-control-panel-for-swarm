"""P5: тактическое избегание (ORCA) — геометрия, границы поля, прозрачность, прогон без эшелонов."""
import numpy as np

from swarmsim import apply_overrides, run
from swarmsim.agents.safety import boundary_lines, orca_velocity


def _fly(pa, pb, va, vb, steps=80, dt=0.1, R=5.5):
    mind = np.inf
    for _ in range(steps):
        na, _ = orca_velocity(pa, va, tuple(va), [(pb, vb, R)], R, 5.0, dt, 15.0)
        nb, _ = orca_velocity(pb, vb, tuple(vb), [(pa, va, R)], R, 5.0, dt, 15.0)
        va, vb = np.array(na), np.array(nb)
        pa, pb = pa + va * dt, pb + vb * dt
        mind = min(mind, float(np.linalg.norm(pa - pb)))
    return mind


def test_head_on_keeps_separation():
    mind = _fly(np.array([0.0, 0.0]), np.array([60.0, 0.0]), np.array([12.0, 0.0]), np.array([-12.0, 0.0]))
    assert mind >= 2 * 5.5 * 0.95                     # сумма радиусов (с допуском на дискретизацию)


def test_crossing_keeps_separation():
    mind = _fly(np.array([0.0, 0.0]), np.array([30.0, -30.0]), np.array([12.0, 0.0]), np.array([0.0, 12.0]))
    assert mind >= 2 * 5.5 * 0.95


def test_no_conflict_means_unchanged_velocity():
    v, active = orca_velocity((0, 0), (12, 0), (12.0, 0.0), [((0, 200.0), (12, 0), 5.5)], 5.5, 5.0, 0.1, 15.0)
    assert np.allclose(v, (12, 0))


def test_boundary_line_forbids_leaving_field():
    walls = boundary_lines((500.0, 0.0), (1000.0, 1000.0), 1.0, 15.0)
    v, _ = orca_velocity((500, 0), (0, 0), (0.0, -10.0), [], 5.5, 5.0, 0.1, 15.0, walls)
    assert v[1] >= -1e-9                             # на границе y = 0 вниз нельзя


def test_single_layer_needs_orca(small):
    """Все на одной высоте: без ORCA возможны столкновения, с ORCA их нет и геозона соблюдена."""
    over = {"agents.count": 8, "agents.altitude_layers": 1, "allocator": {"kind": "central_greedy"}}
    s = run(apply_overrides(small, {**over, "safety.filter": "orca"}), 1).summary
    assert s["N_col"] == 0 and s["N_geo"] == 0 and s["valid"]
    assert s["N_conf"] > 0


def test_orca_transparent_with_layers(small):
    """Эшелоны развели всех: фильтр ни разу не вмешался, прогон идентичен прогону без фильтра."""
    over = {"allocator": {"kind": "central_greedy"}}
    a = run(apply_overrides(small, {**over, "safety.filter": "orca"}), 2).summary
    b = run(apply_overrides(small, over), 2).summary
    assert a["N_conf"] == 0 and a["makespan"] == b["makespan"] and a["U"] == b["U"]
