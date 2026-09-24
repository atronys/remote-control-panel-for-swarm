import math

import numpy as np
import pytest

from swarmsim.agents.base import MotionCommand
from swarmsim.agents.kinematics import DoubleIntegratorMotion, KinematicMotion


def test_kinematic_constant_speed_no_overshoot():
    m = KinematicMotion([0, 0], alt=50, v_max=15)
    target = np.array([10.0, 0.0])
    m.step(0.1, MotionCommand.goto(target, 12))
    assert m.state.p == pytest.approx([1.2, 0.0])
    for _ in range(20):
        m.step(0.1, MotionCommand.goto(target, 12))
    assert m.state.p == pytest.approx(target)          # не перелетает


def test_kinematic_velocity_is_clipped():
    m = KinematicMotion([0, 0], alt=50, v_max=15)
    m.step(1.0, MotionCommand.set_velocity([30, 40]))
    assert m.state.speed == pytest.approx(15)


def test_double_integrator_respects_limits():
    dt, a_max, v_max, yaw = 0.1, 3.0, 15.0, math.radians(90)
    m = DoubleIntegratorMotion([0, 0], 50, v_max, a_max, yaw)
    rng = np.random.default_rng(0)
    for _ in range(3000):
        v_old, psi_old = m.state.v.copy(), m.state.psi
        if rng.random() < 0.5:
            cmd = MotionCommand.goto(rng.uniform(-300, 300, 2), rng.uniform(1, 20))
        else:
            cmd = MotionCommand.set_velocity(rng.uniform(-20, 20, 2))
        m.step(dt, cmd)
        s = m.state
        assert np.linalg.norm(s.v - v_old) / dt <= a_max + 1e-9
        assert s.speed <= v_max + 1e-9
        if np.linalg.norm(v_old) >= m.turn_min_speed and s.speed > 1e-6:
            dpsi = (s.psi - psi_old + math.pi) % (2 * math.pi) - math.pi
            assert abs(dpsi) <= yaw * dt + 1e-9


def test_double_integrator_reaches_waypoint_and_stops():
    m = DoubleIntegratorMotion([0, 0], 50, 15, 3, math.radians(90))
    target = np.array([200.0, 50.0])
    for _ in range(1000):
        m.step(0.1, MotionCommand.goto(target, 12))
    assert np.linalg.norm(m.state.p - target) < 0.5
    assert m.state.speed < 0.5


def test_double_integrator_is_slower_than_kinematic():
    """Разгон и торможение делают переход длиннее, чем в безмассовой модели."""
    target = np.array([300.0, 0.0])

    def steps(m):
        for k in range(10_000):
            if np.linalg.norm(m.state.p - target) < 1.0:
                return k
            m.step(0.1, MotionCommand.goto(target, 12))

    slow = steps(DoubleIntegratorMotion([0, 0], 50, 15, 3, math.radians(90)))
    assert slow > steps(KinematicMotion([0, 0], 50, 15))
