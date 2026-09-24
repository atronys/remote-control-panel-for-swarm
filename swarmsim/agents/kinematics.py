"""Модели движения (docs/02, раздел 3.1).

KinematicMotion         — MVP-0: безмассовая, мгновенная смена курса.
DoubleIntegratorMotion  — MVP-2: ṗ = v, v̇ = a, |a| ≤ a_max, |v| ≤ v_max, |ψ̇| ≤ ψ̇_max.

6-DOF, аэродинамику и моторы сознательно не моделируем — это даст PX4 SITL в фазе P6.
"""
from __future__ import annotations

import math

import numpy as np

from ..datatypes import VehicleState
from .base import MotionCommand, MotionSource

_EPS = 1e-9


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = math.hypot(v[0], v[1])
    if n > limit:
        return v * (limit / n)
    return v


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class KinematicMotion(MotionSource):
    """p(t+dt) = p(t) + v_c · normalize(wp − p) · dt, без перелёта за точку."""

    def __init__(self, p0, alt: float, v_max: float, psi0: float = 0.0):
        self.v_max = v_max
        self._state = VehicleState(p=np.array(p0, dtype=float), v=np.zeros(2), psi=psi0, alt=alt)

    def step(self, dt: float, cmd: MotionCommand) -> VehicleState:
        s = self._state
        if cmd.velocity is not None:
            v = _clip_norm(np.asarray(cmd.velocity, dtype=float), self.v_max)
        elif cmd.target is not None:
            d = cmd.target - s.p
            dist = math.hypot(d[0], d[1])
            if dist < _EPS:
                v = np.zeros(2)
            else:
                speed = min(cmd.speed, self.v_max, dist / dt)
                v = d * (speed / dist)
        else:
            v = np.zeros(2)
        s.p = s.p + v * dt
        s.v = v
        if math.hypot(v[0], v[1]) > _EPS:
            s.psi = math.atan2(v[1], v[0])
        return s


class DoubleIntegratorMotion(MotionSource):
    """Двойной интегратор с ограничениями ускорения, скорости и скорости поворота курса.

    Наведение на точку: желаемая скорость направлена на цель, её модуль ограничен
    профилем торможения sqrt(2·a_max·d), чтобы подходить к точке без перелёта.
    """

    def __init__(self, p0, alt: float, v_max: float, a_max: float,
                 yaw_rate_max: float, psi0: float = 0.0, turn_min_speed: float = 0.5):
        self.v_max = v_max
        self.a_max = a_max
        self.yaw_rate_max = yaw_rate_max
        self.turn_min_speed = turn_min_speed   # ниже этой скорости мультиротор разворачивается на месте
        self._state = VehicleState(p=np.array(p0, dtype=float), v=np.zeros(2), psi=psi0, alt=alt)

    def _desired_velocity(self, dt: float, cmd: MotionCommand) -> np.ndarray:
        s = self._state
        if cmd.velocity is not None:
            return _clip_norm(np.asarray(cmd.velocity, dtype=float), self.v_max)
        if cmd.target is None:
            return np.zeros(2)
        d = cmd.target - s.p
        dist = math.hypot(d[0], d[1])
        if dist < _EPS:
            return np.zeros(2)
        speed = min(cmd.speed, self.v_max, math.sqrt(2.0 * self.a_max * dist), dist / dt)
        return d * (speed / dist)

    def step(self, dt: float, cmd: MotionCommand) -> VehicleState:
        s = self._state
        v_old = s.v
        v_des = self._desired_velocity(dt, cmd)

        a = _clip_norm((v_des - v_old) / dt, self.a_max)
        v_new = _clip_norm(v_old + a * dt, self.v_max)
        v_new = self._limit_turn(v_old, v_new, self.yaw_rate_max * dt)

        s.p = s.p + 0.5 * (v_old + v_new) * dt
        s.v = v_new
        if math.hypot(v_new[0], v_new[1]) > _EPS:
            s.psi = math.atan2(v_new[1], v_new[0])
        return s

    def _limit_turn(self, v_old: np.ndarray, v_new: np.ndarray, max_dpsi: float) -> np.ndarray:
        # Поворот v_new к курсу v_old при том же модуле только уменьшает |v_new − v_old|,
        # поэтому ограничение |a| ≤ a_max сохраняется.
        s_old = math.hypot(v_old[0], v_old[1])
        s_new = math.hypot(v_new[0], v_new[1])
        if s_old < self.turn_min_speed or s_new < _EPS:
            return v_new
        a_old = math.atan2(v_old[1], v_old[0])
        dpsi = _wrap(math.atan2(v_new[1], v_new[0]) - a_old)
        if abs(dpsi) <= max_dpsi:
            return v_new
        heading = a_old + math.copysign(max_dpsi, dpsi)
        return np.array([s_new * math.cos(heading), s_new * math.sin(heading)])
