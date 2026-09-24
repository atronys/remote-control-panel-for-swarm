"""Локальный слой безопасности (тактическое избегание, R2, MVP-3).

none — команда без изменений (стратегическая деконфликтуация: эшелоны по высоте, docs/02).
orca — ORCA (van den Berg et al., «Reciprocal n-body collision avoidance», 2011), порт RVO2:
    для каждого соседа на том же эшелоне (|Δh| < vertical_separation) в радиусе обзора строится
    полуплоскость допустимых скоростей (половина ответственности на каждом — взаимность);
    ближайшая к желаемой скорость, удовлетворяющая всем полуплоскостям и |v| ≤ v_max,
    находится линейным программированием в 2D (LP2/LP3 из RVO2). Если ничего не мешает —
    команда возвращается как есть (фильтр прозрачен и не меняет траектории без конфликтов).

Соседи — по бортовому обзору (sense_range), а не по каналу связи: тактическое избегание не
должно зависеть от доставки сообщений.
"""
from __future__ import annotations

import math

import numpy as np

from ..interfaces import SafetyFilter, register_safety
from .base import MotionCommand

_EPS = 1e-9


@register_safety("none")
class PassthroughSafety(SafetyFilter):
    def filter(self, me, cmd, neighbors):
        return cmd


# ------------------------------------------------------------------ 2D LP из RVO2
def _det(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _lp1(lines, no, radius, opt, dir_opt):
    p, d = lines[no]
    dot = p[0] * d[0] + p[1] * d[1]
    disc = dot * dot + radius * radius - (p[0] * p[0] + p[1] * p[1])
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t_left, t_right = -dot - sq, -dot + sq
    for i in range(no):
        pi, di = lines[i]
        den = _det(d, di)
        num = _det(di, (p[0] - pi[0], p[1] - pi[1]))
        if abs(den) <= _EPS:
            if num < 0.0:
                return None
            continue
        t = num / den
        if den >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None
    if dir_opt:
        t = t_right if opt[0] * d[0] + opt[1] * d[1] > 0.0 else t_left
    else:
        t = min(max(d[0] * (opt[0] - p[0]) + d[1] * (opt[1] - p[1]), t_left), t_right)
    return (p[0] + t * d[0], p[1] + t * d[1])


def _lp2(lines, radius, opt, dir_opt):
    if dir_opt:
        res = (opt[0] * radius, opt[1] * radius)
    elif opt[0] * opt[0] + opt[1] * opt[1] > radius * radius:
        n = math.hypot(opt[0], opt[1])
        res = (opt[0] / n * radius, opt[1] / n * radius)
    else:
        res = (opt[0], opt[1])
    for i, (p, d) in enumerate(lines):
        if _det(d, (p[0] - res[0], p[1] - res[1])) > 0.0:
            r = _lp1(lines, i, radius, opt, dir_opt)
            if r is None:
                return i, res
            res = r
    return len(lines), res


def _lp3(lines, begin, radius, res):
    dist = 0.0
    for i in range(begin, len(lines)):
        pi, di = lines[i]
        if _det(di, (pi[0] - res[0], pi[1] - res[1])) > dist:
            proj = []
            for j in range(i):
                pj, dj = lines[j]
                det = _det(di, dj)
                if abs(det) <= _EPS:
                    if di[0] * dj[0] + di[1] * dj[1] > 0.0:
                        continue
                    pt = (0.5 * (pi[0] + pj[0]), 0.5 * (pi[1] + pj[1]))
                else:
                    t = _det(dj, (pi[0] - pj[0], pi[1] - pj[1])) / det
                    pt = (pi[0] + t * di[0], pi[1] + t * di[1])
                dx, dy = dj[0] - di[0], dj[1] - di[1]
                n = math.hypot(dx, dy)
                if n <= _EPS:
                    continue
                proj.append((pt, (dx / n, dy / n)))
            fail, r = _lp2(proj, radius, (-di[1], di[0]), True)
            if fail >= len(proj):
                res = r
            dist = _det(di, (pi[0] - res[0], pi[1] - res[1]))
    return res


def orca_lines(p, v, others, radius: float, tau: float, dt: float):
    """Полуплоскости ORCA: список (точка, направление). others — [(p_j, v_j, r_j)]."""
    lines = []
    inv_tau, inv_dt = 1.0 / tau, 1.0 / dt
    for pj, vj, rj in others:
        rp = (pj[0] - p[0], pj[1] - p[1])
        rv = (v[0] - vj[0], v[1] - vj[1])
        dsq = rp[0] * rp[0] + rp[1] * rp[1]
        R = radius + rj
        Rsq = R * R
        if dsq > Rsq:
            w = (rv[0] - inv_tau * rp[0], rv[1] - inv_tau * rp[1])
            wsq = w[0] * w[0] + w[1] * w[1]
            dot1 = w[0] * rp[0] + w[1] * rp[1]
            if dot1 < 0.0 and dot1 * dot1 > Rsq * wsq:
                wl = math.sqrt(wsq)
                uw = (w[0] / wl, w[1] / wl)
                direction = (uw[1], -uw[0])
                k = R * inv_tau - wl
                u = (k * uw[0], k * uw[1])
            else:
                leg = math.sqrt(dsq - Rsq)
                if _det(rp, w) > 0.0:
                    direction = ((rp[0] * leg - rp[1] * R) / dsq, (rp[0] * R + rp[1] * leg) / dsq)
                else:
                    direction = (-(rp[0] * leg + rp[1] * R) / dsq, -(-rp[0] * R + rp[1] * leg) / dsq)
                dot2 = rv[0] * direction[0] + rv[1] * direction[1]
                u = (dot2 * direction[0] - rv[0], dot2 * direction[1] - rv[1])
        else:
            # уже внутри суммы радиусов: разойтись за один шаг
            w = (rv[0] - inv_dt * rp[0], rv[1] - inv_dt * rp[1])
            wl = math.hypot(w[0], w[1])
            if wl <= _EPS:
                continue
            uw = (w[0] / wl, w[1] / wl)
            direction = (uw[1], -uw[0])
            k = R * inv_dt - wl
            u = (k * uw[0], k * uw[1])
        lines.append(((v[0] + 0.5 * u[0], v[1] + 0.5 * u[1]), direction))
    return lines


def boundary_lines(p, size, horizon: float, reach: float):
    """Границы поля как полуплоскости скоростей: v·n ≥ −d/horizon для каждой близкой стены
    (n — нормаль внутрь поля, d — расстояние до стены). Дальние стены (d > reach) не мешают."""
    lines = []
    for n, d in (((1.0, 0.0), p[0]), ((-1.0, 0.0), size[0] - p[0]),
                 ((0.0, 1.0), p[1]), ((0.0, -1.0), size[1] - p[1])):
        if d <= reach:
            k = -max(d, 0.0) / horizon
            lines.append(((n[0] * k, n[1] * k), (n[1], -n[0])))
    return lines


def orca_velocity(p, v, v_pref, others, radius: float, tau: float, dt: float, v_max: float, walls=()):
    lines = list(walls) + orca_lines(p, v, others, radius, tau, dt)
    if not lines:
        return (v_pref[0], v_pref[1]), False
    fail, res = _lp2(lines, v_max, v_pref, False)
    if fail < len(lines):
        res = _lp3(lines, fail, v_max, res)
    return res, True


@register_safety("orca")
class OrcaSafety(SafetyFilter):
    """ORCA на горизонтальной плоскости среди соседей своего эшелона."""

    def __init__(self, scenario):
        super().__init__(scenario)
        s = scenario.safety
        self.radius = 0.5 * s.orca_radius_factor * s.min_separation
        self.tau = s.orca_tau
        self.dt = scenario.dt
        self.v_max = scenario.agents.v_max
        self.size = tuple(float(x) for x in scenario.world.size)
        self.n_conf = 0            # N_conf: шаги, на которых фильтр изменил команду

    def _preferred(self, me, cmd: MotionCommand):
        if cmd.velocity is not None:
            return (float(cmd.velocity[0]), float(cmd.velocity[1]))
        if cmd.target is None:
            return (0.0, 0.0)
        d = cmd.target - me.p
        dist = math.hypot(d[0], d[1])
        if dist < _EPS:
            return (0.0, 0.0)
        sp = min(cmd.speed, self.v_max, dist / self.dt)
        return (d[0] / dist * sp, d[1] / dist * sp)

    def filter(self, me, cmd, neighbors):
        if not neighbors:
            return cmd
        others = [(nb.p, nb.v, self.radius) for nb in neighbors]
        v_pref = self._preferred(me, cmd)
        # границы поля — жёстче соседей: уклоняясь, не вылететь за геозону
        walls = boundary_lines(me.p, self.size, 1.0, self.v_max * 1.0)
        v_new, _ = orca_velocity(me.p, me.v, v_pref, others, self.radius, self.tau, self.dt, self.v_max, walls)
        if abs(v_new[0] - v_pref[0]) + abs(v_new[1] - v_pref[1]) < 1e-6:
            return cmd
        self.n_conf += 1
        return MotionCommand.set_velocity(np.array(v_new))
