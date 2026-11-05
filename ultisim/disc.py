"""Throw types, flight paths (quadratic Bezier), timing, error and mark-coverage geometry."""
from __future__ import annotations

import math

import numpy as np

THROW_TYPES = ("backhand", "forehand", "hammer")
BH, FH, HAM = 0, 1, 2


def type_cfg(cfg, t: int):
    return getattr(cfg.throws, THROW_TYPES[t])


def control_point(p0: np.ndarray, p1: np.ndarray, curve: np.ndarray, cfg) -> np.ndarray:
    """Bezier control point. curve = +1 bends the flight toward the thrower's left of the line, -1 right."""
    d_vec = p1 - p0
    d = np.hypot(d_vec[..., 0], d_vec[..., 1])
    left = np.stack([-d_vec[..., 1], d_vec[..., 0]], axis=-1) / np.maximum(d, 1e-9)[..., None]
    mid = 0.5 * (p0 + p1)
    return mid + left * (np.asarray(curve) * cfg.throws.curve.amount * d)[..., None]


def bezier(p0, c, p1, s):
    s = np.asarray(s)[..., None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * c + s * s * p1


def flight_time(cfg, ttype: np.ndarray, dist: np.ndarray, huck: np.ndarray, loft=None) -> np.ndarray:
    """Flight time. A lofted (touch) throw is slower and hangs longer: it goes over a defender in the lane,
    but gives everyone more time to close."""
    ttype = np.asarray(ttype)
    dist = np.asarray(dist, dtype=float)
    loft = np.zeros(np.shape(dist), bool) if loft is None else np.asarray(loft, bool)
    tcf = cfg.throws
    out = np.empty(np.shape(dist))
    for t in range(3):
        tc = type_cfg(cfg, t)
        m = ttype == t
        if not m.any():
            continue
        if t == HAM:
            out[m] = tc.hang + dist[m] / tc.speed
        else:
            hk = huck[m]
            lf = loft[m] & ~hk
            flat = tc.hang + dist[m] / tc.speed
            lofted = tc.hang + tcf.loft_hang_extra + dist[m] / (tc.speed * tcf.loft_speed_mult)
            out[m] = np.where(hk, tc.huck_hang + dist[m] / tc.huck_speed, np.where(lf, lofted, flat))
    return out


def throw_peak(cfg, ttype, dist, huck, loft) -> np.ndarray:
    """Peak height (m) of a throw: flat bullets stay low, lofted touch passes rise to 3-4 m, hucks and
    hammers go higher still."""
    hc = cfg.throws.height
    ttype = np.asarray(ttype)
    dist = np.asarray(dist, dtype=float)
    flat = hc.flat_base + hc.flat_per_m * dist
    lofted = np.minimum(hc.loft_base + hc.loft_per_m * dist, hc.loft_max)
    hk = hc.huck_base + hc.huck_per_m * dist
    hm = hc.hammer_base + hc.hammer_per_m * dist
    return np.where(ttype == HAM, hm, np.where(huck, hk, np.where(loft, lofted, flat)))


def flight_height(cfg, peak, s, T, h0=None):
    """Disc height at flight fraction s: a quadratic from release height through `peak` (reached at
    s = 0.5) to catch height at s = 1; during the tail (s > 1) it sinks toward the ground."""
    hc = cfg.throws.height
    h0 = hc.release if h0 is None else h0
    h1 = hc.catch
    peak = np.asarray(peak, dtype=float)
    s = np.asarray(s, dtype=float)
    ctrl = 2.0 * peak - 0.5 * (h0 + h1)
    sc = np.clip(s, 0.0, 1.0)
    h = (1 - sc) ** 2 * h0 + 2 * (1 - sc) * sc * ctrl + sc * sc * h1
    e = np.clip((s - 1.0) * T / cfg.throws.tail_time, 0.0, 1.0)
    return np.where(s > 1.0, h1 - (h1 - hc.ground) * e, h)


def unreachable_window(cfg, ttype: int, huck: bool) -> tuple[float, float]:
    """Fraction of the flight [lo, hi) during which the disc is too high to touch."""
    if ttype == HAM:
        return cfg.throws.hammer.unreachable_lo, cfg.throws.hammer.unreachable_hi
    if huck:
        return cfg.throws.huck.unreachable_lo, cfg.throws.huck.unreachable_hi
    return 0.0, 0.0


def height_profile(ttype: int, huck: bool, s: float) -> float:
    """Approximate disc height (m) for drawing and for 'is it catchable'."""
    peak = 6.0 if ttype == HAM else (5.0 if huck else 1.2)
    return 1.3 + peak * math.sin(math.pi * min(max(s, 0.0), 1.0)) * (1.0 - 0.3 * s)


def error_sigma(cfg, thrower_skill, ttype, dist, is_break, marked, curved) -> np.ndarray:
    tc = cfg.throws
    skill_mult = tc.skill_err_worst + (tc.skill_err_best - tc.skill_err_worst) * thrower_skill
    type_mult = np.choose(np.asarray(ttype), [tc.backhand.err_mult, tc.forehand.err_mult, tc.hammer.err_mult])
    sig = (tc.err_a + tc.err_b * dist + tc.err_c * dist * dist) * skill_mult * type_mult
    sig = sig * np.where(is_break, tc.break_err_mult, 1.0)
    sig = sig * np.where(marked, tc.marked_err_mult, 1.0)
    sig = sig * np.where(curved, tc.curve_err_mult, 1.0)
    return sig


def release_point(thrower_pos: np.ndarray, ttype: np.ndarray, cfg, pivot: float = 1.0) -> np.ndarray:
    """Release point offset to the thrower's left (+y) for backhands, right (-y) for forehands/hammers."""
    side = np.choose(np.asarray(ttype), [cfg.throws.backhand.release_side, cfg.throws.forehand.release_side,
                                         cfg.throws.hammer.release_side])
    rp = np.broadcast_to(thrower_pos, np.shape(side) + (2,)).copy()
    rp[..., 1] += side * pivot
    return rp


def angle_deg(v: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def wrap_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def mark_block_prob(cfg, thrower_pos, rel_pts, ttype, mark_center_deg, mark_half_width_deg,
                    marking, break_skill, fake_active: bool) -> np.ndarray:
    """Probability that a set marker blocks a throw.

    rel_pts: (..., 2) a point ~2 m along the early flight path for each candidate.
    Two parts: the marker's body cone (highest dead-centre, zero at the cone edge) and a break-side
    tail - a set marker shuffles and reaches for anything released toward their side, fading linearly
    to zero at mark_tail_deg off the marker (well before the open side). Fakes, pivots (curves) and hammers are how throwers beat it.
    """
    dc = cfg.defense
    ang = angle_deg(rel_pts - thrower_pos)
    delta = np.abs(wrap_deg(ang - mark_center_deg))
    cover = np.clip(1.0 - (delta / max(mark_half_width_deg, 1e-6)) ** 2, 0.0, 1.0)
    skill = (1.0 + dc.mark_skill_weight * (2.0 * marking - 1.0)) * (1.0 - dc.break_skill_weight * (2.0 * break_skill - 1.0) * 0.5)
    tmult = np.choose(np.asarray(ttype), [cfg.throws.backhand.mark_mult, cfg.throws.forehand.mark_mult,
                                          cfg.throws.hammer.mark_mult])
    fake = (0.35 + 0.4 * marking) if fake_active else 1.0
    # fixed blocking capacity: a wedge wider than the reference spreads the same arms/legs thinner
    spread = min(1.0, dc.mark_width_ref / max(mark_half_width_deg, 1e-6))
    tail = dc.mark_break_tail * np.clip(1.0 - delta / dc.mark_tail_deg, 0.0, 1.0) * (1.0 - cover)
    return np.clip(dc.mark_block_max * (spread * cover + tail) * skill * tmult * fake, 0.0, 0.95)


def flight_point(p0, c, p1, T: float, s, tail_time: float):
    """Disc position at flight fraction s. For s > 1 the disc floats on along its final tangent
    (slowing down) for up to `tail_time` seconds before it hits the ground."""
    s = np.asarray(s, dtype=float)
    base = bezier(p0, c, p1, np.minimum(s, 1.0))
    tan = p1 - c
    tn = math.hypot(tan[0], tan[1])
    u = tan / tn if tn > 1e-9 else np.zeros(2)
    v_end = 2.0 * tn / max(T, 1e-6)          # Bezier speed at s = 1
    e = np.clip((s - 1.0) * T, 0.0, tail_time)
    glide = v_end * (e - 0.35 * e * e / max(tail_time, 1e-6))
    return base + u * glide[..., None]
