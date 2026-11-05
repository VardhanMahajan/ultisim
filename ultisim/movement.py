"""Point-mass player kinematics and closed-form time-to-reach.

Players accelerate at `accel` but brake at `brake` (>= accel, scaled by agility). A sharp cut
therefore costs time naturally: velocity toward the old direction must be braked away first.
"""
from __future__ import annotations

import numpy as np


def brake_power(accel: np.ndarray, agility: np.ndarray, cfg) -> np.ndarray:
    m = cfg.movement
    return accel * (m.brake_base + m.brake_agility * agility)


def step(pos: np.ndarray, vel: np.ndarray, target: np.ndarray, speed: np.ndarray,
         accel: np.ndarray, brake: np.ndarray, top: np.ndarray, dt: float,
         frozen: np.ndarray | None = None, target_vel: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Advance all players one tick toward `target` (N,2) at desired `speed` (N,).

    Uses an "arrive" profile: the desired speed is capped by sqrt(2 * brake * distance) so players
    stop on their target instead of overshooting. `target_vel` (N,2) is the target's own velocity
    (e.g. a defender's read of their attacker): it is fed forward so trackers don't lag behind a
    moving target.
    """
    delta = target - pos
    dist = np.hypot(delta[:, 0], delta[:, 1])
    safe = np.maximum(dist, 1e-9)
    cap = np.minimum(speed, top)
    want = np.minimum(cap, np.sqrt(2.0 * brake * np.maximum(dist - 0.05, 0.0)))
    v_des = delta / safe[:, None] * want[:, None]
    if target_vel is not None:
        v_des = v_des + target_vel
        vd = np.hypot(v_des[:, 0], v_des[:, 1])
        over_cap = vd > cap
        if over_cap.any():
            v_des[over_cap] *= (cap[over_cap] / vd[over_cap])[:, None]
    dv = v_des - vel
    # Accelerating vs braking: the component of dv opposing current velocity uses brake power.
    vmag = np.hypot(vel[:, 0], vel[:, 1])
    vhat = np.where(vmag[:, None] > 1e-6, vel / np.maximum(vmag, 1e-9)[:, None], 0.0)
    along = np.sum(dv * vhat, axis=1)
    amax = np.where(along < 0.0, brake, accel)
    dvmag = np.hypot(dv[:, 0], dv[:, 1])
    scale = np.minimum(1.0, amax * dt / np.maximum(dvmag, 1e-9))
    new_vel = vel + dv * scale[:, None]
    nv = np.hypot(new_vel[:, 0], new_vel[:, 1])
    over = nv > top
    if over.any():
        new_vel[over] *= (top[over] / nv[over])[:, None]
    if frozen is not None and frozen.any():
        new_vel[frozen] = 0.0
    new_pos = pos + 0.5 * (vel + new_vel) * dt
    if frozen is not None and frozen.any():
        new_pos[frozen] = pos[frozen]
    return new_pos, new_vel


def time_to_reach(pos: np.ndarray, vel: np.ndarray, top: np.ndarray, accel: np.ndarray,
                  brake: np.ndarray, reach: np.ndarray, points: np.ndarray,
                  delay: np.ndarray | float = 0.0, paired: bool = False,
                  extra_reach: np.ndarray | float = 0.0) -> np.ndarray:
    """Approximate earliest arrival time of each player at each point.

    pos, vel: (P,2); top/accel/brake/reach/delay: (P,).
    paired=False: points (..., 2) shared by all players -> returns (P, ...).
    paired=True:  points (P, ..., 2), row i belongs to player i -> returns (P, ...).
    The player must close to within `reach` (+ `extra_reach`, broadcastable to the output shape).
    """
    P = pos.shape[0]
    extra = points.ndim - 2 if paired else points.ndim - 1
    shp = (P,) + (1,) * extra
    p = pos.reshape(shp + (2,))
    v = vel.reshape(shp + (2,))
    d_vec = points - p
    d = np.hypot(d_vec[..., 0], d_vec[..., 1])
    d_eff = np.maximum(d - reach.reshape(shp) - extra_reach, 0.0)
    u = d_vec / np.maximum(d, 1e-9)[..., None]
    v_par = np.sum(v * u, axis=-1)
    v_perp = np.abs(v[..., 0] * u[..., 1] - v[..., 1] * u[..., 0])
    a = accel.reshape(shp)
    b = brake.reshape(shp)
    vmax = top.reshape(shp)
    # Kill sideways and backward velocity first (braking), then accelerate along the line.
    t_turn = (np.maximum(-v_par, 0.0) + 0.5 * v_perp) / b
    v0 = np.clip(v_par, 0.0, None)
    d_acc = (vmax * vmax - v0 * v0) / (2.0 * a)
    t_acc_full = (vmax - v0) / a
    t_short = (-v0 + np.sqrt(v0 * v0 + 2.0 * a * d_eff)) / a
    t_long = t_acc_full + (d_eff - d_acc) / vmax
    t_run = np.where(d_eff <= d_acc, t_short, t_long)
    dl = np.asarray(delay).reshape(shp) if np.ndim(delay) else delay
    # Reaction delay only matters if the player has to move: someone already in the spot is there.
    return np.where(d_eff <= 0.0, 0.0, t_turn + t_run + dl)
