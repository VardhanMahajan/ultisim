"""Player stats and roster sampling.

A roster is 7 players. Stats are drawn from the distributions in [players] of the config,
correlated through two latent factors:
  * athletic factor -> top speed, acceleration, vertical, agility (partly)
  * handler factor  -> throw skill, break skill, ranges, release time, decision quality
Archetypes shift the factor means (e.g. a "fast" team has athletic mean +1 sd).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.special import ndtr

N_PER_TEAM = 7

# stat name -> (factor, loading sign). Loading magnitude comes from config (*_corr).
_FACTOR = {
    "top_speed": ("ath", 1.0), "accel": ("ath", 1.0), "vertical": ("ath", 1.0), "agility": ("ath", 0.6),
    "throw_skill": ("han", 1.0), "break_skill": ("han", 1.0), "decision_noise": ("han", -0.8),
    "release_time": ("han", -0.6), "range_backhand": ("han", 0.8), "range_forehand": ("han", 0.8),
    "range_hammer": ("han", 0.6),
}
_NORMAL = ["top_speed", "accel", "reaction_time", "height"]
_UNIFORM = ["agility", "vertical", "hands", "throw_skill", "break_skill", "decision_noise", "marking",
            "release_time", "range_backhand", "range_forehand", "range_hammer"]

ARCHETYPES = {
    "random": {},
    "fast": {"ath": 1.0},
    "throwing": {"han": 1.0},
    "tall": {"height": 0.07},
    "slow_throwing": {"ath": -0.7, "han": 1.0},
}


@dataclass
class Roster:
    top_speed: np.ndarray
    accel: np.ndarray
    agility: np.ndarray
    reaction_time: np.ndarray
    height: np.ndarray
    vertical: np.ndarray
    hands: np.ndarray
    throw_skill: np.ndarray
    break_skill: np.ndarray
    decision_noise: np.ndarray
    marking: np.ndarray
    release_time: np.ndarray
    range_backhand: np.ndarray
    range_forehand: np.ndarray
    range_hammer: np.ndarray

    @classmethod
    def stat_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def player(self, i: int) -> dict:
        return {k: round(float(getattr(self, k)[i]), 3) for k in self.stat_names()}

    def concat(self, other: "Roster") -> "Roster":
        """14-player roster: self = ids 0-6, other = ids 7-13."""
        return Roster(**{k: np.concatenate([getattr(self, k), getattr(other, k)]) for k in self.stat_names()})

    def subset(self, idx) -> "Roster":
        return Roster(**{k: getattr(self, k)[idx].copy() for k in self.stat_names()})


def sample_roster(rng: np.random.Generator, cfg, archetype: str = "random", n: int = N_PER_TEAM) -> Roster:
    pc = cfg.players
    shift = ARCHETYPES[archetype]
    lat = {"ath": rng.standard_normal(n) + shift.get("ath", 0.0),
           "han": rng.standard_normal(n) + shift.get("han", 0.0)}
    load = {"ath": pc.athletic_corr, "han": pc.handler_corr}

    def z_for(name: str) -> np.ndarray:
        eps = rng.standard_normal(n)
        if name not in _FACTOR:
            return eps
        fac, sign = _FACTOR[name]
        rho = float(np.clip(load[fac] * abs(sign), 0.0, 0.99))
        return np.sign(sign) * rho * lat[fac] + np.sqrt(1.0 - rho * rho) * eps

    out = {}
    for name in _NORMAL:
        d = getattr(pc, name)
        v = d.mean + d.sd * z_for(name)
        if name == "height":
            v = v + shift.get("height", 0.0)
        out[name] = np.clip(v, d.lo, d.hi)
    for name in _UNIFORM:
        d = getattr(pc, name)
        u = ndtr(z_for(name))  # correlated uniform via the normal CDF
        out[name] = d.lo + u * (d.hi - d.lo)
    return Roster(**out)
