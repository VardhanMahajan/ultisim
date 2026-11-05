"""Label evolved strategies with a structure family, from what the genome says the formation looks like.

Offense: place the disc in the middle of the field and compute the 6 slot home positions (force-aware
frame: dx downfield, dy toward the open side). Cutter slots (not handlers) define the structure:
  vertical   - cutters strung out along the field, roughly in the middle
  side       - strung out along the field, but off to one side
  horizontal - cutters spread across the field at a similar depth
  iso        - one cutter isolated well in front of a clump of the others
  spread     - cutters scattered (no dominant axis)
  other      - degenerate shapes (e.g. fewer than 2 cutters)
Descriptors are also returned so the dashboard / MAP-Elites archive can use the continuous values.
"""
from __future__ import annotations

import numpy as np

from .genome import FORCES, DefenseGenome, OffenseGenome

FAMILIES = ["vertical", "horizontal", "side", "iso", "spread", "other"]


def offense_homes(g: OffenseGenome, width: float = 37.0) -> np.ndarray:
    """Slot homes relative to a disc at the centre of the field (anchor blend is then irrelevant)."""
    return np.stack([np.asarray(g.home_dx, float), np.asarray(g.home_dy, float)], axis=1)


def offense_descriptors(g: OffenseGenome) -> dict:
    H = offense_homes(g)
    handler = np.asarray(g.handler) > 0.5
    cut = H[~handler]
    d = {"n_handlers": int(handler.sum()), "n_cutters": int(len(cut))}
    if len(cut) == 0:
        d.update(depth=0.0, spread_x=0.0, spread_y=0.0, lateral=0.0, iso_gap=0.0)
        return d
    d["depth"] = float(cut[:, 0].mean())
    d["spread_x"] = float(cut[:, 0].std())
    d["spread_y"] = float(cut[:, 1].std())
    d["lateral"] = float(cut[:, 1].mean())
    # iso: the cutter nearest the disc vs the centroid of the rest
    if len(cut) >= 3:
        order = np.argsort(cut[:, 0])
        front, rest = cut[order[0]], cut[order[1:]]
        d["iso_gap"] = float(np.hypot(*(rest.mean(0) - front)))
        d["iso_rest_spread"] = float(np.hypot(rest[:, 0].std(), rest[:, 1].std()))
    else:
        d["iso_gap"] = 0.0
        d["iso_rest_spread"] = 0.0
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def offense_family(g: OffenseGenome, d: dict | None = None) -> str:
    d = d or offense_descriptors(g)
    if d["n_cutters"] < 2:
        return "other"
    sx, sy = d["spread_x"], d["spread_y"]
    if d["n_cutters"] >= 3 and d["iso_gap"] > 9.0 and d["iso_rest_spread"] < 5.5 and sx + sy > 4.0:
        return "iso"
    if sx >= 1.6 * sy and sx > 2.0:
        return "vertical" if abs(d["lateral"]) < 6.0 else "side"
    if sy >= 1.6 * sx and sy > 2.0:
        return "horizontal"
    return "spread"


def defense_scheme(g: DefenseGenome) -> str:
    if g.is_zone:
        return "zone"
    if g.is_flex:
        return f"flex_{g.n_zone}z{7 - g.n_zone}p"
    return f"person_{g.force_name}"


DEF_SCHEMES = [f"person_{f}" for f in FORCES] + [f"flex_{k}z{7 - k}p" for k in range(1, 7)] + ["zone"]
