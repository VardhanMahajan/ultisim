"""Expected-value field V(x, y): how good it is for the offense to hold the disc at (x, y).

Phase 1 uses a hand-made heuristic. Phase 3/4 replaces it with a grid learned from simulation
(start possessions from many spots, measure score rate, smooth): see docs/SIMULATION.md.
"""
from __future__ import annotations

import numpy as np


class ValueField:
    def __init__(self, cfg, grid: np.ndarray | None = None):
        self.L = cfg.field.length
        self.W = cfg.field.width
        self.ez = cfg.field.endzone_depth
        self.grid = grid  # optional learned (nx, ny) grid over the field
        vc = cfg.value
        self.base, self.gain, self.power, self.edge = vc.base, vc.gain, vc.power, vc.edge_penalty

    def value(self, pts: np.ndarray) -> np.ndarray:
        """V for offense-frame points (..., 2). Points in the attacking end zone are worth 1."""
        x = pts[..., 0]
        y = pts[..., 1]
        if self.grid is not None:
            nx, ny = self.grid.shape
            ix = np.clip((x / self.L * nx).astype(int), 0, nx - 1)
            iy = np.clip((y / self.W * ny).astype(int), 0, ny - 1)
            v = self.grid[ix, iy]
        else:
            u = np.clip((x - self.ez) / (self.L - 2 * self.ez), 0.0, 1.0)
            v = self.base + self.gain * u ** self.power
            edge = np.minimum(y, self.W - y)
            v = v - self.edge * np.clip(1.0 - edge / 5.0, 0.0, 1.0)
        inside_ez = (x >= self.L - self.ez) & (x <= self.L) & (y >= 0) & (y <= self.W)
        return np.where(inside_ez, 1.0, v)

    def turnover_cost(self, pts: np.ndarray) -> np.ndarray:
        """How bad a turnover at these points is: the opponent's value from the mirrored spot."""
        mirrored = np.stack([self.L - pts[..., 0], self.W - pts[..., 1]], axis=-1)
        mirrored[..., 0] = np.clip(mirrored[..., 0], self.ez, self.L - self.ez - 0.01)
        return self.value(mirrored)
