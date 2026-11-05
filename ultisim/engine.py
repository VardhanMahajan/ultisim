"""Fast engine: Python front-end to the C core (csim/). Same rules as engine_ref.py.

    from ultisim.engine import CEngine
    eng = CEngine(cfg)
    res = eng.possessions(offense_genome, defense_genome, roster_pairs, seeds)     # structured array
    res = eng.run(offenses, defenses, specs)                                        # general batch

Games run in parallel inside C (OpenMP, all cores by default); Python only packs inputs.
"""
from __future__ import annotations

import ctypes
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import cabi
from .genome import DEFENSE_SCHEMA, OFFENSE_SCHEMA, to_vector
from .roster import Roster

ROOT = Path(__file__).resolve().parent.parent
SPEC_DTYPE = np.dtype(cabi.UsGameSpec)
RESULT_DTYPE = np.dtype(cabi.UsResult)
OUTCOMES = cabi.OUTCOMES


def _get(cfg, path: str):
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return float(obj)


def make_config(cfg) -> cabi.UsConfig:
    """Flatten the TOML config (+ learned throw model) into the C config struct."""
    c = cabi.UsConfig()
    for name, path in cabi.CONFIG_FIELDS:
        if path != "<computed>":
            setattr(c, name, _get(cfg, path))
    model_path = getattr(cfg.throws, "learned_model", "")
    full = ROOT / model_path if model_path else None
    if full is not None and full.exists():
        d = json.loads(full.read_text())
        from .engine_ref import FEATURE_NAMES
        assert d["features"] == FEATURE_NAMES, "throw model features changed - refit"
        c.use_model = 1.0
        for i, w in enumerate(d["weights"]):
            setattr(c, f"model_w{i}", float(w))
    else:
        c.use_model = 0.0
    return c


def pack_offenses(genomes) -> ctypes.Array:
    arr = (cabi.UsOffense * len(genomes))()
    for i, g in enumerate(genomes):
        v = np.ascontiguousarray(to_vector(g, OFFENSE_SCHEMA), dtype=np.float64)
        ctypes.memmove(ctypes.byref(arr, i * ctypes.sizeof(cabi.UsOffense)), v.ctypes.data, v.nbytes)
    return arr


def pack_defenses(genomes) -> ctypes.Array:
    arr = (cabi.UsDefense * len(genomes))()
    for i, g in enumerate(genomes):
        v = np.ascontiguousarray(to_vector(g, DEFENSE_SCHEMA), dtype=np.float64)
        ctypes.memmove(ctypes.byref(arr, i * ctypes.sizeof(cabi.UsDefense)), v.ctypes.data, v.nbytes)
    return arr


def roster_matrix(rosters: list[Roster]) -> np.ndarray:
    """(n, 7, n_stats) array in cabi.PLAYER_FIELDS order."""
    return np.stack([np.stack([getattr(r, k) for k in cabi.PLAYER_FIELDS], axis=1) for r in rosters])


def make_specs(team0: np.ndarray, team1: np.ndarray, seeds, off_idx=(0, -1), def_idx=(-1, 0), mode: int = 0,
               start_xy: tuple[float, float] | None = None) -> np.ndarray:
    """team0/team1: (n, 7, n_stats) roster matrices. off_idx/def_idx: per-team genome index, scalars or (n,) arrays."""
    n = len(seeds)
    specs = np.zeros(n, dtype=SPEC_DTYPE)
    players = specs["players"]                     # (n, 14) structured
    for k, name in enumerate(cabi.PLAYER_FIELDS):
        players[name][:, :7] = team0[:, :, k]
        players[name][:, 7:] = team1[:, :, k]
    specs["off_idx"][:, 0] = off_idx[0]
    specs["off_idx"][:, 1] = off_idx[1]
    specs["def_idx"][:, 0] = def_idx[0]
    specs["def_idx"][:, 1] = def_idx[1]
    specs["mode"] = mode
    specs["seed"] = np.asarray(seeds, dtype=np.uint64)
    if start_xy is None:
        specs["start_x"] = 0.0
        specs["start_y"] = -1.0
    else:
        specs["start_x"], specs["start_y"] = start_xy
    return specs


class CEngine:
    def __init__(self, cfg, threads: int = 0):
        self.cfg = cfg
        self.ccfg = make_config(cfg)
        self.lib = cabi.lib()
        # explicit thread count: scripts pin OMP_NUM_THREADS=1 for numpy, which OpenMP would otherwise obey
        self.threads = threads or (os.cpu_count() or 1)
        self.openmp = bool(self.lib.us_has_openmp())
        assert SPEC_DTYPE.itemsize == ctypes.sizeof(cabi.UsGameSpec)
        assert RESULT_DTYPE.itemsize == ctypes.sizeof(cabi.UsResult)

    def run(self, offenses, defenses, specs: np.ndarray) -> np.ndarray:
        offs = offenses if isinstance(offenses, ctypes.Array) else pack_offenses(offenses)
        defs = defenses if isinstance(defenses, ctypes.Array) else pack_defenses(defenses)
        specs = np.ascontiguousarray(specs)
        out = np.zeros(len(specs), dtype=RESULT_DTYPE)

        def call(lo, hi, threads):
            rc = self.lib.us_sim_batch(
                ctypes.byref(self.ccfg), offs, len(offs), defs, len(defs),
                ctypes.cast(int(specs.ctypes.data) + int(lo) * SPEC_DTYPE.itemsize, ctypes.POINTER(cabi.UsGameSpec)), int(hi - lo),
                ctypes.cast(int(out.ctypes.data) + int(lo) * RESULT_DTYPE.itemsize, ctypes.POINTER(cabi.UsResult)), int(threads))
            if rc != 0:
                raise RuntimeError(f"us_sim_batch failed with code {rc}")

        n = len(specs)
        if self.openmp or self.threads <= 1 or n < 2 * self.threads:
            call(0, n, self.threads)
        else:
            # No OpenMP in this build (e.g. Apple clang without libomp): split the batch over Python threads.
            # ctypes releases the GIL during the C call, so this uses every core. Chunks interleave games of
            # different lengths for balance.
            k = self.threads * 4
            bounds = np.linspace(0, n, k + 1).astype(int)
            with ThreadPoolExecutor(self.threads) as ex:
                list(ex.map(lambda i: call(bounds[i], bounds[i + 1], 1), range(k)))
        return out

    def possessions(self, og, dg, roster_pairs, seeds, start_xy=None) -> np.ndarray:
        """One possession per (roster_a, roster_b) pair: team 0 (roster_a) runs `og` against `dg`."""
        A = roster_matrix([a for a, _ in roster_pairs])
        B = roster_matrix([b for _, b in roster_pairs])
        specs = make_specs(A, B, seeds, off_idx=(0, 0), def_idx=(0, 0), mode=0, start_xy=start_xy)
        return self.run([og], [dg], specs)

    def points(self, ogs, dgs, roster_pairs, seeds) -> np.ndarray:
        """Full points: team 0 receives. ogs/dgs = (team0 genome, team1 genome)."""
        A = roster_matrix([a for a, _ in roster_pairs])
        B = roster_matrix([b for _, b in roster_pairs])
        specs = make_specs(A, B, seeds, off_idx=(0, 1), def_idx=(0, 1), mode=1)
        return self.run(list(ogs), list(dgs), specs)


def outcome_name(code: int) -> str:
    return OUTCOMES[int(code)]
