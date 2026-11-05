"""Exploit audit: re-evaluate a run's defense population against the canonical offenses with many
possessions and print the strongest genomes in full.

    python scripts/audit_defenses.py --run smoke -n 400
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ultisim.classify import defense_scheme  # noqa: E402
from ultisim.config import load_config  # noqa: E402
from ultisim.engine import CEngine, make_specs, pack_defenses, pack_offenses, roster_matrix  # noqa: E402
from ultisim.genome import CANONICAL_DEFENSES, CANONICAL_OFFENSES, DEFENSE_SCHEMA, DefenseGenome, from_vector  # noqa: E402
from ultisim.roster import sample_roster  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("-n", type=int, default=400, help="possessions per (defense, offense)")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    st = pickle.load(open(ROOT / "runs" / a.run / "state.pkl", "rb"))
    cfg = load_config()
    eng = CEngine(cfg)
    dg = [from_vector(v, DEFENSE_SCHEMA, DefenseGenome) for v in st["dfn"]] + [f() for f in CANONICAL_DEFENSES.values()]
    labels = [f"pop{i}" for i in range(len(st["dfn"]))] + list(CANONICAL_DEFENSES)
    offs = pack_offenses([f() for f in CANONICAL_OFFENSES.values()])
    defs = pack_defenses(dg)
    rng = np.random.default_rng(7)
    pairs = [(sample_roster(rng, cfg), sample_roster(rng, cfg)) for _ in range(a.n)]
    A, B = roster_matrix([p[0] for p in pairs]), roster_matrix([p[1] for p in pairs])
    conc = np.zeros(len(dg))
    br = np.zeros(len(dg))
    for di in range(len(dg)):
        s, b, t = 0.0, 0.0, 0.0
        for oi in range(len(offs)):
            r = eng.run(offs, defs, make_specs(A, B, np.arange(a.n) + 1000 * oi, off_idx=(oi, 0), def_idx=(0, di)))
            s += (r["outcome"] == 0).mean() / len(offs)
            b += r["breaks"].sum()
            t += r["throws"].sum()
        conc[di], br[di] = s, b / max(t, 1)
    order = np.argsort(conc)
    print(f"{'defense':18s} {'scheme':20s} {'conceded':>8s} {'break%':>7s}")
    for i in order:
        print(f"{labels[i]:18s} {defense_scheme(dg[i]):20s} {conc[i]:8.3f} {br[i] * 100:7.1f}")
    for i in order[:a.top]:
        print(f"\n{labels[i]} full genome:")
        for name, n, lo, hi in DEFENSE_SCHEMA:
            v = getattr(dg[i], name)
            print(f"  {name:14s} {np.round(v, 2) if n > 1 else round(float(v), 3)}   [{lo}, {hi}]")


if __name__ == "__main__":
    main()
