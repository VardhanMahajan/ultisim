"""Throughput benchmark for the C engine (run after every engine change).

    python scripts/bench.py              # all threads
    python scripts/bench.py --threads 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultisim.config import load_config  # noqa: E402
from ultisim.engine import CEngine, make_specs, pack_defenses, pack_offenses, roster_matrix  # noqa: E402
from ultisim.genome import CANONICAL_DEFENSES, CANONICAL_OFFENSES  # noqa: E402
from ultisim.roster import sample_roster  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config()
    eng = CEngine(cfg, threads=a.threads)
    rng = np.random.default_rng(1)
    rosters = [sample_roster(rng, cfg) for _ in range(512)]
    R = roster_matrix(rosters)
    offs = pack_offenses([f() for f in CANONICAL_OFFENSES.values()])
    defs = pack_defenses([f() for f in CANONICAL_DEFENSES.values()])
    i = np.arange(a.n)
    specs = make_specs(R[i % 512], R[(i * 7 + 3) % 512], i, off_idx=(i % len(offs), 0), def_idx=(0, i % len(defs)))
    eng.run(offs, defs, specs[:64])  # warm up
    t = time.perf_counter()
    res = eng.run(offs, defs, specs)
    el = time.perf_counter() - t
    print(f"{a.n} possessions in {el:.2f}s = {a.n / el:,.0f} possessions/s "
          f"(threads={a.threads or 'all'}; mean {res['throws'].mean():.1f} throws, score {np.mean(res['outcome'] == 0):.3f})")


if __name__ == "__main__":
    main()
