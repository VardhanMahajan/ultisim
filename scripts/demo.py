"""Phase 1 demo: canonical offenses vs canonical defenses on the reference engine.

    python scripts/demo.py                      # 100 possessions per matchup, 8 processes
    python scripts/demo.py -n 400 --workers 8
    python scripts/demo.py --replays-only       # just write example replays to runs/demo/

Prints a score-rate table with 95% confidence intervals and per-matchup stats, and writes
example replays (one possession per matchup + a full point) for the viewer.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one math thread per worker process

import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultisim.config import load_config  # noqa: E402
from ultisim.engine_ref import Game, simulate_point, simulate_possession  # noqa: E402
from ultisim.genome import CANONICAL_DEFENSES, CANONICAL_OFFENSES  # noqa: E402
from ultisim.roster import sample_roster  # noqa: E402

OFFENSES = ["vertical", "horizontal", "side", "iso"]
DEFENSES = ["person_forehand", "person_backhand", "person_straight_up", "person_middle", "cup_zone", "wall_zone"]


def rosters_for(seed: int, cfg):
    """Common random numbers: possession k uses the same two rosters for every matchup."""
    rng = np.random.default_rng(10_000 + seed)
    return sample_roster(rng, cfg), sample_roster(rng, cfg)


def run_chunk(args):
    on, dn, seeds = args
    cfg = load_config()
    out = []
    for k in seeds:
        ra, rb = rosters_for(k, cfg)
        g = Game(cfg, (ra, rb), (CANONICAL_OFFENSES[on](), None), (None, CANONICAL_DEFENSES[dn]()), seed=k)
        r = g.run_possession()
        n_ok = sum(t["outcome"] in ("catch", "score") for t in g.throw_log)
        out.append((r.outcome, r.throws, n_ok, r.hucks, r.huck_completions, r.breaks, r.break_completions,
                    r.duration, r.max_stall))
    return on, dn, out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def write_replays(cfg, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for on in OFFENSES:
        for dn in ["person_forehand", "cup_zone"]:
            # pick a seed whose possession is reasonably long (more to watch)
            for k in range(30):
                ra, rb = rosters_for(k, cfg)
                res, rec = simulate_possession(cfg, ra, rb, CANONICAL_OFFENSES[on](), CANONICAL_DEFENSES[dn](),
                                               seed=k, record=True, title=f"{on} stack vs {dn}")
                if res.throws >= 6 or k == 29:
                    paths.append(rec.save(outdir / f"{on}_vs_{dn}.json"))
                    break
    ra, rb = rosters_for(7, cfg)
    _, rec = simulate_point(cfg, (ra, rb), (CANONICAL_OFFENSES["vertical"](), CANONICAL_OFFENSES["horizontal"]()),
                            (CANONICAL_DEFENSES["cup_zone"](), CANONICAL_DEFENSES["person_forehand"]()), seed=7,
                            record=True, title="Full point: vertical/cup zone vs horizontal/person FH")
    paths.append(rec.save(outdir / "full_point_vertical_vs_horizontal.json"))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100, help="possessions per matchup")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replays-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    paths = write_replays(cfg, ROOT / "runs" / "demo")
    print(f"wrote {len(paths)} replays to runs/demo/")
    if a.replays_only:
        return
    chunk = max(1, a.n // a.workers)
    jobs = [(on, dn, list(range(i, min(i + chunk, a.n)))) for on in OFFENSES for dn in DEFENSES
            for i in range(0, a.n, chunk)]
    t0 = time.perf_counter()
    agg: dict = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for on, dn, out in ex.map(run_chunk, jobs):
            agg.setdefault((on, dn), []).extend(out)
    el = time.perf_counter() - t0
    total = sum(len(v) for v in agg.values())
    print(f"\n{total} possessions in {el:.0f}s ({total / el:.1f}/s on {a.workers} processes)\n")
    print("Score rate per possession from the goal line (95% CI):")
    print(f"{'':12s}" + "".join(f"{d:>22s}" for d in DEFENSES))
    for on in OFFENSES:
        row = f"{on:12s}"
        for dn in DEFENSES:
            o = agg[(on, dn)]
            p, lo, hi = wilson(sum(x[0] == "score" for x in o), len(o))
            row += f"{p:9.2f} [{lo:.2f}-{hi:.2f}]  "
        print(row)
    print("\nOverall per offense / defense:")
    for on in OFFENSES:
        o = [x for dn in DEFENSES for x in agg[(on, dn)]]
        print(f"  {on:12s} score {np.mean([x[0] == 'score' for x in o]):.3f}")
    for dn in DEFENSES:
        o = [x for on in OFFENSES for x in agg[(on, dn)]]
        print(f"  {dn:20s} conceded {np.mean([x[0] == 'score' for x in o]):.3f}")
    allx = [x for v in agg.values() for x in v]
    thr = sum(x[1] for x in allx)
    print("\nRealism check (all matchups):")
    print(f"  completion rate      {sum(x[2] for x in allx) / max(thr, 1):.3f}   (club target ~0.90-0.95)")
    print(f"  throws / possession  {np.mean([x[1] for x in allx]):.1f}")
    sc = [x for x in allx if x[0] == "score"]
    print(f"  throws / score       {np.mean([x[1] for x in sc]) if sc else float('nan'):.1f}   (target ~8-15)")
    print(f"  huck completion      {sum(x[4] for x in allx) / max(sum(x[3] for x in allx), 1):.3f}  ({sum(x[3] for x in allx)} hucks; target ~0.45-0.60)")
    print(f"  break completion     {sum(x[6] for x in allx) / max(sum(x[5] for x in allx), 1):.3f}  ({sum(x[5] for x in allx)} breaks)")
    print(f"  mean duration        {np.mean([x[7] for x in allx]):.1f}s")
    print(f"  turnover causes      {dict(Counter(x[0] for x in allx if x[0] != 'score').most_common())}")


if __name__ == "__main__":
    main()
