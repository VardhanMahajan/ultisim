"""Statistical parity: C engine vs. Python reference on the canonical matchups.

    python scripts/parity.py --n-py 48 --n-c 1500
    python scripts/parity.py --defs person_forehand watch4:26     # <run>:<index> = an evolved defense from runs/<run>/state.pkl

Both engines play the same rosters (common random numbers for players); their game RNGs differ, so
we compare statistics. A metric FAILS if |C - Python| > 3 combined standard errors.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultisim.config import load_config  # noqa: E402
from ultisim.engine import OUTCOMES, CEngine, make_specs, pack_defenses, pack_offenses, roster_matrix  # noqa: E402
from ultisim.engine_ref import Game  # noqa: E402
from ultisim.genome import CANONICAL_DEFENSES, CANONICAL_OFFENSES, DEFENSE_SCHEMA, DefenseGenome, from_vector  # noqa: E402
from ultisim.roster import sample_roster  # noqa: E402

OFF = ["vertical", "horizontal", "side", "iso"]
DEF = ["person_forehand", "person_backhand", "person_middle", "cup_zone"]
FIELDS = ["outcome", "throws", "completions", "hucks", "huck_completions", "breaks", "break_completions", "duration"]


def get_defense(name):
    """A canonical defense by name, or '<run>:<index>' for member <index> of a run's checkpointed defense population."""
    if name in CANONICAL_DEFENSES:
        return CANONICAL_DEFENSES[name]()
    run, idx = name.rsplit(":", 1)
    with open(ROOT / "runs" / run / "state.pkl", "rb") as f:
        st = pickle.load(f)
    return from_vector(np.asarray(st["dfn"][int(idx)], float), DEFENSE_SCHEMA, DefenseGenome, name=name)


def rosters(k, cfg):
    rng = np.random.default_rng(int(os.environ.get("PARITY_ROSTER_SEED", 90_000)) + k)
    return sample_roster(rng, cfg), sample_roster(rng, cfg)


def py_job(args):
    on, dn, seeds = args
    cfg = load_config()
    rows = []
    for k in seeds:
        ra, rb = rosters(k, cfg)
        g = Game(cfg, (ra, rb), (CANONICAL_OFFENSES[on](), None), (None, get_defense(dn)), seed=k)
        r = g.run_possession()
        rows.append((on, dn, OUTCOMES.index(r.outcome), r.throws, r.completions, r.hucks, r.huck_completions,
                     r.breaks, r.break_completions, r.duration))
    return rows


def summarize(rows):
    """rows: list of tuples (on, dn, outcome, throws, comps, hucks, hcomps, breaks, bcomps, duration)."""
    a = np.array([r[2:] for r in rows], dtype=float)
    out, thr, comp, hk, hkc, br, brc, dur = a.T
    tos = out != 0

    def ratio(num, den):
        """ratio estimate with a delta-method standard error over possessions"""
        n = len(num)
        R = num.sum() / max(den.sum(), 1e-9)
        resid = num - R * den
        se = np.sqrt(np.sum(resid ** 2) / max(n - 1, 1) / n) / max(den.mean(), 1e-9)
        return R, se

    def mean(x):
        return x.mean(), x.std(ddof=1) / np.sqrt(len(x))

    m = {"score_rate": mean((out == 0).astype(float)), "completion": ratio(comp, thr),
         "throws_per_poss": mean(thr), "huck_completion": ratio(hkc, hk), "huck_share": ratio(hk, thr),
         "break_share": ratio(br, thr), "duration_s": mean(dur)}
    for name, codes in [("d_share", (1, 2, 7)), ("drop_share", (3,)), ("throwaway_share", (4, 5))]:
        m[name] = ratio(np.isin(out, codes).astype(float), tos.astype(float))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-py", type=int, default=48)
    ap.add_argument("--n-c", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--off", nargs="*", default=OFF)
    ap.add_argument("--defs", nargs="*", default=DEF)
    a = ap.parse_args()
    cfg = load_config()

    t0 = time.perf_counter()
    chunk = max(1, a.n_py // 2)
    jobs = [(on, dn, list(range(i, min(i + chunk, a.n_py)))) for on in a.off for dn in a.defs for i in range(0, a.n_py, chunk)]
    py_rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(py_job, jobs):
            py_rows += r
    t_py = time.perf_counter() - t0

    eng = CEngine(cfg)
    offs = pack_offenses([CANONICAL_OFFENSES[o]() for o in OFF])
    defs = pack_defenses([get_defense(d) for d in a.defs])
    # C replays exactly Python's rosters (many game seeds each) so both estimate the same quantity
    pairs = [rosters(k % a.n_py, cfg) for k in range(a.n_c)]
    A = roster_matrix([p[0] for p in pairs])
    B = roster_matrix([p[1] for p in pairs])
    c_rows = []
    t0 = time.perf_counter()
    for oi, on in enumerate(OFF):
        if on not in a.off:
            continue
        for di, dn in enumerate(a.defs):
            seeds = np.arange(a.n_c) + 1_000_003 * (oi * 4 + di + 1)
            res = eng.run(offs, defs, make_specs(A, B, seeds, off_idx=(oi, 0), def_idx=(0, di)))
            c_rows += [(on, dn, *[res[f][i] for f in FIELDS]) for i in range(len(res))]
    t_c = time.perf_counter() - t0

    print(f"Python reference: {len(py_rows)} possessions in {t_py:.0f}s ({len(py_rows) / t_py:.1f}/s)")
    print(f"C engine:         {len(c_rows)} possessions in {t_c:.1f}s ({len(c_rows) / t_c:.0f}/s)\n")
    fails = 0
    for label, sel in [("ALL", lambda r: True)] + [(f"off={o}", lambda r, o=o: r[0] == o) for o in a.off] \
            + [(f"def={d}", lambda r, d=d: r[1] == d) for d in a.defs]:
        mp = summarize([r for r in py_rows if sel(r)])
        mc = summarize([r for r in c_rows if sel(r)])
        lines = []
        for k in mp:
            (vp, sp), (vc, sc) = mp[k], mc[k]
            z = (vc - vp) / max(np.hypot(sp, sc), 1e-9)
            flag = "FAIL" if abs(z) > 3 else ("~" if abs(z) > 2 else "ok")
            fails += flag == "FAIL"
            if label == "ALL" or flag != "ok":
                lines.append(f"  {k:16s} python {vp:7.3f} +-{sp:.3f}   C {vc:7.3f} +-{sc:.3f}   z={z:+5.1f}  {flag}")
        if lines:
            print(label)
            print("\n".join(lines))
    print(f"\n{'PARITY OK' if fails == 0 else f'{fails} metric(s) outside 3 SE'}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
