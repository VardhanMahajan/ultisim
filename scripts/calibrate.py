"""Phase 2 calibration harness: canonical-vs-canonical stats against club-level target bands.

    python scripts/calibrate.py -n 60                       # possessions per matchup
    python scripts/calibrate.py -n 60 --points 150          # + full points for the hold rate
    python scripts/calibrate.py --set defense.block_prob=0.8 --set players.reaction_time.mean=0.3
    python scripts/calibrate.py --og risk_aversion=2.0 --og patience=1.2   # override all canonical offenses
    python scripts/calibrate.py --json runs/calib/last.json  # also save the metrics

Every override is a dotted path into configs/default.toml (or an offense/defense genome field),
so a sweep is just a loop around this script.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one math thread per worker process

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultisim.config import load_config  # noqa: E402
from ultisim.engine_ref import Game  # noqa: E402
from ultisim.genome import CANONICAL_DEFENSES, CANONICAL_OFFENSES  # noqa: E402
from ultisim.roster import sample_roster  # noqa: E402

OFFENSES = ["vertical", "horizontal", "side", "iso"]
DEFENSES = ["person_forehand", "person_backhand", "person_middle", "cup_zone"]

# (metric, lo, hi): club-level open, possessions starting at the offense's own goal line.
TARGETS = [
    ("completion", 0.90, 0.95),
    ("score_rate", 0.40, 0.60),       # per possession from the goal line
    ("hold_rate", 0.55, 0.72),        # full points: team receiving the pull scores first
    ("throws_per_score", 8.0, 15.0),
    ("huck_completion", 0.45, 0.62),
    ("huck_share", 0.04, 0.15),       # hucks / throws
    ("break_share", 0.05, 0.25),      # break-side throws / throws
    ("stall_share", 0.0, 0.08),       # stall-outs / turnovers
    ("drop_share", 0.10, 0.25),       # drops / turnovers
    ("d_share", 0.25, 0.50),          # blocks + interceptions + mark blocks / turnovers
    ("throwaway_share", 0.30, 0.60),  # throwaways + OOB / turnovers
    ("mean_stall_at_throw", 2.5, 5.0),
]


def parse_val(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def nested(pairs: list[str]) -> dict:
    out: dict = {}
    for kv in pairs:
        k, v = kv.split("=", 1)
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = parse_val(v)
    return out


def apply_genome(g, pairs: list[str]):
    for kv in pairs:
        k, v = kv.split("=", 1)
        cur = getattr(g, k)
        if isinstance(cur, np.ndarray):
            setattr(g, k, np.full_like(cur, float(v)))
        else:
            setattr(g, k, float(v))
    return g


def rosters_for(seed: int, cfg):
    rng = np.random.default_rng(20_000 + seed)
    return sample_roster(rng, cfg), sample_roster(rng, cfg)


def job(args):
    kind, on, dn, seeds, cfg_over, og_over, dg_over = args
    cfg = load_config(overrides=cfg_over)
    rows = []
    for k in seeds:
        ra, rb = rosters_for(k, cfg)
        if kind == "poss":
            og = apply_genome(CANONICAL_OFFENSES[on](), og_over)
            dg = apply_genome(CANONICAL_DEFENSES[dn](), dg_over)
            g = Game(cfg, (ra, rb), (og, None), (None, dg), seed=k)
            r = g.run_possession()
            rows.append(dict(kind=kind, on=on, dn=dn, outcome=r.outcome, throws=r.throws, dur=r.duration,
                             tl=g.throw_log, dl=g.decision_log))
        else:  # full point: team 0 receives the pull
            o2 = OFFENSES[(OFFENSES.index(on) + 1) % len(OFFENSES)]
            d2 = DEFENSES[(DEFENSES.index(dn) + 1) % len(DEFENSES)]
            ogs = (apply_genome(CANONICAL_OFFENSES[on](), og_over), apply_genome(CANONICAL_OFFENSES[o2](), og_over))
            dgs = (apply_genome(CANONICAL_DEFENSES[d2](), dg_over), apply_genome(CANONICAL_DEFENSES[dn](), dg_over))
            g = Game(cfg, (ra, rb), ogs, dgs, seed=k, mode="point")
            res = g.run_point()
            rows.append(dict(kind=kind, hold=res[-1].outcome == "score" and res[-1].offense == 0,
                             n_poss=len(res)))
    return rows


def metrics(rows: list[dict]) -> dict:
    P = [r for r in rows if r["kind"] == "poss"]
    tl = [t for r in P for t in r["tl"]]
    dl = [d for r in P for d in r["dl"]]
    ok = np.array([t["outcome"] in ("catch", "score") for t in tl])
    outs = Counter(r["outcome"] for r in P)
    tos = sum(v for k, v in outs.items() if k != "score")
    hk = [t for t in tl if t["huck"]]
    m = {
        "n_possessions": len(P),
        "completion": float(ok.mean()) if len(ok) else float("nan"),
        "score_rate": outs["score"] / max(len(P), 1),
        "throws_per_score": float(np.mean([r["throws"] for r in P if r["outcome"] == "score"])) if outs["score"] else float("nan"),
        "throws_per_poss": float(np.mean([r["throws"] for r in P])),
        "huck_completion": float(np.mean([t["outcome"] in ("catch", "score") for t in hk])) if hk else float("nan"),
        "huck_share": len(hk) / max(len(tl), 1),
        "break_share": float(np.mean([t["is_break"] for t in tl])) if tl else float("nan"),
        "break_completion": float(np.mean([t["outcome"] in ("catch", "score") for t in tl if t["is_break"]]) if any(t["is_break"] for t in tl) else float("nan")),
        "stall_share": outs["stall"] / max(tos, 1),
        "drop_share": outs["drop"] / max(tos, 1),
        "d_share": (outs["block"] + outs["interception"] + outs["mark_block"]) / max(tos, 1),
        "throwaway_share": (outs["throwaway"] + outs["out_of_bounds"]) / max(tos, 1),
        "mean_stall_at_throw": float(np.mean([t["stall"] for t in tl])) if tl else float("nan"),
        "pred_p_chosen": float(np.mean([t["p"] for t in tl])) if tl else float("nan"),
        "best_p_available": float(np.mean([d[1] for d in dl])) if dl else float("nan"),
        "share_decisions_best_p_ge_95": float(np.mean([d[1] >= 0.95 for d in dl])) if dl else float("nan"),
        "turnovers": dict(outs.most_common()),
    }
    # model calibration: predicted vs actual completion by predicted-p bucket
    if tl:
        Pp = np.array([t["p"] for t in tl])
        cal = {}
        for lo, hi in [(0, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]:
            mk = (Pp >= lo) & (Pp < hi)
            if mk.any():
                cal[f"{lo:.2f}-{min(hi, 1):.2f}"] = (int(mk.sum()), round(float(Pp[mk].mean()), 3), round(float(ok[mk].mean()), 3))
        m["calibration_pred_vs_actual"] = cal
        D = np.array([t["dist"] for t in tl])
        m["completion_by_dist"] = {f"{lo}-{hi}m": (int(((D >= lo) & (D < hi)).sum()), round(float(ok[(D >= lo) & (D < hi)].mean()), 3))
                                   for lo, hi in [(0, 10), (10, 20), (20, 35), (35, 80)] if ((D >= lo) & (D < hi)).any()}
    Q = [r for r in rows if r["kind"] == "point"]
    if Q:
        m["hold_rate"] = float(np.mean([r["hold"] for r in Q]))
        m["possessions_per_point"] = float(np.mean([r["n_poss"] for r in Q]))
    m["by_matchup_score"] = {}
    for r in P:
        key = f"{r['on']}|{r['dn']}"
        m["by_matchup_score"].setdefault(key, []).append(r["outcome"] == "score")
    m["by_matchup_score"] = {k: round(float(np.mean(v)), 3) for k, v in m["by_matchup_score"].items()}
    return m


def report(m: dict):
    print(f"\n{m['n_possessions']} possessions")
    print(f"{'metric':24s}{'value':>10s}   target")
    for name, lo, hi in TARGETS:
        v = m.get(name, float("nan"))
        flag = "  " if (v == v and lo <= v <= hi) else "!!"
        print(f"{flag}{name:22s}{v:10.3f}   [{lo:g}, {hi:g}]")
    for key, label in [("throws_per_poss", "throws_per_poss"), ("break_completion", "break_completion"),
                       ("pred_p_chosen", "pred_p_chosen"), ("best_p_available", "best_p_available"),
                       ("share_decisions_best_p_ge_95", "decisions w/ best>=.95")]:
        if key in m:
            print(f"  {label:22s}{m[key]:10.3f}")
    if "possessions_per_point" in m:
        print(f"  {'possessions_per_point':22s}{m['possessions_per_point']:10.2f}")
    print(f"  turnovers: {m['turnovers']}")
    if "calibration_pred_vs_actual" in m:
        print(f"  model calibration (n, predicted, actual): {m.get('calibration_pred_vs_actual')}")
        print(f"  completion by distance (n, actual): {m.get('completion_by_dist')}")


def run_c(a, over) -> dict:
    """Same matchups on the C engine: aggregate metrics from per-game counters (no per-throw log)."""
    from ultisim.engine import CEngine, make_specs, pack_defenses, pack_offenses, roster_matrix
    cfg = load_config(overrides=over)
    eng = CEngine(cfg)
    offs = pack_offenses([apply_genome(CANONICAL_OFFENSES[o](), a.og) for o in OFFENSES])
    defs = pack_defenses([apply_genome(CANONICAL_DEFENSES[d](), a.dg) for d in DEFENSES])
    pairs = [rosters_for(k, cfg) for k in range(a.n)]
    A, B = roster_matrix([p[0] for p in pairs]), roster_matrix([p[1] for p in pairs])
    res, by = [], {}
    for oi, on in enumerate(OFFENSES):
        for di, dn in enumerate(DEFENSES):
            r = eng.run(offs, defs, make_specs(A, B, np.arange(a.n) + 7919 * (oi * 8 + di + 1), off_idx=(oi, 0), def_idx=(0, di)))
            res.append(r)
            by[f"{on}|{dn}"] = round(float((r["outcome"] == 0).mean()), 3)
    r = np.concatenate(res)
    thr = r["throws"].sum()
    outs = {o: int(r[f"n_{o}"].sum()) for o in ("score", "block", "interception", "drop", "throwaway",
                                                  "out_of_bounds", "stall", "mark_block", "timeout")}
    tos = sum(v for k, v in outs.items() if k != "score")
    m = {"n_possessions": len(r), "completion": r["completions"].sum() / max(thr, 1),
         "score_rate": float((r["outcome"] == 0).mean()),
         "throws_per_score": float(r["throws"][r["outcome"] == 0].mean()) if outs["score"] else float("nan"),
         "throws_per_poss": float(r["throws"].mean()),
         "huck_completion": r["huck_completions"].sum() / max(r["hucks"].sum(), 1), "huck_share": r["hucks"].sum() / max(thr, 1),
         "break_share": r["breaks"].sum() / max(thr, 1), "break_completion": r["break_completions"].sum() / max(r["breaks"].sum(), 1),
         "stall_share": outs["stall"] / max(tos, 1), "drop_share": outs["drop"] / max(tos, 1),
         "d_share": (outs["block"] + outs["interception"] + outs["mark_block"]) / max(tos, 1),
         "throwaway_share": (outs["throwaway"] + outs["out_of_bounds"]) / max(tos, 1),
         "mean_stall_at_throw": r["sum_stall_at_throw"].sum() / max(thr, 1), "pred_p_chosen": r["sum_p_chosen"].sum() / max(thr, 1),
         "turnovers": dict(sorted(outs.items(), key=lambda kv: -kv[1])), "by_matchup_score": by}
    if a.points:
        holds, nposs = [], []
        per = max(1, a.points // (len(OFFENSES) * len(DEFENSES)))
        pp = [rosters_for(1000 + k, cfg) for k in range(per)]
        PA, PB = roster_matrix([p[0] for p in pp]), roster_matrix([p[1] for p in pp])
        for oi, on in enumerate(OFFENSES):
            for di, dn in enumerate(DEFENSES):
                o2, d2 = (oi + 1) % len(OFFENSES), (di + 1) % len(DEFENSES)
                q = eng.run(offs, defs, make_specs(PA, PB, np.arange(per) + 104729 * (oi * 8 + di + 1),
                                                   off_idx=(oi, o2), def_idx=(d2, di), mode=1))
                holds.append(q["scorer"] == 0)
                nposs.append(q["n_poss"])
        m["hold_rate"] = float(np.mean(np.concatenate(holds)))
        m["possessions_per_point"] = float(np.mean(np.concatenate(nposs)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=60, help="possessions per matchup")
    ap.add_argument("--points", type=int, default=0, help="full points for the hold rate")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--set", action="append", default=[], help="config override, dotted.path=value")
    ap.add_argument("--og", action="append", default=[], help="offense genome override field=value")
    ap.add_argument("--dg", action="append", default=[], help="defense genome override field=value")
    ap.add_argument("--json", default="")
    ap.add_argument("--matchups", action="store_true", help="print per-matchup score rates")
    ap.add_argument("--engine", choices=["ref", "c"], default="ref",
                    help="ref = Python reference (full diagnostics); c = C engine (fast, aggregate metrics)")
    a = ap.parse_args()
    over = nested(a.set)
    if a.engine == "c":
        t0 = time.perf_counter()
        m = run_c(a, over)
        report(m)
        if a.matchups:
            for k, v in sorted(m["by_matchup_score"].items()):
                print(f"    {k:32s} {v:.2f}")
        print(f"({time.perf_counter() - t0:.0f}s, C engine)")
        if a.json:
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            m["overrides"] = {"set": a.set, "og": a.og, "dg": a.dg, "engine": "c"}
            Path(a.json).write_text(json.dumps(m, indent=1, default=float))
        return
    chunk = max(1, a.n // 2)
    jobs = [("poss", on, dn, list(range(i, min(i + chunk, a.n))), over, a.og, a.dg)
            for on in OFFENSES for dn in DEFENSES for i in range(0, a.n, chunk)]
    if a.points:
        per = max(1, a.points // (len(OFFENSES) * len(DEFENSES)))
        jobs += [("point", on, dn, list(range(1000, 1000 + per)), over, a.og, a.dg) for on in OFFENSES for dn in DEFENSES]
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(job, jobs):
            rows.extend(r)
    m = metrics(rows)
    report(m)
    if a.matchups:
        for k, v in sorted(m["by_matchup_score"].items()):
            print(f"    {k:32s} {v:.2f}")
    print(f"({time.perf_counter() - t0:.0f}s)")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        m["overrides"] = {"set": a.set, "og": a.og, "dg": a.dg}
        Path(a.json).write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
