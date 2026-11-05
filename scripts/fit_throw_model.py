"""Fit the learned throw-completion model from self-play.

    python scripts/fit_throw_model.py --rounds 3 -n 40

Each round:
  1. plays possessions with the CURRENT model, with exploration (throwers sometimes pick a random top-12
     throw) so the data covers more than the throws the model already likes:
       - every hand-built offense against every hand-built defense, and
       - random pairs drawn from a diverse pool: heavily mutated hand-built strategies plus evolved
         offenses/defenses from earlier runs (runs/*/state.pkl), so the model has seen defenses unlike
         the textbook ones. Without this, evolution finds defenses where the model is badly
         over-confident (watch4: believed 0.91, actual 0.76) and throwers walk into them;
  2. fits an L2 logistic regression: features of the chosen throw -> completed or not;
  3. reports held-out log-loss and calibration vs. the analytic model, then saves configs/throw_model.json.

Rounds repeat because changing the model changes which throws get chosen (like AlphaZero's value net
being refit on games played with the previous net).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one math thread per worker process

import argparse
import datetime as dt
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultisim.config import load_config  # noqa: E402
from ultisim.engine_ref import FEATURE_NAMES, Game  # noqa: E402
from ultisim.genome import (CANONICAL_DEFENSES, CANONICAL_OFFENSES, DEFENSE_SCHEMA, OFFENSE_SCHEMA,  # noqa: E402
                            DefenseGenome, OffenseGenome, from_vector, schema_bounds, to_vector)
from ultisim.roster import sample_roster  # noqa: E402

MODEL_PATH = ROOT / "configs" / "throw_model.json"


def mutant(v, schema, rng, rate=0.35, scale=0.15):
    """A heavily mutated copy of genome vector v (about a third of genes moved by 15% of their range)."""
    lo, hi = schema_bounds(schema)
    m = rng.random(len(v)) < rate
    v = v.copy()
    v[m] += rng.normal(0.0, scale, m.sum()) * (hi - lo)[m]
    return np.clip(v, lo, hi)


def evolved_pool(key, n_genes):
    """Population + Hall of Fame vectors from earlier runs' checkpoints, if any (schema length must match)."""
    out = []
    for f in sorted((ROOT / "runs").glob("*/state.pkl")):
        try:
            with open(f, "rb") as fh:
                st = pickle.load(fh)
        except Exception:  # noqa: BLE001 - an unreadable or old checkpoint is just skipped
            continue
        vs = list(st.get(key, [])) + (list(st.get("hof", [])) if key == "dfn" else [])
        out += [np.asarray(v, float) for v in vs if len(v) == n_genes]
    return out


def diverse_pairs(n_pairs, rng, use_runs):
    offs = [to_vector(f(), OFFENSE_SCHEMA) for f in CANONICAL_OFFENSES.values()]
    defs = [to_vector(f(), DEFENSE_SCHEMA) for f in CANONICAL_DEFENSES.values()]
    ev_o = evolved_pool("off", len(offs[0])) if use_runs else []
    ev_d = evolved_pool("dfn", len(defs[0])) if use_runs else []
    pairs = []
    for _ in range(n_pairs):
        if ev_o and rng.random() < 0.5:
            o = ev_o[rng.integers(len(ev_o))]
        else:
            o = mutant(offs[rng.integers(len(offs))], OFFENSE_SCHEMA, rng)
        if ev_d and rng.random() < 0.5:
            d = ev_d[rng.integers(len(ev_d))]
        else:
            d = mutant(defs[rng.integers(len(defs))], DEFENSE_SCHEMA, rng)
        pairs.append((o, d))
    return pairs, len(ev_o), len(ev_d)


def collect(args):
    ov, dv, seeds, eps, use_model = args
    over = {"sim": {"explore_eps": eps}}
    if not use_model:
        over["throws"] = {"learned_model": ""}
    cfg = load_config(overrides=over)
    og = from_vector(np.asarray(ov, float), OFFENSE_SCHEMA, OffenseGenome)
    dg = from_vector(np.asarray(dv, float), DEFENSE_SCHEMA, DefenseGenome)
    X, y = [], []
    for k in seeds:
        rng = np.random.default_rng(50_000 + k)
        g = Game(cfg, (sample_roster(rng, cfg), sample_roster(rng, cfg)), (og, None), (None, dg), seed=k)
        g.run_possession()
        for t in g.throw_log:
            if t["outcome"] is None:
                continue
            X.append(t["x"])
            y.append(t["outcome"] in ("catch", "score"))
    return X, y


def fit_logistic(X, y, l2=1.0, iters=50):
    w = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], l2)
    reg[0] = 1e-6  # don't shrink the bias
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + reg * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calib_table(p, y):
    rows = []
    for lo, hi in [(0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum() >= 5:
            rows.append(f"{lo:.2f}-{min(hi, 1):.2f}: n={m.sum():5d} pred {p[m].mean():.3f} actual {y[m].mean():.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("-n", type=int, default=40, help="possessions per matchup per round")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--l2", type=float, default=2.0)
    ap.add_argument("--fresh", action="store_true", help="start round 1 from the analytic model")
    ap.add_argument("--diverse", type=int, default=80,
                    help="random (mutated or evolved offense, defense) pairs per round, on top of the canonical grid")
    ap.add_argument("--diverse-n", type=int, default=12, help="possessions per diverse pair")
    ap.add_argument("--no-runs", action="store_true", help="don't draw evolved strategies from runs/*/state.pkl")
    a = ap.parse_args()
    Xall, yall = [], []
    for rnd in range(1, a.rounds + 1):
        use_model = MODEL_PATH.exists() and not (a.fresh and rnd == 1)
        t0 = time.perf_counter()
        chunk = max(1, a.n // 2)
        base = rnd * 100_000
        jobs = []
        for on, of in CANONICAL_OFFENSES.items():
            for dn, df in CANONICAL_DEFENSES.items():
                ov, dv = to_vector(of(), OFFENSE_SCHEMA), to_vector(df(), DEFENSE_SCHEMA)
                for i in range(0, a.n, chunk):
                    jobs.append((ov, dv, list(range(base + i, base + min(i + chunk, a.n))), a.eps, use_model))
                base += 1_000
        pairs, n_eo, n_ed = diverse_pairs(a.diverse, np.random.default_rng(7_000 + rnd), not a.no_runs)
        for ov, dv in pairs:
            jobs.append((ov, dv, list(range(base, base + a.diverse_n)), a.eps, use_model))
            base += 1_000
        with ProcessPoolExecutor(a.workers) as ex:
            for X, y in ex.map(collect, jobs):
                Xall += X
                yall += y
        X = np.array(Xall, float)
        y = np.array(yall, float)
        rng = np.random.default_rng(rnd)
        test = rng.random(len(y)) < 0.2
        w = fit_logistic(X[~test], y[~test], l2=a.l2)
        p_new = 1 / (1 + np.exp(-np.clip(X[test] @ w, -30, 30)))
        p_old = 1 / (1 + np.exp(-X[test][:, FEATURE_NAMES.index("logit_p")]))
        print(f"\nround {rnd}: {len(y)} throws total ({time.perf_counter() - t0:.0f}s), completion in data {y.mean():.3f}"
              f" [{'learned' if use_model else 'analytic'} model playing; {len(pairs)} diverse pairs,"
              f" evolved pool {n_eo} offenses / {n_ed} defenses]")
        print(f"  held-out log-loss: model {logloss(p_old, y[test]):.4f} -> refit {logloss(p_new, y[test]):.4f}")
        print("  refit calibration:")
        for r in calib_table(p_new, y[test]):
            print("    " + r)
        w_full = fit_logistic(X, y, l2=a.l2)
        MODEL_PATH.write_text(json.dumps({
            "features": FEATURE_NAMES, "weights": [round(float(v), 6) for v in w_full], "n_throws": int(len(y)),
            "completion_in_data": round(float(y.mean()), 4), "fitted": dt.datetime.now().isoformat(timespec="seconds"),
            "note": "logistic regression on chosen-throw features from self-play with exploration: canonical grid plus"
                    " mutated and evolved strategies",
        }, indent=1))
        print(f"  saved {MODEL_PATH.relative_to(ROOT)}")
    print("\nweights:")
    for n, v in zip(FEATURE_NAMES, w_full):
        print(f"  {n:16s} {v:+.3f}")


if __name__ == "__main__":
    main()
