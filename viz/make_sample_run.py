#!/usr/bin/env python3
"""Write a plausible *fake* evolution run (viz/RUN_FORMAT.md) for testing the dashboard.

    python viz/make_sample_run.py [--out runs/_sample_run] [--gens 25] [--planned 40]
                                  [--seed 3] [--live SECONDS] [--state running]

Writes meta.json, history.jsonl, status.json, archive.json and latest_best.json (a copy of
runs/_sample/sample.json, generated with viz/make_sample.py if it is missing).

--live N keeps going after the initial generations: one more generation is appended every
N seconds (status/archive rewritten as the real run does) until --planned is reached, then
status becomes "finished". Ctrl+C leaves it "stopped". Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime

VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VIZ_DIR)

FAMILIES = ["vertical", "horizontal", "side", "iso", "spread", "other"]
DEF_SCHEMES = ["person_forehand", "person_backhand", "person_straight_up", "person_middle",
               "person_sideline", "zone"]
OFF_POP, DEF_POP, POSS_PER_OFF = 48, 32, 240

# how attractive each family becomes over the run (drives population shares)
FAMILY_FITNESS = {"vertical": 0.9, "horizontal": 1.4, "side": 0.4, "iso": 0.6, "spread": 0.8, "other": 0.1}
FAMILY_CEIL = {"vertical": 0.60, "horizontal": 0.64, "side": 0.50, "iso": 0.55, "spread": 0.57, "other": 0.42}
SCHEME_FITNESS = {"person_forehand": 1.2, "person_backhand": 0.7, "person_straight_up": 0.2,
                  "person_middle": 0.8, "person_sideline": 0.1, "zone": 0.6}


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def write_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    for _ in range(20):   # Windows: a reader holding the file open makes replace() fail briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


def apportion(weights: dict, total: int, rng: random.Random) -> dict:
    """Integer counts summing to total, roughly proportional to weights (with a little noise)."""
    w = {k: max(1e-6, v * rng.uniform(0.85, 1.15)) for k, v in weights.items()}
    s = sum(w.values())
    raw = {k: total * v / s for k, v in w.items()}
    out = {k: int(v) for k, v in raw.items()}
    rest = total - sum(out.values())
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:rest]:
        out[k] += 1
    return out


class FakeRun:
    def __init__(self, seed: int, planned: int):
        self.rng = random.Random(seed)
        self.planned = planned
        self.gen = -1
        self.elapsed = 0.0
        self.possessions = 0
        self.family_best: dict = {f: None for f in FAMILIES}
        self.archive: dict = {}
        self.off_best = 0.36
        self.def_best = 0.30
        self.genes = {"risk_aversion": 1.0, "patience": 1.05, "huck_bias": 0.0, "break_bias": 0.0,
                      "lead": 1.5, "give_go": 0.3, "reset_stall": 6.0, "panic_stall": 8.5}
        self.dgenes = {"mark_dist": 1.8, "mark_angle": 40.0, "mark_width": 55.0, "angle_near": 65.0,
                       "angle_far": 150.0, "cushion_base": 2.0, "poach": 0.1, "help_margin": 0.25}

    def _descriptors(self, fam: str) -> dict:
        r = self.rng
        base = {
            "vertical": dict(depth=18.0, spread_x=4.5, spread_y=0.5, lateral=0.0, iso_gap=8.0),
            "horizontal": dict(depth=21.0, spread_x=1.2, spread_y=9.5, lateral=0.5, iso_gap=3.0),
            "side": dict(depth=16.0, spread_x=4.5, spread_y=0.5, lateral=-12.0, iso_gap=8.0),
            "iso": dict(depth=24.0, spread_x=2.0, spread_y=6.0, lateral=0.0, iso_gap=26.0),
            "spread": dict(depth=16.0, spread_x=10.0, spread_y=8.5, lateral=5.0, iso_gap=12.0),
            "other": dict(depth=12.0, spread_x=6.0, spread_y=6.0, lateral=3.0, iso_gap=5.0),
        }[fam]
        d = {"n_handlers": r.choice([2, 2, 3]), "n_cutters": 4}
        for k, v in base.items():
            d[k] = round(v + r.gauss(0, 0.8), 2)
        d["iso_rest_spread"] = round(abs(r.gauss(3.2, 0.6)), 2)
        return d

    def step(self) -> dict:
        r = self.rng
        self.gen += 1
        g = self.gen
        t = g / max(1, self.planned - 1)
        gen_s = r.uniform(22.0, 28.0) * (1.0 + 0.15 * t)
        self.elapsed += gen_s
        self.possessions += OFF_POP * POSS_PER_OFF

        # family shares drift from roughly uniform toward the fitness ranking
        k = min(1.0, g / 12.0)
        weights = {f: (1 - k) * 1.0 + k * FAMILY_FITNESS[f] ** 2 for f in FAMILIES}
        fams = apportion(weights, OFF_POP, r)

        # offense improves on a saturating curve; per-generation best is noisy (re-evaluated)
        target = 0.36 + 0.27 * (1 - math.exp(-g / 7.0))
        self.off_best = max(0.30, target + r.gauss(0, 0.012))
        mean = self.off_best - 0.17 + 0.07 * (1 - math.exp(-g / 9.0)) + r.gauss(0, 0.008)
        median = mean + r.gauss(0.008, 0.006)
        worst = max(0.0, mean - r.uniform(0.25, 0.33))

        # defense co-evolves: conceded first drops, then creeps up as offense gets better
        self.def_best = 0.30 - 0.06 * (1 - math.exp(-g / 5.0)) + 0.002 * g + r.gauss(0, 0.01)
        def_mean = self.def_best + r.uniform(0.12, 0.17)

        champ_fam = max((f for f in FAMILIES if fams[f] > 0),
                        key=lambda f: FAMILY_CEIL[f] * (0.9 + 0.1 * fams[f] / OFF_POP) + r.gauss(0, 0.035))
        for gk in self.genes:
            self.genes[gk] = round(self.genes[gk] + r.gauss(0, 0.05), 3)
        champion = {"family": champ_fam, "score_rate": round(self.off_best, 4),
                    "descriptors": self._descriptors(champ_fam), "genes": dict(self.genes)}

        # per-family best (archive): only families present this generation can improve
        for f in FAMILIES:
            if fams[f] == 0:
                continue
            if f == "other" and g < 6:
                continue   # leave one family null for a while
            cand = FAMILY_CEIL[f] * (0.62 + 0.38 * (1 - math.exp(-g / 8.0))) + r.gauss(0, 0.01)
            if f == champ_fam:
                cand = max(cand, self.off_best)
            cand = round(cand, 4)
            if self.family_best[f] is None or cand > self.family_best[f]:
                self.family_best[f] = cand
                self.archive[f] = {"score_rate": cand, "gen": g, "descriptors": self._descriptors(f),
                                   "genes": {gk: round(v + r.gauss(0, 0.1), 3) for gk, v in self.genes.items()}}

        schemes = apportion({s: (1 - k) + k * SCHEME_FITNESS[s] ** 2 for s in DEF_SCHEMES}, DEF_POP, r)
        top_scheme = max(schemes, key=schemes.get)
        for gk in self.dgenes:
            self.dgenes[gk] = round(self.dgenes[gk] + r.gauss(0, 0.05 * (1 + abs(self.dgenes[gk]) / 20)), 3)
        scheme, _, force = top_scheme.partition("_")
        def_champion = {"scheme": scheme, "force": force or "forehand", "conceded": round(self.def_best, 4),
                        "genes": dict(self.dgenes)}

        return {
            "gen": g,
            "elapsed_s": round(self.elapsed, 1),
            "gen_s": round(gen_s, 2),
            "possessions": self.possessions,
            "off": {"best": round(self.off_best, 4), "mean": round(mean, 4),
                    "median": round(median, 4), "worst": round(worst, 4)},
            "def": {"best": round(self.def_best, 4), "mean": round(def_mean, 4)},
            "families": fams,
            "family_best": dict(self.family_best),
            "champion": champion,
            "def_champion": def_champion,
            "def_schemes": schemes,
            "stats": {"completion": round(0.92 + 0.02 * t + r.gauss(0, 0.004), 4),
                      "huck_share": round(0.03 + 0.03 * t + r.gauss(0, 0.004), 4),
                      "break_share": round(0.16 + 0.04 * t + r.gauss(0, 0.006), 4),
                      "throws_per_poss": round(8.2 - 0.8 * t + r.gauss(0, 0.15), 3),
                      "games": OFF_POP * POSS_PER_OFF},
        }


def ensure_sample_replay() -> str:
    src = os.path.join(ROOT_DIR, "runs", "_sample", "sample.json")
    if not os.path.isfile(src):
        subprocess.check_call([sys.executable, os.path.join(VIZ_DIR, "make_sample.py"), "--out", src])
    return src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write a fake evolution run for the dashboard.")
    ap.add_argument("--out", default=os.path.join(ROOT_DIR, "runs", "_sample_run"))
    ap.add_argument("--gens", type=int, default=25, help="generations to write up front")
    ap.add_argument("--planned", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--live", type=float, default=0.0, metavar="SECONDS",
                    help="then append one generation every SECONDS until --planned")
    ap.add_argument("--state", default="running", choices=["running", "finished", "stopped", "error"],
                    help="status.json state when not using --live (default running)")
    args = ap.parse_args(argv)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(out)
    for fn in ("history.jsonl", "latest_best.json"):
        try:
            os.remove(os.path.join(out, fn))
        except FileNotFoundError:
            pass

    started = time.time()
    write_atomic(os.path.join(out, "meta.json"), {
        "run": name, "created": now_iso(), "kind": "offense",
        "generations_planned": args.planned, "offense_pop": OFF_POP, "defense_pop": DEF_POP,
        "possessions_per_offense": POSS_PER_OFF,
        "families": FAMILIES, "def_schemes": DEF_SCHEMES,
        "settings": {"synthetic": True, "seed": args.seed},
    })

    run = FakeRun(args.seed, args.planned)
    hist_path = os.path.join(out, "history.jsonl")
    replay_src = ensure_sample_replay()

    def status(state: str, phase: str, gen: int, message: str = "") -> None:
        write_atomic(os.path.join(out, "status.json"), {
            "run": name, "state": state, "gen": gen, "generations_planned": args.planned,
            "phase": phase, "updated": now_iso(), "elapsed_s": round(run.elapsed, 1), "message": message})

    def one_gen() -> None:
        line = run.step()
        with open(hist_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(line) + "\n")
        write_atomic(os.path.join(out, "archive.json"), {"families": run.archive})
        if line["gen"] >= 2 and not os.path.exists(os.path.join(out, "latest_best.json")):
            shutil.copyfile(replay_src, os.path.join(out, "latest_best.json"))

    for _ in range(min(args.gens, args.planned)):
        one_gen()
    done = run.gen + 1

    if args.live <= 0:
        state = "finished" if done >= args.planned else args.state
        status(state, "done" if state == "finished" else "evaluating", done,
               "synthetic sample run (viz/make_sample_run.py)")
        print(f"wrote {done} generations to {out} (state {state})")
        return 0

    print(f"wrote {done} generations to {out}; appending one every {args.live:g} s (Ctrl+C to stop)",
          flush=True)
    try:
        while run.gen + 1 < args.planned:
            nxt = run.gen + 1
            t_end = time.time() + args.live
            phase_i = 0
            while time.time() < t_end:
                status("running", ["evaluating", "breeding"][phase_i % 2], nxt)
                phase_i += 1
                time.sleep(min(1.0, max(0.0, t_end - time.time())))
            one_gen()
            print(f"  gen {run.gen}", flush=True)
        status("finished", "done", run.gen + 1)
    except KeyboardInterrupt:
        status("stopped", "interrupted", run.gen + 1, "stopped by user")
    print(f"done after {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
