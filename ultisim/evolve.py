"""Coevolution of offensive and defensive strategies (docs/EVOLUTION.md).

    from ultisim.evolve import EvoConfig, Evolver
    Evolver(EvoConfig(run="pilot", generations=20)).run()

Each generation:
  1. sample fresh rosters (common to every genome this generation)
  2. every offense plays 6 opponents: 3 from the defense population (balanced round-robin),
     1 from the Hall of Fame of past defense champions, 2 canonical defenses
  3. racing: everyone plays `stage1` possessions; the top `race_frac` play up to `stage2`
  4. fitness: offense = score rate; defense = 1 - score rate conceded
  5. breed: elitism + tournament selection + slot-block crossover + Gaussian mutation
  6. log (history.jsonl / status.json / archive.json), checkpoint (state.pkl), render a champion replay
All file formats: viz/RUN_FORMAT.md. Simulation runs in the C engine (ultisim/engine.py).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .classify import DEF_SCHEMES, FAMILIES, defense_scheme, offense_descriptors, offense_family
from .config import load_config
from .engine import CEngine, make_specs, pack_defenses, pack_offenses, roster_matrix
from .genome import (CANONICAL_DEFENSES, CANONICAL_OFFENSES, DEFENSE_SCHEMA, N_SLOTS, OFFENSE_SCHEMA,
                     DefenseGenome, OffenseGenome, from_vector, schema_bounds, to_vector)
from .roster import sample_roster

ROOT = Path(__file__).resolve().parent.parent
OFF_GENES_SHOWN = ["risk_aversion", "patience", "huck_bias", "break_bias", "hammer_bias", "lead", "give_go", "reset_stall", "panic_stall"]
DEF_GENES_SHOWN = ["zone", "mark_dist", "mark_angle", "angle_near", "angle_far", "cushion_base", "poach", "switch",
                   "help_cut", "stack_sag", "flex_focus", "transition_throws", "trap"]


@dataclass
class EvoConfig:
    run: str = "run"
    kind: str = "offense"
    generations: int = 100
    off_pop: int = 48
    def_pop: int = 32
    n_rosters: int = 64               # roster pairs sampled per generation (common random numbers)
    stage1: int = 96                  # possessions per offense for everyone (racing stage 1)
    stage2: int = 288                 # total possessions for the top race_frac offenses
    race_frac: float = 0.34
    elite: int = 4
    tournament: int = 3
    crossover: float = 0.5
    mut_rate: float = 0.15            # per-gene mutation probability
    mut_scale: float = 0.08           # mutation sd as a fraction of the gene's range
    seed_canonical: float = 0.6       # share of the initial populations derived from canonical strategies
    hof_size: int = 20
    # --- open-endedness (novelty search + quality diversity)
    novelty_weight: float = 0.35      # parent selection = rank(score) + novelty_weight * rank(novelty)
    novelty_k: int = 10               # novelty = mean distance to the k nearest behaviours (population + archive)
    novelty_archive: int = 600
    niche_elites: bool = True         # the best member of every structure / scheme family always survives
    immigrants: int = 3               # fresh genomes injected each generation (random or macro-mutated)
    macro_mut: float = 0.06           # chance a child gets one whole formation slot redrawn
    replay_every: int = 2             # generations between champion replays (0 = never)
    seed: int = 0
    threads: int = 0
    config_overrides: dict = field(default_factory=dict)


def replace_retry(src: Path, dst: Path, tries: int = 40):
    """os.replace, retried: on Windows it fails while a reader (the dashboard server) has dst open."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05)


def _atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    replace_retry(tmp, path)


class Evolver:
    def __init__(self, ec: EvoConfig, resume: bool = False):
        self.ec = ec
        self.cfg = load_config(overrides=ec.config_overrides or None)
        self.eng = CEngine(self.cfg, threads=ec.threads)
        self.dir = ROOT / "runs" / ec.run
        self.dir.mkdir(parents=True, exist_ok=True)
        self.olo, self.ohi = schema_bounds(OFFENSE_SCHEMA)
        self.dlo, self.dhi = schema_bounds(DEFENSE_SCHEMA)
        self.canon_def = [to_vector(f(), DEFENSE_SCHEMA) for f in CANONICAL_DEFENSES.values()]
        self.canon_def_names = list(CANONICAL_DEFENSES)
        self._replay_thread: threading.Thread | None = None
        if resume and (self.dir / "state.pkl").exists():
            self._load()
        else:
            self.rng = np.random.default_rng(ec.seed)
            self.gen = 0
            self.elapsed = 0.0
            self.possessions = 0
            self.off = self._init_pop(OFFENSE_SCHEMA, [to_vector(f(), OFFENSE_SCHEMA) for f in CANONICAL_OFFENSES.values()],
                                      ec.off_pop, self.olo, self.ohi, offense=True)
            self.dfn = self._init_pop(DEFENSE_SCHEMA, self.canon_def, ec.def_pop, self.dlo, self.dhi, offense=False)
            self.hof: list[np.ndarray] = []
            self.archive: dict = {}
            self.nov_off: list = []
            self.nov_def: list = []
            (self.dir / "history.jsonl").write_text("")
        _atomic_json(self.dir / "meta.json", {
            "run": ec.run, "created": dt.datetime.now().isoformat(timespec="seconds"), "kind": ec.kind,
            "generations_planned": ec.generations, "offense_pop": ec.off_pop, "defense_pop": ec.def_pop,
            "possessions_per_offense": ec.stage2, "racing": {"stage1": ec.stage1, "stage2": ec.stage2, "top_frac": ec.race_frac},
            "families": FAMILIES, "def_schemes": DEF_SCHEMES, "settings": asdict(ec)})

    # ------------------------------------------------------------------ population helpers
    def _random_vec(self, lo, hi, offense: bool) -> np.ndarray:
        v = lo + self.rng.random(len(lo)) * (hi - lo)
        if offense:  # sensible handler count: 1-3 handler slots
            g = from_vector(v, OFFENSE_SCHEMA, OffenseGenome)
            g.handler = np.zeros(N_SLOTS)
            g.handler[self.rng.choice(N_SLOTS, self.rng.integers(1, 4), replace=False)] = 1.0
            v = to_vector(g, OFFENSE_SCHEMA)
        return v

    def _init_pop(self, schema, canon, n, lo, hi, offense) -> np.ndarray:
        pop = []
        n_canon = int(round(n * self.ec.seed_canonical))
        for i in range(n_canon):
            base = canon[i % len(canon)].copy()
            pop.append(base if i < len(canon) else self._mutate(base, lo, hi, scale=1.5))
        while len(pop) < n:
            pop.append(self._random_vec(lo, hi, offense))
        return np.array(pop)

    def _mutate(self, v, lo, hi, scale=1.0) -> np.ndarray:
        ec = self.ec
        m = self.rng.random(len(v)) < ec.mut_rate
        v = v.copy()
        v[m] += self.rng.normal(0.0, ec.mut_scale * scale, m.sum()) * (hi - lo)[m]
        return np.clip(v, lo, hi)

    def _crossover(self, a, b, schema) -> np.ndarray:
        """Slot-block crossover: each formation slot (all its per-slot genes) comes from one parent;
        scalar genes are mixed uniformly."""
        child = a.copy()
        take_b_slot = self.rng.random(N_SLOTS) < 0.5
        i = 0
        for _, n, _, _ in schema:
            if n == N_SLOTS:
                child[i:i + n] = np.where(take_b_slot, b[i:i + n], a[i:i + n])
            else:
                if self.rng.random() < 0.5:
                    child[i:i + n] = b[i:i + n]
            i += n
        return child

    def _macro(self, v, schema, lo, hi) -> np.ndarray:
        """Redraw every gene of one formation slot (offense) / zone spot (defense): a structural jump."""
        v = v.copy()
        k = int(self.rng.integers(N_SLOTS))
        i = 0
        for _, n, _, _ in schema:
            if n == N_SLOTS:
                j = i + k
                v[j] = lo[j] + self.rng.random() * (hi[j] - lo[j])
            i += n
        return v

    @staticmethod
    def _ranks(x) -> np.ndarray:
        r = np.empty(len(x))
        r[np.argsort(x)] = np.arange(len(x))
        return r / max(len(x) - 1, 1)

    def _novelty(self, B: np.ndarray, archive: list) -> np.ndarray:
        """Mean distance of each behaviour to its k nearest neighbours among the population and the archive."""
        ref = np.vstack([B] + ([np.array(archive)] if archive else []))
        d = np.sqrt(((B[:, None, :] - ref[None, :, :]) ** 2).sum(-1))
        d[np.arange(len(B)), np.arange(len(B))] = np.inf   # not yourself
        k = min(self.ec.novelty_k, ref.shape[0] - 1)
        return np.sort(d, axis=1)[:, :k].mean(axis=1)

    def _breed(self, pop, fit, schema, lo, hi, novelty=None, labels=None, offense=True) -> np.ndarray:
        ec = self.ec
        order = np.argsort(-fit)
        keep = list(order[:ec.elite])
        if ec.niche_elites and labels is not None:   # best of every family survives (quality diversity)
            for fam in sorted(set(labels)):
                best = max((i for i in range(len(pop)) if labels[i] == fam), key=lambda i: fit[i])
                if best not in keep:
                    keep.append(best)
        new = [pop[i].copy() for i in keep[:len(pop) // 2]]
        sel = self._ranks(fit)
        if novelty is not None and ec.novelty_weight > 0:
            sel = sel + ec.novelty_weight * self._ranks(novelty)

        def pick():
            c = self.rng.choice(len(pop), ec.tournament, replace=False)
            return pop[c[np.argmax(sel[c])]]
        n_imm = min(ec.immigrants, len(pop) - len(new))
        while len(new) < len(pop) - n_imm:
            a = pick()
            child = self._crossover(a, pick(), schema) if self.rng.random() < ec.crossover else a.copy()
            if self.rng.random() < ec.macro_mut:
                child = self._macro(child, schema, lo, hi)
            new.append(self._mutate(child, lo, hi))
        for _ in range(n_imm):   # immigrants: half brand-new random genomes, half heavily reshaped survivors
            if self.rng.random() < 0.5:
                new.append(self._random_vec(lo, hi, offense))
            else:
                v = pop[int(self.rng.integers(len(pop)))]
                new.append(self._mutate(self._macro(self._macro(v, schema, lo, hi), schema, lo, hi), lo, hi, scale=2.0))
        return np.array(new)

    def _off_behaviour(self, beh: np.ndarray) -> np.ndarray:
        """Behaviour fingerprint per offense: what it does in games + the shape of its formation."""
        rows = []
        for v, b in zip(self.off, beh):
            d = offense_descriptors(from_vector(v, OFFENSE_SCHEMA, OffenseGenome))
            rows.append([b[0] * 10, b[1] * 4, b[2] * 20, b[3] / 10, b[4] / 40,
                         d["depth"] / 20, d["spread_x"] / 8, d["spread_y"] / 10, d["lateral"] / 10, d["n_handlers"] / 3])
        return np.array(rows)

    def _def_behaviour(self) -> np.ndarray:
        rows = []
        for v in self.dfn:
            g = from_vector(v, DEFENSE_SCHEMA, DefenseGenome)
            rows.append([g.n_zone / 7, g.force / 4, g.mark_dist / 3, g.cushion_base / 5, g.poach, g.switch, g.help_cut,
                         g.stack_sag, g.trap, min(g.transition_throws, 10) / 10, (g.angle_near % 360) / 360])
        return np.array(rows)

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self):
        ec = self.ec
        nO = len(self.off)
        # defense pool: population | hall of fame | canonical
        hof = self.hof if self.hof else [self.canon_def[0]]
        pool = list(self.dfn) + list(hof) + self.canon_def
        p_hof0 = len(self.dfn)
        p_can0 = p_hof0 + len(hof)
        # opponents per offense: 3 population (balanced round-robin), 1 HoF, 2 canonical
        perm = self.rng.permutation(np.repeat(np.arange(len(self.dfn)), int(np.ceil(3 * nO / len(self.dfn)))))
        opp = np.zeros((nO, 6), int)
        opp[:, :3] = perm[:3 * nO].reshape(nO, 3)
        opp[:, 3] = p_hof0 + self.rng.integers(0, len(hof), nO)
        opp[:, 4:] = p_can0 + self.rng.integers(0, len(self.canon_def), (nO, 2))
        # rosters (common to all genomes this generation)
        pairs = [(sample_roster(self.rng, self.cfg), sample_roster(self.rng, self.cfg)) for _ in range(ec.n_rosters)]
        A = roster_matrix([p[0] for p in pairs])
        B = roster_matrix([p[1] for p in pairs])
        offs = pack_offenses([from_vector(v, OFFENSE_SCHEMA, OffenseGenome) for v in self.off])
        defs = pack_defenses([from_vector(v, DEFENSE_SCHEMA, DefenseGenome) for v in pool])
        gen_seed = (self.ec.seed + 1) * 1_000_003 + self.gen * 7_919_993

        def games(off_ids, k0, k1):
            """possessions k0..k1-1 (per offense) for the given offenses; opponent slot = k % 6."""
            k = np.arange(k0, k1)
            oi = np.repeat(off_ids, len(k))
            kk = np.tile(k, len(off_ids))
            di = opp[oi, kk % 6]
            ridx = kk % ec.n_rosters
            specs = make_specs(A[ridx], B[ridx], gen_seed + kk * 131 + (kk % 6),
                               off_idx=(oi, 0), def_idx=(0, di))
            return oi, di, self.eng.run(offs, defs, specs)

        rows = []
        oi, di, r = games(np.arange(nO), 0, ec.stage1)
        rows.append((oi, di, r))
        s1 = np.bincount(oi, weights=(r["outcome"] == 0), minlength=nO) / ec.stage1
        top = np.argsort(-s1)[:max(1, int(round(ec.race_frac * nO)))]
        if ec.stage2 > ec.stage1:
            rows.append(games(top, ec.stage1, ec.stage2))
        oi = np.concatenate([x[0] for x in rows])
        di = np.concatenate([x[1] for x in rows])
        r = np.concatenate([x[2] for x in rows])
        scored = (r["outcome"] == 0).astype(float)
        n_o = np.bincount(oi, minlength=nO)
        off_fit = np.bincount(oi, weights=scored, minlength=nO) / np.maximum(n_o, 1)
        n_d = np.bincount(di, minlength=len(pool))[:len(self.dfn)]
        conceded = np.bincount(di, weights=scored, minlength=len(pool))[:len(self.dfn)] / np.maximum(n_d, 1)
        conceded[n_d == 0] = 1.0
        self.possessions += len(r)
        thr = r["throws"].sum()
        stats = {"completion": float(r["completions"].sum() / max(thr, 1)), "huck_share": float(r["hucks"].sum() / max(thr, 1)),
                 "break_share": float(r["breaks"].sum() / max(thr, 1)), "hammer_share": float(r["hammers"].sum() / max(thr, 1)),
                 "throws_per_poss": float(r["throws"].mean()), "games": int(len(r))}
        t_o = np.maximum(np.bincount(oi, weights=r["throws"], minlength=nO), 1)
        beh = np.stack([np.bincount(oi, weights=r["hucks"], minlength=nO) / t_o,
                        np.bincount(oi, weights=r["breaks"], minlength=nO) / t_o,
                        np.bincount(oi, weights=r["hammers"], minlength=nO) / t_o,
                        t_o / np.maximum(n_o, 1),
                        np.bincount(oi, weights=r["duration"], minlength=nO) / np.maximum(n_o, 1)], axis=1)
        self._last_beh = beh
        return off_fit, n_o, conceded, n_d, stats

    # ------------------------------------------------------------------ logging
    def _status(self, phase, state="running", message=""):
        _atomic_json(self.dir / "status.json", {
            "run": self.ec.run, "state": state, "gen": self.gen, "generations_planned": self.ec.generations,
            "phase": phase, "updated": dt.datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(self.elapsed, 1), "message": message})

    def _genes(self, g, names):
        return {k: round(float(getattr(g, k)), 3) for k in names}

    def _log(self, off_fit, conceded, stats, gen_s):
        og = [from_vector(v, OFFENSE_SCHEMA, OffenseGenome) for v in self.off]
        fams = [offense_family(g) for g in og]
        dg = [from_vector(v, DEFENSE_SCHEMA, DefenseGenome) for v in self.dfn]
        schemes = [defense_scheme(g) for g in dg]
        for g, f, s, v in zip(og, fams, off_fit, self.off):
            cur = self.archive.get(f)
            if cur is None or s > cur["score_rate"]:
                self.archive[f] = {"score_rate": round(float(s), 4), "gen": self.gen, "descriptors": offense_descriptors(g),
                                   "genes": self._genes(g, OFF_GENES_SHOWN), "vector": v.tolist()}
        b = int(np.argmax(off_fit))
        bd = int(np.argmin(conceded))
        line = {
            "gen": self.gen, "elapsed_s": round(self.elapsed, 1), "gen_s": round(gen_s, 2), "possessions": int(self.possessions),
            "off": {"best": round(float(off_fit.max()), 4), "mean": round(float(off_fit.mean()), 4),
                    "median": round(float(np.median(off_fit)), 4), "worst": round(float(off_fit.min()), 4)},
            "def": {"best": round(float(conceded.min()), 4), "mean": round(float(conceded.mean()), 4)},
            "families": {f: fams.count(f) for f in FAMILIES},
            "family_best": {f: (self.archive[f]["score_rate"] if f in self.archive else None) for f in FAMILIES},
            "champion": {"family": fams[b], "score_rate": round(float(off_fit[b]), 4), "descriptors": offense_descriptors(og[b]),
                         "genes": self._genes(og[b], OFF_GENES_SHOWN)},
            "def_champion": {"scheme": "zone" if dg[bd].is_zone else ("flex" if dg[bd].is_flex else "person"),
                             "n_zone": dg[bd].n_zone, "force": dg[bd].force_name,
                             "conceded": round(float(conceded[bd]), 4), "genes": self._genes(dg[bd], DEF_GENES_SHOWN)},
            "def_schemes": {s: schemes.count(s) for s in DEF_SCHEMES},
            "stats": {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
            "novelty": getattr(self, "last_novelty", None),
        }
        with open(self.dir / "history.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        _atomic_json(self.dir / "archive.json", {"families": {k: {kk: vv for kk, vv in v.items() if kk != "vector"}
                                                              for k, v in self.archive.items()}})
        return line, og[b], dg[bd]

    def _save(self):
        tmp = self.dir / "state.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"gen": self.gen, "elapsed": self.elapsed, "possessions": self.possessions, "off": self.off,
                         "dfn": self.dfn, "hof": self.hof, "archive": self.archive,
                         "nov_off": self.nov_off, "nov_def": self.nov_def,
                         "rng": self.rng.bit_generator.state, "ec": asdict(self.ec)}, f)
        replace_retry(tmp, self.dir / "state.pkl")

    def _load(self):
        with open(self.dir / "state.pkl", "rb") as f:
            s = pickle.load(f)
        self.gen, self.elapsed, self.possessions = s["gen"], s["elapsed"], s["possessions"]
        self.off, self.dfn, self.hof, self.archive = s["off"], s["dfn"], s["hof"], s["archive"]
        self.nov_off, self.nov_def = s.get("nov_off", []), s.get("nov_def", [])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = s["rng"]

    # ------------------------------------------------------------------ champion replay (Python reference, background)
    def _render_replay(self, og: OffenseGenome, dg: DefenseGenome, label: str):
        if self._replay_thread is not None and self._replay_thread.is_alive():
            return  # previous one still rendering; skip this generation

        def work(seed0):
            from .engine_ref import simulate_possession
            best = None
            rng = np.random.default_rng(seed0)
            for k in range(4):
                ra, rb = sample_roster(rng, self.cfg), sample_roster(rng, self.cfg)
                res, rec = simulate_possession(self.cfg, ra, rb, og, dg, seed=seed0 + k, record=True, title=label)
                key = (res.outcome == "score", res.throws)
                if best is None or key > best[0]:
                    best = (key, rec)
            best[1].save(self.dir / "latest_best.json")

        og.name, dg.name = label.split(" vs ")[0], defense_scheme(dg)
        self._replay_thread = threading.Thread(target=work, args=(int(self.rng.integers(1 << 30)),), daemon=True)
        self._replay_thread.start()

    # ------------------------------------------------------------------ main loop
    def run(self):
        ec = self.ec
        self._status("starting")
        try:
            while self.gen < ec.generations:
                t0 = time.perf_counter()
                self._status("evaluating")
                off_fit, n_o, conceded, n_d, stats = self._evaluate()
                gen_s = time.perf_counter() - t0
                self.elapsed += gen_s
                line, champ, dchamp = self._log(off_fit, conceded, stats, gen_s)
                print(f"gen {self.gen:3d}  off best {line['off']['best']:.3f} mean {line['off']['mean']:.3f}  "
                      f"def best {line['def']['best']:.3f}  champ {line['champion']['family']:10s}  "
                      f"{gen_s:5.1f}s  ({stats['games']} games)", flush=True)
                if ec.replay_every and self.gen % ec.replay_every == 0:
                    self._render_replay(champ, dchamp, f"Gen {self.gen} champion ({line['champion']['family']}, "
                                                       f"{line['champion']['score_rate']:.2f}) vs best D")
                # hall of fame + next generation
                self.hof.append(self.dfn[int(np.argmin(conceded))].copy())
                self.hof = self.hof[-ec.hof_size:]
                self._status("breeding")
                Bo, Bd = self._off_behaviour(self._last_beh), self._def_behaviour()
                nov_o, nov_d = self._novelty(Bo, self.nov_off), self._novelty(Bd, self.nov_def)
                for arch, B, fitv in ((self.nov_off, Bo, off_fit), (self.nov_def, Bd, 1.0 - conceded)):
                    for i in set([int(np.argmax(fitv))] + list(self.rng.choice(len(B), 3, replace=False))):
                        arch.append(B[i].tolist())
                    del arch[:-ec.novelty_archive]
                self.last_novelty = {"off_mean": round(float(nov_o.mean()), 3), "def_mean": round(float(nov_d.mean()), 3),
                                     "archive": len(self.nov_off)}
                off_lab = [offense_family(from_vector(v, OFFENSE_SCHEMA, OffenseGenome)) for v in self.off]
                def_lab = [defense_scheme(from_vector(v, DEFENSE_SCHEMA, DefenseGenome)).split("_")[0] for v in self.dfn]
                self.off = self._breed(self.off, off_fit, OFFENSE_SCHEMA, self.olo, self.ohi, nov_o, off_lab, offense=True)
                self.dfn = self._breed(self.dfn, 1.0 - conceded, DEFENSE_SCHEMA, self.dlo, self.dhi, nov_d, def_lab,
                                       offense=False)
                self.gen += 1
                self._save()
            self._status("done", state="finished")
        except KeyboardInterrupt:
            self._save()
            self._status("interrupted", state="stopped", message="stopped by user; resume with --resume")
            print("stopped; checkpoint saved")
        except Exception as e:  # keep the dashboard honest
            self._status("error", state="error", message=repr(e))
            raise
        finally:
            if self._replay_thread is not None:
                self._replay_thread.join(timeout=120)
