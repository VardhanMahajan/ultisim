# Ultimate Frisbee Strategy Lab: Plan

**Goal:** Build a simulator of 7v7 Ultimate, then have strategies evolve by playing against each other (self-play / coevolution) to answer two questions:

1. **Run A (Offense):** For teams with randomly drawn player stats, which offensive structure scores most often? Candidates include vertical stack, horizontal stack, side stack, iso, and anything the search comes up with on its own.
2. **Run B (Defense):** Against a vertical stack and against a horizontal stack, which defensive scheme works best? This covers force choice, marking, positioning, poaching, zones and switching.

It ships with a **visualization engine** so you can watch the current best strategies play, and a **dashboard** so you can follow the evolution.

Wind is out of scope for v1. **The heavy simulation runs in a C engine** (a shared library). Python stays in charge of everything else: evolution, analysis, the viewer and experiments. A small pure-Python reference simulator is kept for prototyping rules and checking the C engine's correctness.

---

## 0. Why a GA and not literally AlphaZero, and how we stay faithful to the idea

AlphaZero combines tree search (MCTS) with a neural net, and it works for **turn-based, perfect-information, two-player** games. Ultimate has **14 agents moving at the same time in continuous space**, and the outcome depends heavily on chance (throw errors, drops). Plain MCTS doesn't fit that, and training a neural policy from scratch to produce recognisable team structure would need a huge compute budget.

What *does* carry over from AlphaZero/AlphaStar is the core loop: **play yourself, keep what wins, repeat.** We do that with:

- **A structured strategy genome.** A strategy is a vector of interpretable parameters: formation slots, cut rules, throw decision weights, defensive positioning, force, and so on. Evolution searches this space.
- **Coevolution.** An offense population and a defense population evolve against each other. A **Hall of Fame / league** of past champions (the AlphaStar trick) keeps a strategy from overfitting to the opponent of the moment and stops rock‑paper‑scissors cycling.
- **An open-ended formation encoding.** We don't hard-code "vertical" or "horizontal". The genome encodes 7 generic player slots (home position, role, cut behaviour). Afterwards a **structure classifier** labels what evolved (aligned with the field = vertical, across it = horizontal, and so on). That way the search can also discover structures we didn't name.
- **Later (Phase 7, optional):** swap the hand-structured behaviours for small neural policies trained by neuroevolution or PPO self-play, using the GA champions as starting opponents.

The most important caveat: **conclusions are only as good as the simulator.** Evolution is brilliant at exploiting simulator bugs. Before we trust any result, the plan has explicit **calibration** (Phase 2) and **exploit-audit** steps (watching replays of the champions).

---

## 1. The simulated game

### 1.1 Field and rules (WFDF metric by default)
| Item | Value |
|---|---|
| Field | 100 m × 37 m: 64 m playing proper plus two 18 m end zones |
| Players | 7 v 7 |
| Time step | 0.1 s for movement and decisions; disc flight is evaluated analytically, not stepped |
| Possession start | Offense starts with the disc on its own goal line, at a random x across the width (a standardized pull). A full-point mode with turnovers going back and forth is used for display and for final evaluation |
| Score | A catch inside the attacking end zone |
| Turnovers | Throwaway or out of bounds, drop, block or interception, stall-out |
| Stall | The count starts only when the marker is within 3 m. 1 count per second, turnover at 10 |
| Thrower | Cannot run. Can pivot; the release point sits within about 1.2 m of the pivot foot, on the side chosen |
| Not modeled in v1 | Fouls, picks, travels, pull hang time and bricks, double teams (they're illegal anyway), timeouts, fatigue (comes in v2), wind |

### 1.2 Players: stats and variability
Each player gets stats drawn from distributions at roster-generation time. A "random team" is 7 independent draws. We average results over **many random rosters**, and to cancel roster luck we use **mirrored matchups**: the same two rosters play with offense and defense swapped.

| Stat | Default distribution (club level) | Effect |
|---|---|---|
| Top speed | N(7.5, 0.6) m/s, clipped to [6.0, 9.2] | Running |
| Acceleration | N(4.0, 0.5) m/s² | How fast they reach top speed |
| Agility | U(0, 1) | Speed kept through a direction change (≥ 90° cut: keeps 30% to 60% of speed) |
| Reaction time | N(0.27, 0.04) s | **How late a defender reacts** to their attacker's change of direction. This is the cutter's head start and the main thing that makes cuts work |
| Height / vertical | Height N(1.80, 0.07) m, vertical U(0.4, 0.8) m | Wins contested high discs; catch/block radius |
| Hands | Uncontested catch probability U(0.965, 0.995) | Drops |
| Throwing (per throw type) | Range, accuracy, release time | See 1.3 |
| Break skill | U(0, 1) | Getting a break throw past the mark |
| Decision noise | U(0, 1) | How accurately the thrower *perceives* open lanes (noise added to its own completion estimate) |
| Marking skill | U(0, 1) | How much a marker takes away, and their hand-block chance |
| Bid aggressiveness | Set by the genome, not a stat | Layout attempts (see 1.4) |

Stats can be correlated where that's realistic: an athletic factor ties speed, acceleration and vertical together, and a "handler-ness" factor ties throwing and break skill together. That way rosters look like real teams. **Archetype rosters** (fast team, throwing team, tall team) are used for the "does the best structure depend on the roster?" analysis.

### 1.3 Throws and disc flight
Throw types in v1: **backhand, forehand, hammer**, plus **inside-out / outside-in** variants of backhand and forehand (curve direction) and a **huck** flag (floaty, high release, long hang time).

- **Flight path:** a quadratic Bézier from release point to target. The control point sets the curve (IO/OI). Speed profile: `t_flight = d / v0 + hang`, where v0 depends on throw type and distance: bullets 18 to 25 m/s, hucks lower with extra hang, hammers slow and turning over.
- **Max range per type per player:** backhand 45 to 70 m, forehand 35 to 60 m, hammer 15 to 40 m.
- **Error:** the actual landing point = intended target + Gaussian error with σ = a + b·distance. σ is multiplied up for off-hand / break throws, throws under a tight mark, hammers, and quick releases. A landing point out of bounds is a turnover (the receiver may still catch it in bounds before then, which is checked).
- **Release time:** 0.3 to 0.6 s from catch to first possible throw (skill). A throw must come off before the stall reaches 10.
- **Pivot and fake:** the thrower can spend 0.4 s on a fake. That shifts the marker's coverage for about 0.3 s (marker skill shortens this) and opens a break window.

### 1.4 When a D happens (the "fair" turnover model)
1. **Mark block (hand/foot):** the marker covers an angular sector around their body. Which side depends on the **force** (forehand, backhand, straight-up, middle, sideline). A throw released *through* the covered sector is blocked with probability `p_mark(angle, marker skill, thrower break skill, throw type, faked?)`. Hammers and high releases sharply reduce `p_mark`, and IO/OI release points change the angle. **This is where "breaking the force" lives.** Open-side throws are free from the mark. Break throws are possible but risky, and faking, pivot reach and break skill all shift the odds.
2. **Interception / point block:** for every throw we compute, for each defender, the **earliest time it can reach any point on the disc path** given position, velocity, acceleration, top speed and reach radius. This is the "pitch control" idea from soccer analytics, and it vectorizes cleanly. If a defender gets there clearly before the receiver, it's a likely D. If they arrive at about the same time, it's a **contested catch**, resolved by a logistic function of arrival margin, height/vertical, positioning (in front vs behind) and hands.
3. **Layout bid:** a defender can bid to extend reach by 1.5 m. If the bid fails they are on the ground for 1.5 s, which is the real cost of bidding. Bid aggressiveness is a defensive gene.
4. **Drop:** uncontested catch probability comes from the hands stat. It drops further with disc speed and with contest pressure.
5. **Stall-out:** reaching 10.
6. **Throwaway:** an error landing out of bounds, or a disc nobody reaches.

### 1.5 Movement
Players are point masses with an acceleration limit, a speed limit and **turning cost** (agility). Each tick, a player's behaviour gives it a target point (and urgency), and a steering controller moves it there. Defenders react to a **delayed** copy of their attacker's motion (reaction-time ring buffer). That delay is the source of separation on cuts, so it is one of the most important fairness knobs and gets calibrated.

All of these constants live in one config file (`configs/default.toml`) so you can argue with them.

---

## 2. Strategy genome

### 2.1 Offense genome
- **Formation:** 7 slots, each with:
  - a role weight (handler ↔ cutter)
  - a home position, expressed as a blend of *disc-relative* (dx downfield, dy toward the open side) and *field-absolute* coordinates
  - an x-spacing and y-spacing jitter tolerance.

  Coordinates are in a **force-aware frame** (open side vs break side), so one genome works when mirrored.
- **Cutting rules (per slot):** cut trigger (when the disc is caught / when the stall reaches k / when the lane in front is clear / when the previous cutter clears), the mix of cut types (under, deep, break-side, in-cut, fill), cut depth and angle, a **fake-step** before the cut, the clearing path afterwards, and cut **order / staggering** (who goes first, second).
- **Handler behaviour:** reset trigger (stall count), reset position (break-side dump, open-side, behind), swing tendency, and whether handlers cut upline after throwing ("give-go").
- **Thrower decision model:** at each tick the thrower scores candidate throws: each receiver × a few lead points × throw types × open/break release. The score is
  `U = p̂_complete · V(target) − (1 − p̂_complete) · C(turnover location)`,
  where `p̂` is the thrower's *noisy* estimate from the interception model and `V` is a learned **expected-points field** (see 3.3). Genes set the risk tolerance, the huck threshold, the break-throw willingness, the throw-type preferences, and a stall-dependent "must throw" threshold that falls as the count rises.

### 2.2 Defense genome
- **Scheme:** person-to-person / zone (cup + wings + deep) / hybrid (clam: switching on cutters). The mix is a categorical gene.
- **Force:** forehand, backhand, straight-up, middle, sideline. It can depend on field position, e.g. "force middle near the sideline".
- **Marker:** mark distance, sector width (wider covers more but is easier to break with a fake), and hammer awareness.
- **Person defense, per defender:** position relative to the assigned attacker. That covers fronting vs trailing, open-side vs break-side shading, and cushion as a function of distance from the disc. Also: **poach** rules (deep poach / lane poach and when to recover), help-D threshold (leave your attacker to go for a disc in the air), switching rules, bid aggressiveness, and matchup assignment (fastest on fastest, or by role).
- **Zone:** the slot positions of cup, wings, middle-middle and deep in disc-relative coordinates, plus rotation rules and the trap at the sideline.

### 2.3 Canonical seeds (baselines)
Hand-written genomes for **vertical stack, horizontal stack (ho stack), side stack, split stack, German/iso**, and on defense **person D with each force, cup zone, wall zone and clam**. They serve three purposes: (a) a sanity check that the engine produces sensible stats, (b) seeds and benchmarks for evolution, (c) the fixed opponents for Run B.

---

## 3. Evolution / self-play

### 3.1 Algorithm
- Two populations, **Offense (O)** and **Defense (D)**, about 64 genomes each.
- **Fitness of an offense genome** = P(score per possession). It is measured against a sample of D-population members **plus** a Hall of Fame of past D champions **plus** the canonical defensive baselines. Defense fitness is the mirror image.
- **Evaluation noise control:** each genome gets 200 to 1,000 possessions per generation (adaptive: more for contenders). **Common random numbers** mean every genome faces the same rosters and seeds in a generation. Rosters are resampled every generation so nobody overfits to one team.
- **Selection/variation:** tournament selection, Gaussian mutation with self-adaptive step sizes, crossover by whole slot (swap complete player-slot blocks), and elitism.
- **Diversity:** **MAP-Elites** archive keyed on structure descriptors (stack orientation, number of handlers, average depth). We get the best strategy *per structure family*, not just one winner. That directly answers "vertical vs horizontal".
- **Alternative optimizer:** CMA-ES for fine-tuning a family's champion once the GA has found the region.

### 3.2 Run A: best offensive structure
1. Seed O with canonical stacks plus random genomes. Seed D with canonical defenses plus random genomes.
2. Coevolve for N generations, with the Hall of Fame active.
3. **Final tournament:** the champion of each structure family vs a fixed, strong defensive league, over 50k+ possessions, with random rosters and 95% confidence intervals.
4. Report: score rate per structure, its breakdown (turnover causes, throws per point, huck rate, break-throw rate), and **roster-dependence** (does vertical win with fast teams but horizontal with good throwers?).

### 3.3 Expected-points field V(x, y)
The thrower needs to know how good a field position is. We **learn it from the simulator**: start possessions from a grid of positions, measure the scoring rate, and smooth it. Re-estimate a few times as strategies improve (a simple value iteration). This works like EPV in real ultimate analytics, and a bonus is that we get a "value of field position" map.

### 3.4 Run B: defending vertical and horizontal stacks
1. Freeze the offense: the **canonical** vertical and horizontal stacks *and* the **evolved champions** of each family from Run A.
2. Evolve D separately against each one (and against a mix).
3. Report the winning scheme per opponent: force, mark, positioning, poaching, zone vs person, with confidence intervals. Then **re-open the offense** for a few generations to check whether it adapts back. That tells us whether the defensive answer is robust or just exploitable.

---

## 4. Performance plan: C engine ("optimize the shit out of it")

### 4.1 Split of work
| Python (`ultisim/`) | C (`csim/`, built as `ultisim_core.dll` / `.so`) |
|---|---|
| Config loading, roster sampling | The possession loop: movement, mark and stall, throw choice, disc flight, intercepts, catches, all turnovers |
| GA / coevolution / MAP-Elites / Hall of Fame | Per-game random numbers (fast generator, seeded per game so runs can be replayed exactly) |
| Value-field V(x, y) fitting | Writing out the stats for each game (score, turnover cause, throws, hucks, breaks) |
| Analysis, tournaments, reports | Optional replay recording (positions at 10 Hz) for the games you want to watch |
| Viewer server and dashboard | |

**One call simulates a whole batch.** Python hands C flat arrays: the config constants, rosters, one offense and one defense genome per game, and seeds. A single function `sim_batch(...)` plays N possessions and fills the result arrays. There are no per-tick calls into C, so the cost of crossing between Python and C is negligible.

**Interface:** `ctypes` from the standard library, with no extra dependency. The data contract is a set of C structs mirrored by numpy structured dtypes (`csim/ultisim.h` ↔ `ultisim/cabi.py`), with a version number checked at load time so the two can't silently drift apart.

**Multithreading:** `ctypes` releases the Python lock (the GIL) while a C function runs. So Python splits a batch into 8 chunks and calls `sim_batch` from 8 threads, and all cores run C at full speed. That means no multiprocessing, no pickling, and no OpenMP requirement (it can still be added if the compiler supports it).

### 4.2 What makes the C fast
1. **Scalar loops over games.** Each game is a tight loop over ticks and players with all state in a small struct that fits in cache. This is simpler and faster than the numpy batching it replaces, because games can end at different times without wasted work.
2. **Throw evaluation is the hot spot.** Each decision checks receivers × lead points × throw types against 7 defenders, so it's optimized specifically:
   - closed-form "time to reach a point" for each defender (acceleration plus top speed, no stepping)
   - about 8 sample points per flight path
   - pruning of impossible candidates early (out of range, a receiver moving away)
   - decisions only every 0.2 to 0.3 s, or when something changes
3. **Compiler flags:** `-O3 -march=native -ffast-math`, where the rules allow it. Tests check that results stay statistically identical.
4. **Profiling gate:** a `bench.py` script that we run after every engine change. No silent slowdowns.

**Target:** ≥ 20k possessions/second on 8 cores, about 2 to 3 s per generation. That gives room for bigger populations and more samples, which means less noisy answers. `bench.py` decides whether we've hit the target, not this estimate.

### 4.3 Correctness: the C engine must match the reference
- The Python reference sim and the C engine run the same rules. A **parity test** checks that they produce the same game for the same seed, either exactly (deterministic pieces) or within confidence intervals over 100k possessions (whole-game statistics).
- New rules are prototyped in Python first (fast to change, easy to watch in the viewer), then ported to C once they're settled.
- In debug builds the C code checks for bad states (NaN positions, a player outside the field by more than 5 m, stall > 10) and aborts with the game seed so the game can be reproduced.

### 4.4 Toolchain (Windows)
**Installed (2026-11-05):** WinLibs GCC 16.2.0 (UCRT, POSIX threads) via winget, user scope. It is at `%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_*\mingw64\bin` and is on PATH in new shells. Smoke test passed:
- a `-O3 -march=native -fopenmp` DLL loads via ctypes
- OpenMP sees 8 threads
- 8 Python threads calling C give a **7.9× speedup**, which confirms that ctypes releases the GIL.

OpenMP links dynamically (`libgomp-1.dll`), so `build_c.py` copies the runtime DLLs next to `ultisim_core.dll`.

The options that were considered:
- **GCC via WinLibs/MinGW-w64** (recommended): `winget install BrechtSanders.WinLibs.POSIX.UCRT`, about 250 MB, no admin needed. It's fast, supports OpenMP, and builds the same way on Linux/Mac later.
- **MSVC Build Tools:** the official Microsoft compiler, a large install (several GB) and admin rights.
- **zig via pip** (`pip install ziglang`, then `python -m ziglang cc`): no system install at all and the lightest option, but no OpenMP. We don't need OpenMP anyway (see 4.1).

The build is a single `scripts/build_c.py` that finds the compiler and builds the library. No CMake.

### 4.5 Later
CUDA/GPU batching or SIMD-vectorized intercepts if we ever need 100× more.

---

## 5. Visualization engine

A **browser-based viewer** (HTML5 Canvas plus plain JS, no build step), served by a tiny stdlib Python server:

- **Replay viewer:** top-down field with players (offense/defense colours, numbers, stat-tinted), the disc and its planned flight arc, marker sector and force arrow, stall count, the cut paths of the last few seconds, and event flags (D, drop, huck, break). Controls: play/pause, speed 0.25× to 8×, scrub, step tick by tick, and click a player to see their stats.
- **Live mode:** the trainer writes `runs/<run>/latest_best.json` (a replay of the current champions) every K generations. The viewer polls it and auto-loads the new champion.
- **Evolution dashboard:** best/mean fitness per generation for O and D, the share of each structure family in the population over time, the MAP-Elites grid (structure vs score rate), a formation heatmap of the current champion (where each slot spends its time), the turnover-cause breakdown, and the V(x, y) field.
- **Side-by-side mode:** the same roster and seed, two different strategies, so we can compare.

Replays are compact (positions quantized to cm as int16, at 10 Hz), so thousands of them are cheap to store.

---

## 6. Project layout

```
frisbee_strat/
  PLAN.md
  pyproject.toml
  configs/default.toml        # every fairness/physics knob in one place (stdlib tomllib)
  ultisim/
    config.py                 # load/validate config
    roster.py                 # stat distributions, archetypes, correlated sampling
    disc.py                   # throw types, flight paths, error model
    intercept.py              # time-to-reach / pitch-control math (reference)
    engine_ref.py             # pure-Python reference sim (rule prototyping, parity tests)
    cabi.py                   # ctypes bindings + numpy dtypes mirroring ultisim.h
    engine.py                 # thin wrapper: threads sim_batch() over 8 cores
    offense.py                # offense genome -> per-tick targets + throw choice
    defense.py                # defense genome -> positioning, mark, poach, zone
    genome.py                 # encoding, mutation, crossover, canonical seeds
    value_field.py            # learned expected-points field V(x, y)
    evolve.py                 # GA / coevolution / Hall of Fame / MAP-Elites
    classify.py               # structure descriptors and family labelling
    replay.py                 # record/serialize replays
    analysis.py               # tournaments, CIs, reports
  csim/
    ultisim.h                 # data contract: structs, ABI version
    sim.c                     # possession loop, rules, events
    move.c, throw.c, intercept.c, rng.c, replay.c
  viz/
    server.py                 # stdlib http server for runs/
    index.html, viewer.js, dashboard.js
  scripts/
    build_c.py, bench.py, calibrate.py, run_offense.py, run_defense.py, tournament.py
  tests/                      # rules, geometry, determinism, regressions
  runs/                       # outputs (gitignored)
```

---

## 7. Phases and milestones

| Phase | Deliverable | Done when |
|---|---|---|
| **0. Setup** | venv, numpy/pytest, C compiler, repo skeleton, config file, a "hello" DLL loaded via ctypes | `pytest` runs; `build_c.py` builds and Python calls into C |
| **1. Core engine** | Single-game, readable reference sim: movement, mark/stall, throws, intercepts, all turnover types, canonical scripted strategies | Unit tests pass. Vertical and ho stacks *look right* in the viewer |
| **1b. Viewer v1** | Replay viewer for the reference sim | You can watch a possession and pick out every rule event |
| **2. Calibration** | Tune constants until canonical-vs-canonical stats look like real club ultimate: completion rate ~90 to 95%, hold rate ~55 to 70%, ~8 to 15 throws per scoring possession, huck completion ~45 to 60%, some breaks, rare stalls | Stats are inside the target bands. **You sign off on realism** |
| **3. C engine** | Port the calibrated rules to `csim/`, add ctypes bindings and threaded batches, parity tests and the benchmark | Parity tests pass (exact where deterministic, within CIs otherwise); ≥ 20k possessions/s |
| **4. Run A** | Offensive coevolution, MAP-Elites, final tournament, report | Ranked structure table with CIs. Champion replays audited for exploits |
| **5. Run B** | Defense vs vertical, vs horizontal, vs mix, plus the counter-adaptation check | Ranked defensive scheme table per opponent |
| **6. Dashboard + report** | Live evolution dashboard and a write-up of findings | You can watch a run live and read the conclusions |
| **7. (Later)** | Fatigue, wind, mixed rosters, neural policies, GPU/SIMD engine | Separate plan |

**Exploit audit (runs every time a champion appears):** watch its replays and check its stats for physically silly behaviour, such as a defender standing at an exact spot, a 0 m mark, or 60 m hammers. Each one we find becomes a rule/engine fix and a regression test, and the run restarts.

---

## 8. Main risks and how we handle them

- **The simulator is wrong, so the conclusions are wrong.** Phase 2 calibration, exploit audits, and sensitivity analysis: rerun the final tournament with key constants ±20% (reaction time, throw error, hands) and check that the ranking holds.
- **Noisy fitness.** Common random numbers, adaptive sample counts, and CIs on every claim.
- **Coevolution cycling or collapse.** Hall of Fame, a league of canonical opponents, and MAP-Elites diversity.
- **The genome is too narrow to discover new structures.** The generic 7-slot formation, plus the optional neural-policy phase.
- **Python is too slow.** The hot loop is in C, with threaded batches across all 8 cores.
- **The C and Python versions drift apart.** Parity tests in CI, an ABI version check at load time, and every rule change going through the reference sim first.
- **Crashes or memory bugs in C.** Debug builds with sanitizer and assert checks, every failing game reproducible from its seed, and Python validating every result array.

---

## 9. Decisions (settled 2026-11-05)

| # | Decision | Choice |
|---|---|---|
| 1 | Rules and units | **WFDF metric**: 100 × 37 m, 18 m end zones |
| 2 | Level | **Club/college** stat distributions (section 1.2) |
| 3 | Division | **Open/single-gender** for now; mixed later |
| 4 | Evaluation unit | **Possessions for training**, with **full points** (turnovers back and forth) to validate finalists |
| 5 | Zones in Run B | **Included**: person D (every force), cup, wall, clam/hybrid |
| 6 | Viewer | **Browser-based** (Canvas + live dashboard, local stdlib server) |
| 7 | Fairness constants | **Plan defaults** (sections 1.2 to 1.4); Phase 2 calibration tunes them |
| 8 | C compiler | **WinLibs GCC 16.2** via winget, installed and smoke-tested (section 4.4) |
| 9 | When C comes in | **After calibration**: Python reference first, C port in Phase 3 with parity tests |

---

## 10. Status log

### 2026-11-05: Phase 0 and Phase 1 done (reference engine + viewer v1)

**How to run**
```bash
.venv/Scripts/python -m pytest -q                 # 43 tests
.venv/Scripts/python scripts/build_c.py           # C core (Phase 0 stub: ABI + smoke test)
.venv/Scripts/python scripts/demo.py -n 100       # canonical matchup table + replays in runs/demo/
.venv/Scripts/python viz/server.py                # viewer at http://127.0.0.1:8765/
```

**What exists**
- `ultisim/engine_ref.py`, the readable reference engine. It covers possession and full-point modes; movement with braking/turn cost and the defender reaction delay; force-aware marking with a coverage wedge, fakes and stall; BH/FH/hammer throws with IO/OI curves, hucks and error; pitch-control-style intercepts; contested catches; layout bids; drops; OOB; and the double-team rule.
- Canonical offenses: vertical, horizontal, side, iso. Canonical defenses: person D with each force, cup zone, wall zone.
- Genomes with a flat-vector schema, ready for the GA.
- Replay recorder and browser viewer.
- Reference speed: about 16 possessions/s on 8 processes. That's enough for calibration; the C port (Phase 3) is for the GA.

**Bugs found while building (fixed, so behaviour is now sensible)**
- Defenders could block inside 3 m of the thrower. Now only the mark can.
- Defenders could retry a failed block on every tick.
- The thrower's model didn't match the sim on defender reaction time.
- The disc stopped dead at its aim point. It now glides on for 0.35 s.
- Defenders lagged moving attackers. They now match the attacker's velocity.
- At the start of a possession, defenders were placed deep of their attackers instead of at their scheme positions.

**Realism gap: the input to Phase 2 calibration** (1,920 canonical-vs-canonical possessions)
| Metric | Sim now | Club target |
|---|---|---|
| Completion rate | 0.76 | 0.90 to 0.95 |
| Score rate per possession from the goal line | ~0.20 | ~0.55 to 0.70 |
| Throws per scoring possession | 5.5 | 8 to 15 |
| Huck completion | ~0.79 | 0.45 to 0.60 |
| Main turnover causes | interceptions > blocks > drops | |

Throwers let go at stall ~2.5 on throws the sim completes ~85% of the time, and they rarely hold for better options. Likely levers, in order:
1. The value field and the hold-vs-throw rule (`patience`, `risk_aversion`, the shape of V/C). These make throwers too willing.
2. Cutters don't get open enough: reaction-time head start, cut timing, jab effect.
3. The interception margin (the 0.12 s sigmoid).
4. Huck hang time and error, which make hucks too easy.

The canonical-matchup table is **not meaningful yet**. Don't read strategy conclusions from it until calibration is signed off.

### 2026-11-05: Phase 2 calibration, awaiting your sign-off

**How to rerun**
```bash
.venv/Scripts/python scripts/fit_throw_model.py --rounds 2 -n 16 --eps 0.25 --fresh   # refit after physics changes
.venv/Scripts/python scripts/calibrate.py -n 48 --points 320 --matchups              # stats vs target bands
.venv/Scripts/python scripts/calibrate.py -n 24 --set throws.err_b=0.036              # any dotted-path override
```

**Result** (768 canonical possessions plus 320 full points; saved in `runs/calib/calibrated.json`)
| Metric | Before | Now | Club target |
|---|---|---|---|
| Completion rate | 0.76 | **0.948** | 0.90 to 0.95 |
| Score rate per possession (from the goal line) | 0.21 | **0.517** | 0.40 to 0.60 |
| Hold rate (full points) | n/a | **0.566** | 0.55 to 0.72 |
| Throws per scoring possession | 5.3 | **9.4** | 8 to 15 |
| Huck share / completion | 12% / 0.81 | 5% / **0.79** | 4 to 15% / 0.45 to 0.62 |
| Break share / completion | 21% / 0.74 | 19% / 0.91 | |
| Completion by distance (0–10 / 10–20 / 20–35 / 35+ m) | | 0.97 / 0.94 / 0.91 / 0.78 | |
| Turnovers: drops / Ds / throwaways+OOB / timeouts | | 34% / 35% / 23% / 7% | ~20 / ~30 / ~50 / 0 |

**Engine changes from calibration**
1. **Learned throw-completion model** (`configs/throw_model.json`, fitted by `scripts/fit_throw_model.py`). It's a logistic regression on 22 physical features, fitted from self-play with exploration, and it replaces the hand-built estimate. Held-out log-loss improved from about 0.55 to 0.33, and its predictions are calibrated within about 2 points in every bucket. **Refit it whenever the physics change.**
2. Fixes to the thrower's model:
   - defenders keep moving during their reaction delay
   - throw error widens the lane a defender can reach into
   - the receiver's defender shadows them only once they've read the throw
   - the catch-point formula is less generous to late defenders
3. **New rules:**
   - Help defenders in person D read throws 0.35 s later, because they're watching their own player.
   - A disc that reaches a defender before they've read it is only a reflex chance (×0.35).
   - Hucks are read 0.3 s later by everyone.
   - Huck catches are harder (−0.12), being over the shoulder at a sprint.
   - Throwers get impatient based on how long they've held the disc, as well as the stall count.
4. Swept touch test now uses relative motion, so a disc can no longer pass through a fast receiver between ticks.
5. **Constants changed:**
   - reach 0.75 → 0.90 m (includes a lunge step)
   - hands range 0.98 to 0.998
   - smaller multipliers stacking on short-throw error
   - long-throw error grows with distance squared
   - faster, flatter hucks (17.5 m/s, 0.45 s hang)
   - field value curve 0.45 + 0.45·u^1.5
   - less thrower misjudgement
6. **Canonical person D now shades the open side** (angle 65°, 2 m cushion) instead of fronting, which is closer to club standard.

**Sensitivity (±20%, n = 20 per matchup, score-rate SE ≈ 0.03):** there are no cliffs.
- Throw accuracy matters most: `err_b` ±20% gives a score rate of 0.54 / 0.39.
- Defender reaction time hardly matters (0.475 / 0.472).
- `block_prob` 0.70 gives 0.47; help read delay 0.28 / 0.42 gives 0.52 / 0.50.

**Known deviations (yours to accept or push back on)**
1. **Hucks complete 79% but are only 5% of throws.** Throwers now huck only when the deep cutter is wide open, so the real-world mix of 50-50 hucks doesn't appear.
2. **The turnover mix leans toward drops (34%) and away from throwaways (23%).** Completion is at the top of the band, so the other causes are rare.
3. **No stall-outs.** Instead, about 3% of possessions hit the 120 s cap from endless dump-swing. These count as turnovers, so the GA won't be rewarded for them.
4. **Defender reaction time barely moves outcomes**, so cut-vs-defender footwork is not what decides possessions right now. Throw accuracy and the thrower's decisions are.
5. **The cup zone holds every stack offense to 21–38%.** That's plausible, because none of the canonical offenses is a zone offense. Finding one is the GA's job.

### 2026-11-05: Phase 3, C engine done

**How to use**
```bash
.venv/Scripts/python scripts/build_c.py                         # regenerates csim/us_structs.h, builds build/ultisim_core.dll
.venv/Scripts/python scripts/bench.py -n 4000                    # throughput
.venv/Scripts/python scripts/parity.py --n-py 48 --n-c 1500      # C vs Python reference, statistically
.venv/Scripts/python scripts/calibrate.py --engine c -n 400 --points 1600   # full calibration in ~30 s
```
```python
from ultisim.engine import CEngine
eng = CEngine(cfg)                                       # all cores
res = eng.possessions(offense_genome, defense_genome, roster_pairs, seeds)   # numpy structured array
res = eng.run(offenses, defenses, specs)                 # general batch: many genomes, one call
```

**What exists**
- `csim/sim.c` is a function-by-function port of `engine_ref.py`, with every rule, the learned throw model and full-point mode. Its random-number generator is xoshiro256**, seeded per game, so a result never depends on the thread count.
- One C call simulates a whole batch across all cores (OpenMP). Python only packs the inputs as numpy arrays.
- **The struct layouts are generated from `ultisim/cabi.py`** when the C code is built, and checked again at load time (ABI version and struct sizes), so Python and C can't silently disagree. Genomes cross the boundary as the same flat vectors the GA will mutate.
- **Tests:** 54 in total. The exact-parity tests cover time-to-reach, flight path, throw error, mark block, movement step and the matching algorithm (all match Python to 1e-9). The C-engine tests check valid results, determinism across thread counts, full points, and that genome changes affect outcomes.

**Statistical parity** (same rosters, 16 canonical matchups)
- No metric differs by more than 3 SE.
- Score rate 0.444 vs 0.453, completion 0.941 vs 0.941, throws per possession 8.8 vs 8.7, D share 0.381 vs 0.396 (rerun with a larger sample).
- **Watch item:** drop share comes out about 2 SE lower in C in both runs. That's within tolerance, but worth re-checking if it grows.

**Speed** (i7-10610U: 4 cores / 8 threads, 1.8 GHz ultrabook, timings vary ±25% with heat)
| | possessions/s |
|---|---|
| Python reference, per process | ~1.4–4 |
| C, 1 thread | ~90–135 |
| C, all threads | **~270–470** |

**This is far below the 20k/s target.** That target was set before calibration made each throw decision much richer. A decision now scores 84 candidate throws against 7 defenders at 10 points along each flight (lane, drift, shadowing and reflex rules), then runs the learned model. That happens about 120 times per possession, roughly 10,000 candidate evaluations each.

Optimizations already in:
- AVX2-vectorized defender kernel
- fast `exp`
- skipping defenders who provably can't matter (exact once the ±2 s feature clipping is applied)
- reusing duplicate candidates

**What it means for Phase 4:** 64 offense genomes × 300 possessions is about 19k possessions, roughly 45–70 s per generation on this laptop. 100 generations takes about 1.5–2 h; coevolving defense doubles that.

Ways to go faster, in order of payoff:
1. A machine with more cores. Speed scales roughly linearly, so a 16–32-core desktop or cloud box gives 4–8×.
2. Behaviour changes (need your OK and a recalibration): decide every 0.3–0.4 s instead of 0.2 s, and trim the 7-throw menu. Roughly 2–3×.
3. float32 with hand-written AVX2/AVX-512 intrinsics: roughly 1.5–2×.
4. GPU batching.

**Not ported (by design):**
- **Replay recording:** champion replays are rendered with the Python reference, which plays the same strategy.
- **Per-throw logs:** the throw model is refit with the Python reference (`scripts/fit_throw_model.py`).

### 2026-11-05: Phase 3 speed pass

**Result: about 860–1,010 possessions/s on all threads (272/s on one), up from 270–470.** That's roughly 2.5–3× faster, and about 70–200× the Python reference per core.

| Change | Game changes? | Gain |
|---|---|---|
| Throwers re-read the field every **0.4 s** instead of 0.2 s (`sim.decision_interval`) | yes, small (see below) | ~1.6× |
| Curved (IO/OI) throws and hammers only considered for **break-side** targets (`throws.specials_everywhere = false`) | yes, small | ~1.15× |
| Chase logic for discs in the air: arrival times computed in one vectorized pass instead of early-exit loops | no (identical logic) | ticks 4.8 → 2.75 µs |
| Probability loop skips defender/point pairs > 2 s behind the disc (sigmoid < 6e-8; the margin features are still exact) | no (negligible) | ~1.1× |

**Tried and reverted:**
- A single-precision 8-lane defender kernel was *slower*, because it gives up skipping far-away defenders, and that skipping is worth more than wider vectors.
- A Padé sigmoid didn't vectorize the probability loop either.

**Game check after the change:**
- Throw model refit on 12.8k throws (`--rounds 3 -n 32`). The previous model is saved as `runs/throw_model_phase2.json`.
- Calibration on the C engine (6,400 possessions + 1,600 points):

| Metric | Value |
|---|---|
| Completion | 0.938 |
| Score rate | 0.454 |
| Hold rate | 0.559 |
| Throws per score | 8.3 |
| Hucks | 3.4% of throws, 71% completed |
| Breaks | 16% of throws, 88% completed |

  These are all within the signed-off profile. The known deviations are the same as before: drop-heavy turnover mix, few but easy hucks, no stall-outs.
- Parity, C vs Python on fresh rosters (1,792 vs 32,000 possessions): every metric within 2.1 SE. Completion 0.938 vs 0.938, score 0.451 vs 0.452.

**GA budget now:** a generation of 64 genomes × 300 possessions takes about 20–25 s on this laptop. 100 generations is about 40 min (offense only), or about 1.5 h coevolving.

**Further speed would need:** more cores (linear), a slimmer learned throw model with cheaper features, or GPU batching. Evaluation-side savings (racing / successive halving of genomes, common random numbers) belong in Phase 4.

### 2026-11-05: Phase 4 code ready (small tests only, overnight run to be done on the Mac mini)

**What exists**
- `ultisim/classify.py`: labels each offense with a structure family (vertical / horizontal / side / iso / spread / other) from its cutter formation, plus continuous descriptors. It labels all the canonical strategies correctly.
- `ultisim/evolve.py` and `scripts/run_offense.py`: coevolution of offenses and defenses.
  - **Opponents:** each offense plays 3 population defenses (balanced round-robin), 1 from the Hall of Fame and 2 canonical defenses.
  - **Fairness:** common rosters per generation.
  - **Racing:** everyone plays `stage1` possessions; the top third play up to `stage2`.
  - **Breeding:** elitism, tournament selection, slot-block crossover and Gaussian mutation.
  - **Records kept:** a best-per-family archive, a checkpoint every generation (`--resume`), the files the dashboard reads (`viz/RUN_FORMAT.md`), and champion replays rendered in the background.
- `scripts/audit_defenses.py`: re-evaluates a run's defenses against the canonical offenses with many possessions (the exploit audit).
- A new canonical **zone offense** (`zone_o`).
- **Portable C build:** it probes the CPU and OpenMP flags. Apple clang works with or without libomp; without OpenMP, `CEngine` splits batches over Python threads (tested: 1,268 vs 1,396 possessions/s here, with identical results).
- `RUNNING.md`: the Mac mini setup and overnight guide.

**Found by smoke runs**
1. **Exploit, fixed:** a wider mark wedge had no downside, so evolution pushed it to 140°+ and defenses conceded only 9–12%. Blocking capacity is now fixed: wedges wider than 50° spread the same peak thinner (`defense.mark_width_ref`). There's a regression test.
2. **Watch item, not a bug:** evolved zones hold the canonical *stack* offenses to 2–23% (the canonical cup zone allows 17–35%). A hand-built zone offense scores 41% against the cup zone and 25% against the best evolved zone, so the simulator isn't simply zone-biased. Offenses have to discover zone offense. If zones still dominate after a long run, audit them before believing it.

**Tests:** 62 passing.

### 2026-11-05: Defense abilities, hammer realism, open-ended evolution

**Exploits found by watch runs (fixed)**
- **Wide mark wedge:** a wider wedge had no downside. Fixed first with a fixed blocking capacity, then properly with a **body-based mark**: coverage half-width = atan(`mark_reach` / `mark_dist`), and block chance × (1 − exp(−dist/0.8)), so tight marks react later. The `mark_width` gene is gone.
- **Hammers:** they went over the mark and couldn't be touched mid-flight, with no downside. They now get a −0.10 catch penalty and a ×2.0 error multiplier. The hand-built zone offense no longer has a hammer preference. Hammer share in evolution is now about 0.01% of throws.

**New defensive abilities** (genes in both engines; struct ABI 6)
- **Flex/hybrid zones:** `zone` = share of 7 players in zone. k = 0 is person D, 7 is full zone, 1–6 is the marker plus k−1 zone spots with person D on the cutters picked by `flex_focus`.
- **Clam switching:** `switch`.
- **Help defense on a beaten cutter:** `help_cut`.
- **Surround-the-stack sag:** `stack_sag`.
- **Zone-to-person transition** after N completions: `transition_throws`.
- **Sideline trap:** `trap`.
- **New seeds:** clam, help_d, flex_deep, junk_cup, zone_to_person, cup_trap.
- **Labels:** flex is shown as `flex_<k>z<7-k>p`.

**New offensive ability**
- `seek_space` (per slot): bends cuts toward open space, away from nearby defenders.

**Open-ended evolution** (`ultisim/evolve.py`)
- **Novelty search:** parents are selected on rank(score) + 0.35 × rank(novelty). Novelty is the distance to the 10 nearest behaviour fingerprints in the population plus an archive. Offense fingerprint: huck/break/hammer share, tempo and formation shape. Defense fingerprint: scheme genes.
- **Niche elites:** the best member of every offense family and every defense family (person/flex/zone) always survives.
- **Fresh ideas:** 3 immigrants per generation (random, or heavily reshaped survivors), plus a 6% chance of a macro-mutation that redraws a whole formation slot.

**Parity:** passes at 3 SE on all new defenses. `scripts/parity.py` now replays Python's exact rosters in C; before, different roster sets caused false alarms.
- **Open item:** in some samples cup-zone matchups lean lower in C (z ≈ −2.5); a fresh-roster check agreed. Settle it with a dedicated large comparison before the overnight run.

### 2026-11-05: Run watch3 results, and the break exploit fixed

**What watch3 showed** (100 generations, 768k possessions, 11 min; novelty search on)
- **Diversity held.** Every offense family stayed alive. The champion family rotated: iso 39 generations, vertical 24, side 16, spread 14, horizontal 6. Defense schemes moved from person-middle to flex_1z6p to person-sideline.
- **Arms race.** Average offense score fell from 0.42 to 0.34 and average conceded from 0.43 to 0.28, so defenses learned faster.
- **Evolved defense.** Person D forcing sideline, marker at the 3 m limit, defenders fronting (between attacker and disc) with a cushion that grows with distance from the disc, constant switching, always laying out. It concedes 11–19% to the standard offenses.
- **Evolved offense.** A 3–4-handler pod within 10 m of the disc, one mid cutter at about 20 m and one deep at about 35 m, no resets until stall 9, frequent fakes and seek-space cuts. It scored 0.57–0.63 against all standard defenses, but **mostly through an exploit**: it threw 45–92% of its throws as breaks at 89–97% completion.

**The break exploit and the fix**
- **Cause 1:** the mark only covered a body cone. Anything outside the cone was unguarded.
- **Cause 2:** the mark judged a throw by its first 2 m of flight, so a curved around-break escaped completely.
- **Fix, part 1:** a break-side tail of block coverage (`mark_break_tail` = 0.6, fading linearly to zero at `mark_tail_deg` = 65° off the marker).
- **Fix, part 2:** throws are judged along the bisector of their release and target directions.
- **Fix, part 3:** curved throws carry ×1.25 error.
- **Fix, part 4:** "break" now means aimed within 50° of the marker (`break_side_deg`), not "inside the narrow cone".
- **Effect:** the standard stacks break 9–16% of throws against person D at 0.77–0.81 completion (real: 10–20%, 0.80–0.85). The watch3 champion offenses fall back to 0.47–0.49, level with the standard stacks.

**Checks:**
- Throw model refit on 20.9k throws; the previous model is saved as `runs/throw_model_pre_breaktail.json`.
- Calibration on C: completion 0.929, score 0.42, hold 0.56, 8.5 throws per score, breaks 5.3% overall (about 10% against person D, about 1% against zones) at 0.80 completion.
- Parity: all overall metrics within 1.5 SE.
- 92 tests pass.
- No new evolution run yet.


### 2026-11-06: Disc heights, reach envelopes and lofted throws

**What changed** (both engines plus the thrower's model; ABI 11)
- **Height profile per throw.** Release at 1.0 m (hammers overhead at 2.2 m), a peak at mid-flight, catch at 1.2 m, and a tail sinking to 0.3 m. Peaks: flat 1.0 + 0.025·d, lofted 2.0 + 0.05·d (capped at 4.5 m), huck 2.5 + 0.09·d, hammer 6 + 0.05·d. This replaces the old fixed "untouchable windows" for hucks and hammers.
- **Reach envelope.** Standing reach = 1.25 × height; jumping adds the vertical but drops the lateral radius to ×0.6. A disc above someone's envelope can't be touched by them. Discs below 0.4 m catch/block at ×0.6. No layouts for discs overhead.
- **Flat vs lofted.** Straight backhand/forehand throws from 6 to 35 m can also be lofted: 25% slower, 0.35 s extra hang. The learned model gets a `lofted` feature (23 features).
- **Zone read delay.** Zone spots follow where the disc was 0.45 s ago, so swings can beat a cup. Before this, the instantly-shifting cup plus the new height model held stack offenses to 0.09 with 40% time-outs.
- **Possession urgency.** After 40 s of possession, patience and risk aversion fade linearly to 40% by 80 s. Time-outs went from 13% of possessions to 0.

**Results**
- Throw model refit on 27k throws. Earlier models are saved as `runs/throw_model_pre_height.json` and `runs/throw_model_pre_breaktail.json`.
- Calibration on C: completion 0.924, score 0.389, hold 0.537, 9.3 throws per score, about 25% of throws lofted (completing 0.90 vs 0.86 for flat), breaks 5% overall at 0.82, hucks 3.6% at 0.66, Ds 52% of turnovers.
- Score and hold sit about 3% under their bands. Lowering `block_prob` to 0.78–0.82 didn't move them, because throwers adapt, so it's left at 0.88.
- Hammers are now almost never chosen.
- Parity: all overall metrics within 1.4 SE.
- 94 tests pass.


### 2026-11-06: The watch4 defense exploit (fronting)

**Diagnosis**
- watch4's `flex_1z6p` champion conceded 0.043 to the hand-built offenses; person forehand concedes about 0.39.
- Not re-matching: turning `switch` off gave 0.044, and the same genes as plain person D gave 0.045.
- Gene ablations: resetting `angle_near` gave 0.167, `angle_far` 0.183, `mark_angle` 0.148. The defense **fronts** (angle about -4 deg, cushion 1.9 m + 0.22 m per m).
- Against it, 48% of possessions ended in downfield blocks (15% against person D). Throwers expected 0.93 completion and got 0.83. 90% of the Ds came from the receiver's own defender, standing about halfway along a ~9 m throw at release.

**Root causes and fixes** (both engines; ABI 12)
1. **In-flight positioning used the live disc.** Person defenders not chasing positioned against `flight_pos_now()`, so a fronting defender tracked the disc onto the lane with no read. Now all positional defenders use the disc position `zone_read_delay` (0.45 s) ago, as zone players already did. Effect: 0.048 -> 0.084.
2. **The thrower's read assumed drift.** The evaluator moved defenders who hadn't read the throw on their current velocity. Now the receiver's person defender holds their genome spot (`person_spot`, shared with movement): the spot slides with the receiver to the catch point, and they close any gap at top speed. Regression test: `test_thrower_sees_fronting_defender_step_into_the_lane`. Effect: -> 0.107.
3. **Blind side.** A defender fronting their attacker reacts up to `blind_delay` = 0.35 s later to the attacker's moves. Small effect on its own.
4. **Throw model on diverse defenses.** `fit_throw_model.py` now also plays 80 random pairs per round of heavily mutated hand-built strategies and evolved ones from `runs/*/state.pkl`. On that data the old model's held-out log-loss was 0.485 and the refit's 0.358. The previous model is saved as `runs/throw_model_pre_diverse.json`.

**Results**
- The champion concedes **0.12** to the hand-built offenses (person FH 0.39, cup 0.34).
- **It has a counter.** An offense-only GA against the frozen champion (32 offenses, 30 generations, 200 possessions each) went from 0.10-0.17 to **0.33** on 1,500 fresh possessions. The evolved offenses throw more aggressively (risk aversion about 0) and deeper, and still score 0.34-0.39 against person D. So fronting is a strong defense that coevolution can answer, not an exploit.
- Calibration on C: completion 0.927, score 0.399, hold 0.546, 9.6 throws per score, hucks 3.2% at 0.64, Ds 46% of turnovers (back in band).
- Parity on the canonical matchups and on the watch4 champion: all overall metrics within 1.6 SE.
- 95 tests pass. About 400-540 possessions/s.
- Runs watch1-4 predate this fix.
