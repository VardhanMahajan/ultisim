# UltiSim: evolving Ultimate Frisbee strategy

UltiSim is a 7v7 Ultimate Frisbee simulator plus a **self-play evolution engine** that discovers offensive and defensive strategies by making them play each other millions of times. The idea is the AlphaZero / AlphaStar loop (play yourself, keep what wins, repeat) applied to a sport with 14 players moving continuously at once.

It answers questions like:

- **Which offensive structure scores most often** (vertical stack, horizontal stack, side stack, iso, something new) for a team with average, random player stats?
- **Which defense works best** against a given offense: person D with each force, zones, flex/hybrid zones, clam switching, help defense?

```
offense genome ─┐                        ┌─> score rate per possession ─┐
                ├─> 7v7 simulator (C) ───┤                              ├─> selection + novelty ─> next generation
defense genome ─┘   millions of games    └─> turnovers, hucks, breaks ──┘
```

**Status:** research prototype. The simulator is calibrated to club-level statistics and the evolution loop runs end to end. Results so far are in [Findings so far](#findings-so-far). Read them with the caveats there.

---

## Quick start

Requirements: Python 3.12+ and a C compiler (GCC, Clang or Apple clang). Full per-OS instructions are in [docs/SETUP.md](docs/SETUP.md).

```bash
python -m venv .venv
.venv/bin/pip install numpy scipy pytest          # Windows: .venv\Scripts\pip ...
.venv/bin/python scripts/build_c.py               # builds the C engine (probes CPU + OpenMP flags)
.venv/bin/python -m pytest -q                      # ~95 tests, about 2 minutes
.venv/bin/python scripts/bench.py -n 4000          # possessions per second on your machine
```

Run a small evolution and watch it live:

```bash
.venv/bin/python scripts/run_offense.py --run demo --gens 30 --off-pop 48 --def-pop 32
.venv/bin/python viz/server.py                     # in a second terminal
# open http://127.0.0.1:8765/dashboard.html?run=demo
```

For long or overnight runs, see [RUNNING.md](RUNNING.md).

---

## What's inside

| Part | What it does | Where |
|---|---|---|
| **Simulator, reference** | Readable pure-Python 7v7 engine. It's the source of truth for the rules and the place new mechanics are prototyped. | `ultisim/engine_ref.py` |
| **Simulator, fast** | Function-by-function C port, with OpenMP batches. About 70–200× faster per core. | `csim/sim.c`, `ultisim/engine.py` |
| **Learned throw model** | Logistic regression (23 features) the thrower uses to estimate completion odds. Fitted from self-play. | `scripts/fit_throw_model.py`, `configs/throw_model.json` |
| **Strategy genomes** | 97 numbers per offense and 32 per defense. 5 hand-built offenses and 13 defenses serve as seeds. | `ultisim/genome.py` |
| **Evolution** | Coevolution with racing, Hall of Fame, novelty search, niche elites and immigrants. | `ultisim/evolve.py`, `scripts/run_offense.py` |
| **Structure classifier** | Labels offenses vertical / horizontal / side / iso / spread from their formation. | `ultisim/classify.py` |
| **Viewer and dashboard** | Browser replay viewer and live evolution dashboard (no dependencies). | `viz/` |
| **Calibration and checks** | Club-level target bands, C vs Python parity, benchmarks, exploit audits. | `scripts/` |

Documentation:

- [docs/SIMULATION.md](docs/SIMULATION.md): the game model. Field, players, movement, throws, the mark, what causes a D, defensive schemes, and calibration targets and results.
- [docs/EVOLUTION.md](docs/EVOLUTION.md): full gene reference, the evolutionary algorithm, run outputs, the dashboard, and the exploit history.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): code layout, the two engines, the C data contract (generated from Python), determinism and performance.
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): tests, parity checks, and how to add a new ability or gene end to end.
- [docs/SETUP.md](docs/SETUP.md): installation on Windows, macOS and Linux.
- [RUNNING.md](RUNNING.md): long and overnight runs, resuming, and watching remotely.
- [docs/PLAN.md](docs/PLAN.md): the original project plan and a dated development log with every decision and measurement.

---

## How it works (short version)

1. **The game.**
   - 7v7 on a WFDF field (100 × 37 m, 18 m end zones).
   - Every player has randomly drawn stats: speed, acceleration, agility, reaction time, height, hands, throwing range and accuracy, break skill, marking.
   - Defenders react to cuts after their reaction time.
   - Throws (backhand, forehand, hammer, inside-out/outside-in curves, hucks, flat or lofted) fly along curved paths with a height profile and distance-dependent error.
   - A defender can only touch the disc if it's inside their reach envelope (standing reach, plus vertical when jumping).
   - A D happens when the marker blocks the release, a defender reaches the flight path first, a contested catch is lost, a layout bid succeeds, a pass is dropped or thrown away, or the stall count reaches 10.
2. **The thrower decides like a player.** Every 0.4 s it scores about 30–100 candidate throws, each a receiver × lead × throw type × curve × flat or lofted. Each candidate gets a completion estimate from the learned model and a value for the field position it reaches. The thrower throws if that beats holding the disc, or waits.
3. **Strategies are genomes.**
   - An offense is 6 formation slots: home spot, handler role, two cut options, timing, jab, seek-space. Plus team genes: reset timing, risk appetite, huck/break/hammer bias, patience, fakes, give-and-go.
   - A defense is its scheme (person / flex / zone), force, mark distance and angle, positioning, poaching, switching, help, stack sag, zone spots, a zone-to-person transition and a sideline trap.
4. **Evolution.** The offense and defense populations coevolve. Each generation, every offense plays current defenses, a Hall of Fame of past champion defenses, and the hand-built defenses.
   - **Fitness:** score rate per possession (or 1 − conceded for defenses).
   - **Selection:** score rank plus a **novelty** bonus (how different the strategy's behaviour is from everything seen before), so the search keeps exploring instead of converging on one strategy.

---

## Findings so far

The evolution runs so far were **short (100 generations, about 10 minutes each)**. They are best read as tests of the machinery.

**Exploits found and fixed.** Each was found by looking at what evolution converged on. This is the most important practical lesson: evolution finds every loophole in the simulator.

| What evolution found | Why it worked in the simulator | Fix |
|---|---|---|
| An extremely wide mark wedge took over every defense | A wider wedge had no downside | Coverage now comes from the marker's body: wider when closer, but a tight mark reacts later |
| Hammers on up to 27% of throws (hand-built zone offense) | Hammers went over the mark and couldn't be touched mid-flight, with no downside | Catch penalty and more error for hammers |
| Evolved offenses broke the mark on 45–92% of throws at 90–97% completion | The mark only covered a narrow cone, and curved "around" breaks escaped it | Break-side coverage, curves judged along their whole path, more error on curves |
| `flex_1z6p` defenses (a dedicated marker plus six person defenders) conceding 3–4% | Not the flex or switching: the defenders **fronted** (stood between attacker and disc). In flight they positioned against the live disc, sliding onto every throwing lane for free, and the thrower's read assumed they would just drift | Positional defenders read the disc in flight 0.45 s late; the thrower predicts the receiver's defender holding their spot; fronting reacts later (blind side); throw model refit on mutated and evolved strategies |

**Run 3 results** (before the break fix, so treat the offensive results with caution):

- **Diversity held.** With novelty search, every structure family survived all 100 generations. The champion family rotated: iso led 39 generations, vertical 24, side 16, spread 14, horizontal 6.
- **Defense evolved faster than offense.** The average offense score fell from 0.42 to 0.34, and the average defense's conceded rate fell from 0.43 to 0.28.
- **A non-textbook defense emerged.** It's person D forcing sideline with the marker sitting back at the 3 m limit. Defenders front their attackers (stand between them and the disc), give more cushion the further their attacker is from the disc, switch constantly and always bid. It held the hand-built offenses to 11–19% scoring; standard defenses allow 30–55%. This result does not depend on the break exploit.
- **The evolved offense** was a 3–4-handler pod near the disc, one mid cutter about 20 m deep and one deep cutter about 35 m deep, frequent fakes and seek-space cuts. Most of its edge came from the break exploit, which is now fixed.

**Run 4** showed the next exploit. Defenses converged on `flex_1z6p` and, in the audit, conceded only 3–4%. After the fix, the same champion concedes about 12% to the hand-built offenses (hand-built defenses: 33–49%). A short offense-only evolution against it reached 33% in 30 generations by throwing more aggressively and deeper. So fronting is now a strong defense with a counter, not an unbeatable one. Runs 1–4 predate the fix; rerun before trusting their defensive results. See [docs/EVOLUTION.md](docs/EVOLUTION.md#7-results-so-far).

**Calibrated baseline** (hand-built strategies, current simulator with disc heights and lofted throws): completion 0.93, 0.40 score rate per possession from the goal line, 0.55 hold rate, 9.6 throws per score, breaks on about 10% of throws against person D at 0.82 completion, about a quarter of throws lofted. Scoring and hold rate sit slightly under their target bands. Known deviations from real club play: hucks are rarer but easier, turnovers lean toward drops over throwaways, and there are no stall-outs. Details are in [docs/SIMULATION.md](docs/SIMULATION.md).

---

## Repository layout

```
configs/      default.toml (every physics/rules constant, commented) and throw_model.json (learned throw model)
csim/         C engine: sim.c (game), core.c (ABI info), ultisim.h (API); us_structs.h is generated at build time
ultisim/      Python package: reference engine, C bindings, genomes, evolution, classifier, replays, rosters
scripts/      build_c, bench, calibrate, fit_throw_model, parity, run_offense, audit_defenses, demo
viz/          server.py, replay viewer (index.html/viewer.js), dashboard (dashboard.html/js/css), file formats
tests/        pytest suite: rules, physics, exact C-vs-Python parity, C engine behaviour
docs/         documentation and the development log
runs/         run outputs (git-ignored)
```

## Contributing

Start with [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). The golden rules:

1. **New mechanics go into the Python reference first, then C.** Prove they match with `scripts/parity.py`.
2. **After any physics change:** refit the throw model, rerun calibration, and check the numbers against the target bands.
3. **When evolution converges fast on something surprising:** audit it before believing it.
