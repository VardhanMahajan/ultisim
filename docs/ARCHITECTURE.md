# Architecture

## The big picture
```
configs/default.toml ──> ultisim/config.py ──┬──> ultisim/engine_ref.py   (Python reference engine: rules, prototyping)
configs/throw_model.json ────────────────────┤
                                             └──> ultisim/engine.py ──ctypes──> build/ultisim_core.(dll|so)  (C engine)
ultisim/genome.py (schemas, seeds) ──────────────┘           ▲
ultisim/cabi.py  (struct field lists) ──generates──> csim/us_structs.h

ultisim/evolve.py ──> CEngine.run(offenses, defenses, specs) ──> results ──> fitness, novelty ──> next generation
                 └──> engine_ref (background thread) ──> runs/<run>/latest_best.json ──> viz/ viewer + dashboard
```

## Two engines, one game
| | Python reference (`engine_ref.py`) | C engine (`csim/sim.c`) |
|---|---|---|
| Purpose | Readable source of truth; prototyping new mechanics; replays; per-throw logs | Speed for evolution, calibration and benchmarks |
| Speed | ~1–4 possessions/s per process | ~200–300 possessions/s per core; batches run on all cores |
| Randomness | numpy `default_rng(seed)` | xoshiro256** seeded per game |
| Outputs | `PossessionResult`, replays, `throw_log`, `decision_log` | `UsResult` counters per game (outcome, throws, hucks, breaks, hammers, fakes, stall, ...) |

The C engine is a **function-by-function port**: same function names, same order of operations, same formulas. Because the random-number generators differ, individual games differ. Parity is checked in two layers:

1. **Exact.** Deterministic building blocks are exported from C and compared with Python to 1e-9: time-to-reach, flight path, throw error, mark block probability, the movement step, and the matching algorithm. See `tests/test_c_parity_exact.py`.
   It also covers the disc height profile (`us_flight_height`).
2. **Statistical.** `scripts/parity.py` plays the same rosters in both engines and compares score rate, completion, throws, hucks, breaks, durations and turnover mix. It flags anything more than 3 standard errors apart.

Some deliberate, documented approximations in C, all negligible:
- defenders who provably can't reach the disc within 2 s are skipped
- sigmoids below 6e-8 are skipped
- a fast `exp`.

## The C data contract
`ultisim/cabi.py` is the **single source of truth** for every struct crossing the Python/C boundary:
- `UsConfig`: every config constant plus the learned-model weights, all doubles
- `UsPlayer`: 15 stats
- `UsOffense` / `UsDefense`: the genome schemas, so the GA's flat vectors are the structs byte for byte
- `UsGameSpec`: 14 players, genome indices per team, mode, seed, start position
- `UsResult`.

`scripts/build_c.py` generates `csim/us_structs.h` from those lists before compiling. At load time Python checks the ABI version and every struct size. A genome field that's a C keyword gets a trailing underscore (`switch` becomes `switch_`).

**Adding a config constant or gene** therefore means: add it to the TOML or the genome schema, add it to `CONFIG_FIELDS` if it's a config value, bump `ABI_VERSION`, use it in both engines, and rebuild. See [DEVELOPMENT.md](DEVELOPMENT.md).

## Batches and threads
`CEngine.run(offenses, defenses, specs)` packs everything into numpy structured arrays and calls `us_sim_batch` once:
- **With OpenMP:** the C loop runs `schedule(dynamic)` over games.
- **Without OpenMP** (e.g. Apple clang without `libomp`): Python splits the batch into chunks and calls C from a thread pool. ctypes releases the GIL during the call, so all cores are used.

Results are identical either way and independent of the thread count (tested).

## Determinism
- Same seed + same inputs = same game, in both engines.
- An evolution run is deterministic given `--seed`, apart from champion-replay rendering, which uses separate seeds.
- A checkpoint saves the RNG state, so `--resume` continues the same stream.

## Performance notes
- Throw evaluation is about 80% of the cost. Each decision scores up to 84 candidate throws × 7 defenders × 10 path points.
- The defender kernel is a branch-free loop that GCC vectorizes with AVX2.
- An 8-lane float version was tried and was *slower*, because it gives up skipping far-away defenders.
- Biggest levers if you need more speed:
  1. more cores (scales linearly)
  2. a cheaper learned throw model
  3. GPU batching.
- `scripts/bench.py` is the throughput reference. Run it after engine changes.

## The learned throw model
`engine_ref.throw_features` and the matching block in `sim.c` compute 23 features per candidate throw. `configs/throw_model.json` stores the logistic-regression weights and the feature names, and the names are checked at load time. `scripts/fit_throw_model.py` fits the model from Python-reference self-play with exploration (throwers sometimes pick a random top-12 option, so the data covers more than the throws the model already likes). Each round plays every hand-built matchup plus 80 random pairs of **heavily mutated hand-built strategies and evolved strategies from `runs/*/state.pkl`** (`--no-runs` skips those). A model fitted only on textbook defenses is badly over-confident against the odd defenses evolution invents. Refit it after any change to the physics.

## Visualization
`viz/server.py` is a stdlib HTTP server bound to 127.0.0.1. It serves:
- `viz/`
- `runs/`
- `/api/replays` and `/api/runs`.

The viewer and dashboard are plain HTML/JS/SVG with no build step. File formats are in `viz/REPLAY_FORMAT.md` and `viz/RUN_FORMAT.md`. Writers replace files atomically, with retries on Windows, where an open reader can block a replace.
