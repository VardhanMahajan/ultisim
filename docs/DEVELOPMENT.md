# Development guide

## Everyday commands
Commands are shown with `.venv/bin/python`; on Windows use `.venv\Scripts\python`.

```bash
.venv/bin/python scripts/build_c.py                 # rebuild the C engine (also regenerates csim/us_structs.h)
.venv/bin/python -m pytest -q                        # full suite, about 1-2 min
.venv/bin/python scripts/bench.py -n 4000            # throughput
.venv/bin/python scripts/calibrate.py --engine c -n 400 --points 1600     # calibration on C (~30 s)
.venv/bin/python scripts/calibrate.py -n 24          # calibration on the Python reference, with per-throw diagnostics (minutes)
.venv/bin/python scripts/parity.py --n-py 40 --n-c 2000                   # C vs Python, same rosters (~4 min)
.venv/bin/python scripts/parity.py --defs person_forehand watch4:26       # ... including an evolved defense (<run>:<index>)
.venv/bin/python scripts/fit_throw_model.py --rounds 3 -n 24 --eps 0.25   # refit the throw model (~15 min on 12 cores)
.venv/bin/python scripts/demo.py --replays-only      # example replays into runs/demo/
```

Any config value can be overridden for a quick experiment, e.g. `scripts/calibrate.py --engine c --set defense.mark_break_tail=0.8`. Genome fields can be overridden for every hand-built strategy with `--og` / `--dg`.

## Tests
| file | covers |
|---|---|
| `tests/test_engine.py` | config, rosters, genomes; movement and time-to-reach; flight paths; the mark (including the "wider wedge adds no coverage" regression); possession rules for every hand-built matchup; double team; full points; replay format |
| `tests/test_c_parity_exact.py` | struct layouts; exact C vs Python for every deterministic building block |
| `tests/test_engine_c.py` | valid results; determinism independent of thread count; full points; genome changes affect outcomes |
| `tests/test_c_core.py` | ABI and the threading smoke test |

## Adding a new ability (gene) end to end
Example: a new defense gene `deny_deep` (worked examples in the git history: `switch`, `help_cut`, `trap`, `seek_space`).

1. **Genome.** Add the field (with a default that keeps today's behaviour) to `DefenseGenome` in `ultisim/genome.py`, and append `("deny_deep", 1, lo, hi)` to the **end** of `DEFENSE_SCHEMA`. Old checkpoints won't load after a schema change, so start a new run.
2. **Python reference.** Implement the behaviour in `ultisim/engine_ref.py`. Keep it readable; this is the spec.
3. **Config constants** (if any). Add them to `configs/default.toml` with a comment and to `CONFIG_FIELDS` in `ultisim/cabi.py`, then **bump `ABI_VERSION`**.
4. **C engine.** Port the same logic to `csim/sim.c`, mirroring structure, order of operations and RNG use (draw counts don't need to match, distributions do). Rebuild.
5. **Tests.** Run `pytest`. Add a test if the ability has a crisp invariant.
6. **Parity.** Run `scripts/parity.py` on a seed defense that uses the gene. Add such a seed to `CANONICAL_DEFENSES` if useful; the parametrized tests then cover it automatically.
7. **Physics changed?** If the change affects what throwers face, refit the throw model, then rerun the calibration and compare with the target bands.
8. **Dashboard and log.** Add the gene to `DEF_GENES_SHOWN` / `OFF_GENES_SHOWN` in `ultisim/evolve.py` if it should appear on the champion cards. Note the change in [PLAN.md](PLAN.md).

## Rules of thumb
- **Python first, C second, parity third.** Never change behaviour only in C.
- **Every gene needs a cost.** If a gene has no physical downside, evolution will max it out (see the exploit table in [EVOLUTION.md](EVOLUTION.md#6-exploit-audits-read-this-before-believing-a-result)).
- **Calibrate with hand-built strategies, audit with evolved ones.** Calibration can't find exploits that hand-built strategies never use. Short evolution runs plus `scripts/audit_defenses.py` do.
- **Windows:** the C library can't be rebuilt while a run has it loaded. Stop runs before `build_c.py`.
- **Encoding:** read and write text files as UTF-8 explicitly in scripts. Windows' default code page silently corrupts en dashes and similar characters.
