# Long and overnight runs

Install first: [docs/SETUP.md](docs/SETUP.md). Commands use `.venv/bin/python`; on Windows use `.venv\Scripts\python`.

## 1. Measure your machine
```bash
.venv/bin/python scripts/bench.py -n 8000
```

Seconds per generation ≈ games per generation ÷ bench rate, where games per generation = `off_pop × stage1 + 0.34 × off_pop × (stage2 − stage1)`.

| settings | games per generation | at 1,000 possessions/s | at 4,000 possessions/s |
|---|---|---|---|
| defaults: 48/32 pop, 96/288 | ~7,700 | ~8 s | ~2 s |
| overnight: 96/64 pop, 128/384 | ~20,700 | ~21 s | ~5 s |

Possessions get longer as defenses improve, so expect generations to slow down somewhat over a run.

## 2. Smoke test (1 minute)
```bash
.venv/bin/python scripts/run_offense.py --run smoke --gens 3 --off-pop 16 --def-pop 12 --stage1 48 --stage2 96
```

## 3. Start the run
macOS (`caffeinate` keeps the Mac awake; `nohup ... &` survives closing the terminal):
```bash
mkdir -p runs && caffeinate -dimsu nohup .venv/bin/python scripts/run_offense.py --run overnight1 \
    --gens 600 --off-pop 96 --def-pop 64 --stage1 128 --stage2 384 > runs/overnight1.log 2>&1 &
```

Linux: the same command without `caffeinate -dimsu`.

Windows (PowerShell; keep the window open and stop Windows from sleeping):
```powershell
.venv\Scripts\python scripts\run_offense.py --run overnight1 --gens 600 --off-pop 96 --def-pop 64 --stage1 128 --stage2 384
```

- **Stop:** Ctrl+C, or `kill %1` if it's running in the background. A checkpoint is saved after every generation.
- **Resume:** run the same command again with `--resume`.

## 4. Watch it
```bash
.venv/bin/python viz/server.py        # http://127.0.0.1:8765/dashboard.html?run=overnight1
```

The server binds to localhost only. To watch from another computer, tunnel it:
```bash
ssh -L 8765:127.0.0.1:8765 you@the-run-machine.local
```
Then open the same URL locally.

## 5. Sanity checks while it runs
- **`runs/<run>.log`** gets one line per generation. If it stops growing, check `runs/<run>/status.json` for an error.
- **One family suddenly wins everything by a wide margin** (e.g. defenses conceding under 10–15%): audit before believing it.
  ```bash
  .venv/bin/python scripts/audit_defenses.py --run overnight1 -n 400
  ```
  Then watch the champion replay. See the exploit history in [docs/EVOLUTION.md](docs/EVOLUTION.md#6-exploit-audits-read-this-before-believing-a-result).

## 6. Sharing results
A run's folder `runs/<run>/` is self-contained: settings, history, archive, replays and the checkpoint. Copy it to another machine's `runs/` to open it in the dashboard or analyse it there.
