# Evolution run output format v1

Written by `ultisim/evolve.py` into `runs/<run_name>/`, and read by `viz/dashboard.html` (served by `viz/server.py`).
Every file is written atomically (temp file + rename). The one exception is `history.jsonl`, which only ever has whole lines appended.

```
runs/<run_name>/
  meta.json            run settings, written once at start (rewritten on resume)
  history.jsonl        one JSON object per completed generation (append-only)
  status.json          live status, rewritten every few seconds while running
  latest_best.json     replay of the current champion (viz/REPLAY_FORMAT.md), refreshed every few generations
  archive.json         best offense found per structure family (rewritten each generation)
  state.pkl            checkpoint for --resume (binary, the dashboard ignores it)
```

## meta.json
```jsonc
{
  "run": "pilot1",
  "created": "2026-11-05T21:00:00",
  "kind": "offense",                  // Phase 4 = "offense" (Run A); Phase 5 = "defense" (Run B)
  "generations_planned": 20,
  "offense_pop": 48, "defense_pop": 32,
  "possessions_per_offense": 240,
  "families": ["vertical", "horizontal", "side", "iso", "spread", "other"]   // fixed order, use for colours
}
```

## history.jsonl (one line per generation)
```jsonc
{
  "gen": 3,                            // 0-based
  "elapsed_s": 81.2,                   // since the run started (including resumed time)
  "gen_s": 24.9,                       // wall time of this generation
  "possessions": 46080,                // cumulative possessions simulated
  "off": {"best": 0.58, "mean": 0.44, "median": 0.45, "worst": 0.12},   // offense score rate per possession
  "def": {"best": 0.31, "mean": 0.47},                                   // score rate CONCEDED (lower = better defense)
  "families": {"vertical": 12, "horizontal": 20, "side": 6, "iso": 3, "spread": 5, "other": 2},  // counts in the offense population
  "family_best": {"vertical": 0.55, "horizontal": 0.58, "side": 0.49, "iso": 0.47, "spread": 0.40, "other": null},
                                       // best score rate ever seen per family (archive), null if never seen
  "champion": {
    "family": "horizontal", "score_rate": 0.58,
    "descriptors": {"n_handlers": 2, "depth": 16.1, "spread_x": 1.2, "spread_y": 9.8, "lateral": 0.3},
    "genes": {"risk_aversion": 0.8, "patience": 1.1, "huck_bias": 0.05, "lead": 1.9, "give_go": 0.4}   // a few readable genes
  },
  "def_champion": {"scheme": "person", "force": "forehand", "conceded": 0.31,
                   "genes": {"mark_dist": 1.4, "angle_near": 70.0, "angle_far": 150.0, "poach": 0.1}},
  "def_schemes": {"person_forehand": 10, "person_backhand": 6, "person_straight_up": 2, "person_middle": 5,
                  "person_sideline": 1, "zone": 8},  // counts in the defense population
  "stats": {"completion": 0.93, "huck_share": 0.05, "break_share": 0.17, "throws_per_poss": 8.1}   // across all games this generation
}
```

## status.json
```jsonc
{"run": "pilot1", "state": "running",       // "running" | "finished" | "stopped" | "error"
 "gen": 4, "generations_planned": 20, "phase": "evaluating",   // free text: "evaluating", "breeding", "rendering replay", ...
 "updated": "2026-11-05T21:03:12", "elapsed_s": 101.5, "message": ""}
```

## archive.json
```jsonc
{"families": {
   "horizontal": {"score_rate": 0.58, "gen": 3, "descriptors": {...}, "genes": {...}},
   "vertical": {...}
 }}
```
