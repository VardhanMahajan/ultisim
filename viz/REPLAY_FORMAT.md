# Replay format v1 (`ultisim-replay`)

One JSON file per replay. It's written by `ultisim/replay.py` and read by `viz/viewer.js`.
All coordinates are **absolute field coordinates in metres**:
- x runs along the field, 0 to 100. y runs across it, 0 to 37.
- End zones are x in [0, 18] and [82, 100].
- **Team 0 attacks toward x = 100. Team 1 attacks toward x = 0.**

Draw with x horizontal (left to right) and y vertical, with y = 0 at the **bottom** of the canvas. Seen that way, +y is the thrower's left when facing +x.

```jsonc
{
  "format": "ultisim-replay",
  "version": 1,
  "meta": {
    "title": "Vertical stack vs Person D (force forehand)",
    "mode": "possession",          // or "point" (turnovers flip possession)
    "seed": 1234,
    "dt": 0.1,                      // seconds between frames
    "teams": [
      {"name": "Vertical stack", "strategy": "vertical"},
      {"name": "Person FH", "strategy": "person_fh"}
    ],
    "created": "2026-11-05T17:00:00"
  },
  "field": {"length": 100.0, "width": 37.0, "endzone": 18.0},

  // 14 players: ids 0-6 are team 0, ids 7-13 are team 1.
  "players": [
    {"id": 0, "team": 0, "num": 1,
     "stats": {"top_speed": 7.9, "accel": 4.1, "agility": 0.6, "reaction_time": 0.25,
               "height": 1.83, "vertical": 0.62, "hands": 0.98, "throw_skill": 0.7,
               "break_skill": 0.4, "marking": 0.5}}
    // ... (stat keys may grow; show whatever is present)
  ],

  // One entry per frame (frame i is at time i * dt).
  "frames": [
    {
      "p": [x0, y0, x1, y1, "...", x13, y13], // 28 numbers, 2 decimals
      "st": [0, 1, 0, "..."],                 // 14 state codes, see below
      "d": [x, y, z],                          // disc position; z = height in m (approx)
      "ds": 0,                                 // disc state: 0 = held, 1 = in air, 2 = on ground / dead
      "h": 3,                                  // id of the player holding it, or -1
      "o": 0,                                  // team currently on offense
      "s": 4.2,                                // stall count (float, 0-10), or null when not counting
      "m": [a, w]                              // marker coverage: centre angle a (deg, 0 = +x, counter-clockwise)
                                               //   and half-width w (deg) around the thrower; null if no set mark
    }
  ],

  // Every throw, for drawing the flight arc while the disc is in the air.
  "throws": [
    {"t0": 3.4, "t1": 5.1, "thrower": 2, "receiver": 5,
     "type": "backhand",               // backhand | forehand | hammer
     "huck": false, "break": true, "curve": -1,   // curve: -1, 0, +1
     "path": [[x0, y0], [cx, cy], [x1, y1]],      // quadratic Bezier: release, control, landing
     "outcome": "catch"}               // catch | score | block | interception | drop | throwaway | out_of_bounds
  ],

  // Discrete events, for the timeline / ticker.
  "events": [
    {"t": 5.1, "type": "catch", "player": 5, "text": "#6 catches (backhand, 18 m, break)"}
    // types: pull | catch | throw | score | block | interception | drop | throwaway |
    //        out_of_bounds | stall | mark_block | layout | turnover | timeout | fake
  ],

  // Possession segments (one in "possession" mode, several in "point" mode).
  "possessions": [
    {"t0": 0.0, "t1": 22.3, "offense": 0, "force": "forehand", "outcome": "score"}
  ],

  "result": {"outcome": "score", "team": 0, "duration": 22.3, "throws": 7, "turnovers": 0}
}
```

## Player state codes (`st`)
| code | meaning |
|---|---|
| 0 | set / standing in formation |
| 1 | cutting (sprinting) |
| 2 | clearing / jogging back |
| 3 | chasing the disc in the air |
| 4 | on the ground after a layout bid |
| 5 | marking the thrower |
| 6 | holding the disc |
| 7 | defending (person D tracking) |
| 8 | zone position |
