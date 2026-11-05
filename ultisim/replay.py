"""Record a game into the replay format described in viz/REPLAY_FORMAT.md."""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, cfg, title: str, mode: str, seed: int, team_names, team_strategies, roster14):
        self.L = cfg.field.length
        self.W = cfg.field.width
        self.data = {
            "format": "ultisim-replay",
            "version": 1,
            "meta": {
                "title": title, "mode": mode, "seed": int(seed), "dt": cfg.sim.dt,
                "teams": [{"name": n, "strategy": s} for n, s in zip(team_names, team_strategies)],
                "created": _dt.datetime.now().isoformat(timespec="seconds"),
            },
            "field": {"length": self.L, "width": self.W, "endzone": cfg.field.endzone_depth},
            "players": [{"id": i, "team": i // 7, "num": i % 7 + 1, "stats": roster14.player(i)} for i in range(14)],
            "frames": [], "throws": [], "events": [], "possessions": [], "result": None,
        }

    # coordinate conversion: engine works in the offense frame
    def pt(self, p, flipped: bool):
        p = np.asarray(p, dtype=float)
        if flipped:
            p = np.stack([self.L - p[..., 0], self.W - p[..., 1]], axis=-1)
        return p

    def frame(self, pos, states, disc_xyz, disc_state, holder, offense, stall, mark, flipped):
        P = self.pt(pos, flipped)
        d = self.pt(disc_xyz[:2], flipped)
        m = None
        if mark is not None:
            a, w = mark
            m = [round(float(a + 180.0 if flipped else a), 1), round(float(w), 1)]
            m[0] = round((m[0] + 180.0) % 360.0 - 180.0, 1)
        self.data["frames"].append({
            "p": [round(float(v), 2) for v in P.reshape(-1)],
            "st": [int(s) for s in states],
            "d": [round(float(d[0]), 2), round(float(d[1]), 2), round(float(disc_xyz[2]), 2)],
            "ds": int(disc_state), "h": int(holder), "o": int(offense),
            "s": None if stall is None else round(float(stall), 1), "m": m,
        })

    def throw(self, t0, t1, thrower, receiver, ttype, huck, is_break, curve, p0, c, p1, flipped) -> int:
        path = [[round(float(v), 2) for v in self.pt(q, flipped)] for q in (p0, c, p1)]
        self.data["throws"].append({"t0": round(t0, 2), "t1": round(t1, 2), "thrower": int(thrower),
                                    "receiver": int(receiver), "type": ttype, "huck": bool(huck),
                                    "break": bool(is_break), "curve": int(curve), "path": path, "outcome": None})
        return len(self.data["throws"]) - 1

    def throw_outcome(self, idx: int, outcome: str, t1: float | None = None):
        self.data["throws"][idx]["outcome"] = outcome
        if t1 is not None:
            self.data["throws"][idx]["t1"] = round(t1, 2)

    def event(self, t, etype, player, text):
        self.data["events"].append({"t": round(t, 2), "type": etype, "player": int(player), "text": text})

    def possession(self, t0, t1, offense, force, outcome):
        self.data["possessions"].append({"t0": round(t0, 2), "t1": round(t1, 2), "offense": int(offense),
                                         "force": force, "outcome": outcome})

    def finish(self, result: dict):
        self.data["result"] = result

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, separators=(",", ":"))   # "format" stays the first key
        for i in range(40):  # atomic: the live viewer never sees a half-written file
            try:
                os.replace(tmp, path)
                break
            except PermissionError:  # Windows: a reader has the old file open; retry briefly
                if i == 39:
                    raise
                time.sleep(0.05)
        return path
