#!/usr/bin/env python3
"""Write a synthetic (but plausible) replay to runs/_sample/sample.json.

It follows viz/REPLAY_FORMAT.md (v1) so the viewer can be tested before the simulator
exists. This is a scripted choreography, not a simulation: cutters follow scripted cut
paths, everyone else steers toward simple targets (stack slots, person-D positions, the
mark), and the disc follows each throw's quadratic Bezier.

Sequence (team 0, vertical stack, attacking +x; team 1 person D forcing forehand):
  1. open-side forehand to an out cut
  2. around-break backhand (stall gets to ~4 first)
  3. open-side forehand huck, defender lays out and misses
  4. pump fake, then a short forehand into the end zone: score

    python viz/make_sample.py [--out runs/_sample/sample.json] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime

DT = 0.1
L, W, EZ = 100.0, 37.0, 18.0
N_PER_TEAM = 7
MARK_ANGLE = 55.0      # deg: marker sits on the thrower's backhand side, upfield (force forehand)
MARK_HALF = 45.0       # deg half-width of the blocked wedge
MARK_DIST = 1.6        # m from thrower
STALL_RANGE = 3.0      # count only when the marker is within this

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_players(rng: random.Random):
    nums = [[7, 12, 3, 21, 9, 44, 18], [2, 5, 11, 14, 23, 31, 88]]
    players = []
    for pid in range(2 * N_PER_TEAM):
        team = pid // N_PER_TEAM
        stats = {
            "top_speed": round(rng.uniform(6.8, 8.6), 2),
            "accel": round(rng.uniform(3.4, 4.8), 2),
            "agility": round(rng.uniform(0.35, 0.85), 2),
            "reaction_time": round(rng.uniform(0.18, 0.32), 2),
            "height": round(rng.uniform(1.65, 1.95), 2),
            "vertical": round(rng.uniform(0.45, 0.80), 2),
            "hands": round(rng.uniform(0.90, 0.99), 3),
            "throw_skill": round(rng.uniform(0.40, 0.90), 2),
            "break_skill": round(rng.uniform(0.20, 0.80), 2),
            "marking": round(rng.uniform(0.30, 0.80), 2),
        }
        players.append({"id": pid, "team": team, "num": nums[team][pid % N_PER_TEAM], "stats": stats})
    return players


def bezier(path, u):
    (x0, y0), (cx, cy), (x1, y1) = path
    a, b, c = (1 - u) ** 2, 2 * (1 - u) * u, u * u
    return a * x0 + b * cx + c * x1, a * y0 + b * cy + c * y1


def cut_ease(u):
    """0->1, starts at rest, ends at full (average) speed, peak 4/3 of average."""
    return 2 * u * u - u ** 3


class Body:
    def __init__(self, x, y, vmax, accel):
        self.x, self.y = x, y
        self.vx = self.vy = 0.0
        self.vmax, self.accel = vmax, accel

    def steer(self, tx, ty, vmax=None, gain=1.6):
        vmax = self.vmax if vmax is None else vmax
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        sp = min(vmax, dist * gain)
        dvx = (dx / dist * sp if dist > 1e-6 else 0.0) - self.vx
        dvy = (dy / dist * sp if dist > 1e-6 else 0.0) - self.vy
        dv = math.hypot(dvx, dvy)
        lim = self.accel * 1.5 * DT
        if dv > lim:
            dvx, dvy = dvx / dv * lim, dvy / dv * lim
        self.vx += dvx
        self.vy += dvy
        self.x += self.vx * DT
        self.y += self.vy * DT

    def speed(self):
        return math.hypot(self.vx, self.vy)


def build(seed: int):
    rng = random.Random(seed)
    players = make_players(rng)
    num = {p["id"]: p["num"] for p in players}

    # ---- script --------------------------------------------------------------------
    # Ticks (frame index) for each throw: release, flight length, thrower, receiver,
    # catch spot, cut start tick, and throw attributes.
    script = [
        dict(rel=30, fl=9, thrower=0, recv=2, spot=(29.5, 7.0), cut=12,
             type="forehand", huck=False, brk=False, curve=1, outcome="catch", peak=1.2),
        dict(rel=92, fl=12, thrower=2, recv=3, spot=(36.0, 27.0), cut=84,
             type="backhand", huck=False, brk=True, curve=1, outcome="catch", peak=1.8),
        dict(rel=116, fl=30, thrower=3, recv=6, spot=(79.0, 11.0), cut=92,
             type="forehand", huck=True, brk=False, curve=-1, outcome="catch", peak=6.5),
        dict(rel=182, fl=8, thrower=6, recv=4, spot=(87.0, 9.0), cut=160,
             type="forehand", huck=False, brk=False, curve=1, outcome="score", peak=1.0),
    ]
    layout = dict(player=13, start=143, end=157)   # defender of #6 bids on the huck
    fake_tick = 174
    score_tick = script[-1]["rel"] + script[-1]["fl"]
    n_frames = score_tick + 10 + 1

    # ---- initial positions --------------------------------------------------------------
    start = {0: (18.0, 12.0), 1: (14.5, 21.0), 2: (30.0, 18.5), 3: (33.5, 18.5),
             4: (37.0, 18.5), 5: (40.5, 18.5), 6: (44.0, 18.5)}
    bodies = {}
    for p in players:
        s = p["stats"]
        if p["team"] == 0:
            x, y = start[p["id"]]
        else:
            ox, oy = start[p["id"] - N_PER_TEAM]
            x, y = ox + 2.0, oy + 1.0
        bodies[p["id"]] = Body(x, y, s["top_speed"], s["accel"])

    holder = 0
    disc_state = 0
    active = None            # throw dict while disc is in the air
    cut_from = {}            # receiver -> (x, y) at cut start
    decel = {}               # receiver -> ticks of post-catch drift remaining
    handler_tgt = {}         # handler id -> last reset target
    stall = None
    throws_out, events = [], []
    frames = []

    def ev(tick, typ, player, text):
        events.append({"t": round(tick * DT, 2), "type": typ, "player": player, "text": text})

    for tick in range(n_frames):
        t = tick * DT
        disc_z = 1.0
        hb = bodies[holder] if holder >= 0 else None

        # ---- throw release ----------------------------------------------------------
        for th in script:
            if tick == th["rel"]:
                thrower = bodies[th["thrower"]]
                side = -1.0 if th["type"] == "forehand" else 1.0   # forehand released on the right (-y)
                rx, ry = thrower.x + 0.25, thrower.y + 0.7 * side
                ex, ey = th["spot"]
                dx, dy = ex - rx, ey - ry
                dist = math.hypot(dx, dy)
                nx, ny = -dy / dist, dx / dist                     # left normal
                bend = th["curve"] * min(0.14 * dist, 5.0)
                cx, cy = (rx + ex) / 2 + nx * bend, (ry + ey) / 2 + ny * bend
                th["path"] = [[round(rx, 2), round(ry, 2)], [round(cx, 2), round(cy, 2)], [ex, ey]]
                th["dist"] = dist
                active = th
                disc_state, holder, stall = 1, -1, None
                tags = ", ".join(x for x in (("huck" if th["huck"] else ""), ("break" if th["brk"] else "")) if x)
                ev(tick, "throw", th["thrower"],
                   f"#{num[th['thrower']]} throws a {th['type']} to #{num[th['recv']]}"
                   + (f" ({tags})" if tags else ""))
                throws_out.append({
                    "t0": round(t, 2), "t1": round((th["rel"] + th["fl"]) * DT, 2),
                    "thrower": th["thrower"], "receiver": th["recv"], "type": th["type"],
                    "huck": th["huck"], "break": th["brk"], "curve": th["curve"],
                    "path": th["path"], "outcome": th["outcome"],
                })
            if tick == th["cut"]:
                b = bodies[th["recv"]]
                cut_from[th["recv"]] = (b.x, b.y, tick)

        # ---- catch ------------------------------------------------------------------
        if active is not None and tick == active["rel"] + active["fl"]:
            th = active
            holder, disc_state, active = th["recv"], 0, None
            b = bodies[holder]
            ex, ey = th["spot"]
            T = (th["rel"] + th["fl"] - cut_from[holder][2]) * DT
            sx, sy, _ = cut_from.pop(holder)
            b.vx, b.vy = (ex - sx) / T, (ey - sy) / T
            b.x, b.y = ex, ey
            decel[holder] = 6
            desc = f"{th['type']}, {th['dist']:.0f} m" + (", huck" if th["huck"] else "") + (", break" if th["brk"] else "")
            if th["outcome"] == "score":
                ev(tick, "catch", holder, f"#{num[holder]} catches in the end zone ({desc})")
                ev(tick, "score", holder, f"SCORE: #{num[th['thrower']]} to #{num[holder]}")
            else:
                ev(tick, "catch", holder, f"#{num[holder]} catches ({desc})")

        if tick == layout["start"]:
            ev(tick, "layout", layout["player"],
               f"#{num[layout['player']]} lays out for the block, just misses")
        if tick == fake_tick:
            ev(tick, "fake", holder, f"#{num[holder]} pump fakes the around")

        # ---- move players -------------------------------------------------------------
        st = [0] * (2 * N_PER_TEAM)
        hb = bodies[holder] if holder >= 0 else None
        disc_x = hb.x if hb else active["spot"][0]
        disc_xy = (hb.x, hb.y) if hb else None
        after_score = tick >= score_tick

        # offense
        stack_start = min(disc_x + 10.0, 86.0)
        spacing = min(3.5, max(1.5, (97.0 - stack_start) / 4))
        stack_ids = [i for i in range(2, 7) if i != holder and i not in cut_from]
        for pid in range(N_PER_TEAM):
            b = bodies[pid]
            if pid == holder:
                if pid in decel and decel[pid] > 0:
                    b.vx *= 0.55
                    b.vy *= 0.55
                    b.x += b.vx * DT
                    b.y += b.vy * DT
                    decel[pid] -= 1
                else:
                    b.vx = b.vy = 0.0
                st[pid] = 6
                continue
            if pid in cut_from:
                sx, sy, t0 = cut_from[pid]
                th = next(x for x in script if x["recv"] == pid and x["cut"] == t0)
                T = th["rel"] + th["fl"] - t0
                u = min(1.0, (tick - t0) / T)
                e = cut_ease(u)
                ex, ey = th["spot"]
                nx, ny = sx + (ex - sx) * e, sy + (ey - sy) * e
                b.vx, b.vy = (nx - b.x) / DT, (ny - b.y) / DT
                b.x, b.y = nx, ny
                st[pid] = 3 if (active is th) else 1
                continue
            if after_score:
                b.steer(b.x, b.y)
                st[pid] = 0
                continue
            if pid in (0, 1):   # handlers: reset spots behind the disc on either side
                if hb is not None:
                    if pid == 1:
                        handler_tgt[pid] = (hb.x - 4.0, min(W - 3.0, hb.y + 7.0))
                    else:
                        handler_tgt[pid] = (hb.x - 2.5, max(3.0, hb.y - 7.0))
                tx, ty = handler_tgt.get(pid, (b.x, b.y))
                b.steer(tx, ty, vmax=5.0)
            else:
                k = stack_ids.index(pid) if pid in stack_ids else 0
                sx_, sy_ = stack_start + k * spacing, W / 2
                far = math.hypot(sx_ - b.x, sy_ - b.y) > 6.0
                b.steer(sx_, sy_, vmax=6.8 if (far or disc_state == 1) else 4.5)
            st[pid] = 2 if b.speed() > 2.5 else 0

        # defense (person D, defender id = offender id + 7)
        for pid in range(N_PER_TEAM, 2 * N_PER_TEAM):
            b = bodies[pid]
            mark_of = pid - N_PER_TEAM
            o = bodies[mark_of]
            if layout["start"] <= tick < layout["end"] and pid == layout["player"]:
                if tick == layout["start"]:
                    ang = math.atan2(11.0 - b.y, 79.6 - b.x)
                    b.vx, b.vy = 6.5 * math.cos(ang), 6.5 * math.sin(ang)
                b.vx *= 0.72
                b.vy *= 0.72
                b.x += b.vx * DT
                b.y += b.vy * DT
                st[pid] = 4
                continue
            if mark_of == holder and not after_score:
                a = math.radians(MARK_ANGLE)
                b.steer(o.x + MARK_DIST * math.cos(a), o.y + MARK_DIST * math.sin(a), gain=3.0)
                st[pid] = 5
                continue
            # person D: downfield (+x) and shading the break side (+y)
            lag = 1.0 if mark_of in cut_from else 0.0
            b.steer(o.x + 1.6 - lag, o.y + 0.9, vmax=b.vmax * (0.93 if mark_of in cut_from else 1.0))
            st[pid] = 7

        # ---- disc, stall, mark ----------------------------------------------------------
        mark = None
        if holder >= 0:
            hb = bodies[holder]
            d = [hb.x + 0.3, hb.y - 0.3, 1.0]
            mk = bodies[holder + N_PER_TEAM]
            gap = math.hypot(mk.x - hb.x, mk.y - hb.y)
            # a little hysteresis once the count is on, so a settling marker doesn't flicker
            if not after_score and gap <= (STALL_RANGE + 1.0 if stall is not None else STALL_RANGE):
                stall = 0.0 if stall is None else stall + DT
                mark = [MARK_ANGLE, MARK_HALF]
            else:
                stall = None
        else:
            th = active
            u = (tick - th["rel"]) / th["fl"]
            x, y = bezier(th["path"], u)
            disc_z = 1.0 + th["peak"] * math.sin(math.pi * u)
            d = [x, y, disc_z]

        frames.append({
            "p": [round(v, 2) for pid in range(2 * N_PER_TEAM) for v in (bodies[pid].x, bodies[pid].y)],
            "st": st,
            "d": [round(d[0], 2), round(d[1], 2), round(d[2], 2)],
            "ds": disc_state,
            "h": holder,
            "o": 0,
            "s": None if stall is None else round(min(stall, 10.0), 2),
            "m": mark,
        })

    events.sort(key=lambda e: e["t"])
    duration = round(score_tick * DT, 2)
    return {
        "format": "ultisim-replay",
        "version": 1,
        "meta": {
            "title": "Sample: vertical stack vs person D (force forehand)",
            "mode": "possession",
            "seed": seed,
            "dt": DT,
            "teams": [
                {"name": "Vertical stack", "strategy": "vertical"},
                {"name": "Person FH", "strategy": "person_fh"},
            ],
            "created": datetime.now().replace(microsecond=0).isoformat(),
            "synthetic": True,
        },
        "field": {"length": L, "width": W, "endzone": EZ},
        "players": players,
        "frames": frames,
        "throws": throws_out,
        "events": events,
        "possessions": [{"t0": 0.0, "t1": duration, "offense": 0, "force": "forehand", "outcome": "score"}],
        "result": {"outcome": "score", "team": 0, "duration": duration,
                   "throws": len(throws_out), "turnovers": 0},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "_sample", "sample.json"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    rep = build(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rep, f, separators=(",", ":"))
    os.replace(tmp, args.out)
    print(f"wrote {args.out}: {len(rep['frames'])} frames, {len(rep['throws'])} throws, "
          f"{len(rep['events'])} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
