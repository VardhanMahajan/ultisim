"""Pure-Python reference simulator (readable, single game). The C engine (Phase 3) must match it.

Frame: the engine always works in the CURRENT OFFENSE's frame (offense attacks +x). On a turnover
in full-point mode the whole state is rotated 180 degrees (x -> L - x, y -> W - y) and `flipped`
toggles; the recorder converts back to absolute field coordinates.

Player ids: 0-6 team 0, 7-13 team 1.

Tick order:  thrower decision -> behaviour targets -> movement -> disc flight / catches -> stall.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, ndtr

from . import disc as D
from .genome import DefenseGenome, OffenseGenome
from .movement import brake_power, step, time_to_reach
from .replay import Recorder
from .roster import Roster
from .value_field import ValueField

# player state codes (shared with the viewer, see viz/REPLAY_FORMAT.md)
SET, CUT, CLEAR, CHASE, GROUND, MARK, HOLD, DEF, ZONE = range(9)
JAB, RESET = 101, 102  # internal offense modes, shown as CUT
TURNOVER_TYPES = ("block", "interception", "drop", "throwaway", "out_of_bounds", "stall", "mark_block", "timeout")


@dataclass
class Flight:
    thrower: int
    receiver: int
    ttype: int
    huck: bool
    curve: int
    is_break: bool
    p0: np.ndarray
    c: np.ndarray
    p1: np.ndarray
    intended: np.ndarray
    t0: float
    T: float
    lo: float
    hi: float
    dist: float
    s_end: float = 1.0          # flight fraction at which the disc hits the ground (> 1: tail)
    peak: float = 1.5           # peak height (m) of the flight
    loft: bool = False          # lofted touch pass (slower, higher)
    h0: float = 1.0             # release height (hammers go overhead)
    rec_idx: int = -1
    prev_pos: np.ndarray | None = None
    bid_done: set = field(default_factory=set)
    touched: set = field(default_factory=set)   # defenders who already got a fingertip on it


@dataclass
class PossessionResult:
    outcome: str          # "score" or one of TURNOVER_TYPES
    offense: int
    duration: float
    throws: int = 0
    completions: int = 0
    hucks: int = 0
    huck_completions: int = 0
    breaks: int = 0
    break_completions: int = 0
    hammers: int = 0
    fakes: int = 0
    max_stall: float = 0.0
    end_pos: tuple = (0.0, 0.0)   # offense-frame location of the disc at the end


def _sig(x):
    return expit(x)


FEATURE_NAMES = ["bias", "logit_p", "log_1m_pdef", "log_precv", "log_pok", "log_1m_pmark", "log_pin", "hands",
                 "mid_margin", "end_margin", "mid_margin_neg", "end_margin_neg", "dist30", "dist30_sq", "flight_T",
                 "huck", "hammer", "curved", "break", "stall", "rcv_speed", "rcv_late", "lofted"]


def throw_features(p, p_def, p_recv, p_ok, p_mark, p_in, hands, mid_margin, end_margin, dist, T, huck, ttype,
                   curve, is_break, stall, rcv_speed, trec, loft) -> np.ndarray:
    """Features for the learned completion model (one row per candidate throw)."""
    lg = lambda v: np.log(np.clip(v, 1e-4, 1.0))
    mm = np.clip(mid_margin, -2.0, 2.0)
    em = np.clip(end_margin, -2.0, 2.0)
    return np.stack([
        np.ones_like(p), np.log(p / (1 - p)), lg(1 - p_def), lg(p_recv), lg(p_ok), lg(1 - p_mark), lg(p_in),
        np.clip(hands, 0, 1), mm, em, np.minimum(mm, 0.0), np.minimum(em, 0.0), dist / 30.0, (dist / 30.0) ** 2, T,
        huck.astype(float), (ttype == D.HAM).astype(float), (curve != 0).astype(float), is_break.astype(float),
        np.full_like(p, stall), rcv_speed / 8.0, np.clip(trec - T, -2.0, 2.0), loft.astype(float),
    ], axis=1)


_MODEL_CACHE: dict = {}


def load_throw_model(cfg):
    """Learned completion-model weights (configs/throw_model.json), or None to use the analytic model."""
    import json
    from pathlib import Path
    path = getattr(cfg.throws, "learned_model", "")
    if not path:
        return None
    full = Path(__file__).resolve().parent.parent / path
    if not full.exists():
        return None
    key = (str(full), full.stat().st_mtime)
    if key not in _MODEL_CACHE:
        d = json.loads(full.read_text())
        assert d["features"] == FEATURE_NAMES, "throw model features changed - refit (scripts/fit_throw_model.py)"
        _MODEL_CACHE[key] = np.array(d["weights"], dtype=float)
    return _MODEL_CACHE[key]


class Game:
    def __init__(self, cfg, rosters: tuple[Roster, Roster],
                 offense_genomes: tuple[OffenseGenome, OffenseGenome | None],
                 defense_genomes: tuple[DefenseGenome | None, DefenseGenome],
                 seed: int = 0, record: bool = False, title: str = "", mode: str = "possession",
                 value_field: ValueField | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.R = rosters[0].concat(rosters[1])
        self.og = offense_genomes
        self.dg = defense_genomes
        self.vf = value_field or ValueField(cfg)
        self.L, self.W = cfg.field.length, cfg.field.width
        self.ez = cfg.field.endzone_depth
        self.dt = cfg.sim.dt
        R = self.R
        m = cfg.movement
        self.brake = brake_power(R.accel, R.agility, cfg)
        self.reach = m.reach_base + m.reach_height * (R.height - 1.6) + m.reach_vert * R.vertical
        self.eff_height = R.height + 0.6 * R.vertical
        # reach envelope: standing reach from height; jumping adds the vertical but costs lateral reach
        self.standing = R.height * m.standing_reach_ratio
        self.max_reach_h = self.standing + R.vertical
        self.ranges = np.stack([R.range_backhand, R.range_forehand, R.range_hammer], axis=1)

        self.pos = np.zeros((14, 2))
        self.vel = np.zeros((14, 2))
        self.mode = np.full(14, SET)
        self.mtarget = np.zeros((14, 2))
        self.cut_end = np.zeros((14, 2))
        self.muntil = np.zeros(14)
        self.ground_until = np.full(14, -1.0)
        self.next_cut = np.zeros(14)
        self.slot = np.full(14, -1)
        self.zslot = np.full(14, -1)
        self.match = {}            # defender id -> attacker id (person-role defenders)
        self.next_switch = 0.0
        self.k_eff = 0              # zone players right now (transition can switch to person)
        self.t = 0.0
        self.offense = 0
        self.flipped = False
        self.holder = -1
        self.disc_pos = np.zeros(2)
        self.flight: Flight | None = None
        self.stall = 0.0
        self.ready_time = 0.0
        self.grace_until = 0.0
        self.next_decision = 0.0
        self.fake_until = -1.0
        self.fake_effect_until = -1.0
        self.last_fake = -10.0
        self.s_open = 1.0
        self.reset_player = -1
        self.hist = deque(maxlen=int(math.ceil(0.6 / self.dt)) + 2)
        self.res: PossessionResult | None = None
        self.throw_model = load_throw_model(cfg)
        self.throw_log: list[dict] = []   # per-throw diagnostics: predicted p vs. outcome
        self.decision_log: list[tuple] = []  # per-decision: (stall, best true p available, nearest-defender gap of best)
        self.mode_name = mode
        self.rec = None
        if record:
            names = [offense_genomes[0].name if offense_genomes[0] else "-",
                     defense_genomes[1].name if defense_genomes[1] else "-"]
            if mode == "point":
                names = [f"{offense_genomes[0].name} / {defense_genomes[0].name}",
                         f"{offense_genomes[1].name} / {defense_genomes[1].name}"]
            strategies = [offense_genomes[0].name if offense_genomes[0] else "",
                          defense_genomes[1].name if defense_genomes[1] else ""]
            self.rec = Recorder(cfg, title or f"{names[0]} vs {names[1]}", mode, seed, names, strategies, self.R)

    # ------------------------------------------------------------------ helpers
    @property
    def O(self) -> np.ndarray:
        return np.arange(7) + 7 * self.offense

    @property
    def Dd(self) -> np.ndarray:
        return np.arange(7) + 7 * (1 - self.offense)

    @property
    def ogen(self) -> OffenseGenome:
        return self.og[self.offense]

    @property
    def dgen(self) -> DefenseGenome:
        return self.dg[1 - self.offense]

    def num(self, i: int) -> str:
        return f"#{i % 7 + 1}{'AB'[i // 7]}"

    def event(self, etype, player, text):
        if self.rec:
            self.rec.event(self.t, etype, player, text)

    def disc_ref(self) -> np.ndarray:
        if self.flight is not None:
            return self.flight.intended
        return self.disc_pos

    def compute_s_open(self, y: float) -> float:
        f = self.dgen.force_name
        toward_mid = 1.0 if y <= self.W / 2 else -1.0
        if f == "forehand":
            return -1.0
        if f == "backhand":
            return 1.0
        if f == "sideline":
            return -toward_mid
        return toward_mid  # middle / straight up: open toward the middle of the field

    def frame_vec(self, dx, dy):
        return np.stack([np.asarray(dx, float), self.s_open * np.asarray(dy, float)], axis=-1)

    def dir_from_angle(self, ang_deg):
        a = np.radians(ang_deg)
        return np.stack([np.cos(a), self.s_open * np.sin(a)], axis=-1)

    def clip_field(self, p, margin=0.8):
        p = np.array(p, dtype=float)
        p[..., 0] = np.clip(p[..., 0], margin, self.L - margin)
        p[..., 1] = np.clip(p[..., 1], margin, self.W - margin)
        return p

    def mark_geometry(self):
        """(centre angle deg in field frame, half-width) of the marker's coverage wedge, or None."""
        g = self.dgen
        if self.dgen.force_name == "straight_up":
            ma = 0.0
        else:
            ma = g.mark_angle
        center = math.degrees(math.atan2(-self.s_open * math.sin(math.radians(ma)), math.cos(math.radians(ma))))
        if self.t < self.fake_effect_until:
            center += self.s_open * self.cfg.throws.fake_shift_deg  # pulled toward the open side by the fake
        # body-based coverage: the marker's reach seen from the thrower; closer mark = wider angle
        width = math.degrees(math.atan2(self.cfg.defense.mark_reach, g.mark_dist))
        return center, width

    def urgency(self) -> float:
        """Possession urgency: after urgency_start seconds of possession the thrower's patience and risk
        aversion fade linearly (to urgency_floor after another urgency_span seconds); nobody swings forever."""
        sc = self.cfg.sim
        u = min(max((self.t - self.poss_t0 - sc.urgency_start) / sc.urgency_span, 0.0), 1.0)
        return 1.0 - (1.0 - sc.urgency_floor) * u

    def mark_quality(self) -> float:
        """A tight mark has less time to react to the release: block chance scales by 1 - exp(-dist / d0)."""
        return 1.0 - math.exp(-self.dgen.mark_dist / self.cfg.defense.mark_react_dist)

    def defender_of(self, a: int) -> int:
        for d, b in self.match.items():
            if b == a:
                return d
        return -1

    def rest_mode(self, d: int) -> int:
        return ZONE if self.zslot[d] >= 0 else DEF

    def help_delay(self, d: int, receiver: int) -> float:
        """Extra read delay for person-role defenders who are not guarding the receiver."""
        if d in self.match and self.match[d] != receiver:
            return self.cfg.defense.help_read_delay
        return 0.0

    def marker_id(self) -> int:
        if self.holder < 0:
            return -1
        if self.k_eff >= 1:   # zone / flex: whoever holds zone spot 0 marks
            ids = [d for d in self.Dd if self.zslot[d] == 0]
            return ids[0] if ids else -1
        return self.defender_of(self.holder)

    def marker_set(self) -> bool:
        mk = self.marker_id()
        return mk >= 0 and np.hypot(*(self.pos[mk] - self.pos[self.holder])) <= self.cfg.defense.mark_set_radius

    def delayed(self, ids: np.ndarray, delays: np.ndarray):
        """Positions/velocities of `ids` as they were `delays` seconds ago (linear interpolation)."""
        if not self.hist:
            return self.pos[ids].copy(), self.vel[ids].copy()
        H = len(self.hist)
        k = np.clip(delays / self.dt, 0.0, max(H - 1.000001, 0.0))
        k0 = np.clip(np.floor(k).astype(int), 0, H - 1)
        w = (k - k0)[:, None]
        P = np.empty((len(ids), 2))
        V = np.empty((len(ids), 2))
        for j, (i, a, b) in enumerate(zip(ids, k0, k0 + 1)):
            pa, va = self.hist[H - 1 - a]
            pb, vb = self.hist[H - 1 - min(b, H - 1)]
            P[j] = (1 - w[j]) * pa[i] + w[j] * pb[i]
            V[j] = (1 - w[j]) * va[i] + w[j] * vb[i]
        return P, V

    # ------------------------------------------------------------------ setup
    def homes(self, ref: np.ndarray) -> np.ndarray:
        g = self.ogen
        x = ref[0] + g.home_dx
        y_disc = ref[1] + self.s_open * g.home_dy
        y_mid = self.W / 2 + self.s_open * g.home_dy
        y = (1 - g.y_anchor) * y_disc + g.y_anchor * y_mid
        return self.clip_field(np.stack([x, y], axis=1), margin=1.2)

    def assign_slots(self, players: np.ndarray, ref: np.ndarray):
        H = self.homes(ref)
        g = self.ogen
        cost = np.hypot(*(self.pos[players][:, None, :] - H[None, :, :]).transpose(2, 0, 1))
        cost = cost + 6.0 * (g.handler[None, :] > 0.5) * (1.0 - self.R.throw_skill[players])[:, None]
        r, c = linear_sum_assignment(cost)
        self.slot[players] = -1
        self.slot[players[r]] = c

    def assign_zone(self):
        ids = self.Dd
        pts = self.zone_points(self.disc_ref())
        cost = np.hypot(*(self.pos[ids][:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
        r, c = linear_sum_assignment(cost)
        self.zslot[ids[r]] = c

    def assign_flex(self, k: int):
        """k defenders take zone spots 0..k-1 (spot 0 = marker); the other 7-k play person on the cutters
        chosen by flex_focus (deepest first at 1, closest to the disc first at 0)."""
        g = self.dgen
        Dd = self.Dd
        ref = self.disc_ref()
        pts = self.zone_points(ref)[:k]
        att = np.array([a for a in self.O if a != self.holder])
        depth = self.pos[att, 0] - ref[0]
        chosen = att[np.argsort(-(2.0 * g.flex_focus - 1.0) * depth, kind="stable")[:7 - k]]
        cols = np.vstack([pts, self.pos[chosen]])
        cost = np.hypot(*(self.pos[Dd][:, None, :] - cols[None, :, :]).transpose(2, 0, 1))
        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            d = int(Dd[i])
            if j < k:
                self.zslot[d] = j
            else:
                self.match[d] = int(chosen[j - k])

    def assign_defense(self, by_distance: bool):
        k = self.k_eff
        self.zslot[self.Dd] = -1
        self.match = {}
        if k == 0:
            self.assign_matchups(by_distance)
        elif k == 7:
            self.assign_zone()
        else:
            self.assign_flex(k)
        self.next_switch = self.t + self.cfg.defense.switch_interval

    def zone_points(self, ref) -> np.ndarray:
        g = self.dgen
        mc, _ = self.mark_geometry()
        mk = ref + g.mark_dist * np.array([math.cos(math.radians(mc)), math.sin(math.radians(mc))])
        rest = ref + self.frame_vec(g.zone_dx, g.zone_dy)
        if g.trap > 0.0:
            edge = min(ref[1], self.W - ref[1])
            tf = g.trap * float(np.clip(1.0 - edge / self.cfg.defense.trap_zone, 0.0, 1.0))
            if tf > 0.0:
                short = np.hypot(*(rest - ref).T) < 15.0
                rest[short] = rest[short] + 0.4 * tf * (ref - rest[short])
        return self.clip_field(np.vstack([mk[None, :], rest]))

    def assign_matchups(self, by_distance: bool):
        O, Dd = self.O, self.Dd
        g = self.dgen
        sp_o = self.R.top_speed[O]
        sp_d = self.R.top_speed[Dd]
        cost = g.match_speed * np.abs(sp_d[:, None] - sp_o[None, :])
        if by_distance:
            cost = cost + 0.3 * np.hypot(*(self.pos[Dd][:, None, :] - self.pos[O][None, :, :]).transpose(2, 0, 1))
        r, c = linear_sum_assignment(cost)
        self.match = {int(Dd[i]): int(O[j]) for i, j in zip(r, c)}

    def start_possession(self, disc_xy: np.ndarray, fresh: bool):
        """fresh=True: set formation (after a pull / possession-mode start). Else keep positions."""
        O = self.O
        self.flight = None
        self.disc_pos = np.array(disc_xy, float)
        self.s_open = self.compute_s_open(self.disc_pos[1])
        self.stall = 0.0
        self.reset_player = -1
        self.fake_until = self.fake_effect_until = -1.0
        if fresh:
            g = self.ogen
            H = self.homes(self.disc_pos)
            spots = np.vstack([self.disc_pos[None, :], H])
            handlerish = np.concatenate([[1.0], (g.handler > 0.5).astype(float)])
            cost = -np.outer(self.R.throw_skill[O], handlerish) * 3.0 + self.rng.random((7, 7)) * 0.1
            r, c = linear_sum_assignment(cost)
            self.pos[O[r]] = spots[c]
            self.vel[O] = 0.0
            self.holder = int(O[r][c == 0][0])
            others = O[O != self.holder]
            self.assign_slots(others, self.disc_pos)
            grace = 0.0
        else:
            d = np.hypot(*(self.pos[O] - self.disc_pos).T)
            self.holder = int(O[np.argmin(d)])
            self.pos[self.holder] = self.disc_pos
            self.vel[self.holder] = 0.0
            others = O[O != self.holder]
            self.assign_slots(others, self.disc_pos)
            grace = self.cfg.rules.pickup_grace
        self.mode[O] = SET
        self.mode[self.holder] = HOLD
        self.next_cut[O] = 0.0
        self.ground_until[:] = -1.0
        self.ready_time = self.t + self.R.release_time[self.holder] + grace
        self.grace_until = self.t + grace
        self.next_decision = self.ready_time
        self.k_eff = self.dgen.n_zone
        k = self.k_eff
        if fresh and k >= 1:  # zone spots first (as if set after the pull), then assign by distance
            pts = self.zone_points(self.disc_pos)
            self.pos[self.Dd] = pts[self.rng.permutation(7)]
            self.vel[self.Dd] = 0.0
        self.assign_defense(by_distance=not (fresh and k == 0))
        if fresh and k < 7:  # person-role defenders start in their scheme positions
            for d, a in self.match.items():
                self.pos[d] = self.pos[a] + np.array([1.5, 0.0])
            self.vel[self.Dd] = 0.0
            self.hist.clear()
            for _ in range(2):
                t_, s_, c_ = self.pos.copy(), np.zeros(14), np.zeros(14, int)
                for d in self.Dd:
                    self.mode[d] = self.rest_mode(d)
                self.defense_targets(t_, s_, c_)
                self.pos[self.Dd] = t_[self.Dd]
        for d in self.Dd:
            self.mode[d] = self.rest_mode(d)
        self.hist.clear()
        self.poss_t0 = self.t
        self.res = PossessionResult(outcome="", offense=self.offense, duration=0.0)
        self.event("pull" if fresh else "turnover", self.holder,
                   f"{self.num(self.holder)} starts with the disc (force {self.dgen.force_name})")

    # ------------------------------------------------------------------ offense behaviour
    def offense_targets(self, tgt, spd, codes):
        g = self.ogen
        O = self.O
        ref = self.disc_ref()
        H = self.homes(ref)
        top = self.R.top_speed
        jog = self.cfg.movement.jog_fraction
        have_disc = self.holder >= 0
        # reset trigger
        if have_disc and self.stall >= g.reset_stall and self.reset_player < 0:
            cands = [i for i in O if i != self.holder and self.slot[i] >= 0 and g.handler[self.slot[i]] > 0.5
                     and self.mode[i] not in (GROUND,)]
            if cands:
                rs = self.disc_pos + self.frame_vec(g.reset_dx, g.reset_dy)
                self.reset_player = min(cands, key=lambda i: np.hypot(*(self.pos[i] - rs)))
                self.mode[self.reset_player] = RESET
        for i in O:
            md = self.mode[i]
            if self.t < self.ground_until[i]:
                tgt[i] = self.pos[i]; spd[i] = 0.0; codes[i] = GROUND
                continue
            if md == GROUND:
                self.mode[i] = md = CLEAR
            if md == HOLD:
                tgt[i] = self.pos[i]; spd[i] = 0.0; codes[i] = HOLD
                continue
            if md == CHASE:
                tgt[i] = self.mtarget[i]; spd[i] = top[i]; codes[i] = CHASE
                continue
            k = self.slot[i]
            if k < 0:
                tgt[i] = self.pos[i]; spd[i] = 0.0; codes[i] = SET
                continue
            home = H[k]
            if md == RESET:
                rs = self.clip_field(self.disc_pos + self.frame_vec(g.reset_dx, g.reset_dy))
                tgt[i] = rs; spd[i] = top[i]; codes[i] = CUT
                continue
            if md == SET:
                if (have_disc and self.t >= self.ready_time - 0.3 + g.cut_delay[k] and self.t >= self.next_cut[i]):
                    self.begin_cut(i, k)
                    md = self.mode[i]
                else:
                    dist = np.hypot(*(home - self.pos[i]))
                    tgt[i] = home
                    spd[i] = top[i] * (jog if dist < 6 else 0.85)
                    codes[i] = SET if dist < 1.5 else CLEAR
                    continue
            if md == JAB:
                if self.t >= self.muntil[i]:
                    self.mode[i] = md = CUT
                    self.mtarget[i] = self.cut_end[i]
                    self.muntil[i] = self.t + np.hypot(*(self.cut_end[i] - self.pos[i])) / top[i] + 1.2
                else:
                    tgt[i] = self.mtarget[i]; spd[i] = top[i] * 0.7; codes[i] = CUT
                    continue
            if md == CUT:
                if np.hypot(*(self.mtarget[i] - self.pos[i])) < 1.0 or self.t >= self.muntil[i]:
                    self.mode[i] = md = CLEAR
                    self.next_cut[i] = self.t + g.recut_delay[k]
                else:
                    tgt[i] = self.mtarget[i]; spd[i] = top[i]; codes[i] = CUT
                    continue
            if md == CLEAR:
                if np.hypot(*(home - self.pos[i])) < 2.0:
                    self.mode[i] = SET
                tgt[i] = home; spd[i] = top[i] * 0.8; codes[i] = CLEAR

    def begin_cut(self, i: int, k: int):
        g = self.ogen
        dirA = self.dir_from_angle(g.cutA_ang[k])
        dirB = self.dir_from_angle(g.cutB_ang[k])
        away = np.zeros(2)
        mydef = self.defender_of(i)
        if mydef >= 0:
            v = self.pos[i] - self.pos[mydef]
        else:
            Dd = self.Dd
            dn = Dd[np.argmin(np.hypot(*(self.pos[Dd] - self.pos[i]).T))]
            v = self.pos[i] - self.pos[dn]
        nv = np.hypot(*v)
        if nv > 0.3:
            away = v / nv
        la = g.cutA_bias[k] + g.read_def[k] * float(dirA @ away)
        lb = g.read_def[k] * float(dirB @ away)
        pa = _sig(la - lb)
        if self.rng.random() < pa:
            d, ln = dirA, g.cutA_len[k]
        else:
            d, ln = dirB, g.cutB_len[k]
        if g.seek_space[k] > 0.0:
            rel = self.pos[i] - self.pos[self.Dd]
            dist = np.hypot(*rel.T)
            near = dist < 10.0
            if near.any():
                sv = (rel[near] / np.maximum(dist[near], 0.5)[:, None] ** 2).sum(axis=0)
                sn = np.hypot(*sv)
                if sn > 1e-6:
                    d = (1.0 - g.seek_space[k]) * d + g.seek_space[k] * sv / sn
                    d = d / max(np.hypot(*d), 1e-6)
        end = self.clip_field(self.pos[i] + d * ln, margin=0.7)
        if not hasattr(self, "cut_end"):
            self.cut_end = np.zeros((14, 2))
        self.cut_end[i] = end
        if self.rng.random() < g.jab[k]:
            self.mode[i] = JAB
            self.mtarget[i] = self.clip_field(self.pos[i] - d * 1.5)
            self.muntil[i] = self.t + 0.35
        else:
            self.mode[i] = CUT
            self.mtarget[i] = end
            self.muntil[i] = self.t + ln / self.R.top_speed[i] + 1.2

    # ------------------------------------------------------------------ defense behaviour
    def defense_targets(self, tgt, spd, codes, tvel=None):
        g = self.dgen
        cfg = self.cfg
        Dd = self.Dd
        top = self.R.top_speed
        ref = self.disc_pos if self.flight is None else self.flight_pos_now()
        mk = self.marker_id()
        chasing = set(i for i in Dd if self.mode[i] == CHASE)
        Pa, Va = self.delayed(self.O, np.full(7, 0.25))          # what defenders see of the offense (all roles)
        if self.flight is not None:   # zone players read the disc late: their spots follow where it was
            f_ = self.flight
            zref = self.fpos(f_, min(max((self.t - cfg.defense.zone_read_delay - f_.t0) / f_.T, 0.0), f_.s_end))
        else:
            zref = ref
        pts = self.zone_points(zref) if any(self.zslot[d] >= 0 for d in Dd) else None
        person = [int(d) for d in Dd if int(d) in self.match]
        PD = {}
        if person:
            Pd_, Vd_ = self.delayed(np.array([self.match[d] for d in person]), self.R.reaction_time[person])
            PD = {d: (Pd_[n], Vd_[n]) for n, d in enumerate(person)}
        for i in Dd:
            if self.t < self.ground_until[i]:
                tgt[i] = self.pos[i]; spd[i] = 0.0; codes[i] = GROUND
                continue
            if self.mode[i] == GROUND:
                self.mode[i] = self.rest_mode(int(i))
            if i in chasing:
                tgt[i] = self.mtarget[i]; spd[i] = top[i]; codes[i] = CHASE
                continue
            if i == mk:
                mc, _ = self.mark_geometry()
                tgt[i] = self.pos[self.holder] + g.mark_dist * np.array([math.cos(math.radians(mc)),
                                                                          math.sin(math.radians(mc))])
                spd[i] = top[i]; codes[i] = MARK
                continue
            if self.zslot[i] >= 0 or int(i) not in self.match:
                p = pts[self.zslot[i]] if (pts is not None and self.zslot[i] >= 0) else self.pos[i]
                dd = np.hypot(*(Pa - p).T)
                n = int(np.argmin(dd))
                if dd[n] < 8.0:
                    sh = g.zone_shade * (Pa[n] - p)
                    ln = np.hypot(*sh)
                    if ln > 5.0:
                        sh *= 5.0 / ln
                    p = p + sh
                tgt[i] = self.clip_field(p); spd[i] = top[i]; codes[i] = ZONE
                continue
            Pdj, Vdj = PD[int(i)]
            pa = Pdj + g.anticipation * Vdj * self.R.reaction_time[i]
            to_disc = ref - pa
            dist = np.hypot(*to_disc)
            u = to_disc / max(dist, 1e-6)
            w = np.clip((dist - 15.0) / 15.0, 0.0, 1.0)
            th = math.radians((1 - w) * g.angle_near + w * g.angle_far)
            dvec = math.cos(th) * u + math.sin(th) * np.array([0.0, self.s_open])
            dvec /= max(np.hypot(*dvec), 1e-6)
            cushion = g.cushion_base + g.cushion_per_m * dist
            p = pa + cushion * dvec
            slow = np.hypot(*Vdj) < 3.0
            if slow and dist > 12.0:
                p = (1 - g.poach) * p + g.poach * (ref + 0.55 * (pa - ref))
            if slow and g.stack_sag > 0.0:
                # surround the stack: defenders of stationary attackers in a cluster sag into it
                a_idx = self.match[int(i)] - self.O[0]
                da = np.hypot(*(Pa - Pa[a_idx]).T)
                nb = da < cfg.defense.stack_radius
                if nb.sum() >= 3:  # the attacker + at least 2 neighbours
                    cen = Pa[nb].mean(axis=0)
                    tc = ref - cen
                    sag = cen + 2.0 * tc / max(np.hypot(*tc), 1e-6)
                    wsag = 0.6 * g.stack_sag
                    p = (1 - wsag) * p + wsag * sag
            tgt[i] = self.clip_field(p, margin=0.3); spd[i] = top[i]; codes[i] = DEF
            if tvel is not None:
                tvel[i] = Vdj   # match the attacker's (delayed) velocity, then correct position
        self.help_on_cutter(tgt, tvel, mk, ref, Pa, Va)
        self.enforce_double_team(tgt, mk, tvel)

    def help_on_cutter(self, tgt, tvel, mk, ref, Pa, Va):
        """Help defense: when a cutter has clearly beaten their defender, the nearest free defender
        (guarding a stationary attacker away from the disc) steps into that cutter's throwing lane."""
        g = self.dgen
        cfg = self.cfg
        if g.help_cut <= 0.0 or self.holder < 0 or self.flight is not None or len(self.match) < 2:
            return
        O0 = self.O[0]
        best = None
        for d, a in sorted(self.match.items()):
            if d == mk or a == self.holder or self.mode[d] in (CHASE, GROUND):
                continue
            sep = np.hypot(*(Pa[a - O0] - self.pos[d]))
            if np.hypot(*Va[a - O0]) > 3.0 and sep > cfg.defense.help_sep and (best is None or sep > best[0]):
                best = (sep, a, d)
        if best is None:
            return
        _, a, d = best
        lane = Pa[a - O0] + 0.35 * (ref - Pa[a - O0])
        helpers = [h for h, b in sorted(self.match.items())
                   if h not in (d, mk) and self.mode[h] not in (CHASE, GROUND)
                   and np.hypot(*Va[b - O0]) < 3.0 and np.hypot(*(Pa[b - O0] - ref)) > 8.0]
        if not helpers:
            return
        h = min(helpers, key=lambda x: np.hypot(*(self.pos[x] - lane)))
        tgt[h] = self.clip_field((1 - g.help_cut) * tgt[h] + g.help_cut * lane, margin=0.3)
        if tvel is not None:
            tvel[h] = 0.0

    def maybe_switch(self):
        """Switching (clam): every switch_interval, re-match person defenders to the nearest attackers if
        that saves enough running; switch = 1 always re-matches, 0 never."""
        g = self.dgen
        cfg = self.cfg
        if g.switch <= 0.0 or self.t < self.next_switch or len(self.match) < 2:
            return
        self.next_switch = self.t + cfg.defense.switch_interval
        mk = self.marker_id()
        ds = [d for d in sorted(self.match) if d != mk and self.mode[d] not in (CHASE, GROUND)]
        if len(ds) < 2:
            return
        att = [self.match[d] for d in ds]
        cost = np.hypot(*(self.pos[ds][:, None, :] - self.pos[att][None, :, :]).transpose(2, 0, 1))
        cur = float(np.trace(cost))
        r, c = linear_sum_assignment(cost)
        if cur - cost[r, c].sum() > (1.0 - g.switch) * cfg.defense.switch_gain:
            for i, j in zip(r, c):
                self.match[ds[i]] = att[j]

    def enforce_double_team(self, tgt, mk, tvel=None):
        """Only the marker may stand within the double-team radius of the thrower."""
        if self.holder < 0:
            return
        hp = self.pos[self.holder]
        r = self.cfg.rules.double_team_radius + 0.2
        for i in self.Dd:
            if i == mk or self.mode[i] == CHASE:
                continue
            att = self.match.get(int(i), -1)
            if att >= 0 and np.hypot(*(self.pos[att] - hp)) < r:
                continue  # legal: guarding another attacker who is also within the radius
            v = tgt[i] - hp
            d = np.hypot(*v)
            if d < r:
                tgt[i] = hp + (v / d if d > 1e-6 else np.array([1.0, 0.0])) * r
                if tvel is not None:
                    tvel[i] = 0.0

    # ------------------------------------------------------------------ throwing
    def predict_receivers(self, R: np.ndarray, taus: np.ndarray) -> np.ndarray:
        """Where each receiver will be after tau seconds (R,), (R,) -> (R,2)."""
        p = self.pos[R]
        v = self.vel[R]
        out = np.empty_like(p)
        for j, i in enumerate(R):
            if self.mode[i] in (CUT, JAB, RESET):
                goal = self.mtarget[i] if self.mode[i] != JAB else self.cut_end[i]
                dg = goal - p[j]
                ln = np.hypot(*dg)
                travel = min(np.hypot(*v[j]) * taus[j] + 0.5 * self.R.accel[i] * taus[j] ** 2 * 0.5,
                             self.R.top_speed[i] * taus[j], ln + 3.0)
                out[j] = p[j] + (dg / max(ln, 1e-6)) * travel
            else:
                goal = self.mtarget[i] if self.mode[i] == CLEAR else p[j]
                travel = min(np.hypot(*v[j]) * taus[j], np.hypot(*(goal - p[j])) + 1.0)
                vh = v[j] / max(np.hypot(*v[j]), 1e-6)
                out[j] = p[j] + vh * travel
        return out

    def evaluate_throws(self):
        """Score every candidate throw. Returns dict of candidate arrays + perceived utility."""
        cfg = self.cfg
        g = self.ogen
        h = self.holder
        hp = self.pos[h]
        R = np.array([i for i in self.O if i != h and self.t >= self.ground_until[i]])
        Dd = self.Dd
        if len(R) == 0:
            return None
        # throw menu: (type, curve, lofted)
        menu = [(D.BH, 0, 0), (D.BH, 0, 1), (D.BH, 1, 0), (D.BH, -1, 0),
                (D.FH, 0, 0), (D.FH, 0, 1), (D.FH, 1, 0), (D.FH, -1, 0), (D.HAM, 0, 0)]
        leads = [0.0, g.lead]
        # receiver predicted targets per lead (approximate flight time with backhand speed)
        targets = []
        rec_ids = []
        for lead in leads:
            tau = np.full(len(R), 0.5)
            for _ in range(2):
                tp = self.predict_receivers(R, tau)
                dd = np.hypot(*(tp - hp).T)
                tau = cfg.throws.backhand.hang + dd / cfg.throws.backhand.speed
            vhat = self.vel[R] / np.maximum(np.hypot(*self.vel[R].T), 1e-6)[:, None]
            moving = (np.hypot(*self.vel[R].T) > 2.0)[:, None]
            tp = tp + np.where(moving, vhat * lead, 0.0)
            targets.append(tp)
            rec_ids.append(R)
        tg = np.concatenate(targets)          # (Rn*2, 2)
        rid = np.concatenate(rec_ids)
        nT = len(tg)
        ttype = np.repeat([[m[0] for m in menu]], nT, axis=0).reshape(-1)
        curve = np.repeat([[m[1] for m in menu]], nT, axis=0).reshape(-1)
        loft = np.repeat([[m[2] for m in menu]], nT, axis=0).reshape(-1).astype(bool)
        tgt = np.repeat(tg, len(menu), axis=0)
        rcv = np.repeat(rid, len(menu))
        rp = D.release_point(hp, ttype, cfg)
        dvec = tgt - rp
        dist = np.hypot(*dvec.T)
        huck = (dist > cfg.throws.huck_min_dist) & (ttype != D.HAM)
        valid = (dist >= cfg.throws.min_throw_dist) & (dist <= self.ranges[h][ttype])
        valid &= ~((ttype == D.HAM) & (dist < 10.0))
        valid &= ~((curve != 0) & (dist < 8.0))
        valid &= ~(loft & (huck | (dist < cfg.throws.loft_min_dist) | (dist > cfg.throws.loft_max_dist)))
        if not cfg.throws.specials_everywhere:
            # curves (IO/OI) and hammers exist to beat the mark: only consider them for break-side targets
            brk_pre = np.zeros(len(tgt), bool)
            if self.marker_set():
                mc_, mw_ = self.mark_geometry()
                brk_pre = np.abs(D.wrap_deg(D.angle_deg(tgt - hp) - mc_)) < cfg.defense.break_side_deg
            valid &= ~(((curve != 0) | (ttype == D.HAM)) & ~brk_pre)
        if not valid.any():
            return None
        idx = np.nonzero(valid)[0]
        ttype, curve, tgt, rcv, rp, dist, huck = ttype[idx], curve[idx], tgt[idx], rcv[idx], rp[idx], dist[idx], huck[idx]
        loft = loft[idx]
        C = len(idx)
        T = D.flight_time(cfg, ttype, dist, huck, loft)
        peak = D.throw_peak(cfg, ttype, dist, huck, loft)
        ctrl = D.control_point(rp, tgt, curve, cfg)
        K = 10
        s = np.linspace(0.1, 1.0, K)
        pts = D.bezier(rp[:, None, :], ctrl[:, None, :], tgt[:, None, :], s[None, :])   # (C,K,2)
        tk = s[None, :] * T[:, None]
        h0_ = np.where(ttype == D.HAM, cfg.throws.height.hammer_release, cfg.throws.height.release)
        hk_ = D.flight_height(cfg, peak[:, None], s[None, :], T[:, None], h0_[:, None])   # (C,K) disc height
        reachable = (s[None, :] * dist[:, None] >= cfg.rules.double_team_radius)
        # reach envelope per defender and path point: out of reach above standing + vertical; jumping
        # (above standing reach) shrinks the lateral radius
        mv = cfg.movement
        allowed = hk_[None, :, :] <= self.max_reach_h[Dd][:, None, None]                    # (7,C,K)
        r_env = self.reach[Dd][:, None, None] * np.where(hk_[None, :, :] > self.standing[Dd][:, None, None],
                                                           mv.jump_reach_mult, 1.0)
        low = np.where(hk_ < cfg.throws.height.low_disc, cfg.throws.height.low_mult, 1.0)    # (C,K)
        # mark geometry and throw error first: error widens the lane defenders can reach into
        marked = self.marker_set()
        is_break = np.zeros(C, bool)
        p_mark = np.zeros(C)
        mk = self.marker_id()
        if marked:
            mc, mw = self.mark_geometry()
            ang = D.angle_deg(tgt - hp)
            is_break = np.abs(D.wrap_deg(ang - mc)) < cfg.defense.break_side_deg   # toward the marker's side
            sm = np.minimum(1.0, 2.0 / np.maximum(dist, 1e-6))
            early = D.bezier(rp, ctrl, tgt, sm)
            # judge the throw by the bisector of its release direction and its target direction: a curve
            # can bend a break around the mark, but not launder it completely
            u1 = (early - hp) / np.maximum(np.hypot(*(early - hp).T), 1e-9)[:, None]
            u2 = (tgt - hp) / np.maximum(np.hypot(*(tgt - hp).T), 1e-9)[:, None]
            ub = u1 + u2
            ub = ub / np.maximum(np.hypot(*ub.T), 1e-9)[:, None]
            early = hp + 2.0 * ub
            p_mark = D.mark_block_prob(cfg, hp, early, ttype, mc, mw, self.R.marking[mk], self.R.break_skill[h],
                                       self.t < self.fake_effect_until) * self.mark_quality()
        sig = D.error_sigma(cfg, self.R.throw_skill[h], ttype, dist, is_break, marked, curve != 0)
        # the real flight wanders ~sig * s off the aimed line by fraction s of the flight
        lane = cfg.throws.model_error_margin * sig[None, :, None] * s[None, None, :]       # (1,C,K)
        # defenders: before they react to the throw they keep moving on their current velocity
        # person-D help defenders (not guarding the receiver) are watching their own player and read
        # the throw later; the receiver's defender and zone players are facing the disc.
        def_of = np.full(14, -1)
        for d_, a_ in self.match.items():
            def_of[a_] = d_
        base_dl = self.R.reaction_time[Dd] + cfg.movement.disc_read_delay
        person_role = np.array([int(d) in self.match for d in Dd])
        help_dl = cfg.defense.help_read_delay if person_role.any() else 0.0
        guarding = (Dd[:, None] == def_of[rcv][None, :])                                  # (7,C)
        dlC = base_dl[:, None] + np.where(guarding | ~person_role[:, None], 0.0, help_dl)  # (7,C)
        tdef = None
        for extra in ({0.0, help_dl} if help_dl else {0.0}):
            dl = base_dl + extra
            p_react = self.pos[Dd] + self.vel[Dd] * dl[:, None]
            t_ = time_to_reach(p_react, self.vel[Dd], self.R.top_speed[Dd], self.R.accel[Dd],
                               self.brake[Dd], np.zeros(7), pts, extra_reach=r_env + lane) + dl[:, None, None]  # (7,C,K)
            sel = (dlC == dl[:, None])[:, :, None]
            tdef = np.where(sel, t_, tdef) if tdef is not None else t_
        # ...and anyone already standing in / drifting through the lane can get a hand on it
        # (a disc reaching them before they have read the throw is only a reflex chance)
        in_lane = np.zeros(tdef.shape, bool)
        for frac in (0.0, 0.5, 1.0):
            along = self.pos[Dd][:, None, None, :] + self.vel[Dd][:, None, None, :] * (frac * np.minimum(tk[None], dlC[:, :, None]))[..., None]
            in_lane |= np.hypot(*(along - pts[None]).transpose(3, 0, 1, 2)) <= r_env + lane
        in_lane &= allowed
        tdef = np.where(in_lane, np.minimum(tdef, tk[None] - 0.3), tdef)
        tdef = np.where(allowed, tdef, 1e9)            # disc above this defender's reach at that point
        reflex = np.where(in_lane & (tk[None] < dlC[:, :, None]), cfg.defense.reflex_block, 1.0)
        # hucks are hard to judge early: everyone reads them later
        tdef = tdef + cfg.movement.huck_read_extra * huck[None, :, None]
        # receiver arrival at the landing point
        trec = time_to_reach(self.pos[rcv], self.vel[rcv], self.R.top_speed[rcv], self.R.accel[rcv],
                             self.brake[rcv], self.reach[rcv], tgt, paired=True)
        # the receiver's own defender shadows them to the catch point (lagging by their current gap)
        if True:
            dc = def_of[rcv]
            has = dc >= 0
            if has.any():
                cc = np.nonzero(has)[0]
                d = dc[cc]
                j = d - Dd[0]
                gap = np.maximum(np.hypot(*(self.pos[d] - self.pos[rcv[cc]]).T) - self.reach[d] - lane[0, cc, -1], 0.0)
                # follows the receiver with a lag, but can only start closing the gap once the throw is read
                react = self.R.reaction_time[d] + cfg.movement.disc_read_delay
                track = np.maximum(trec[cc] + 0.5 * self.R.reaction_time[d], react) + gap / self.R.top_speed[d]
                tdef[j, cc, -1] = np.minimum(tdef[j, cc, -1], track)
        bp = cfg.defense.block_prob
        p_mid = (_sig((tk[None, :, :-1] - tdef[:, :, :-1]) / 0.12) * reachable[None, :, :-1] * bp * reflex[:, :, :-1]
                 * low[None, :, :-1])
        # at the catch: the defender must beat the disc AND not arrive much later than the receiver
        p_end = (_sig((T[None, :] - tdef[:, :, -1]) / 0.15) * _sig((np.minimum(trec, T)[None, :] - tdef[:, :, -1] + 0.1) / 0.12)
                 * bp * 0.8 * reachable[None, :, -1])
        p_d = np.maximum(p_mid.max(axis=2), p_end)            # (7, C)
        p_def = 1.0 - np.prod(1.0 - p_d, axis=0)
        p_recv = _sig((T + 0.15 - trec) / 0.15)
        radius = self.reach[rcv] + 0.3 + cfg.throws.receiver_adjust * np.maximum(0.0, T - trec)
        p_ok = 1.0 - np.exp(-radius ** 2 / (2 * sig ** 2))
        p_in = ndtr(tgt[:, 1] / sig) * ndtr((self.W - tgt[:, 1]) / sig) * ndtr((self.L - tgt[:, 0]) / sig)
        vdisc = dist / T
        near_def = (np.min(np.hypot(*(self.pos[Dd][:, None, :] - tgt[None, :, :]).transpose(2, 0, 1)), axis=0) < 2.0)
        hands = (self.R.hands[rcv] - cfg.catching.speed_drop_per_ms * np.maximum(vdisc - 12.0, 0)
                 - cfg.catching.contest_drop * near_def - cfg.catching.huck_drop * huck
                 - cfg.catching.hammer_drop * (ttype == D.HAM))
        p = (1 - p_mark) * (1 - p_def) * p_recv * p_ok * p_in * np.clip(hands, 0.0, 1.0)
        p = np.clip(p, 1e-4, 1 - 1e-4)
        # time margins of the most dangerous defender: mid-flight and at the catch
        mid_margin = np.where(reachable[None, :, :-1], tdef[:, :, :-1] - tk[None, :, :-1], 9.0).min(axis=(0, 2))
        end_margin = (tdef[:, :, -1] - np.minimum(trec, T)[None, :]).min(axis=0)
        X = throw_features(p, p_def, p_recv, p_ok, p_mark, p_in, hands, mid_margin, end_margin, dist, T, huck,
                           ttype, curve, is_break, self.stall / self.cfg.rules.stall_max,
                           np.hypot(*self.vel[rcv].T), trec, loft)
        if self.throw_model is not None:
            p = np.clip(_sig(X @ self.throw_model), 1e-4, 1 - 1e-4)
        noise = cfg.throws.read_noise_base + cfg.throws.read_noise_skill * self.R.decision_noise[h]  # logit-space
        p_hat = _sig(np.log(p / (1 - p)) + self.rng.normal(0.0, noise, C))
        V = self.vf.value(tgt)
        Cst = self.vf.turnover_cost(tgt)
        U = p_hat * V - (1 - p_hat) * g.risk_aversion * self.urgency() * Cst
        U += g.huck_bias * huck + g.break_bias * is_break + g.hammer_bias * (ttype == D.HAM)
        return dict(U=U, p=p, p_hat=p_hat, p_mark=p_mark, p_d=p_d, tdef=tdef, pts=pts, tk=tk, X=X, loft=loft,
                    parts=dict(p_def=p_def, p_recv=p_recv, p_ok=p_ok, p_in=p_in, hands=hands, trec=trec,
                               mid_margin=mid_margin, end_margin=end_margin), ttype=ttype, curve=curve, tgt=tgt, rcv=rcv, rp=rp,
                    ctrl=ctrl, dist=dist, huck=huck, T=T, is_break=is_break, sig=sig, marked=marked)

    def thrower_decision(self):
        g = self.ogen
        ev = self.evaluate_throws()
        if ev is None:
            return
        b = int(np.argmax(ev["U"]))
        eps = self.cfg.sim.explore_eps
        if eps > 0 and self.rng.random() < eps:
            top = np.argsort(-ev["U"])[:12]
            b = int(self.rng.choice(top))
            ev["U"] = ev["U"].copy()
            ev["U"][b] = np.inf  # force this throw
        self.decision_log.append((self.stall, float(ev["p"].max()), float(ev["p"][b])))
        here = self.vf.value(self.pos[self.holder][None, :])[0]
        # urgency follows the stall count, or time held if nobody is counting (no marker in range)
        s_eff = max(self.stall, self.t - self.ready_time)
        hold = here * g.patience * self.urgency() * max(1.0 - (s_eff / self.cfg.rules.stall_max) ** 2, 0.0)
        must = s_eff >= g.panic_stall
        if ev["U"][b] < hold and not must:
            # consider a fake if the best option is a break that the mark is taking away
            if (ev["is_break"][b] and ev["p_mark"][b] > 0.12 and self.t - self.last_fake > 1.2
                    and self.rng.random() < g.fake_rate * self.cfg.sim.decision_interval * 2):
                self.do_fake()
            return
        if ev["is_break"][b] and ev["p_mark"][b] > 0.2 and self.t - self.last_fake > 1.2 and not must \
                and self.rng.random() < g.fake_rate:
            self.do_fake()
            return
        self.release(ev, b)

    def do_fake(self):
        tc = self.cfg.throws
        self.last_fake = self.t
        self.fake_until = self.t + tc.fake_time
        dur = tc.fake_effect_time * (1.2 - 0.6 * self.R.marking[self.marker_id()]) if self.marker_id() >= 0 else 0.0
        self.fake_effect_until = self.fake_until + dur
        self.next_decision = self.fake_until
        self.res.fakes += 1
        self.event("fake", self.holder, f"{self.num(self.holder)} pivots and fakes")

    def release(self, ev, b):
        cfg = self.cfg
        h = self.holder
        ttype, curve, huck, is_break = int(ev["ttype"][b]), int(ev["curve"][b]), bool(ev["huck"][b]), bool(ev["is_break"][b])
        loft = bool(ev["loft"][b])
        rcv = int(ev["rcv"][b])
        intended = ev["tgt"][b].copy()
        rp = ev["rp"][b].copy()
        tname = D.THROW_TYPES[ttype]
        self.res.throws += 1
        self.res.hucks += huck
        self.res.breaks += is_break
        self.res.hammers += ttype == D.HAM
        label = f"{tname}{' huck' if huck else ''}{' break' if is_break else ''}"
        self.throw_log.append(dict(p=float(ev["p"][b]), p_mark=float(ev["p_mark"][b]), dist=float(ev["dist"][b]),
                                   ttype=ttype, huck=huck, is_break=is_break, loft=loft, stall=self.stall, outcome=None,
                                   best_p=float(ev["p"].max()), x=ev["X"][b].tolist()))
        # mark block?
        if ev["marked"] and self.rng.random() < ev["p_mark"][b]:
            u = (intended - rp) / max(np.hypot(*(intended - rp)), 1e-6)
            spot = self.clip_field(self.pos[h] + u * 1.5)
            self.event("mark_block", self.marker_id(), f"{self.num(self.marker_id())} blocks the {label} at the mark")
            if self.rec:
                ri = self.rec.throw(self.t, self.t + 0.2, h, rcv, tname, huck, is_break, curve, rp,
                                    0.5 * (rp + spot), spot, self.flipped)
                self.rec.throw_outcome(ri, "block")
            self.throw_log[-1]["outcome"] = "mark_block"
            self.turnover("mark_block", spot)
            return
        err = self.rng.normal(0.0, ev["sig"][b], 2)
        p1 = intended + err
        dist = float(np.hypot(*(p1 - rp)))
        T = float(D.flight_time(cfg, np.array([ttype]), np.array([dist]), np.array([huck]), np.array([loft]))[0])
        peak = float(D.throw_peak(cfg, np.array([ttype]), np.array([dist]), np.array([huck]), np.array([loft]))[0])
        ctrl = D.control_point(rp, p1, np.array(curve, float), cfg)
        lo, hi = D.unreachable_window(cfg, ttype, huck)
        self.flight = Flight(thrower=h, receiver=rcv, ttype=ttype, huck=huck, curve=curve, is_break=is_break,
                             p0=rp, c=ctrl, p1=p1, intended=intended, t0=self.t, T=T, lo=lo, hi=hi, dist=dist,
                             prev_pos=rp.copy(), s_end=1.0 + cfg.throws.tail_time / T, peak=peak, loft=loft,
                             h0=cfg.throws.height.hammer_release if ttype == D.HAM else cfg.throws.height.release)
        if self.rec:
            self.flight.rec_idx = self.rec.throw(self.t, self.t + T, h, rcv, tname, huck, is_break, curve,
                                                 rp, ctrl, p1, self.flipped)
        self.event("throw", h, f"{self.num(h)} throws a {label} to {self.num(rcv)} ({dist:.0f} m, p~{ev['p'][b]:.2f})")
        self.holder = -1
        self.stall = 0.0
        self.reset_player = -1
        g = self.ogen
        # thrower: give-go or clear; others: resets clear; re-slot around the intended catch point
        if self.rng.random() < g.give_go:
            if not hasattr(self, "cut_end"):
                self.cut_end = np.zeros((14, 2))
            self.mode[h] = CUT
            self.mtarget[h] = self.clip_field(self.pos[h] + self.dir_from_angle(15.0) * 10.0)
            self.muntil[h] = self.t + 2.0
        else:
            self.mode[h] = CLEAR
        for i in self.O:
            if self.mode[i] == RESET:
                self.mode[i] = CLEAR
        self.mode[rcv] = CHASE
        self.mtarget[rcv] = intended
        others = np.array([i for i in self.O if i != rcv])
        self.assign_slots(others, intended)

    # ------------------------------------------------------------------ disc flight
    def fpos(self, f: Flight, s):
        return D.flight_point(f.p0, f.c, f.p1, f.T, s, self.cfg.throws.tail_time)

    def flight_pos_now(self) -> np.ndarray:
        f = self.flight
        return self.fpos(f, min(max((self.t - f.t0) / f.T, 0.0), f.s_end))

    def update_chasers(self):
        """Players decide whether to go for the disc in the air and where to intercept it."""
        f = self.flight
        cfg = self.cfg
        since = self.t - f.t0
        s_now = since / f.T
        ss = np.linspace(min(s_now + 0.02, f.s_end), f.s_end, 18)
        pts = self.fpos(f, ss)
        td = ss * f.T - since
        hss = D.flight_height(cfg, f.peak, ss, f.T, f.h0)
        ok_base = ss * f.dist >= cfg.rules.double_team_radius

        def ok_for(i):   # reachable path points for player i (height envelope); the landing point always counts
            o = ok_base & (hss <= self.max_reach_h[i])
            o[-1] = True
            return o
        read_dl = cfg.movement.disc_read_delay + (cfg.movement.huck_read_extra if f.huck else 0.0)
        read = since >= read_dl
        # offense: intended receiver tracks the real disc after the read delay
        r = f.receiver
        if self.mode[r] == CHASE:
            if read:
                tr = time_to_reach(self.pos[[r]], self.vel[[r]], self.R.top_speed[[r]], self.R.accel[[r]],
                                   self.brake[[r]], self.reach[[r]], pts)[0]
                good = np.nonzero(ok_for(r) & (tr <= td))[0]
                self.mtarget[r] = pts[good[0]] if len(good) else pts[-1]
            else:
                self.mtarget[r] = f.intended
        # defense: help if they can get there in time
        Dd = self.Dd
        g = self.dgen
        if not read:
            return
        rd = self.defender_of(r)
        free = np.array([i for i in Dd if self.t >= self.ground_until[i]
                         and since >= self.R.reaction_time[i] + read_dl + self.help_delay(int(i), r)])
        if len(free) == 0:
            return
        td_def = time_to_reach(self.pos[free], self.vel[free], self.R.top_speed[free], self.R.accel[free],
                               self.brake[free], self.reach[free], pts)
        for j, i in enumerate(free):
            margin = g.help_margin + (0.6 if i == rd else 0.0)
            good = np.nonzero(ok_for(int(i)) & (td_def[j] <= td + margin))[0]
            if len(good):
                self.mode[i] = CHASE
                self.mtarget[i] = pts[good[0]]
            elif self.mode[i] == CHASE:
                self.mode[i] = self.rest_mode(int(i))

    def advance_flight(self):
        """Called after movement. Returns True if the flight ended this tick."""
        f = self.flight
        cfg = self.cfg
        s = (self.t - f.t0) / f.T
        s_c = min(s, f.s_end)
        cur = self.fpos(f, s_c)
        prev = f.prev_pos
        f.prev_pos = cur
        h_disc = float(D.flight_height(cfg, f.peak, s_c, f.T, f.h0))
        catchable = s_c * f.dist >= self.cfg.rules.double_team_radius
        if catchable:
            # swept test in each player's own frame (both disc and player move during the tick),
            # so a receiver attacking the disc can't tunnel through it between ticks
            P = self.pos
            P0 = self.hist[-1][0] if self.hist else P
            r0 = prev[None, :] - P0                      # disc relative to player at tick start
            r1 = cur[None, :] - P                        # ... and at tick end
            dr = r1 - r0
            dl2 = np.sum(dr * dr, axis=1)
            u = np.where(dl2 > 1e-9, np.clip(-np.sum(r0 * dr, axis=1) / np.maximum(dl2, 1e-9), 0.0, 1.0), 1.0)
            rel = r0 + u[:, None] * dr
            dmin = np.hypot(*rel.T)
            active = self.t >= self.ground_until
            can = h_disc <= self.max_reach_h                                   # within the reach envelope
            radius = self.reach * np.where(h_disc > self.standing, cfg.movement.jump_reach_mult, 1.0)
            touch = active & can & (dmin <= radius)
            for i in f.touched:
                touch[i] = False
            scale = np.full(14, cfg.throws.height.low_mult if h_disc < cfg.throws.height.low_disc else 1.0)
            # layout bids by chasers just out of reach as the disc passes / lands (not for discs overhead)
            ext = cfg.movement.layout_extension
            passing = (u < 1.0) | (s >= f.s_end)
            bidders = active & ~touch & (self.mode == CHASE) & (dmin <= self.reach + ext) & passing & (h_disc <= self.standing)
            for i in np.nonzero(bidders)[0]:
                if i in f.bid_done:
                    continue
                f.bid_done.add(int(i))
                aggr = self.og[i // 7].bid_aggr if (i // 7) == self.offense else self.dg[i // 7].bid_aggr
                if self.rng.random() < aggr:
                    touch[i] = True
                    scale[i] *= cfg.movement.layout_success_scale * (1.0 - 0.5 * (dmin[i] - self.reach[i]) / ext)
                    self.ground_until[i] = self.t + cfg.movement.layout_ground_time
                    self.mode[i] = GROUND
                    self.event("layout", int(i), f"{self.num(int(i))} lays out!")
            if touch.any():
                return self.resolve_touch(touch, dmin, scale, cur)
        if s >= f.s_end:
            p1 = cur
            out = not (0.0 <= p1[0] <= self.L and 0.0 <= p1[1] <= self.W)
            self.close_throw("out_of_bounds" if out else "throwaway")
            self.event("out_of_bounds" if out else "throwaway", f.thrower,
                       f"{'out of bounds' if out else 'throw lands untouched'} - turnover")
            self.turnover("out_of_bounds" if out else "throwaway", p1)
            return True
        return False

    def close_throw(self, outcome):
        if self.throw_log and self.throw_log[-1]["outcome"] is None:
            self.throw_log[-1]["outcome"] = outcome
        if self.rec and self.flight and self.flight.rec_idx >= 0:
            self.rec.throw_outcome(self.flight.rec_idx, outcome, self.t)

    def resolve_touch(self, touch, dmin, scale, cur) -> bool:
        cfg = self.cfg
        f = self.flight
        O_t = [i for i in self.O if touch[i]]
        D_t = [i for i in self.Dd if touch[i]]
        # a defender who has not yet read the throw only gets a reflex chance at it
        since = self.t - f.t0
        for d in D_t:
            huck_dl = cfg.movement.huck_read_extra if f.huck else 0.0
            if since < self.R.reaction_time[d] + cfg.movement.disc_read_delay + huck_dl + self.help_delay(int(d), f.receiver):
                scale[d] *= cfg.defense.reflex_block
        near_def = bool(np.any(np.hypot(*(self.pos[self.Dd] - cur).T) < 2.0))
        vdisc = f.dist / f.T

        def try_catch(i, contested: bool) -> bool:
            p = self.R.hands[i] - cfg.catching.speed_drop_per_ms * max(vdisc - 12.0, 0.0)
            if contested or near_def:
                p -= cfg.catching.contest_drop
            if f.huck:
                p -= cfg.catching.huck_drop  # over-the-shoulder catch at full sprint
            if f.ttype == D.HAM:
                p -= cfg.catching.hammer_drop  # upside-down disc dropping in
            return self.rng.random() < p * scale[i]

        def best(ids):
            return min(ids, key=lambda i: dmin[i] - 0.5 * (i == f.receiver))

        if D_t and O_t:
            o, d = best(O_t), best(D_t)
            logit = (cfg.defense.contest_base + cfg.defense.contest_height_weight * (self.eff_height[o] - self.eff_height[d])
                     + cfg.defense.contest_position_weight * (1.0 if dmin[o] < dmin[d] else -1.0))
            if self.rng.random() < _sig(logit):
                if try_catch(o, True):
                    return self.catch(o, cur)
                return self.drop(o, cur)
            if self.rng.random() < cfg.defense.block_prob * scale[d]:
                return self.block(d, cur, contested=True)
            if try_catch(o, True):
                return self.catch(o, cur)
            return self.drop(o, cur)
        if D_t:
            d = best(D_t)
            if self.rng.random() < cfg.defense.block_prob * scale[d]:
                return self.block(d, cur, contested=False)
            f.touched.add(int(d))
            return False  # got a fingertip on it; the disc flies on
        o = best(O_t)
        if try_catch(o, False):
            return self.catch(o, cur)
        return self.drop(o, cur)

    def catch(self, i, where) -> bool:
        f = self.flight
        p = self.pos[i].copy()
        self.res.completions += 1
        self.res.huck_completions += f.huck
        self.res.break_completions += f.is_break
        inb = 0.0 <= p[0] <= self.L and 0.0 <= p[1] <= self.W
        if not inb:
            self.close_throw("out_of_bounds")
            self.event("out_of_bounds", i, f"{self.num(i)} catches it out of bounds - turnover")
            self.turnover("out_of_bounds", p)
            return True
        if p[0] >= self.L - self.ez:
            self.close_throw("score")
            self.event("score", i, f"SCORE! {self.num(i)} catches it in the end zone")
            self.end_possession("score", p)
            return True
        self.close_throw("catch")
        self.event("catch", i, f"{self.num(i)} catches ({D.THROW_TYPES[f.ttype]}, {f.dist:.0f} m{', break' if f.is_break else ''})")
        self.flight = None
        self.holder = i
        if p[0] < self.ez:
            p[0] = self.ez  # walked up to the goal line
            self.pos[i] = p
        self.disc_pos = p
        self.s_open = self.compute_s_open(p[1])
        self.stall = 0.0
        self.mode[i] = HOLD
        self.ready_time = self.t + self.R.release_time[i]
        self.next_decision = self.ready_time
        self.reset_player = -1
        for j in self.O:
            if j != i and self.mode[j] == CHASE:
                self.mode[j] = CLEAR
        for j in self.Dd:
            if self.mode[j] == CHASE:
                self.mode[j] = self.rest_mode(int(j))
        others = np.array([j for j in self.O if j != i])
        self.assign_slots(others, p)
        if self.k_eff >= 1:   # zone and flex re-form around the new thrower ...
            g = self.dgen
            if g.transition_throws < 9.5 and self.res.completions >= int(math.floor(g.transition_throws + 0.5)):
                self.k_eff = 0   # ... or, after N completions, switch to person D (zone-to-person flex)
            self.assign_defense(by_distance=True)
            for j in self.Dd:
                if self.mode[j] in (ZONE, DEF):
                    self.mode[j] = self.rest_mode(int(j))
        return False

    def drop(self, i, where) -> bool:
        self.close_throw("drop")
        self.event("drop", i, f"{self.num(i)} drops it - turnover")
        self.turnover("drop", where)
        return True

    def block(self, d, where, contested) -> bool:
        f = self.flight
        kind = "block" if contested or np.hypot(*(self.pos[f.receiver] - where)) < 3.0 else "interception"
        self.close_throw(kind)
        self.event(kind, d, f"{self.num(d)} gets the D ({kind})!")
        self.turnover(kind, where)
        return True

    def turnover(self, kind: str, where):
        self.end_possession(kind, np.asarray(where, float))

    def end_possession(self, outcome: str, where):
        self.res.outcome = outcome
        self.res.duration = self.t - self.poss_t0
        self.res.end_pos = (float(where[0]), float(where[1]))
        self.flight = None
        self.holder = -1
        if self.rec:
            self.rec.possession(self.poss_t0, self.t, self.offense, self.dgen.force_name, outcome)

    # ------------------------------------------------------------------ main loop
    def tick(self):
        cfg = self.cfg
        # 1. thrower
        if self.holder >= 0:
            if self.t >= self.ready_time and self.t >= self.next_decision and self.t >= self.fake_until:
                self.next_decision = self.t + cfg.sim.decision_interval
                self.thrower_decision()
                if self.res.outcome:
                    return
        # 2. behaviour
        tgt = self.pos.copy()
        spd = np.zeros(14)
        codes = np.zeros(14, int)
        if self.flight is not None:
            self.update_chasers()
        self.maybe_switch()
        tvel = np.zeros((14, 2))
        self.offense_targets(tgt, spd, codes)
        self.defense_targets(tgt, spd, codes, tvel)
        # 3. movement
        frozen = self.t < self.ground_until
        self.hist.append((self.pos.copy(), self.vel.copy()))
        self.pos, self.vel = step(self.pos, self.vel, tgt, spd, self.R.accel, self.brake, self.R.top_speed,
                                  self.dt, frozen, target_vel=tvel)
        s = cfg.movement.boundary_slack
        self.pos[:, 0] = np.clip(self.pos[:, 0], -s, self.L + s)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -s, self.W + s)
        if self.holder >= 0:
            self.disc_pos = self.pos[self.holder].copy()
        self.t += self.dt
        # 4. disc
        if self.flight is not None:
            if self.advance_flight():
                self.record(codes)
                return
        # 5. stall
        if self.holder >= 0:
            if self.t >= self.grace_until and self.marker_set():
                self.stall += cfg.rules.stall_rate * self.dt
            self.res.max_stall = max(self.res.max_stall, self.stall)
            if self.stall >= cfg.rules.stall_max:
                mk = self.marker_id()
                self.event("stall", mk, f"Stall 10 - {self.num(self.holder)} is stalled out")
                self.turnover("stall", self.disc_pos)
        if self.t - self.poss_t0 > cfg.sim.max_possession_time and not self.res.outcome:
            self.event("timeout", -1, "possession time cap reached")
            self.turnover("timeout", self.disc_pos if self.flight is None else self.flight.p1)
        self.record(codes)

    def record(self, codes):
        if not self.rec:
            return
        if self.flight is not None:
            f = self.flight
            s = min(max((self.t - f.t0) / f.T, 0.0), f.s_end)
            dp = self.fpos(f, s)
            dz = float(D.flight_height(self.cfg, f.peak, s, f.T, f.h0))
            ds = 1
        else:
            dp = self.disc_pos
            dz = 1.0 if self.holder >= 0 else 0.0
            ds = 0 if self.holder >= 0 else 2
        mark = self.mark_geometry() if (self.holder >= 0 and self.marker_set()) else None
        stall = self.stall if self.holder >= 0 else None
        self.rec.frame(self.pos, codes, np.array([dp[0], dp[1], dz]), ds, self.holder, self.offense, stall,
                       mark, self.flipped)

    def run_possession(self, disc_xy=None, fresh=True) -> PossessionResult:
        if disc_xy is None:
            m = self.cfg.sim.start_y_margin
            disc_xy = (self.cfg.sim.start_x, self.rng.uniform(m, self.W - m))
        self.start_possession(np.array(disc_xy, float), fresh)
        codes = np.where(np.isin(np.arange(14), self.O), SET, DEF)
        codes[self.holder] = HOLD
        self.record(codes)
        while not self.res.outcome:
            self.tick()
        return self.res

    def flip(self):
        """Rotate the state 180 degrees and hand possession to the other team."""
        L, W = self.L, self.W
        self.pos = np.stack([L - self.pos[:, 0], W - self.pos[:, 1]], axis=1)
        self.vel = -self.vel
        self.hist = deque(((np.stack([L - p[:, 0], W - p[:, 1]], axis=1), -v) for p, v in self.hist),
                          maxlen=self.hist.maxlen)
        self.flipped = not self.flipped
        self.offense = 1 - self.offense

    def run_point(self) -> list[PossessionResult]:
        """Full point: possessions alternate on turnovers until someone scores."""
        results = [self.run_possession(fresh=True)]
        while results[-1].outcome != "score" and len(results) < self.cfg.sim.max_point_possessions:
            ex, ey = results[-1].end_pos
            self.flip()
            nx = float(np.clip(self.L - ex, self.ez, self.L - self.ez - 0.01))
            ny = float(np.clip(self.W - ey, 0.5, self.W - 0.5))
            results.append(self.run_possession((nx, ny), fresh=False))
        return results

    def result_summary(self, results: list[PossessionResult]) -> dict:
        last = results[-1]
        return {"outcome": last.outcome, "team": int(last.offense), "duration": round(self.t, 2),
                "throws": int(sum(r.throws for r in results)), "turnovers": int(sum(r.outcome != "score" for r in results))}


# ---------------------------------------------------------------------- convenience API
def simulate_possession(cfg, roster_off, roster_def, og, dg, seed=0, record=False, title="", value_field=None):
    g = Game(cfg, (roster_off, roster_def), (og, None), (None, dg), seed=seed, record=record, title=title,
             mode="possession", value_field=value_field)
    res = g.run_possession()
    if g.rec:
        g.rec.finish(g.result_summary([res]))
    return res, g.rec


def simulate_point(cfg, rosters, ogs, dgs, seed=0, record=False, title="", value_field=None):
    g = Game(cfg, rosters, ogs, dgs, seed=seed, record=record, title=title, mode="point", value_field=value_field)
    results = g.run_point()
    if g.rec:
        g.rec.finish(g.result_summary(results))
    return results, g.rec
