"""Strategy genomes (offense / defense), flat-vector schema, and canonical seed strategies.

Coordinates in genomes use the FORCE-AWARE frame:
  dx = metres downfield from the disc (negative = behind the disc)
  dy = metres toward the OPEN side (negative = break side)
  angles: 0 = straight downfield, +90 = toward the open side, 180 = back upfield, -90 = break side
So one genome works whichever way the disc is forced; the engine mirrors it per throw.

Offense: 6 formation slots (the 6 non-throwers). After each throw, players are re-matched to
slots (Hungarian assignment), so roles flow the way real stacks do.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields

import numpy as np

N_SLOTS = 6
FORCES = ("forehand", "backhand", "straight_up", "middle", "sideline")


def _a(*v) -> np.ndarray:
    return np.array(v, dtype=float)


@dataclass
class OffenseGenome:
    name: str = "offense"
    # --- formation slots (6,) ---
    home_dx: np.ndarray = field(default_factory=lambda: np.zeros(N_SLOTS))
    home_dy: np.ndarray = field(default_factory=lambda: np.zeros(N_SLOTS))
    y_anchor: np.ndarray = field(default_factory=lambda: np.zeros(N_SLOTS))   # 0 disc-relative y, 1 field-centre-relative y
    handler: np.ndarray = field(default_factory=lambda: np.zeros(N_SLOTS))    # >0.5 = handler slot
    # --- cutting (6,) ---
    cutA_ang: np.ndarray = field(default_factory=lambda: np.full(N_SLOTS, 150.0))
    cutA_len: np.ndarray = field(default_factory=lambda: np.full(N_SLOTS, 10.0))
    cutB_ang: np.ndarray = field(default_factory=lambda: np.full(N_SLOTS, 10.0))
    cutB_len: np.ndarray = field(default_factory=lambda: np.full(N_SLOTS, 22.0))
    cutA_bias: np.ndarray = field(default_factory=lambda: np.zeros(N_SLOTS))  # logit preference for cut A
    read_def: np.ndarray = field(default_factory=lambda: np.ones(N_SLOTS))    # weight: cut away from the defender
    cut_delay: np.ndarray = field(default_factory=lambda: np.ones(N_SLOTS))   # s after thrower ready
    recut_delay: np.ndarray = field(default_factory=lambda: np.full(N_SLOTS, 2.5))
    jab: np.ndarray = field(default_factory=lambda: np.full(N_SLOTS, 0.4))    # probability of a jab step first
    seek_space: np.ndarray = field(default_factory=lambda: np.zeros(N_SLOTS))  # bend cuts toward open space (0..1)
    # --- team-level ---
    reset_stall: float = 6.0
    reset_dx: float = -3.0
    reset_dy: float = -5.0
    give_go: float = 0.3
    risk_aversion: float = 1.0
    huck_bias: float = 0.0
    break_bias: float = 0.0
    hammer_bias: float = -0.05
    patience: float = 1.05
    panic_stall: float = 8.5
    fake_rate: float = 0.3
    lead: float = 1.5
    bid_aggr: float = 0.5


@dataclass
class DefenseGenome:
    name: str = "defense"
    zone: float = 0.0             # share of the 7 defenders playing zone: round(zone * 7) = k.
                                  # k = 0 person D, k = 7 full zone, 1..6 flex (marker + k-1 zone spots,
                                  # the other 7-k play person on the cutters picked by flex_focus)
    force: float = 0.0            # index into FORCES
    mark_dist: float = 1.6
    mark_angle: float = 55.0      # marker stands this many degrees from downfield toward the break side
    mark_width: float = 50.0      # DEPRECATED (not in the schema): coverage now follows from mark_dist
    # person D
    cushion_base: float = 2.0
    cushion_per_m: float = 0.06
    angle_near: float = 65.0      # offset angle vs. attacker, 0 = between attacker and disc, 180 = deep side,
                                  # +90 = open side. Default: shade the open side (club-standard person D)
    angle_far: float = 150.0
    anticipation: float = 0.2     # 0..0.6: how much defenders extrapolate their (delayed) read
    poach: float = 0.1            # sag into lanes off non-moving attackers
    help_margin: float = 0.25     # s of slack when deciding to go for a disc in the air
    bid_aggr: float = 0.5
    match_speed: float = 0.5      # matchup cost weight on speed mismatch
    # zone (6 non-marker slots)
    zone_dx: np.ndarray = field(default_factory=lambda: _a(3.0, 3.0, 12.0, 14.0, 14.0, 30.0))
    zone_dy: np.ndarray = field(default_factory=lambda: _a(4.0, -4.0, 0.0, 12.0, -12.0, 0.0))
    zone_shade: float = 0.3
    # team defense
    switch: float = 0.0           # 0 never switch; 1 always re-match to the nearest attacker (clam)
    help_cut: float = 0.0         # how far a free defender steps into the lane of a cutter who has beaten their man
    stack_sag: float = 0.0        # defenders of stationary attackers in a cluster sag into the stack
    flex_focus: float = 1.0       # flex: person defenders take the deepest (1) or the closest-to-disc (0) cutters
    transition_throws: float = 10.0  # zone/flex switch to person D after this many completions (>= 9.5: never)
    trap: float = 0.0             # near a sideline the short zone players collapse on the thrower

    @property
    def n_zone(self) -> int:
        return int(np.floor(float(np.clip(self.zone, 0.0, 1.0)) * 7 + 0.5))   # same rounding as the C engine

    @property
    def force_name(self) -> str:
        return FORCES[int(np.clip(self.force, 0, len(FORCES) - 1e-9))]

    @property
    def is_zone(self) -> bool:
        return self.n_zone == 7

    @property
    def is_flex(self) -> bool:
        return 0 < self.n_zone < 7


# ---------------------------------------------------------------------------
# Flat-vector schema (for the GA and, later, the C ABI): (name, size, lo, hi)
OFFENSE_SCHEMA = [
    ("home_dx", N_SLOTS, -8.0, 35.0), ("home_dy", N_SLOTS, -16.0, 16.0), ("y_anchor", N_SLOTS, 0.0, 1.0),
    ("handler", N_SLOTS, 0.0, 1.0), ("cutA_ang", N_SLOTS, -180.0, 180.0), ("cutA_len", N_SLOTS, 3.0, 35.0),
    ("cutB_ang", N_SLOTS, -180.0, 180.0), ("cutB_len", N_SLOTS, 3.0, 35.0), ("cutA_bias", N_SLOTS, -3.0, 3.0),
    ("read_def", N_SLOTS, 0.0, 3.0), ("cut_delay", N_SLOTS, 0.0, 8.0), ("recut_delay", N_SLOTS, 0.5, 6.0),
    ("jab", N_SLOTS, 0.0, 1.0),
    ("reset_stall", 1, 2.0, 9.0), ("reset_dx", 1, -8.0, 4.0), ("reset_dy", 1, -12.0, 12.0),
    ("give_go", 1, 0.0, 1.0), ("risk_aversion", 1, 0.0, 3.0), ("huck_bias", 1, -0.3, 0.3),
    ("break_bias", 1, -0.3, 0.3), ("hammer_bias", 1, -0.3, 0.3), ("patience", 1, 0.6, 1.6),
    ("panic_stall", 1, 6.0, 9.5), ("fake_rate", 1, 0.0, 1.0), ("lead", 1, 0.0, 5.0), ("bid_aggr", 1, 0.0, 1.0),
    ("seek_space", N_SLOTS, 0.0, 1.0),
]
DEFENSE_SCHEMA = [
    ("zone", 1, 0.0, 1.0), ("force", 1, 0.0, len(FORCES) - 1e-6), ("mark_dist", 1, 0.9, 3.0),
    ("mark_angle", 1, 0.0, 110.0), ("cushion_base", 1, 0.3, 5.0),
    ("cushion_per_m", 1, 0.0, 0.25), ("angle_near", 1, -180.0, 180.0), ("angle_far", 1, -180.0, 180.0),
    ("anticipation", 1, 0.0, 0.6), ("poach", 1, 0.0, 1.0), ("help_margin", 1, -0.3, 1.0),
    ("bid_aggr", 1, 0.0, 1.0), ("match_speed", 1, 0.0, 2.0),
    ("zone_dx", N_SLOTS, -6.0, 40.0), ("zone_dy", N_SLOTS, -18.0, 18.0), ("zone_shade", 1, 0.0, 1.0),
    ("switch", 1, 0.0, 1.0), ("help_cut", 1, 0.0, 1.0), ("stack_sag", 1, 0.0, 1.0), ("flex_focus", 1, 0.0, 1.0),
    ("transition_throws", 1, 0.0, 10.0), ("trap", 1, 0.0, 1.0),
]


def schema_bounds(schema) -> tuple[np.ndarray, np.ndarray]:
    lo = np.concatenate([np.full(n, l) for _, n, l, _ in schema])
    hi = np.concatenate([np.full(n, h) for _, n, _, h in schema])
    return lo, hi


def to_vector(g, schema) -> np.ndarray:
    return np.concatenate([np.atleast_1d(np.asarray(getattr(g, k), dtype=float)) for k, _, _, _ in schema])


def from_vector(vec: np.ndarray, schema, cls, name: str = "evolved"):
    g = cls(name=name)
    i = 0
    for k, n, lo, hi in schema:
        v = np.clip(vec[i:i + n], lo, hi)
        setattr(g, k, v.copy() if n > 1 else float(v[0]))
        i += n
    return g


def clone(g, name: str | None = None):
    c = copy.deepcopy(g)
    if name:
        c.name = name
    return c


# ---------------------------------------------------------------------------
# Canonical offenses
def vertical_stack() -> OffenseGenome:
    return OffenseGenome(
        name="vertical",
        # slot: 0 dump handler, 1 open-side handler, 2-5 the stack (front .. back)
        home_dx=_a(-2.5, -0.5, 12.0, 16.0, 20.0, 24.0),
        home_dy=_a(-7.0, 8.0, 0.0, 0.0, 0.0, 0.0),
        y_anchor=_a(0, 0, 1, 1, 1, 1),
        handler=_a(1, 1, 0, 0, 0, 0),
        cutA_ang=_a(35.0, 25.0, 140.0, 135.0, 140.0, 135.0),   # handlers upline; cutters under to the open side
        cutA_len=_a(8.0, 10.0, 12.0, 12.0, 13.0, 13.0),
        cutB_ang=_a(170.0, 165.0, 10.0, 15.0, 5.0, 10.0),      # handlers back for the dump; cutters deep
        cutB_len=_a(4.0, 5.0, 24.0, 22.0, 22.0, 20.0),
        cutA_bias=_a(0.0, 0.5, 1.0, 0.3, 0.0, -0.7),
        read_def=_a(1.0, 1.0, 1.2, 1.2, 1.2, 1.2),
        cut_delay=_a(3.0, 4.0, 0.6, 2.6, 1.4, 3.6),
        recut_delay=_a(2.0, 2.0, 2.6, 2.6, 2.6, 2.6),
        jab=_a(0.3, 0.3, 0.5, 0.5, 0.5, 0.5),
        reset_stall=6.0, reset_dx=-3.0, reset_dy=-5.0,
    )


def horizontal_stack() -> OffenseGenome:
    return OffenseGenome(
        name="horizontal",
        # slot: 0,1 handlers either side of the thrower; 2-5 cutters spread across ~16 m downfield
        home_dx=_a(0.5, -1.5, 16.0, 17.0, 17.0, 16.0),
        home_dy=_a(10.0, -9.0, -13.0, -4.5, 4.5, 13.0),
        y_anchor=_a(0.3, 0.3, 1, 1, 1, 1),
        handler=_a(1, 1, 0, 0, 0, 0),
        cutA_ang=_a(25.0, 35.0, 150.0, 165.0, 165.0, 150.0),
        cutA_len=_a(10.0, 8.0, 10.0, 11.0, 11.0, 10.0),
        cutB_ang=_a(165.0, 175.0, 5.0, -15.0, 15.0, 5.0),
        cutB_len=_a(5.0, 4.0, 22.0, 22.0, 22.0, 22.0),
        cutA_bias=_a(0.5, 0.0, -0.6, 0.8, 0.8, -0.6),          # inside cutters under, outside cutters deep
        read_def=_a(1.0, 1.0, 1.3, 1.3, 1.3, 1.3),
        cut_delay=_a(3.5, 3.0, 2.2, 0.6, 1.4, 3.0),
        recut_delay=_a(2.0, 2.0, 2.8, 2.8, 2.8, 2.8),
        jab=_a(0.3, 0.3, 0.5, 0.5, 0.5, 0.5),
        reset_stall=5.5, reset_dx=-2.5, reset_dy=-6.0,
    )


def side_stack() -> OffenseGenome:
    return OffenseGenome(
        name="side",
        # stack along the break-side line, cutting into the open field
        home_dx=_a(-2.5, -0.5, 10.0, 14.0, 18.0, 22.0),
        home_dy=_a(-7.0, 8.0, -12.0, -12.0, -12.0, -12.0),
        y_anchor=_a(0, 0, 1, 1, 1, 1),
        handler=_a(1, 1, 0, 0, 0, 0),
        cutA_ang=_a(35.0, 25.0, 125.0, 120.0, 115.0, 110.0),
        cutA_len=_a(8.0, 10.0, 14.0, 15.0, 16.0, 16.0),
        cutB_ang=_a(170.0, 165.0, 25.0, 25.0, 20.0, 15.0),
        cutB_len=_a(4.0, 5.0, 24.0, 24.0, 22.0, 20.0),
        cutA_bias=_a(0.0, 0.5, 0.8, 0.3, 0.0, -0.5),
        cut_delay=_a(3.0, 4.0, 0.6, 2.4, 1.4, 3.4),
        reset_stall=6.0, reset_dx=-3.0, reset_dy=-5.0,
    )


def iso() -> OffenseGenome:
    return OffenseGenome(
        name="iso",
        # one isolated cutter in the middle of the field; the rest clear out deep on the break side
        home_dx=_a(-2.5, -0.5, 15.0, 26.0, 29.0, 32.0),
        home_dy=_a(-7.0, 8.0, 0.0, -13.0, -13.0, -13.0),
        y_anchor=_a(0, 0, 1, 1, 1, 1),
        handler=_a(1, 1, 0, 0, 0, 0),
        cutA_ang=_a(35.0, 25.0, 145.0, 150.0, 150.0, 150.0),
        cutA_len=_a(8.0, 10.0, 12.0, 12.0, 12.0, 12.0),
        cutB_ang=_a(170.0, 165.0, 10.0, 20.0, 20.0, 20.0),
        cutB_len=_a(4.0, 5.0, 24.0, 15.0, 15.0, 15.0),
        cutA_bias=_a(0.0, 0.5, 0.6, 0.5, 0.5, 0.5),
        read_def=_a(1.0, 1.0, 1.6, 1.0, 1.0, 1.0),
        cut_delay=_a(3.0, 4.0, 0.3, 6.0, 7.0, 8.0),
        recut_delay=_a(2.0, 2.0, 1.5, 4.0, 4.0, 4.0),
        reset_stall=6.0, reset_dx=-3.0, reset_dy=-5.0,
    )


def zone_offense() -> OffenseGenome:
    return OffenseGenome(
        name="zone_o",
        # 0,1 swing handlers either side of the thrower (3 handlers incl. thrower); 2,3 poppers in the seams
        # just behind the cup; 4,5 deep wings
        home_dx=_a(-1.0, -2.0, 8.0, 9.0, 24.0, 24.0),
        home_dy=_a(11.0, -11.0, 4.5, -4.5, 12.0, -12.0),
        y_anchor=_a(0.3, 0.3, 0.0, 0.0, 1.0, 1.0),
        handler=_a(1, 1, 0, 0, 0, 0),
        cutA_ang=_a(20.0, -20.0, 60.0, -60.0, 160.0, -160.0),   # handlers step up; poppers slide into seams; wings come under
        cutA_len=_a(5.0, 5.0, 6.0, 6.0, 10.0, 10.0),
        cutB_ang=_a(170.0, -170.0, 120.0, -120.0, 10.0, -10.0),  # handlers drop back for the swing; poppers come back; wings go deep
        cutB_len=_a(4.0, 4.0, 5.0, 5.0, 15.0, 15.0),
        cutA_bias=_a(0.0, 0.0, 0.5, 0.5, 0.3, 0.3),
        read_def=_a(1.5, 1.5, 2.0, 2.0, 1.2, 1.2),
        cut_delay=_a(1.0, 1.5, 0.5, 1.0, 2.5, 3.0),
        recut_delay=_a(1.5, 1.5, 1.2, 1.2, 2.5, 2.5),
        jab=_a(0.2, 0.2, 0.4, 0.4, 0.4, 0.4),
        reset_stall=4.0, reset_dx=-3.0, reset_dy=-6.0, risk_aversion=1.2, patience=1.0, hammer_bias=0.0,
    )


CANONICAL_OFFENSES = {"vertical": vertical_stack, "horizontal": horizontal_stack, "side": side_stack, "iso": iso,
                      "zone_o": zone_offense}


# ---------------------------------------------------------------------------
# Canonical defenses
def person(force: str) -> DefenseGenome:
    g = DefenseGenome(name=f"person_{force}", force=float(FORCES.index(force)))
    if force == "straight_up":
        g.mark_angle, g.mark_width = 0.0, 40.0
    return g


def cup_zone() -> DefenseGenome:
    return DefenseGenome(name="cup_zone", zone=1.0, force=float(FORCES.index("sideline")),
                         mark_dist=1.8, mark_angle=40.0, mark_width=55.0,
                         zone_dx=_a(3.5, 3.0, 12.0, 15.0, 15.0, 32.0),
                         zone_dy=_a(4.5, -4.0, 0.0, 11.0, -11.0, 0.0), zone_shade=0.25)


def wall_zone() -> DefenseGenome:
    # 1-3-3 style "wall": marker, a flat wall of three ~8 m off, two deeper wings + deep
    return DefenseGenome(name="wall_zone", zone=1.0, force=float(FORCES.index("middle")),
                         mark_dist=1.6, mark_angle=60.0, mark_width=50.0,
                         zone_dx=_a(8.0, 8.0, 8.0, 20.0, 20.0, 33.0),
                         zone_dy=_a(7.0, -7.0, 0.0, 9.0, -9.0, 0.0), zone_shade=0.35)


def clam() -> DefenseGenome:
    # person D that switches on crossing cuts and helps a beaten teammate
    g = person("forehand")
    g.name, g.switch, g.help_cut = "clam", 0.8, 0.5
    return g


def help_d() -> DefenseGenome:
    # person D that sags into the stack and steps into the lane of an open cutter
    g = person("forehand")
    g.name, g.help_cut, g.stack_sag, g.poach = "help_d", 0.8, 0.4, 0.2
    return g


def flex_deep() -> DefenseGenome:
    # marker + one deep zone player in the middle; person D on the 5 cutters closest to the disc
    g = DefenseGenome(name="flex_deep", zone=2 / 7, force=float(FORCES.index("forehand")), flex_focus=0.0,
                      zone_dx=_a(30.0, 12.0, 14.0, 14.0, 3.0, 3.0), zone_dy=_a(0.0, 0.0, 11.0, -11.0, 4.0, -4.0))
    return g


def junk_cup() -> DefenseGenome:
    # marker + 2 cup players around the disc; person D on the 4 deepest cutters
    g = DefenseGenome(name="junk_cup", zone=3 / 7, force=float(FORCES.index("middle")), flex_focus=1.0,
                      mark_dist=1.8, mark_angle=40.0,
                      zone_dx=_a(3.5, 3.0, 12.0, 15.0, 15.0, 30.0), zone_dy=_a(4.5, -4.0, 0.0, 11.0, -11.0, 0.0))
    return g


def zone_to_person() -> DefenseGenome:
    # "flex" in the time sense: start in a cup zone, switch to person D after 3 completions
    g = cup_zone()
    g.name, g.transition_throws = "zone_to_person", 3.0
    return g


def cup_trap() -> DefenseGenome:
    # cup zone forcing sideline that collapses into a trap near the line
    g = cup_zone()
    g.name, g.trap, g.force = "cup_trap", 0.8, float(FORCES.index("sideline"))
    return g


CANONICAL_DEFENSES = {
    "person_forehand": lambda: person("forehand"),
    "person_backhand": lambda: person("backhand"),
    "person_straight_up": lambda: person("straight_up"),
    "person_middle": lambda: person("middle"),
    "person_sideline": lambda: person("sideline"),
    "cup_zone": cup_zone,
    "wall_zone": wall_zone,
    "clam": clam,
    "help_d": help_d,
    "flex_deep": flex_deep,
    "junk_cup": junk_cup,
    "zone_to_person": zone_to_person,
    "cup_trap": cup_trap,
}
