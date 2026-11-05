import json

import numpy as np
import pytest

from ultisim import disc as D
from ultisim.config import load_config
from ultisim.engine_ref import TURNOVER_TYPES, Game, simulate_point, simulate_possession
from ultisim.genome import (CANONICAL_DEFENSES, CANONICAL_OFFENSES, DEFENSE_SCHEMA, OFFENSE_SCHEMA, DefenseGenome,
                            OffenseGenome, from_vector, to_vector, vertical_stack)
from ultisim.movement import step, time_to_reach
from ultisim.roster import Roster, sample_roster

CFG = load_config()


def rosters(seed=0):
    rng = np.random.default_rng(seed)
    return sample_roster(rng, CFG), sample_roster(rng, CFG)


# ---------------------------------------------------------------- config / rosters / genomes
def test_config_overrides():
    c = load_config(overrides={"players": {"top_speed": {"mean": 8.0}}})
    assert c.players.top_speed.mean == 8.0 and c.players.top_speed.sd == CFG.players.top_speed.sd


def test_roster_ranges_and_correlation():
    rng = np.random.default_rng(1)
    rs = [sample_roster(rng, CFG) for _ in range(300)]
    for name in Roster.stat_names():
        v = np.concatenate([getattr(r, name) for r in rs])
        d = getattr(CFG.players, name)
        assert v.min() >= d.lo - 1e-9 and v.max() <= d.hi + 1e-9, name
    sp = np.concatenate([r.top_speed for r in rs])
    ac = np.concatenate([r.accel for r in rs])
    th = np.concatenate([r.throw_skill for r in rs])
    assert np.corrcoef(sp, ac)[0, 1] > 0.15          # athletic factor
    assert abs(np.corrcoef(sp, th)[0, 1]) < 0.1      # independent factors


def test_genome_vector_roundtrip():
    g = vertical_stack()
    v = to_vector(g, OFFENSE_SCHEMA)
    g2 = from_vector(v, OFFENSE_SCHEMA, OffenseGenome)
    assert np.allclose(to_vector(g2, OFFENSE_SCHEMA), v)
    d = CANONICAL_DEFENSES["cup_zone"]()
    assert np.allclose(to_vector(from_vector(to_vector(d, DEFENSE_SCHEMA), DEFENSE_SCHEMA, DefenseGenome),
                                 DEFENSE_SCHEMA), to_vector(d, DEFENSE_SCHEMA))


# ---------------------------------------------------------------- physics
def test_movement_respects_speed_and_arrives():
    pos = np.zeros((3, 2)); vel = np.zeros((3, 2))
    tgt = np.array([[30.0, 0.0], [0.0, 5.0], [-10.0, -10.0]])
    top = np.array([7.0, 8.0, 9.0]); acc = np.full(3, 4.0); brk = np.full(3, 6.0)
    vmax = 0.0
    for _ in range(200):
        pos, vel = step(pos, vel, tgt, top, acc, brk, top, 0.1)
        vmax = max(vmax, float(np.max(np.hypot(*vel.T) - top)))
    assert vmax <= 1e-9
    assert np.allclose(pos, tgt, atol=0.3)


def test_sharp_cut_costs_time():
    """Reversing direction at full speed must take longer than starting from rest."""
    top, acc, brk = np.array([8.0]), np.array([4.0]), np.array([6.0])
    t_rest = time_to_reach(np.zeros((1, 2)), np.zeros((1, 2)), top, acc, brk, np.zeros(1), np.array([[-10.0, 0.0]]))
    t_rev = time_to_reach(np.zeros((1, 2)), np.array([[8.0, 0.0]]), top, acc, brk, np.zeros(1), np.array([[-10.0, 0.0]]))
    assert t_rev[0, 0] > t_rest[0, 0] + 1.0


def test_time_to_reach_zero_inside_reach_and_monotonic():
    pts = np.array([[0.5, 0.0], [5.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    t = time_to_reach(np.zeros((1, 2)), np.zeros((1, 2)), np.array([8.0]), np.array([4.0]), np.array([6.0]),
                      np.array([1.0]), pts, delay=np.array([0.3]))[0]
    assert t[0] == 0.0 and np.all(np.diff(t) > 0)


def test_flight_path_endpoints_and_tail_continuity():
    p0, p1 = np.array([10.0, 10.0]), np.array([30.0, 20.0])
    c = D.control_point(p0, p1, np.array(1.0), CFG)
    assert np.allclose(D.flight_point(p0, c, p1, 1.5, 0.0, 0.35), p0)
    assert np.allclose(D.flight_point(p0, c, p1, 1.5, 1.0, 0.35), p1)
    a = D.flight_point(p0, c, p1, 1.5, 0.999, 0.35)
    b = D.flight_point(p0, c, p1, 1.5, 1.001, 0.35)
    assert np.hypot(*(a - b)) < 0.1


def test_mark_blocks_break_side_not_open_side():
    hp = np.array([40.0, 18.0])
    center = -55.0  # force forehand: marker on the backhand (+y?) side -> here a wedge pointing at -55 deg
    pts = hp + 2.0 * np.array([[np.cos(np.radians(center)), np.sin(np.radians(center))], [1.0, 0.0],
                               [np.cos(np.radians(60)), np.sin(np.radians(60))]])
    p = D.mark_block_prob(CFG, hp, pts, np.array([0, 0, 0]), center, 50.0, 0.5, 0.5, False)
    assert p[0] > 0.4 and p[1] < p[0] and p[2] == 0.0
    ph = D.mark_block_prob(CFG, hp, pts[:1], np.array([D.HAM]), center, 50.0, 0.5, 0.5, False)
    assert ph[0] < 0.2 * p[0]


# ---------------------------------------------------------------- game rules
def test_possession_deterministic():
    ra, rb = rosters(3)
    a, _ = simulate_possession(CFG, ra, rb, vertical_stack(), CANONICAL_DEFENSES["person_forehand"](), seed=11)
    b, _ = simulate_possession(CFG, ra, rb, vertical_stack(), CANONICAL_DEFENSES["person_forehand"](), seed=11)
    assert (a.outcome, a.throws, a.duration) == (b.outcome, b.throws, b.duration)


@pytest.mark.parametrize("on", list(CANONICAL_OFFENSES))
@pytest.mark.parametrize("dn", list(CANONICAL_DEFENSES))
def test_every_canonical_matchup_ends_validly(on, dn):
    for k in range(2):
        ra, rb = rosters(k)
        res, _ = simulate_possession(CFG, ra, rb, CANONICAL_OFFENSES[on](), CANONICAL_DEFENSES[dn](), seed=k)
        assert res.outcome == "score" or res.outcome in TURNOVER_TYPES
        assert 0.0 < res.duration <= CFG.sim.max_possession_time + 1.0
        assert res.max_stall <= CFG.rules.stall_max + 1e-6


def test_stall_out_when_nobody_moves():
    """Receivers that never cut and a defense that smothers them -> the thrower eventually stalls or turns it over."""
    g = vertical_stack()
    g.cut_delay[:] = 99.0
    g.reset_stall = 99.0
    g.panic_stall = 9.9
    g.risk_aversion = 3.0
    ra, rb = rosters(5)
    res, _ = simulate_possession(CFG, ra, rb, g, CANONICAL_DEFENSES["person_forehand"](), seed=5)
    assert res.outcome != "score" or res.throws >= 1


def test_double_team_rule_keeps_helpers_out():
    """No defender other than the marker may *stay* inside the double-team radius (brief transients
    after a catch are physical; guarding an attacker who is also near the thrower is legal)."""
    ra, rb = rosters(2)
    game = Game(CFG, (ra, rb), (vertical_stack(), None), (None, CANONICAL_DEFENSES["person_forehand"]()), seed=2)
    game.start_possession(np.array([40.0, 18.0]), fresh=True)
    inside = np.zeros(14)
    worst = 0.0
    r = CFG.rules.double_team_radius
    while not game.res.outcome and game.t < 30:
        game.tick()
        if game.holder < 0:
            inside[:] = 0.0
            continue
        mk = game.marker_id()
        hp = game.pos[game.holder]
        for i in game.Dd:
            att = game.match[int(i)]
            illegal = (i != mk and np.hypot(*(game.pos[i] - hp)) < r - 0.5
                       and np.hypot(*(game.pos[att] - hp)) >= r)
            inside[i] = inside[i] + CFG.sim.dt if illegal else 0.0
        worst = max(worst, inside.max())
    assert worst <= 1.0


def test_flip_is_an_involution():
    ra, rb = rosters(1)
    g = Game(CFG, (ra, rb), (vertical_stack(), vertical_stack()),
             (CANONICAL_DEFENSES["person_forehand"](), CANONICAL_DEFENSES["person_forehand"]()), seed=1)
    g.start_possession(np.array([30.0, 10.0]), fresh=True)
    p0, v0, o0 = g.pos.copy(), g.vel.copy(), g.offense
    g.flip(); g.flip()
    assert np.allclose(g.pos, p0) and np.allclose(g.vel, v0) and g.offense == o0


def test_full_point_ends_with_score_or_cap():
    ra, rb = rosters(4)
    res, _ = simulate_point(CFG, (ra, rb), (vertical_stack(), CANONICAL_OFFENSES["horizontal"]()),
                            (CANONICAL_DEFENSES["cup_zone"](), CANONICAL_DEFENSES["person_forehand"]()), seed=4)
    assert res[-1].outcome == "score" or len(res) == CFG.sim.max_point_possessions
    assert all(r.offense != s.offense for r, s in zip(res, res[1:]))


# ---------------------------------------------------------------- replay format
def test_replay_matches_format(tmp_path):
    ra, rb = rosters(6)
    _, rec = simulate_point(CFG, (ra, rb), (vertical_stack(), CANONICAL_OFFENSES["horizontal"]()),
                            (CANONICAL_DEFENSES["cup_zone"](), CANONICAL_DEFENSES["person_forehand"]()), seed=6,
                            record=True)
    path = rec.save(tmp_path / "r.json")
    raw = path.read_text()
    assert raw.startswith('{"format":"ultisim-replay"')
    d = json.loads(raw)
    assert d["version"] == 1 and len(d["players"]) == 14 and d["result"]["outcome"]
    for f in d["frames"]:
        assert len(f["p"]) == 28 and len(f["st"]) == 14 and len(f["d"]) == 3 and f["ds"] in (0, 1, 2)
        assert f["o"] in (0, 1)
        xs, ys = np.array(f["p"][0::2]), np.array(f["p"][1::2])
        assert xs.min() > -5 and xs.max() < 105 and ys.min() > -5 and ys.max() < 42
    for t in d["throws"]:
        assert len(t["path"]) == 3 and t["type"] in D.THROW_TYPES and t["outcome"]


def test_wider_mark_wedge_does_not_add_coverage():
    """Exploit fix: total blocking (integrated over throw angles) must not grow with wedge width."""
    hp = np.array([40.0, 18.0])
    angs = np.radians(np.linspace(-180, 180, 3601))
    pts = hp + 2.0 * np.stack([np.cos(angs), np.sin(angs)], 1)
    total = {w: D.mark_block_prob(CFG, hp, pts, np.zeros(len(pts), int), 0.0, w, 0.5, 0.5, False).sum()
             for w in (30.0, 50.0, 80.0)}
    assert total[80.0] <= total[50.0] * 1.01 and total[30.0] < total[50.0]


def test_height_profile_and_loft():
    """Release height -> peak at mid-flight -> catch height; the tail sinks; lofts rise above flat throws."""
    hc = CFG.throws.height
    for peak in (1.5, 3.5, 6.0):
        assert D.flight_height(CFG, peak, 0.0, 1.0) == pytest.approx(hc.release)
        assert D.flight_height(CFG, peak, 0.5, 1.0) == pytest.approx(peak)
        assert D.flight_height(CFG, peak, 1.0, 1.0) == pytest.approx(hc.catch)
    T = 1.0
    tail_end = 1.0 + CFG.throws.tail_time / T
    assert D.flight_height(CFG, 3.0, tail_end, T) == pytest.approx(hc.ground)
    d = np.array([20.0])
    flat = D.throw_peak(CFG, np.array([D.BH]), d, np.array([False]), np.array([False]))[0]
    loft = D.throw_peak(CFG, np.array([D.BH]), d, np.array([False]), np.array([True]))[0]
    huck = D.throw_peak(CFG, np.array([D.BH]), np.array([40.0]), np.array([True]), np.array([False]))[0]
    ham = D.throw_peak(CFG, np.array([D.HAM]), d, np.array([False]), np.array([False]))[0]
    assert flat < 2.5 < loft < huck < ham
    t_flat = D.flight_time(CFG, np.array([D.BH]), d, np.array([False]), np.array([False]))[0]
    t_loft = D.flight_time(CFG, np.array([D.BH]), d, np.array([False]), np.array([True]))[0]
    assert t_loft > t_flat + 0.3


def _fronting_scene(angle, cushion, lat_err):
    """Thrower at (40,18); a stationary receiver 11 m away; their defender sits `cushion` m off the receiver,
    toward the disc (angle 0, fronting) or away from it (angle 180), and `lat_err` m off that spot sideways."""
    import copy
    ra, rb = rosters(4)
    dg = copy.deepcopy(CANONICAL_DEFENSES["person_forehand"]())
    dg.angle_near = dg.angle_far = angle
    dg.cushion_base, dg.cushion_per_m = cushion, 0.0
    g = Game(CFG, (ra, rb), (vertical_stack(), None), (None, dg), seed=1)
    g.start_possession(np.array([40.0, 18.0]), fresh=True)
    while g.holder < 0 or g.t < g.ready_time:
        g.tick()
    h = g.holder
    hp = g.pos[h].copy()
    r = next(int(o) for o in g.O if o != h)
    d = next(dd for dd, a in g.match.items() if a == r)
    for i in list(g.O) + list(g.Dd):   # everyone else far away
        if i not in (h, r, d, g.marker_id()):
            g.pos[i] = np.array([95.0, 2.0 + (i % 7) * 5.0])
            g.vel[i] = 0.0
    g.pos[r] = hp + np.array([11.0, 4.0])
    g.vel[r] = 0.0
    u = (hp - g.pos[r]) / np.hypot(*(hp - g.pos[r]))
    g.pos[d] = g.pos[r] + cushion * (u if angle == 0.0 else -u) - np.array([0.0, lat_err])
    g.vel[d] = 0.0
    ev = g.evaluate_throws()
    k = np.nonzero((ev["rcv"] == r) & (ev["curve"] == 0) & ~ev["loft"] & (ev["ttype"] != D.HAM)
                   & (np.hypot(*(ev["tgt"] - g.pos[r]).T) < 0.3))[0]
    return ev["p_d"][d - g.Dd[0]][k]


def test_thrower_sees_fronting_defender_step_into_the_lane():
    """Exploit fix (watch4): a person defender fronting the receiver, a step off the lane, will close onto it
    while the disc is in the air. The thrower's read must see that (it used to assume they just drift)."""
    for lat in (1.8, 2.5):
        assert _fronting_scene(0.0, 3.5, lat).min() > 0.5
    assert _fronting_scene(180.0, 1.5, 2.5).max() < 0.3   # a trailing defender is not in the lane
