"""C engine behaviour: valid results, determinism (independent of threading), both modes."""
import numpy as np
import pytest

from ultisim.config import load_config
from ultisim.genome import CANONICAL_DEFENSES, CANONICAL_OFFENSES
from ultisim.roster import sample_roster

CFG = load_config()


@pytest.fixture(scope="module")
def eng():
    try:
        from ultisim.engine import CEngine
        return CEngine(CFG)
    except FileNotFoundError:
        pytest.skip("C core not built (python scripts/build_c.py)")


def pairs(n, seed=0):
    rng = np.random.default_rng(seed)
    return [(sample_roster(rng, CFG), sample_roster(rng, CFG)) for _ in range(n)]


def test_possessions_valid(eng):
    from ultisim.engine import pack_defenses, pack_offenses, make_specs, roster_matrix
    P = pairs(40)
    A, B = roster_matrix([a for a, _ in P]), roster_matrix([b for _, b in P])
    offs = pack_offenses([f() for f in CANONICAL_OFFENSES.values()])
    defs = pack_defenses([f() for f in CANONICAL_DEFENSES.values()])
    for oi in range(len(offs)):
        for di in range(len(defs)):
            r = eng.run(offs, defs, make_specs(A, B, np.arange(40) + 100 * oi + di, off_idx=(oi, 0), def_idx=(0, di)))
            assert np.all((r["outcome"] >= 0) & (r["outcome"] <= 8))
            assert np.all(r["n_poss"] == 1)
            assert np.all(r["completions"] <= r["throws"])
            assert np.all(r["duration"] > 0) and np.all(r["duration"] <= CFG.sim.max_possession_time + 1)
            assert np.all(r["max_stall"] <= CFG.rules.stall_max + 1e-9)
            counts = np.stack([r[f"n_{o}"] for o in ("score", "block", "interception", "drop", "throwaway",
                                                     "out_of_bounds", "stall", "mark_block", "timeout")], 1)
            assert np.all(counts.sum(1) == 1)


def test_deterministic_and_thread_independent(eng):
    from ultisim.engine import CEngine
    P = pairs(64, 1)
    og, dg = CANONICAL_OFFENSES["horizontal"](), CANONICAL_DEFENSES["cup_zone"]()
    a = eng.possessions(og, dg, P, np.arange(64))
    b = CEngine(CFG, threads=1).possessions(og, dg, P, np.arange(64))
    assert np.array_equal(a, b)
    c = eng.possessions(og, dg, P, np.arange(64) + 1)
    assert not np.array_equal(a["duration"], c["duration"])


def test_full_points(eng):
    P = pairs(60, 2)
    r = eng.points((CANONICAL_OFFENSES["vertical"](), CANONICAL_OFFENSES["horizontal"]()),
                   (CANONICAL_DEFENSES["cup_zone"](), CANONICAL_DEFENSES["person_forehand"]()), P, np.arange(60))
    done = r["scorer"] >= 0
    assert np.all(done | (r["n_poss"] == CFG.sim.max_point_possessions))
    assert np.all(r["n_score"][done] == 1)
    # possessions alternate, so the scorer's parity matches the possession count (team 0 receives)
    assert np.all(r["scorer"][done] == (r["n_poss"][done] - 1) % 2)


def test_genome_changes_matter(eng):
    """A thrower who never risks anything and never cuts should score less than the canonical stack."""
    P = pairs(300, 3)
    g = CANONICAL_OFFENSES["vertical"]()
    passive = CANONICAL_OFFENSES["vertical"]()
    passive.cut_delay[:] = 8.0
    passive.risk_aversion = 3.0
    passive.patience = 1.6
    dg = CANONICAL_DEFENSES["person_forehand"]()
    good = eng.possessions(g, dg, P, np.arange(300))
    bad = eng.possessions(passive, dg, P, np.arange(300))
    assert (good["outcome"] == 0).mean() > (bad["outcome"] == 0).mean()
