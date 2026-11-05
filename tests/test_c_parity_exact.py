"""Exact parity: the C engine's deterministic building blocks vs. the Python reference."""
import ctypes

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from ultisim import cabi
from ultisim import disc as D
from ultisim.config import load_config
from ultisim.movement import step, time_to_reach

CFG = load_config()


@pytest.fixture(scope="module")
def C():
    try:
        L = cabi.lib()
    except FileNotFoundError:
        pytest.skip("C core not built (python scripts/build_c.py)")
    from ultisim.engine import make_config
    return L, make_config(CFG)


def test_layouts_match(C):
    L, _ = C
    assert L.us_sizeof_config() == ctypes.sizeof(cabi.UsConfig)
    assert L.us_sizeof_spec() == ctypes.sizeof(cabi.UsGameSpec)
    assert L.us_sizeof_result() == ctypes.sizeof(cabi.UsResult)


def test_time_to_reach(C):
    L, _ = C
    rng = np.random.default_rng(0)
    for _ in range(2000):
        p, v, q = rng.uniform(-20, 20, 2), rng.uniform(-8, 8, 2), rng.uniform(-20, 20, 2)
        top, acc, brk, reach = rng.uniform(6, 9), rng.uniform(3, 5), rng.uniform(4, 9), rng.uniform(0.8, 1.5)
        delay, extra = rng.uniform(0, 0.6), rng.uniform(0, 2)
        ref = time_to_reach(p[None], v[None], np.array([top]), np.array([acc]), np.array([brk]), np.array([reach]),
                            q[None], delay=np.array([delay]), extra_reach=extra)[0, 0]
        got = L.us_time_to_reach(p[0], p[1], v[0], v[1], top, acc, brk, reach, q[0], q[1], delay, extra)
        assert got == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_flight_point(C):
    L, _ = C
    rng = np.random.default_rng(1)
    out = (ctypes.c_double * 2)()
    for _ in range(500):
        p0, p1 = rng.uniform(0, 60, 2), rng.uniform(0, 60, 2)
        c = D.control_point(p0, p1, np.array(rng.choice([-1.0, 0.0, 1.0])), CFG)
        T, s = rng.uniform(0.3, 4), rng.uniform(0, 1.5)
        ref = D.flight_point(p0, c, p1, T, s, CFG.throws.tail_time)
        L.us_flight_point(p0[0], p0[1], c[0], c[1], p1[0], p1[1], T, s, CFG.throws.tail_time, out)
        assert np.allclose(list(out), ref, rtol=1e-9, atol=1e-9)


def test_error_sigma(C):
    L, ccfg = C
    rng = np.random.default_rng(2)
    for _ in range(500):
        skill, t, d = rng.uniform(), int(rng.integers(3)), rng.uniform(3, 60)
        b, m, cv = (int(x) for x in rng.integers(0, 2, 3))
        ref = D.error_sigma(CFG, skill, np.array(t), d, bool(b), bool(m), bool(cv))
        assert L.us_error_sigma(ctypes.byref(ccfg), skill, t, d, b, m, cv) == pytest.approx(float(ref), rel=1e-12)


def test_mark_block_prob(C):
    L, ccfg = C
    rng = np.random.default_rng(3)
    for _ in range(500):
        hp, e = rng.uniform(10, 30, 2), rng.uniform(10, 30, 2)
        t, center, width = int(rng.integers(3)), rng.uniform(-180, 180), rng.uniform(20, 80)
        mk, bs, fk = rng.uniform(), rng.uniform(), bool(rng.integers(2))
        ref = D.mark_block_prob(CFG, hp, e[None], np.array([t]), center, width, mk, bs, fk)[0]
        got = L.us_mark_block_prob(ctypes.byref(ccfg), hp[0], hp[1], e[0], e[1], t, center, width, mk, bs, int(fk))
        assert got == pytest.approx(float(ref), rel=1e-9, abs=1e-12)


def test_step(C):
    L, _ = C
    rng = np.random.default_rng(4)
    n = 14
    for _ in range(50):
        pos, vel, tgt = rng.uniform(0, 50, (n, 2)), rng.uniform(-8, 8, (n, 2)), rng.uniform(0, 50, (n, 2))
        top, acc = rng.uniform(6, 9, n), rng.uniform(3, 5, n)
        brk, spd, tv = acc * rng.uniform(1.2, 2.0, n), rng.uniform(0, 9, n), rng.uniform(-3, 3, (n, 2))
        frozen = rng.random(n) < 0.2
        ref_p, ref_v = step(pos, vel, tgt, spd, acc, brk, top, 0.1, frozen, target_vel=tv)
        P = lambda a: np.ascontiguousarray(a, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        px, py, vx, vy = (np.ascontiguousarray(a, dtype=np.float64) for a in (pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1]))
        tx, ty = np.ascontiguousarray(tgt[:, 0]), np.ascontiguousarray(tgt[:, 1])
        tvf = np.ascontiguousarray(tv.reshape(-1))
        fr = np.ascontiguousarray(frozen.astype(np.int32))
        L.us_step(n, P(px), P(py), P(vx), P(vy), P(tx), P(ty), P(spd), P(acc), P(brk), P(top), P(tvf), 0.1,
                  fr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        assert np.allclose(np.stack([px, py], 1), ref_p, atol=1e-9)
        assert np.allclose(np.stack([vx, vy], 1), ref_v, atol=1e-9)


def test_hungarian_is_optimal(C):
    L, _ = C
    rng = np.random.default_rng(5)
    for n in (6, 7):
        for _ in range(200):
            cost = rng.uniform(-5, 20, (n, n))
            out = (ctypes.c_int32 * n)()
            flat = np.ascontiguousarray(cost.reshape(-1))
            L.us_hungarian(n, n, flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), out)
            r, c = linear_sum_assignment(cost)
            assert sorted(out) == list(range(n))
            assert cost[np.arange(n), list(out)].sum() == pytest.approx(cost[r, c].sum(), abs=1e-9)


def test_flight_height(C):
    L, ccfg = C
    rng = np.random.default_rng(6)
    for _ in range(500):
        peak, s, T, h0 = rng.uniform(1.0, 8.0), rng.uniform(0, 1.6), rng.uniform(0.3, 4.0), rng.choice([1.0, 2.2])
        ref = float(D.flight_height(CFG, peak, s, T, h0))
        assert L.us_flight_height(ctypes.byref(ccfg), peak, s, T, h0) == pytest.approx(ref, rel=1e-9, abs=1e-12)
