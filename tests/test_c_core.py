import pytest

from ultisim import cabi


def _lib():
    try:
        return cabi.lib()
    except FileNotFoundError:
        pytest.skip("C core not built (python scripts/build_c.py)")


def test_abi_and_smoke():
    L = _lib()
    assert L.us_abi_version() == cabi.ABI_VERSION
    assert L.us_max_threads() >= 1
    assert abs(L.us_smoke(100000) - 1.6449240668982423) < 1e-9
