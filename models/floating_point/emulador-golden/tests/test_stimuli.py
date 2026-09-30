"""Estímulos (paso P3 de GUIA_ISI_CD.md)."""
import numpy as np
import pytest

from emulador.stimuli import map_gray, prbs


# --------------------------------------------------------------------------
# prbs
# --------------------------------------------------------------------------
@pytest.mark.parametrize("order", [7, 9, 11, 15])
def test_prbs_period(order):
    period = 2**order - 1
    b = prbs(order, 2 * period)
    assert np.array_equal(b[:period], b[period:])
    # el período es exactamente 2^order - 1 (no un divisor)
    for d in (1, period // 3, period // 7):
        if d and period % d == 0:
            assert not np.array_equal(b[:period], np.roll(b[:period], -d))


def test_prbs15_balance():
    b = prbs(15, 2**15 - 1)
    assert int(b.sum()) == 2**14


def test_prbs31_recurrence():
    # x^31 + x^28 + 1  ->  b[n] = b[n-31] xor b[n-28]
    b = prbs(31, 5000, seed=0x5A5A5A5)
    n = np.arange(31, 5000)
    assert np.array_equal(b[n], b[n - 31] ^ b[n - 28])


def test_prbs_first_bits_are_seed():
    b = prbs(15, 15, seed=0b100000000000001)
    assert b.tolist() == [1] + [0] * 13 + [1]


def test_prbs_dtype_and_values():
    b = prbs(15, 1000)
    assert b.dtype == np.uint8
    assert set(np.unique(b)) <= {0, 1}


def test_prbs_invert():
    assert np.array_equal(prbs(15, 1000, invert=True), 1 - prbs(15, 1000))


def test_prbs_reproducible():
    assert np.array_equal(prbs(31, 3000, seed=7), prbs(31, 3000, seed=7))
    assert not np.array_equal(prbs(31, 3000, seed=7), prbs(31, 3000, seed=8))


@pytest.mark.parametrize("seed", [0, 2**15])
def test_prbs_rejects_invalid_seed(seed):
    with pytest.raises(ValueError):
        prbs(15, 10, seed=seed)


def test_prbs_rejects_unknown_order():
    with pytest.raises(ValueError):
        prbs(16, 10)


# --------------------------------------------------------------------------
# map_gray
# --------------------------------------------------------------------------
def _all_labels(bps):
    """Todas las etiquetas de bps bits, concatenadas (MSB primero)."""
    return np.array([(v >> (bps - 1 - i)) & 1 for v in range(2**bps) for i in range(bps)])


def test_pam4_levels():
    s = map_gray([0, 0, 0, 1, 1, 1, 1, 0], "pam4")
    assert np.allclose(s * np.sqrt(5), [-3, -1, 1, 3], rtol=0, atol=1e-12)


def test_qam16_i_then_q():
    # I = 10 -> +3, Q = 00 -> -3
    s = map_gray([1, 0, 0, 0], "qam16")
    assert np.allclose(s * np.sqrt(10), [3 - 3j], rtol=0, atol=1e-12)


@pytest.mark.parametrize("mod, bps", [("pam4", 2), ("qam16", 4)])
def test_gray_neighbors_differ_in_one_bit(mod, bps):
    labels = np.arange(2**bps)
    pts = map_gray(_all_labels(bps), mod)
    d = np.abs(pts[:, None] - pts[None, :])
    dmin = d[d > 1e-9].min()
    n_pairs = 0
    for a in labels:
        for b in labels:
            if a < b and np.isclose(d[a, b], dmin):
                assert bin(a ^ b).count("1") == 1
                n_pairs += 1
    assert n_pairs == {"pam4": 3, "qam16": 24}[mod]


@pytest.mark.parametrize("mod, bps", [("pam4", 2), ("qam16", 4)])
def test_constellation_unit_power(mod, bps):
    pts = map_gray(_all_labels(bps), mod)
    assert len(np.unique(np.round(pts, 12))) == 2**bps
    assert np.isclose(np.mean(np.abs(pts) ** 2), 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mod, n", [("pam4", 3), ("qam16", 6)])
def test_map_gray_rejects_bad_length(mod, n):
    with pytest.raises(ValueError):
        map_gray(np.zeros(n, dtype=np.uint8), mod)


def test_map_gray_rejects_unknown_mod():
    with pytest.raises(ValueError):
        map_gray(np.zeros(4, dtype=np.uint8), "qpsk")
