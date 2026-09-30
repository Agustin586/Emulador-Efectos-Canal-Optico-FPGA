"""Estímulos (paso P3 de GUIA_ISI_CD.md)."""
import numpy as np
import pytest

from emulador.stimuli import prbs


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
