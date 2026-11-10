"""Utilidades de cuantización (paso C2 de GUIA_ISI_CD.md; decisiones 25 y 26)."""
import numpy as np
import pytest

from emulador.quantization import quantize, to_int

W, I = 14, 1  # entrada de la CD y de la ISI (decisión 32): q = 2^-13, rango [-1, 1 - q]
Q = 2.0**-(W - I)


def _grid(W, I):
    q = 2.0 ** -(W - I)
    return np.arange(-(2 ** (W - 1)), 2 ** (W - 1)) * q


@pytest.mark.parametrize("W, I", [(14, 1), (20, 2), (18, 1), (14, 2), (10, 2), (8, 4)])
def test_representable_values_unchanged(W, I):
    v = _grid(W, I)
    for rounding in ("half_even", "half_up", "trunc"):
        assert np.array_equal(quantize(v, W, I, rounding=rounding), v)


def test_rounding_error_at_most_half_step():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1 - Q, 100_000)
    e = quantize(x, W, I) - x
    assert np.max(np.abs(e)) <= Q / 2


def test_trunc_error_in_minus_q_zero():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1 - Q, 100_000)
    e = quantize(x, W, I, rounding="trunc") - x
    assert np.all((e <= 0) & (e > -Q))


def test_saturation_ac_fixed_14_1():
    """±10 con ac_fixed<14, 1>: extremos del rango, -1 y 1 - 2^-13 (exacto)."""
    y = quantize(np.array([10.0, -10.0, 1.0, -1.0 - Q]), W, I)
    assert np.array_equal(y, [1 - Q, -1.0, 1 - Q, -1.0])


def test_rounding_up_to_max_saturates():
    """1 - q/4 redondea a 1, que no es representable: satura a 1 - q."""
    assert quantize(1 - Q / 4, W, I) == 1 - Q


def test_wrap_without_saturation():
    """saturate=False: desborde en complemento a dos (AC_WRAP)."""
    assert quantize(1.0, W, I, saturate=False) == -1.0
    assert quantize(-1.0 - Q, W, I, saturate=False) == 1 - Q
    assert quantize(3.0, 10, 2, saturate=False) == -1.0  # rango [-2, 2): 3 -> 3 - 4


def test_iq_independent():
    rng = np.random.default_rng(2)
    a, b = rng.uniform(-1.5, 1.5, (2, 1000))
    y = quantize(a + 1j * b, W, I)
    assert np.array_equal(y.real, quantize(a, W, I))
    assert np.array_equal(y.imag, quantize(b, W, I))


def test_half_even_ties():
    """Decisión 26 (AC_RND_CONV): empates a par."""
    y = quantize(np.array([0.5, 1.5, -0.5, 2.5, -1.5]) * Q, W, I)
    assert np.array_equal(y, np.array([0.0, 2.0, 0.0, 2.0, -2.0]) * Q)


def test_half_up_ties():
    """AC_RND: empates hacia +inf."""
    y = quantize(np.array([0.5, -0.5, 1.5, -1.5]) * Q, W, I, rounding="half_up")
    assert np.array_equal(y, np.array([1.0, 0.0, 2.0, -1.0]) * Q)


def test_default_is_half_even():
    x = np.array([0.5, 1.5, -0.5]) * Q
    assert np.array_equal(quantize(x, W, I), quantize(x, W, I, rounding="half_even"))
    assert not np.array_equal(quantize(x, W, I), quantize(x, W, I, rounding="half_up"))


def test_real_stays_real_and_shape_preserved():
    x = np.random.default_rng(3).uniform(-1, 1, (2, 7))
    y = quantize(x, W, I)
    assert y.shape == x.shape and not np.iscomplexobj(y)
    z = quantize(x + 1j * x, W, I)
    assert z.shape == x.shape and np.iscomplexobj(z)
    assert np.ndim(quantize(0.3, W, I)) == 0


def test_to_int_extremes_and_values():
    k = to_int(np.array([1 - Q, -1.0, 0.0, Q, -Q, 10.0, 0.5]), W, I)
    assert k.dtype == np.int64
    assert np.array_equal(k, [2**13 - 1, -(2**13), 0, 1, -1, 2**13 - 1, 2**12])


def test_to_int_consistent_with_quantize():
    rng = np.random.default_rng(4)
    x = rng.uniform(-2, 2, 10_000)
    for W_, I_ in ((14, 1), (20, 2), (18, 1)):
        k = to_int(x, W_, I_)
        assert np.array_equal(k * 2.0 ** -(W_ - I_), quantize(x, W_, I_))
        assert k.min() >= -(2 ** (W_ - 1)) and k.max() <= 2 ** (W_ - 1) - 1


def test_to_int_complex_returns_i_and_q():
    ki, kq = to_int(np.array([Q + 1j * (1 - Q), -1.0 - 1j]), W, I)
    assert np.array_equal(ki, [1, -(2**13)]) and np.array_equal(kq, [2**13 - 1, -(2**13)])


@pytest.mark.parametrize(
    "kwargs", [{"W": 0, "I": 0}, {"W": 14, "I": 1, "rounding": "nearest"}, {"W": 64, "I": 1}]
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        quantize(0.1, **kwargs)
