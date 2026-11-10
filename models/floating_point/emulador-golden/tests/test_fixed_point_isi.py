"""FIR de la ISI en punto fijo, bit a bit (paso C3.1 de GUIA_ISI_CD.md; decisiones 26, 27 y 32)."""
import numpy as np
import pytest

from emulador.blocks import ISI
from emulador.fixed_point import isi_fir_fixed, round_shift, saturate_int
from emulador.isi import isi_apply
from emulador.metrics import nrmse
from emulador.quantization import quantize, to_int


def _signal(n=4000, seed=0, peak=0.95):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return x * peak / np.max(np.abs(np.r_[x.real, x.imag]))


def _bessel(K=16, B_rel=0.6):
    return ISI(profile="bessel", B_rel=B_rel, order=4, K=K).h


# --------------------------------------------------------------------------
# Enteros: redondeo y saturación con la semántica de ac_fixed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rounding", ["half_even", "half_up", "trunc"])
def test_round_shift_matches_to_int(rounding):
    """round_shift(k, s) = to_int(k·2^-s) en el rango donde el float es exacto, empates incluidos."""
    rng = np.random.default_rng(1)
    k = np.r_[rng.integers(-(2**40), 2**40, 20_000), np.arange(-64, 65)]
    for s in (1, 2, 5, 17):
        got = round_shift(k, s, rounding)
        # formato entero <53, 53>: no satura con |k| < 2^41
        expected = to_int(k * 2.0**-s, 53, 53, rounding=rounding)
        assert np.array_equal(got, expected)


def test_round_shift_exact_beyond_float():
    """Con |k| > 2^53 sigue exacto (aritmética entera de Python)."""
    k = np.array([2**60 + 2**4 + 2**3, -(2**60) - 2**3, 2**61 + 3 * 2**3], dtype=np.int64)
    assert round_shift(k, 4, "half_even").tolist() == [2**56 + 2, -(2**56), 2**57 + 2]
    assert round_shift(k, 4, "half_up").tolist() == [2**56 + 2, -(2**56), 2**57 + 2]
    assert round_shift(k, 4, "trunc").tolist() == [2**56 + 1, -(2**56) - 1, 2**57 + 1]


def test_round_shift_zero_is_identity():
    k = np.array([-5, 0, 7])
    assert np.array_equal(round_shift(k, 0, "half_even"), k)


def test_saturate_int_counts():
    k, n_sat = saturate_int(np.array([-200, -128, 0, 127, 128, 300]), 8)
    assert k.tolist() == [-128, -128, 0, 127, 127, 127] and n_sat == 3


# --------------------------------------------------------------------------
# FIR en punto fijo
# --------------------------------------------------------------------------
def _reference_codes(xr, xi, hr, hi, F_full, shift_acc, out_shift, acc_rounding, out_rounding, W_out):
    """FIR bit a bit con enteros de Python, lazo explícito (referencia independiente)."""
    n, K = len(xr), len(hr)

    def rnd(v, s, mode):
        if s == 0:
            return v
        q, r = divmod(v, 1 << s)
        half = 1 << (s - 1)
        if mode == "trunc":
            return q
        if mode == "half_up":
            return q + (r >= half)
        return q + (r > half or (r == half and q % 2 == 1))

    lo, hi_ = -(1 << (W_out - 1)), (1 << (W_out - 1)) - 1
    yr, yi = [], []
    for m in range(n):
        ar = ai = 0
        for k in range(K):
            if m - k < 0:
                continue
            a, b, c, d = int(xr[m - k]), int(xi[m - k]), int(hr[k]), int(hi[k])
            ar += rnd(a * c, shift_acc, acc_rounding) - rnd(b * d, shift_acc, acc_rounding)
            ai += rnd(a * d, shift_acc, acc_rounding) + rnd(b * c, shift_acc, acc_rounding)
        yr.append(min(max(rnd(ar, out_shift, out_rounding), lo), hi_))
        yi.append(min(max(rnd(ai, out_shift, out_rounding), lo), hi_))
    return np.array(yr), np.array(yi)


@pytest.mark.parametrize("acc_frac", [None, 20, 14])
@pytest.mark.parametrize("acc_rounding", ["half_even", "trunc"])
def test_bit_exact_against_integer_loop(acc_frac, acc_rounding):
    x = _signal(300, seed=2)
    h = _bessel()
    res = isi_fir_fixed(x, h, acc_frac=acc_frac, acc_rounding=acc_rounding)
    xr, xi = to_int(x, 14, 1)
    hr, hi = to_int(h, 18, 1)
    F_full = 13 + 17
    shift_acc = 0 if acc_frac is None else F_full - acc_frac
    F_acc = F_full - shift_acc
    yr, yi = _reference_codes(xr, xi, hr, hi, F_full, shift_acc, F_acc - 13, acc_rounding, "half_even", 14)
    assert np.array_equal(res.codes[0], yr) and np.array_equal(res.codes[1], yi)
    assert np.array_equal(res.y, (yr + 1j * yi) * 2.0**-13)


def test_identity_tap_reproduces_quantized_input():
    """h = [1] satura a 1 - 2^-17 en <18, 1>, pero la salida redondea de vuelta a x cuantizado."""
    x = _signal(2000, seed=3)
    res = isi_fir_fixed(x, np.array([1.0 + 0j]))
    assert np.array_equal(res.y, quantize(x, 14, 1))
    assert res.saturations["h"] == 1


def test_wide_formats_match_float_fir():
    """Con formatos amplios, coincide con isi_apply (el FIR flotante del bloque)."""
    x = _signal(3000, seed=4, peak=0.5)
    h = _bessel(K=32, B_rel=0.4)
    res = isi_fir_fixed(x, h, x_fmt=(28, 1), h_fmt=(28, 1), out_fmt=(50, 3))
    assert nrmse(res.y, isi_apply(x, h)) < 1e-7


def test_starting_point_error_small_and_nonzero():
    """Punto de partida (decisión 32): error interno ≤ 1e-3 contra el flotante con la misma entrada."""
    x = quantize(_signal(8000, seed=5), 14, 1)
    h = _bessel()
    res = isi_fir_fixed(x, h, out_fmt=(14, 2))
    e = nrmse(res.y, isi_apply(x, h))
    assert 1e-6 < e < 1e-3


def test_accumulator_format_and_growth():
    """Acumulador: <W_x + W_h + 1 + ceil(log2 K), I_x + I_h + 1 + ceil(log2 K)> con precisión completa."""
    h = _bessel(K=16)
    res = isi_fir_fixed(_signal(100), h)
    assert res.acc_fmt == (14 + 18 + 1 + 4, 1 + 1 + 1 + 4)
    res = isi_fir_fixed(_signal(100), h, acc_frac=20)
    assert res.acc_fmt == (7 + 20, 7)
    assert res.saturations["acc"] == 0


def test_saturation_counts():
    x = _signal(1000, seed=6, peak=1.5)  # componentes fuera de [-1, 1)
    res = isi_fir_fixed(x, np.array([0.9, 0.9, 0.9]), out_fmt=(14, 1))
    assert res.saturations["x"] > 0
    assert res.saturations["out"] > 0
    res = isi_fir_fixed(_signal(1000, seed=6), _bessel(), out_fmt=(14, 2))
    assert res.saturations == {"x": 0, "h": 0, "acc": 0, "out": 0}


def test_output_length_and_causality():
    x = np.zeros(50, complex)
    x[10] = 0.5
    h = _bessel()
    res = isi_fir_fixed(x, h)
    assert res.y.shape == (50,)
    assert np.all(res.y[:10] == 0)
    assert np.allclose(res.y[10 : 10 + h.size], 0.5 * h, atol=2**-13)


def test_real_input_accepted():
    res = isi_fir_fixed(np.full(20, 0.25), np.array([0.5]))
    assert np.allclose(res.y, 0.125)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_fmt": (34, 1), "h_fmt": (28, 1)},  # el acumulador no entra en int64
        {"acc_frac": 31},  # más bits que la precisión completa (30)
        {"acc_frac": 0},
    ],
)
def test_invalid_formats(kwargs):
    with pytest.raises(ValueError):
        isi_fir_fixed(_signal(10), _bessel(), **kwargs)


def test_rejects_2d_input():
    with pytest.raises(ValueError):
        isi_fir_fixed(np.zeros((2, 10), complex), _bessel())
