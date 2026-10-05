"""ISI genérica (Parte A de GUIA_ISI_CD.md)."""
from functools import partial

import numpy as np
import pytest

from emulador.blocks import ISI
from emulador.isi import (
    K_rule,
    echo,
    energy_captured,
    gaussian,
    isi_apply,
    isi_apply_stream,
    measured,
    normalize_energy,
)
from emulador.isi.profiles import _interp_table
from emulador.stimuli import impulse

TOL = 1e-12


def _crandn(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# --------------------------------------------------------------------------
# fir: isi_apply, isi_apply_stream (A1.1)
# --------------------------------------------------------------------------
def test_impulse_gives_taps():
    h = _crandn(np.random.default_rng(0), 7)
    y = isi_apply(impulse(20), h)
    assert np.array_equal(y[:7], h)
    assert np.array_equal(y[7:], np.zeros(13))


@pytest.mark.parametrize("K", [1, 7, 16, 32])
def test_matches_numpy_convolve(K):
    rng = np.random.default_rng(K)
    h = _crandn(rng, K)
    x = _crandn(rng, 500)
    y = isi_apply(x, h)
    assert y.shape == x.shape
    assert np.max(np.abs(y - np.convolve(x, h)[: x.size])) < TOL


def test_linearity():
    rng = np.random.default_rng(1)
    h = _crandn(rng, 16)
    x1, x2 = _crandn(rng, 300), _crandn(rng, 300)
    a, b = 0.7 - 1.3j, -2.1 + 0.4j
    lhs = isi_apply(a * x1 + b * x2, h)
    rhs = a * isi_apply(x1, h) + b * isi_apply(x2, h)
    assert np.max(np.abs(lhs - rhs)) < TOL


def test_two_polarizations():
    rng = np.random.default_rng(2)
    h = _crandn(rng, 16)
    x = _crandn(rng, 2, 400)
    y = isi_apply(x, h)
    assert y.shape == (2, 400)
    assert np.array_equal(y[0], isi_apply(x[0], h))
    assert np.array_equal(y[1], isi_apply(x[1], h))


@pytest.mark.parametrize("K", [1, 16, 32])
@pytest.mark.parametrize("shape", [(1000,), (2, 1000)])
def test_stream_equals_full(K, shape):
    rng = np.random.default_rng(K)
    h = _crandn(rng, K)
    x = _crandn(rng, *shape)
    # trozos de largos distintos, incluidos algunos más cortos que K - 1 y
    # uno vacío al final
    cuts = [3, 10, 11, 100, 337, 340, 700, 1000]
    chunks = np.split(x, cuts, axis=-1)
    y = isi_apply_stream(chunks, h)
    assert len(y) == len(chunks)
    assert all(yc.shape == xc.shape for yc, xc in zip(y, chunks))
    full = isi_apply(x, h)
    assert np.max(np.abs(np.concatenate(y, axis=-1) - full)) < TOL


def test_rejects_empty_taps():
    with pytest.raises(ValueError):
        isi_apply(np.ones(4, complex), np.array([]))


# --------------------------------------------------------------------------
# Bloque ISI con taps (A1.2)
# --------------------------------------------------------------------------
def test_block_default_is_bypass():
    x = _crandn(np.random.default_rng(3), 256)
    blk = ISI()
    assert np.array_equal(blk(x), x)
    assert np.array_equal(blk.process_hw(x), x)
    assert blk.delay_samples == 0


def test_block_impulse_hw_gives_taps():
    h = _crandn(np.random.default_rng(4), 9)
    y = ISI(taps=h).process_hw(impulse(30))
    assert np.array_equal(y[:9], h)


def test_block_delay_and_alignment():
    h = np.array([0.1, 0.2, 1.0, 0.3])
    blk = ISI(taps=h)
    assert blk.delay_samples == 2
    x = _crandn(np.random.default_rng(5), 200)
    y, y_hw = blk(x), blk.process_hw(x)
    assert y.shape == y_hw.shape == x.shape
    assert np.array_equal(y[:-2], y_hw[2:])
    # la cola alineada sale de seguir filtrando con ceros a la entrada
    assert np.array_equal(y[-2:], isi_apply(np.concatenate([x, np.zeros(2)]), h)[-2:])


def test_block_impulse_aligned_puts_cursor_at_zero():
    h = np.array([0.1, 0.2, 1.0, 0.3])
    y = ISI(taps=h)(impulse(10, 5))
    assert np.array_equal(y[3:7], h)  # el cursor (k0 = 2) cae en la posición 5


def test_block_two_polarizations():
    rng = np.random.default_rng(6)
    blk = ISI(taps=_crandn(rng, 16))
    x = _crandn(rng, 2, 300)
    y, y_hw = blk(x), blk.process_hw(x)
    for p in range(2):
        assert np.array_equal(y[p], blk(x[p]))
        assert np.array_equal(y_hw[p], blk.process_hw(x[p]))


def test_block_coefficients():
    h = [1, 0.3j]
    c = ISI(taps=h).coefficients()
    assert set(c) == {"h"}
    assert c["h"].dtype == complex
    assert np.array_equal(c["h"], np.array(h, dtype=complex))


@pytest.mark.parametrize("taps", [np.ones(33), np.array([]), np.ones((2, 4))])
def test_block_rejects_invalid_taps(taps):
    with pytest.raises(ValueError):
        ISI(taps=taps)


def test_block_accepts_K32():
    assert ISI(taps=np.ones(32)).coefficients()["h"].size == 32


def test_block_rejects_unknown_profile():
    with pytest.raises(ValueError):
        ISI(profile="nope")


# --------------------------------------------------------------------------
# profiles: normalize_energy, echo, K_rule (A2.1)
# --------------------------------------------------------------------------
def test_normalize_energy():
    rng = np.random.default_rng(7)
    for h in (_crandn(rng, 16), rng.standard_normal(5), np.array([3.0])):
        g = normalize_energy(h)
        assert abs(np.sum(np.abs(g) ** 2) - 1) < TOL
        # solo cambia la escala (PDF Ec. 9)
        assert np.max(np.abs(g * np.sqrt(np.sum(np.abs(h) ** 2)) - h)) < TOL


def test_normalize_energy_rejects_zero():
    with pytest.raises(ValueError):
        normalize_energy(np.zeros(4))


def test_echo_example():
    """echo([(2, 0.5)]): antes de normalizar, h = [1, 0, 0.5]."""
    h = echo([(2, 0.5)])
    assert np.max(np.abs(h - normalize_energy(np.array([1, 0, 0.5])))) < TOL
    assert abs(np.sum(np.abs(h) ** 2) - 1) < TOL


def test_echo_several_complex_and_K():
    h = echo([(1, 0.2j), (4, -0.1)], K=8)
    raw = np.array([1, 0.2j, 0, 0, -0.1, 0, 0, 0])
    assert h.size == 8
    assert np.max(np.abs(h - normalize_energy(raw))) < TOL
    assert int(np.argmax(np.abs(h))) == 0  # el cursor sigue en k = 0


def test_echo_without_echoes_is_bypass():
    assert np.array_equal(echo([]), np.array([1.0 + 0j]))


@pytest.mark.parametrize(
    "echoes, K",
    [
        ([(2, 1.0)], None),  # |a| >= 1
        ([(2, 1.2j)], None),
        ([(0, 0.3)], None),  # d = 0 pisa el tap principal
        ([(32, 0.3)], None),  # d > 31
        ([(2, 0.3), (2, 0.1)], None),  # d repetido
        ([(1.5, 0.3)], None),  # d no entero
        ([(5, 0.3)], 5),  # K <= max(d)
        ([(5, 0.3)], 33),  # K > 32
    ],
)
def test_echo_rejects_invalid(echoes, K):
    with pytest.raises(ValueError):
        echo(echoes, K=K)


@pytest.mark.parametrize(
    "B_rel, m, K",
    [(0.3, 4, 32), (0.4, 4, 32), (0.4, 6, 32), (0.4, 3, 16), (0.41, 4, 16), (0.6, 4, 16), (0.8, 2, 16)],
)
def test_K_rule(B_rel, m, K):
    """Decisión 6: K = 32 si B <= 0,4 Rs y m >= 4; 16 si no (perfil principal: 16)."""
    for Rs in (32e9, 100e9):
        assert K_rule(B_rel * Rs, Rs, m) == K


# --------------------------------------------------------------------------
# profiles: gaussian, energy_captured (A2.2)
# --------------------------------------------------------------------------
RS = 32e9
B_RELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _H_rel_db(h, f, fs):
    """20·log10(|H(f)| / |H(0)|) con H(f) = Σ h[k] exp(-j 2π f k Ts)."""
    k = np.arange(h.size)
    H = lambda f_: np.sum(h * np.exp(-2j * np.pi * f_ * k / fs))  # noqa: E731
    return 20 * np.log10(abs(H(f)) / abs(H(0.0)))


@pytest.mark.parametrize("B_rel", B_RELS)
@pytest.mark.parametrize("K", [16, 32])
def test_gaussian_energy_and_3dB(B_rel, K):
    fs = 2 * RS
    h = gaussian(B_rel * RS, fs, K)
    assert h.shape == (K,)
    assert abs(np.sum(np.abs(h) ** 2) - 1) < TOL
    # relativo a f = 0: la normalización por energía cambia |H(0)|
    assert _H_rel_db(h, B_rel * RS, fs) == pytest.approx(-3.0, abs=0.3)


@pytest.mark.parametrize("K", [15, 16, 32])
def test_gaussian_symmetric_around_center(K):
    h = gaussian(0.6 * RS, 2 * RS, K)
    k0 = K // 2
    assert int(np.argmax(np.abs(h))) == k0
    for i in range(1, K - k0):
        assert abs(h[k0 - i] - h[k0 + i]) < TOL


def test_gaussian_scales_with_scenario():
    """Mismo B/Rs y sps a 32 y 100 GBd: mismos taps (el diseño depende de B/fs)."""
    h32 = gaussian(0.6 * 32e9, 2 * 32e9, 16)
    h100 = gaussian(0.6 * 100e9, 2 * 100e9, 16)
    assert np.max(np.abs(h32 - h100)) < TOL


@pytest.mark.parametrize("B_rel", B_RELS)
def test_gaussian_energy_captured(B_rel):
    B = B_rel * RS
    K = K_rule(B, RS, 4)
    assert energy_captured(partial(gaussian, B, 2 * RS), K) >= 0.985


def test_energy_captured_detects_truncation():
    """Con banda angosta la respuesta es larga: truncada a pocos taps pierde energía."""
    design = partial(gaussian, 0.1 * RS, 2 * RS)
    assert energy_captured(design, 4) < 0.9
    assert energy_captured(design, 32) > energy_captured(design, 4)


# --------------------------------------------------------------------------
# profiles: measured; perfiles echo/gaussian/measured en el bloque (A2.3)
# --------------------------------------------------------------------------
def _H_gauss(B):
    return lambda f: np.exp(-np.log(2) / 2 * f**2 / B**2)


@pytest.mark.parametrize("K", [16, 32])
def test_measured_function_equals_gaussian(K):
    B, fs = 0.6 * RS, 2 * RS
    h = measured(_H_gauss(B), fs, K)
    assert np.max(np.abs(h - gaussian(B, fs, K))) < TOL


def test_measured_table_on_design_grid_equals_gaussian():
    """Tabla (f, H) en los mismos puntos de la FFT de diseño: la interpolación es exacta."""
    B, fs, L = 0.5 * RS, 2 * RS, 8192
    f = np.fft.fftfreq(L, 1 / fs)
    h = measured((f, _H_gauss(B)(f)), fs, 16, L=L)
    assert np.max(np.abs(h - gaussian(B, fs, 16, L=L))) < TOL


def test_measured_coarse_table():
    """Tabla gruesa (17 puntos en toda la banda): la interpolación en dB alcanza."""
    B, fs = 0.6 * RS, 2 * RS
    f = np.linspace(-fs / 2, fs / 2, 17)
    h = measured((f, _H_gauss(B)(f)), fs, 16)
    assert abs(np.sum(np.abs(h) ** 2) - 1) < TOL
    assert _H_rel_db(h, B, fs) == pytest.approx(-3.0, abs=0.3)
    assert np.max(np.abs(h - gaussian(B, fs, 16))) < 1e-3


def test_measured_table_is_zero_outside_range():
    """Fuera del rango de la tabla, H = 0; adentro, interpola magnitud y fase."""
    f_t = np.array([-10e9, 0.0, 10e9])
    H_t = np.array([0.5, 1.0, 0.5 * np.exp(1j)])
    f = np.array([-20e9, -10e9, 5e9, 10e9, 20e9])
    Hf = _interp_table((f_t, H_t), f)
    assert Hf[0] == 0 and Hf[-1] == 0
    assert abs(Hf[1] - 0.5) < TOL and abs(Hf[3] - 0.5 * np.exp(1j)) < TOL
    # a mitad de camino: -3,01 dB (promedio en dB de 0 y -6,02) y fase 0,5 rad
    assert abs(Hf[2] - np.sqrt(0.5) * np.exp(0.5j)) < TOL


def test_measured_keeps_asymmetric_response():
    """Una H con retardo de 2 muestras da taps complejos con el cursor corrido 2."""
    B, fs, K = 0.6 * RS, 2 * RS, 16
    h = measured(lambda f: _H_gauss(B)(f) * np.exp(-2j * np.pi * f * 2 / fs), fs, K)
    assert int(np.argmax(np.abs(h))) == K // 2 + 2


def test_measured_rejects_bad_table():
    with pytest.raises(ValueError):
        measured((np.array([0.0, 1e9]), np.array([1.0])), 2 * RS, 16)


def test_block_gaussian_default_params():
    """Perfil gaussiano con los valores por defecto: B = 0,6·Rs, 32 GBd, 2 sps, K = 16."""
    blk = ISI(profile="gaussian")
    h = blk.coefficients()["h"]
    assert h.size == 16
    assert np.max(np.abs(h - gaussian(0.6 * 32e9, 64e9, 16))) < TOL
    assert blk.delay_samples == 8


def test_block_gaussian_K_rule_and_override():
    assert ISI(profile="gaussian", B_rel=0.4).coefficients()["h"].size == 32
    assert ISI(profile="gaussian", B_rel=0.4, order=2).coefficients()["h"].size == 16
    assert ISI(profile="gaussian", K=24).coefficients()["h"].size == 24


def test_block_gaussian_scales_with_Rs():
    h32 = ISI(profile="gaussian", Rs=32e9).coefficients()["h"]
    h100 = ISI(profile="gaussian", Rs=100e9).coefficients()["h"]
    assert np.max(np.abs(h32 - h100)) < TOL


def test_block_echo():
    blk = ISI(profile="echo", echoes=((2, 0.5),))
    assert np.max(np.abs(blk.coefficients()["h"] - echo([(2, 0.5)]))) < TOL
    assert blk.delay_samples == 0
    assert ISI(profile="echo", echoes=((2, 0.5),), K=8).coefficients()["h"].size == 8


def test_block_measured():
    B = 0.6 * 32e9
    blk = ISI(profile="measured", H_measured=_H_gauss(B))
    assert np.max(np.abs(blk.coefficients()["h"] - gaussian(B, 64e9, 16))) < TOL


def test_block_measured_requires_H():
    with pytest.raises(ValueError):
        ISI(profile="measured")


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"taps": [1, 0.3j, -0.2]},
        {"profile": "echo", "echoes": ((1, 0.3), (5, -0.2j))},
        {"profile": "gaussian"},
        {"profile": "gaussian", "B_rel": 0.3},
        {"profile": "measured", "H_measured": _H_gauss(0.6 * 32e9)},
    ],
)
def test_block_profiles_energy_and_shape(kw):
    """Todos los perfiles: energía unitaria (salvo taps arbitrarios) y salida de la misma forma."""
    blk = ISI(**kw)
    h = blk.coefficients()["h"]
    if blk.profile != "taps":
        assert abs(np.sum(np.abs(h) ** 2) - 1) < TOL
    x = _crandn(np.random.default_rng(8), 2, 256)
    assert blk(x).shape == x.shape
    assert np.iscomplexobj(blk(x))


@pytest.mark.parametrize("kw", [{"Rs": 0.0}, {"sps": 0}, {"B_rel": 0.0}, {"profile": "gaussian", "K": 33}])
def test_block_rejects_invalid_params(kw):
    with pytest.raises(ValueError):
        ISI(**{"profile": "gaussian", **kw})
