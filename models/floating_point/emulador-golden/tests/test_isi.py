"""ISI genérica (Parte A de GUIA_ISI_CD.md)."""
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from emulador.blocks import ISI
from emulador.isi import (
    K_rule,
    analog,
    analog_prototype,
    cd_short,
    echo,
    energy_captured,
    gaussian,
    isi_apply,
    isi_apply_stream,
    measured,
    normalize_energy,
)
from emulador.isi.profiles import _interp_table
from emulador.cd import cd_reference, cd_transfer
from emulador.config import D_si, Scenario
from emulador.metrics import group_delay, nrmse
from emulador.stimuli import comm_signal, impulse

TOL = 1e-12
DATA = Path(__file__).parent / "data"


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


# --------------------------------------------------------------------------
# profiles: analog (Bessel / Butterworth) (A3.1)
# --------------------------------------------------------------------------
KINDS = ["bessel", "butter"]
ORDERS = [2, 3, 4, 5, 6]


def _analog_h(kind, B_rel, m, sps=2, K=None):
    fs = sps * RS
    B = B_rel * RS
    K = K if K is not None else K_rule(B, RS, m)
    return analog(kind, B, fs, m, K), fs


def test_bessel_norm_mag():
    """norm="mag": B es el punto de -3 dB del prototipo (si no, B es otra cosa)."""
    B = 0.6 * RS
    b, a = analog_prototype("bessel", B, 4)
    _, H = signal.freqs(b, a, worN=[0.0, 2 * np.pi * B])
    assert 20 * np.log10(abs(H[1]) / abs(H[0])) == pytest.approx(-10 * np.log10(2), abs=1e-6)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("m", ORDERS)
@pytest.mark.parametrize("B_rel", [0.3, 0.4, 0.5])
def test_analog_3dB_and_energy_2sps(kind, m, B_rel):
    """Decisión 19: a 2 sps, -3 dB ±0,3 y energía >= 98,5 % para B <= 0,5·Rs."""
    h, fs = _analog_h(kind, B_rel, m)
    assert abs(np.sum(np.abs(h) ** 2) - 1) < TOL
    assert _H_rel_db(h, B_rel * RS, fs) == pytest.approx(-3.0, abs=0.3)
    K = h.size
    design = partial(analog, kind, B_rel * RS, fs, m, L=4096)
    assert energy_captured(design, K, L=4096) >= 0.985


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("m", ORDERS)
@pytest.mark.parametrize("B_rel", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
def test_analog_3dB_and_energy_4sps(kind, m, B_rel):
    """Decisión 19: a 4 sps vale todo el rango (peor desvío medido: 0,11 dB).

    A 2 sps, con B > 0,5·Rs, el prototipo no está limitado en banda en fs/2 y el
    -3 dB se corre hasta 0,66 dB: es el muestreo, no el diseño. K y d van al
    doble que a 2 sps para cubrir el mismo tiempo (la regla de K está en
    muestras a 2 sps).
    """
    fs, B = 4 * RS, B_rel * RS
    K2 = K_rule(B, RS, m)
    K, d = 2 * K2, 2 * (6 if K2 >= 32 else 3)
    h = analog(kind, B, fs, m, K, d=d)
    assert _H_rel_db(h, B, fs) == pytest.approx(-3.0, abs=0.3)
    design = partial(analog, kind, B, fs, m, d=d, L=4096)
    assert energy_captured(design, K, L=4096) >= 0.985


@pytest.mark.parametrize("kind", KINDS)
def test_analog_main_profile_2sps(kind):
    """Perfil principal (Bessel m = 4, B = 0,6·Rs, K = 16) y su par Butterworth, a 2 sps."""
    h, fs = _analog_h(kind, 0.6, 4)
    assert h.size == 16
    assert _H_rel_db(h, 0.6 * RS, fs) == pytest.approx(-3.0, abs=0.3)


def test_analog_is_causal_with_pre_delay():
    """El pico cae después del pre-retardo d (d = 3 con K = 16, d = 6 con K = 32)."""
    h16, _ = _analog_h("bessel", 0.6, 4)
    h32, _ = _analog_h("bessel", 0.4, 4)
    assert h32.size == 32
    assert 3 <= int(np.argmax(np.abs(h16))) < 16
    assert 6 <= int(np.argmax(np.abs(h32))) < 32
    assert np.isrealobj(h16)


def test_analog_rejects_bad_kind():
    with pytest.raises(ValueError):
        analog("cheby", 0.6 * RS, 2 * RS, 4, 16)


# --------------------------------------------------------------------------
# profiles: fidelidad contra el prototipo analógico (A3.2)
# --------------------------------------------------------------------------
NF = 8192  # grilla fina para comparar respuestas


def _fidelity(H_test, H_ref, fs, B, d=0.0):
    """Peor error en |f| <= B: magnitud relativa a f = 0 [dB] y retardo [muestras].

    H_test y H_ref en la grilla de fftfreq(NF, 1/fs). Al retardo de H_test se le
    descuenta d (el pre-retardo del diseño).
    """
    f, tau_t = group_delay(H_test, fs)
    _, tau_r = group_delay(H_ref, fs)
    Ht, Hr = np.fft.fftshift(H_test), np.fft.fftshift(H_ref)
    band = np.abs(f) <= B
    mag_t = 20 * np.log10(np.abs(Ht[band]) / np.abs(Ht[NF // 2]))
    mag_r = 20 * np.log10(np.abs(Hr[band]) / np.abs(Hr[NF // 2]))
    d_err = (tau_t[band] - tau_r[band]) * fs - d
    return np.max(np.abs(mag_t - mag_r)), np.max(np.abs(d_err))


def _prototype_on_grid(kind, B, m, fs):
    b, a = analog_prototype(kind, B, m)
    _, H = signal.freqs(b, a, worN=2 * np.pi * np.fft.fftfreq(NF, 1 / fs))
    return H


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("m", [3, 4, 5, 6])
@pytest.mark.parametrize("B_rel", [0.3, 0.4, 0.5, 0.6, 0.7])
def test_analog_fidelity(kind, m, B_rel):
    """Decisión 19: en la banda pasante, < 0,5 dB y < 0,4 muestras (m >= 3, B <= 0,7·Rs)."""
    h, fs = _analog_h(kind, B_rel, m)
    d = 6 if h.size >= 32 else 3
    B = B_rel * RS
    mag, dly = _fidelity(np.fft.fft(h, NF), _prototype_on_grid(kind, B, m, fs), fs, B, d)
    assert mag < 0.5
    assert dly < 0.4


def test_analog_main_profile_fidelity():
    """Perfil principal: 0,06 dB y 0,03 muestras (Ref. A.3: 0,04 dB y 0,03 muestras)."""
    h, fs = _analog_h("bessel", 0.6, 4)
    B = 0.6 * RS
    mag, dly = _fidelity(np.fft.fft(h, NF), _prototype_on_grid("bessel", B, 4, fs), fs, B, 3)
    assert mag < 0.1
    assert dly < 0.05


def test_bilinear_bessel_fails_the_same_criteria():
    """Decisión 6: el Bessel digital (transformación bilineal) se aparta del
    prototipo 0,53 dB y su retardo varía ~1 muestra en la banda; el diseño en
    frecuencia, 0,06 dB y 0,03 muestras."""
    fs, B, m = 2 * RS, 0.6 * RS, 4
    bz, az = signal.bessel(m, B / (fs / 2), norm="mag")  # digital: bilineal
    _, Hz = signal.freqz(bz, az, worN=2 * np.pi * np.fft.fftfreq(NF, 1 / fs) / fs)
    HP = _prototype_on_grid("bessel", B, m, fs)
    mag, _ = _fidelity(Hz, HP, fs, B)
    f, tau_z = group_delay(Hz, fs)
    _, tau_p = group_delay(HP, fs)
    dt = ((tau_z - tau_p) * fs)[np.abs(f) <= B]
    assert mag > 0.5
    assert np.max(np.abs(dt - dt.mean())) > 0.4  # sin pre-retardo: se compara la variación


# --------------------------------------------------------------------------
# Perfiles bessel/butter en el bloque y regresión del perfil principal (A3.3)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_block_analog_profiles(kind):
    blk = ISI(profile=kind, B_rel=0.5, order=3)
    expected = analog(kind, 0.5 * 32e9, 64e9, 3, 16)
    assert np.max(np.abs(blk.coefficients()["h"] - expected)) < TOL
    assert ISI(profile=kind, B_rel=0.4, order=4).coefficients()["h"].size == 32


def test_block_main_profile_regression():
    """Perfil principal (decisión 6): Bessel m = 4, B = 0,6·Rs, K = 16, a 32 GBd y 2 sps.

    tests/data/isi_bessel_m4_B06_K16.npy se generó con ``analog`` después de
    revisar los coeficientes a ojo y de que pasaran los tests de A3.1 y A3.2.
    """
    blk = ISI(profile="bessel")
    h = blk.coefficients()["h"]
    ref = np.load(DATA / "isi_bessel_m4_B06_K16.npy")
    assert h.size == 16
    assert np.max(np.abs(h - ref)) < TOL
    assert blk.delay_samples == 4  # pre-retardo d = 3 + ~1 muestra del Bessel


def test_block_main_profile_same_at_other_rate():
    """Mismo B/Rs y sps a 100 GBd: mismos taps (el diseño solo depende de B/fs)."""
    h32 = ISI(profile="bessel").coefficients()["h"]
    h100 = ISI(profile="bessel", Rs=100e9).coefficients()["h"]
    assert np.max(np.abs(h32 - h100)) < TOL


# --------------------------------------------------------------------------
# Perfil CD corta (A4.1)
# --------------------------------------------------------------------------
D_SMF = D_si(17.0)
LAM = 1550e-9
KM = 1e3


def test_cd_short_length():
    assert cd_short(64e9, D_SMF, LAM, 10 * KM).size == 5


def test_cd_short_constant_magnitude_and_quadratic_phase():
    h = cd_short(64e9, D_SMF, LAM, 50 * KM)
    assert abs(np.sum(np.abs(h) ** 2) - 1) < TOL
    assert np.max(np.abs(np.abs(h) - np.abs(h[0]))) < TOL
    n = np.arange(h.size) - h.size // 2
    phase = np.unwrap(np.angle(h))
    assert np.max(np.abs(phase - np.polyval(np.polyfit(n, phase, 2), n))) < 1e-10


def test_cd_short_sign_matches_cd_transfer():
    """Curvatura de la fase de H_FIR (FIR centrado) con el mismo signo que H_CD."""
    fs, z = 64e9, 30 * KM
    h = cd_short(fs, D_SMF, LAM, z)
    L = 8192
    H = np.fft.fftshift(np.fft.fft(np.roll(np.pad(h, (0, L - h.size)), -(h.size // 2))))
    f = np.fft.fftshift(np.fft.fftfreq(L, 1 / fs))
    band = np.abs(f) < fs / 8
    curv = np.polyfit(f[band], np.unwrap(np.angle(H[band])), 2)[0]
    curv_ref = np.polyfit(f[band], np.unwrap(np.angle(np.fft.fftshift(cd_transfer(L, fs, D_SMF, LAM, z))[band])), 2)[0]
    assert np.sign(curv) == np.sign(curv_ref) > 0


def test_cd_short_limit():
    """R <= 31 (K <= 32 y R impar): a 64 GSa/s, hasta ~55 km de SSMF."""
    assert cd_short(64e9, D_SMF, LAM, 55 * KM).size == 31
    with pytest.raises(ValueError):
        cd_short(64e9, D_SMF, LAM, 100 * KM)


def test_cd_short_R1_is_identity():
    """Decisión 20: con R = 1 (z <= 3 km a 64 GSa/s) el tap de Savory sería una
    rotación de 45° (NRMSE ~0,7); [1] queda a NRMSE <= 0,15 de la referencia."""
    assert np.array_equal(cd_short(64e9, D_SMF, LAM, 2 * KM), np.array([1.0 + 0j]))
    assert np.array_equal(cd_short(64e9, D_SMF, LAM, 0.0), np.array([1.0 + 0j]))


@pytest.mark.parametrize("z_km, max_nrmse", [(10, 0.40), (20, 0.25), (30, 0.25), (40, 0.20), (55, 0.20)])
def test_cd_short_against_reference(z_km, max_nrmse):
    """Decisión 20: NRMSE contra cd_reference (QAM16, 4096 símbolos, 2 sps).

    Medido: 0,38 (10 km), 0,23 (20), 0,19 (30), 0,17 (40), 0,14 (55). El
    truncamiento abrupto del chirp deja 2 a 5 dB de ondulación en la banda: el
    perfil es ilustrativo, la CD de verdad va por overlap-save.
    """
    z = z_km * KM
    x = comm_signal(Scenario(), n_symbols=4096).samples
    y = ISI(profile="cd_short", length_km=z_km)(x)
    assert nrmse(y, cd_reference(x, 64e9, D_SMF, LAM, z)) < max_nrmse


def test_block_cd_short():
    blk = ISI(profile="cd_short", length_km=30.0)
    h = blk.coefficients()["h"]
    assert np.max(np.abs(h - cd_short(64e9, D_SMF, LAM, 30 * KM))) < TOL
    assert blk.delay_samples == h.size // 2  # magnitud constante: el cursor es el tap central
    assert np.array_equal(ISI(profile="cd_short").coefficients()["h"], np.array([1.0 + 0j]))
    with pytest.raises(ValueError):
        ISI(profile="cd_short", length_km=100.0)
