"""Dispersión cromática (Parte B de GUIA_ISI_CD.md)."""
import numpy as np
import pytest

from emulador.blocks import ISI, ChromaticDispersion
from emulador.cd import (
    beta3_residual,
    betas,
    cd_N,
    cd_R,
    cd_hbar,
    cd_overlap_save,
    cd_reference,
    cd_reference_pad,
    cd_summary,
    cd_transfer,
    savory_taps,
)
from emulador.config import C, D_si, HardwareParams, S_si, Scenario
from emulador.metrics import group_delay, nrmse, rms_width
from emulador.stimuli import centered_time, comm_signal, gaussian_pulse, impulse

TOL = 1e-12
FS = 64e9
LAM = 1550e-9
D = D_si(17.0)
S = S_si(0.057)
KM = 1e3


# --------------------------------------------------------------------------
# physics: betas, cd_transfer (B1.1)
# --------------------------------------------------------------------------
def test_betas_table_values():
    """PDF Tabla 15: SMF a 1550 nm, D = 17 ps/(nm·km), S = 0,057 ps/(nm²·km)."""
    b2, b3 = betas(D, LAM, S)
    assert b2 / 1e-27 == pytest.approx(-21.7, rel=0.01)  # ps²/km = 1e-27 s²/m
    assert b3 / 1e-39 == pytest.approx(0.13, rel=0.03)  # ps³/km = 1e-39 s³/m


def test_betas_S0_gives_beta3_exactly_zero():
    b2, b3 = betas(D, LAM)
    assert b3 == 0.0
    assert betas(D, LAM, 0.0) == (b2, 0.0)


def test_cd_transfer_S0_identical_to_beta2_only():
    L, z = 4096, 100 * KM
    b2, _ = betas(D, LAM)
    w = 2 * np.pi * np.fft.fftfreq(L, 1 / FS)
    H_b2 = np.exp(-1j * (b2 / 2 * w**2) * z)
    assert np.array_equal(cd_transfer(L, FS, D, LAM, z), H_b2)
    assert np.array_equal(cd_transfer(L, FS, D, LAM, z, S=0.0), H_b2)


def test_cd_transfer_positive_sign():
    """Decisión 2: H_CD(f) = exp(+j π D λ² z f² / c). Fase < π en |f| < 2 GHz."""
    L, z = 4096, 10 * KM
    f = np.fft.fftfreq(L, 1 / FS)
    band = np.abs(f) < 2e9
    expected = np.pi * D * LAM**2 * z * f[band] ** 2 / C
    assert expected.max() < np.pi
    phase = np.angle(cd_transfer(L, FS, D, LAM, z)[band])
    assert np.max(np.abs(phase - expected)) < TOL


@pytest.mark.parametrize("z_km", [100, 2000])
@pytest.mark.parametrize("S_", [0.0, S])
def test_cd_transfer_all_pass(z_km, S_):
    H = cd_transfer(4096, FS, D, LAM, z_km * KM, S=S_)
    assert np.max(np.abs(np.abs(H) - 1)) < TOL


@pytest.mark.parametrize("S_", [0.0, S])
def test_cd_transfer_composition(S_):
    L, z1, z2 = 2048, 37 * KM, 63 * KM
    H1 = cd_transfer(L, FS, D, LAM, z1, S=S_)
    H2 = cd_transfer(L, FS, D, LAM, z2, S=S_)
    H12 = cd_transfer(L, FS, D, LAM, z1 + z2, S=S_)
    assert np.max(np.abs(H1 * H2 - H12)) < TOL


def test_cd_transfer_zero_length_is_identity():
    assert np.array_equal(cd_transfer(512, FS, D, LAM, 0.0, S=S), np.ones(512, complex))


# --------------------------------------------------------------------------
# physics: retardo de grupo (B1.2)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("z_km", [10, 100, 1000])
def test_group_delay_slope(z_km):
    """τ(f) = -(D λ² z / c)·f con β₃ = 0. Detecta el factor ½ y el signo invertido."""
    z = z_km * KM
    f, tau = group_delay(cd_transfer(8192, FS, D, LAM, z), FS)
    band = np.abs(f) < FS / 4  # los bordes tienen error numérico del gradiente
    slope, _ = np.polyfit(f[band], tau[band], 1)
    expected = -D * LAM**2 * z / C
    assert slope == pytest.approx(expected, rel=1e-3, abs=0)


def test_group_delay_slope_value_100km():
    """Valor de referencia: SSMF, 100 km, 1550 nm -> -13,6 ps/GHz."""
    f, tau = group_delay(cd_transfer(8192, FS, D, LAM, 100 * KM), FS)
    band = np.abs(f) < FS / 4
    slope, _ = np.polyfit(f[band], tau[band], 1)
    assert slope / (1e-12 / 1e9) == pytest.approx(-13.6, rel=0.01)


def test_group_delay_parabola_with_beta3():
    """Con β₃: τ(ω) = β₂ z ω + (β₃ z / 2) ω².

    ``abs=0`` en los approx: los coeficientes en SI son ~1e-20 y ~1e-35, muy por
    debajo de la tolerancia absoluta por defecto de pytest (1e-12).
    """
    z = 1000 * KM
    b2, b3 = betas(D, LAM, S)
    f, tau = group_delay(cd_transfer(8192, FS, D, LAM, z, S=S), FS)
    band = np.abs(f) < FS / 4
    c2, c1, _ = np.polyfit(2 * np.pi * f[band], tau[band], 2)
    assert c1 == pytest.approx(b2 * z, rel=0.01, abs=0)
    assert c2 == pytest.approx(b3 * z / 2, rel=0.01, abs=0)


# --------------------------------------------------------------------------
# physics: largo del chirp R y tamaño de bloque N (B1.3)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fs, z_km, R",
    [
        (40e9, 10, 3),
        (40e9, 500, 109),
        (64e9, 10, 5),
        (64e9, 100, 55),
        (64e9, 500, 279),
        (64e9, 1000, 559),
        (64e9, 2000, 1117),
    ],
)
def test_cd_R_table(fs, z_km, R):
    """Ref. B.2 (PDF Ec. 26). Con c = 3e8 en lugar de C exacta, 1000 y 2000 km dan 557 y 1115."""
    assert cd_R(fs, D, LAM, z_km * KM) == R


def test_cd_R_zero_length_and_sign_of_D():
    assert cd_R(FS, D, LAM, 0.0) == 1
    assert cd_R(FS, -D, LAM, 500 * KM) == cd_R(FS, D, LAM, 500 * KM)


@pytest.mark.parametrize(
    "z_km, N",
    [(0, 128), (10, 128), (100, 128), (200, 128), (500, 512), (1000, 1024), (2000, 2048)],
)
def test_cd_N_table(z_km, N):
    """Decisión 5: N = max(128, menor potencia de 2 >= R)."""
    assert cd_N(FS, D, LAM, z_km * KM) == N


def test_cd_N_default_N_max_is_hardware_params():
    assert cd_N(FS, D, LAM, 2000 * KM) <= HardwareParams().N_max


def test_cd_N_exceeds_N_max():
    with pytest.raises(ValueError):
        cd_N(FS, D, LAM, 2000 * KM, N_max=1024)


def test_cd_N_custom_N_min():
    assert cd_N(FS, D, LAM, 10 * KM, N_min=16) == 16
    assert cd_N(FS, D, LAM, 100 * KM, N_min=16) == 64


# --------------------------------------------------------------------------
# physics: taps de Savory y resumen (B1.4)
# --------------------------------------------------------------------------
def _fir_phase_curvature(h, band_frac=0.25, L=8192):
    """Curvatura (coef. de f²) de la fase de la respuesta del FIR centrado en n = 0."""
    R = h.size
    H = np.fft.fft(np.roll(np.pad(h, (0, L - R)), -(R // 2)))
    f = np.fft.fftshift(np.fft.fftfreq(L, 1 / FS))
    H = np.fft.fftshift(H)
    band = np.abs(f) < band_frac * FS
    return np.polyfit(f[band], np.unwrap(np.angle(H[band])), 2)[0]


def test_savory_length_is_R():
    for z_km in (10, 100, 500):
        assert savory_taps(FS, D, LAM, z_km * KM).size == cd_R(FS, D, LAM, z_km * KM)
    assert savory_taps(FS, D, LAM, 10 * KM).size == 5


def test_savory_constant_magnitude():
    z = 100 * KM
    h = savory_taps(FS, D, LAM, z)
    expected = np.sqrt(C / (D * LAM**2 * z)) / FS  # sqrt(c Ts² / (D λ² z))
    assert np.max(np.abs(np.abs(h) - expected)) < TOL


def test_savory_quadratic_phase():
    h = savory_taps(FS, D, LAM, 100 * KM)
    n = np.arange(h.size) - h.size // 2
    phase = np.unwrap(np.angle(h))
    resid = phase - np.polyval(np.polyfit(n, phase, 2), n)
    assert np.max(np.abs(resid)) < 1e-10


@pytest.mark.parametrize("z_km", [10, 100])
def test_savory_sign_matches_cd_transfer(z_km):
    """Los taps emulan el canal (no lo compensan): misma curvatura de fase que H_CD."""
    z = z_km * KM
    curv = _fir_phase_curvature(savory_taps(FS, D, LAM, z))
    expected = np.pi * D * LAM**2 * z / C
    assert np.sign(curv) == np.sign(expected)
    if z_km >= 100:  # con pocos taps la truncación deforma la fase
        assert curv == pytest.approx(expected, rel=0.01, abs=0)


def test_savory_negative_D_is_conjugate():
    z = 50 * KM
    assert np.max(np.abs(savory_taps(FS, -D, LAM, z) - np.conj(savory_taps(FS, D, LAM, z)))) < TOL


def test_savory_no_dispersion_is_identity():
    assert np.array_equal(savory_taps(FS, D, LAM, 0.0), np.array([1.0 + 0j]))
    assert np.array_equal(savory_taps(FS, 0.0, LAM, 100 * KM), np.array([1.0 + 0j]))


def test_cd_summary_table_15():
    """PDF Tabla 15: SSMF, 100 km, 64 GSa/s (32 GBd, 2 sps)."""
    s = cd_summary(Scenario(), D, 100 * KM, S)
    assert s["beta2_ps2_km"] == pytest.approx(-21.7, rel=0.01)
    assert s["beta3_ps3_km"] == pytest.approx(0.13, rel=0.03)
    assert s["gd_slope_ps_GHz"] == pytest.approx(-13.6, rel=0.01)
    assert s["R"] == 55
    assert s["N"] == 128
    assert s["L_D_km"] == pytest.approx(45, rel=0.01)
    assert s["Dz_ps_nm"] == pytest.approx(1700)


def test_cd_summary_S0_and_worst_case():
    s = cd_summary(Scenario(), D, 100 * KM)
    assert s["beta3_ps3_km"] == 0.0
    # PDF Tabla 14, peor caso: D = 23 ps/(nm·km), 2000 km, 64 GSa/s -> R = 1509, N = 2048
    s = cd_summary(Scenario(), D_si(23.0), 2000 * KM)
    assert (s["R"], s["N"]) == (1509, 2048)
    assert s["N_fits_hw"] is True


def test_cd_summary_N_beyond_hw():
    s = cd_summary(Scenario(), D, 4000 * KM)
    assert s["N"] == 4096
    assert s["N_fits_hw"] is False


def test_cd_summary_no_dispersion():
    s = cd_summary(Scenario(), 0.0, 100 * KM)
    assert s["R"] == 1
    assert s["L_D_km"] == np.inf


# --------------------------------------------------------------------------
# reference: CD ideal con una FFT de toda la señal (B2.1)
# --------------------------------------------------------------------------
def _qam(n_symbols=2048):
    return comm_signal(Scenario(), mod="qam16", n_symbols=n_symbols).samples


def test_reference_identity():
    x = _qam()
    assert nrmse(cd_reference(x, FS, D, LAM, 0.0), x) < 1e-14


def test_reference_alignment_and_length():
    """full=False: mismo largo que x y y[n] corresponde a x[n] (= full[pad + n])."""
    x, z = _qam(), 100 * KM
    y = cd_reference(x, FS, D, LAM, z, S=S)
    y_full = cd_reference(x, FS, D, LAM, z, S=S, full=True)
    pad = cd_reference_pad(FS, D, LAM, z)
    assert y.shape == x.shape
    assert np.array_equal(y, y_full[pad : pad + x.size])
    # con β₃ = 0 el chirp es simétrico: un impulso se ensancha sin moverse
    n0 = 300
    y_imp = np.abs(cd_reference(impulse(600, n0), FS, D, LAM, z))
    assert np.max(np.abs(y_imp[n0 - 50 : n0] - y_imp[n0 + 50 : n0 : -1])) < 1e-12


def test_reference_energy_full():
    """Decisión 18: full=True devuelve las L muestras, así la energía se conserva."""
    x = _qam()
    y = cd_reference(x, FS, D, LAM, 500 * KM, S=S, full=True)
    assert abs(np.sum(np.abs(y) ** 2) / np.sum(np.abs(x) ** 2) - 1) < 1e-12


@pytest.mark.parametrize("z_km", [100, 500, 2000])
def test_reference_reversibility(z_km):
    """(D, S) y después (-D, -S). Decisión 18: NRMSE < 1e-5, no 1e-12.

    Las colas del chirp discreto decaen como ~1/n y el relleno de R + 64 no las
    contiene: el piso crece con z y baja con el largo de la señal. Con 4096
    símbolos QAM16: 1,0e-6 (100 km), 2,8e-6 (500 km) y 7,3e-6 (2000 km); con
    2048 símbolos, 2000 km da 2,0e-5. Muy por debajo del 1e-2 del overlap-save.
    """
    x, z = _qam(4096), z_km * KM
    y = cd_reference(x, FS, D, LAM, z, S=S, full=True)
    back = cd_reference(y, FS, -D, LAM, z, S=-S, full=True)
    off = cd_reference_pad(FS, D, LAM, z) + cd_reference_pad(FS, -D, LAM, z)
    assert nrmse(back[off : off + x.size], x) < 1e-5


def test_reference_reversibility_same_grid_exact():
    """Sobre la misma grilla de L puntos, H* deshace H exactamente."""
    x, z = _qam(), 500 * KM
    y = cd_reference(x, FS, D, LAM, z, S=S, full=True)
    H = cd_transfer(y.size, FS, D, LAM, z, S=S)
    back = np.fft.ifft(np.fft.fft(y) * np.conj(H))
    pad = cd_reference_pad(FS, D, LAM, z)
    assert nrmse(back[pad : pad + x.size], x) < 1e-12
    assert np.max(np.abs(back[: pad])) < 1e-12


def test_reference_two_polarizations():
    x = np.stack([_qam(), 1j * _qam()[::-1]])
    for full in (False, True):
        y = cd_reference(x, FS, D, LAM, 100 * KM, S=S, full=full)
        assert y.shape[0] == 2
        for p in range(2):
            assert np.max(np.abs(y[p] - cd_reference(x[p], FS, D, LAM, 100 * KM, S=S, full=full))) < 1e-12


# --------------------------------------------------------------------------
# reference: ensanchamiento del pulso gaussiano (B2.2)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("z_km", [25, 50, 75, 100])
def test_reference_gaussian_pulse_broadening(z_km):
    """PDF Ec. 22-23: T(z)/T0 = sqrt(1 + (z/L_D)²), L_D = T0²/|β₂| (11,2 km).

    fs = 256 GSa/s (T0 = 4·Ts, sin aliasing) y n = 4096 muestras (16 ns): a
    100 km el pulso llega a ~9·T0 y queda dentro de la ventana.
    """
    T0, fs, n, z = 15.6e-12, 256e9, 4096, z_km * KM
    b2, _ = betas(D, LAM)
    L_D = T0**2 / abs(b2)
    assert L_D / KM == pytest.approx(11.2, rel=0.01)
    t = centered_time(n, fs)
    a0 = gaussian_pulse(T0, fs, n)
    az = cd_reference(a0, fs, D, LAM, z)
    ratio = rms_width(az, t) / rms_width(a0, t)
    assert ratio == pytest.approx(np.sqrt(1 + (z / L_D) ** 2), rel=0.01)


# --------------------------------------------------------------------------
# overlap_save: cd_hbar, cd_overlap_save (B3.1)
# --------------------------------------------------------------------------
def _noise(n, seed=0, shape=()):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape + (n,)) + 1j * rng.standard_normal(shape + (n,))) / np.sqrt(2)


def _ols_vs_reference(x, z, N=None):
    """NRMSE del overlap-save contra la referencia, descontando el retardo N/2."""
    N = N if N is not None else cd_N(FS, D, LAM, z)
    y = cd_overlap_save(x, cd_hbar(N, FS, D, LAM, z, S=S))
    r = cd_reference(x, FS, D, LAM, z, S=S)
    m = x.size - 2 * N
    return nrmse(y[N // 2 : N // 2 + m], r[:m])


@pytest.mark.parametrize("z_km", [100, 500, 1000, 2000])
def test_overlap_save_against_reference_qam(z_km):
    """Decisión 21: con QAM16 y N = cd_N, NRMSE < 1e-2 (medido 1,6e-3 … 2e-4)."""
    x = _qam(10_000)
    assert _ols_vs_reference(x, z_km * KM) < 1e-2


def test_overlap_save_worst_point_of_N128():
    """229 km es el último z con N = 128 (R = 127): NRMSE 9,8e-3 con QAM16."""
    assert cd_N(FS, D, LAM, 229 * KM) == 128
    assert _ols_vs_reference(_qam(10_000), 229 * KM) < 1.5e-2


@pytest.mark.parametrize("z_km", [100, 500, 1000, 2000])
def test_overlap_save_against_reference_noise(z_km):
    """Decisión 21: ruido blanco (toda la banda, peor caso), NRMSE < 2e-2 (medido 0,7e-2 … 1,4e-2).

    Las colas del chirp que no entran en N muestras son las de frecuencia
    instantánea cerca de Nyquist: la QAM16 casi no tiene energía ahí, el ruido sí.
    """
    assert _ols_vs_reference(_noise(20_000, seed=z_km), z_km * KM) < 2e-2


def test_hbar_all_pass_and_delay_factor():
    N, z = 256, 300 * KM
    Hbar = cd_hbar(N, FS, D, LAM, z, S=S)
    assert Hbar.shape == (2 * N,)
    assert np.max(np.abs(np.abs(Hbar) - 1)) < TOL
    k = np.fft.fftfreq(2 * N) * 2 * N
    expected = cd_transfer(2 * N, FS, D, LAM, z, S=S) * np.exp(-2j * np.pi * k * (N / 2) / (2 * N))
    assert np.max(np.abs(Hbar - expected)) < TOL


def test_hbar_S0_identical_to_beta2_only():
    assert np.array_equal(cd_hbar(128, FS, D, LAM, 100 * KM), cd_hbar(128, FS, D, LAM, 100 * KM, S=0.0))


def test_overlap_save_zero_length_is_pure_delay():
    """Con z = 0, H̄ es un retardo puro de N/2: la salida es x corrida N/2 muestras."""
    N = 128
    x = _noise(1000)
    y = cd_overlap_save(x, cd_hbar(N, FS, D, LAM, 0.0))
    assert y.size >= x.size
    assert np.max(np.abs(y[: N // 2])) < 1e-12
    assert np.max(np.abs(y[N // 2 : N // 2 + x.size - N // 2] - x[: x.size - N // 2])) < 1e-12


def test_overlap_save_two_polarizations():
    N, z = 256, 300 * KM
    Hbar = cd_hbar(N, FS, D, LAM, z)
    x = _noise(3000, shape=(2,))
    y = cd_overlap_save(x, Hbar)
    for p in range(2):
        assert np.max(np.abs(y[p] - cd_overlap_save(x[p], Hbar))) < 1e-12


def test_overlap_save_impulse_is_centered_chirp():
    """Un impulso en n0 sale como el chirp centrado en n0 + N/2 (β₃ = 0: simétrico en módulo)."""
    N, z, n0 = 512, 400 * KM, 100
    x = np.zeros(4 * N, complex)
    x[n0] = 1.0
    y = np.abs(cd_overlap_save(x, cd_hbar(N, FS, D, LAM, z)))
    c = n0 + N // 2
    assert np.max(np.abs(y[c - 100 : c] - y[c + 100 : c : -1])) < 1e-12


# --------------------------------------------------------------------------
# Bloque ChromaticDispersion (B3.2)
# --------------------------------------------------------------------------
def test_block_default_is_identity():
    """length_km = 0: H̄ es un retardo puro de N/2 y process lo descuenta."""
    x = _qam()
    blk = ChromaticDispersion()
    assert blk.N == 128 and blk.delay_samples == 64
    assert nrmse(blk(x), x) < 1e-14


def test_block_hbar_matches_functions():
    blk = ChromaticDispersion(length_km=300.0, S_ps_nm2_km=0.057)
    N = cd_N(FS, D, LAM, 300 * KM)
    c = blk.coefficients()
    assert c["N"] == N == blk.N
    assert np.max(np.abs(c["Hbar"] - cd_hbar(N, FS, D, LAM, 300 * KM, S=S))) < TOL
    assert np.max(np.abs(np.abs(c["Hbar"]) - 1)) < TOL


def test_block_S0_hbar_identical_to_beta2_only():
    a = ChromaticDispersion(length_km=100.0).coefficients()["Hbar"]
    b = ChromaticDispersion(length_km=100.0, S_ps_nm2_km=0.0).coefficients()["Hbar"]
    assert np.array_equal(a, b)
    assert np.array_equal(a, cd_hbar(128, FS, D, LAM, 100 * KM))


@pytest.mark.parametrize("z_km", [100, 500])
def test_block_aligned_vs_raw(z_km):
    """process = process_hw adelantado N/2 (exacto); el retardo declarado es N/2."""
    blk = ChromaticDispersion(length_km=z_km, S_ps_nm2_km=0.057)
    x = _qam()
    y, y_hw = blk(x), blk.process_hw(x)
    h = blk.delay_samples
    assert h == blk.N // 2
    assert y.shape == y_hw.shape == x.shape
    assert np.array_equal(y[: x.size - h], y_hw[h:])


@pytest.mark.parametrize("z_km", [100, 500, 2000])
def test_block_against_reference(z_km):
    """La salida alineada sigue a la referencia (QAM16, decisión 21), sin bordes."""
    blk = ChromaticDispersion(length_km=z_km, S_ps_nm2_km=0.057)
    x = _qam(10_000)
    r = cd_reference(x, FS, D, LAM, z_km * KM, S=S)
    N = blk.N
    assert nrmse(blk(x)[N : x.size - N], r[N : x.size - N]) < 1e-2


def test_block_two_polarizations():
    blk = ChromaticDispersion(length_km=200.0)
    x = np.stack([_qam(), 1j * _qam()[::-1]])
    y, y_hw = blk(x), blk.process_hw(x)
    assert y.shape == x.shape
    for p in range(2):
        assert np.max(np.abs(y[p] - blk(x[p]))) < 1e-12
        assert np.max(np.abs(y_hw[p] - blk.process_hw(x[p]))) < 1e-12


def test_block_explicit_N_and_validation():
    assert ChromaticDispersion(length_km=100.0, N=512).N == 512
    for bad in (127, 0, 64):  # impar, cero, menor que R = 111 a 200 km
        with pytest.raises(ValueError):
            ChromaticDispersion(length_km=200.0, N=bad)
    with pytest.raises(ValueError):  # cd_N supera N_max = 2048
        ChromaticDispersion(length_km=4000.0)


def test_block_reversible_with_negative_D():
    """+D y luego -D (fibra compensadora): vuelve a la entrada fuera de los bordes."""
    x = _qam(8192)
    y = ChromaticDispersion(length_km=500.0)(x)
    back = ChromaticDispersion(length_km=500.0, D_ps_nm_km=-17.0)(y)
    N = 512
    assert nrmse(back[2 * N : -2 * N], x[2 * N : -2 * N]) < 1e-2


def test_from_scenario():
    esc = Scenario(Rs=100e9, sps=2, wavelength=1560e-9)
    cd = ChromaticDispersion.from_scenario(esc, length_km=50.0)
    assert cd.fs == 200e9 and cd.wavelength_nm == pytest.approx(1560.0) and cd.length_km == 50.0
    isi = ISI.from_scenario(esc, profile="gaussian")
    assert (isi.Rs, isi.sps) == (100e9, 2)
    assert np.max(np.abs(isi.coefficients()["h"] - ISI(profile="gaussian").coefficients()["h"])) < TOL


# --------------------------------------------------------------------------
# Residuo de β₃ tras compensar solo β₂ (B4.1, PDF Tabla 17)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("Rs, evm", [(32e9, 0.007), (64e9, 0.056), (100e9, 0.22)])
def test_beta3_residual_table_17(Rs, evm):
    """1000 km de SSMF (S = 0,057), compensación con -D y S = 0. ±5 % relativo.

    Medido: 0,69 / 5,49 / 21,2 %.
    """
    r = beta3_residual(Scenario(Rs=Rs), D, 1000 * KM, S)
    assert r.evm == pytest.approx(evm, rel=0.05)


def test_beta3_residual_200GBd_destroys_constellation():
    """Decisión 23: EVM 69 % (SNR equivalente 3,2 dB). Con evm_opt (ganancia
    óptima) la EVM no puede pasar de 100 %, así que el criterio es > 50 %."""
    r = beta3_residual(Scenario(Rs=200e9), D, 1000 * KM, S)
    assert r.evm > 0.5


@pytest.mark.parametrize("Rs", [32e9, 100e9])
def test_beta3_residual_control_without_beta3(Rs):
    """S = 0 en el canal y en la compensación: queda el piso del RRC (~0,1 %)."""
    r = beta3_residual(Scenario(Rs=Rs), D, 1000 * KM, 0.0)
    assert r.evm < 0.002


@pytest.mark.parametrize("Rs, evm", [(32e9, 0.0029), (64e9, 0.022), (100e9, 0.083)])
def test_beta3_residual_ideal_timing(Rs, evm):
    """Decisión 23: con temporización ideal queda solo la ISI de β₃ (±10 %).

    La Tabla 17 (grilla de 2 sps) mezcla un error de temporización:
    0,69 % -> 0,29 %, 5,5 % -> 2,2 % y 21 % -> 8,3 %.
    """
    r = beta3_residual(Scenario(Rs=Rs), D, 1000 * KM, S, timing="ideal")
    assert r.evm == pytest.approx(evm, rel=0.10)
    assert r.evm <= beta3_residual(Scenario(Rs=Rs), D, 1000 * KM, S).evm + 1e-12


def test_beta3_residual_ideal_is_monotonic_in_rate():
    """Con temporización ideal el residuo crece con la tasa (con la grilla de 2 sps no siempre)."""
    evm = [beta3_residual(Scenario(Rs=Rs * 1e9), D, 500 * KM, S, timing="ideal").evm for Rs in (32, 50, 64, 100, 150, 200)]
    assert all(a < b for a, b in zip(evm, evm[1:]))


def test_beta3_residual_grows_with_rate_and_length():
    evm = {(Rs, z): beta3_residual(Scenario(Rs=Rs), D, z * KM, S).evm for Rs in (64e9, 100e9) for z in (500, 1000)}
    assert evm[64e9, 500] < evm[64e9, 1000] < evm[100e9, 1000]
    assert evm[100e9, 500] < evm[100e9, 1000]
