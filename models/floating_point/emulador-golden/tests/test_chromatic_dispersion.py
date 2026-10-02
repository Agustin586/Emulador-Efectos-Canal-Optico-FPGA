"""Dispersión cromática (Parte B de GUIA_ISI_CD.md)."""
import numpy as np
import pytest

from emulador.cd import betas, cd_N, cd_R, cd_summary, cd_transfer, savory_taps
from emulador.config import C, D_si, HardwareParams, S_si, Scenario
from emulador.metrics import group_delay

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
