"""Dispersión cromática (Parte B de GUIA_ISI_CD.md)."""
import numpy as np
import pytest

from emulador.cd import betas, cd_transfer
from emulador.config import C, D_si, S_si

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
