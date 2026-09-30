"""Métricas comunes (paso P2 de GUIA_ISI_CD.md). Se validan contra casos analíticos."""
import numpy as np
import pytest

from emulador.config import Scenario
from emulador.metrics import group_delay, nrmse

FS = Scenario().fs
TS = Scenario().Ts


def _noise(n=4096, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


# --------------------------------------------------------------------------
# nrmse
# --------------------------------------------------------------------------
def test_nrmse_of_identical_signals_is_zero():
    x = _noise()
    assert nrmse(x, x) == 0.0


def test_nrmse_of_scaled_signal():
    x = _noise()
    assert np.isclose(nrmse(1.01 * x, x), 0.01, rtol=0, atol=1e-12)


def test_nrmse_is_relative_to_reference_energy():
    x = _noise()
    e = 1e-3 * _noise(seed=1)
    expected = np.sqrt(np.sum(np.abs(e) ** 2) / np.sum(np.abs(x) ** 2))
    assert np.isclose(nrmse(100 * (x + e), 100 * x), expected, rtol=1e-12, atol=0)


def test_nrmse_accepts_two_polarizations():
    x = np.stack([_noise(seed=2), _noise(seed=3)])
    assert np.isclose(nrmse(1.01 * x, x), 0.01, rtol=0, atol=1e-12)


def test_nrmse_rejects_different_shapes():
    with pytest.raises(ValueError):
        nrmse(np.ones(10), np.ones(11))


def test_nrmse_rejects_zero_reference():
    with pytest.raises(ValueError):
        nrmse(np.ones(10), np.zeros(10))


# --------------------------------------------------------------------------
# group_delay
# --------------------------------------------------------------------------
def test_group_delay_of_pure_delay():
    """H = exp(-j 2 pi f d Ts) es un retardo de d muestras: tau = d·Ts en toda la banda."""
    L, d = 1024, 5
    f = np.fft.fftfreq(L, 1 / FS)
    H = np.exp(-2j * np.pi * f * d * TS)
    f_out, tau = group_delay(H, FS)
    assert np.allclose(tau, d * TS, rtol=1e-9, atol=0)


def test_group_delay_frequency_axis_is_sorted():
    L = 256
    f_out, _ = group_delay(np.ones(L, complex), FS)
    assert np.all(np.diff(f_out) > 0)
    assert np.allclose(np.sort(f_out), np.sort(np.fft.fftfreq(L, 1 / FS)), rtol=0, atol=0)


def test_group_delay_of_quadratic_phase():
    """Fase cuadrática phi = a·f^2 (como la CD): tau = -dphi/domega = -a·f/pi, exacto también en los bordes."""
    L, a = 2048, 3e-20  # a en rad/Hz^2: ~30 rad en el borde (obliga a unwrap)
    f = np.fft.fftfreq(L, 1 / FS)
    f_out, tau = group_delay(np.exp(1j * a * f**2), FS)
    assert np.allclose(tau, -a * f_out / np.pi, rtol=0, atol=1e-9 * np.max(np.abs(tau)))


def test_group_delay_of_symmetric_fir():
    """Un FIR real simétrico de K taps tiene fase lineal: retardo de (K-1)/2 muestras."""
    h = np.array([0.1, 0.3, 1.0, 0.3, 0.1])
    L = 512
    f_out, tau = group_delay(np.fft.fft(h, L), FS)
    assert np.allclose(tau, 2 * TS, rtol=1e-9, atol=0)
