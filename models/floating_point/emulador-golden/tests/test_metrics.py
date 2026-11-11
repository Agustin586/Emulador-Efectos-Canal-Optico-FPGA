"""Métricas comunes (paso P2 de GUIA_ISI_CD.md). Se validan contra casos analíticos."""
import numpy as np
import pytest

from emulador.config import Scenario
from emulador.metrics import (
    evm_opt,
    group_delay,
    isi_power_ratio,
    nrmse,
    peak_distortion,
    rms_width,
    snr_eq_db,
    symbol_rate_response,
)

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


# --------------------------------------------------------------------------
# evm_opt y snr_eq_db
# --------------------------------------------------------------------------
def _orthogonal_error(ref, rel_power, seed=5):
    """Error e ortogonal a ref (sum conj(ref)·e = 0) con |e|^2 = rel_power·|ref|^2."""
    e = _noise(len(ref), seed)
    e = e - np.vdot(ref, e) / np.vdot(ref, ref) * ref
    return e * np.sqrt(rel_power) * np.linalg.norm(ref) / np.linalg.norm(e)


def test_evm_opt_ignores_gain_and_phase():
    ref = _noise()
    assert evm_opt(2 * np.exp(0.7j) * ref, ref) <= 1e-12


def test_evm_opt_with_orthogonal_error():
    """Con ganancia óptima, EVM^2 = N/(S+N) (no N/S): acá N/S = 0.01."""
    ref = _noise()
    rx = ref + _orthogonal_error(ref, 0.01)
    assert np.isclose(evm_opt(rx, ref) ** 2, 0.01 / 1.01, rtol=1e-12, atol=0)


def test_evm_opt_discard_edges():
    ref = _noise(1000)
    rx = ref.copy()
    rx[:10] = 0
    rx[-10:] = 0
    assert evm_opt(rx, ref) > 0.1
    assert evm_opt(rx, ref, discard=10) <= 1e-12


def test_evm_opt_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        evm_opt(np.ones(10), np.ones(11))
    with pytest.raises(ValueError):
        evm_opt(np.zeros(10), np.ones(10))


def test_snr_eq_db():
    """N/S = 0.01 -> EVM^2 = 1/101 -> SNR_eq = 10·log10(101) = 20.04 dB."""
    ref = _noise()
    rx = ref + _orthogonal_error(ref, 0.01)
    assert np.isclose(snr_eq_db(rx, ref), 10 * np.log10(101), rtol=0, atol=1e-9)


def test_snr_eq_db_without_error_is_infinite():
    ref = _noise()
    assert snr_eq_db(ref, ref) == np.inf


# --------------------------------------------------------------------------
# rms_width
# --------------------------------------------------------------------------
def _gaussian(T0, t, t1=0.0):
    return np.exp(-((t - t1) ** 2) / (2 * T0**2))


def test_rms_width_of_gaussian_pulse():
    """a = exp(-t^2/2T0^2) -> |a|^2 = exp(-t^2/T0^2) -> sigma = T0/sqrt(2)."""
    T0 = 15.6e-12
    t = np.arange(-4096, 4096) * 0.5e-12
    assert np.isclose(rms_width(_gaussian(T0, t), t), T0 / np.sqrt(2), rtol=1e-3, atol=0)


def test_rms_width_ignores_shift_scale_and_phase():
    T0 = 15.6e-12
    t = np.arange(-4096, 4096) * 0.5e-12
    ref = rms_width(_gaussian(T0, t), t)
    a = 3.0 * np.exp(1j * 1e11 * t) * _gaussian(T0, t, t1=200e-12)
    assert np.isclose(rms_width(a, t), ref, rtol=1e-9, atol=0)


def test_rms_width_rejects_invalid_inputs():
    t = np.arange(10.0)
    with pytest.raises(ValueError):
        rms_width(np.ones(9), t)
    with pytest.raises(ValueError):
        rms_width(np.zeros(10), t)


# --------------------------------------------------------------------------
# peak_distortion e isi_power_ratio
# --------------------------------------------------------------------------
def test_no_isi():
    p = np.array([0.0, 1.0, 0.0])
    assert peak_distortion(p) == 0.0
    assert isi_power_ratio(p) == 0.0


def test_single_echo():
    a = 0.3
    p = np.array([1.0, a])
    assert np.isclose(peak_distortion(p), a, rtol=0, atol=1e-12)
    assert np.isclose(isi_power_ratio(p), a**2, rtol=0, atol=1e-12)


def test_cursor_is_largest_magnitude_even_if_negative_or_complex():
    p = np.array([0.2, -1.0, 0.3j])
    assert np.isclose(peak_distortion(p), 0.5, rtol=0, atol=1e-12)
    assert np.isclose(isi_power_ratio(p), 0.13, rtol=0, atol=1e-12)


def test_isi_metrics_reject_zero_response():
    with pytest.raises(ValueError):
        peak_distortion(np.zeros(4))
    with pytest.raises(ValueError):
        isi_power_ratio(np.zeros(4))


# --------------------------------------------------------------------------
# symbol_rate_response
# --------------------------------------------------------------------------
def test_symbol_rate_response_without_channel():
    """RRC * RRC (coseno realzado) muestreado en el instante óptimo: cursor 1 y
    ISI < -40 dB, que es el piso del RRC truncado (roll-off 0.1, span 32)."""
    p = symbol_rate_response([1.0], sps=2, rolloff=0.1)
    assert np.isclose(np.max(np.abs(p)), 1.0, rtol=0, atol=1e-12)
    assert isi_power_ratio(p) < 1e-4


def test_symbol_rate_response_with_one_symbol_echo():
    """Con 2 sps, un eco de un símbolo está 2 taps más adelante: ISI ~ |a|^2."""
    a = 0.3
    p = symbol_rate_response([1.0, 0.0, a], sps=2, rolloff=0.1)
    assert np.isclose(isi_power_ratio(p), a**2, rtol=0, atol=1e-3)


def test_symbol_rate_response_sampling_phase():
    """Un retardo de una muestra (medio símbolo a 2 sps) se corrige con phase=1."""
    h = [0.0, 1.0]
    assert isi_power_ratio(symbol_rate_response(h, sps=2, rolloff=0.1, phase=1)) < 1e-4
    assert isi_power_ratio(symbol_rate_response(h, sps=2, rolloff=0.1, phase=0)) > 0.1


def test_symbol_rate_response_rejects_invalid_phase():
    with pytest.raises(ValueError):
        symbol_rate_response([1.0], sps=2, rolloff=0.1, phase=2)
