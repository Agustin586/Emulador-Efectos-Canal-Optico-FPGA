"""Tests específicos para el bloque PhaseNoise (Ruido de fase del láser)."""
import numpy as np
import pytest

from emulador.blocks.phase_noise import PhaseNoise
from emulador.common import gen_signal


def test_phase_noise_variance_growth():
    """Comprueba que la varianza de la fase crezca linealmente con el tiempo según el modelo de Wiener.

    Var{theta[n] - theta[0]} = 2 * pi * linewidth * n * Ts
    """
    linewidth = 500e3  # 500 kHz
    fs = 2.0e9         # 2 GHz
    ts = 1.0 / fs
    n_samples = 2000
    n_runs = 800

    # Coleccionamos las trayectorias de fase de múltiples corridas
    x = np.ones(n_samples, dtype=complex)
    phases = np.zeros((n_runs, n_samples))

    for run_idx in range(n_runs):
        block = PhaseNoise(linewidth_hz=linewidth, fs=fs, seed=run_idx)
        y = block(x)
        phases[run_idx, :] = np.unwrap(np.angle(y))

    # Calculamos la varianza en función de n
    var_empirical = np.var(phases, axis=0)
    n_indices = np.arange(n_samples)
    var_theoretical = 2.0 * np.pi * linewidth * n_indices * ts

    # Ajuste lineal de la pendiente empírica
    slope_emp, _ = np.polyfit(n_indices[50:], var_empirical[50:], 1)
    slope_theor = 2.0 * np.pi * linewidth * ts

    assert np.isclose(slope_emp, slope_theor, rtol=0.10)


def test_phase_noise_envelope_preservation():
    """Comprueba que el ruido de fase preserve estrictamente la envolvente (|y[n]| == |x[n]|)."""
    sig = gen_signal("qam16", n_symbols=2048, sps=2)
    x = sig.samples

    block = PhaseNoise(linewidth_hz=1e6, fs=2e9, seed=12)
    y = block(x)

    assert np.allclose(np.abs(y), np.abs(x), atol=1e-12)


def test_phase_noise_zero_linewidth_noop():
    """Comprueba que un linewidth de 0 Hz devuelva exactamente la señal de entrada."""
    sig = gen_signal("qam16", n_symbols=1024, sps=2)
    x = sig.samples

    block = PhaseNoise(linewidth_hz=0.0, fs=2e9)
    y = block(x)

    assert np.array_equal(y, x)


def test_phase_noise_null_case():
    """Comprueba exhaustivamente el caso nulo: linewidth <= 0 Hz o configuración neutra.

    Verifica que el bloque se comporte como un no-op binario estricto en 1D y 2D.
    """
    sig = gen_signal("qam16", n_symbols=1024, sps=2)
    x = sig.samples

    # Linewidth 0 Hz exacto
    assert np.array_equal(PhaseNoise(linewidth_hz=0.0, fs=2e9)(x), x)

    # Parámetros por defecto (linewidth_hz=0.0)
    assert np.array_equal(PhaseNoise()(x), x)

    # Un linewidth negativo es una configuración inválida.
    with pytest.raises(ValueError, match="linewidth_hz"):
        PhaseNoise(linewidth_hz=-10e3, fs=2e9)

    # Polarización dual en caso nulo
    x_dual = np.stack([x, 2.0 * x])
    assert np.array_equal(PhaseNoise(linewidth_hz=0.0, fs=2e9)(x_dual), x_dual)


def test_phase_noise_dual_polarization_shared():
    """Comprueba que en polarización dual (2, n), ambas polarizaciones compartan el mismo láser."""
    sig = gen_signal("qam16", n_symbols=2048, sps=2)
    x = np.stack([sig.samples, 2.0 * sig.samples])  # (2, n)

    block = PhaseNoise(linewidth_hz=1e6, fs=2e9, seed=42)
    y = block(x)

    assert y.shape == x.shape
    # El fasor aplicado a X debe ser idéntico al aplicado a Y
    phasor_x = y[0] / x[0]
    phasor_y = y[1] / x[1]
    assert np.allclose(phasor_x, phasor_y, atol=1e-12)


def test_phase_noise_seed_determinism():
    """Comprueba el determinismo estricto mediante semilla (seed):
    - Mismo seed garantiza una trayectoria bit a bit idéntica (np.array_equal).
    - Distinto seed garantiza trayectorias estocásticamente independientes (not np.allclose).
    - Determinismo estricto tanto en 1D como en 2D (polarización dual).
    """
    sig = gen_signal("qam16", n_symbols=1024, sps=2)
    x = sig.samples

    # Caso 1D: Mismo seed debe ser idéntico bit a bit
    y1 = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=12345)(x)
    y2 = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=12345)(x)
    assert np.array_equal(y1, y2)

    # Distinto seed debe divergir
    y3 = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=54321)(x)
    assert not np.allclose(y1, y3)

    # Caso 2D (polarización dual): Determinismo estricto
    x_dual = np.stack([x, 1.5 * x])
    y_dual1 = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=777)(x_dual)
    y_dual2 = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=777)(x_dual)
    assert np.array_equal(y_dual1, y_dual2)


def test_phase_noise_seed_reproducibility():
    """Comprueba la reproducibilidad mediante la semilla."""
    sig = gen_signal("qam16", n_symbols=1024, sps=2)
    x = sig.samples

    y1 = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=99)(x)
    y2 = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=99)(x)
    y3 = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=100)(x)

    assert np.array_equal(y1, y2)
    assert not np.allclose(y1, y3)


def test_phase_noise_chunked_processing_matches_single_run():
    """La fase y el RNG deben continuar correctamente entre chunks."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=4096) + 1j * rng.normal(size=4096)

    whole = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=21)(x)
    chunked_block = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=21)
    chunked = np.concatenate([chunked_block(x[:1300]), chunked_block(x[1300:])])

    assert np.allclose(chunked, whole, rtol=0.0, atol=1e-13)


def test_phase_noise_reset_restarts_trajectory():
    """reset() debe restaurar la misma trayectoria para la misma semilla."""
    x = np.ones(512, dtype=complex)
    block = PhaseNoise(linewidth_hz=500e3, fs=2e9, seed=22)
    first = block(x)
    block.reset()
    second = block(x)

    assert np.array_equal(first, second)


def test_phase_noise_invalid_parameters():
    for linewidth, fs in [(np.nan, 2e9), (-1.0, 2e9), (1.0, 0.0), (1.0, -1.0), (1.0, np.nan)]:
        with pytest.raises(ValueError, match="phase_noise"):
            PhaseNoise(linewidth_hz=linewidth, fs=fs)
