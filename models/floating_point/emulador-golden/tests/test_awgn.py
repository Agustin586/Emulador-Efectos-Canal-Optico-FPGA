"""Tests específicos para el bloque AWGN."""
import numpy as np
import pytest

from emulador.blocks.awgn import AWGN
from emulador.common import gen_signal


def test_awgn_snr_accuracy():
    """Comprueba que la SNR medida coincida con la pedida en varios valores."""
    sig = gen_signal("qam16", n_symbols=50000, sps=2, seed=123)
    x = sig.samples

    for target_snr in [10.0, 20.0, 30.0]:
        block = AWGN(snr_db=target_snr, seed=42)
        y = block(x)
        noise = y - x

        p_s = np.mean(np.abs(x) ** 2)
        p_n = np.mean(np.abs(noise) ** 2)
        measured_snr = 10.0 * np.log10(p_s / p_n)

        # Tolerancia estadística para N = 100,000 muestras
        assert np.isclose(measured_snr, target_snr, atol=0.25)


def test_awgn_circular_symmetry():
    """Comprueba que el ruido sea complejo circularmente simétrico (I y Q equipotentes e incorrelados)."""
    x = np.ones(100000, dtype=complex)
    block = AWGN(snr_db=20.0, seed=7)
    noise = block(x) - x

    var_i = np.var(noise.real)
    var_q = np.var(noise.imag)
    cov_iq = np.mean((noise.real - np.mean(noise.real)) * (noise.imag - np.mean(noise.imag)))

    # Varianza I y Q deben ser estadísticamente indistinguibles
    assert np.isclose(var_i, var_q, rtol=0.05)
    # Covarianza cruzada debe ser cercana a cero
    assert np.isclose(cov_iq, 0.0, atol=1e-3)


def test_awgn_seed_reproducibility():
    """Comprueba que la misma semilla genere la misma corrida y distinta semilla genere otra."""
    sig = gen_signal("qam16", n_symbols=1024, sps=2)
    x = sig.samples

    y1 = AWGN(snr_db=25.0, seed=100)(x)
    y2 = AWGN(snr_db=25.0, seed=100)(x)
    y3 = AWGN(snr_db=25.0, seed=200)(x)

    assert np.array_equal(y1, y2)
    assert not np.allclose(y1, y3)


def test_awgn_dual_polarization():
    """Comprueba que funcione sobre señales con dos polarizaciones (2, n) e inyecte ruido independiente en cada una."""
    sig = gen_signal("qam16", n_symbols=4096, sps=2)
    x = np.stack([sig.samples, sig.samples])  # (2, n)

    block = AWGN(snr_db=20.0, seed=42)
    y = block(x)

    assert y.shape == x.shape
    noise_x = y[0] - x[0]
    noise_y = y[1] - x[1]

    # Las dos polarizaciones deben tener ruidos diferentes e independientes
    corr = np.abs(np.mean(noise_x * np.conj(noise_y))) / np.sqrt(
        np.mean(np.abs(noise_x) ** 2) * np.mean(np.abs(noise_y) ** 2)
    )
    assert corr < 0.05


def test_awgn_null_case():
    """Comprueba el caso nulo: +inf dB o potencia nula devuelven la señal intacta."""
    sig = gen_signal("qam16", n_symbols=1024, sps=2)
    x = sig.samples

    # +inf dB significa ruido nulo.
    y_inf = AWGN(snr_db=float("inf"), seed=42)(x)
    assert np.array_equal(y_inf, x)

    y_np_inf = AWGN(snr_db=np.inf, seed=42)(x)
    assert np.array_equal(y_np_inf, x)

    # Caso señal de potencia nula (ceros): devuelve ceros sin errores numéricos
    zeros = np.zeros(512, dtype=complex)
    y_zeros = AWGN(snr_db=20.0, seed=42)(zeros)
    assert np.array_equal(y_zeros, zeros)


def test_awgn_invalid_snr():
    """-inf y NaN no representan una SNR válida y deben rechazarse."""
    for snr in (-np.inf, np.nan):
        with pytest.raises(ValueError, match="snr_db"):
            AWGN(snr_db=snr)
