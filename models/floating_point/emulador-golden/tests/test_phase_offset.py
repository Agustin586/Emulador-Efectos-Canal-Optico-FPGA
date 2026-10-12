"""Tests específicos para el bloque PhaseOffset."""
import numpy as np
import pytest

from emulador.blocks.phase_offset import PhaseOffset
from emulador.common import gen_signal


def test_phase_offset_rotation():
    """Comprueba que rote todos los puntos exactamente el ángulo especificado."""
    sig = gen_signal("qam16", n_symbols=1024, sps=2)
    x = sig.samples

    delta = 0.45
    block = PhaseOffset(delta_theta_rad=delta)
    y = block(x)

    assert np.allclose(y, x * np.exp(1j * delta))
    assert np.allclose(np.abs(y), np.abs(x), atol=1e-12)


def test_phase_offset_zero():
    """Con delta_theta_rad = 0 no debe modificar la señal."""
    sig = gen_signal("qam16", n_symbols=512, sps=2)
    x = sig.samples

    y = PhaseOffset(delta_theta_rad=0.0)(x)
    assert np.array_equal(y, x)


def test_phase_offset_invalid_angle():
    with pytest.raises(ValueError, match="delta_theta_rad"):
        PhaseOffset(delta_theta_rad=np.nan)
