"""Tests propios del bloque PMD (además de los de test_contract.py)."""
import numpy as np

from emulador.blocks import PMD
from emulador.common import gen_signal

FS = 64e9  # 32 GBd x 2 sps


def _signal(n_symbols=1024):
    return gen_signal("qam16", n_symbols, sps=2).samples


def test_dgd_de_dos_muestras_corre_una_muestra_cada_polarizacion():
    """Con DGD = 2 muestras, X se adelanta 1 muestra e Y se atrasa 1 muestra.

    La FFT es circular, por eso la comparación es contra np.roll.
    """
    x = _signal()
    dgd_ps = 2 / FS * 1e12  # 31.25 ps
    y = PMD(fs=FS, dgd_ps=dgd_ps)(np.stack([x, x]))
    assert np.allclose(y[0], np.roll(x, -1))  # X adelantada
    assert np.allclose(y[1], np.roll(x, +1))  # Y atrasada
