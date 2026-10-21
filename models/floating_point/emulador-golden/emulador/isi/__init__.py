"""ISI genérica: lógica pura en SI (GUIA_ISI_CD.md, Parte A).

  - fir          : filtro FIR causal y su versión por bloques (estructura del hardware).
  - profiles     : diseño de los taps h[k] (energía unitaria).
  - characterize : SNR equivalente de la ISI (Tabla 4 del PDF).
"""
from .characterize import isi_snr_eq
from .fir import isi_apply, isi_apply_stream
from .profiles import (
    K_rule,
    analog,
    analog_prototype,
    echo,
    energy_captured,
    gaussian,
    measured,
    normalize_energy,
)

__all__ = [
    "K_rule",
    "analog",
    "analog_prototype",
    "echo",
    "energy_captured",
    "gaussian",
    "isi_apply",
    "isi_apply_stream",
    "isi_snr_eq",
    "measured",
    "normalize_energy",
]
