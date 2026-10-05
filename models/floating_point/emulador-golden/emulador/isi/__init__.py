"""ISI genérica: lógica pura en SI (GUIA_ISI_CD.md, Parte A).

  - fir      : filtro FIR causal y su versión por bloques (estructura del hardware).
  - profiles : diseño de los taps h[k] (energía unitaria).
"""
from .fir import isi_apply, isi_apply_stream
from .profiles import K_rule, echo, energy_captured, gaussian, measured, normalize_energy

__all__ = [
    "K_rule",
    "echo",
    "energy_captured",
    "gaussian",
    "isi_apply",
    "isi_apply_stream",
    "measured",
    "normalize_energy",
]
