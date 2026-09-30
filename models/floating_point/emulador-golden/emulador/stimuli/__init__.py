"""Estímulos de prueba (GUIA_ISI_CD.md §1, decisión 10, y paso P3).

  - prbs         : bits pseudoaleatorios reproducibles en hardware (ITU-T O.150).
  - mapper       : bits -> símbolos PAM4/QAM16 con Gray, potencia 1.
  - pulse        : RRC (shape) y señal completa PRBS -> Gray -> RRC (comm_signal).
  - test_signals : impulso, pulso gaussiano y tono (pruebas físicas).
"""
from .mapper import BITS_PER_SYMBOL, map_gray
from .prbs import prbs
from .pulse import CommSignal, comm_signal, shape
from .test_signals import centered_time, gaussian_pulse, impulse, tone

__all__ = [
    "BITS_PER_SYMBOL",
    "CommSignal",
    "centered_time",
    "comm_signal",
    "gaussian_pulse",
    "impulse",
    "map_gray",
    "prbs",
    "shape",
    "tone",
]
