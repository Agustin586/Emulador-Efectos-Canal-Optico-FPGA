"""Estímulos de prueba (GUIA_ISI_CD.md §1, decisión 10, y paso P3).

  - prbs         : bits pseudoaleatorios reproducibles en hardware (ITU-T O.150).
  - mapper       : bits -> símbolos PAM4/QAM16 con Gray, potencia 1.
"""
from .mapper import BITS_PER_SYMBOL, map_gray
from .prbs import prbs

__all__ = ["BITS_PER_SYMBOL", "map_gray", "prbs"]
