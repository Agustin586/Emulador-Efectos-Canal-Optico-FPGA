"""Dispersión cromática: lógica pura en SI (GUIA_ISI_CD.md, Parte B).

  - physics : β₂, β₃, H_CD(ω) con la convención de signo positiva, largo del
              chirp R, tamaño de bloque N, taps de Savory y resumen.
"""
from .physics import betas, cd_N, cd_R, cd_summary, cd_transfer, savory_taps

__all__ = [
    "betas",
    "cd_N",
    "cd_R",
    "cd_summary",
    "cd_transfer",
    "savory_taps",
]
