"""Dispersión cromática: lógica pura en SI (GUIA_ISI_CD.md, Parte B).

  - physics : β₂, β₃, H_CD(ω) con la convención de signo positiva, largo del
              chirp R y tamaño de bloque N.
"""
from .physics import betas, cd_N, cd_R, cd_transfer

__all__ = [
    "betas",
    "cd_N",
    "cd_R",
    "cd_transfer",
]
