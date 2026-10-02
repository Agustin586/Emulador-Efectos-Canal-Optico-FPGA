"""Dispersión cromática: lógica pura en SI (GUIA_ISI_CD.md, Parte B).

  - physics : β₂, β₃ y H_CD(ω) con la convención de signo positiva.
"""
from .physics import betas, cd_transfer

__all__ = [
    "betas",
    "cd_transfer",
]
