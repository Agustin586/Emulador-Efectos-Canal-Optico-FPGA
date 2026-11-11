"""Dispersión cromática: lógica pura en SI (GUIA_ISI_CD.md, Parte B).

  - physics      : β₂, β₃, H_CD(ω) con la convención de signo positiva, largo del
                   chirp R, tamaño de bloque N y resumen.
  - reference    : CD ideal con una FFT de toda la señal (verdad física).
  - overlap_save : CD por bloques, el algoritmo del hardware.
  - characterize : residuo de β₃ tras compensar solo β₂ (Tabla 17 del PDF).
"""
from .characterize import beta3_residual
from .overlap_save import cd_hbar, cd_overlap_save
from .physics import betas, cd_N, cd_R, cd_summary, cd_transfer
from .reference import cd_reference, cd_reference_pad

__all__ = [
    "beta3_residual",
    "betas",
    "cd_N",
    "cd_R",
    "cd_hbar",
    "cd_overlap_save",
    "cd_reference",
    "cd_reference_pad",
    "cd_summary",
    "cd_transfer",
]
