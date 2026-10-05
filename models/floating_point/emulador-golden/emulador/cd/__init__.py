"""Dispersión cromática: lógica pura en SI (GUIA_ISI_CD.md, Parte B).

  - physics      : β₂, β₃, H_CD(ω) con la convención de signo positiva, largo del
                   chirp R, tamaño de bloque N, taps de Savory y resumen.
  - reference    : CD ideal con una FFT de toda la señal (verdad física).
  - overlap_save : CD por bloques, el algoritmo del hardware.
"""
from .overlap_save import cd_hbar, cd_overlap_save
from .physics import betas, cd_N, cd_R, cd_summary, cd_transfer, savory_taps
from .reference import cd_reference, cd_reference_pad

__all__ = [
    "betas",
    "cd_N",
    "cd_R",
    "cd_hbar",
    "cd_overlap_save",
    "cd_reference",
    "cd_reference_pad",
    "cd_summary",
    "cd_transfer",
    "savory_taps",
]
