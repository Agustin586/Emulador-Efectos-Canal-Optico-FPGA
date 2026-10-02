"""ISI genérica: lógica pura en SI (GUIA_ISI_CD.md, Parte A).

  - fir : filtro FIR causal y su versión por bloques (estructura del hardware).
"""
from .fir import isi_apply, isi_apply_stream

__all__ = [
    "isi_apply",
    "isi_apply_stream",
]
