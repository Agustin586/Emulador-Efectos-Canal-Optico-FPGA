"""Modelos en punto fijo bit a bit para la exploración de bits (paso C3 de GUIA_ISI_CD.md).

Utilidades fuera del núcleo del golden: no las usan los bloques ni ``REGISTRY``.
Se construyen sobre ``emulador.quantization`` y aritmética entera, con la
semántica de ``ac_fixed`` (decisiones 25, 26 y 32), así que sirven de
referencia bit a bit para el modelo de HALCON (decisiones 14 y 33).

  - integer : ``round_shift`` y ``saturate_int`` (cambios de formato en enteros).
  - isi_fir : ``isi_fir_fixed``, el FIR complejo de la ISI (C3.1).
"""
from .integer import round_shift, saturate_int
from .isi_fir import IsiFixedResult, isi_fir_fixed

__all__ = ["IsiFixedResult", "isi_fir_fixed", "round_shift", "saturate_int"]
