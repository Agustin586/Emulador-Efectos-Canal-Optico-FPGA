"""Bloque 03 - Dispersión cromática, CD (Nivel 3)  [PENDIENTE]

En simple: cada color (longitud de onda) viaja a distinta velocidad por la fibra,
así que el pulso se ensancha en el tiempo. Es un filtro que solo cambia la fase:
no altera la potencia de cada frecuencia, pero desfasa más las frecuencias altas.

Modelo (dominio de la frecuencia, f en Hz, banda base):
    H(f) = exp( -j * pi * D * lambda^2 * z * f^2 / c )
que equivale a exp(-j * D * lambda^2 * z * w^2 / (4*pi*c)) con w = 2*pi*f.
El signo depende de la convención de fase: definirlo y documentarlo.

Parámetros:
  fs            : frecuencia de muestreo de la señal, en Hz.
  D_ps_nm_km    : parámetro de dispersión (típico SMF: ~17 ps/(nm*km)).
  wavelength_nm : longitud de onda de la portadora (típico: 1550 nm).
  length_km     : largo del enlace. Por defecto 0 (sin efecto).

ATENCIÓN: pasar todo a unidades SI (s, m, Hz) antes de calcular. Además, revisar
la fórmula de la guía: en las referencias no aparece un factor 1/2 adicional.

TODO:
  - Implementar ``process`` con FFT: multiplicar el espectro por H(f) y volver.
  - Comprobar que la potencia se conserva y que aplicar el conjugado de H
    devuelve la señal original.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Block


@dataclass
class ChromaticDispersion(Block):
    fs: float = 2.0e9
    D_ps_nm_km: float = 17.0
    wavelength_nm: float = 1550.0
    length_km: float = 0.0
    name = "chromatic_dispersion"

    def process(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("CD: pendiente de implementar")
