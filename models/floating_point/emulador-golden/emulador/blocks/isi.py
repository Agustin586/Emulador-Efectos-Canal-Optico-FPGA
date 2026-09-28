"""Bloque 01 - ISI genérica (Nivel 1)  [PENDIENTE]

En simple: la energía de cada símbolo se "chorrea" sobre los vecinos, como si el
canal tuviera memoria. Se modela con un filtro FIR de respuesta configurable.

Modelo:  y[n] = sum_k h[k] * x[n-k]

Parámetros:
  taps : coeficientes h[k]. Por defecto un solo tap igual a 1 (sin distorsión).

TODO:
  - Implementar ``process`` con una convolución sobre el último eje, de modo que
    funcione para forma (n,) y para (2, n).
  - Decidir cómo se maneja el retardo/transitorio de la salida y documentarlo.
  - Elegir un canal de prueba y mostrar constelación y espectro antes y después.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common import Block


@dataclass
class ISI(Block):
    taps: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j]))
    name = "isi"

    def process(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ISI: pendiente de implementar")
