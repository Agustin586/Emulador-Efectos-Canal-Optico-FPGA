"""Bloque 05 - Offset de fase de portadora (Nivel 1)  [EJEMPLO COMPLETO]

Es el bloque de referencia: muestra cómo se ve uno terminado. Los demás siguen
esta misma forma.

En simple: el oscilador local del receptor y la portadora recibida no están
alineados en fase, así que toda la señal aparece girada un ángulo constante.

Modelo:  y[n] = x[n] * exp(j * delta_theta)

Parámetros:
  delta_theta_rad : giro constante, en radianes. Por defecto 0 (sin efecto).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Block


@dataclass
class PhaseOffset(Block):
    delta_theta_rad: float = 0.0
    name = "phase_offset"

    def process(self, x: np.ndarray) -> np.ndarray:
        return x * np.exp(1j * self.delta_theta_rad)
