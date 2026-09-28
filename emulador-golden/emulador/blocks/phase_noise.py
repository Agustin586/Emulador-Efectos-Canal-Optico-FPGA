"""Bloque 06 - Ruido de fase del láser (Nivel 2)  [PENDIENTE]

En simple: la fase de la luz del láser "deambula" al azar con el tiempo. En cada
muestra la fase se corre un poco hacia un lado u otro y esos corrimientos se
acumulan (caminata aleatoria). Cuanto más ancho el láser, más deambula.

Modelo (proceso de Wiener discreto):
    theta[n] = theta[n-1] + d[n],    d[n] ~ N(0, sigma^2),   sigma^2 = 2*pi*linewidth*Ts
    y[n]     = x[n] * exp(j * theta[n])
con Ts = 1/fs el período de muestreo de la señal que entra al bloque.

Parámetros:
  linewidth_hz : ancho de línea del láser. Por defecto 0 (sin efecto).
  fs           : frecuencia de muestreo de la señal, en Hz.
  seed         : semilla.

TODO:
  - Implementar ``process`` con ``np.cumsum`` sobre incrementos gaussianos.
  - Verificar que la varianza de theta[n] - theta[0] crece linealmente con n.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Block


@dataclass
class PhaseNoise(Block):
    linewidth_hz: float = 0.0
    fs: float = 2.0e9
    seed: int = 0
    name = "phase_noise"

    def process(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Ruido de fase: pendiente de implementar")
