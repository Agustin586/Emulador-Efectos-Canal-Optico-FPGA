"""Bloque 02 - AWGN, ruido blanco gaussiano aditivo (Nivel 2)  [PENDIENTE]

En simple: se le suma a la señal un ruido aleatorio cuya potencia fija la
relación señal/ruido (SNR) del enlace.

Modelo:  y[n] = x[n] + w[n],   w ~ ruido gaussiano complejo de varianza sigma^2

Parámetros:
  snr_db : relación señal/ruido en dB.
  seed   : semilla del generador, para poder repetir una corrida.

A DEFINIR ANTES DE IMPLEMENTAR (ver README):
  ¿La SNR es por muestra o por símbolo (Es/N0)? Con más de una muestra por
  símbolo la diferencia es un factor sps. Hay que acordar una sola definición.

TODO:
  - Implementar ``process`` con ``np.random.default_rng(seed)``.
  - Verificar que la SNR medida coincida con la pedida.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Block


@dataclass
class AWGN(Block):
    snr_db: float = 30.0
    seed: int = 0
    name = "awgn"

    def process(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("AWGN: pendiente de implementar")
