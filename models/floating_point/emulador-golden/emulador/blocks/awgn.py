"""Bloque 02 - AWGN, ruido blanco gaussiano aditivo (Nivel 2)

En simple: se le suma a la señal un ruido aleatorio cuya potencia fija la
relación señal/ruido (SNR) del enlace.

Modelo:
    y[n] = x[n] + w[n]
    w[n] ~ CN(0, sigma^2),  con sigma^2 = P_s / (10^(snr_db / 10))

Parámetros:
    snr_db : relación señal/ruido en dB (por defecto 30.0 dB).
             Corresponde a la SNR por muestra en la banda de simulación.
    seed   : semilla para el generador pseudoaleatorio (default_rng).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..common import Block


@dataclass
class AWGN(Block):
    snr_db: float = 30.0
    seed: int = 0
    name: ClassVar[str] = "awgn"

    def process(self, x: np.ndarray) -> np.ndarray:
        p_signal = float(np.mean(np.abs(x) ** 2))
        if p_signal == 0.0 or self.snr_db is None or np.isposinf(self.snr_db) or np.isneginf(self.snr_db):
            return x

        snr_linear = 10.0 ** (self.snr_db / 10.0)
        p_noise = p_signal / snr_linear

        # Ruido gaussiano complejo circularmente simétrico:
        # Cada componente (I y Q) aporta varianza p_noise / 2
        sigma_comp = np.sqrt(p_noise / 2.0)
        rng = np.random.default_rng(self.seed)

        noise = sigma_comp * (
            rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
        )
        return x + noise
