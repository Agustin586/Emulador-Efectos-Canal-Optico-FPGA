"""Bloque 06 - Ruido de fase del láser (Nivel 2)

En simple: la fase de la luz del láser "deambula" al azar con el tiempo. En cada
muestra la fase se corre un poco hacia un lado u otro y esos corrimientos se
acumulan (caminata aleatoria / proceso de Wiener). Cuanto más ancho el espectro
del láser (mayor linewidth), más rápida y pronunciada es la difusión de fase.

Modelo (proceso de Wiener discreto):
    theta[n] = theta[n-1] + d[n],   d[n] ~ N(0, sigma^2)
    sigma^2  = 2 * pi * linewidth_hz * Ts = 2 * pi * linewidth_hz / fs
    y[n]     = x[n] * exp(j * theta[n])

Parámetros:
    linewidth_hz : ancho de línea a mitad de altura (FWHM) del láser, en Hz.
                   Por defecto 0.0 (sin efecto).
    fs           : frecuencia de muestreo de la señal, en Hz (por defecto 2.0 GHz).
    seed         : semilla para el generador pseudoaleatorio (default_rng).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..common import Block


@dataclass
class PhaseNoise(Block):
    linewidth_hz: float = 0.0
    fs: float = 2.0e9
    seed: int = 0
    name: ClassVar[str] = "phase_noise"

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.linewidth_hz <= 0.0:
            return x

        ts = 1.0 / self.fs
        sigma = np.sqrt(2.0 * np.pi * self.linewidth_hz * ts)

        n_samples = x.shape[-1]
        rng = np.random.default_rng(self.seed)

        # Incrementos gaussianos i.i.d. de la caminata aleatoria
        delta_theta = rng.normal(loc=0.0, scale=sigma, size=n_samples)
        theta = np.cumsum(delta_theta)

        # Multiplica por exp(j * theta).
        # Si x es 1D (n,) o 2D (2, n), theta broadcast naturalmente sobre el último eje.
        return x * np.exp(1j * theta)
