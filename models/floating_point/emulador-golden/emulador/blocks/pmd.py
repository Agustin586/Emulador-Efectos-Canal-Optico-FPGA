"""Bloque 04 - PMD, dispersión por modo de polarización (Nivel 3)  [PENDIENTE]

En simple: la fibra no es perfectamente simétrica, así que las dos polarizaciones
de la luz viajan a velocidades apenas distintas y se mezclan entre sí. Se modela
como una cascada de K tramos de fibra birefringente, cada uno con una rotación de
polarización y un retardo entre las dos polarizaciones (DGD).

Este bloque necesita señal de DOS polarizaciones: forma (2, n), filas X e Y.

Modelo (una sección, en frecuencia, w = 2*pi*f):
    J_k(w) = R(theta_k) @ diag( exp(+j*w*tau_k/2), exp(-j*w*tau_k/2) ) @ R(theta_k)^H
    J_total(w) = J_K(w) @ ... @ J_1(w)
    [Yx(w), Yy(w)]^T = J_total(w) @ [Xx(w), Xy(w)]^T
  con tau_k el DGD de la sección y R(theta) una rotación 2x2.

Parámetros:
  fs         : frecuencia de muestreo, en Hz.
  n_sections : cantidad de secciones K (más secciones = estadística más fiel).
  dgd_ps     : DGD por sección, en ps. Por defecto 0 (sin efecto).
  seed       : semilla para los ángulos aleatorios de cada sección.

TODO:
  - Implementar ``process`` aplicando J_total(w) en frecuencia (FFT por polarización).
  - Verificar que con ángulos aleatorios el DGD total sigue una distribución
    Maxwelliana (simular muchas realizaciones y comparar el histograma).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Block


@dataclass
class PMD(Block):
    fs: float = 64e9
    n_sections: int = 10
    dgd_ps: float = 0.0
    seed: int = 0
    name = "pmd"

    def process(self, x: np.ndarray) -> np.ndarray:
        """Procesa la señal de entrada x (2, n) y devuelve la salida (2, n).

        Args:
            x: Señal de entrada, forma (2, n), filas X e Y.

        Returns:
            Señal de salida, forma (2, n), filas X e Y.
        """

        tao = self.dgd_ps * 1e-12  # DGD por sección en segundos
        n = x.shape[-1]
        w = 2 * np.pi * np.fft.fftfreq(n, 1 / self.fs)  # vector de frecuencias
        X = np.fft.fft(x, axis=-1)  # FFT por polarización
        H = np.stack([np.exp(+1j * w * tao / 2),      # X: se adelanta tao/2
                      np.exp(-1j * w * tao / 2)])     # Y: se atrasa tao/2
        return np.fft.ifft(H * X, axis=-1)

        
