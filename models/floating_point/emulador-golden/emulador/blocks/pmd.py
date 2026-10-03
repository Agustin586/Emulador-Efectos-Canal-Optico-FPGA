"""Bloque 04 - PMD, dispersión por modo de polarización (Nivel 3)  [EN CURSO]

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

Estado: una sección (paso 2). La física está en ``emulador/pmd/physics.py``.

TODO:
  - Cascada de ``n_sections`` secciones con ángulos aleatorios (paso 3).
  - Verificar que con ángulos aleatorios el DGD total sigue una distribución
    Maxwelliana (simular muchas realizaciones y comparar el histograma).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Block
from ..pmd import pmd_section


@dataclass
class PMD(Block):
    fs: float = 64e9
    n_sections: int = 10
    dgd_ps: float = 0.0
    seed: int = 0
    name = "pmd"

    def process(self, x: np.ndarray) -> np.ndarray:
        """Procesa la señal de entrada x (2, n) y devuelve la salida (2, n).

        Por ahora aplica UNA sección, con un ángulo theta sorteado con ``seed``
        (paso 2). La cascada de ``n_sections`` secciones es el paso 3.

        Args:
            x: Señal de entrada, forma (2, n), filas X e Y.

        Returns:
            Señal de salida, forma (2, n), filas X e Y.
        """
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError(f"pmd: la entrada tiene que tener forma (2, n): forma {x.shape}")
        

        tau = self.dgd_ps * 1e-12  # DGD por sección en segundos
    

        for i in range (self.n_sections):
            theta = np.random.default_rng(self.seed + i).uniform(0, 2 * np.pi)
            x = pmd_section(x, self.fs, tau, theta)
            print(f"Sección {i+1}/{self.n_sections}: tau={tau*1e12:.2f} ps, theta={theta:.2f} rad")

        return x
