"""Bloque 04 - PMD, dispersión por modo de polarización (Nivel 3)  [EN CURSO]

En simple: la fibra no es perfectamente simétrica, así que las dos polarizaciones
de la luz viajan a velocidades apenas distintas y se mezclan entre sí. Se modela
como una cascada de K tramos de fibra birefringente, cada uno con una rotación de
polarización y un retardo entre las dos polarizaciones (DGD).

Este bloque necesita señal de DOS polarizaciones: forma (2, n), filas X e Y.

Modelo (una sección, en frecuencia, w = 2*pi*f):
    J_k(w) = R_k @ diag( exp(+j*w*tau_k/2), exp(-j*w*tau_k/2) ) @ R_k^H
    J_total(w) = J_K(w) @ ... @ J_1(w)
    [Yx(w), Yy(w)]^T = J_total(w) @ [Xx(w), Xy(w)]^T
  con tau_k el DGD de la sección y R_k = R(theta_k, phi_k) una matriz 2x2 unitaria.
  Los ejes se sortean uniformes en la esfera de Poincaré; con solo theta el DGD
  total sale Rayleigh y no Maxwelliano (ver ``emulador/pmd/physics.py``).

Parámetros:
  fs         : frecuencia de muestreo, en Hz.
  n_sections : cantidad de secciones K (más secciones = estadística más fiel).
               Con K <= 10 el DGD total se aparta un poco de la Maxwelliana (nunca
               supera K*dgd_ps, la cola queda recortada); con K >= 15 coincide.
  dgd_ps     : DGD por sección, en ps. Por defecto 0 (sin efecto).
               DGD total medio = dgd_ps * sqrt(8*K/(3*pi))  (~2.9 * dgd_ps con K = 10).
  seed       : semilla para los ángulos aleatorios de cada sección.

Estado: golden model de referencia completo (cascada de K secciones, DGD total
Maxwelliano verificado con K = 20 en tests/test_pmd.py). Histograma:
``python -m scripts.pmd_maxwell``.

TODO:
  - Modelo de la arquitectura (rotaciones + FIR de Lagrange) y comparación contra
    esta referencia (paso 4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Block
from ..pmd import pmd_section, random_angles, total_dgd


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
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError(f"pmd: la entrada tiene que tener forma (2, n): forma {x.shape}")
        

        tau = self.dgd_ps * 1e-12  # DGD por sección en segundos
        thetas, phis = self.angles()

        for theta, phi in zip(thetas, phis):
            x = pmd_section(x, self.fs, tau, theta, phi)

        return x

    def angles(self) -> tuple[np.ndarray, np.ndarray]:
        """Ángulos (thetas, phis) de las K secciones de esta fibra.

        Un solo generador para los K ángulos: semillas distintas dan fibras
        independientes. Ejes uniformes en la esfera de Poincaré (``random_angles``).
        """
        return random_angles(np.random.default_rng(self.seed), self.n_sections)

    def total_dgd_ps(self) -> float:
        """DGD total de esta fibra (esta semilla), en ps, medido en la portadora."""
        thetas, phis = self.angles()
        return total_dgd(self.dgd_ps * 1e-12, thetas, phis) * 1e12
