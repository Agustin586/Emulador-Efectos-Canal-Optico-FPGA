"""Física del PMD: rotación de polarización y retardo diferencial (DGD), en SI.

Señal: forma (2, n), filas X e Y; el tiempo es el último eje.

Una sección de fibra birrefringente (docstring de ``blocks/pmd.py``):
    J(w) = R(theta) @ diag( exp(+j*w*tau/2), exp(-j*w*tau/2) ) @ R(theta)^H
  1. R^H : proyecta la señal sobre los ejes rápido/lento de la fibra (girados theta).
  2. D   : un eje se adelanta tau/2 y el otro se atrasa tau/2.
  3. R   : vuelve a expresar la señal en X e Y.
"""
from __future__ import annotations

import numpy as np


def rotation(theta: float) -> np.ndarray:
    """Matriz de rotación 2x2 (real, unitaria): R(theta).

    - theta: ángulo de rotación, en radianes.
    - return: array (2, 2).
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def dgd_delay(x: np.ndarray, fs: float, tau: float) -> np.ndarray:
    """Retardo diferencial entre las dos filas, aplicado en frecuencia.

    La fila 0 se adelanta tau/2 (exp(+j*w*tau/2)) y la fila 1 se atrasa tau/2
    (exp(-j*w*tau/2)). Todos los factores tienen módulo 1: no cambia la potencia.
    La FFT es circular: lo que sale por un extremo entra por el otro.

    - x: señal (2, n).
    - fs: frecuencia de muestreo, en Hz.
    - tau: DGD, en segundos.
    - return: señal (2, n).
    """
    n = x.shape[-1]
    w = 2 * np.pi * np.fft.fftfreq(n, 1 / fs)  # frecuencia angular de cada bin
    X = np.fft.fft(x, axis=-1)  # FFT por polarización
    H = np.stack([np.exp(+1j * w * tau / 2),   # fila 0: se adelanta tau/2
                  np.exp(-1j * w * tau / 2)])  # fila 1: se atrasa tau/2
    return np.fft.ifft(H * X, axis=-1)


def pmd_section(x: np.ndarray, fs: float, tau: float, theta: float) -> np.ndarray:
    """Una sección de PMD: y = R(theta) @ D(tau) @ R(theta)^H @ x.

    R no depende de la frecuencia, así que se aplica en el tiempo. Como R es
    real, R^H = R.T.

    - x: señal (2, n), filas X e Y.
    - fs: frecuencia de muestreo, en Hz.
    - tau: DGD de la sección, en segundos.
    - theta: ángulo de los ejes de la fibra respecto de X/Y, en radianes.
    - return: señal (2, n).
    """
    R = rotation(theta)
    return R @ dgd_delay(R.T @ x, fs, tau)
