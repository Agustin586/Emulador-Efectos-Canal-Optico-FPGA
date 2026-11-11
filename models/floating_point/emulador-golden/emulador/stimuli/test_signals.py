"""Señales de prueba física: impulso, pulso gaussiano y tono.

  impulse        : valida h[n] (la salida de un FIR a un impulso son sus taps).
  gaussian_pulse : valida el ensanchamiento por CD (tiene solución analítica).
  tone           : valida |H(f)| y la fase en un solo bin de la FFT.

Todas devuelven complex128, forma (n,).
"""
from __future__ import annotations

import numpy as np


def impulse(n: int, pos: int = 0) -> np.ndarray:
    """Impulso unitario δ[m - pos] de largo n.

    Ejemplo:
        >>> impulse(4, 1).real.tolist()
        [0.0, 1.0, 0.0, 0.0]
    """
    if not (0 <= pos < n):
        raise ValueError(f"pos fuera de rango [0, {n}): {pos!r}")
    x = np.zeros(n, dtype=complex)
    x[pos] = 1.0
    return x


def centered_time(n: int, fs: float) -> np.ndarray:
    """Eje de tiempo [s] con t = 0 en la muestra n // 2: t[m] = (m - n//2) / fs."""
    return (np.arange(n) - n // 2) / fs


def gaussian_pulse(T0: float, fs: float, n: int) -> np.ndarray:
    """Pulso gaussiano A(t) = exp(-t^2 / (2 T0^2)), centrado en la muestra n // 2.

    T0 : semiancho a 1/e de intensidad [s]. |A|^2 = exp(-t^2/T0^2) tiene ancho
         rms T0/sqrt(2) (ver ``metrics.rms_width``).
    fs : frecuencia de muestreo [Sa/s].
    n  : largo; tiene que alcanzar para que el pulso decaiga en los bordes (y,
         con CD, para que quepa ensanchado).

    El eje de tiempo correspondiente es ``centered_time(n, fs)``.
    """
    if T0 <= 0:
        raise ValueError(f"T0 debe ser positivo: {T0!r}")
    t = centered_time(n, fs)
    return np.exp(-(t**2) / (2 * T0**2)).astype(complex)


def tone(k: int, n: int) -> np.ndarray:
    """Tono complejo exp(j 2π k m / n), m = 0 … n-1.

    Cae exactamente en el bin k de ``np.fft.fft`` de largo n (sin fuga), cuya
    frecuencia es ``np.fft.fftfreq(n, 1/fs)[k]``. k negativo equivale a n + k.
    """
    m = np.arange(n)
    return np.exp(2j * np.pi * k * m / n)
