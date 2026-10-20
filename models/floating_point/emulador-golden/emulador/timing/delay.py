"""Retardo fraccionario de una señal muestreada.

Señal: el tiempo es el último eje; vale para (n,) y para (2, n).
Convención: mu > 0 ATRASA la señal, y[n] = x(n - mu). Con mu = 1, y = np.roll(x, 1).
"""
from __future__ import annotations

import numpy as np


def ideal_delay(x: np.ndarray, mu: float) -> np.ndarray:
    """Retardo fraccionario exacto, aplicado en frecuencia: H(w) = exp(-j*w*mu).

    w en rad/muestra. Todos los factores tienen módulo 1: no cambia la potencia.
    La FFT es circular: lo que sale por un extremo entra por el otro.

    - x: señal, el tiempo es el último eje.
    - mu: retardo, en muestras (puede ser no entero o negativo).
    - return: señal de la misma forma que x.
    """
    n = x.shape[-1]
    w = 2 * np.pi * np.fft.fftfreq(n)  # frecuencia angular de cada bin, en rad/muestra
    return np.fft.ifft(np.fft.fft(x, axis=-1) * np.exp(-1j * w * mu), axis=-1)


def linear_delay(x: np.ndarray, mu: float) -> np.ndarray:
    """Retardo fraccionario aproximado con un interpolador lineal (FIR de 2 taps).

        y[n] = (1 - mu) * x[n] + mu * x[n-1]

    Une con una recta las dos muestras vecinas y toma el punto a distancia mu.
    Respuesta en frecuencia: H(w) = (1 - mu) + mu * exp(-j*w). Es un pasabajos:
    atenúa las frecuencias altas, más cuanto más cerca está mu de 0.5.
    Borde: x[-1] = 0, como un registro de FPGA recién reseteado.

    - x: señal, el tiempo es el último eje.
    - mu: retardo, en muestras (0 <= mu <= 1).
    - return: señal de la misma forma que x.
    """
    if not 0 <= mu <= 1:
        raise ValueError(f"linear_delay: mu tiene que estar en [0, 1]: mu = {mu}")
    x_prev = np.concatenate([np.zeros_like(x[..., :1]), x[..., :-1]], axis=-1)  # x[n-1]
    return (1 - mu) * x + mu * x_prev


def _shift(x: np.ndarray, d: int) -> np.ndarray:
    """x[n - d] con ceros en los bordes (no circular). d > 0 atrasa, d < 0 adelanta."""
    y = np.zeros_like(x)
    if d >= 0:
        y[..., d:] = x[..., : x.shape[-1] - d]
    else:
        y[..., :d] = x[..., -d:]
    return y


def cubic_taps(mu: float) -> np.ndarray:
    """Coeficientes del interpolador de Lagrange de 3er orden, para x[n+1], x[n], x[n-1], x[n-2].

    Pasa un polinomio de grado 3 por las 4 muestras y lo evalúa en n - mu:
        h_d = prod_{k != d} (mu - k) / (d - k),   d, k en {-1, 0, 1, 2}

    - mu: retardo, en muestras (0 <= mu <= 1).
    - return: array (4,), coeficientes de x[n+1], x[n], x[n-1], x[n-2].
    """
    return np.array([
        -mu * (mu - 1) * (mu - 2) / 6,
        (mu + 1) * (mu - 1) * (mu - 2) / 2,
        -(mu + 1) * mu * (mu - 2) / 2,
        (mu + 1) * mu * (mu - 1) / 6,
    ])


def cubic_delay(x: np.ndarray, mu: float) -> np.ndarray:
    """Retardo fraccionario aproximado con un interpolador de Lagrange cúbico (FIR de 4 taps).

        y[n] = h0 * x[n+1] + h1 * x[n] + h2 * x[n-1] + h3 * x[n-2]

    Usa 2 muestras a cada lado del punto buscado (entre x[n-1] y x[n]), así que
    aproxima mejor que el lineal. Es exacto sobre polinomios de grado <= 3.
    Usa x[n+1] (una muestra "del futuro"): en hardware se agrega 1 muestra de
    latencia y queda causal. Bordes: ceros afuera de la señal.

    - x: señal, el tiempo es el último eje.
    - mu: retardo, en muestras (0 <= mu <= 1).
    - return: señal de la misma forma que x.
    """
    if not 0 <= mu <= 1:
        raise ValueError(f"cubic_delay: mu tiene que estar en [0, 1]: mu = {mu}")
    h = cubic_taps(mu)
    return sum(h_d * _shift(x, d) for h_d, d in zip(h, (-1, 0, 1, 2)))


def delay_error_db(y: np.ndarray, y_ref: np.ndarray, discard: int = 64) -> float:
    """Error de un retardo aproximado contra la referencia, en dB (más negativo = mejor).

        10 * log10( sum|y - y_ref|^2 / sum|y_ref|^2 )

    Descarta ``discard`` muestras de cada borde (transitorio circular del ideal y
    ceros de los interpoladores).
    """
    y, y_ref = y[..., discard:-discard], y_ref[..., discard:-discard]
    return float(10 * np.log10(np.sum(np.abs(y - y_ref) ** 2) / np.sum(np.abs(y_ref) ** 2)))
