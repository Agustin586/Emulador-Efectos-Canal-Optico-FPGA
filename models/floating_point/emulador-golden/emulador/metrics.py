"""Métricas comunes del golden model de ISI y CD.

Comparan los tres niveles de cada bloque (referencia ideal, modelo por bloques en
flotante y modelo en punto fijo) y caracterizan la ISI. Todos los tests y
notebooks las usan, por eso cada una se valida contra casos analíticos
(tests/test_metrics.py).

Convención de frecuencias (GUIA_ISI_CD.md §1 y referencia §3): los espectros
entran en el orden de numpy.fft (fftfreq); fftshift se aplica solo adentro, para
derivar o graficar.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Comparación entre modelos
# --------------------------------------------------------------------------
def nrmse(y: np.ndarray, ref: np.ndarray) -> float:
    """Error cuadrático medio normalizado (referencia §1, PDF Ec. 32).

        NRMSE = sqrt( sum |y - ref|^2 / sum |ref|^2 )

    ``y`` y ``ref`` tienen que tener la misma forma y estar YA alineados
    (descontado el retardo de cada modelo): alinear es responsabilidad del que
    llama. Acepta (n,) o (2, n); con dos polarizaciones suma sobre ambas.
    """
    y = np.asarray(y)
    ref = np.asarray(ref)
    if y.shape != ref.shape:
        raise ValueError(f"nrmse: formas distintas {y.shape} y {ref.shape} (¿faltó alinear?)")
    energy_ref = np.sum(np.abs(ref) ** 2)
    if energy_ref == 0:
        raise ValueError("nrmse: la referencia tiene energía nula")
    return float(np.sqrt(np.sum(np.abs(y - ref) ** 2) / energy_ref))


# --------------------------------------------------------------------------
# Respuesta en frecuencia
# --------------------------------------------------------------------------
def group_delay(H: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Retardo de grupo tau(f) = -dphi/domega de una respuesta en frecuencia.

    H  : respuesta en la grilla de una FFT de L puntos, en orden numpy.fft
         (f = fftfreq(L, 1/fs)).
    fs : frecuencia de muestreo [Hz].

    Devuelve (f, tau) con f ordenada de -fs/2 a fs/2 y tau en segundos.

    La fase se desenrolla (unwrap) antes de derivar: sin eso, los saltos de 2*pi
    aparecen como picos de retardo. La derivada usa diferencias de segundo orden
    también en los bordes (edge_order=2), así es exacta para fases lineales
    (retardo puro, FIR simétrico) y cuadráticas (CD sin beta3).

    El unwrap supone que la fase cambia menos de pi entre muestras de frecuencia:
    si no se cumple (retardos de más de L/2 muestras), usar una FFT más larga.
    """
    H = np.asarray(H)
    L = H.shape[-1]
    f = np.fft.fftshift(np.fft.fftfreq(L, d=1 / fs))
    phase = np.unwrap(np.angle(np.fft.fftshift(H)))
    tau = -np.gradient(phase, 2 * np.pi * f, edge_order=2)
    return f, tau
