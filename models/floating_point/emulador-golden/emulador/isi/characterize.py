"""Caracterización de la ISI de los perfiles (paso A5, PDF §3.5.5 y Tabla 4).

La ISI se mide de forma determinista, sin símbolos aleatorios: se arma la
respuesta total TX (RRC) -> canal h -> RX (RRC) muestreada a la tasa de
símbolos (``metrics.symbol_rate_response``) y se calcula la SNR equivalente,
-10·log10(potencia de ISI / potencia del cursor) (``metrics.isi_power_ratio``).

Dos formas de muestrear:
  - instante óptimo: el perfil se diseña con ``oversample`` veces más sps (K y
    d escalados para cubrir el mismo tiempo) y se elige la fase de muestreo
    que maximiza la SNR. Es la de la Tabla 4 del PDF.
  - grilla fija: a ``sps`` (2), muestreando en el cursor entero k0 = argmax|h|,
    como haría un receptor sin recuperación de temporización. El retardo
    fraccionario de los perfiles causales queda sin corregir.
"""
from __future__ import annotations

import numpy as np

from ..metrics import isi_power_ratio, symbol_rate_response
from .profiles import K_rule, analog, gaussian


def _taps(kind: str, B: float, fs: float, m: int, K: int, d: int) -> np.ndarray:
    L = max(8192, 4 * K)
    if kind == "gaussian":
        return gaussian(B, fs, K, L=L)
    if kind in ("bessel", "butter"):
        return analog(kind, B, fs, m, K, d=d, L=L)
    raise ValueError(f'isi_snr_eq: kind tiene que ser "gaussian", "bessel" o "butter": {kind!r}')


def _snr_db(p: np.ndarray) -> float:
    return float(-10 * np.log10(isi_power_ratio(p)))


def isi_snr_eq(
    kind: str,
    B_rel: float,
    m: int = 4,
    sps: int = 2,
    rolloff: float = 0.1,
    span: int = 32,
    oversample: int = 8,
) -> tuple[float, float]:
    """SNR equivalente de la ISI de un perfil pasa-bajos, en dB.

    - kind: "gaussian", "bessel" o "butter".
    - B_rel: ancho de banda a -3 dB relativo a Rs.
    - m: orden (Bessel y Butterworth; también entra en la regla de K).
    - sps: muestras por símbolo del emulador (grilla fija).
    - rolloff, span: RRC del transmisor y del filtro adaptado.
    - oversample: factor de sobremuestreo para buscar el instante óptimo.
    - return: (instante óptimo, grilla fija). Mayor es mejor.

    K = ``K_rule`` (en muestras a ``sps``) y d = 3 o 6, como el bloque ``ISI``.
    El resultado solo depende de B/Rs, no de Rs.
    """
    Rs = 1.0  # todo escala con Rs: se trabaja normalizado
    B = B_rel * Rs
    K = K_rule(B, Rs, m)
    d = 6 if K >= 32 else 3

    h = _taps(kind, B, sps * Rs, m, K, d)
    k0 = int(np.argmax(np.abs(h)))
    grid = _snr_db(symbol_rate_response(h, sps, rolloff, span, phase=k0 % sps))

    sps_f = sps * oversample
    h_f = _taps(kind, B, sps_f * Rs, m, K * oversample, d * oversample)
    optimal = max(_snr_db(symbol_rate_response(h_f, sps_f, rolloff, span, phase=ph)) for ph in range(sps_f))
    return optimal, grid
