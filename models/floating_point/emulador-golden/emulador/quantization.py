"""Utilidades de cuantización para la exploración de bits (paso C2 de GUIA_ISI_CD.md).

Convención de formato (decisión 25): la de ``ac_fixed<W, I, true>`` de Algorithmic C,
que usa HALCON.
  - W: bits totales; I: bits enteros, incluido el signo; W - I fraccionarios.
  - Paso q = 2^-(W - I); rango [-2^(I-1), 2^(I-1) - q].
  - Ejemplo: ``ac_fixed<14, 1>`` (entrada de la CD y de la ISI, decisión 32) tiene
    q = 2^-13 y rango [-1, 1 - 2^-13].

Redondeo y desborde (decisión 26):
  - ``"half_even"`` (por defecto, ``AC_RND_CONV``): al más cercano, empates a par.
    Es ``np.round``, así que Python y HALCON coinciden bit a bit.
  - ``"half_up"`` (``AC_RND``): floor(x/q + 0,5), empates hacia +inf. La decisión 32
    lo usa en los productos de la CD.
  - ``"trunc"`` (``AC_TRN``): floor(x/q), hacia -inf.
  - ``saturate=True`` (``AC_SAT``, por defecto) satura al rango; ``False``
    (``AC_WRAP``) desborda en complemento a dos.

Los complejos se cuantizan por separado en la parte real y en la imaginaria.
Es la herramienta de la exploración (C3) y la referencia bit a bit del modelo en
punto fijo: no depende de dónde viva ese modelo (decisión 14).
"""
from __future__ import annotations

import numpy as np

ROUNDINGS = ("half_even", "half_up", "trunc")
_W_MAX = 53  # con W <= 53 los enteros caben exactos en la mantisa de un float64


def _check(W: int, I: int, rounding: str) -> None:
    if not (1 <= W <= _W_MAX) or int(W) != W or int(I) != I:
        raise ValueError(f"W tiene que ser entero entre 1 y {_W_MAX} e I entero: W={W!r}, I={I!r}")
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding tiene que ser uno de {ROUNDINGS}: {rounding!r}")


def _to_int_real(x: np.ndarray, W: int, I: int, rounding: str, saturate: bool) -> np.ndarray:
    """Entero k con signo (valor = k·q) de un array real."""
    s = x * 2.0 ** (W - I)  # x/q, exacto: q es potencia de 2
    if rounding == "half_even":
        k = np.round(s)
    elif rounding == "half_up":
        k = np.floor(s + 0.5)
    else:
        k = np.floor(s)
    lo, hi = -(2 ** (W - 1)), 2 ** (W - 1) - 1
    if saturate:
        k = np.clip(k, lo, hi)
        return k.astype(np.int64)
    # complemento a dos: módulo 2^W, llevado a [lo, hi]
    k = np.mod(k, 2.0**W)
    k = np.where(k > hi, k - 2.0**W, k)
    return k.astype(np.int64)


def quantize(x, W: int, I: int, rounding: str = "half_even", saturate: bool = True):
    """Cuantiza x al formato ``ac_fixed<W, I, true>`` y devuelve el valor (float).

    - x: escalar o array, real o complejo.
    - W, I: bits totales y enteros con signo (decisión 25).
    - rounding: ``"half_even"`` (por defecto), ``"half_up"`` o ``"trunc"``.
    - saturate: True satura (``AC_SAT``); False desborda en complemento a dos.
    - return: mismo tipo (real o complejo) y forma que x; cada componente es un
      múltiplo de q = 2^-(W - I) dentro del rango.

    Lanza ``ValueError`` si W no está entre 1 y 53 o si rounding no es válido.
    """
    _check(W, I, rounding)
    x = np.asarray(x)
    q = 2.0 ** -(W - I)
    if np.iscomplexobj(x):
        re = _to_int_real(x.real, W, I, rounding, saturate) * q
        im = _to_int_real(x.imag, W, I, rounding, saturate) * q
        return re + 1j * im
    y = _to_int_real(x.astype(float), W, I, rounding, saturate) * q
    return y[()] if y.ndim == 0 else y


def to_int(x, W: int, I: int, rounding: str = "half_even", saturate: bool = True):
    """Enteros con signo (complemento a dos de W bits) del valor cuantizado: x ≈ k·q.

    Los usa la exportación de vectores (C4) para los ``.hex``: el código de W bits
    es k mod 2^W.

    - x, W, I, rounding, saturate: como en ``quantize``.
    - return: array int64 en [-2^(W-1), 2^(W-1) - 1] con la forma de x; si x es
      complejo, la tupla (k_I, k_Q).
    """
    _check(W, I, rounding)
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return (_to_int_real(x.real, W, I, rounding, saturate), _to_int_real(x.imag, W, I, rounding, saturate))
    return _to_int_real(x.astype(float), W, I, rounding, saturate)
