"""Secuencias binarias pseudoaleatorias (PRBS) según ITU-T O.150.

Los vectores que se exportan al RTL usan estos bits, no ``rng.integers``: así el
hardware puede regenerar el mismo estímulo con un LFSR (referencia §C.1).

CONVENCIÓN (tiene que coincidir con el LFSR del RTL; GUIA_ISI_CD.md §1, decisión 16):
  - LFSR de Fibonacci con polinomio x^m + x^k + 1. La secuencia de salida cumple
        b[n] = b[n-m] xor b[n-k]
  - Los primeros m bits de salida son la semilla, del bit más significativo al
    menos significativo. Es decir, la semilla es el contenido inicial del
    registro, y el registro saca primero su MSB.
  - Sin inversión por defecto. O.150 especifica la salida invertida para
    PRBS-15, PRBS-23 y PRBS-31; ``invert=True`` la reproduce. Lo que importa es
    que el RTL use la misma opción.
"""
from __future__ import annotations

import numpy as np

# orden m -> exponente k del término intermedio del polinomio x^m + x^k + 1
POLYNOMIALS = {
    7: 6,    # x^7  + x^6  + 1
    9: 5,    # x^9  + x^5  + 1
    11: 9,   # x^11 + x^9  + 1
    15: 14,  # x^15 + x^14 + 1
    23: 18,  # x^23 + x^18 + 1
    31: 28,  # x^31 + x^28 + 1
}


def prbs(order: int, n_bits: int, seed: int = 1, invert: bool = False) -> np.ndarray:
    """Genera ``n_bits`` de una PRBS de orden ``order`` (período 2^order - 1).

    order  : 7, 9, 11, 15, 23 o 31 (ver ``POLYNOMIALS``).
    n_bits : largo de la salida.
    seed   : estado inicial del registro, entero en [1, 2^order - 1] (0 es el
             estado muerto del LFSR).
    invert : invierte la salida, como pide O.150 para PRBS-15/23/31.

    Devuelve un arreglo ``uint8`` de ceros y unos.

    Ejemplo:
        >>> prbs(7, 10).tolist()
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    """
    if order not in POLYNOMIALS:
        raise ValueError(f"orden de PRBS no soportado: {order!r} (usar {sorted(POLYNOMIALS)})")
    if not (1 <= seed < 2**order):
        raise ValueError(f"la semilla debe estar en [1, 2^{order} - 1]: {seed!r}")
    if n_bits < 0:
        raise ValueError(f"n_bits debe ser >= 0: {n_bits!r}")

    k = POLYNOMIALS[order]
    b = np.empty(max(n_bits, order), dtype=np.uint8)
    b[:order] = [(seed >> (order - 1 - i)) & 1 for i in range(order)]
    # La recurrencia b[n] = b[n-m] ^ b[n-k] solo mira k o más muestras atrás
    # (k < m), así que se pueden calcular k bits por vez sin lazo bit a bit.
    n = order
    while n < len(b):
        stop = min(n + k, len(b))
        b[n:stop] = b[n - order : stop - order] ^ b[n - k : stop - k]
        n = stop
    b = b[:n_bits]
    return b ^ 1 if invert else b
