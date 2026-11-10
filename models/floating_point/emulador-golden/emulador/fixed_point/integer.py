"""Aritmética entera con la semántica de ``ac_fixed`` (decisiones 25 y 26).

Un valor de formato ``ac_fixed<W, I>`` se guarda como su código entero k (valor
k·2^-(W - I)). Con códigos en int64, los productos y las sumas son exactos (como
``ac_fixed``, que hace crecer el formato), y los únicos puntos con pérdida son
los cambios de formato:

  - ``round_shift``: descarta s bits fraccionarios con un redondeo, igual que
    asignar un ``ac_fixed`` con más bits fraccionarios a uno con menos;
  - ``saturate_int``: lleva al rango de W bits con signo (``AC_SAT``).

Son exactos para |k| < 2^62, más allá de los 53 bits de ``quantize``/``to_int``
(que pasan por float64); en el rango común coinciden con ``to_int`` (test).
"""
from __future__ import annotations

import numpy as np

ROUNDINGS = ("half_even", "half_up", "trunc")


def round_shift(k, s: int, rounding: str = "half_even") -> np.ndarray:
    """Divide los códigos k por 2^s y redondea al entero (s bits fraccionarios menos).

    - k: entero o array de enteros (int64).
    - s: bits que se descartan, s >= 0.
    - rounding: ``"half_even"`` (``AC_RND_CONV``), ``"half_up"`` (``AC_RND``) o
      ``"trunc"`` (``AC_TRN``, hacia -inf).
    - return: array int64, sin saturar.
    """
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding tiene que ser uno de {ROUNDINGS}: {rounding!r}")
    if s < 0 or int(s) != s:
        raise ValueError(f"s tiene que ser un entero >= 0: {s!r}")
    k = np.asarray(k, dtype=np.int64)
    if s == 0:
        return k.copy()
    q = k >> s  # piso (desplazamiento aritmético)
    if rounding == "trunc":
        return q
    r = k & ((1 << s) - 1)  # resto en [0, 2^s)
    half = 1 << (s - 1)
    if rounding == "half_up":
        return q + (r >= half)
    return q + ((r > half) | ((r == half) & ((q & 1) == 1)))


def saturate_int(k, W: int) -> tuple[np.ndarray, int]:
    """Satura los códigos al rango de W bits con signo [-2^(W-1), 2^(W-1) - 1].

    - return: (códigos saturados en int64, cantidad de valores que saturaron).
    """
    k = np.asarray(k, dtype=np.int64)
    lo, hi = -(1 << (W - 1)), (1 << (W - 1)) - 1
    n_sat = int(np.count_nonzero((k < lo) | (k > hi)))
    return np.clip(k, lo, hi), n_sat
