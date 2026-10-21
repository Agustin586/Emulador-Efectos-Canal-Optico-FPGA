"""Mapeo de bits a símbolos con codificación Gray, potencia media unitaria.

  PAM4  : 2 bits por símbolo.  00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3, todo / sqrt(5).
  QAM16 : 4 bits por símbolo. Los 2 primeros van a I y los 2 siguientes a Q,
          cada par como PAM4 Gray; todo / sqrt(10).

Con Gray, dos símbolos vecinos difieren en un solo bit: un error de decisión al
vecino más cercano cuesta un bit, no dos.

A diferencia de ``common.constellation`` (sin Gray, usada por ``gen_signal``),
acá el orden de los puntos importa, porque los bits vienen de una PRBS.
"""
from __future__ import annotations

import numpy as np

BITS_PER_SYMBOL = {"pam4": 2, "qam16": 4}

# índice = 2·b0 + b1  ->  nivel Gray: 00 -> -3, 01 -> -1, 10 -> +3, 11 -> +1
_PAM4_GRAY = np.array([-3.0, -1.0, 3.0, 1.0])


def map_gray(bits: np.ndarray, mod: str) -> np.ndarray:
    """Convierte bits en símbolos complejos de potencia media 1.

    bits : arreglo de ceros y unos; su largo tiene que ser múltiplo de los bits
           por símbolo (``BITS_PER_SYMBOL[mod]``).
    mod  : "pam4" o "qam16".

    Ejemplo:
        >>> (map_gray([0, 0, 0, 1, 1, 1, 1, 0], "pam4") * np.sqrt(5)).real.tolist()
        [-3.0, -1.0, 1.0, 3.0]
    """
    mod = mod.lower()
    if mod not in BITS_PER_SYMBOL:
        raise ValueError(f"modulación no soportada: {mod!r} (usar {sorted(BITS_PER_SYMBOL)})")
    bits = np.asarray(bits, dtype=np.int64)
    bps = BITS_PER_SYMBOL[mod]
    if bits.ndim != 1 or len(bits) % bps:
        raise ValueError(f"{mod}: el largo de bits debe ser múltiplo de {bps}: {bits.shape}")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("los bits tienen que ser 0 o 1")

    b = bits.reshape(-1, bps)
    i = _PAM4_GRAY[2 * b[:, 0] + b[:, 1]]
    if mod == "pam4":
        return (i / np.sqrt(5.0)).astype(complex)
    q = _PAM4_GRAY[2 * b[:, 2] + b[:, 3]]
    return (i + 1j * q) / np.sqrt(10.0)
