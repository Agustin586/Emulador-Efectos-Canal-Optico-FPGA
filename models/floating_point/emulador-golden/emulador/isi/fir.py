"""FIR causal de la ISI (Ref. A.1, PDF Ec. 8).

    y[n] = sum_{k=0}^{K-1} h[k] x[n-k],   taps a f_s.

Es la estructura del hardware: todos los perfiles de ISI solo cambian los h[k]
que se cargan. Las funciones operan sobre el último eje, así que sirven para
forma (n,) y (2, n), y devuelven el mismo largo que la entrada (causal, sin
recortar el retardo: eso lo decide el bloque).
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.signal import lfilter


def _taps(h) -> np.ndarray:
    h = np.asarray(h, dtype=complex)
    if h.ndim != 1 or h.size == 0:
        raise ValueError(f"h tiene que ser 1-D y no vacío: forma {h.shape}")
    return h


def isi_apply(x: np.ndarray, h) -> np.ndarray:
    """FIR causal y[n] = Σ h[k] x[n-k], con estado inicial en ceros.

    Igual a ``np.convolve(x, h)[:n]`` fila por fila. No usar
    ``np.convolve(..., "same")``: centra el filtro y rompe la causalidad.
    """
    return lfilter(_taps(h), 1.0, np.asarray(x, dtype=complex), axis=-1)


def isi_apply_stream(chunks: Iterable[np.ndarray], h) -> list[np.ndarray]:
    """Aplica el FIR a una secuencia de bloques consecutivos (PDF §3.6).

    Conserva entre bloques las últimas K-1 muestras de estado (``zi``/``zf`` de
    ``lfilter``), como el hardware, que procesa la señal en flujo. Arranca en
    ceros, así que concatenar las salidas da lo mismo que ``isi_apply`` sobre
    la señal completa. Los bloques pueden tener largos distintos (incluso
    menores que K-1, o vacíos); todos tienen que compartir las dimensiones
    previas al último eje.
    """
    h = _taps(h)
    out = []
    zi = None
    for x in chunks:
        x = np.asarray(x, dtype=complex)
        if zi is None:
            zi = np.zeros(x.shape[:-1] + (h.size - 1,), dtype=complex)
        if x.shape[-1] == 0:  # lfilter no acepta entradas vacías
            out.append(x.copy())
            continue
        y, zi = lfilter(h, 1.0, x, axis=-1, zi=zi)
        out.append(y)
    return out
