"""Referencia ideal de la CD: una sola FFT del largo de toda la señal (Ref. B.3).

Es la "verdad física" contra la que se mide el overlap-save (paso B3): separa
el error del método por bloques del de cuantización que aparece después
(PDF §4.7.3). No es implementable en hardware (la FFT es del largo de la señal).

Se rellena x con pad = R + 64 ceros a cada lado, para que la convolución
circular no lleve las colas del chirp de un extremo al otro, y se usa una FFT
de L = menor potencia de 2 >= len(x) + 2·pad.

Piso numérico (GUIA_ISI_CD.md §1, decisión 18): la respuesta al impulso
discreta de H_CD no termina en R muestras, sus colas decaen como ~1/n. Lo que
cae fuera del buffer de L muestras da la vuelta, así que aplicar (D, S) y luego
(-D, -S) devuelve x con NRMSE ~1e-6, no 1e-15. Es cuatro órdenes menor que el
error que se le admite al overlap-save (1e-2).
"""
from __future__ import annotations

import numpy as np

from .physics import cd_R, cd_transfer


def cd_reference_pad(fs: float, D: float, lam: float, z: float) -> int:
    """Ceros que ``cd_reference`` agrega a cada lado de x: R + 64.

    Con ``full=True``, x[n] corresponde a y[pad + n].
    """
    return cd_R(fs, D, lam, z) + 64


def cd_reference(
    x: np.ndarray,
    fs: float,
    D: float,
    lam: float,
    z: float,
    S: float = 0.0,
    full: bool = False,
) -> np.ndarray:
    """Aplica la CD con una FFT de toda la señal (rellenada con ceros).

    - x: señal de forma (..., n); opera sobre el último eje.
    - fs, D, lam, z, S: como en ``cd_transfer`` (SI).
    - full:
        - False: devuelve (..., n), alineada: y[n] corresponde a x[n] (el chirp
          está centrado, no hay retardo). Se pierden las colas que caen fuera.
        - True: devuelve las L muestras del buffer, con las colas (decisión
          18). Conserva la energía; x[n] corresponde a y[pad + n], con
          pad = ``cd_reference_pad(fs, D, lam, z)``.
    - return: array complejo.
    """
    x = np.asarray(x, dtype=complex)
    n = x.shape[-1]
    pad = cd_reference_pad(fs, D, lam, z)
    widths = [(0, 0)] * (x.ndim - 1) + [(pad, pad)]
    xp = np.pad(x, widths)
    L = 1 << (xp.shape[-1] - 1).bit_length()
    y = np.fft.ifft(np.fft.fft(xp, L, axis=-1) * cd_transfer(L, fs, D, lam, z, S), axis=-1)
    if full:
        return y
    return y[..., pad : pad + n]
