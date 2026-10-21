"""CD por bloques con overlap-save: el algoritmo que ejecuta la FPGA (Ref. B.4).

PDF §4.7.2, Ec. 27 y 28, Fig. 8. Se separa en dos funciones, como en hardware:

  - cd_hbar         : el host calcula una vez el filtro H̄ de L = 2N puntos.
  - cd_overlap_save : la FPGA procesa la señal en bloques de N muestras nuevas.

H̄[k] = H_CD(2π f_k) · exp(-j 2π k (N/2) / L): la fase lineal centra el chirp
(no causal) en la ventana. Por eso la salida sale **retrasada N/2 muestras**.
Hace falta N >= R (``cd_N``) para que la convolución circular de cada bloque
coincida con la lineal (decisión 5).
"""
from __future__ import annotations

import numpy as np

from .physics import cd_transfer


def cd_hbar(N: int, fs: float, D: float, lam: float, z: float, S: float = 0.0) -> np.ndarray:
    """Filtro H̄ de L = 2N puntos para overlap-save, con retardo de N/2 (PDF Ec. 28).

    - N: muestras nuevas por bloque (par, >= R).
    - fs, D, lam, z, S: como en ``cd_transfer`` (SI).
    - return: array complejo de forma (2N,), en el orden de ``np.fft``, |H̄| = 1.
    """
    if N < 2 or N % 2:
        raise ValueError(f"cd_hbar: N tiene que ser par y >= 2: {N!r}")
    L = 2 * N
    k = np.fft.fftfreq(L) * L
    return cd_transfer(L, fs, D, lam, z, S) * np.exp(-2j * np.pi * k * (N / 2) / L)


def cd_overlap_save(x: np.ndarray, Hbar: np.ndarray) -> np.ndarray:
    """Overlap-save con FFT de L = 2N puntos. La salida sale retrasada N/2 muestras.

    Por bloque b: ventana [N muestras anteriores | N nuevas] (el primer bloque
    va precedido por N ceros) -> FFT -> × H̄ -> IFFT -> se descartan las primeras
    N muestras y se entregan las N últimas.

    - x: señal de forma (..., n); opera sobre el último eje.
    - Hbar: filtro de ``cd_hbar``, forma (2N,).
    - return: forma (..., nb·N), nb = ⌈n/N⌉ (la última ventana se completa con
      ceros). y[n] corresponde a x[n - N/2].
    """
    x = np.asarray(x, dtype=complex)
    L = Hbar.shape[-1]
    N = L // 2
    n = x.shape[-1]
    nb = -(-n // N)
    widths = [(0, 0)] * (x.ndim - 1) + [(N, nb * N - n)]
    xz = np.pad(x, widths)
    frames = xz[..., np.arange(nb)[:, None] * N + np.arange(L)]  # (..., nb, L)
    y = np.fft.ifft(np.fft.fft(frames, axis=-1) * Hbar, axis=-1)[..., N:]
    return y.reshape(x.shape[:-1] + (nb * N,))
