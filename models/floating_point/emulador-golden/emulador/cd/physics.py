"""Física de la dispersión cromática (Ref. B.1, PDF Ec. 16 y 18).

Todo en SI: D en s/m², S en s/m³, λ y z en m, β₂ en s²/m, β₃ en s³/m, fs en Sa/s.
Las unidades de hoja de datos se convierten en ``config`` (``D_si``, ``S_si``).

    H_CD(ω) = exp[-j (β₂ ω²/2 + β₃ ω³/6) z]

    β₂ = -D λ² / (2π c)
    β₃ = (λ² / (2π c))² (S + 2D/λ)

Con β₃ = 0 queda H_CD(f) = exp(+j π D λ² z f² / c): convención de signo
positiva (GUIA_ISI_CD.md §1, decisión 2), sin el factor ½ de la guía original
(decisión 3). El compensador (CDC) usa H*.
"""
from __future__ import annotations

import numpy as np

from ..config import C, HardwareParams


def betas(D: float, lam: float, S: float = 0.0) -> tuple[float, float]:
    """β₂ [s²/m] y β₃ [s³/m] a partir de D [s/m²], λ [m] y S [s/m³].

    Con S = 0 devuelve β₃ = 0 exacto (decisión 4): el modelo queda idéntico al
    de solo β₂, aunque el término 2D/λ no sea nulo.

    - D: coeficiente de dispersión cromática, s/m².
    - lam: longitud de onda, m.
    - S: pendiente de dispersión, s/m³.
    - return: (β₂, β₃) en s²/m y s³/m.

    Ejemplo (SMF, 1550 nm, 17 ps/(nm·km)): β₂ ≈ -2.17e-26 s²/m = -21,7 ps²/km.
    """
    k = lam**2 / (2 * np.pi * C)
    beta2 = -D * k
    beta3 = k**2 * (S + 2 * D / lam) if S != 0.0 else 0.0
    return beta2, beta3


def cd_transfer(L: int, fs: float, D: float, lam: float, z: float, S: float = 0.0) -> np.ndarray:
    """H_CD en la grilla de una FFT de L puntos, en el orden de ``np.fft``.

    - L: largo de la FFT, entero positivo.
    - fs: frecuencia de muestreo, Sa/s.
    - D: coeficiente de dispersión cromática, s/m².
    - lam: longitud de onda, m.
    - z: distancia de propagación, m.
    - S: pendiente de dispersión, s/m³.
    - return: array de forma (L,) con H_CD(ω) en la grilla de la FFT, orden de ``np.fft``. 

    ω = 2π · fftfreq(L, 1/fs). |H| = 1 (pasa-todo), H(z₁)·H(z₂) = H(z₁ + z₂) y
    H⁻¹ = H*. Con z = 0 da unos exactos.
    """
    w = 2 * np.pi * np.fft.fftfreq(L, d=1 / fs)
    b2, b3 = betas(D, lam, S)
    return np.exp(-1j * (b2 / 2 * w**2 + b3 / 6 * w**3) * z)


def cd_R(fs: float, D: float, lam: float, z: float) -> int:
    """Largo del chirp de la CD en muestras (Ref. B.2, PDF Ec. 26).

        R = 2 ⌊|D| λ² z / (2 c Ts²)⌋ + 1

    Es la cantidad de muestras que ocupa la respuesta al impulso dentro de la
    banda de Nyquist (Savory, 2008). Siempre impar; con z = 0 da 1.

    - fs: frecuencia de muestreo, Sa/s.
    - D: coeficiente de dispersión cromática, s/m² (se usa |D|).
    - lam: longitud de onda, m.
    - z: distancia de propagación, m.
    - return: R, entero impar >= 1.

    Ejemplo (64 GSa/s, SMF, 1550 nm, 100 km): R = 55.
    """
    Ts = 1 / fs
    return 2 * int(np.floor(abs(D) * lam**2 * z / (2 * C * Ts**2))) + 1


def cd_N(
    fs: float,
    D: float,
    lam: float,
    z: float,
    N_min: int = 128,
    N_max: int = HardwareParams().N_max,
) -> int:
    """Tamaño de bloque N del overlap-save (decisión 5).

        N = max(N_min, menor potencia de 2 >= R)

    La FFT del overlap-save tiene L = 2N puntos.

    - fs, D, lam, z: como en ``cd_R``.
    - N_min: N mínimo (potencia de 2).
    - N_max: N máximo que admite el hardware; por defecto el de
      ``HardwareParams`` (provisorio).
    - return: N, potencia de 2.

    Lanza ``ValueError`` si N supera N_max.
    """
    R = cd_R(fs, D, lam, z)
    N = max(N_min, 1 << (R - 1).bit_length())
    if N > N_max:
        raise ValueError(f"cd_N: R = {R} requiere N = {N}, mayor que N_max = {N_max}")
    return N
