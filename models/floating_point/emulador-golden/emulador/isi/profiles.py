"""Perfiles de ISI: cómo se calculan los taps h[k] que se cargan en el FIR.

Ref. A.2 y A.3, PDF §3.5. Todos los perfiles devuelven taps con energía
unitaria, Σ|h[k]|² = 1 (PDF Ec. 9): la ISI redistribuye la potencia entre
símbolos vecinos pero no la cambia (la SNR la fija el bloque AWGN).

  - normalize_energy : h / sqrt(Σ|h|²).
  - echo             : canal de ecos discretos (multicamino).
  - gaussian         : pasa-bajos gaussiano diseñado en frecuencia (fase lineal).
  - K_rule           : cantidad de taps según el ancho de banda y el orden.
  - energy_captured  : qué fracción de la energía queda en los K taps.

Diseño en frecuencia (Ref. A.3, PDF §3.5.3): se muestrea H(f) en la grilla de
una FFT de L puntos, se antitransforma y se trunca a K taps. No se muestrea la
h(t) continua: con B >= 0,5·Rs produce aliasing y el -3 dB se corre.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

K_MAX = 32  # taps del FIR en hardware (decisión 6)


def normalize_energy(h) -> np.ndarray:
    """Escala h para que Σ|h[k]|² = 1 (PDF Ec. 9).

    - h: taps, array 1-D (real o complejo).
    - return: h / sqrt(Σ|h|²), con el mismo dtype que h.

    Lanza ``ValueError`` si h tiene energía nula.
    """
    h = np.asarray(h)
    energy = np.sum(np.abs(h) ** 2)
    if energy == 0:
        raise ValueError("normalize_energy: h tiene energía nula")
    return h / np.sqrt(energy)


def echo(echoes: Iterable[tuple[int, complex]], K: int | None = None) -> np.ndarray:
    """Canal de ecos: un camino principal y copias atenuadas y retrasadas.

        h[0] = 1,  h[d_i] = a_i,  normalizado en energía

    - echoes: pares (d_i, a_i): retardo en muestras (entero, 1 <= d_i <= 31,
      sin repetir) y amplitud compleja (|a_i| < 1, así el cursor sigue en k = 0).
    - K: cantidad de taps; por defecto max(d_i) + 1. Tiene que cumplir
      max(d_i) < K <= 32.
    - return: array complejo de forma (K,).

    Sin ecos devuelve [1] (sin distorsión).

    Ejemplo: ``echo([(2, 0.5)])`` es [1, 0, 0.5] normalizado.
    """
    echoes = list(echoes)
    ds = [d for d, _ in echoes]
    for d, a in echoes:
        if int(d) != d or not (1 <= d <= K_MAX - 1):
            raise ValueError(f"echo: retardo fuera de rango (entero de 1 a {K_MAX - 1}): {d!r}")
        if abs(a) >= 1:
            raise ValueError(f"echo: |a| tiene que ser < 1: {a!r}")
    if len(set(ds)) != len(ds):
        raise ValueError(f"echo: retardos repetidos: {ds}")
    d_max = max(ds, default=0)
    if K is None:
        K = int(d_max) + 1
    if not (d_max < K <= K_MAX):
        raise ValueError(f"echo: K tiene que cumplir max(d) = {d_max} < K <= {K_MAX}: {K!r}")
    h = np.zeros(K, dtype=complex)
    h[0] = 1.0
    for d, a in echoes:
        h[int(d)] = a
    return normalize_energy(h)


def gaussian(B: float, fs: float, K: int, L: int = 8192) -> np.ndarray:
    """Pasa-bajos gaussiano con -3 dB en B, diseñado en frecuencia (PDF Ec. 10 y 11).

        H_G(f) = exp(-(ln 2 / 2) · f² / B²)

    Se antitransforma, se centra con ``fftshift`` y se trunca centrado:
    h[k] = h̃[L/2 - ⌊K/2⌋ + k]. Es simétrico (fase lineal) con el cursor en
    k0 = ⌊K/2⌋. Al ser simétrico y real, es el perfil de validación del método.

    - B: ancho de banda a -3 dB, Hz.
    - fs: frecuencia de muestreo, Sa/s.
    - K: cantidad de taps, 1 <= K <= L.
    - L: largo de la FFT de diseño.
    - return: array real de forma (K,), con energía unitaria.

    Solo depende de B/fs: el mismo B/Rs con el mismo sps da los mismos taps a
    cualquier tasa de símbolos.
    """
    if B <= 0:
        raise ValueError(f"gaussian: B tiene que ser positivo: {B!r}")
    if not (1 <= K <= L):
        raise ValueError(f"gaussian: K tiene que cumplir 1 <= K <= L = {L}: {K!r}")
    f = np.fft.fftfreq(L, d=1 / fs)
    H = np.exp(-np.log(2) / 2 * f**2 / B**2)
    h = np.fft.fftshift(np.fft.ifft(H)).real
    start = L // 2 - K // 2
    return normalize_energy(h[start : start + K])


def energy_captured(design: Callable[[int], np.ndarray], K: int, L: int = 8192) -> float:
    """Fracción de la energía de la respuesta completa que queda en K taps.

    - design: función K -> taps (por ejemplo ``partial(gaussian, B, fs)``),
      que use una FFT de diseño de L puntos.
    - K: cantidad de taps a evaluar.
    - L: largo de la respuesta completa (se diseña con K = L, sin truncar).
    - return: Σ|h_L[ventana]|² / Σ|h_L|², entre 0 y 1.

    La ventana es la posición de los K taps dentro de la respuesta completa;
    se ubica por correlación, así sirve para perfiles truncados centrados
    (gaussiano) o desde el inicio (Bessel, Butterworth).
    """
    full = normalize_energy(design(L))
    short = design(K)
    # posición de la ventana: máximo de |Σ full[o + k]·conj(short[k])|
    corr = np.abs(np.correlate(full, short, mode="valid"))
    o = int(np.argmax(corr))
    return float(np.sum(np.abs(full[o : o + K]) ** 2))


def K_rule(B: float, Rs: float, m: int) -> int:
    """Cantidad de taps de los perfiles pasa-bajos (decisión 6).

    K = 32 si B <= 0,4·Rs y m >= 4 (respuesta larga: banda angosta y orden
    alto); K = 16 en los demás casos. El perfil principal (Bessel m = 4,
    B = 0,6·Rs) usa K = 16.

    - B: ancho de banda a -3 dB, Hz.
    - Rs: tasa de símbolos, Bd.
    - m: orden del filtro.
    """
    return 32 if (B <= 0.4 * Rs and m >= 4) else 16
