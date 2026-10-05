"""Perfiles de ISI: cómo se calculan los taps h[k] que se cargan en el FIR.

Ref. A.2 y A.3, PDF §3.5. Todos los perfiles devuelven taps con energía
unitaria, Σ|h[k]|² = 1 (PDF Ec. 9): la ISI redistribuye la potencia entre
símbolos vecinos pero no la cambia (la SNR la fija el bloque AWGN).

  - normalize_energy : h / sqrt(Σ|h|²).
  - echo             : canal de ecos discretos (multicamino).
  - gaussian         : pasa-bajos gaussiano diseñado en frecuencia (fase lineal).
  - measured         : respuesta medida, como función H(f) o tabla (f, H).
  - analog           : Bessel o Butterworth desde el prototipo analógico (causal).
  - cd_short         : CD de pocos km con los taps de Savory (vincula ISI y CD).
  - K_rule           : cantidad de taps según el ancho de banda y el orden.
  - energy_captured  : qué fracción de la energía queda en los K taps.

Diseño en frecuencia (Ref. A.3, PDF §3.5.3): se muestrea H(f) en la grilla de
una FFT de L puntos, se antitransforma y se trunca a K taps. No se muestrea la
h(t) continua: con B >= 0,5·Rs produce aliasing y el -3 dB se corre.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from scipy import signal

from ..cd.physics import savory_taps

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
    f = np.fft.fftfreq(L, d=1 / fs)
    H = np.exp(-np.log(2) / 2 * f**2 / B**2)
    return _centered_taps(H, K, real=True)


def measured(H, fs: float, K: int, L: int = 8192) -> np.ndarray:
    """Taps a partir de una respuesta en frecuencia medida, con el método del gaussiano.

    - H: la respuesta, de una de dos formas:
        - función ``H(f)`` que acepta un array de frecuencias en Hz;
        - tabla ``(f, H)``: frecuencias en Hz (en cualquier orden, puede cubrir
          solo parte de la banda) y valores complejos. Se interpola la magnitud
          en dB y la fase desenrollada con ``np.interp``; fuera del rango de la
          tabla, H = 0.
    - fs: frecuencia de muestreo, Sa/s.
    - K: cantidad de taps, 1 <= K <= L.
    - L: largo de la FFT de diseño.
    - return: array complejo de forma (K,), con energía unitaria.

    Centra antes de truncar, como ``gaussian``, pero conserva la parte
    imaginaria: una respuesta medida no tiene por qué ser simétrica. Con
    H = H_G da los mismos taps que ``gaussian``.
    """
    f = np.fft.fftfreq(L, d=1 / fs)
    if callable(H):
        Hf = np.asarray(H(f), dtype=complex)
    else:
        Hf = _interp_table(H, f)
    return _centered_taps(Hf, K)


def _interp_table(table, f: np.ndarray) -> np.ndarray:
    """Interpola una tabla (f_t, H_t) en f: magnitud en dB y fase desenrollada."""
    f_t, H_t = (np.asarray(a) for a in table)
    if f_t.ndim != 1 or f_t.shape != H_t.shape or f_t.size < 2:
        raise ValueError(f"measured: la tabla (f, H) necesita dos arrays 1-D iguales: {f_t.shape}, {H_t.shape}")
    order = np.argsort(f_t)
    f_t, H_t = f_t[order], H_t[order].astype(complex)
    mag_db = 20 * np.log10(np.maximum(np.abs(H_t), 1e-300))
    phase = np.unwrap(np.angle(H_t))
    Hf = 10 ** (np.interp(f, f_t, mag_db) / 20) * np.exp(1j * np.interp(f, f_t, phase))
    Hf[(f < f_t[0]) | (f > f_t[-1])] = 0.0
    return Hf


def _centered_taps(H: np.ndarray, K: int, real: bool = False) -> np.ndarray:
    """Antitransforma H (grilla de ``fftfreq``), centra y trunca K taps; normaliza.

    h[k] = h̃[L/2 - ⌊K/2⌋ + k], con h̃ = fftshift(ifft(H)) (PDF Ec. 10 y 11).
    """
    L = H.size
    if not (1 <= K <= L):
        raise ValueError(f"K tiene que cumplir 1 <= K <= L = {L}: {K!r}")
    h = np.fft.fftshift(np.fft.ifft(H))
    if real:
        h = h.real
    start = L // 2 - K // 2
    return normalize_energy(h[start : start + K])


def analog_prototype(kind: str, B: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Coeficientes (b, a) del prototipo analógico de orden m con -3 dB en B.

    - kind: "bessel" (fase casi lineal) o "butter" (máximamente plano, más ISI
      por fase).
    - B: ancho de banda a -3 dB, Hz.
    - m: orden.
    - return: (b, a) de H_P(s), con s en rad/s (``signal.freqs``).

    El Bessel usa ``norm="mag"``: con la normalización por defecto, B no es el
    punto de -3 dB.
    """
    wc = 2 * np.pi * B
    if kind == "bessel":
        return signal.bessel(m, wc, analog=True, norm="mag")
    if kind == "butter":
        return signal.butter(m, wc, analog=True)
    raise ValueError(f'analog: kind tiene que ser "bessel" o "butter": {kind!r}')


def analog(kind: str, B: float, fs: float, m: int, K: int, d: int | None = None, L: int = 4096) -> np.ndarray:
    """Bessel o Butterworth diseñado en frecuencia desde el prototipo analógico.

    Ref. A.3, PDF Ec. 12:

        h[k] = Re{ IFFT_L{ H_P(f_l) · exp(-j 2π f_l d Ts) } }[k],  k = 0 … K-1

    Se evalúa H_P en la grilla de ``fftfreq(L, 1/fs)`` y se le agrega un
    pre-retardo de d muestras, que conserva la oscilación previa al pulso que
    produce el recorte en Nyquist. **Nunca** con la transformación bilineal
    (decisión 6): a 2 sps deforma el corte y el Bessel pierde la fase lineal.

    - kind, B, m: como en ``analog_prototype``.
    - fs: frecuencia de muestreo, Sa/s.
    - K: cantidad de taps, 1 <= K <= L.
    - d: pre-retardo en muestras; por defecto 6 si K >= 32 y 3 si no.
    - L: largo de la FFT de diseño.
    - return: array real de forma (K,), causal, con energía unitaria.

    Límite del muestreo (decisión 19): a 2 sps y B > 0,5·Rs el prototipo no está
    limitado en banda en fs/2 y el -3 dB se corre hasta ~0,7 dB (m = 2). A 4 sps,
    con K y d al doble (mismo tiempo), desaparece.
    """
    if not (1 <= K <= L):
        raise ValueError(f"analog: K tiene que cumplir 1 <= K <= L = {L}: {K!r}")
    if d is None:
        d = 6 if K >= 32 else 3
    b, a = analog_prototype(kind, B, m)
    f = np.fft.fftfreq(L, d=1 / fs)
    _, H = signal.freqs(b, a, worN=2 * np.pi * f)
    H = H * np.exp(-2j * np.pi * f * d / fs)
    return normalize_energy(np.fft.ifft(H).real[:K])


def cd_short(fs: float, D: float, lam: float, z: float) -> np.ndarray:
    """CD corta en el FIR de la ISI: taps de Savory normalizados (Ref. A.3, PDF Ec. 25).

    - fs: frecuencia de muestreo, Sa/s.
    - D: coeficiente de dispersión, s/m².
    - lam: longitud de onda, m.
    - z: distancia, m.
    - return: array complejo de forma (R,), R = ``cd_R`` impar <= 31, con energía
      unitaria y el tap central en R // 2.

    Lanza ``ValueError`` si R > 31 (no entra en K <= 32): a 64 GSa/s con SSMF,
    más de ~55 km.

    Decisión 20: es una aproximación **ilustrativa**. El truncamiento abrupto
    del chirp deja 2 a 5 dB de ondulación en la banda (NRMSE contra
    ``cd_reference`` de 0,38 a 10 km y 0,14 a 55 km, QAM16 a 2 sps; con R = 3 es
    peor que no hacer nada). Con R = 1 (z <= 3 km a 64 GSa/s) devuelve [1]: el
    único tap de Savory sería una rotación de 45°. La CD de verdad va por
    overlap-save (bloque ``ChromaticDispersion``).
    """
    h = savory_taps(fs, D, lam, z)
    if h.size > K_MAX - 1:
        raise ValueError(f"cd_short: R = {h.size} taps no entra en K <= {K_MAX} (usar el bloque de CD)")
    if h.size == 1:
        return np.array([1.0 + 0j])
    return normalize_energy(h)


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
