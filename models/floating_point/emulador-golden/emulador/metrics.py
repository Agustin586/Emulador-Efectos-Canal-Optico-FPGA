"""Métricas comunes del golden model de ISI y CD.

Comparan los tres niveles de cada bloque (referencia ideal, modelo por bloques en
flotante y modelo en punto fijo) y caracterizan la ISI. Todos los tests y
notebooks las usan, por eso cada una se valida contra casos analíticos
(tests/test_metrics.py).

Convención de frecuencias (GUIA_ISI_CD.md §1 y referencia §3): los espectros
entran en el orden de numpy.fft (fftfreq); fftshift se aplica solo adentro, para
derivar o graficar.
"""
from __future__ import annotations

import numpy as np

from .common import rrc_taps


# --------------------------------------------------------------------------
# Comparación entre modelos
# --------------------------------------------------------------------------
def nrmse(y: np.ndarray, ref: np.ndarray) -> float:
    """Error cuadrático medio normalizado (referencia §1, PDF Ec. 32).

        NRMSE = sqrt( sum |y - ref|^2 / sum |ref|^2 )

    ``y`` y ``ref`` tienen que tener la misma forma y estar YA alineados
    (descontado el retardo de cada modelo): alinear es responsabilidad del que
    llama. Acepta (n,) o (2, n); con dos polarizaciones suma sobre ambas.
    """
    y = np.asarray(y)
    ref = np.asarray(ref)
    if y.shape != ref.shape:
        raise ValueError(f"nrmse: formas distintas {y.shape} y {ref.shape} (¿faltó alinear?)")
    energy_ref = np.sum(np.abs(ref) ** 2)
    if energy_ref == 0:
        raise ValueError("nrmse: la referencia tiene energía nula")
    return float(np.sqrt(np.sum(np.abs(y - ref) ** 2) / energy_ref))


# --------------------------------------------------------------------------
# Calidad de la constelación
# --------------------------------------------------------------------------
def evm_opt(rx: np.ndarray, ref: np.ndarray, discard: int = 0) -> float:
    """EVM (como fracción, no en %) con ganancia Y fase óptimas.

        g   = sum conj(rx)·ref / sum |rx|^2      (ganancia compleja de mínimo error)
        EVM = ||g·rx - ref|| / ||ref||

    Diferencia con ``common.evm_percent``: aquella normaliza solo la potencia y NO
    corrige la fase, así que una rotación cuenta como error. Acá la rotación y la
    ganancia constantes no cuentan: lo que queda es la distorsión propiamente
    dicha. Hace falta para la CD, cuyo cursor principal es complejo y rota la
    constelación. Las dos se usan, según qué se quiera medir.

    Con un error e ortogonal a ref (ruido, ISI no correlacionada), la ganancia
    óptima absorbe parte de la potencia y resulta EVM^2 = N/(S+N), no N/S.

    ``discard`` descarta símbolos de cada borde (transitorios del filtrado).
    """
    rx = np.asarray(rx)
    ref = np.asarray(ref)
    if rx.shape != ref.shape:
        raise ValueError(f"evm_opt: formas distintas {rx.shape} y {ref.shape}")
    if discard:
        rx = rx[..., discard:-discard]
        ref = ref[..., discard:-discard]
    energy_rx = np.sum(np.abs(rx) ** 2)
    if energy_rx == 0 or np.sum(np.abs(ref) ** 2) == 0:
        raise ValueError("evm_opt: rx o ref tienen energía nula")
    g = np.vdot(rx, ref) / energy_rx
    return float(np.linalg.norm(g * rx - ref) / np.linalg.norm(ref))


def snr_eq_db(rx: np.ndarray, ref: np.ndarray, discard: int = 0) -> float:
    """SNR equivalente en dB: -20·log10(evm_opt).

    Expresa una distorsión determinista (ISI, residuo de beta3) en la misma
    escala que el ruido del enlace, para compararlos (PDF Fig. 10). Con error
    ortogonal a la referencia vale 10·log10(1 + S/N), que a SNR alta es S/N.
    Devuelve inf si no hay error.
    """
    evm = evm_opt(rx, ref, discard)
    return float(np.inf) if evm == 0 else float(-20 * np.log10(evm))


# --------------------------------------------------------------------------
# Forma de pulso
# --------------------------------------------------------------------------
def rms_width(a: np.ndarray, t: np.ndarray) -> float:
    """Ancho rms de un pulso, ponderado por su intensidad |a|^2.

        sigma = sqrt( sum t^2 |a|^2 / sum |a|^2 - (sum t |a|^2 / sum |a|^2)^2 )

    No depende de la posición, la amplitud ni la fase del pulso. Para
    a = exp(-t^2/2T0^2) vale T0/sqrt(2); bajo CD crece en el mismo factor
    sqrt(1 + (z/L_D)^2) que T0 (PDF Ec. 23), que es lo que verifica el paso B2.
    """
    a = np.asarray(a)
    t = np.asarray(t, dtype=float)
    if a.shape != t.shape:
        raise ValueError(f"rms_width: formas distintas {a.shape} y {t.shape}")
    w = np.abs(a) ** 2
    energy = np.sum(w)
    if energy == 0:
        raise ValueError("rms_width: pulso de energía nula")
    mean = np.sum(t * w) / energy
    return float(np.sqrt(np.sum((t - mean) ** 2 * w) / energy))


# --------------------------------------------------------------------------
# Respuesta en frecuencia
# --------------------------------------------------------------------------
def group_delay(H: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Retardo de grupo tau(f) = -dphi/domega de una respuesta en frecuencia.

    H  : respuesta en la grilla de una FFT de L puntos, en orden numpy.fft
         (f = fftfreq(L, 1/fs)).
    fs : frecuencia de muestreo [Hz].

    Devuelve (f, tau) con f ordenada de -fs/2 a fs/2 y tau en segundos.

    La fase se desenrolla (unwrap) antes de derivar: sin eso, los saltos de 2*pi
    aparecen como picos de retardo. La derivada usa diferencias de segundo orden
    también en los bordes (edge_order=2), así es exacta para fases lineales
    (retardo puro, FIR simétrico) y cuadráticas (CD sin beta3).

    El unwrap supone que la fase cambia menos de pi entre muestras de frecuencia:
    si no se cumple (retardos de más de L/2 muestras), usar una FFT más larga.
    """
    H = np.asarray(H) # Asegura que sea un array de numpy
    L = H.shape[-1]   # Número de puntos de la FFT
    f = np.fft.fftshift(np.fft.fftfreq(L, d=1 / fs)) # Frecuencias ordenadas de -fs/2 a fs/2
    phase = np.unwrap(np.angle(np.fft.fftshift(H))) # Desenrolla la fase
    # Calcula el retardo de grupo como la derivada negativa de la fase respecto a la frecuencia angular 
    tau = -np.gradient(phase, 2 * np.pi * f, edge_order=2) 
    return f, tau


# --------------------------------------------------------------------------
# Severidad de la ISI (PDF §3.7)
# --------------------------------------------------------------------------
def _split_cursor(p: np.ndarray) -> tuple[float, np.ndarray]:
    """Separa el cursor principal (el de mayor módulo) del resto de la respuesta."""
    p = np.asarray(p)
    if not np.any(p):
        raise ValueError("respuesta nula: no hay cursor principal")
    n0 = int(np.argmax(np.abs(p)))
    return float(np.abs(p[n0])), np.delete(p, n0)


def peak_distortion(p: np.ndarray) -> float:
    """Distorsión de pico D_p = sum_{n != n0} |p[n]| / |p[n0]| (PDF Ec. 13).

    ``p`` es la respuesta total muestreada a la tasa de símbolos (ver
    ``symbol_rate_response``); el cursor principal n0 es el de mayor módulo.
    D_p < 1 indica ojo abierto para modulaciones de dos niveles.
    """
    cursor, rest = _split_cursor(p)
    return float(np.sum(np.abs(rest)) / cursor)


def isi_power_ratio(p: np.ndarray) -> float:
    """Potencia de ISI relativa al cursor: sum_{n != n0} |p[n]|^2 / |p[n0]|^2.

    Su inversa en dB, -10·log10(isi_power_ratio), es la "SNR equivalente" de la
    Tabla 4 del PDF, que se compara directamente con la SNR del enlace.
    """
    cursor, rest = _split_cursor(p)
    return float(np.sum(np.abs(rest) ** 2) / cursor**2)


def symbol_rate_response(
    h: np.ndarray, sps: int, rolloff: float, span: int = 32, phase: int = 0
) -> np.ndarray:
    """Respuesta total RRC * h * RRC muestreada a la tasa de símbolos.

    Es la p[n] de ``peak_distortion`` e ``isi_power_ratio``: transmisor (RRC),
    canal (h, a fs) y filtro adaptado (RRC), tomando una muestra cada ``sps``
    a partir de ``phase`` (0 <= phase < sps). Sin canal (h = [1]) y con
    phase = 0 se muestrea en el pico del coseno realzado, que vale 1 porque
    ``rrc_taps`` tiene energía unitaria.

    El RRC truncado deja un piso de ISI propio: con roll-off 0.1 es de -44.6 dB
    con span = 32 y de -35.5 dB con span = 16. Por eso span = 32 por defecto.

    Para el instante óptimo con un retardo fraccionario, probar todas las fases
    o usar un escenario con más sps.
    """
    if not 0 <= phase < sps:
        raise ValueError(f"phase debe estar entre 0 y sps - 1 = {sps - 1}: {phase!r}")
    g = rrc_taps(rolloff, span, sps)
    total = np.convolve(np.convolve(g, np.asarray(h)), g)
    return total[phase::sps]
