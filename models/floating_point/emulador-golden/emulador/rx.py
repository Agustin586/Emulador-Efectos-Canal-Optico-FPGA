"""Receptor mínimo para medir y graficar. NO es parte del emulador.

Generaliza ``common.recover_symbols``, que asume retardo cero y fase de portadora
nula. Acá se busca el instante de muestreo y el retardo entero, y la EVM se mide
con ganancia y fase óptimas (``metrics.evm_opt``). Sin esa búsqueda, la EVM de
Bessel/Butterworth (retardo con parte fraccionaria) o de la CD (rota la
constelación) refleja el receptor y no el canal: más de 10 dB de diferencia
(PDF §3.5.5).

Para analizar el instante óptimo con más resolución que media muestra, generar
la señal con ``Scenario(sps=16)``: los perfiles escalan con B/Rs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags

from .common import rrc_taps
from .config import Scenario
from .metrics import evm_opt
from .stimuli import CommSignal


def matched_filter(x: np.ndarray, esc: Scenario, span: int = 64) -> np.ndarray:
    """Filtro adaptado: convolución con el mismo RRC del transmisor, modo "same".

    El RRC es simétrico y de largo impar: no agrega retardo. span por defecto =
    64 símbolos (GUIA_ISI_CD.md §1, decisión 15).
    """
    return np.convolve(x, rrc_taps(esc.rolloff, span, esc.sps), mode="same")


@dataclass
class RxResult:
    """Resultado de ``recover``.

    symbols : símbolos recibidos, alineados con ``ref`` y corregidos con la
              ganancia compleja óptima (listos para graficar la constelación).
    ref     : símbolos transmitidos correspondientes.
    phase   : fase de muestreo elegida, en muestras (0 … sps-1).
    delay   : retardo entero, en símbolos.
    evm     : ``evm_opt`` (fracción, no %) sobre ``symbols``/``ref``.
    """

    symbols: np.ndarray
    ref: np.ndarray
    phase: int
    delay: int
    evm: float

    def delay_samples(self, sps: int) -> int:
        """Retardo total en muestras: delay·sps + phase."""
        return self.delay * sps + self.phase


def _align(s: np.ndarray, ref: np.ndarray, delay: int, discard: int):
    """Pares (s[k + delay], ref[k]) solapados, sin ``discard`` símbolos por borde."""
    k0 = max(0, -delay)
    k1 = min(len(ref), len(s) - delay)
    k = np.arange(k0 + discard, k1 - discard)
    if len(k) == 0:
        raise ValueError("no quedan símbolos para medir: señal corta o retardo excesivo")
    return s[k + delay], ref[k]


def _best_delay(s: np.ndarray, ref: np.ndarray) -> int:
    """Retardo entero (en símbolos) que maximiza |correlación cruzada|."""
    c = correlate(s, ref, mode="full", method="fft")
    lags = correlation_lags(len(s), len(ref), mode="full")
    return int(lags[np.argmax(np.abs(c))])


def recover(x: np.ndarray, sig: CommSignal, search: bool = True) -> RxResult:
    """Filtro adaptado, muestreo en el instante óptimo y medición de EVM.

    x      : señal recibida (n,), con sig.scenario.sps muestras por símbolo.
    sig    : la señal transmitida (da el escenario, el span y los símbolos).
    search : True -> para cada fase 0 … sps-1 se estima el retardo entero por
             correlación cruzada con ``sig.symbols`` y se elige la combinación
             de menor ``evm_opt``. False -> fase 0 y retardo 0.

    La EVM descarta ``sig.span`` símbolos en cada borde (transitorios del
    filtrado).
    """
    esc = sig.scenario
    y = matched_filter(np.asarray(x), esc, sig.span)
    candidates = range(esc.sps) if search else (0,)
    best = None
    for phase in candidates:
        s = y[phase :: esc.sps]
        delay = _best_delay(s, sig.symbols) if search else 0
        rx, ref = _align(s, sig.symbols, delay, sig.span)
        e = evm_opt(rx, ref)
        if best is None or e < best[0]:
            best = (e, phase, delay, rx, ref)
    e, phase, delay, rx, ref = best
    g = np.vdot(rx, ref) / np.vdot(rx, rx)  # ganancia compleja óptima, como en evm_opt
    return RxResult(g * rx, ref, phase, delay, float(e))
