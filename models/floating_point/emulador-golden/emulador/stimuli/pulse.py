"""Conformación de pulso RRC y señal de comunicación completa.

Cadena de ``comm_signal``:  PRBS -> map_gray -> sobremuestreo (× sps) -> RRC.

A diferencia de ``common.gen_signal`` (roll-off 0.2, sin Gray, ``rng.integers``),
acá el roll-off sale del ``Scenario`` (0.1 por defecto), el mapeo es Gray y los
bits vienen de una PRBS, así los vectores exportados son reproducibles en el RTL.
``gen_signal`` queda para ``test_contract`` y ``run_block`` (GUIA_ISI_CD.md §5).
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from ..common import rrc_taps
from ..config import Scenario
from .mapper import BITS_PER_SYMBOL, map_gray
from .prbs import prbs


def shape(symbols: np.ndarray, esc: Scenario, span: int = 64) -> np.ndarray:
    """Sobremuestrea los símbolos a ``esc.sps`` y los filtra con un RRC.

    Inserta sps - 1 ceros entre símbolos y convoluciona con
    ``rrc_taps(esc.rolloff, span, esc.sps)`` en modo "same", como ``gen_signal``.
    El RRC tiene largo impar (span·sps + 1) y es simétrico, así que el símbolo k
    queda centrado en la muestra k·sps: no agrega retardo.

    La salida se normaliza a potencia media 1 (la ISI supone potencia 1).

    span : largo del RRC en símbolos. Con roll-off 0.1 el piso de EVM
           back-to-back es 0.58 % con 32 y 0.12 % con 64 (GUIA §1, decisión 15).
    """
    symbols = np.asarray(symbols, dtype=complex)
    up = np.zeros(len(symbols) * esc.sps, dtype=complex)
    up[:: esc.sps] = symbols
    x = np.convolve(up, rrc_taps(esc.rolloff, span, esc.sps), mode="same")
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def _package_version() -> str:
    try:
        return version("emulador-golden")
    except PackageNotFoundError:
        return "unknown"


@dataclass
class CommSignal:
    """Señal de comunicación con todo lo necesario para reproducirla y medirla."""

    samples: np.ndarray  # señal conformada, esc.sps muestras por símbolo, potencia media 1
    symbols: np.ndarray  # símbolos transmitidos (referencia para la EVM)
    bits: np.ndarray     # bits de la PRBS que generaron los símbolos
    scenario: Scenario
    seed: int
    mod: str
    prbs_order: int
    span: int

    def meta(self) -> dict:
        """Metadatos para el meta.json de los vectores exportados."""
        return {
            "scenario": self.scenario.to_dict(),
            "seed": self.seed,
            "prbs_order": self.prbs_order,
            "mod": self.mod,
            "n_symbols": len(self.symbols),
            "n_samples": len(self.samples),
            "span": self.span,
            "version": _package_version(),
        }


def comm_signal(
    esc: Scenario,
    mod: str = "qam16",
    n_symbols: int = 4096,
    prbs_order: int = 15,
    seed: int = 1,
    span: int = 64,
) -> CommSignal:
    """Genera una señal PRBS -> Gray -> RRC reproducible.

    esc        : escenario (sps y roll-off).
    mod        : "pam4" o "qam16".
    n_symbols  : cantidad de símbolos.
    prbs_order : orden de la PRBS (15 o 31 para los vectores del RTL).
    seed       : estado inicial del LFSR (ver ``prbs``).
    span       : largo del RRC en símbolos.
    """
    mod = mod.lower()
    bits = prbs(prbs_order, n_symbols * BITS_PER_SYMBOL[mod], seed=seed)
    symbols = map_gray(bits, mod)
    return CommSignal(shape(symbols, esc, span), symbols, bits, esc, seed, mod, prbs_order, span)
