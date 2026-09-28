"""Utilidades compartidas: contrato de bloque, señal de prueba y EVM.

CONTRATO PROVISORIO DE INTERFAZ (a confirmar con el grupo, ver README):
  - Una señal es un ``np.ndarray`` complejo. El último eje es el tiempo.
  - Forma ``(n,)`` = una polarización. Forma ``(2, n)`` = dos polarizaciones (X, Y).
  - Un bloque devuelve un arreglo de la MISMA forma que recibe.
  - Con ``enabled=False`` el bloque no hace nada (bypass): devuelve la entrada intacta.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


# --------------------------------------------------------------------------
# Contrato de bloque
# --------------------------------------------------------------------------
@dataclass
class Block(ABC):
    """Clase base de todos los deterioros.

    Cada bloque: (1) hereda de ``Block``, (2) declara sus parámetros como campos
    de dataclass CON valor por defecto nulo (sin efecto), (3) implementa ``process``.
    """

    enabled: bool = True
    name: ClassVar[str] = "block"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if not self.enabled:
            return x
        y = self.process(x)
        if y.shape != x.shape:
            raise ValueError(f"{self.name}: la salida debe tener la forma de la entrada")
        return y

    @abstractmethod
    def process(self, x: np.ndarray) -> np.ndarray:
        """Aplica el efecto en punto flotante (golden model)."""


# --------------------------------------------------------------------------
# Señal de prueba
# --------------------------------------------------------------------------
def constellation(mod: str) -> np.ndarray:
    """Puntos de la constelación, normalizados a potencia media 1."""
    mod = mod.lower()
    if mod == "pam4":
        pts = np.array([-3, -1, 1, 3], dtype=complex)
    elif mod == "qam16":
        lv = np.array([-3, -1, 1, 3])
        pts = (lv[:, None] + 1j * lv[None, :]).ravel().astype(complex)
    else:
        raise ValueError(f"Modulación no soportada: {mod!r} (usar 'pam4' o 'qam16')")
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def rrc_taps(beta: float, span: int, sps: int) -> np.ndarray:
    """Filtro raíz de coseno realzado (RRC) de energía unitaria.

    beta: roll-off (0..1). span: largo en símbolos. sps: muestras por símbolo.
    """
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps  # tiempo en símbolos
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4.0 * beta)):
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))


@dataclass
class TestSignal:
    """Señal de prueba lista para pasarle a un bloque."""

    __test__ = False  # evita que pytest lo tome por una clase de tests

    samples: np.ndarray  # señal conformada, sps muestras por símbolo, potencia media 1
    symbols: np.ndarray  # símbolos ideales transmitidos (referencia)
    sps: int
    beta: float
    span: int
    mod: str


def gen_signal(
    mod: str = "qam16",
    n_symbols: int = 4096,
    sps: int = 2,
    beta: float = 0.2,
    span: int = 32,
    seed: int = 0,
) -> TestSignal:
    """Genera símbolos aleatorios, los sobremuestrea y los conforma con un RRC.

    Con los valores por defecto, la cadena TX-RX sin canal deja una EVM base de
    ~0.2 % (el filtro truncado no es perfecto): es el piso contra el que se miden
    los efectos.
    """
    rng = np.random.default_rng(seed)
    pts = constellation(mod)
    sym = pts[rng.integers(0, len(pts), n_symbols)]
    up = np.zeros(n_symbols * sps, dtype=complex)
    up[::sps] = sym
    x = np.convolve(up, rrc_taps(beta, span, sps), mode="same")
    x /= np.sqrt(np.mean(np.abs(x) ** 2))
    return TestSignal(x, sym, sps, beta, span, mod)


# --------------------------------------------------------------------------
# Métrica: EVM
# --------------------------------------------------------------------------
def recover_symbols(x: np.ndarray, sig: TestSignal) -> np.ndarray:
    """Receptor ideal mínimo: filtro adaptado (RRC) + muestreo en el instante óptimo.

    Sirve solo para MIRAR el efecto de un bloque; no es el receptor de las tesis
    consumidoras.
    """
    y = np.convolve(x, rrc_taps(sig.beta, sig.span, sig.sps), mode="same")
    return y[:: sig.sps] / np.sqrt(sig.sps)  # sin canal, coincide con los símbolos ideales


def evm_percent(rx: np.ndarray, ref: np.ndarray, discard: int = 32) -> float:
    """EVM (%) entre símbolos recibidos y de referencia.

    Normaliza la potencia de ``rx`` a la de ``ref`` (una ganancia real no cuenta
    como error), pero NO corrige fase: una rotación sí aparece como error.
    ``discard`` descarta símbolos de los bordes (transitorios del filtrado).
    """
    rx = rx[discard:-discard]
    ref = ref[discard:-discard]
    rx = rx * np.sqrt(np.mean(np.abs(ref) ** 2) / np.mean(np.abs(rx) ** 2))
    return float(100 * np.sqrt(np.mean(np.abs(rx - ref) ** 2) / np.mean(np.abs(ref) ** 2)))
