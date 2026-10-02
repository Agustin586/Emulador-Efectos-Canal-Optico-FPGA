"""Bloque 01 - ISI genérica (Nivel 1)

En simple: la energía de cada símbolo se "chorrea" sobre los vecinos, como si el
canal tuviera memoria. Se modela con un filtro FIR de respuesta configurable.

Modelo:  y[n] = sum_{k=0}^{K-1} h[k] * x[n-k]   (taps a f_s, K <= 32)

Parámetros:
  taps    : coeficientes h[k], complejos. Por defecto un solo tap igual a 1
            (sin distorsión). Se usan con ``profile="taps"``.
  profile : origen de los taps. Por ahora solo "taps" (taps arbitrarios); los
            perfiles de la Parte A de GUIA_ISI_CD.md se agregan en A2 a A4.

Retardo (GUIA_ISI_CD.md §1, decisión 8):
  - ``delay_samples`` es el cursor k0 = argmax |h[k]|.
  - ``process_hw(x)`` es el FIR causal, como el hardware: la salida sale
    retrasada k0 muestras. Es la salida que se exporta para el RTL.
  - ``process(x)`` (y ``__call__``) devuelve la salida alineada: el cursor cae en
    la misma posición que en la entrada. Se rellena x con k0 ceros al final, se
    filtra y se descartan las primeras k0 muestras, así el largo se mantiene.
    El retardo fraccionario (si el cursor no está bien definido) no se corrige:
    le toca al receptor.
  - ``coefficients()`` devuelve {"h": h}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common import Block
from ..isi import isi_apply

K_MAX = 32  # taps del FIR en hardware (decisión 6)
PROFILES = ("taps",)


@dataclass
class ISI(Block):
    taps: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j]))
    profile: str = "taps"
    name = "isi"

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"isi: perfil desconocido {self.profile!r} (opciones: {PROFILES})")
        h = np.asarray(self.taps, dtype=complex)
        if h.ndim != 1 or not (1 <= h.size <= K_MAX):
            raise ValueError(f"isi: taps tiene que ser 1-D con 1 <= K <= {K_MAX}: forma {h.shape}")
        self.h = h
        self.delay_samples = int(np.argmax(np.abs(h)))

    def process_hw(self, x: np.ndarray) -> np.ndarray:
        """FIR causal, retrasado ``delay_samples`` muestras (como el hardware)."""
        return isi_apply(x, self.h)

    def process(self, x: np.ndarray) -> np.ndarray:
        """FIR con la salida alineada (descuenta ``delay_samples``)."""
        k0 = self.delay_samples
        if k0 == 0:
            return isi_apply(x, self.h)
        pad = np.zeros(np.shape(x)[:-1] + (k0,), dtype=complex)
        return isi_apply(np.concatenate([x, pad], axis=-1), self.h)[..., k0:]

    def coefficients(self) -> dict[str, np.ndarray]:
        return {"h": self.h}
