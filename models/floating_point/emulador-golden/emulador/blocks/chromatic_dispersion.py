"""Bloque 03 - Dispersión cromática, CD (Nivel 3)

En simple: cada color (longitud de onda) viaja a distinta velocidad por la fibra,
así que el pulso se ensancha en el tiempo. Es un filtro que solo cambia la fase:
no altera la potencia de cada frecuencia, pero desfasa más las frecuencias altas.

Modelo (banda base, ω = 2πf, GUIA_ISI_CD.md §1, decisiones 2 a 4):
    H(ω) = exp[-j (β₂ ω²/2 + β₃ ω³/6) z]
    β₂ = -D λ² / (2π c),   β₃ = (λ² / (2π c))² (S + 2D/λ)   (β₃ = 0 si S = 0)
Con β₃ = 0: H(f) = exp(+j π D λ² z f² / c). Convención de signo **positiva**,
sin el factor ½ de la guía original. El compensador (CDC) usa H*.

Implementación (decisión 5): muestreo en frecuencia + overlap-save con FFT de
L = 2N puntos (``emulador/cd/overlap_save.py``), el mismo algoritmo que la FPGA.
N = ``cd_N`` (max(128, menor potencia de 2 >= R), hasta N_max) o el que se pase,
que tiene que cumplir N >= R.

Parámetros (unidades de hoja de datos; el cálculo interno es en SI):
  fs            : frecuencia de muestreo, Hz (por defecto 64 GSa/s: 32 GBd, 2 sps).
  D_ps_nm_km    : parámetro de dispersión (SSMF: 17 ps/(nm·km); negativo emula
                  fibra compensadora).
  wavelength_nm : longitud de onda de la portadora (1550 nm).
  length_km     : largo del enlace. Por defecto 0 (sin efecto).
  S_ps_nm2_km   : pendiente de dispersión (activa β₃; SSMF: 0,057). Por defecto 0.
  N             : muestras nuevas por bloque; None -> ``cd_N``.

Retardo (decisión 8):
  - ``delay_samples`` = N/2.
  - ``process_hw(x)``: salida cruda del overlap-save, retrasada N/2, recortada
    al largo de x. Es la que se exporta para el RTL.
  - ``process(x)`` (y ``__call__``): salida alineada, del mismo largo que x.
  - ``coefficients()`` = {"Hbar": H̄, "N": N}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cd import cd_hbar, cd_N, cd_overlap_save, cd_R
from ..common import Block
from ..config import D_si, S_si, Scenario


@dataclass
class ChromaticDispersion(Block):
    fs: float = 64e9
    D_ps_nm_km: float = 17.0
    wavelength_nm: float = 1550.0
    length_km: float = 0.0
    S_ps_nm2_km: float = 0.0
    N: int | None = None
    name = "chromatic_dispersion"

    def __post_init__(self):
        if self.fs <= 0 or self.length_km < 0:
            raise ValueError(f"chromatic_dispersion: fs > 0 y length_km >= 0: {self.fs!r}, {self.length_km!r}")
        D = D_si(self.D_ps_nm_km)
        lam = self.wavelength_nm / 1e9  # / 1e9 (exacto) y no * 1e-9: mismo float que 1550e-9
        z = self.length_km * 1e3
        if self.N is None:
            self.N = cd_N(self.fs, D, lam, z)
        R = cd_R(self.fs, D, lam, z)
        if self.N < 2 or self.N % 2 or self.N < R:
            raise ValueError(f"chromatic_dispersion: N tiene que ser par y >= R = {R}: {self.N!r}")
        self.Hbar = cd_hbar(self.N, self.fs, D, lam, z, S_si(self.S_ps_nm2_km))
        self.delay_samples = self.N // 2

    @classmethod
    def from_scenario(cls, esc: Scenario, **kw) -> ChromaticDispersion:
        """Toma fs y λ del escenario; el resto de los campos va en ``kw``."""
        return cls(fs=esc.fs, wavelength_nm=esc.wavelength * 1e9, **kw)

    def process_hw(self, x: np.ndarray) -> np.ndarray:
        """Overlap-save crudo, retrasado N/2 muestras (como el hardware)."""
        x = np.asarray(x)
        return cd_overlap_save(x, self.Hbar)[..., : x.shape[-1]]

    def process(self, x: np.ndarray) -> np.ndarray:
        """Overlap-save con la salida alineada (descuenta N/2)."""
        x = np.asarray(x)
        n, h = x.shape[-1], self.delay_samples
        pad = np.zeros(x.shape[:-1] + (h,), dtype=complex)
        return cd_overlap_save(np.concatenate([x, pad], axis=-1), self.Hbar)[..., h : h + n]

    def coefficients(self) -> dict:
        return {"Hbar": self.Hbar, "N": self.N}
