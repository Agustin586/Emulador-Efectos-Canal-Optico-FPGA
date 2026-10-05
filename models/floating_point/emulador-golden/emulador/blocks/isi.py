"""Bloque 01 - ISI genérica (Nivel 1)

En simple: la energía de cada símbolo se "chorrea" sobre los vecinos, como si el
canal tuviera memoria. Se modela con un filtro FIR de respuesta configurable.

Modelo:  y[n] = sum_{k=0}^{K-1} h[k] * x[n-k]   (taps a f_s, K <= 32)

Parámetros:
  profile    : de dónde salen los taps (``emulador/isi/profiles.py``):
                 "taps"     : ``taps`` tal cual (por defecto, [1]: sin distorsión);
                 "echo"     : camino principal + ``echoes``;
                 "gaussian" : pasa-bajos gaussiano con -3 dB en B = B_rel·Rs;
                 "measured" : respuesta medida ``H_measured``;
                 "bessel"   : Bessel de orden ``order`` con -3 dB en B (fase casi
                              lineal; perfil principal con m = 4, B = 0,6·Rs, K = 16);
                 "butter"   : Butterworth de orden ``order`` (perfil de estrés);
                 "cd_short" : CD de ``length_km`` con taps de Savory (R <= 31;
                              aproximación ilustrativa, decisión 20).
               Todos menos "taps" salen con energía unitaria.
  taps       : coeficientes h[k], complejos (profile="taps").
  Rs         : tasa de símbolos [Bd]; con sps da fs = Rs·sps.
  sps        : muestras por símbolo.
  B_rel      : ancho de banda a -3 dB relativo a Rs (B = B_rel·Rs).
  order      : orden m del filtro; entra en la regla de K.
  K          : cantidad de taps (<= 32). None -> K_rule(B, Rs, order) para
               gaussian, measured, bessel y butter; max(d_i) + 1 para echo.
  echoes     : pares (d_i, a_i) del perfil "echo": retardo en muestras y
               amplitud compleja, |a_i| < 1.
  H_measured : respuesta medida del perfil "measured": función H(f) con f en Hz,
               o tabla (f, H).
  D_ps_nm_km, wavelength_nm, length_km : fibra del perfil "cd_short". Por
               defecto 17 ps/(nm·km), 1550 nm y 0 km (sin efecto).

Retardo (GUIA_ISI_CD.md §1, decisión 8):
  - ``delay_samples`` es el cursor k0 = argmax |h[k]| (con "cd_short", cuyos
    taps tienen todos la misma magnitud, el tap central K // 2).
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
from ..config import D_si
from ..isi import K_rule, analog, cd_short, echo, gaussian, isi_apply, measured
from ..isi.profiles import K_MAX

PROFILES = ("taps", "echo", "gaussian", "measured", "bessel", "butter", "cd_short")


@dataclass
class ISI(Block):
    taps: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j]))
    profile: str = "taps"
    Rs: float = 32e9
    sps: int = 2
    B_rel: float = 0.6
    order: int = 4
    K: int | None = None
    echoes: tuple = ()
    H_measured: object = None
    D_ps_nm_km: float = 17.0
    wavelength_nm: float = 1550.0
    length_km: float = 0.0
    name = "isi"

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"isi: perfil desconocido {self.profile!r} (opciones: {PROFILES})")
        if self.Rs <= 0 or self.sps < 1 or self.B_rel <= 0:
            raise ValueError(f"isi: Rs, sps y B_rel tienen que ser positivos: {self.Rs!r}, {self.sps!r}, {self.B_rel!r}")
        h = np.asarray(self._design(), dtype=complex)
        if h.ndim != 1 or not (1 <= h.size <= K_MAX):
            raise ValueError(f"isi: taps tiene que ser 1-D con 1 <= K <= {K_MAX}: forma {h.shape}")
        self.h = h
        if self.profile == "cd_short":
            self.delay_samples = h.size // 2
        else:
            self.delay_samples = int(np.argmax(np.abs(h)))

    def _design(self) -> np.ndarray:
        """Taps según ``profile``, en SI (fs = Rs·sps, B = B_rel·Rs)."""
        if self.profile == "taps":
            return self.taps
        if self.profile == "echo":
            return echo(self.echoes, K=self.K)
        fs = self.Rs * self.sps
        if self.profile == "cd_short":
            return cd_short(fs, D_si(self.D_ps_nm_km), self.wavelength_nm * 1e-9, self.length_km * 1e3)
        B = self.B_rel * self.Rs
        K = self.K if self.K is not None else K_rule(B, self.Rs, self.order)
        if not (1 <= K <= K_MAX):
            raise ValueError(f"isi: K tiene que cumplir 1 <= K <= {K_MAX}: {K!r}")
        if self.profile == "gaussian":
            return gaussian(B, fs, K)
        if self.profile in ("bessel", "butter"):
            return analog(self.profile, B, fs, self.order, K)
        if self.H_measured is None:
            raise ValueError('isi: profile="measured" necesita H_measured')
        return measured(self.H_measured, fs, K)

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
