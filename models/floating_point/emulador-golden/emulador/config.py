"""Escenario de simulación, constantes y conversión de unidades.

Todo lo que depende de la velocidad del sistema (fs, Ts, el B de los perfiles de
ISI, R y N de la CD) se deriva de un único ``Scenario``: cambiar de 32 a 100 GBd no
requiere tocar ningún otro módulo (GUIA_ISI_CD.md §1, decisión 1).

CONVENCIONES (referencia_guia_ISI_CD.md §3):
  - Unidades SI en todo el cálculo interno: D en s/m^2, S en s/m^3, lambda y z en m,
    beta2 en s^2/m, beta3 en s^3/m. Las unidades de hoja de datos se convierten acá.
  - Señal: envolvente compleja en banda base, complex128; f es la desviación
    respecto de la portadora, en [-fs/2, fs/2).
  - Transformada: la de numpy.fft, X[k] = sum_n x[n] exp(-j 2 pi k n / L).
  - Eje de frecuencias: siempre np.fft.fftfreq(L, 1/fs); fftshift solo para graficar.
  - Signo de la CD: positivo, H_CD(f) = exp(+j pi D lambda^2 z f^2 / c)
    (GUIA_ISI_CD.md §1, decisión 2).
  - Retardos: cada bloque declara su retardo entero en muestras (``delay_samples``).
  - Aleatoriedad: np.random.default_rng(seed); nunca el estado global de numpy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# --------------------------------------------------------------------------
# Constantes y conversión de unidades
# --------------------------------------------------------------------------
C = 299_792_458.0  # velocidad de la luz en el vacío [m/s] (exacta)


def D_si(D_ps_nm_km: float) -> float:
    """Parámetro de dispersión: ps/(nm·km) -> s/m^2.

    1 ps/(nm·km) = 1e-12 / (1e-9 · 1e3) = 1e-6 s/m^2  (17 ps/nm/km -> 1.7e-5 s/m^2).
    """
    return D_ps_nm_km * 1e-6


def S_si(S_ps_nm2_km: float) -> float:
    """Pendiente de dispersión: ps/(nm^2·km) -> s/m^3.

    1 ps/(nm^2·km) = 1e-12 / (1e-18 · 1e3) = 1e3 s/m^3  (0.057 ps/nm^2/km -> 57 s/m^3).
    """
    return S_ps_nm2_km * 1e3


# --------------------------------------------------------------------------
# Escenario del canal
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """Escenario de simulación. Por defecto: 32 GBd, 2 sps -> fs = 64 GSa/s.

    Rs         : tasa de símbolos [Bd], de 1 a 300 GBd.
    sps        : muestras por símbolo: 1, 2, 4, 8 o 16 (16 solo para análisis fino).
    wavelength : longitud de onda de la portadora [m], banda C (1530 a 1565 nm).
    rolloff    : roll-off del RRC, de 0.05 a 0.5.
    """

    Rs: float = 32e9
    sps: int = 2
    wavelength: float = 1550e-9
    rolloff: float = 0.1

    def __post_init__(self):
        if not (1e9 <= self.Rs <= 300e9):
            raise ValueError(f"Rs fuera de rango (1 a 300 GBd): {self.Rs!r}")
        if self.sps not in (1, 2, 4, 8, 16):
            raise ValueError(f"sps debe ser 1, 2, 4, 8 o 16: {self.sps!r}")
        if not (1530e-9 <= self.wavelength <= 1565e-9):
            raise ValueError(f"wavelength fuera de la banda C (1530 a 1565 nm): {self.wavelength!r}")
        if not (0.05 <= self.rolloff <= 0.5):
            raise ValueError(f"rolloff fuera de rango (0.05 a 0.5): {self.rolloff!r}")

    @property
    def fs(self) -> float:
        """Frecuencia de muestreo [Sa/s]."""
        return self.Rs * self.sps

    @property
    def Ts(self) -> float:
        """Período de muestreo [s]."""
        return 1.0 / self.fs

    def to_dict(self) -> dict:
        """Parámetros y magnitudes derivadas, para los meta.json."""
        return {**asdict(self), "fs": self.fs, "Ts": self.Ts}


# --------------------------------------------------------------------------
# Parámetros de la implementación en hardware
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareParams:
    """Parámetros del hardware, separados del escenario (describen la
    implementación, no el canal). GUIA_ISI_CD.md §1, decisión 7.

    VALORES PROVISORIOS hasta confirmar el modelo de RFSoC (paso C1 de la guía).

    P     : muestras por ciclo de la FFT radix-2^2 (potencia de 2).
    f_clk : frecuencia de reloj [Hz]; None mientras no esté definida.
    N_max : máximo tamaño de bloque del overlap-save (potencia de 2, >= 128).
    """

    P: int = 1
    f_clk: float | None = None
    N_max: int = 2048
    provisional: bool = True

    def __post_init__(self):
        if self.P < 1 or self.P & (self.P - 1):
            raise ValueError(f"P debe ser una potencia de 2: {self.P!r}")
        if self.N_max < 128 or self.N_max & (self.N_max - 1):
            raise ValueError(f"N_max debe ser una potencia de 2 >= 128: {self.N_max!r}")
        if self.f_clk is not None and self.f_clk <= 0:
            raise ValueError(f"f_clk debe ser positiva: {self.f_clk!r}")

    def to_dict(self) -> dict:
        return asdict(self)
