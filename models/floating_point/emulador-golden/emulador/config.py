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

QUIÉN USA CADA ELEMENTO (explicado en notebooks/00_escenario_y_convenciones.ipynb):
  - C, D_si, S_si  : los bloques, para pasar sus campos de hoja de datos a SI, y
                     emulador/cd/, que calcula en SI.
  - Scenario       : el canal (Rs, sps, lambda, roll-off -> fs, Ts). Lo usan los
                     perfiles de ISI (B = B_rel·Rs), la CD (R, N), los estímulos,
                     el receptor y los meta.json de los vectores exportados.
  - HardwareParams : la implementación (P, f_clk, N_CD), separada del canal
                     porque un mismo canal admite distinto hardware. Lo usan cd_N
                     y el modelo bit a bit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from numbers import Real
from typing import ClassVar

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

    Rs         : tasa de símbolos [Bd], de 1 a 480 GBd.
    sps        : muestras por símbolo: entero 1, 2, 4, 8 o 16 (16 solo para análisis
                 fino), o racional up/down entre 1 y 16 con down <= 16, como la
                 tasa del proyecto, 1,2 = 6/5 (``Fraction(6, 5)``; un float como
                 1.2 se convierte a esa fracción). Con sps racional la señal se
                 arma a ``sps_up`` muestras por símbolo y se decima por
                 ``sps_down`` (``stimuli.shape``).
    wavelength : longitud de onda de la portadora [m], banda C (1530 a 1565 nm).
    rolloff    : roll-off del RRC, de 0.05 a 0.5.
    """

    Rs: float = 32e9
    sps: int | Fraction = 2
    wavelength: float = 1550e-9
    rolloff: float = 0.1

    SPS_INT: ClassVar[tuple[int, ...]] = (1, 2, 4, 8, 16)
    SPS_DOWN_MAX: ClassVar[int] = 16

    def __post_init__(self):
        if not (1e9 <= self.Rs <= 480e9):
            raise ValueError(f"Rs fuera de rango (1 a 480 GBd): {self.Rs!r}")
        object.__setattr__(self, "sps", self._normalize_sps(self.sps))
        if not (1530e-9 <= self.wavelength <= 1565e-9):
            raise ValueError(f"wavelength fuera de la banda C (1530 a 1565 nm): {self.wavelength!r}")
        if not (0.05 <= self.rolloff <= 0.5):
            raise ValueError(f"rolloff fuera de rango (0.05 a 0.5): {self.rolloff!r}")

    @classmethod
    def _normalize_sps(cls, sps):
        """int de SPS_INT, o Fraction no entera en (1, 16) con denominador <= 16."""
        if isinstance(sps, bool) or not isinstance(sps, (Real, Fraction)):
            raise ValueError(f"sps tiene que ser un número: {sps!r}")
        f = Fraction(sps).limit_denominator(cls.SPS_DOWN_MAX)
        if abs(float(f) - float(sps)) > 1e-12 * float(sps):
            raise ValueError(f"sps tiene que ser racional con denominador <= {cls.SPS_DOWN_MAX}: {sps!r}")
        if f.denominator == 1:
            if int(f) not in cls.SPS_INT:
                raise ValueError(f"sps entero debe ser 1, 2, 4, 8 o 16: {sps!r}")
            return int(f)
        if not (1 < f < 16):
            raise ValueError(f"sps racional tiene que estar entre 1 y 16: {sps!r}")
        return f

    @property
    def sps_up(self) -> int:
        """Muestras por símbolo con que se arma la señal (numerador de sps)."""
        return Fraction(self.sps).numerator

    @property
    def sps_down(self) -> int:
        """Factor de decimación después de armarla (denominador de sps; 1 si es entero)."""
        return Fraction(self.sps).denominator

    @property
    def fs(self) -> float:
        """Frecuencia de muestreo [Sa/s]."""
        return float(self.Rs * self.sps)

    @property
    def Ts(self) -> float:
        """Período de muestreo [s]."""
        return 1.0 / self.fs

    def to_dict(self) -> dict:
        """Parámetros y magnitudes derivadas, para los meta.json.

        Con sps racional, "sps" va como float y se agregan "sps_up" y "sps_down"
        (con sps entero el diccionario no cambia).
        """
        d = {**asdict(self), "fs": self.fs, "Ts": self.Ts}
        if self.sps_down != 1:
            d["sps"] = float(self.sps)
            d["sps_up"], d["sps_down"] = self.sps_up, self.sps_down
        return d


# --------------------------------------------------------------------------
# Parámetros de la implementación en hardware
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareParams:
    """Parámetros del hardware, separados del escenario (describen la
    implementación, no el canal). GUIA_ISI_CD.md §1, decisión 7.

    P y f_clk son PROVISORIOS (placa ZCU208 confirmada; P y f_clk sin definir).

    P     : muestras por ciclo de la FFT radix-2^2 (potencia de 2).
    f_clk : frecuencia de reloj [Hz]; None mientras no esté definida.
    N_CD  : tamaño de bloque del overlap-save de la CD (FFT de L = 2·N_CD).
            Es un parámetro de SÍNTESIS (dos versiones del hardware), fijo en
            ejecución: en ejecución solo cambia H̄. Solo 512 o 2048 (decisión 30).
    """

    N_CD_VALUES: ClassVar[tuple[int, ...]] = (512, 2048)

    P: int = 1
    f_clk: float | None = None
    N_CD: int = 2048
    provisional: bool = True

    def __post_init__(self):
        if self.P < 1 or self.P & (self.P - 1):
            raise ValueError(f"P debe ser una potencia de 2: {self.P!r}")
        if self.N_CD not in self.N_CD_VALUES:
            raise ValueError(f"N_CD debe ser uno de {self.N_CD_VALUES}: {self.N_CD!r}")
        if self.f_clk is not None and self.f_clk <= 0:
            raise ValueError(f"f_clk debe ser positiva: {self.f_clk!r}")

    def to_dict(self) -> dict:
        return asdict(self)
