"""Réplica bit a bit de la CD en punto fijo (paso C3.2.2 de GUIA_ISI_CD.md).

Cadena (decisión 31): FFT DIF radix-2² → ×H̄ en orden bit invertido → IFFT DIT,
sin reordenamiento, dentro de un overlap-save con L = 2N. Especificación: PDF de
arquitectura de la FFT, secciones "Descomposición por etapas" y "Aritmética bit
a bit". Este módulo la implementa tal cual; no la rediseña.

Etapas (``fft_stages``): si log₂L es impar, primero una R2 suelta; después pares
BF2I + BF2II hasta 4 puntos. Cada etapa es (tipo, d, M), con M = 2d en I y R2 y
M = 4d en II; para la posición j, q = j mod M. En la FFT, por etapa:

  1. solo en BF2II, ×(−j) donde q ≥ 3M/4 (exacto; negar −2 satura);
  2. mariposa: a + b en j y a − b en j + d para (j mod 2d) < d; desplazamiento s;
  3. rotación por W_M^e (``twiddle_exponents``): después de R2, e = q − d si
     q ≥ d; después de BF2II, e = n₃(k₁ + 2k₂); después de BF2I, nada;
  4. rotación trivial si e mod (M/4) = 0: 1, −j, −1 o j, exacta.

La IFFT DIT recorre las etapas al revés: rotación con el twiddle conjugado,
mariposa y, en BF2II, ×(+j) donde q ≥ 3M/4. Entra en bit invertido, sale en orden
natural y no divide por L. El producto por H̄ en la posición p usa H̄[bitrev(p)]
(``hbar_permuted``, la tabla que carga el host).

Aritmética (``CdFixedConfig``, decisión 32):
  - entrada ``<W_in, 1>`` (ADC) → datos ``<W_d, 2>`` por extensión de signo, exacta;
  - mariposa: a ± b exacto con W_d + 1 bits, desplazamiento de s bits como cambio
    exacto de escala y un solo redondeo a W_d (``AC_RND_CONV``) con ``AC_SAT``;
  - rotación no trivial y ×H̄: producto complejo exacto y un solo redondeo por
    componente a W_d (``AC_RND``) con ``AC_SAT``;
  - tablas de twiddles (una sola, W_4096^m) y H̄: ``AC_RND`` a ``<W_t, 1>`` y
    ``<W_h, 1>``; +1 satura al máximo;
  - salida ``<W_out, I_out>`` (DAC): ``AC_RND_CONV`` y ``AC_SAT``.

Escalado (``scaling_default``, "mitad + 1"): s_f = 1 en cada BF2I y R2 y en la
segunda etapa; s_i = 1 − s_f. Se define por versión de L (no es la cola).

Todo en enteros int64 sobre ``integer.round_shift`` y ``integer.saturate_int``.
Hay también un motor en flotante con la misma descomposición (``fft_dif_float``,
``ifft_dit_float``), que sirve para verificar la descomposición contra numpy y
para ubicar en qué etapa se acumula el error (``stage_errors``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import HardwareParams
from ..quantization import to_int
from .integer import ROUNDINGS, round_shift, saturate_int

L_MAX = 4096  # tamaño de la tabla de twiddles (N_CD = 2048, decisión 30)
_W_PROD_MAX = 62  # productos y sumas en int64


# --------------------------------------------------------------------------
# Descomposición por etapas
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    """Etapa de la FFT: tipo ("R2", "I" o "II"), distancia d y tamaño M de la sub-FFT."""

    kind: str
    d: int
    M: int


def _log2(L: int) -> int:
    n = int(L).bit_length() - 1
    if L < 2 or (1 << n) != L:
        raise ValueError(f"L tiene que ser potencia de 2 y >= 2: {L!r}")
    return n


def fft_stages(L: int) -> list[Stage]:
    """Etapas de la FFT DIF de L puntos, en orden (tabla de "Descomposición por etapas")."""
    n = _log2(L)
    stages = []
    m = L
    if n % 2:
        stages.append(Stage("R2", m // 2, m))
        m //= 2
    while m >= 4:
        stages.append(Stage("I", m // 2, m))
        stages.append(Stage("II", m // 4, m))
        m //= 4
    return stages


def twiddle_exponents(stage: Stage, L: int) -> np.ndarray | None:
    """Exponente e de la rotación W_M^e después de la etapa, por posición j (−1: sin rotación).

    None si la etapa no rota (BF2I).
    """
    M, d = stage.M, stage.d
    q = np.arange(L) % M
    if stage.kind == "I":
        return None
    if stage.kind == "R2":
        return np.where(q >= d, q - d, -1)
    k1 = q // (M // 2)
    k2 = (q // (M // 4)) % 2
    n3 = q % (M // 4)
    return n3 * (k1 + 2 * k2)


def bitrev(L: int) -> np.ndarray:
    """Permutación de inversión de bits de log₂L bits: p -> bitrev(p)."""
    n = _log2(L)
    j = np.arange(L)
    r = np.zeros(L, dtype=np.int64)
    for b in range(n):
        r |= ((j >> b) & 1) << (n - 1 - b)
    return r


def scaling_default(L: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Escalado "mitad + 1" de la versión L: (s_f, s_i), un desplazamiento por etapa.

    s_f = 1 en cada BF2I y R2 y en la segunda etapa (⌊log₂L/2⌋ + 1 bits); s_i = 1 − s_f,
    así que s_f + s_i suman log₂L (ganancia 1).
    """
    stages = fft_stages(L)
    s_f = tuple(1 if (st.kind in ("I", "R2") or i == 1) else 0 for i, st in enumerate(stages))
    return s_f, tuple(1 - s for s in s_f)


# --------------------------------------------------------------------------
# Tablas (las genera el host; las leen el RTL y HALCON)
# --------------------------------------------------------------------------
def twiddle_table(W_t: int = 18, L_max: int = L_MAX, rounding: str = "half_up") -> tuple[np.ndarray, np.ndarray]:
    """Tabla de twiddles W_{L_max}^m, m = 0 … L_max − 1, en ``<W_t, 1>`` (``AC_RND``, ``AC_SAT``).

    W_M^e = W_{L_max}^(e·L_max/M): una sola tabla sirve para todas las etapas y versiones.
    - return: (códigos de la parte real, códigos de la parte imaginaria).
    """
    w = np.exp(-2j * np.pi * np.arange(L_max) / L_max)
    return to_int(w, W_t, 1, rounding=rounding)


def hbar_permuted(Hbar, W_h: int = 18, rounding: str = "half_up") -> tuple[np.ndarray, np.ndarray]:
    """H̄ en el orden del flujo (posición p -> H̄[bitrev(p)]), en ``<W_h, 1>`` (``AC_RND``, ``AC_SAT``)."""
    Hbar = np.asarray(Hbar, dtype=complex)
    return to_int(Hbar[bitrev(Hbar.size)], W_h, 1, rounding=rounding)


# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CdFixedConfig:
    """Formatos y modos de la cadena (decisión 32). ``s_f``/``s_i`` None = ``scaling_default(L)``."""

    W_in: int = 14
    W_d: int = 20
    W_t: int = 18
    W_h: int = 18
    W_out: int = 14
    I_out: int = 2
    in_rounding: str = "half_even"
    bf_rounding: str = "half_even"
    mul_rounding: str = "half_up"
    table_rounding: str = "half_up"
    out_rounding: str = "half_even"
    s_f: tuple[int, ...] | None = None
    s_i: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.W_d - 2 < self.W_in - 1:
            raise ValueError(f"CdFixedConfig: W_d − 2 >= W_in − 1 (extensión de signo exacta): W_in={self.W_in}, W_d={self.W_d}")
        if (self.W_d - 1) + (max(self.W_t, self.W_h) - 1) + 1 > _W_PROD_MAX:
            raise ValueError(f"CdFixedConfig: los productos de W_d={self.W_d} y W_t/W_h={max(self.W_t, self.W_h)} no entran en int64")
        for r in (self.in_rounding, self.bf_rounding, self.mul_rounding, self.table_rounding, self.out_rounding):
            if r not in ROUNDINGS:
                raise ValueError(f"CdFixedConfig: redondeo inválido {r!r}")

    def shifts(self, L: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        s_f, s_i = scaling_default(L)
        s_f = s_f if self.s_f is None else tuple(self.s_f)
        s_i = s_i if self.s_i is None else tuple(self.s_i)
        n = _log2(L)
        if len(s_f) != n or len(s_i) != n:
            raise ValueError(f"CdFixedConfig: s_f y s_i tienen que tener {n} etapas para L = {L}")
        return s_f, s_i


# --------------------------------------------------------------------------
# Aritmética: flotante (exacta) y entera (bit a bit)
# --------------------------------------------------------------------------
class _FloatArith:
    """Datos complejos; mismas operaciones sin cuantizar (para verificar y diagnosticar)."""

    def __init__(self):
        self.saturations = 0

    @staticmethod
    def bfly(z, d, s):
        shp = z.shape
        v = z.reshape(shp[:-1] + (-1, 2, d))
        a, b = v[..., 0, :], v[..., 1, :]
        return (np.stack([a + b, a - b], axis=-2) / 2.0**s).reshape(shp)

    @staticmethod
    def mul_j(z, mask, plus):
        out = z.copy()
        out[..., mask] *= 1j if plus else -1j
        return out

    @staticmethod
    def rotate(z, e, M, conj):
        act = e >= 0
        w = np.exp(-2j * np.pi * e[act] / M)
        out = z.copy()
        out[..., act] *= np.conj(w) if conj else w
        return out

    @staticmethod
    def mul_table(z, H):
        return z * H


class _IntArith:
    """Datos como (códigos I, códigos Q) en ``<W_d, 2>``; cuenta saturaciones."""

    def __init__(self, cfg: CdFixedConfig):
        self.cfg = cfg
        self.W = cfg.W_d
        self.wr, self.wi = twiddle_table(cfg.W_t, rounding=cfg.table_rounding)
        self.saturations = 0

    def _sat(self, k):
        k, n = saturate_int(k, self.W)
        self.saturations += n
        return k

    def _neg(self, k):
        return self._sat(-k)

    def bfly(self, z, d, s):
        out = []
        for c in z:
            shp = c.shape
            v = c.reshape(shp[:-1] + (-1, 2, d))
            a, b = v[..., 0, :], v[..., 1, :]
            u = np.stack([a + b, a - b], axis=-2)  # exacto con W_d + 1 bits
            out.append(self._sat(round_shift(u, s, self.cfg.bf_rounding)).reshape(shp))
        return tuple(out)

    def mul_j(self, z, mask, plus):
        re, im = z[0].copy(), z[1].copy()
        a, b = z[0][..., mask], z[1][..., mask]
        if plus:  # ×(+j): (a, b) -> (−b, a)
            re[..., mask], im[..., mask] = self._neg(b), a
        else:  # ×(−j): (a, b) -> (b, −a)
            re[..., mask], im[..., mask] = b, self._neg(a)
        return re, im

    def _cmul(self, a, b, c, d, F, conj):
        # (a + jb)(c ± jd) exacto y un solo redondeo por componente
        if conj:
            pr, pi = a * c + b * d, b * c - a * d
        else:
            pr, pi = a * c - b * d, a * d + b * c
        r = self.cfg.mul_rounding
        return self._sat(round_shift(pr, F, r)), self._sat(round_shift(pi, F, r))

    def rotate(self, z, e, M, conj):
        re, im = z[0].copy(), z[1].copy()
        act = e >= 0
        triv = act & (e % (M // 4) == 0)
        k = np.where(triv, (e // (M // 4)) % 4, 0)
        if conj:
            k = (-k) % 4
        a, b = z
        for kk in (1, 2, 3):
            m = triv & (k == kk)
            if not np.any(m):
                continue
            am, bm = a[..., m], b[..., m]
            if kk == 1:  # ×(−j)
                re[..., m], im[..., m] = bm, self._neg(am)
            elif kk == 2:  # ×(−1)
                re[..., m], im[..., m] = self._neg(am), self._neg(bm)
            else:  # ×(+j)
                re[..., m], im[..., m] = self._neg(bm), am
        nt = act & ~triv
        if np.any(nt):
            idx = e[nt] * (L_MAX // M)
            re[..., nt], im[..., nt] = self._cmul(a[..., nt], b[..., nt], self.wr[idx], self.wi[idx], self.cfg.W_t - 1, conj)
        return re, im

    def mul_table(self, z, H):
        hr, hi = H
        return self._cmul(z[0], z[1], hr, hi, self.cfg.W_h - 1, conj=False)


def _transform(z, L, ar, s, inverse, record=None, prefix=""):
    """FFT DIF (inverse=False) o IFFT DIT (inverse=True) sobre el último eje."""
    stages = fft_stages(L)
    if len(s) != len(stages):
        raise ValueError(f"hacen falta {len(stages)} desplazamientos para L = {L}: {len(s)}")
    j = np.arange(L)
    order = range(len(stages) - 1, -1, -1) if inverse else range(len(stages))
    for n_done, i in enumerate(order):
        st = stages[i]
        e = twiddle_exponents(st, L)
        mask = (j % st.M) >= 3 * st.M // 4
        if not inverse:
            if st.kind == "II":
                z = ar.mul_j(z, mask, plus=False)
            z = ar.bfly(z, st.d, s[i])
            if e is not None:
                z = ar.rotate(z, e, st.M, conj=False)
        else:
            if e is not None:
                z = ar.rotate(z, e, st.M, conj=True)
            z = ar.bfly(z, st.d, s[i])
            if st.kind == "II":
                z = ar.mul_j(z, mask, plus=True)
        if record is not None:
            record(f"{prefix} {n_done}", z)
    return z


def fft_dif_float(x, s=None) -> np.ndarray:
    """FFT DIF radix-2² en flotante (sin cuantizar), salida en orden bit invertido.

    Con s = None (sin desplazamientos) es ``np.fft.fft(x)[..., bitrev(L)]``.
    """
    x = np.asarray(x, dtype=complex)
    L = x.shape[-1]
    s = [0] * len(fft_stages(L)) if s is None else s
    return _transform(x, L, _FloatArith(), s, inverse=False)


def ifft_dit_float(X, s=None) -> np.ndarray:
    """IFFT DIT en flotante: entrada en bit invertido, salida en orden natural, sin 1/L."""
    X = np.asarray(X, dtype=complex)
    L = X.shape[-1]
    s = [0] * len(fft_stages(L)) if s is None else s
    return _transform(X, L, _FloatArith(), s, inverse=True)


def fft_dif_fixed(re, im, cfg: CdFixedConfig, s) -> tuple[np.ndarray, np.ndarray, int]:
    """FFT DIF bit a bit sobre códigos ``<W_d, 2>`` (último eje = L). Devuelve (I, Q, saturaciones)."""
    ar = _IntArith(cfg)
    re, im = np.asarray(re, np.int64), np.asarray(im, np.int64)
    zr, zi = _transform((re, im), re.shape[-1], ar, s, inverse=False)
    return zr, zi, ar.saturations


def ifft_dit_fixed(re, im, cfg: CdFixedConfig, s) -> tuple[np.ndarray, np.ndarray, int]:
    """IFFT DIT bit a bit sobre códigos ``<W_d, 2>`` en bit invertido. Devuelve (I, Q, saturaciones)."""
    ar = _IntArith(cfg)
    re, im = np.asarray(re, np.int64), np.asarray(im, np.int64)
    zr, zi = _transform((re, im), re.shape[-1], ar, s, inverse=True)
    return zr, zi, ar.saturations


# --------------------------------------------------------------------------
# Overlap-save completo
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CdFixedResult:
    """Salida de ``cd_overlap_save_fixed``.

    - y: salida al DAC, valores en la grilla de ``<W_out, I_out>``; forma (nb·N,),
      retrasada N/2 como ``cd_overlap_save``.
    - codes: (k_I, k_Q) de la salida.
    - y_internal: salida de la IFFT antes del DAC, en la grilla de ``<W_d, 2>``.
    - saturations: por punto: "in" (ADC), "fft", "hbar", "ifft" y "out" (DAC).
    - trace: con ``trace=True``, [(nombre, valores (nb, L))] después de cada etapa
      ("fft i", "hbar", "ifft i").
    - stage_saturations: saturaciones de cada etapa, en el mismo orden que trace.
    """

    y: np.ndarray
    codes: tuple[np.ndarray, np.ndarray]
    y_internal: np.ndarray
    saturations: dict[str, int]
    trace: list = field(default_factory=list)
    stage_saturations: list = field(default_factory=list)


def _frames(x, N):
    n = x.shape[-1]
    nb = -(-n // N)
    xz = np.pad(x, (N, nb * N - n))
    return xz[np.arange(nb)[:, None] * N + np.arange(2 * N)], nb


def cd_overlap_save_fixed(x, Hbar, cfg: CdFixedConfig = CdFixedConfig(), trace: bool = False) -> CdFixedResult:
    """Overlap-save bit a bit: ADC → FFT DIF → ×H̄ (bit invertido) → IFFT DIT → DAC.

    - x: señal 1-D compleja, escalada al rango de ``<W_in, 1>`` (decisión 34: pico 0,95).
    - Hbar: filtro de ``cd_hbar`` (orden de np.fft, 2N puntos); se cuantiza y permuta
      con ``hbar_permuted``. N tiene que ser una versión del hardware (512 o 2048,
      decisión 30): ninguna usa la etapa R2, que solo aparece con log₂L impar. Las
      transformadas sueltas (``fft_dif_fixed``, ``ifft_dit_fixed``) aceptan cualquier L.
    - cfg: formatos, modos y escalado.
    - trace: guarda la salida de cada etapa (para ``stage_errors``).
    - return: ``CdFixedResult``; la salida tiene la forma de ``cd_overlap_save``.
    """
    x = np.asarray(x, dtype=complex)
    Hbar = np.asarray(Hbar, dtype=complex)
    if x.ndim != 1:
        raise ValueError(f"cd_overlap_save_fixed: x tiene que ser 1-D: {x.shape}")
    L = Hbar.size
    N = L // 2
    if N not in HardwareParams.N_CD_VALUES or L != 2 * N:
        raise ValueError(f"cd_overlap_save_fixed: N = L/2 tiene que ser uno de {HardwareParams.N_CD_VALUES} (decisión 30): L = {L}")
    s_f, s_i = cfg.shifts(L)
    F_in, F_d, F_out = cfg.W_in - 1, cfg.W_d - 2, cfg.W_out - cfg.I_out
    sat = {}

    # ADC: <W_in, 1> con saturación; después extensión de signo a <W_d, 2> (exacta)
    xr, xi = to_int(x, cfg.W_in, 1, rounding=cfg.in_rounding)
    wr_, wi_ = to_int(x, cfg.W_in + 16, 17, rounding=cfg.in_rounding)  # mismo paso, sin saturar
    sat["in"] = int(np.count_nonzero(wr_ != xr) + np.count_nonzero(wi_ != xi))
    fr, nb = _frames(xr, N)
    fi, _ = _frames(xi, N)
    z = (fr << (F_d - F_in), fi << (F_d - F_in))

    rec, stage_sat = [], []
    ar = _IntArith(cfg)
    last = [0]

    def record(name, zz):
        if trace:
            rec.append((name.strip(), (zz[0] + 1j * zz[1]) * 2.0**-F_d))
            stage_sat.append(ar.saturations - last[0])
            last[0] = ar.saturations

    z = _transform(z, L, ar, s_f, inverse=False, record=record, prefix="fft")
    sat["fft"] = ar.saturations
    n0 = ar.saturations
    z = ar.mul_table(z, hbar_permuted(Hbar, cfg.W_h, rounding=cfg.table_rounding))
    record("hbar", z)
    sat["hbar"] = ar.saturations - n0
    n0 = ar.saturations
    z = _transform(z, L, ar, s_i, inverse=True, record=record, prefix="ifft")
    sat["ifft"] = ar.saturations - n0

    yr, yi = z[0][:, N:].reshape(-1), z[1][:, N:].reshape(-1)
    y_internal = (yr + 1j * yi) * 2.0**-F_d
    # DAC: <W_out, I_out> con un solo redondeo y saturación
    if F_out <= F_d:
        orr, oi = round_shift(yr, F_d - F_out, cfg.out_rounding), round_shift(yi, F_d - F_out, cfg.out_rounding)
    else:
        orr, oi = yr << (F_out - F_d), yi << (F_out - F_d)
    orr, n1 = saturate_int(orr, cfg.W_out)
    oi, n2 = saturate_int(oi, cfg.W_out)
    sat["out"] = n1 + n2
    y = (orr + 1j * oi) * 2.0**-F_out
    return CdFixedResult(y=y, codes=(orr, oi), y_internal=y_internal, saturations=sat, trace=rec, stage_saturations=stage_sat)


def stage_errors(x, Hbar, cfg: CdFixedConfig = CdFixedConfig()) -> list[tuple[str, float, int]]:
    """Error acumulado después de cada etapa: (nombre, NRMSE, saturaciones de la etapa).

    Compara la réplica bit a bit con el mismo grafo en flotante (twiddles y H̄
    exactos, mismos desplazamientos) sobre la misma entrada cuantizada al ADC, así
    que mide solo el error que agrega la cadena hasta esa etapa.
    """
    x = np.asarray(x, dtype=complex)
    Hbar = np.asarray(Hbar, dtype=complex)
    L = Hbar.size
    N = L // 2
    s_f, s_i = cfg.shifts(L)
    res = cd_overlap_save_fixed(x, Hbar, cfg, trace=True)
    F_in = cfg.W_in - 1
    xr, xi = to_int(x, cfg.W_in, 1, rounding=cfg.in_rounding)
    fr, _ = _frames(xr, N)
    fi, _ = _frames(xi, N)
    zf = (fr + 1j * fi) * 2.0**-F_in
    ref = []

    def record(name, zz):
        ref.append(zz)

    zf = _transform(zf, L, _FloatArith(), s_f, inverse=False, record=record, prefix="fft")
    zf = zf * Hbar[bitrev(L)]
    record("hbar", zf)
    _transform(zf, L, _FloatArith(), s_i, inverse=True, record=record, prefix="ifft")
    out = []
    for (name, v), r, ns in zip(res.trace, ref, res.stage_saturations):
        den = np.sqrt(np.sum(np.abs(r) ** 2))
        out.append((name, float(np.sqrt(np.sum(np.abs(v - r) ** 2)) / den) if den > 0 else 0.0, ns))
    return out
