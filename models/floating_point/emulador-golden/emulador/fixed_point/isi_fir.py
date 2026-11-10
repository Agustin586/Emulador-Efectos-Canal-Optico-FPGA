"""FIR complejo de la ISI en punto fijo, bit a bit (paso C3.1 de GUIA_ISI_CD.md).

Estructura (decisión 32, punto de partida de la ISI):

    x[n] --(ADC)--> <W_x, I_x>          h[k] --> <W_h, I_h>
    productos reales exactos <W_x + W_h, I_x + I_h>
    parte real   Σ_k (x_r h_r - x_i h_i),  parte imaginaria  Σ_k (x_r h_i + x_i h_r)
    acumulador <W_acc, I_acc>, I_acc = I_x + I_h + 1 + ceil(log2 K)
    salida --(DAC)--> <W_out, I_out>

Puntos de cuantización:
  - entrada y coeficientes: ``to_int`` con ``rounding`` y saturación;
  - acumulador: con ``acc_frac=None`` (por defecto) guarda la precisión completa
    (F_x + F_h bits fraccionarios) y es exacto; con ``acc_frac`` = F < F_x + F_h,
    cada producto real se redondea a F bits con ``acc_rounding`` antes de
    sumarse (modela un acumulador más angosto). Los log2 K bits de crecimiento
    más el de la suma compleja hacen que no pueda desbordar; igual se satura y
    se cuenta;
  - salida: un solo redondeo (``rounding``, ``AC_RND_CONV`` por defecto) y ``AC_SAT``.

Todo en enteros int64 (``integer.round_shift``, ``integer.saturate_int``): el
resultado es el que daría ``ac_fixed`` con los mismos formatos y modos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..quantization import to_int
from .integer import round_shift, saturate_int

_W_ACC_MAX = 62  # con códigos int64, el acumulador tiene que entrar en 62 bits + signo


@dataclass(frozen=True)
class IsiFixedResult:
    """Salida del FIR en punto fijo.

    - y: salida compleja, valores en la grilla de ``out_fmt`` (float exacto).
    - codes: (k_I, k_Q), códigos de la salida (int64).
    - acc_fmt: (W_acc, I_acc) del acumulador.
    - acc_peak: máximo |código| del acumulador (parte real o imaginaria), para
      medir el crecimiento real contra el reservado.
    - saturations: cantidad de componentes que saturaron en "x", "h", "acc" y "out".
    """

    y: np.ndarray
    codes: tuple[np.ndarray, np.ndarray]
    acc_fmt: tuple[int, int]
    acc_peak: int
    saturations: dict[str, int] = field(default_factory=dict)


def _codes_and_saturations(v: np.ndarray, W: int, I: int, rounding: str) -> tuple[np.ndarray, int]:
    """Códigos de una componente real y cuántos valores saturaron."""
    k = to_int(v, W, I, rounding=rounding)
    extra = min(53, W + 16) - W  # mismo paso, rango más amplio: detecta lo que satura
    wide = to_int(v, W + extra, I + extra, rounding=rounding)
    return k, int(np.count_nonzero(wide != k))


def isi_fir_fixed(
    x,
    h,
    *,
    x_fmt: tuple[int, int] = (14, 1),
    h_fmt: tuple[int, int] = (18, 1),
    out_fmt: tuple[int, int] = (14, 1),
    acc_frac: int | None = None,
    acc_rounding: str = "half_even",
    rounding: str = "half_even",
) -> IsiFixedResult:
    """FIR causal y[n] = Σ h[k] x[n-k] en punto fijo, mismo largo que x (como ``isi_apply``).

    - x: señal 1-D, real o compleja, ya escalada al rango de ``x_fmt``.
    - h: taps del FIR, 1-D, largo K.
    - x_fmt, h_fmt, out_fmt: formatos (W, I) de ``ac_fixed<W, I>`` (decisión 25).
    - acc_frac: bits fraccionarios del acumulador; None = precisión completa.
    - acc_rounding: redondeo de cada producto si ``acc_frac`` recorta bits.
    - rounding: redondeo de la entrada, los coeficientes y la salida (decisión 26).
    - return: ``IsiFixedResult``.

    Lanza ``ValueError`` si x no es 1-D, si el acumulador no entra en int64
    (W_acc > 62) o si ``acc_frac`` no está entre F_out y F_x + F_h.
    """
    x = np.asarray(x)
    h = np.asarray(h)
    if x.ndim != 1 or h.ndim != 1 or h.size == 0:
        raise ValueError(f"isi_fir_fixed: x y h tienen que ser 1-D (h no vacío): {x.shape}, {h.shape}")
    (W_x, I_x), (W_h, I_h), (W_out, I_out) = x_fmt, h_fmt, out_fmt
    F_x, F_h, F_out = W_x - I_x, W_h - I_h, W_out - I_out
    F_full = F_x + F_h
    F_acc = F_full if acc_frac is None else int(acc_frac)
    if not (F_out <= F_acc <= F_full) or F_acc < 1:
        raise ValueError(f"isi_fir_fixed: acc_frac tiene que estar entre {max(F_out, 1)} y {F_full}: {acc_frac!r}")
    K = h.size
    I_acc = I_x + I_h + 1 + int(np.ceil(np.log2(K)))
    W_acc = I_acc + F_acc
    if I_acc + F_full > _W_ACC_MAX:
        raise ValueError(f"isi_fir_fixed: el acumulador de precisión completa necesita {I_acc + F_full} bits (máximo {_W_ACC_MAX})")

    x = x.astype(complex)
    h = h.astype(complex)
    sat = {}
    xr, s_r = _codes_and_saturations(x.real, W_x, I_x, rounding)
    xi, s_i = _codes_and_saturations(x.imag, W_x, I_x, rounding)
    sat["x"] = s_r + s_i
    hr, s_r = _codes_and_saturations(h.real, W_h, I_h, rounding)
    hi, s_i = _codes_and_saturations(h.imag, W_h, I_h, rounding)
    sat["h"] = s_r + s_i

    n = x.size
    shift = F_full - F_acc
    if shift == 0:
        # precisión completa: convolución entera exacta
        def conv(a, b):
            return np.convolve(a, b)[:n]

        acc_r = conv(xr, hr) - conv(xi, hi)
        acc_i = conv(xr, hi) + conv(xi, hr)
    else:
        # cada producto real se redondea a F_acc bits antes de sumarse
        acc_r = np.zeros(n, dtype=np.int64)
        acc_i = np.zeros(n, dtype=np.int64)
        for k in range(min(K, n)):
            a = np.r_[np.zeros(k, np.int64), xr[: n - k]]
            b = np.r_[np.zeros(k, np.int64), xi[: n - k]]

            def p(u, c):
                return round_shift(u * c, shift, acc_rounding)

            acc_r += p(a, hr[k]) - p(b, hi[k])
            acc_i += p(a, hi[k]) + p(b, hr[k])

    acc_peak = int(max(np.max(np.abs(acc_r), initial=0), np.max(np.abs(acc_i), initial=0)))
    acc_r, s_r = saturate_int(acc_r, W_acc)
    acc_i, s_i = saturate_int(acc_i, W_acc)
    sat["acc"] = s_r + s_i

    out_r, s_r = saturate_int(round_shift(acc_r, F_acc - F_out, rounding), W_out)
    out_i, s_i = saturate_int(round_shift(acc_i, F_acc - F_out, rounding), W_out)
    sat["out"] = s_r + s_i

    y = (out_r + 1j * out_i) * 2.0**-F_out
    return IsiFixedResult(y=y, codes=(out_r, out_i), acc_fmt=(W_acc, I_acc), acc_peak=acc_peak, saturations=sat)
