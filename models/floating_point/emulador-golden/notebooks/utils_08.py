"""Utilidades del notebook 08_escenario_60gbd_1p2sps (tarea D1.1).

Escenario del documento común del equipo (21/11/2026): DP-16QAM, 60 GBd,
1,2 sps (6/5), fs = 72 GSa/s, 4 canales reales XI, XQ, YI, YQ en Q1.15.

NO es parte del núcleo del golden: usa sus funciones (``prbs``, ``rrc_taps``,
``cd_reference``, ``gaussian``, ``analog``, ``K_rule``, ``nrmse``, ``quantize``)
sin modificarlas. Solo mide; no toma decisiones de diseño.

Contenido:
  - señal común (4 PRBS15, Gray /3, RRC span 32 a 6 sps, diezmado por 5);
  - estadísticas de rango en Q1.15 (T1);
  - ajustes por mínimos cuadrados de la CD (T2, T3) y su precisión reducida (T4);
  - prototipos largos de la ISI y los cuatro criterios de K (T5);
  - costo de DSP por fórmula con FFA (T7).
"""
from __future__ import annotations

import csv
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.signal import fftconvolve

from emulador.cd import cd_R, cd_reference
from emulador.common import rrc_taps
from emulador.config import D_si, S_si, Scenario
from emulador.isi.profiles import analog, gaussian
from emulador.metrics import nrmse
from emulador.quantization import quantize
from emulador.stimuli import prbs

# --------------------------------------------------------------------------
# Escenario y parámetros del documento común
# --------------------------------------------------------------------------
ESC = Scenario(Rs=60e9, sps=Fraction(6, 5))
RS, FS = ESC.Rs, ESC.fs
N_SYM = 81_920
SPAN_TX = 32  # señal común (documento del equipo)
SPAN_RX = 64  # filtro adaptado de rx (decisión 15); este notebook no lo usa
RMS_TARGET = 0.25
CHANNELS = ("XI", "XQ", "YI", "YQ")
PRBS_ORDER = 15
PERIOD = 2**PRBS_ORDER - 1
OFFSETS = tuple(i * PERIOD // 4 for i in range(4))  # 0, P/4, P/2, 3P/4 (enteros)
BASE_SEED = 1  # estado del que se cuentan los desplazamientos
GRAY = np.array([-3.0, -1.0, 3.0, 1.0])  # índice 2·b0 + b1: 00, 01, 10, 11

D_PS = 17.0
S_PS = (0.0, 0.057)
LAM = ESC.wavelength
Q115 = (16, 1)  # ac_fixed<16, 1>: Q1.15
FS_MAX = 1 - 2.0**-15

OUT = Path(__file__).resolve().parents[1] / "docs" / "golden_model" / "resultados" / "08_escenario_60gbd"


# --------------------------------------------------------------------------
# Señal común
# --------------------------------------------------------------------------
def tributary_seeds() -> list[int]:
    """Estados iniciales (15 bits, MSB primero) separados OFFSETS bits de la PRBS con BASE_SEED."""
    b = prbs(PRBS_ORDER, PERIOD + PRBS_ORDER, seed=BASE_SEED)
    return [int("".join(map(str, b[o : o + PRBS_ORDER])), 2) for o in OFFSETS]


def common_signal() -> dict:
    """Señal común: dict con channels (4, n), levels (4, N_SYM), seeds, scale, rms."""
    p, q = ESC.sps_num, ESC.sps_den
    g = rrc_taps(ESC.rolloff, SPAN_TX, p)
    seeds = tributary_seeds()
    chans, levels = [], []
    for s in seeds:
        bits = prbs(PRBS_ORDER, 2 * N_SYM, seed=s).reshape(-1, 2)  # 2 bits por símbolo, MSB primero
        lv = GRAY[2 * bits[:, 0] + bits[:, 1]] / 3.0
        up = np.zeros(N_SYM * p)
        up[::p] = lv
        chans.append(np.convolve(up, g, mode="same")[::q])
        levels.append(lv)
    chans = np.array(chans)
    rms_raw = np.sqrt(np.mean(chans**2, axis=1))
    scale = RMS_TARGET / rms_raw.mean()
    chans = chans * scale
    return {
        "channels": chans,
        "levels": np.array(levels),
        "seeds": seeds,
        "scale": float(scale),
        "rms_raw": rms_raw,
        "rms": np.sqrt(np.mean(chans**2, axis=1)),
    }


def xcorr_max(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, int]:
    """Máximo de |ρ(l)| con |l| <= max_lag; ρ normalizado sin sesgo (por el solapamiento)."""
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    n = len(a)
    L = 1 << (2 * n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(a, L) * np.conj(np.fft.rfft(b, L)), L)
    lags = np.r_[np.arange(0, n), np.arange(-(n - 1), 0)]
    vals = np.r_[c[:n], c[L - (n - 1) :]]
    keep = np.abs(lags) <= max_lag
    rho = vals[keep] / (n - np.abs(lags[keep]))
    i = int(np.argmax(np.abs(rho)))
    return float(abs(rho[i])), int(lags[keep][i])


# --------------------------------------------------------------------------
# T1: rango en Q1.15
# --------------------------------------------------------------------------
def range_stats(v: np.ndarray) -> dict:
    """Estadísticas de un canal real: RMS, pico, cresta, |v| >= 1 y SQNR por Q1.15.

    sqnr_full_db : cuantización completa (redondeo a par + saturación) contra v.
    sqnr_sat_db  : solo el error de saturación: Q1.15 con saturación contra
                   Q1.15 sin saturar (mismo redondeo). inf si no satura.
    n_sat        : muestras que la saturación modifica.
    """
    v = np.asarray(v, float)
    W, I = Q115
    vq = quantize(v, W, I)
    k = np.round(v * 2.0 ** (W - I)) * 2.0 ** -(W - I)  # redondeo a par, sin saturar
    e_sat = vq - k
    ps = np.sum(v**2)
    es = np.sum(e_sat**2)
    rms = float(np.sqrt(np.mean(v**2)))
    peak = float(np.max(np.abs(v)))
    return {
        "rms": rms,
        "peak": peak,
        "crest": peak / rms,
        "n_ge1": int(np.sum(np.abs(v) >= 1.0)),
        "n_sat": int(np.sum(e_sat != 0)),
        "sqnr_full_db": float(10 * np.log10(ps / np.sum((vq - v) ** 2))),
        "sqnr_sat_db": float(10 * np.log10(ps / es)) if es > 0 else np.inf,
    }


def worst_case(rows: list[dict]) -> dict:
    """Resumen de 4 canales: RMS medio, pico máx., cresta máx., totales y SQNR mín."""
    return {
        "rms_mean": float(np.mean([r["rms"] for r in rows])),
        "peak_max": max(r["peak"] for r in rows),
        "crest_max": max(r["crest"] for r in rows),
        "n_ge1_total": sum(r["n_ge1"] for r in rows),
        "n_sat_total": sum(r["n_sat"] for r in rows),
        "sqnr_full_min_db": min(r["sqnr_full_db"] for r in rows),
        "sqnr_sat_min_db": min(r["sqnr_sat_db"] for r in rows),
    }


def cd_apply(X: np.ndarray, z_km: float, S_ps: float) -> np.ndarray:
    """cd_reference (FFT completa) con D = 17 ps/nm/km, λ = 1550 nm, z en km."""
    return cd_reference(X, FS, D_si(D_PS), LAM, z_km * 1e3, S_si(S_ps))


def cd_R_samples(z_km: float) -> int:
    return cd_R(FS, D_si(D_PS), LAM, z_km * 1e3)


# --------------------------------------------------------------------------
# T2 a T4: CD a distancias cortas
# --------------------------------------------------------------------------
def lag_matrix(x: np.ndarray, idx: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """A[i, j] = x[idx[i] - lags[j]] (los índices tienen que quedar dentro de x)."""
    return x[idx[:, None] - lags[None, :]]


def split_indices(n: int, edge: int) -> tuple[np.ndarray, np.ndarray]:
    """Primera mitad (ajuste) y segunda mitad (medición), sin ``edge`` muestras por borde."""
    h = n // 2
    return np.arange(edge, h - edge), np.arange(h + edge, n - edge)


def fit_fir_complex(x, y, K_max, train, test):
    """FIR complejo centrado de K impar (1 … K_max) por mínimos cuadrados.

    Devuelve {K: (h, nrmse_test)}; h[k] multiplica a x[n - (k - K//2)].
    """
    M = K_max // 2
    lags = np.arange(-M, M + 1)
    A_tr, A_te = lag_matrix(x, train, lags), lag_matrix(x, test, lags)
    out = {}
    for K in range(1, K_max + 1, 2):
        sel = slice(M - K // 2, M + K // 2 + 1)
        h = np.linalg.lstsq(A_tr[:, sel], y[train], rcond=None)[0]
        out[K] = (h, nrmse(A_te[:, sel] @ h, y[test]))
    return out


def asymmetry(h: np.ndarray) -> float:
    """||h - h invertido|| / ||h||: 0 si h[k] = h[-k]."""
    return float(np.linalg.norm(h - h[::-1]) / np.linalg.norm(h))


def _sym_columns(x, idx, K):
    """Columnas x[n-m] + x[n+m] (m = 1 … K//2) y x[n] (m = 0), para un filtro real simétrico."""
    cols = [x[idx]]
    for m in range(1, K // 2 + 1):
        cols.append(x[idx - m] + x[idx + m])
    return np.stack(cols, axis=1)


def sym_taps(u: np.ndarray) -> np.ndarray:
    """Coeficientes únicos u (u[0] central) -> filtro simétrico completo de 2·len(u) - 1."""
    return np.r_[u[:0:-1], u]


def fit_first_order(x, y, Ks, Kc, train, test):
    """y ≈ x + (c * x) + j·(s * x), con s y c reales y simétricos (Kc = 0: sin c).

    Devuelve (s, c, nrmse_test); s y c completos (simétricos).
    """
    def design(idx):
        Bs = 1j * _sym_columns(x, idx, Ks)
        if Kc:
            Bs = np.concatenate([Bs, _sym_columns(x, idx, Kc)], axis=1)
        return Bs

    B_tr = design(train)
    r = y[train] - x[train]
    A = np.concatenate([B_tr.real, B_tr.imag])
    b = np.concatenate([r.real, r.imag])
    u = np.linalg.lstsq(A, b, rcond=None)[0]
    y_te = x[test] + design(test) @ u
    ns = Ks // 2 + 1
    s = sym_taps(u[:ns])
    c = sym_taps(u[ns:]) if Kc else np.zeros(1)
    return s, c, nrmse(y_te, y[test])


def quant_coef(c: np.ndarray, bh: int) -> np.ndarray:
    """Coeficientes a bh bits con signo y escala potencia de 2 ajustada al máximo.

    Escala común para parte real e imaginaria: max|c| < 2^e con
    e = floor(log2(max(|Re|, |Im|))) + 1, formato ac_fixed<bh, e + 1> (paso
    2^(e + 1 - bh)), redondeo a par y saturación.
    """
    c = np.asarray(c)
    m = max(np.max(np.abs(c.real)), np.max(np.abs(np.imag(c))))
    e = int(np.floor(np.log2(m))) + 1
    return quantize(c, bh, e + 1)


def conv_centered(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """y[n] = Σ_k h[k] x[n - (k - K//2)], mismo largo que x (K impar)."""
    K = len(h)
    return np.convolve(x, h)[K // 2 : K // 2 + len(x)]


def sqnr_db(y_fx, y_fl) -> float:
    return float(10 * np.log10(np.sum(np.abs(y_fl) ** 2) / np.sum(np.abs(y_fx - y_fl) ** 2)))


def t4_fir(xq, h, bx, bh, sl):
    """FIR completo: golden (h flotante sobre xq) contra corrección (h - δ) en precisión reducida."""
    y_fl = conv_centered(xq, h)
    c = h.copy()
    c[len(h) // 2] -= 1.0
    xb = quantize(xq, bx, 1)
    y_fx = quantize(xq + conv_centered(xb, quant_coef(c, bh)), *Q115)
    return sqnr_db(y_fx[sl], y_fl[sl])


def t4_first_order(xq, s, bx, bh, sl):
    """Primer orden: golden xq + j(s * xq) contra xq + j(s_q * x_b) redondeado a Q1.15."""
    y_fl = xq + 1j * conv_centered(xq, s)
    xb = quantize(xq, bx, 1)
    y_fx = quantize(xq + 1j * conv_centered(xb, quant_coef(s, bh)), *Q115)
    return sqnr_db(y_fx[sl], y_fl[sl])


# --------------------------------------------------------------------------
# T5: ISI a 1,2 sps
# --------------------------------------------------------------------------
L_REF = 8192  # largo del prototipo sin truncar
D_REF = 2048  # pre-retardo del prototipo (muestras a fs)
PROFILES = (("gaussian", None), ("bessel", 2), ("bessel", 4), ("bessel", 6),
            ("butter", 2), ("butter", 4), ("butter", 6))
B_RELS = (0.3, 0.4, 0.5, 0.6)
K_MAX_SEARCH = 64
TOLS = (1e-2, 6e-3, 2e-3)
ENERGY_MIN = 0.985
LAG_MIN, LAG_MAX = -64, 128  # posiciones candidatas relativas a la ventana de 64 de más energía
BAND_EDGE = (1 + ESC.rolloff) * RS / 2  # 33 GHz


def profile_name(kind, m):
    return "gaussiano" if kind == "gaussian" else f"{'Bessel' if kind == 'bessel' else 'Butterworth'} m={m}"


def h_reference(kind: str, m: int | None, B_rel: float) -> np.ndarray:
    """Prototipo sin truncar (L_REF coeficientes, energía unitaria), con las funciones del golden."""
    B = B_rel * RS
    if kind == "gaussian":
        return gaussian(B, FS, K=L_REF, L=L_REF)
    return analog(kind, B, FS, m, K=L_REF, d=D_REF, L=L_REF)


def best_window(h: np.ndarray, K: int) -> tuple[int, float]:
    """Posición o de K coeficientes contiguos con más energía y fracción capturada."""
    e = np.r_[0.0, np.cumsum(np.abs(h) ** 2)]
    w = e[K:] - e[:-K]
    o = int(np.argmax(w))
    return o, float(w[o] / e[-1])


def apply_at(x: np.ndarray, h: np.ndarray, start: int, idx: np.ndarray) -> np.ndarray:
    """y[n] = Σ_k h[k] x[n - start - k] en los índices idx (x vale 0 fuera de rango)."""
    P = 256
    xp = np.r_[np.zeros(P, x.dtype), x, np.zeros(P, x.dtype)]
    full = np.convolve(xp, h)
    return full[idx - start + P]


class LagGram:
    """Matrices de Gram sobre la señal (para d) y sobre la banda (para c), compartidas.

    Señal: G[l1, l2] = Re Σ_n conj(x[n - l1]) x[n - l2] en los índices de ajuste.
    Banda: G[l1, l2] = Σ_f cos(2π f (l1 - l2) / fs), |f| <= BAND_EDGE en la grilla de L_REF.
    """

    def __init__(self, x: np.ndarray, train: np.ndarray):
        self.x, self.train = x, train
        self.lags = np.arange(LAG_MIN, LAG_MAX)
        G = np.zeros((len(self.lags),) * 2)
        for i in range(0, len(train), 8192):
            A = lag_matrix(x, train[i : i + 8192], self.lags)
            G += (A.conj().T @ A).real
        self.G_sig = G
        f = np.fft.fftfreq(L_REF, 1 / FS)
        self.band = np.abs(f) <= BAND_EDGE
        self.f_band = f[self.band]
        # Toeplitz: solo depende de l1 - l2
        r = np.cos(2 * np.pi * np.outer(np.arange(len(self.lags)), self.f_band) / FS).sum(-1)
        self.G_freq = scipy.linalg.toeplitz(r)

    def c_signal(self, y_al: np.ndarray) -> np.ndarray:
        c = np.zeros(len(self.lags))
        for i in range(0, len(self.train), 8192):
            idx = self.train[i : i + 8192]
            A = lag_matrix(self.x, idx, self.lags)
            c += (A.conj().T @ y_al[idx]).real
        return c

    def c_freq(self, T_band: np.ndarray) -> np.ndarray:
        a = np.exp(-2j * np.pi * self.f_band[None, :] * self.lags[:, None] / FS)
        return (a.conj() @ T_band).real


def best_ls(G, c, E, K):
    """Mejor posición de K coeficientes por mínimos cuadrados: (start, h, error relativo)."""
    best = None
    for s in range(0, G.shape[0] - K + 1):
        sl = slice(s, s + K)
        try:
            h = scipy.linalg.solve(G[sl, sl], c[sl], assume_a="pos")
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            h = np.linalg.lstsq(G[sl, sl], c[sl], rcond=None)[0]
        err = (E - c[sl] @ h) / E
        if best is None or err < best[2]:
            best = (s, h, err)
    return best


def isi_case(x, kind, m, B_rel, train, test, gram: LagGram, Ks=range(1, K_MAX_SEARCH + 1)):
    """Los cuatro criterios para un perfil y un B: dict con curvas por K y los h elegidos."""
    h = h_reference(kind, m, B_rel)
    o64, _ = best_window(h, 64)
    y_full = fftconvolve(x, h)
    # ỹ[n] = y[n + o64]: un coeficiente en la posición o64 + l multiplica a x[n - l]
    y_al = np.zeros(len(x), complex)
    n_ok = min(len(x), len(y_full) - o64)
    y_al[:n_ok] = y_full[o64 : o64 + n_ok]
    # banda: T(f) = H_ref(f)·exp(+j2πf·o64/fs)
    f = np.fft.fftfreq(L_REF, 1 / FS)
    T = np.fft.fft(h) * np.exp(2j * np.pi * f * o64 / FS)
    T_band = T[gram.band]
    c_sig, E_sig = gram.c_signal(y_al), float(np.sum(np.abs(y_al[train]) ** 2))
    c_frq, E_frq = gram.c_freq(T_band), float(np.sum(np.abs(T_band) ** 2))
    ref_te = y_al[test]
    res = {"energy": {}, "trunc": {}, "lsf": {}, "lsd": {}, "h": {}, "o64": o64}
    for K in Ks:
        # (a) energía en la mejor ventana
        o, frac = best_window(h, K)
        res["energy"][K] = frac
        # (b) truncado en esa ventana, solo ganancia (ajustada en la primera mitad)
        ht = h[o : o + K]
        s_t = o - o64
        y_tr = apply_at(x, ht, s_t, train)
        g = np.real(np.vdot(y_tr, y_al[train])) / np.real(np.vdot(y_tr, y_tr))
        res["trunc"][K] = nrmse(g * apply_at(x, ht, s_t, test), ref_te)
        # (c) mínimos cuadrados en frecuencia (banda de la señal), mejor retardo entero
        s_f, h_f, _ = best_ls(gram.G_freq, c_frq, E_frq, K)
        res["lsf"][K] = nrmse(apply_at(x, h_f, int(gram.lags[s_f]), test), ref_te)
        # (d) mínimos cuadrados sobre la señal (primera mitad), mejor retardo entero
        s_d, h_d, _ = best_ls(gram.G_sig, c_sig, E_sig, K)
        res["lsd"][K] = nrmse(apply_at(x, h_d, int(gram.lags[s_d]), test), ref_te)
        res["h"][K] = {"trunc": (g * ht, s_t), "lsf": (h_f, int(gram.lags[s_f])), "lsd": (h_d, int(gram.lags[s_d]))}
    res["h_ref"] = h
    return res


def min_K(curve: dict, ok) -> int | None:
    """Menor K de la curva que cumple ``ok(valor)``; None si ninguno."""
    for K in sorted(curve):
        if ok(curve[K]):
            return K
    return None


# --------------------------------------------------------------------------
# T7: DSP por fórmula
# --------------------------------------------------------------------------
DSP_TOTAL = 4272


def _vp(n: int, p: int) -> int:
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


def ffa_cost(K: int, P: int) -> dict:
    """Menor costo K_p·(3/4)^a·(2/3)^b sobre (a, b) con 2^a | P y 3^b | P."""
    best = None
    for a in range(_vp(P, 2) + 1):
        for b in range(_vp(P, 3) + 1):
            u = 2**a * 3**b
            Kp = -(-K // u) * u
            cost = Kp * 0.75**a * (2 / 3) ** b
            if best is None or cost < best["mult_per_sample"] - 1e-12:
                best = {"a": a, "b": b, "K_p": Kp, "mult_per_sample": cost}
    return best


# --------------------------------------------------------------------------
# Guardado
# --------------------------------------------------------------------------
def save_fig(fig, name: str) -> None:
    (OUT / "fig").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig" / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig" / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})


def _fmt(v, digits=4):
    if isinstance(v, float):
        if np.isinf(v):
            return "inf"
        return f"{v:.{digits}g}"
    return "" if v is None else str(v)


def save_csv(rows: list[dict], name: str) -> Path:
    """Guarda una lista de dicts (mismas claves) como CSV en OUT/csv/<name>.csv."""
    (OUT / "csv").mkdir(parents=True, exist_ok=True)
    path = OUT / "csv" / f"{name}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v, 6) for k, v in r.items()})
    return path


def md_table(rows: list[dict], cols: list[str] | None = None, digits: int = 4) -> str:
    """Tabla markdown de una lista de dicts (para mostrar con IPython.display.Markdown)."""
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c), digits) for c in cols) + " |")
    return "\n".join(out)


def git_info() -> dict:
    root = Path(__file__).resolve().parents[1]

    def run(*a):
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": run("status", "--porcelain", "--", ".").splitlines(),
    }


def save_meta(meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float))
