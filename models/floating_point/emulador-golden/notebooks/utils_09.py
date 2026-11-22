"""Utilidades del notebook 09_evidencias_cd_isi_72gsps (tarea D2.1).

Evidencias de CD e ISI para el escenario de 72 GS/s (60 GBd, 1,2 sps) con el
rango de distancias nuevo (CD acumulada 0 a 200 ps/nm, opcional hasta 800).

NO es parte del núcleo del golden. Reutiliza ``utils_08`` (señal común, ajustes
por mínimos cuadrados, prototipos de la ISI, FFA) y agrega solo lo que falta:

  - overlap-save con bloque L y solapamiento V cualquiera (``os_hbar``,
    ``os_apply``): el del núcleo fija V = L/2;
  - CD parametrizada por CD acumulada D·z;
  - FIR complejo por ecuaciones normales (rápido hasta K = 81);
  - respuesta efectiva del overlap-save por posición dentro del bloque;
  - señal común con desfasajes y largo configurables (corridas de E5);
  - diseño (e) de la ISI para un K dado.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.signal import fftconvolve

import utils_08 as u8
from emulador.cd import cd_R, cd_reference, cd_transfer
from emulador.common import rrc_taps
from emulador.config import D_si, S_si
from emulador.metrics import nrmse
from emulador.quantization import quantize
from emulador.stimuli import prbs

ESC, FS, RS, LAM = u8.ESC, u8.FS, u8.RS, u8.LAM
Q115 = u8.Q115
D_PS = u8.D_PS  # 17 ps/nm/km
S_ALT = 0.057  # ps/nm²/km

CD_GRID = (0, 5, 10, 17, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800)  # ps/nm
CD_S_REPEAT = (200, 800)
CD_NOMINAL = 200
CD_OPTIONAL = 800
OS_PAIRS = ((128, 32), (256, 64), (512, 128))
OS_PAIRS_E7 = ((256, 64), (512, 128), (1024, 256), (2048, 512))
TOLS_CD = (1e-2, 1e-3)
K_B_MAX = 81
PS = (48, 96, 192)
DSP_TOTAL = u8.DSP_TOTAL

OUT = Path(__file__).resolve().parents[1] / "docs" / "golden_model" / "resultados" / "09_evidencias_cd_isi"


# --------------------------------------------------------------------------
# CD por CD acumulada
# --------------------------------------------------------------------------
def z_m(cd_ps_nm: float, D_ps: float = D_PS) -> float:
    """Distancia en m para una CD acumulada D·z (ps/nm) con D en ps/nm/km."""
    return cd_ps_nm / D_ps * 1e3


def cd_ref(x: np.ndarray, cd_ps_nm: float, S_ps: float = 0.0, D_ps: float = D_PS, lam: float = LAM) -> np.ndarray:
    """``cd_reference`` (FFT de toda la señal) para una CD acumulada dada."""
    return cd_reference(x, FS, D_si(D_ps), lam, z_m(cd_ps_nm, D_ps), S_si(S_ps))


def cd_R_of(cd_ps_nm: float, D_ps: float = D_PS) -> int:
    return cd_R(FS, D_si(D_ps), LAM, z_m(cd_ps_nm, D_ps))


# --------------------------------------------------------------------------
# Overlap-save con L y V cualesquiera
# --------------------------------------------------------------------------
def os_hbar(L: int, V: int, cd_ps_nm: float, S_ps: float = 0.0, D_ps: float = D_PS, lam: float = LAM) -> np.ndarray:
    """H̄[k] = H_CD(f_k)·exp(-j2πk(V/2)/L), FFT de L puntos (orden de ``np.fft``).

    Con V = L/2 es ``cd_hbar(L/2, ...)`` del núcleo.
    """
    if V % 2 or not 0 < V < L:
        raise ValueError(f"os_hbar: V tiene que ser par y 0 < V < L: L={L}, V={V}")
    k = np.fft.fftfreq(L) * L
    H = cd_transfer(L, FS, D_si(D_ps), lam, z_m(cd_ps_nm, D_ps), S_si(S_ps))
    return H * np.exp(-2j * np.pi * k * (V / 2) / L)


def os_apply(x: np.ndarray, Hbar: np.ndarray, V: int) -> np.ndarray:
    """Overlap-save: tramas de L = len(Hbar) con V muestras de la trama anterior y L - V nuevas.

    FFT -> × H̄ -> IFFT -> se descartan las primeras V muestras. La primera trama
    va precedida por V ceros. Salida de largo nb·(L - V), nb = ⌈n/(L - V)⌉;
    y[n] corresponde a x[n - V/2]. Opera sobre el último eje.
    """
    x = np.asarray(x, dtype=complex)
    L = Hbar.shape[-1]
    M = L - V
    n = x.shape[-1]
    nb = -(-n // M)
    widths = [(0, 0)] * (x.ndim - 1) + [(V, nb * M - n)]
    xz = np.pad(x, widths)
    frames = xz[..., np.arange(nb)[:, None] * M + np.arange(L)]
    y = np.fft.ifft(np.fft.fft(frames, axis=-1) * Hbar, axis=-1)[..., V:]
    return y.reshape(x.shape[:-1] + (nb * M,))


def os_aligned(x: np.ndarray, L: int, V: int, cd_ps_nm: float, S_ps: float = 0.0) -> np.ndarray:
    """Overlap-save sin el retardo V/2: devuelve (..., n) alineada con x."""
    x = np.asarray(x)
    n = x.shape[-1]
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, V // 2)])  # para que salgan las últimas V/2
    y = os_apply(xp, os_hbar(L, V, cd_ps_nm, S_ps), V)
    return y[..., V // 2 : V // 2 + n]


def os_effective_response(L: int, V: int, cd_ps_nm: float, S_ps: float = 0.0, over: int = 16):
    """Respuesta efectiva del overlap-save para cada posición de salida del bloque.

    La salida en la posición i (V … L-1) de la trama es un FIR de L coeficientes
    de la extensión periódica de h̄ = IFFT(H̄), con lags i-L+1 … i. Devuelve
    (f, G, Hid): f (Lf,) en Hz, G (L-V, Lf) con la respuesta de cada posición y
    Hid (Lf,) = H_CD(f)·exp(-j2πf(V/2)/fs) en la grilla fina de Lf = over·L.
    """
    Hb = os_hbar(L, V, cd_ps_nm, S_ps)
    hb = np.fft.ifft(Hb)
    Lf = over * L
    f = np.fft.fftfreq(Lf, 1 / FS)
    G = np.empty((L - V, Lf), complex)
    for j, i in enumerate(range(V, L)):
        start = i - L + 1  # lag del primer coeficiente
        seq = np.roll(hb, -start)  # seq[n] = hb[(n + start) mod L]
        G[j] = np.fft.fft(seq, Lf) * np.exp(-2j * np.pi * f * start / FS)
    Hid = cd_transfer(Lf, FS, D_si(D_PS), LAM, z_m(cd_ps_nm), S_si(S_ps)) * np.exp(-2j * np.pi * f * (V / 2) / FS)
    return f, G, Hid


def min_L_pow2(R: int) -> int:
    """Menor L potencia de 2 con V = L/4 >= R - 1 (referencia, sin simular)."""
    L = 4
    while L // 4 < R - 1:
        L *= 2
    return L


# --------------------------------------------------------------------------
# FIR complejo por mínimos cuadrados (ecuaciones normales)
# --------------------------------------------------------------------------
def fir_curve(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, K_max: int = K_B_MAX) -> dict:
    """NRMSE (segunda mitad) del FIR complejo centrado de K impar = 1 … K_max.

    Mismo modelo que ``utils_08.fit_fir_complex`` (h[k] multiplica a
    x[n - (k - K//2)]), resuelto con la matriz de Gram para que sea rápido.
    """
    M = K_max // 2
    lags = np.arange(-M, M + 1)
    G = np.zeros((len(lags),) * 2, complex)
    c = np.zeros(len(lags), complex)
    for i in range(0, len(train), 8192):
        idx = train[i : i + 8192]
        A = u8.lag_matrix(x, idx, lags)
        G += A.conj().T @ A
        c += A.conj().T @ y[idx]
    A_te = u8.lag_matrix(x, test, lags)
    out = {}
    for K in range(1, K_max + 1, 2):
        sl = slice(M - K // 2, M + K // 2 + 1)
        h = scipy.linalg.solve(G[sl, sl], c[sl], assume_a="her")
        out[K] = nrmse(A_te[:, sl] @ h, y[test])
    return out


def min_K_tol(curve: dict, tol: float):
    for K in sorted(curve):
        if curve[K] <= tol:
            return K
    return None


# --------------------------------------------------------------------------
# Señal común con desfasajes y largo configurables (E5)
# --------------------------------------------------------------------------
def signal_run(shift_bits: int, n_sym: int = u8.N_SYM, rms: float = u8.RMS_TARGET) -> dict:
    """Señal común del 08 con los 4 desfasajes corridos ``shift_bits`` bits (mod P).

    Con shift_bits = 0 y n_sym = N_SYM es ``utils_08.common_signal()``.
    El factor común se recalcula para que el RMS medio sea ``rms``.
    """
    p, q = ESC.sps_num, ESC.sps_den
    g = rrc_taps(ESC.rolloff, u8.SPAN_TX, p)
    b = prbs(u8.PRBS_ORDER, u8.PERIOD + u8.PRBS_ORDER, seed=u8.BASE_SEED)
    offs = [(o + shift_bits) % u8.PERIOD for o in u8.OFFSETS]
    seeds = [int("".join(map(str, b[o : o + u8.PRBS_ORDER])), 2) for o in offs]
    chans, levels = [], []
    for s in seeds:
        bits = prbs(u8.PRBS_ORDER, 2 * n_sym, seed=s).reshape(-1, 2)
        lv = u8.GRAY[2 * bits[:, 0] + bits[:, 1]] / 3.0
        up = np.zeros(n_sym * p)
        up[::p] = lv
        chans.append(np.convolve(up, g, mode="same")[::q])
        levels.append(lv)
    chans = np.array(chans)
    scale = rms / np.sqrt(np.mean(chans**2, axis=1)).mean()
    return {"channels": chans * scale, "levels": np.array(levels), "offsets": offs, "seeds": seeds,
            "scale": float(scale)}


def quant_only_sqnr_db(v: np.ndarray) -> float:
    """SQNR del redondeo a Q1.15 sin saturar (solo el paso de cuantización)."""
    W, I = Q115
    k = np.round(v * 2.0 ** (W - I)) * 2.0 ** -(W - I)
    return float(10 * np.log10(np.sum(v**2) / np.sum((k - v) ** 2)))


def stats4(chans, sl: slice) -> dict:
    """Peor caso de los 4 canales reales (``utils_08.worst_case``) en la ventana ``sl``."""
    return u8.worst_case([u8.range_stats(np.asarray(v)[sl]) for v in chans])


# --------------------------------------------------------------------------
# ISI: diseño (e) para un K
# --------------------------------------------------------------------------
def design_e(gram: "u8.LagGram", G_rc: np.ndarray, W_rc: np.ndarray, kind, m, B_rel, K, x, test):
    """Diseño (e) con K coeficientes: (h, start, nrmse_test, h_ref, o64).

    Mismo cálculo que ``utils_08.isi_case`` para el criterio (e), solo para un K.
    """
    h_ref = u8.h_reference(kind, m, B_rel)
    o64, _ = u8.best_window(h_ref, 64)
    f = np.fft.fftfreq(u8.L_REF, 1 / FS)
    T_band = (np.fft.fft(h_ref) * np.exp(2j * np.pi * f * o64 / FS))[gram.band]
    cw = gram.c_freq(T_band, W_rc)
    Ew = float(np.sum(W_rc * np.abs(T_band) ** 2))
    s, h, _ = u8.best_ls(G_rc, cw, Ew, K)
    start = int(gram.lags[s])
    y_full = fftconvolve(x, h_ref)
    y_al = np.zeros(len(x), complex)
    n_ok = min(len(x), len(y_full) - o64)
    y_al[:n_ok] = y_full[o64 : o64 + n_ok]
    err = nrmse(u8.apply_at(x, h, start, test), y_al[test])
    return h, start, err, h_ref, o64


def apply_real_fir(chans: np.ndarray, h: np.ndarray, start: int) -> np.ndarray:
    """y[n] = Σ_k h[k] x[n - start - k] sobre cada canal real (mismo largo)."""
    out = []
    for v in chans:
        full = np.convolve(np.r_[np.zeros(256), v, np.zeros(256)], h)
        idx = np.arange(len(v))
        out.append(full[idx - start + 256])
    return np.array(out)


# --------------------------------------------------------------------------
# Guardado (misma forma que utils_08, carpeta de D2.1)
# --------------------------------------------------------------------------
def save_fig(fig, name: str) -> None:
    (OUT / "fig").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig" / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig" / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})


def save_csv(rows: list[dict], name: str) -> Path:
    (OUT / "csv").mkdir(parents=True, exist_ok=True)
    path = OUT / "csv" / f"{name}.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: u8._fmt(r.get(k), 6) for k in keys})
    return path


def read_csv_08(name: str) -> list[dict]:
    with open(u8.OUT / "csv" / f"{name}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def save_meta(meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float))
