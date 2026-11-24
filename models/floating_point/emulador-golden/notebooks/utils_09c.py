"""Utilidades de E13 del notebook 09 (tarea D2.8): descomposiciones radix-2^k de la FFT de L = 128.

NO es parte del núcleo ni lo modifica. Reutiliza la aritmética de
``emulador.fixed_point.cd_fft`` (mariposa, rotación y tablas) y el modelo de
recursos de ``utils_09b`` (mismos supuestos y mismos factores de calibración).

Modelo de una descomposición (``groups``): la FFT DIF de L = 2^n puntos son n
etapas de mariposas radix-2 en orden de distancia d = L/2, L/4, …, 1 (la etapa t
consume el bit n_{n−1−t} de la entrada y produce el bit k_t de la salida, en el
lugar), con una rotación W_L^e después de cada etapa ("frontera" t). La fase de
la DFT es n·k = Σ n_i·k_l·2^(i+l) (mód. L); cada término con i + l ≤ n − 2 se
aplica en alguna frontera t con l ≤ t < n − 1 − i. La descomposición decide en
cuál:

  - ``groups`` = tamaños de los grupos de etapas (radix-2^b por grupo), p. ej.
    (1, 2, 2, 2) es la radix-2² vigente (R2 + tres pares) y (3, 3, 1) la 2³·2³·2;
  - término entre grupos: al final del grupo de l (Cooley-Tukey: el twiddle
    "grande" va entre grupos);
  - término dentro de un grupo: el grupo se parte como la radix-2² del núcleo
    (R2 primero si b es impar, después pares) y se aplica la misma regla; dentro
    de un par, en la frontera del medio (×(−j) de BF2II). Así, 2³ = 2·2² (W8 en
    la primera frontera) y 2⁴ = 2²·2² (W16 en la frontera del medio).

Todas las variantes son mariposas radix-2 en el mismo orden de bits, así que la
salida de la FFT queda en orden bit invertido y la IFFT DIT (las mismas fronteras
al revés, con el twiddle conjugado) entra en bit invertido y sale en orden
natural: todas son compatibles con H̄ permutado. Con (1, 2, 2, 2) el resultado es
bit a bit igual al ``_transform`` del núcleo.

Plegado en P_FFT carriles como ``utils_09b.lane_rotators``: la posición
j = i + P_FFT·c va al carril i y al ciclo c. Clases por carril y frontera
(criterio de la tarea: un solo valor no trivial, aunque alterne con 1 o con un
trivial, es "constante"):

  none      : todos 1;
  triv      : un solo trivial (±1, ±j) constante, sin costo;
  triv_mux  : alterna entre triviales: multiplexor;
  w8        : un solo W8 (múltiplo impar de π/4) constante: rotador W8 en LUT;
  w8_mux    : un solo W8 que alterna con 1 o triviales: W8 + multiplexor;
  gen       : un solo valor general constante: 3 DSP, coeficiente fijo;
  gen_mux   : un solo valor general que alterna con 1 o triviales: 3 DSP +
              multiplexor del coeficiente (2·W_c LUT, como el "variable" de E9);
  var_w8    : más de un valor no trivial, todos W8: W8 + multiplexor;
  var       : más de un valor no trivial con alguno general: 3 DSP + tabla de
              coeficientes (2·W_c LUT).

Los generales que son múltiplos impares de π/8 (W16) se marcan aparte: son los
que se podrían llevar a LUT con una pieza que no está caracterizada.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import utils_09b as ub
from emulador.fixed_point.cd_fft import CdFixedConfig, _FloatArith, _IntArith, hbar_permuted, twiddle_table
from emulador.fixed_point.integer import round_shift, saturate_int
from emulador.quantization import to_int

L13, V13, P13 = 128, 32, 48
PF13 = 64  # 128 puntos en 2 ciclos con P = 48

# nombre -> grupos de etapas (orden de la FFT DIF)
DECOMPS = {
    "radix-2": (1, 1, 1, 1, 1, 1, 1),
    "radix-2² (vigente)": (1, 2, 2, 2),
    "radix-2³ (2³·2³·2)": (3, 3, 1),
    "radix-2³ (2·2³·2³)": (1, 3, 3),
    "radix-2⁴ (2⁴·2³)": (4, 3),
    "radix-2⁴ (2³·2⁴)": (3, 4),
}
BASE13 = "radix-2² (vigente)"

ROT_KINDS = ("none", "triv", "triv_mux", "w8", "w8_mux", "gen", "gen_mux", "var_w8", "var")


# --------------------------------------------------------------------------
# Fronteras y exponentes
# --------------------------------------------------------------------------
def _log2(L: int) -> int:
    n = int(L).bit_length() - 1
    assert L >= 2 and (1 << n) == L, L
    return n


def _spans(start: int, sizes) -> list[tuple[int, int]]:
    out, s = [], start
    for b in sizes:
        out.append((s, b))
        s += b
    return out


def _sub_split(b: int) -> tuple[int, ...]:
    """Partición radix-2² de un grupo de b etapas: R2 primero si b es impar, después pares."""
    return ((1,) if b % 2 else ()) + (2,) * (b // 2)


def _which(spans, x: int) -> tuple[int, int]:
    return next((s, b) for s, b in spans if s <= x < s + b)


def term_boundary(i: int, l: int, n: int, groups) -> int:
    """Frontera t en la que se aplica el término n_i·k_l·2^(i+l) (i + l ≤ n − 2)."""
    s = n - 1 - i  # etapa que consume n_i
    assert l < s, (i, l)
    spans = _spans(0, groups)
    ga, gb = _which(spans, l), _which(spans, s)
    if ga != gb:
        return ga[0] + ga[1] - 1
    sub = _spans(ga[0], _sub_split(ga[1]))
    sa, sb = _which(sub, l), _which(sub, s)
    if sa != sb:
        return sa[0] + sa[1] - 1
    return l  # dentro de un par: la frontera del medio


def boundary_exponents(L: int, groups) -> np.ndarray:
    """Exponentes e[t, j] (mód. L) de la rotación W_L^e después de la etapa t, por posición j."""
    n = _log2(L)
    assert sum(groups) == n, (groups, n)
    j = np.arange(L)
    bit = lambda b: (j >> b) & 1
    E = np.zeros((n, L), dtype=np.int64)
    for l in range(n):
        for i in range(n - 1 - l):  # i + l ≤ n − 2
            t = term_boundary(i, l, n, groups)
            E[t] += bit(i) * bit(n - 1 - l) << (i + l)  # n_i en el bit i; k_l en el bit n − 1 − l
    return E % L


def bitrev(L: int) -> np.ndarray:
    n = _log2(L)
    j = np.arange(L)
    return np.array([int(f"{v:0{n}b}"[::-1], 2) for v in j])


# --------------------------------------------------------------------------
# Transformadas (flotante y entera)
# --------------------------------------------------------------------------
def _transform(z, L, ar, groups, s, inverse):
    n = _log2(L)
    E = boundary_exponents(L, groups)
    ts = range(n - 1, -1, -1) if inverse else range(n)
    for t in ts:
        d = L >> (t + 1)
        if not inverse:
            z = ar.bfly(z, d, s[t])
            z = ar.rotate(z, E[t], L, conj=False)
        else:
            z = ar.rotate(z, E[t], L, conj=True)
            z = ar.bfly(z, d, s[t])
    return z


def fft_dif(x, groups, s=None):
    """FFT DIF en flotante con la descomposición ``groups``; salida en orden bit invertido."""
    x = np.asarray(x, complex)
    L = x.shape[-1]
    return _transform(x, L, _FloatArith(), groups, s or [0] * _log2(L), inverse=False)


def ifft_dit(X, groups, s=None):
    """IFFT DIT en flotante: entrada en bit invertido, salida en orden natural, sin 1/L."""
    X = np.asarray(X, complex)
    L = X.shape[-1]
    return _transform(X, L, _FloatArith(), groups, s or [0] * _log2(L), inverse=True)


def check_float(groups, L: int = L13, trials: int = 8, seed: int = 0) -> dict:
    """Error relativo máx. contra np.fft.fft en bit invertido y de FFT → IFFT contra la entrada."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((trials, L)) + 1j * rng.standard_normal((trials, L))
    br = bitrev(L)
    Xr = np.fft.fft(x)[:, br]
    X = fft_dif(x, groups)
    e_fft = float(np.max(np.linalg.norm(X - Xr, axis=-1) / np.linalg.norm(Xr, axis=-1)))
    y = ifft_dit(X, groups) / L
    e_rt = float(np.max(np.linalg.norm(y - x, axis=-1) / np.linalg.norm(x, axis=-1)))
    # IFFT sola contra numpy (entrada en bit invertido)
    yi = ifft_dit(Xr, groups) / L
    e_ifft = float(np.max(np.linalg.norm(yi - x, axis=-1) / np.linalg.norm(x, axis=-1)))
    return {"FFT contra np.fft.fft (bit inv.)": e_fft, "IFFT contra np.fft (bit inv.)": e_ifft, "FFT → IFFT": e_rt}


def os_fixed_decomp(x, Hbar, V, cfg: CdFixedConfig, groups):
    """``utils_09b.os_fixed`` con la FFT/IFFT de la descomposición ``groups`` (misma aritmética y escalado)."""
    x = np.asarray(x, complex)
    L = Hbar.size
    M = L - V
    s_f, s_i = cfg.shifts(L)
    F_in, F_d, F_out = cfg.W_in - 1, cfg.W_d - 2, cfg.W_out - cfg.I_out
    xr, xi = to_int(x, cfg.W_in, 1, rounding=cfg.in_rounding)
    n = x.size
    nb = -(-n // M)
    idx = np.arange(nb)[:, None] * M + np.arange(L)
    fr = np.pad(xr, (V, nb * M - n))[idx]
    fi = np.pad(xi, (V, nb * M - n))[idx]
    ar = _IntArith(cfg)
    z = (fr << (F_d - F_in), fi << (F_d - F_in))
    z = _transform(z, L, ar, groups, s_f, inverse=False)
    z = ar.mul_table(z, hbar_permuted(Hbar, cfg.W_h, rounding=cfg.table_rounding))
    z = _transform(z, L, ar, groups, s_i, inverse=True)
    yr, yi = z[0][:, V:].reshape(-1), z[1][:, V:].reshape(-1)
    y_int = (yr + 1j * yi) * 2.0**-F_d
    orr, n1 = saturate_int(round_shift(yr, F_d - F_out, cfg.out_rounding), cfg.W_out)
    oi, n2 = saturate_int(round_shift(yi, F_d - F_out, cfg.out_rounding), cfg.W_out)
    return (orr + 1j * oi) * 2.0**-F_out, y_int, ar.saturations + n1 + n2


# --------------------------------------------------------------------------
# Clases de rotadores
# --------------------------------------------------------------------------
def _vclass(e: int, L: int) -> str:
    return ub._rot_class(int(e), L)[0]  # one, triv, w8, gen


def is_w16(e: int, L: int) -> bool:
    """General que es múltiplo impar de π/8."""
    return _vclass(e, L) == "gen" and (e * 16) % L == 0


def lane_class(vals, L: int) -> str:
    cls = [_vclass(v, L) for v in vals]
    nt = {v for v, c in zip(vals, cls) if c in ("w8", "gen")}
    alt = len(set(vals)) > 1
    if not nt:
        if all(c == "one" for c in cls):
            return "none"
        return "triv_mux" if alt else "triv"
    if len(nt) == 1:
        k = _vclass(next(iter(nt)), L)
        return k + ("_mux" if alt else "")
    return "var" if any(_vclass(v, L) == "gen" for v in nt) else "var_w8"


def lane_table(L: int, Pf: int, groups) -> list[dict]:
    """Por frontera y carril: clase y valores (exponentes de W_L) en los C = L/Pf ciclos."""
    E = boundary_exponents(L, groups)
    C = L // Pf
    out = []
    for t in range(E.shape[0]):
        for i in range(Pf):
            vals = tuple(int(E[t, i + Pf * c]) for c in range(C))
            k = lane_class(vals, L)
            w16 = k in ("gen", "gen_mux") and any(is_w16(v, L) for v in vals)
            out.append({"frontera": t, "carril": i, "clase": k, "valores": vals, "w16": w16})
    return out


def rot_summary(L: int, Pf: int, groups) -> dict:
    """Conteo por transformada (carriles): clases, agregados de la tarea y fronteras que redondean."""
    tab = lane_table(L, Pf, groups)
    c = {k: 0 for k in ROT_KINDS}
    for r in tab:
        c[r["clase"]] += 1
    c["W16 (generales)"] = sum(r["w16"] for r in tab)
    c["triviales"] = c["triv"] + c["triv_mux"]
    c["W8"] = c["w8"] + c["w8_mux"] + c["var_w8"]
    c["constantes generales"] = c["gen"] + c["gen_mux"]
    c["variables"] = c["var"] + c["var_w8"]
    c["variables (con general)"] = c["var"]
    c["con multiplexor"] = c["triv_mux"] + c["w8_mux"] + c["gen_mux"] + c["var_w8"] + c["var"]
    c["rotadores en DSP"] = c["gen"] + c["gen_mux"] + c["var"]
    E = boundary_exponents(L, groups)
    nontriv = lambda t: any(_vclass(e, L) in ("w8", "gen") for e in E[t])
    c["fronteras no triviales"] = sum(nontriv(t) for t in range(E.shape[0]))
    c["fronteras con general"] = sum(any(_vclass(e, L) == "gen" for e in E[t]) for t in range(E.shape[0]))
    return c


def op_counts(L: int, groups) -> dict:
    """Operaciones por transformada de L puntos sin plegar (una por posición rotada)."""
    E = boundary_exponents(L, groups)
    cls = [_vclass(e, L) for e in E.ravel()]
    return {"triviales": cls.count("triv"), "W8": cls.count("w8"), "generales": cls.count("gen"),
            "W16": int(sum(is_w16(e, L) for e in E.ravel()))}


def split_radix_ops(N: int) -> dict:
    """Twiddles de la split-radix DIF de N puntos, sin plegar (referencia de conteo, sin implementar).

    SR(N): mariposa en L, W_N^m y W_N^(3m) para m = 0 … N/4 − 1, después SR(N/2) y dos SR(N/4).
    """
    tot = {"triviales": 0, "W8": 0, "generales": 0, "W16": 0}

    def rec(M: int):
        if M < 4:
            return
        for m in range(M // 4):
            for e in (m, 3 * m):
                k = _vclass(e, M)
                if k == "triv":
                    tot["triviales"] += 1
                elif k == "w8":
                    tot["W8"] += 1
                elif k == "gen":
                    tot["generales"] += 1
                    tot["W16"] += int((e * 16) % M == 0)
        rec(M // 2)
        rec(M // 4)
        rec(M // 4)

    rec(N)
    return tot


# --------------------------------------------------------------------------
# Recursos (mismo modelo que utils_09b.cd_count con gen_dsp=True)
# --------------------------------------------------------------------------
def cd_count_decomp(P: int, W_d: int, W_c: int, groups, L: int = L13, V: int = V13, w16_lut: bool = False) -> ub.CdCount:
    """Conteo de la CD (overlap-save, 2 polarizaciones) con la descomposición ``groups``.

    Igual que ``utils_09b.cd_count(..., gen_dsp=True)`` salvo los rotadores, que salen de
    ``lane_table``. w16_lut: los generales W16 constantes en LUT (CSD sin compartir; pieza
    sin caracterizar), en lugar de DSP.
    """
    base = ub.cd_count(P, W_d, W_c, gen_dsp=True, L=L, V=V)
    Pf = L // ((L - V) // P)
    nT = 2 * ub.N_POL
    wr, wi = twiddle_table(W_c)
    Wp = W_d + W_c - 1
    a8 = max(len(ub.csd(int(wr[512]))) - 1, 0)
    w8_lut = 2 * (W_d + 1) + 2 * a8 * Wp
    tab = lane_table(L, Pf, groups)
    acc = {k: {"LUT": 0, "FF": 0, "DSP": 0, "n": 0} for k in
           ("gen", "gen_mux", "var", "w8", "w8_mux", "var_w8", "triv_mux", "w16_lut")}
    for r in tab:
        k = r["clase"]
        if k == "gen" and w16_lut and r["w16"]:
            (e,) = set(r["valores"])
            m = e * (4096 // L)
            b, _ = ub.const_mult_adders(int(wr[m]), int(wi[m]))
            a = acc["w16_lut"]
            a["LUT"] += 2 * b * Wp + 2 * Wp
            a["FF"] += 2 * W_d
        elif k == "gen":
            acc[k]["DSP"] += 3
        elif k in ("gen_mux", "var"):
            acc[k]["DSP"] += 3
            acc[k]["LUT"] += 2 * W_c
        elif k == "w8":
            acc[k]["LUT"] += w8_lut
            acc[k]["FF"] += 2 * W_d
        elif k in ("w8_mux", "var_w8"):
            acc[k]["LUT"] += w8_lut + 2 * W_d
            acc[k]["FF"] += 2 * W_d
        elif k == "triv_mux":
            acc[k]["LUT"] += 2 * W_d
        else:
            continue
        if k in acc:
            acc[k]["n"] += 1
    names = {
        "gen": "rotadores constantes generales (en DSP)",
        "gen_mux": "rotadores generales con mux contra 1 o trivial (3 DSP + mux de coeficiente)",
        "var": "rotadores variables (3 DSP)",
        "w8": "rotadores W8",
        "w8_mux": "W8 alternantes (W8 + mux de signo)",
        "var_w8": "W8 variables (W8 + mux)",
        "triv_mux": "triviales alternantes (mux)",
        "w16_lut": "rotadores constantes generales",  # en LUT: se calibran con K["rot"]
    }
    rows = []
    for r in base.rows:
        c = r["componente"]
        if c.startswith(("rotadores", "W8 alternantes", "triviales alternantes", "×(−j) alternantes")):
            continue
        rows.append(r)
    for k, name in names.items():
        a = acc[k]
        if k == "w16_lut" and not w16_lut:
            continue
        rows.append({"componente": name, "LUT": nT * a["LUT"], "FF": nT * a["FF"], "DSP": nT * a["DSP"],
                     "nota": f"{a['n']} por transformada" + (" (W16 en LUT, CSD sin compartir; sin caracterizar)" if k == "w16_lut" else "")})
    # latencia: 1 por mariposa; 4 por frontera con DSP; 1 por frontera con W8 en LUT
    n = _log2(L)
    lat_st = n
    for t in range(n):
        ks = {r["clase"] for r in tab if r["frontera"] == t}
        lat_st += 4 if ks & {"gen", "gen_mux", "var"} else (1 if ks & {"w8", "w8_mux", "var_w8"} else 0)
    cross = [L >> (t + 1) for t in range(n) if (L >> (t + 1)) >= Pf]
    lin, fin, lout, fout, lam_i, lam_o = ub.reorder_cost(P, Pf, L, V)
    lat = lam_i + 2 * (lat_st + sum(d // Pf for d in cross)) + 4 + lam_o
    mem = dict(base.mem)
    depth = lat + math.ceil((V // 2) / P)
    mem["bypass"] = ub.mem_options(P * ub.N_CH * 16, depth)
    mem["bypass_depth"] = depth
    return ub.CdCount(P, W_d, W_c, rows, mem, lat)


KCOMP = {"sumadores de mariposa": "bfly", "rotadores constantes generales": "rot", "rotadores W8": "w8",
         "W8 alternantes (W8 + mux de signo)": "w8", "W8 variables (W8 + mux)": "w8"}


def calibrated(c: ub.CdCount, K: dict, ROTDSP: dict, calib: bool = True) -> dict:
    """Totales de un CdCount con el modelo de ``cd_totals`` del notebook 09 (E9, variante D2.3)."""
    lut = ff = dsp = 0.0
    for r in c.rows:
        if r["componente"].endswith("(en DSP)"):
            n_rot = r["DSP"] // 3
            if calib:
                lut += n_rot * ROTDSP["LUT"]; ff += n_rot * ROTDSP["FF"]; dsp += n_rot * ROTDSP["DSP"]
            else:
                dsp += r["DSP"]
            continue
        k = KCOMP.get(r["componente"])
        kl = K.get(k, 1.0) if (calib and k) else 1.0
        lut += r["LUT"] * kl
        ff += r["FF"] * (K["ff"] if calib else 1.0)
        dsp += r["DSP"] * (K["dsp"] if calib else 1.0)
    for name in ("H̄ doble buffer", "bypass"):
        opts = c.mem[name]
        best = ub.best_mem(opts)
        if best in ("SRL (LUT)", "LUTRAM (LUT)"):
            lut += opts[best]
        elif best == "FF":
            ff += opts[best]
    return {"LUT": lut, "FF": ff, "DSP": dsp}


@dataclass
class Legacy:
    """Clases del conteo de la decisión 36 (``utils_09b.lane_rotators``) en la etapa R2 de L = 128."""

    by_values: dict


def legacy_r2_lanes(L: int = L13, Pf: int = PF13) -> dict:
    """Carriles de la R2 que ``lane_rotators`` cuenta como variables y su clase con el criterio de la tarea.

    ``twiddle_exponents`` marca con e = −1 las posiciones sin rotación de la R2 y
    ``lane_rotators`` las clasifica como W_L^(L−1) (general): cada carril ve entonces
    dos valores no triviales y cuenta como variable. Con e = −1 como 1, la clase es
    la de ``lane_class``.
    """
    rows = [r for r in ub.lane_rotators(L, Pf) if r["etapa"] == 0]
    E = boundary_exponents(L, DECOMPS[BASE13])
    out = {}
    for r in rows:
        i = r["carril"]
        k = lane_class(tuple(int(E[0, i + Pf * c]) for c in range(L // Pf)), L)
        out[(r["clase"], k)] = out.get((r["clase"], k), 0) + 1
    return out
