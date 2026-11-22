"""Utilidades de E9 (recursos) y E10 (rango de parámetros) del notebook 09 (tarea D2.2).

NO es parte del núcleo. Usa el grafo y la aritmética de la FFT en punto fijo del
núcleo (``emulador.fixed_point.cd_fft``) sin modificarlos.

Contenido:
  - overlap-save en punto fijo con L y V cualesquiera (``os_fixed``), con las
    mariposas, rotadores, tablas y redondeos de ``cd_fft`` (el del núcleo fija V = L/2);
  - recorrido del grafo FFT DIF radix-2² / IFFT DIT plegado en P_FFT carriles:
    clase de cada rotador por carril (``lane_rotators``);
  - conteo de LUT, FF, DSP y memoria de la CD (overlap-save 256, 64) y de la ISI
    (FFA con K = 12), todo como **estimación por conteo**;
  - estructura FFA recursiva verificada contra la convolución directa.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

import utils_09 as u9
from emulador.cd import cd_R, cd_reference, cd_transfer
from emulador.config import D_si, S_si
from emulador.fixed_point.cd_fft import (CdFixedConfig, _IntArith, _transform, fft_stages, hbar_permuted,
                                         twiddle_exponents, twiddle_table)
from emulador.fixed_point.integer import round_shift, saturate_int
from emulador.quantization import to_int

FS = u9.FS
L_OS, V_OS = 256, 64
P_LIST = (48, 96, 192)
P_FFT = {48: 64, 96: 128, 192: 256}  # un bloque de 256 cada 4, 2 y 1 ciclos
N_POL, N_CH = 2, 4
W_IN = 16  # Q1.15
W_H = 18  # H̄, decisión 32
WD_COUNT = (14, 16, 17, 18, 20)  # W_d del conteo
WC_COUNT = (12, 14, 16, 18)  # W_c
WD_SQNR = (17, 18, 20)  # W_d medibles con cd_fft (W_d − 2 >= W_in − 1)

# base de partes de Vivado 2024.2, xczu48dr-fsvg1517-2-e
DEVICE = {"LUT": 425_280, "FF": 850_560, "BRAM36": 1080, "URAM": 80, "DSP": 4272}
CRIT = {"LUT": 0.70, "FF": 0.70, "DSP": 0.80, "BRAM36": 0.80}  # criterio de diseño, no del fabricante
EXTERNAL_ROT = {64: (62, 56, 21), 128: (124, 112, 42), 256: (0, 408, 84)}  # variables / generales / W8 por transformada

OUT_CAL = Path(__file__).resolve().parents[1] / "docs" / "golden_model" / "caracterizacion_lut"


# --------------------------------------------------------------------------
# Overlap-save en punto fijo con L y V cualesquiera
# --------------------------------------------------------------------------
def os_fixed(x, Hbar, V, cfg: CdFixedConfig):
    """ADC <W_in,1> → FFT DIF → ×H̄ → IFFT DIT → DAC <W_out,I_out>, tramas de L con V de solapamiento.

    Igual que ``cd_overlap_save_fixed`` salvo el armado de tramas (V muestras de
    la anterior y L − V nuevas). Devuelve (y_DAC, y_interna en <W_d, 2>,
    saturaciones); salida de largo nb·(L − V), retrasada V/2.
    """
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
    z = _transform(z, L, ar, s_f, inverse=False)
    z = ar.mul_table(z, hbar_permuted(Hbar, cfg.W_h, rounding=cfg.table_rounding))
    z = _transform(z, L, ar, s_i, inverse=True)
    yr, yi = z[0][:, V:].reshape(-1), z[1][:, V:].reshape(-1)
    y_int = (yr + 1j * yi) * 2.0**-F_d
    orr = round_shift(yr, F_d - F_out, cfg.out_rounding) if F_out <= F_d else yr << (F_out - F_d)
    oi = round_shift(yi, F_d - F_out, cfg.out_rounding) if F_out <= F_d else yi << (F_out - F_d)
    orr, n1 = saturate_int(orr, cfg.W_out)
    oi, n2 = saturate_int(oi, cfg.W_out)
    return (orr + 1j * oi) * 2.0**-F_out, y_int, ar.saturations + n1 + n2


def cfg_for(W_d: int, W_c: int) -> CdFixedConfig:
    """Entrada y salida Q1.15, datos <W_d, 2>, twiddles <W_c, 1>, H̄ <18, 1>."""
    return CdFixedConfig(W_in=W_IN, W_d=W_d, W_t=W_c, W_h=W_H, W_out=16, I_out=1)


def sqnr_db(y, ref) -> float:
    return float(10 * np.log10(np.sum(np.abs(ref) ** 2) / np.sum(np.abs(y - ref) ** 2)))


# --------------------------------------------------------------------------
# Grafo de la FFT plegado en carriles
# --------------------------------------------------------------------------
def _rot_class(e: int, M: int):
    """Clase de W_M^e: 'one', 'triv' (−j, −1, j), 'w8' (múltiplo impar de π/4) o 'gen'."""
    m = e % M
    if m == 0:
        return ("one", 0)
    if m % (M // 4) == 0:
        return ("triv", m * 4 // M)
    if (m * 8) % M == 0:
        return ("w8", m * 8 // M)
    return ("gen", m * (4096 // M))


def lane_rotators(L: int, P: int) -> list[dict]:
    """Rotadores por carril para el plegado x[i + P·c] -> carril i, ciclo c (bits de ciclo = MSB).

    Para cada etapa con rotación y cada carril, la clase según los valores que
    ve el carril en los C = L/P ciclos del bloque:
      one, triv, w8, gen : constante (un solo valor);
      var_w8             : alterna entre triviales y W8 (W8 con multiplexor de signo);
      var_triv           : alterna solo entre triviales (multiplexor);
      var                : alterna e incluye un valor general (multiplicador variable).
    También cuenta los ×(−j) de BF2II (``mulj``: constante o alternante).
    Devuelve una lista de {etapa, M, carril, clase, valores}.
    """
    C = L // P
    out = []
    for si, st in enumerate(fft_stages(L)):
        j = np.arange(L)
        if st.kind == "II":
            mask = (j % st.M) >= 3 * st.M // 4
            for i in range(P):
                vals = {bool(mask[i + P * c]) for c in range(C)}
                cls = "mulj_var" if len(vals) > 1 else ("mulj" if True in vals else None)
                if cls:
                    out.append({"etapa": si, "M": st.M, "carril": i, "clase": cls, "valores": vals})
        e = twiddle_exponents(st, L)
        if e is None:
            continue
        for i in range(P):
            vals = {_rot_class(int(e[i + P * c]), st.M) for c in range(C)}
            if len(vals) == 1:
                cls = next(iter(vals))[0]
            else:
                kinds = {v[0] for v in vals}
                if "gen" in kinds:
                    cls = "var"
                elif "w8" in kinds:
                    cls = "var_w8"
                else:
                    cls = "var_triv"
            out.append({"etapa": si, "M": st.M, "carril": i, "clase": cls, "valores": vals})
    return out


def rotator_counts(L: int, P: int) -> dict:
    """Conteo por transformada: variables, generales, W8 (constantes + alternantes) y triviales."""
    c = {}
    for r in lane_rotators(L, P):
        c[r["clase"]] = c.get(r["clase"], 0) + 1
    c["W8 total"] = c.get("w8", 0) + c.get("var_w8", 0)
    return c


# --------------------------------------------------------------------------
# Costos por pieza (conteo)
# --------------------------------------------------------------------------
def csd(v: int) -> list[tuple[int, int]]:
    """Dígitos no nulos de la representación CSD de un entero: [(posición, ±1)]."""
    out, k, pos = [], int(v), 0
    while k:
        if k & 1:
            d = 2 - (k & 3)  # 1 si k ≡ 1 (mod 4), −1 si k ≡ 3
            out.append((pos, d))
            k -= d
        k >>= 1
        pos += 1
    return out


def const_mult_adders(c: int, s: int) -> tuple[int, int]:
    """Sumadores para x·c y x·s por desplazamiento y suma (c, s: códigos de cos y sen).

    Devuelve (sin compartir, MCM simple). Sin compartir: (nz_c − 1) + (nz_s − 1).
    MCM simple: los dígitos en la misma posición con el mismo signo (o todos con
    signo opuesto) se suman una vez y se reutilizan: se ahorra max(0, n_común − 1).
    """
    dc, ds = csd(c), csd(s)
    nc, ns = len(dc), len(ds)
    base = max(nc - 1, 0) + max(ns - 1, 0)
    pc, ps = dict(dc), dict(ds)
    same = sum(1 for p, d in pc.items() if ps.get(p) == d)
    opp = sum(1 for p, d in pc.items() if ps.get(p) == -d)
    common = max(same, opp)
    return base, base - max(0, common - 1)


@dataclass
class CdCount:
    """Conteo de la CD (overlap-save, 2 polarizaciones) para un P, W_d y W_c."""

    P: int
    W_d: int
    W_c: int
    rows: list  # desglose: {componente, LUT, FF, DSP, nota}
    mem: dict
    latency_cycles: int

    def total(self, key: str) -> int:
        return int(sum(r.get(key, 0) for r in self.rows))


def reorder_cost(P: int, Pf: int, L: int = L_OS, V: int = V_OS, bits: int = 2 * W_IN):
    """Reordenamiento de entrada (P muestras/ciclo -> tramas de L con V de solapamiento) y de salida.

    Para cada carril (entrada) o palabra de salida, fuentes distintas (posición,
    antigüedad en ciclos) sobre un período de bloques -> tamaño del multiplexor;
    LUT por bit = ⌈m/4⌉ si m > 1 (LUT6 como mux 4:1). Registros: antigüedad máxima
    por posición × bits. Devuelve (LUT, FF, latencia_entrada, latencia_salida).
    """
    M = L - V
    C = L // Pf
    Tb = M // P
    assert C == Tb, (P, Pf)
    nb = 8
    # entrada: el carril i en el ciclo c del bloque b lee la muestra s = M·b − V + Pf·c + i
    need = []
    for b in range(2, 2 + nb):
        for c in range(C):
            for i in range(Pf):
                s = M * b - V + Pf * c + i
                need.append((b, c, i, s // P, s % P))
    lam = max(t - (Tb * b + c) for b, c, i, t, k in need)  # menor retardo fijo con todas las muestras llegadas
    src, maxage = {}, {}
    for b, c, i, t, k in need:
        age = Tb * b + c + lam - t
        src.setdefault(i, set()).add((k, age))
        maxage[k] = max(maxage.get(k, 0), age)
    lut_in = sum(math.ceil(len(v) / 4) for v in src.values() if len(v) > 1) * bits
    ff_in = sum(maxage.values()) * bits
    # salida: la posición j >= V del bloque b (ciclo c = j // Pf, carril j % Pf) es la muestra M·b + j − V
    have = []
    for b in range(2, 2 + nb):
        for j in range(V, L):
            have.append((Tb * b + j // Pf, j % Pf, M * b + j - V))
    lam_o = max(tc - s // P for tc, lane, s in have)
    src_o, age_o = {}, {}
    for tc, lane, s in have:
        k = s % P
        age = s // P + lam_o - tc
        src_o.setdefault(k, set()).add((lane, age))
        age_o[(lane)] = max(age_o.get(lane, 0), age)
    lut_out = sum(math.ceil(len(v) / 4) for v in src_o.values() if len(v) > 1) * bits
    ff_out = sum(age_o.values()) * bits
    return lut_in, ff_in, lut_out, ff_out, lam, lam_o


def mem_options(width_bits: int, depth: int) -> dict:
    """Recursos para una memoria de ``depth`` palabras de ``width_bits`` en cada tecnología."""
    return {
        "FF": width_bits * depth,
        "SRL (LUT)": width_bits * math.ceil(depth / 32),
        "LUTRAM (LUT)": math.ceil(width_bits / 7) * 8 * math.ceil(depth / 64),  # RAM64M8: 64 × 7 en 8 LUT
        "BRAM36": math.ceil(width_bits / 72) * math.ceil(depth / 512),
        "URAM": math.ceil(width_bits / 72) * math.ceil(depth / 4096),
    }


def best_mem(opts: dict) -> str:
    """Opción de menor fracción del dispositivo."""
    frac = {"FF": opts["FF"] / DEVICE["FF"], "SRL (LUT)": opts["SRL (LUT)"] / DEVICE["LUT"],
            "LUTRAM (LUT)": opts["LUTRAM (LUT)"] / DEVICE["LUT"], "BRAM36": opts["BRAM36"] / DEVICE["BRAM36"],
            "URAM": opts["URAM"] / DEVICE["URAM"]}
    return min(frac, key=frac.get)


def cd_count(P: int, W_d: int, W_c: int, mcm: bool = False, gen_dsp: bool = False,
             L: int = L_OS, V: int = V_OS) -> CdCount:
    """Conteo de la CD: FFT + ×H̄ + IFFT por polarización, más reordenamiento, memoria y bypass.

    gen_dsp: rotadores constantes generales en DSP (D2.3): 3 DSP por rotador, coeficiente
    fijo en el puerto B, registros dentro del DSP; 0 LUT y 0 FF en el conteo. Los W8 siguen en LUT.
    """
    # carriles: un bloque de L − V muestras nuevas cada (L − V)/P ciclos -> L/C carriles
    # (con (256, 64) da P_FFT; con (128, 32) y P = 48, 128 puntos en 2 ciclos)
    assert (L - V) % P == 0 and L % ((L - V) // P) == 0, (P, L, V)
    Pf = L // ((L - V) // P)
    wr, wi = twiddle_table(W_c)
    rows = []
    nT = 2 * N_POL  # FFT e IFFT por polarización
    # mariposas
    n_st = len(fft_stages(L))
    rows.append({"componente": "sumadores de mariposa", "LUT": nT * n_st * 2 * Pf * (W_d + 1),
                 "FF": nT * n_st * Pf * 2 * W_d, "DSP": 0,
                 "nota": f"2·{Pf} sumadores reales de {W_d + 1} bits por etapa, {n_st} etapas, {nT} transformadas"})
    # rotadores
    rot = lane_rotators(L, Pf)
    Wp = W_d + W_c - 1  # ancho de los sumadores de un producto por constante
    lut = {"gen": 0, "w8": 0, "var": 0, "var_w8": 0, "var_triv": 0, "mulj_var": 0}
    ff = dict.fromkeys(lut, 0)
    dsp_var = 0
    dsp_gen = 0
    c8 = int(wr[512])  # cos(π/4) en <W_c, 1>
    a8 = max(len(csd(c8)) - 1, 0)
    w8_lut = 2 * (W_d + 1) + 2 * a8 * Wp
    for r in rot:
        k = r["clase"]
        if k == "gen" and gen_dsp:
            dsp_gen += 3
        elif k == "gen":
            (_, m), = r["valores"]
            cr, ci = int(wr[m]), int(wi[m])
            base_r, mcm_r = const_mult_adders(cr, ci)
            base_i, mcm_i = const_mult_adders(cr, ci)  # misma constante para la parte a y la b
            adders = (mcm_r + mcm_i) if mcm else (base_r + base_i)
            lut[k] += adders * Wp + 2 * Wp  # 2 sumas finales del producto complejo
            ff[k] += 2 * W_d
        elif k == "w8":
            lut[k] += w8_lut
            ff[k] += 2 * W_d
        elif k == "var":
            dsp_var += 3
            lut[k] += 2 * W_c  # tabla de coeficientes alternantes indexada por el ciclo (LUT6 = 64 × 1)
        elif k == "var_w8":
            lut[k] += w8_lut + 2 * W_d
            ff[k] += 2 * W_d
        elif k in ("var_triv", "mulj_var"):
            lut[k] += 2 * W_d  # intercambio re/im; el signo va al sumador siguiente (suma/resta)
    names = {"gen": "rotadores constantes generales", "w8": "rotadores W8", "var": "rotadores variables (3 DSP)",
             "var_w8": "W8 alternantes (W8 + mux de signo)", "var_triv": "triviales alternantes (mux)",
             "mulj_var": "×(−j) alternantes de BF2II (mux)"}
    for k, name in names.items():
        dsp = nT * dsp_var if k == "var" else (nT * dsp_gen if k == "gen" else 0)
        note = ""
        if k == "gen":
            note = "en DSP: 3 por rotador, coeficiente en el puerto B" if gen_dsp else ("CSD con MCM simple" if mcm else "CSD sin compartir")
        rows.append({"componente": name + (" (en DSP)" if (k == "gen" and gen_dsp) else ""), "LUT": nT * lut[k],
                     "FF": nT * ff[k], "DSP": dsp, "nota": note})
    # ×H̄
    rows.append({"componente": "×H̄ (multiplicador complejo, 3 DSP)", "LUT": 0, "FF": 0, "DSP": N_POL * Pf * 3,
                 "nota": f"{Pf} por polarización; registros dentro del DSP"})
    # retardos entre ciclos (etapas con d >= P_FFT), FFT e IFFT
    cross = [st.d for st in fft_stages(L) if st.d >= Pf]
    ff_cross = nT * sum(2 * d * 2 * W_d for d in cross)  # estimación: 2·d muestras complejas por etapa cruzada
    rows.append({"componente": "retardos entre ciclos (primeras etapas)", "LUT": 0, "FF": ff_cross, "DSP": 0,
                 "nota": f"etapas con d >= {Pf}: d = {cross}; 2·d muestras por etapa (estimación)"})
    # reordenamiento
    lin, fin, lout, fout, lam_i, lam_o = reorder_cost(P, Pf, L, V)
    rows.append({"componente": f"reordenamiento de entrada (incluye solapamiento de {V})", "LUT": N_POL * lin, "FF": N_POL * fin,
                 "DSP": 0, "nota": "mux por bit, entrada Q1.15 compleja"})
    rows.append({"componente": "reordenamiento de salida", "LUT": N_POL * lout, "FF": N_POL * fout, "DSP": 0, "nota": ""})
    # latencia (estimación)
    st_lat = 0
    for st in fft_stages(L):
        st_lat += 1  # registro de la mariposa
        e = twiddle_exponents(st, L)
        if e is not None:
            kinds = {r["clase"] for r in rot if r["M"] == st.M and r["clase"] not in ("mulj", "mulj_var")}
            st_lat += 4 if "var" in kinds else (1 if kinds - {"one", "triv"} else 0)
    cross_lat = sum(d // Pf for d in cross)
    lat = lam_i + 2 * (st_lat + cross_lat) + 4 + lam_o  # + ×H̄ en DSP (4 ciclos)
    # memoria
    C = L // Pf
    hb = mem_options(Pf * 2 * W_H, 2 * C)  # por carril, profundidad C, doble buffer; compartido entre polarizaciones
    bypass_depth = lat + math.ceil((V // 2) / P)
    byp = mem_options(P * N_CH * 16, bypass_depth)
    mem = {"H̄ doble buffer": hb, "bypass": byp, "bypass_depth": bypass_depth}
    return CdCount(P, W_d, W_c, rows, mem, lat)


# --------------------------------------------------------------------------
# ISI: FFA recursivo
# --------------------------------------------------------------------------
@dataclass
class FfaTally:
    pre: list  # anchos de los sumadores de pre-suma (un elemento = un sumador real)
    post: list
    mults: int
    coefs: int  # coeficientes distintos de los subfiltros


def ffa_conv(x, h, factors, P, tally: FfaTally, w_pre=16, w_post=34, depth_post=0):
    """Convolución lineal y = h * x con la descomposición FFA recursiva (2 y 3 paralelos).

    Cuenta los sumadores como vectores de P/f elementos (un sumador real por fase),
    con su ancho: pre-sumas w_pre + 1 por nivel; post-sumas 34 + 1 por nivel.
    Ecuaciones (2-paralelo): y0 = h0x0 + z⁻¹h1x1, y1 = (h0+h1)(x0+x1) − h0x0 − h1x1.
    (3-paralelo): s0..s5 = h0x0, h1x1, h2x2, (h0+h1)(x0+x1), (h1+h2)(x1+x2), (h0+h1+h2)(x0+x1+x2);
      A = s3 − s1, B = s4 − s1 (compartidos);
      y0 = s0 + z⁻¹(B − s2), y1 = A − s0 + z⁻¹ s2, y2 = s5 − A − B  (8 post-sumas).
    """
    if not factors:
        tally.mults += P * len(h)
        tally.coefs += len(h)
        return np.convolve(x, h)
    f, rest = factors[0], factors[1:]
    Pn = P // f
    n = -(-len(x) // f) * f
    xs = [np.pad(x, (0, n - len(x)))[j::f] for j in range(f)]
    m = -(-len(h) // f) * f
    hs = [np.pad(h, (0, m - len(h)))[j::f] for j in range(f)]
    sub = lambda xx, hh, wp: ffa_conv(xx, hh, rest, Pn, tally, wp, w_post, depth_post)
    wpo = w_post + depth_post + len(rest)  # ancho de las post-sumas de este nivel (34 + 1 por nivel interior)
    if f == 2:
        tally.pre += [w_pre + 1] * Pn
        s0, s1 = sub(xs[0], hs[0], w_pre), sub(xs[1], hs[1], w_pre)
        s2 = sub(xs[0] + xs[1], hs[0] + hs[1], w_pre + 1)
        tally.post += [wpo + 1] * Pn * 3
        k = max(map(len, (s0, s1, s2))) + 1
        pad = lambda v: np.pad(v, (0, k - len(v)))
        y0 = pad(s0) + np.r_[0, pad(s1)[:-1]]
        y1 = pad(s2) - pad(s0) - pad(s1)
        ys = [y0, y1]
    else:
        tally.pre += [w_pre + 1] * Pn * 2 + [w_pre + 2] * Pn
        s0, s1, s2 = (sub(xs[j], hs[j], w_pre) for j in range(3))
        s3 = sub(xs[0] + xs[1], hs[0] + hs[1], w_pre + 1)
        s4 = sub(xs[1] + xs[2], hs[1] + hs[2], w_pre + 1)
        s5 = sub(xs[0] + xs[1] + xs[2], hs[0] + hs[1] + hs[2], w_pre + 2)
        tally.post += [wpo + 1] * Pn * 8
        k = max(map(len, (s0, s1, s2, s3, s4, s5))) + 1
        pad = lambda v: np.pad(v, (0, k - len(v)))
        d = lambda v: np.r_[0, pad(v)[:-1]]
        A, B = pad(s3) - pad(s1), pad(s4) - pad(s1)
        y0 = pad(s0) + d(B - pad(s2))
        y1 = A - pad(s0) + d(s2)
        y2 = pad(s5) - A - B
        ys = [y0, y1, y2]
    y = np.zeros(f * k)
    for j in range(f):
        y[j::f] = ys[j]
    return y[: len(x) + len(h) - 1]


def isi_ffa(P: int, K: int = 12, a: int = 2, b: int = 1, check: bool = True):
    """Los órdenes distintos de factores (a de 2, b de 3): {orden: (tally, error de verificación)}."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4 * P * 3)
    h = rng.standard_normal(K)
    out = {}
    for order in sorted(set(permutations([2] * a + [3] * b))):
        t = FfaTally([], [], 0, 0)
        y = ffa_conv(x, h, list(order), P, t)
        err = float(np.max(np.abs(y - np.convolve(x, h)))) if check else None
        out["-".join(map(str, order))] = (t, err)
    return out


# --------------------------------------------------------------------------
# Calibración con síntesis
# --------------------------------------------------------------------------
def read_calibration() -> list[dict] | None:
    path = OUT_CAL / "resultados.csv"
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# E10: CD con λ, D, S y z directos (sin Scenario)
# --------------------------------------------------------------------------
def cd_eff(D_ps, z_km, lam_nm) -> float:
    """CD efectiva |D|·z·(λ/1550 nm)², ps/nm."""
    return abs(D_ps) * z_km * (lam_nm / 1550.0) ** 2


def os_error_direct(X, D_ps, S_ps, z_km, lam_nm, L=L_OS, V=V_OS, sl=None):
    """NRMSE del overlap-save (L, V) contra cd_reference con (D, S, z, λ) directos."""
    from emulador.metrics import nrmse
    lam, z = lam_nm * 1e-9, z_km * 1e3
    D, S = D_si(D_ps), S_si(S_ps)
    k = np.fft.fftfreq(L) * L
    Hb = cd_transfer(L, FS, D, lam, z, S) * np.exp(-2j * np.pi * k * (V / 2) / L)
    xp = np.pad(X, (0, V // 2))
    y = u9.os_apply(xp, Hb, V)[V // 2 : V // 2 + X.size]
    ref = cd_reference(X, FS, D, lam, z, S)
    sl = slice(None) if sl is None else sl
    return nrmse(y[sl], ref[sl]), cd_R(FS, D, lam, z)
