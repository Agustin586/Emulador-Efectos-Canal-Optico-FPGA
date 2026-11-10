"""Réplica bit a bit de la CD: FFT DIF radix-2² → ×H̄ en bit invertido → IFFT DIT (paso C3.2.2).

Especificación: PDF de arquitectura de la FFT, secciones "Descomposición por etapas" y
"Aritmética bit a bit"; decisiones 31 y 32 de GUIA_ISI_CD.md.
"""
import numpy as np
import pytest

from emulador.blocks import ChromaticDispersion
from emulador.cd import cd_overlap_save
from emulador.fixed_point.cd_fft import (
    L_MAX,
    CdFixedConfig,
    Stage,
    bitrev,
    cd_overlap_save_fixed,
    fft_dif_fixed,
    fft_dif_float,
    fft_stages,
    hbar_permuted,
    ifft_dit_fixed,
    ifft_dit_float,
    scaling_default,
    twiddle_exponents,
    twiddle_table,
)
from emulador.metrics import nrmse
from emulador.quantization import quantize, to_int

POW2 = [2**k for k in range(2, 13)]  # L = 4 ... 4096


def _noise(shape, seed=0, peak=0.95):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return x * peak / np.max(np.abs(np.r_[x.real.ravel(), x.imag.ravel()]))


# --------------------------------------------------------------------------
# Descomposición por etapas
# --------------------------------------------------------------------------
def test_stage_lists_from_pdf_table():
    assert fft_stages(256) == [
        Stage("I", 128, 256), Stage("II", 64, 256), Stage("I", 32, 64), Stage("II", 16, 64),
        Stage("I", 8, 16), Stage("II", 4, 16), Stage("I", 2, 4), Stage("II", 1, 4),
    ]
    assert fft_stages(512)[0] == Stage("R2", 256, 512) and fft_stages(512)[1:] == fft_stages(256)
    assert fft_stages(4096)[:2] == [Stage("I", 2048, 4096), Stage("II", 1024, 4096)]
    assert fft_stages(2048)[0] == Stage("R2", 1024, 2048)


@pytest.mark.parametrize("L_small, L_big, skip", [(1024, 4096, 2), (1024, 2048, 1), (256, 1024, 2), (256, 512, 1)])
def test_chain_is_tail_stages_and_twiddles(L_small, L_big, skip):
    """La cadena de L chico es la cola de la de L grande: mismas etapas y mismos twiddles (índices en W_4096)."""
    small, big = fft_stages(L_small), fft_stages(L_big)
    assert small == big[skip:]
    for st in small:
        e_small = twiddle_exponents(st, L_small)
        e_big = twiddle_exponents(st, L_big)
        if e_small is None:
            assert e_big is None
            continue
        # el exponente depende solo de q = j mod M: el bloque de M posiciones es el mismo
        assert np.array_equal(e_small[: st.M], e_big[: st.M])
        assert np.array_equal(e_small, np.tile(e_small[: st.M], L_small // st.M))


def test_scaling_is_per_version_not_tail():
    """El escalado se define por versión: en 4096 la II 256 tiene s = 0; en 1024 (segunda etapa) s = 1."""
    s4096, _ = scaling_default(4096)
    s1024, _ = scaling_default(1024)
    assert s4096[3] == 0 and s1024[1] == 1
    assert s1024 != s4096[2:]


@pytest.mark.parametrize("L", POW2)
def test_default_scaling_sums_log2L(L):
    s_f, s_i = scaling_default(L)
    n = int(np.log2(L))
    assert len(s_f) == len(s_i) == n
    assert sum(s_f) == n // 2 + 1
    assert sum(s_f) + sum(s_i) == n
    assert all(a + b == 1 for a, b in zip(s_f, s_i))
    for s, st in zip(s_f, fft_stages(L)):
        if st.kind in ("I", "R2"):
            assert s == 1


def test_expensive_rotation_columns():
    """Etapas con rotaciones no triviales: 3, 4, 4, 5 y 5 para L = 256 a 4096 (PDF, Verificación)."""
    counts = []
    for L in (256, 512, 1024, 2048, 4096):
        c = 0
        for st in fft_stages(L):
            e = twiddle_exponents(st, L)
            if e is not None and np.any((e >= 0) & (e % (st.M // 4) != 0)):
                c += 1
        counts.append(c)
    assert counts == [3, 4, 4, 5, 5]


def test_bf2ii_example_L16():
    """Ejemplo del PDF, L = 16 después del primer BF2II."""
    st = fft_stages(16)[1]
    assert st == Stage("II", 4, 16)
    assert twiddle_exponents(st, 16).tolist() == [0, 0, 0, 0, 0, 2, 4, 6, 0, 1, 2, 3, 0, 3, 6, 9]


@pytest.mark.parametrize("L", POW2)
def test_float_fft_matches_numpy_bitrev(L):
    x = _noise((3, L), seed=L)
    X = fft_dif_float(x)
    assert np.max(np.abs(X - np.fft.fft(x)[..., bitrev(L)])) <= 1e-12 * np.max(np.abs(X))


@pytest.mark.parametrize("L", POW2)
def test_float_ifft_dit_from_bitrev(L):
    X = _noise((3, L), seed=L + 1)
    x = ifft_dit_float(X[..., bitrev(L)])
    ref = L * np.fft.ifft(X)
    assert np.max(np.abs(x - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_bitrev_is_involution():
    for L in (16, 1024, 4096):
        p = bitrev(L)
        assert np.array_equal(p[p], np.arange(L))


# --------------------------------------------------------------------------
# Tablas
# --------------------------------------------------------------------------
def test_twiddle_table_rounding_and_saturation():
    wr, wi = twiddle_table(18)
    m = np.arange(L_MAX)
    w = np.exp(-2j * np.pi * m / L_MAX)
    assert np.array_equal(wr, to_int(w.real, 18, 1, rounding="half_up"))
    assert np.array_equal(wi, to_int(w.imag, 18, 1, rounding="half_up"))
    assert wr[0] == 2**17 - 1  # +1 satura al máximo
    assert wi[1024] == -(2**17)  # -1 es representable


def test_hbar_permuted():
    blk = ChromaticDispersion(fs=72e9, length_km=20, N=512)
    hr, hi = hbar_permuted(blk.Hbar, 18)
    p = bitrev(1024)
    assert np.array_equal(hr, to_int(blk.Hbar[p].real, 18, 1, rounding="half_up"))
    assert np.array_equal(hi, to_int(blk.Hbar[p].imag, 18, 1, rounding="half_up"))


# --------------------------------------------------------------------------
# Aritmética: referencia independiente con enteros de Python (L chicos)
# --------------------------------------------------------------------------
def _rnd(v, s, mode):
    if s == 0:
        return v
    q, r = divmod(v, 1 << s)
    half = 1 << (s - 1)
    if mode == "half_up":
        return q + (r >= half)
    return q + (r > half or (r == half and q % 2 == 1))


def _sat(v, W):
    return min(max(v, -(1 << (W - 1))), (1 << (W - 1)) - 1)


def _ref_fft_int(re, im, L, cfg, s_f, inverse=False):
    """FFT DIF (o IFFT DIT) bit a bit, posición por posición, leyendo la especificación al pie de la letra."""
    W, Ft = cfg.W_d, cfg.W_t - 1
    wr, wi = twiddle_table(cfg.W_t)
    z = [(int(a), int(b)) for a, b in zip(re, im)]
    stages = list(zip(fft_stages(L), s_f))
    if inverse:
        stages = stages[::-1]

    def neg(v):
        return _sat(-v, W)

    def rotate(z, st):
        e = twiddle_exponents(st, L)
        if e is None:
            return z
        out = list(z)
        for j in range(L):
            if e[j] < 0:
                continue
            if e[j] % (st.M // 4) == 0:
                k = (e[j] // (st.M // 4)) % 4
                if inverse:
                    k = (-k) % 4
                a, b = z[j]
                # (−j)^k aplicado directo: una sola negación por componente
                out[j] = [(a, b), (b, neg(a)), (neg(a), neg(b)), (neg(b), a)][k]
            else:
                m = e[j] * (L_MAX // st.M)
                c, d = int(wr[m]), int(wi[m])
                a, b = z[j]
                if inverse:
                    pr, pi = a * c + b * d, b * c - a * d
                else:
                    pr, pi = a * c - b * d, a * d + b * c
                out[j] = (_sat(_rnd(pr, Ft, "half_up"), W), _sat(_rnd(pi, Ft, "half_up"), W))
        return out

    def mul_j(z, st, plus):
        if st.kind != "II":
            return z
        out = list(z)
        for j in range(L):
            if j % st.M >= 3 * st.M // 4:
                a, b = z[j]
                out[j] = (neg(b), a) if plus else (b, neg(a))
        return out

    def bfly(z, st, s):
        out = list(z)
        for j in range(L):
            if j % (2 * st.d) < st.d:
                (a, b), (c, d) = z[j], z[j + st.d]
                out[j] = (_sat(_rnd(a + c, s, "half_even"), W), _sat(_rnd(b + d, s, "half_even"), W))
                out[j + st.d] = (_sat(_rnd(a - c, s, "half_even"), W), _sat(_rnd(b - d, s, "half_even"), W))
        return out

    for st, s in stages:
        if not inverse:
            z = mul_j(z, st, plus=False)
            z = bfly(z, st, s)
            z = rotate(z, st)
        else:
            z = rotate(z, st)
            z = bfly(z, st, s)
            z = mul_j(z, st, plus=True)
    return np.array([a for a, _ in z]), np.array([b for _, b in z])


@pytest.mark.parametrize("L", [16, 32, 64])
def test_fixed_fft_and_ifft_bit_exact_against_reference(L):
    cfg = CdFixedConfig()
    rng = np.random.default_rng(L)
    W = cfg.W_d
    lo, hi = -(2 ** (W - 1)), 2 ** (W - 1) - 1
    re = rng.integers(lo // 4, hi // 4, L)
    im = rng.integers(lo // 4, hi // 4, L)
    re[0], im[3] = lo, lo  # incluye el mínimo: la negación satura
    s_f, s_i = scaling_default(L)
    got_r, got_i, _ = fft_dif_fixed(re, im, cfg, s_f)
    exp_r, exp_i = _ref_fft_int(re, im, L, cfg, s_f)
    assert np.array_equal(got_r, exp_r) and np.array_equal(got_i, exp_i)
    got_r, got_i, _ = ifft_dit_fixed(re, im, cfg, s_i)
    exp_r, exp_i = _ref_fft_int(re, im, L, cfg, s_i, inverse=True)
    assert np.array_equal(got_r, exp_r) and np.array_equal(got_i, exp_i)


def test_trivial_rotation_never_rounds():
    """Con datos de un solo LSB, las posiciones con rotación trivial conservan los enteros exactos."""
    cfg = CdFixedConfig()
    L = 64
    st = fft_stages(L)[1]  # II 16, M = 64
    e = twiddle_exponents(st, L)
    trivial = (e >= 0) & (e % (st.M // 4) == 0)
    re = np.arange(L, dtype=np.int64) * 7 + 1  # enteros impares de pocos LSB
    im = -np.arange(L, dtype=np.int64) * 3 - 1
    from emulador.fixed_point.cd_fft import _IntArith

    ar = _IntArith(cfg)
    zr, zi = ar.rotate((re, im), e, st.M, conj=False)
    k = (e[trivial] // (st.M // 4)) % 4
    z = re[trivial] + 1j * im[trivial]
    expected = z * (-1j) ** k
    assert np.array_equal(zr[trivial], expected.real.astype(np.int64))
    assert np.array_equal(zi[trivial], expected.imag.astype(np.int64))
    assert ar.saturations == 0


def test_negation_of_minimum_saturates():
    cfg = CdFixedConfig()
    from emulador.fixed_point.cd_fft import _IntArith

    ar = _IntArith(cfg)
    lo = -(2 ** (cfg.W_d - 1))
    zr, zi = ar.mul_j((np.array([lo, 5]), np.array([3, lo])), np.array([True, True]), plus=False)
    assert zr.tolist() == [3, lo] and zi.tolist() == [2 ** (cfg.W_d - 1) - 1, -5]
    assert ar.saturations == 1


# --------------------------------------------------------------------------
# Cadena completa (overlap-save)
# --------------------------------------------------------------------------
WIDE = CdFixedConfig(W_in=30, W_d=36, W_t=26, W_h=26, W_out=36, I_out=2)


@pytest.mark.parametrize("N_CD, fs, z", [(512, 72e9, 20), (512, 288e9, 10), (2048, 576e9, 20), (2048, 288e9, 0)])
def test_wide_formats_match_overlap_save_and_process_hw(N_CD, fs, z):
    blk = ChromaticDispersion(fs=fs, length_km=z, N=N_CD)
    x = _noise(5 * N_CD + 37, seed=N_CD + z)
    res = cd_overlap_save_fixed(x, blk.Hbar, WIDE)
    y_os = cd_overlap_save(x, blk.Hbar)
    assert res.y.shape == y_os.shape
    assert nrmse(res.y, y_os) < 1e-6
    assert nrmse(res.y[: x.size], blk.process_hw(x)) < 1e-6
    assert all(v == 0 for v in res.saturations.values())


def test_wide_fft_without_shifts_matches_numpy():
    """W_d grande y sin desplazamientos en la FFT: coincide con np.fft.fft (bit invertido) dentro del error de los twiddles."""
    L = 4096
    cfg = CdFixedConfig(W_d=40, W_t=22)
    x = _noise(L, seed=9, peak=2.0**-13)  # |X| ≤ L·√2·pico < 2: entra en <40, 2> sin escalar
    F = cfg.W_d - 2
    re, im = to_int(x, cfg.W_d, 2)
    Xr, Xi, n_sat = fft_dif_fixed(re, im, cfg, [0] * 12)
    X = (Xr + 1j * Xi) * 2.0**-F
    ref = np.fft.fft(quantize(x, cfg.W_d, 2))[bitrev(L)]
    assert n_sat == 0
    assert nrmse(X, ref) < 1e-5


def test_default_formats_internal_error_and_no_saturation():
    """Formatos de la decisión 32: error interno chico y sin saturaciones con una señal tipo ruido."""
    blk = ChromaticDispersion(fs=288e9, length_km=10, N=2048)
    x = quantize(_noise(8 * 2048, seed=3), 14, 1)
    res = cd_overlap_save_fixed(x, blk.Hbar)
    e = nrmse(res.y_internal, cd_overlap_save(x, blk.Hbar))
    assert 1e-6 < e < 1e-4
    assert res.saturations == {"in": 0, "fft": 0, "hbar": 0, "ifft": 0, "out": 0}
    assert np.array_equal(res.y, quantize(res.y_internal, 14, 2))


def test_saturation_counters_by_transform():
    """Sin escalado en la FFT, satura en la FFT; el contador lo separa por transformada."""
    blk = ChromaticDispersion(fs=72e9, length_km=2, N=512)
    x = _noise(4 * 512, seed=4)
    n = 10
    res = cd_overlap_save_fixed(x, blk.Hbar, CdFixedConfig(s_f=(0,) * n, s_i=(1,) * n))
    assert res.saturations["fft"] > 0
    res = cd_overlap_save_fixed(1.5 * x, blk.Hbar)
    assert res.saturations["in"] > 0


def test_output_matches_block_alignment_and_length():
    blk = ChromaticDispersion(fs=72e9, length_km=10, N=512)
    x = _noise(3000, seed=5)
    res = cd_overlap_save_fixed(x, blk.Hbar)
    assert res.y.shape == (6 * 512,)
    assert res.codes[0].shape == res.y.shape


@pytest.mark.parametrize("kwargs", [{"W_d": 12}, {"W_d": 40, "W_t": 26}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        CdFixedConfig(**kwargs)


def test_trace_has_one_entry_per_stage():
    blk = ChromaticDispersion(fs=72e9, length_km=2, N=512)
    res = cd_overlap_save_fixed(_noise(1024), blk.Hbar, trace=True)
    n = 10
    assert [k for k, _ in res.trace] == (
        [f"fft {i}" for i in range(n)] + ["hbar"] + [f"ifft {i}" for i in range(n)]
    )


@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_overlap_save_rejects_N_outside_hardware_versions(N):
    """Solo N_CD ∈ {512, 2048} (decisión 30); ninguna de las dos usa la R2 (L = 1024 y 4096)."""
    Hbar = np.exp(1j * np.linspace(0, 1, 2 * N))
    with pytest.raises(ValueError):
        cd_overlap_save_fixed(_noise(4 * N), Hbar)


def test_hardware_versions_have_no_r2_stage():
    for N in (512, 2048):
        assert all(st.kind != "R2" for st in fft_stages(2 * N))
