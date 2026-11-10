"""Test cruzado: ``quantize``/``to_int`` contra ``ac_fixed`` de HALCON (paso C3.0, decisión 33).

Corre el simulador ``models/fixed_point/emulador-real/quant_crosscheck/`` en el
contenedor ``hawk_workspace`` (``run_crosscheck.sh``) y compara sus enteros con
los de ``emulador.quantization``. Tienen que ser idénticos, empates exactos
incluidos. Caminos (decisión 32):

- double -> ``ac_fixed<W, I>`` (lo que hace ``SET``), para los cuatro formatos de
  la CD, con ``AC_RND_CONV``, ``AC_RND`` y ``AC_TRN``, y ``AC_SAT`` o ``AC_WRAP``;
- a) producto exacto ``<38, 3>`` y suma o resta de dos productos ``<39, 4>`` ->
  ``<20, 2>`` con ``AC_RND`` y ``AC_SAT``;
- b) mariposa: a ± b en ``<21, 3>``, desplazamiento de s bits como cambio exacto de
  escala y redondeo a ``<20, 2>`` con ``AC_RND_CONV`` y ``AC_SAT``. Además, ``>> 1``
  de ``ac_fixed``, que trunca;
- c) negación (por -j) de ``<20, 2>``: -(-2) satura a 2 - q.

Se saltea si no hay Docker, la imagen ``hawk_workspace`` o HALCON (``HALCON_DIR``,
por defecto ``/home/wanda/halcon/tutorials``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from emulador.quantization import to_int

PROJECT = Path(__file__).resolve().parents[3] / "fixed_point" / "emulador-real" / "quant_crosscheck"
SCRIPT = PROJECT / "run_crosscheck.sh"
HALCON_DIR = Path(os.environ.get("HALCON_DIR", "/home/wanda/halcon/tutorials"))
IMAGE = os.environ.get("HAWK_IMAGE", "hawk_workspace")

# formatos de la decisión 32: entrada, interno, twiddles y H̄, salida
FORMATS = [(14, 1), (20, 2), (18, 1), (14, 2)]
MODES = {"conv": "half_even", "rnd": "half_up", "trn": "trunc"}
OVERFLOW = {"sat": True, "wrap": False}

D_W, D_I = 20, 2  # datos internos: F = 18
C_W, C_I = 18, 1  # twiddles y H̄: F = 17
F_D, F_C = D_W - D_I, C_W - C_I
F_P = F_D + F_C  # producto <38, 3>: F = 35


def _image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(
    not (SCRIPT.exists() and (HALCON_DIR / "lib/core/inc/ac_fixed.h").exists() and _image_available()),
    reason="hace falta Docker, la imagen hawk_workspace y HALCON en HALCON_DIR",
)


def _double_inputs(rng) -> np.ndarray:
    """Doubles: aleatorios, empates exactos en la grilla de cada formato y bordes."""
    parts = [rng.uniform(-3, 3, 4000), rng.uniform(-8, 8, 500)]
    for W, I in FORMATS:
        F = W - I
        q = 2.0**-F
        k = rng.integers(-(2 ** (W - 1)) - 8, 2 ** (W - 1) + 8, 400)
        parts.append((k + 0.5) * q)  # empates exactos
        lo, hi = -(2.0 ** (I - 1)), 2.0 ** (I - 1) - q
        parts.append(np.array([lo, hi, lo - q / 2, hi + q / 2, lo - q, hi + q, q / 2, -q / 2, 1.5 * q, -1.5 * q]))
    parts.append(np.array([0.0, -0.0, 1e-300, -1e-300, 1.0, -1.0, 2.0, -2.0, 10.0, -10.0]))
    return np.concatenate(parts)


def _product_inputs(rng) -> np.ndarray:
    """Filas a1 b1 a2 b2 (códigos): aleatorias, con empates exactos y extremos."""
    n = 3000
    a = rng.integers(-(2 ** (D_W - 1)), 2 ** (D_W - 1), (n, 2))
    b = rng.integers(-(2 ** (C_W - 1)), 2 ** (C_W - 1), (n, 2))
    # empates: b1 = ±2^16 (±0,5) con a1 impar deja exactamente medio LSB de <20, 2>;
    # b2 = -2^17 (-1) no cambia los 17 bits que se descartan
    m = n // 3
    b[:m, 0] = rng.choice([-(2**16), 2**16], m)
    a[:m, 0] |= 1
    b[:m, 1] = -(2**17)
    rows = np.column_stack([a[:, 0], b[:, 0], a[:, 1], b[:, 1]])
    amin, cmin, amax, cmax = -(2 ** (D_W - 1)), -(2 ** (C_W - 1)), 2 ** (D_W - 1) - 1, 2 ** (C_W - 1) - 1
    extremes = np.array(
        [
            [amin, cmin, amin, cmin],  # (-2)(-1) + (-2)(-1) = 4: satura
            [amin, cmin, amax, cmax],
            [amax, cmax, amin, cmin],
            [amin, cmax, amin, cmax],
            [amax, cmin, amax, cmin],
            [0, 0, 0, 0],
        ]
    )
    return np.vstack([rows, extremes])


def _butterfly_inputs(rng) -> np.ndarray:
    n = 3000
    ab = rng.integers(-(2 ** (D_W - 1)), 2 ** (D_W - 1), (n, 2))
    lo, hi = -(2 ** (D_W - 1)), 2 ** (D_W - 1) - 1
    extremes = np.array([[lo, lo], [hi, hi], [lo, hi], [hi, lo], [1, 0], [-1, 0], [3, 0], [-3, 0], [0, 0]])
    return np.vstack([ab, extremes])


def _negation_inputs(rng) -> np.ndarray:
    lo, hi = -(2 ** (D_W - 1)), 2 ** (D_W - 1) - 1
    return np.concatenate([rng.integers(lo, hi + 1, 1000), [lo, lo + 1, hi, 0, 1, -1]])


def _write(path: Path, rows) -> None:
    rows = np.atleast_1d(rows)
    if rows.ndim == 1:
        lines = [repr(float(v)) if isinstance(v, (float, np.floating)) else str(int(v)) for v in rows]
    else:
        lines = [" ".join(str(int(v)) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def _read(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64, ndmin=2)


@pytest.fixture(scope="module")
def halcon(tmp_path_factory):
    """Genera las entradas, corre el simulador una vez y devuelve entradas y directorio."""
    data = tmp_path_factory.mktemp("quant_crosscheck")
    rng = np.random.default_rng(33)
    inputs = {
        "double": _double_inputs(rng),
        "product": _product_inputs(rng),
        "butterfly": _butterfly_inputs(rng),
        "negation": _negation_inputs(rng),
    }
    _write(data / "double_in.txt", inputs["double"].astype(float))
    _write(data / "product_in.txt", inputs["product"])
    _write(data / "butterfly_in.txt", inputs["butterfly"])
    _write(data / "negation_in.txt", inputs["negation"])
    r = subprocess.run(
        [str(SCRIPT), str(data)], capture_output=True, text=True, timeout=900, env={**os.environ, "HALCON_DIR": str(HALCON_DIR)}
    )
    assert r.returncode == 0, f"run_crosscheck.sh falló:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return inputs, data


def _mismatch(got, expected, x) -> str:
    bad = np.flatnonzero(got != expected)
    i = bad[:5]
    return f"{bad.size} diferencias; entradas {x[i]!r}, HALCON {got[i]!r}, Python {expected[i]!r}"


@pytest.mark.parametrize("o", list(OVERFLOW))
@pytest.mark.parametrize("q", list(MODES))
@pytest.mark.parametrize("W, I", FORMATS)
def test_double_to_ac_fixed(halcon, W, I, q, o):
    inputs, data = halcon
    x = inputs["double"]
    got = _read(data / f"double_{W}_{I}_{q}_{o}.txt")[:, 0]
    expected = to_int(x, W, I, rounding=MODES[q], saturate=OVERFLOW[o])
    assert np.array_equal(got, expected), _mismatch(got, expected, x)


def test_double_inputs_include_exact_ties(halcon):
    """Cobertura: hay empates exactos (medio LSB) para cada formato."""
    x = halcon[0]["double"]
    for W, I in FORMATS:
        s = x * 2.0 ** (W - I)
        assert np.sum(s - np.floor(s) == 0.5) >= 400


def test_product_path(halcon):
    """a) <38, 3> y <39, 4> exactos -> <20, 2> con AC_RND y AC_SAT."""
    inputs, data = halcon
    a1, b1, a2, b2 = inputs["product"].T
    p1, p2 = a1 * b1, a2 * b2  # enteros con F = 35, exactos en int64 y en float64 (< 2^53)
    got = _read(data / "product_out.txt")
    for col, k in enumerate((p1, p1 + p2, p1 - p2)):
        expected = to_int(k * 2.0**-F_P, D_W, D_I, rounding="half_up", saturate=True)
        assert np.array_equal(got[:, col], expected), _mismatch(got[:, col], expected, k)
    # cobertura: empates exactos (medio LSB de <20, 2>) y saturación
    half = 2 ** (F_P - F_D - 1)
    assert np.sum(np.mod(p1, 2 * half) == half) >= 900
    assert np.sum(np.mod(p1 + p2, 2 * half) == half) >= 900
    assert got[-6, 1] == 2 ** (D_W - 1) - 1  # 4 satura a 2 - q


def test_butterfly_path(halcon):
    """b) a ± b en <21, 3>, s = 0 o 1 como cambio de escala -> <20, 2> con AC_RND_CONV y AC_SAT."""
    inputs, data = halcon
    a, b = inputs["butterfly"].T
    got = _read(data / "butterfly_out.txt")
    for i, k in enumerate((a + b, a - b)):
        for s in (0, 1):
            expected = to_int(k * 2.0 ** -(F_D + s), D_W, D_I, rounding="half_even", saturate=True)
            col = 2 * s + i
            assert np.array_equal(got[:, col], expected), _mismatch(got[:, col], expected, k)
    assert np.sum(np.mod(a + b, 2) == 1) >= 1000  # empates exactos con s = 1


def test_butterfly_shift_operator_truncates(halcon):
    """``>> 1`` de ac_fixed conserva el tipo y trunca: es AC_TRN, no AC_RND_CONV."""
    inputs, data = halcon
    a, b = inputs["butterfly"].T
    got = _read(data / "butterfly_out.txt")
    for i, k in enumerate((a + b, a - b)):
        expected = to_int(k * 2.0 ** -(F_D + 1), D_W, D_I, rounding="trunc", saturate=True)
        assert np.array_equal(got[:, 4 + i], expected), _mismatch(got[:, 4 + i], expected, k)
    assert np.any(got[:, 4] != got[:, 2])  # difiere del redondeo a par en los empates


def test_negation_saturates(halcon):
    """c) -x de <20, 2> exacto en <21, 3> y a <20, 2>: -(-2) satura a 2 - q."""
    inputs, data = halcon
    k = inputs["negation"]
    got = _read(data / "negation_out.txt")[:, 0]
    expected = to_int(-k * 2.0**-F_D, D_W, D_I, rounding="half_even", saturate=True)
    assert np.array_equal(got, expected), _mismatch(got, expected, k)
    assert got[k == -(2 ** (D_W - 1))][0] == 2 ** (D_W - 1) - 1
