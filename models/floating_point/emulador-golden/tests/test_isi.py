"""ISI genérica (Parte A de GUIA_ISI_CD.md)."""
import numpy as np
import pytest

from emulador.isi import isi_apply, isi_apply_stream
from emulador.stimuli import impulse

TOL = 1e-12


def _crandn(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# --------------------------------------------------------------------------
# fir: isi_apply, isi_apply_stream (A1.1)
# --------------------------------------------------------------------------
def test_impulse_gives_taps():
    h = _crandn(np.random.default_rng(0), 7)
    y = isi_apply(impulse(20), h)
    assert np.array_equal(y[:7], h)
    assert np.array_equal(y[7:], np.zeros(13))


@pytest.mark.parametrize("K", [1, 7, 16, 32])
def test_matches_numpy_convolve(K):
    rng = np.random.default_rng(K)
    h = _crandn(rng, K)
    x = _crandn(rng, 500)
    y = isi_apply(x, h)
    assert y.shape == x.shape
    assert np.max(np.abs(y - np.convolve(x, h)[: x.size])) < TOL


def test_linearity():
    rng = np.random.default_rng(1)
    h = _crandn(rng, 16)
    x1, x2 = _crandn(rng, 300), _crandn(rng, 300)
    a, b = 0.7 - 1.3j, -2.1 + 0.4j
    lhs = isi_apply(a * x1 + b * x2, h)
    rhs = a * isi_apply(x1, h) + b * isi_apply(x2, h)
    assert np.max(np.abs(lhs - rhs)) < TOL


def test_two_polarizations():
    rng = np.random.default_rng(2)
    h = _crandn(rng, 16)
    x = _crandn(rng, 2, 400)
    y = isi_apply(x, h)
    assert y.shape == (2, 400)
    assert np.array_equal(y[0], isi_apply(x[0], h))
    assert np.array_equal(y[1], isi_apply(x[1], h))


@pytest.mark.parametrize("K", [1, 16, 32])
@pytest.mark.parametrize("shape", [(1000,), (2, 1000)])
def test_stream_equals_full(K, shape):
    rng = np.random.default_rng(K)
    h = _crandn(rng, K)
    x = _crandn(rng, *shape)
    # trozos de largos distintos, incluidos algunos más cortos que K - 1 y
    # uno vacío al final
    cuts = [3, 10, 11, 100, 337, 340, 700, 1000]
    chunks = np.split(x, cuts, axis=-1)
    y = isi_apply_stream(chunks, h)
    assert len(y) == len(chunks)
    assert all(yc.shape == xc.shape for yc, xc in zip(y, chunks))
    full = isi_apply(x, h)
    assert np.max(np.abs(np.concatenate(y, axis=-1) - full)) < TOL


def test_rejects_empty_taps():
    with pytest.raises(ValueError):
        isi_apply(np.ones(4, complex), np.array([]))
