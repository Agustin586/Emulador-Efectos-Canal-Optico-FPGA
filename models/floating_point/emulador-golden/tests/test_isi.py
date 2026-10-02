"""ISI genérica (Parte A de GUIA_ISI_CD.md)."""
import numpy as np
import pytest

from emulador.blocks import ISI
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


# --------------------------------------------------------------------------
# Bloque ISI con taps (A1.2)
# --------------------------------------------------------------------------
def test_block_default_is_bypass():
    x = _crandn(np.random.default_rng(3), 256)
    blk = ISI()
    assert np.array_equal(blk(x), x)
    assert np.array_equal(blk.process_hw(x), x)
    assert blk.delay_samples == 0


def test_block_impulse_hw_gives_taps():
    h = _crandn(np.random.default_rng(4), 9)
    y = ISI(taps=h).process_hw(impulse(30))
    assert np.array_equal(y[:9], h)


def test_block_delay_and_alignment():
    h = np.array([0.1, 0.2, 1.0, 0.3])
    blk = ISI(taps=h)
    assert blk.delay_samples == 2
    x = _crandn(np.random.default_rng(5), 200)
    y, y_hw = blk(x), blk.process_hw(x)
    assert y.shape == y_hw.shape == x.shape
    assert np.array_equal(y[:-2], y_hw[2:])
    # la cola alineada sale de seguir filtrando con ceros a la entrada
    assert np.array_equal(y[-2:], isi_apply(np.concatenate([x, np.zeros(2)]), h)[-2:])


def test_block_impulse_aligned_puts_cursor_at_zero():
    h = np.array([0.1, 0.2, 1.0, 0.3])
    y = ISI(taps=h)(impulse(10, 5))
    assert np.array_equal(y[3:7], h)  # el cursor (k0 = 2) cae en la posición 5


def test_block_two_polarizations():
    rng = np.random.default_rng(6)
    blk = ISI(taps=_crandn(rng, 16))
    x = _crandn(rng, 2, 300)
    y, y_hw = blk(x), blk.process_hw(x)
    for p in range(2):
        assert np.array_equal(y[p], blk(x[p]))
        assert np.array_equal(y_hw[p], blk.process_hw(x[p]))


def test_block_coefficients():
    h = [1, 0.3j]
    c = ISI(taps=h).coefficients()
    assert set(c) == {"h"}
    assert c["h"].dtype == complex
    assert np.array_equal(c["h"], np.array(h, dtype=complex))


@pytest.mark.parametrize("taps", [np.ones(33), np.array([]), np.ones((2, 4))])
def test_block_rejects_invalid_taps(taps):
    with pytest.raises(ValueError):
        ISI(taps=taps)


def test_block_accepts_K32():
    assert ISI(taps=np.ones(32)).coefficients()["h"].size == 32


def test_block_rejects_unknown_profile():
    with pytest.raises(ValueError):
        ISI(profile="nope")
