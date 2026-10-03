"""Tests propios del bloque PMD (además de los de test_contract.py)."""
import numpy as np
import pytest

from emulador.blocks import PMD
from emulador.common import gen_signal
from emulador.pmd import dgd_delay, pmd_section, rotation

FS = 64e9  # 32 GBd x 2 sps
TAU = 10e-12  # 10 ps, un DGD de prueba


def _signal(n_symbols=1024, seed=0):
    return gen_signal("qam16", n_symbols, sps=2, seed=seed).samples


def _dual_pol():
    """Dos polarizaciones con datos independientes (no la misma señal en X e Y)."""
    return np.stack([_signal(seed=0), _signal(seed=1)])


def _power(x):
    return np.sum(np.abs(x) ** 2)


# --------------------------------------------------------------------------
# Paso 1: DGD sin rotación
# --------------------------------------------------------------------------
def test_dgd_de_dos_muestras_corre_una_muestra_cada_polarizacion():
    """Con DGD = 2 muestras, X se adelanta 1 muestra e Y se atrasa 1 muestra.

    La FFT es circular, por eso la comparación es contra np.roll.
    """
    x = _signal()
    tau = 2 / FS  # 31.25 ps
    y = dgd_delay(np.stack([x, x]), FS, tau)
    assert np.allclose(y[0], np.roll(x, -1))  # X adelantada
    assert np.allclose(y[1], np.roll(x, +1))  # Y atrasada


# --------------------------------------------------------------------------
# Paso 2: una sección con rotación
# --------------------------------------------------------------------------
def test_rotacion_es_unitaria():
    R = rotation(0.7)
    assert np.allclose(R @ R.T, np.eye(2))


def test_sin_rotacion_es_igual_al_paso_1():
    x = _dual_pol()
    assert np.allclose(pmd_section(x, FS, TAU, theta=0.0), dgd_delay(x, FS, TAU))


@pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 4, 2.0])
def test_sin_dgd_la_salida_es_la_entrada(theta):
    """Con tau = 0 queda R @ R^H = identidad, sea cual sea el ángulo."""
    x = _dual_pol()
    assert np.allclose(pmd_section(x, FS, 0.0, theta), x)


@pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 4, 2.0])
def test_conserva_la_potencia_total(theta):
    """J es unitaria: la potencia de X + Y no cambia (aunque se reparta distinto)."""
    x = _dual_pol()
    y = pmd_section(x, FS, TAU, theta)
    assert np.isclose(_power(y), _power(x))


def test_con_ejes_alineados_no_hay_mezcla():
    """theta = 0: los ejes de la fibra coinciden con X/Y, la Y vacía sigue vacía."""
    x = np.stack([_signal(), np.zeros(2048, dtype=complex)])
    y = pmd_section(x, FS, TAU, theta=0.0)
    assert np.allclose(y[1], 0)


def test_con_ejes_girados_hay_mezcla_entre_polarizaciones():
    """theta = pi/4 y entrada solo en X: aparece señal en Y. Eso es el PMD."""
    x = np.stack([_signal(), np.zeros(2048, dtype=complex)])
    y = pmd_section(x, FS, TAU, theta=np.pi / 4)
    assert _power(y[1]) > 0.01 * _power(x)


# --------------------------------------------------------------------------
# Paso 3: cascada de K secciones
# --------------------------------------------------------------------------
def _thetas(seed, k):
    """Los mismos ángulos que sortea el bloque PMD con esa semilla."""
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, k)


def test_cascada_conserva_la_potencia_total():
    """Cada sección es unitaria, así que la cascada también."""
    x = _dual_pol()
    y = PMD(fs=FS, dgd_ps=10, n_sections=10, seed=0)(x)
    assert np.isclose(_power(y), _power(x))


def test_cascada_de_una_seccion_es_pmd_section():
    """Con n_sections=1 el bloque aplica una sola sección con el primer ángulo."""
    x = _dual_pol()
    y = PMD(fs=FS, dgd_ps=10, n_sections=1, seed=5)(x)
    theta = _thetas(seed=5, k=1)[0]
    assert np.allclose(y, pmd_section(x, FS, TAU, theta))


def test_cascada_se_deshace_aplicando_la_inversa():
    """J_total^H deshace el PMD: secciones en orden inverso y con -tau.

    La inversa de una sección R @ D(tau) @ R^H es R @ D(-tau) @ R^H, y la de
    J_K @ ... @ J_1 es J_1^H @ ... @ J_K^H (orden inverso).
    """
    x = _dual_pol()
    k, seed = 10, 7
    y = PMD(fs=FS, dgd_ps=10, n_sections=k, seed=seed)(x)
    for theta in _thetas(seed, k)[::-1]:
        y = pmd_section(y, FS, -TAU, theta)
    assert np.allclose(y, x)


# --------------------------------------------------------------------------
# Bloque
# --------------------------------------------------------------------------
def test_bloque_misma_semilla_mismo_resultado():
    x = _dual_pol()
    assert np.array_equal(PMD(dgd_ps=10, seed=3)(x), PMD(dgd_ps=10, seed=3)(x))


def test_bloque_distinta_semilla_distinto_resultado():
    x = _dual_pol()
    assert not np.allclose(PMD(dgd_ps=10, seed=3)(x), PMD(dgd_ps=10, seed=4)(x))


def test_bloque_rechaza_una_sola_polarizacion():
    with pytest.raises(ValueError):
        PMD(dgd_ps=10)(_signal())
