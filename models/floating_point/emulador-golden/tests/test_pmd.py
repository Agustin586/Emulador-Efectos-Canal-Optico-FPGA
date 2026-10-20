"""Tests propios del bloque PMD (además de los de test_contract.py)."""
import numpy as np
import pytest
from scipy import stats

from emulador.blocks import PMD
from emulador.common import gen_signal
from emulador.pmd import dgd_delay, pmd_section, random_angles, rotation, total_dgd

FS = 64e9  # 32 GBd x 2 sps
TAU = 10e-12  # 10 ps, un DGD de prueba
ANGLES = [(0.0, 0.0), (0.3, 0.0), (np.pi / 4, 0.0), (2.0, 0.0), (0.7, 1.1), (1.2, 4.0)]  # (theta, phi)


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
@pytest.mark.parametrize("theta, phi", ANGLES)
def test_rotacion_es_unitaria(theta, phi):
    R = rotation(theta, phi)
    assert np.allclose(R @ R.conj().T, np.eye(2))


def test_rotacion_con_phi_cero_es_la_rotacion_real():
    c, s = np.cos(0.7), np.sin(0.7)
    assert np.allclose(rotation(0.7), [[c, -s], [s, c]])


def test_sin_rotacion_es_igual_al_paso_1():
    x = _dual_pol()
    assert np.allclose(pmd_section(x, FS, TAU, theta=0.0), dgd_delay(x, FS, TAU))


@pytest.mark.parametrize("theta, phi", ANGLES)
def test_sin_dgd_la_salida_es_la_entrada(theta, phi):
    """Con tau = 0 queda R @ R^H = identidad, sea cual sea la orientación."""
    x = _dual_pol()
    assert np.allclose(pmd_section(x, FS, 0.0, theta, phi), x)


@pytest.mark.parametrize("theta, phi", ANGLES)
def test_conserva_la_potencia_total(theta, phi):
    """J es unitaria: la potencia de X + Y no cambia (aunque se reparta distinto)."""
    x = _dual_pol()
    y = pmd_section(x, FS, TAU, theta, phi)
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
def _angles(seed, k):
    """Los mismos ángulos (thetas, phis) que sortea el bloque PMD con esa semilla."""
    return random_angles(np.random.default_rng(seed), k)


def test_cascada_conserva_la_potencia_total():
    """Cada sección es unitaria, así que la cascada también."""
    x = _dual_pol()
    y = PMD(fs=FS, dgd_ps=10, n_sections=10, seed=0)(x)
    assert np.isclose(_power(y), _power(x))


def test_cascada_de_una_seccion_es_pmd_section():
    """Con n_sections=1 el bloque aplica una sola sección con los primeros ángulos."""
    x = _dual_pol()
    y = PMD(fs=FS, dgd_ps=10, n_sections=1, seed=5)(x)
    thetas, phis = _angles(seed=5, k=1)
    assert np.allclose(y, pmd_section(x, FS, TAU, thetas[0], phis[0]))


def test_cascada_se_deshace_aplicando_la_inversa():
    """J_total^H deshace el PMD: secciones en orden inverso y con -tau.

    La inversa de una sección R @ D(tau) @ R^H es R @ D(-tau) @ R^H, y la de
    J_K @ ... @ J_1 es J_1^H @ ... @ J_K^H (orden inverso).
    """
    x = _dual_pol()
    k, seed = 10, 7
    y = PMD(fs=FS, dgd_ps=10, n_sections=k, seed=seed)(x)
    thetas, phis = _angles(seed, k)
    for theta, phi in zip(thetas[::-1], phis[::-1]):
        y = pmd_section(y, FS, -TAU, theta, phi)
    assert np.allclose(y, x)


# --------------------------------------------------------------------------
# DGD total y estadística Maxwelliana
# --------------------------------------------------------------------------
def test_dgd_de_una_seccion_es_tau():
    """Una sola sección: el DGD total es el de la sección, sea cual sea la orientación."""
    assert np.isclose(total_dgd(TAU, [0.7], [1.1]), TAU, rtol=1e-6)


def test_dgd_con_ejes_alineados_se_suma():
    """K secciones con los mismos ejes: los retardos se suman, DGD total = K * tau."""
    k = 8
    assert np.isclose(total_dgd(TAU, np.zeros(k), np.zeros(k)), k * TAU, rtol=1e-6)


def test_dgd_total_del_bloque_es_maxwelliano():
    """Muchas fibras (semillas): el DGD total sigue una Maxwelliana.

    Teoría (K secciones de DGD tau, ejes al azar): <DGD^2> = K tau^2, que es una
    Maxwelliana de escala a = tau*sqrt(K/3) y media tau*sqrt(8K/(3*pi)).
    Se compara con un test de Kolmogorov-Smirnov y contra la media teórica.
    """
    k, dgd_ps, n_fibras = 20, 1.0, 2000
    dgd = np.array([PMD(dgd_ps=dgd_ps, n_sections=k, seed=s).total_dgd_ps() for s in range(n_fibras)])

    maxwell = stats.maxwell(scale=dgd_ps * np.sqrt(k / 3))
    assert stats.kstest(dgd, maxwell.cdf).pvalue > 0.01
    assert np.isclose(dgd.mean(), dgd_ps * np.sqrt(8 * k / (3 * np.pi)), rtol=0.03)


def test_solo_theta_no_da_maxwelliana():
    """Por qué hace falta phi: con ejes solo en un plano (phi = 0, theta uniforme)
    el DGD total es una caminata 2D y sale Rayleigh.

    Se distingue por <DGD^2>/<DGD>^2: Maxwell = 3*pi/8 = 1.178, Rayleigh = 4/pi = 1.273.
    """
    k, n_fibras = 20, 2000
    rng = np.random.default_rng(0)
    dgd = np.array([total_dgd(TAU, rng.uniform(0, 2 * np.pi, k), np.zeros(k)) for _ in range(n_fibras)])
    ratio = np.mean(dgd**2) / dgd.mean() ** 2
    assert ratio > 1.22  # lejos de Maxwell (1.178), cerca de Rayleigh (1.273)


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
