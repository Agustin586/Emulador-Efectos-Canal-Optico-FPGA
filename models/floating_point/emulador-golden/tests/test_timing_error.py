"""Tests propios del bloque de error de temporización (además de los de test_contract.py)."""
import numpy as np
import pytest

from emulador.blocks import TimingError
from emulador.common import gen_signal
from emulador.timing import cubic_delay, cubic_taps, delay_error_db, ideal_delay, linear_delay

MUS = [0.1, 0.25, 0.5, 0.75, 0.9]


def _signal(n_symbols=1024, seed=0):
    return gen_signal("qam16", n_symbols, sps=2, seed=seed).samples


def _power(x):
    return np.sum(np.abs(x) ** 2)


# --------------------------------------------------------------------------
# Paso 1: retardo ideal con FFT
# --------------------------------------------------------------------------
def test_ideal_sin_retardo_es_la_entrada():
    x = _signal()
    assert np.allclose(ideal_delay(x, 0.0), x)


@pytest.mark.parametrize("m", [1, 2, 5, -3])
def test_ideal_con_retardo_entero_es_np_roll(m):
    """mu entero: corre m muestras. La FFT es circular, por eso la comparación es contra np.roll."""
    x = _signal()
    assert np.allclose(ideal_delay(x, m), np.roll(x, m))


@pytest.mark.parametrize("mu", MUS)
def test_ideal_conserva_la_potencia(mu):
    x = _signal()
    assert np.isclose(_power(ideal_delay(x, mu)), _power(x))


@pytest.mark.parametrize("mu", MUS)
def test_ideal_se_deshace_con_el_retardo_opuesto(mu):
    x = _signal()
    assert np.allclose(ideal_delay(ideal_delay(x, mu), -mu), x)


@pytest.mark.parametrize("mu", MUS)
def test_ideal_dos_retardos_se_suman(mu):
    x = _signal()
    assert np.allclose(ideal_delay(ideal_delay(x, mu), 1 - mu), np.roll(x, 1))


@pytest.mark.parametrize("mu", MUS)
def test_ideal_retarda_una_senoide_exacta(mu):
    """Una exponencial compleja periódica en n muestras se retrasa sin error: x(n - mu)."""
    n, k = 256, 7
    t = np.arange(n)
    x = np.exp(2j * np.pi * k * t / n)
    assert np.allclose(ideal_delay(x, mu), np.exp(2j * np.pi * k * (t - mu) / n))


def test_ideal_vale_para_dos_polarizaciones():
    x = np.stack([_signal(seed=0), _signal(seed=1)])
    y = ideal_delay(x, 0.3)
    assert y.shape == x.shape
    assert np.allclose(y[0], ideal_delay(x[0], 0.3))
    assert np.allclose(y[1], ideal_delay(x[1], 0.3))


# --------------------------------------------------------------------------
# Paso 2: interpolador lineal
# --------------------------------------------------------------------------
def _H_linear(w, mu):
    """Respuesta en frecuencia teórica del interpolador lineal."""
    return (1 - mu) + mu * np.exp(-1j * w)


def test_lineal_sin_retardo_es_la_entrada():
    x = _signal()
    assert np.allclose(linear_delay(x, 0.0), x)


def test_lineal_con_mu_uno_es_la_muestra_anterior():
    """mu = 1: y[n] = x[n-1], con un cero en la primera muestra (registro reseteado)."""
    x = _signal()
    y = linear_delay(x, 1.0)
    assert y[0] == 0
    assert np.allclose(y[1:], x[:-1])


@pytest.mark.parametrize("mu", MUS)
def test_lineal_es_exacto_en_una_rampa(mu):
    """Sobre una recta, interpolar con una recta no tiene error: y[n] = x(n - mu)."""
    n = np.arange(100)
    a, b = 0.3 - 0.2j, 1.5
    y = linear_delay(a * n + b, mu)
    assert np.allclose(y[1:], (a * (n - mu) + b)[1:])  # la muestra 0 es el borde


@pytest.mark.parametrize("mu", MUS)
def test_lineal_respuesta_en_frecuencia(mu):
    """Una exponencial compleja de frecuencia w sale multiplicada por H(w)."""
    w = 0.4 * np.pi
    x = np.exp(1j * w * np.arange(256))
    y = linear_delay(x, mu)
    assert np.allclose(y[1:], _H_linear(w, mu) * x[1:])


def test_lineal_coincide_con_el_ideal_en_baja_frecuencia():
    """Con una senoide lenta (muchas muestras por período) la recta aproxima bien."""
    n, k = 1024, 4
    x = np.exp(2j * np.pi * k * np.arange(n) / n)
    assert np.allclose(linear_delay(x, 0.5)[1:], ideal_delay(x, 0.5)[1:], atol=1e-3)


def test_lineal_no_conserva_la_potencia():
    """Es un pasabajos: la señal pierde potencia (el ideal no)."""
    x = _signal()
    assert _power(linear_delay(x, 0.5)) < 0.9 * _power(x)


def test_lineal_la_peor_atenuacion_es_en_mu_medio():
    w = 0.6 * np.pi  # borde de la banda de la señal (sps = 2, beta = 0.2)
    gains = {mu: abs(_H_linear(w, mu)) for mu in [0.1, 0.25, 0.5, 0.75, 0.9]}
    assert min(gains, key=gains.get) == 0.5


@pytest.mark.parametrize("mu", [0.1, 0.25, 0.4])
def test_lineal_error_simetrico_en_mu_y_uno_menos_mu(mu):
    """mu y 1 - mu atenúan lo mismo: misma potencia de salida."""
    x = _signal()
    assert np.isclose(_power(linear_delay(x, mu)), _power(linear_delay(x, 1 - mu)), rtol=1e-3)


@pytest.mark.parametrize("mu", [-0.1, 1.5])
def test_lineal_rechaza_mu_fuera_de_rango(mu):
    with pytest.raises(ValueError):
        linear_delay(_signal(), mu)


def test_lineal_vale_para_dos_polarizaciones():
    x = np.stack([_signal(seed=0), _signal(seed=1)])
    y = linear_delay(x, 0.3)
    assert y.shape == x.shape
    assert np.allclose(y[0], linear_delay(x[0], 0.3))
    assert np.allclose(y[1], linear_delay(x[1], 0.3))


# --------------------------------------------------------------------------
# Paso 3: el bloque elige el método
# --------------------------------------------------------------------------
def test_bloque_por_defecto_es_el_ideal():
    x = _signal()
    assert np.allclose(TimingError(mu=0.3)(x), ideal_delay(x, 0.3))


def test_bloque_rechaza_un_metodo_desconocido():
    with pytest.raises(ValueError):
        TimingError(mu=0.3, method="sinc")(_signal())


@pytest.mark.parametrize("method, func", [("ideal", ideal_delay), ("linear", linear_delay), ("cubic", cubic_delay)])
def test_bloque_usa_cada_metodo(method, func):
    x = _signal()
    assert np.allclose(TimingError(mu=0.3, method=method)(x), func(x, 0.3))


# --------------------------------------------------------------------------
# Paso 4: interpolador cúbico (Lagrange) y comparación contra el ideal
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mu", MUS)
def test_cubico_los_coeficientes_suman_uno(mu):
    """Una señal constante tiene que salir igual: ganancia 1 en continua."""
    assert np.isclose(cubic_taps(mu).sum(), 1)


def test_cubico_sin_retardo_es_la_entrada():
    x = _signal()
    assert np.allclose(cubic_delay(x, 0.0), x)


def test_cubico_con_mu_uno_es_la_muestra_anterior():
    x = _signal()
    y = cubic_delay(x, 1.0)
    assert y[0] == 0
    assert np.allclose(y[1:], x[:-1])


@pytest.mark.parametrize("mu", MUS)
def test_cubico_es_exacto_en_un_polinomio_de_grado_3(mu):
    """Lagrange de 3er orden no tiene error sobre un polinomio de grado <= 3."""
    t = np.arange(100, dtype=float)
    p = lambda t: (0.5 + 0.1j) * t**3 - 2 * t**2 + 3j * t + 1
    y = cubic_delay(p(t), mu)
    assert np.allclose(y[2:-1], p(t - mu)[2:-1])  # bordes: faltan vecinos


@pytest.mark.parametrize("mu", MUS)
def test_cubico_se_equivoca_menos_que_el_lineal(mu):
    x = _signal()
    ref = ideal_delay(x, mu)
    assert delay_error_db(cubic_delay(x, mu), ref) < delay_error_db(linear_delay(x, mu), ref) - 6


@pytest.mark.parametrize("func", [linear_delay, cubic_delay])
def test_el_peor_error_es_en_mu_medio(func):
    x = _signal()
    errores = {mu: delay_error_db(func(x, mu), ideal_delay(x, mu)) for mu in MUS}
    assert max(errores, key=errores.get) == 0.5


@pytest.mark.parametrize("func", [linear_delay, cubic_delay])
@pytest.mark.parametrize("mu", [0.1, 0.25, 0.4])
def test_error_simetrico_en_mu_y_uno_menos_mu(func, mu):
    x = _signal()
    e1 = delay_error_db(func(x, mu), ideal_delay(x, mu))
    e2 = delay_error_db(func(x, 1 - mu), ideal_delay(x, 1 - mu))
    assert abs(e1 - e2) < 0.5  # dB
