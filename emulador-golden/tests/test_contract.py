"""Contrato que cumplen TODOS los bloques.

Un bloque todavía no implementado se marca como "skipped" (no como error): así
la lista de tests muestra de un vistazo qué bloques ya están listos.
"""
import numpy as np
import pytest

from emulador.blocks import REGISTRY
from emulador.common import gen_signal


def _input_for(name):
    x = gen_signal("qam16", 1024, sps=2).samples
    return np.stack([x, x]) if name == "pmd" else x


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_bypass_returns_input_untouched(name):
    x = _input_for(name)
    block = REGISTRY[name](enabled=False)
    assert block(x) is x or np.array_equal(block(x), x)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_shape_and_dtype_preserved(name):
    x = _input_for(name)
    try:
        y = REGISTRY[name]()(x)
    except NotImplementedError:
        pytest.skip(f"{name}: todavía no implementado")
    assert y.shape == x.shape
    assert np.iscomplexobj(y)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_default_parameters_give_negligible_effect(name):
    """Con los parámetros por defecto el bloque no debería cambiar la señal.

    Excepción razonable: bloques que por definición agregan ruido (awgn);
    quien lo implemente puede ajustar este test para su caso.
    """
    if name == "awgn":
        pytest.skip("awgn: por defecto agrega ruido; definir el test propio")
    x = _input_for(name)
    try:
        y = REGISTRY[name]()(x)
    except NotImplementedError:
        pytest.skip(f"{name}: todavía no implementado")
    assert np.allclose(y, x, atol=1e-6)
