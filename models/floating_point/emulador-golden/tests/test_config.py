"""Escenario, unidades y parámetros de hardware (paso P1 de GUIA_ISI_CD.md)."""
import dataclasses

import numpy as np
import pytest

from emulador.config import C, D_si, HardwareParams, S_si, Scenario


def test_speed_of_light_is_exact():
    assert C == 299_792_458.0


def test_D_si_conversion():
    assert np.isclose(D_si(17.0), 1.7e-5, rtol=1e-12, atol=0)


def test_S_si_conversion():
    assert np.isclose(S_si(0.057), 57.0, rtol=1e-12, atol=0)


def test_default_scenario():
    esc = Scenario()
    assert esc.fs == 64e9
    assert esc.Ts == 1 / 64e9
    assert (esc.Rs, esc.sps, esc.wavelength, esc.rolloff) == (32e9, 2, 1550e-9, 0.1)


def test_fs_follows_scenario():
    assert Scenario(Rs=100e9).fs == 200e9
    assert Scenario(sps=16).fs == 512e9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Rs": 500e9},
        {"Rs": 0.5e9},
        {"sps": 3},
        {"wavelength": 1600e-9},
        {"wavelength": 1310e-9},
        {"rolloff": 0.0},
        {"rolloff": 0.8},
    ],
)
def test_scenario_rejects_out_of_range(kwargs):
    with pytest.raises(ValueError):
        Scenario(**kwargs)


def test_scenario_accepts_range_limits():
    Scenario(Rs=1e9, wavelength=1530e-9, rolloff=0.05)
    Scenario(Rs=300e9, wavelength=1565e-9, rolloff=0.5, sps=16)


def test_scenario_is_immutable():
    esc = Scenario()
    with pytest.raises(dataclasses.FrozenInstanceError):
        esc.Rs = 1


def test_scenario_to_dict():
    d = Scenario().to_dict()
    assert d == {"Rs": 32e9, "sps": 2, "wavelength": 1550e-9, "rolloff": 0.1, "fs": 64e9, "Ts": 1 / 64e9}


def test_hardware_params_are_provisional():
    hw = HardwareParams()
    assert hw.provisional
    assert (hw.P, hw.f_clk, hw.N_CD) == (1, None, 2048)


def test_hardware_params_N_CD_values():
    """Decisión 30: N_CD es de síntesis, solo 512 o 2048."""
    assert HardwareParams.N_CD_VALUES == (512, 2048)
    assert HardwareParams(N_CD=512).N_CD == 512
    assert not hasattr(HardwareParams(), "N_max")


@pytest.mark.parametrize(
    "kwargs",
    [{"P": 3}, {"P": 0}, {"N_CD": 1024}, {"N_CD": 128}, {"N_CD": 4096}, {"N_CD": 1000}, {"f_clk": -1.0}],
)
def test_hardware_params_reject_invalid(kwargs):
    with pytest.raises(ValueError):
        HardwareParams(**kwargs)


# --------------------------------------------------------------------------
# B5.3: sps racional y Rs hasta 480 GBd
# --------------------------------------------------------------------------
from fractions import Fraction  # noqa: E402


def test_rational_sps_project_rate():
    """1,2 sps (6/5): la tasa del proyecto; 1.2 se normaliza a Fraction(6, 5)."""
    for sps in (Fraction(6, 5), 1.2):
        esc = Scenario(Rs=60e9, sps=sps)
        assert esc.sps == Fraction(6, 5)
        assert (esc.sps_up, esc.sps_down) == (6, 5)
        assert esc.fs == pytest.approx(72e9, rel=1e-15)


def test_integer_sps_unchanged():
    esc = Scenario(sps=4)
    assert esc.sps == 4 and isinstance(esc.sps, int)
    assert (esc.sps_up, esc.sps_down) == (4, 1)
    assert Scenario(sps=Fraction(2, 1)).sps == 2 and isinstance(Scenario(sps=Fraction(2, 1)).sps, int)


def test_rs_up_to_480_gbd():
    esc = Scenario(Rs=480e9, sps=Fraction(6, 5))
    assert esc.fs == pytest.approx(576e9, rel=1e-15)
    with pytest.raises(ValueError):
        Scenario(Rs=481e9)


@pytest.mark.parametrize("sps", [Fraction(1, 2), 0.9, Fraction(17, 1), Fraction(33, 17), 3, 1.25 + 1e-9, "2"])
def test_rational_sps_rejected(sps):
    with pytest.raises(ValueError):
        Scenario(sps=sps)


def test_to_dict_rational_sps_is_json_ready():
    import json

    d = Scenario(Rs=60e9, sps=Fraction(6, 5)).to_dict()
    assert d["sps"] == 1.2 and (d["sps_up"], d["sps_down"]) == (6, 5)
    json.dumps(d)
