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
    assert (hw.P, hw.f_clk, hw.N_max) == (1, None, 2048)


@pytest.mark.parametrize("kwargs", [{"P": 3}, {"P": 0}, {"N_max": 64}, {"N_max": 1000}, {"f_clk": -1.0}])
def test_hardware_params_reject_invalid(kwargs):
    with pytest.raises(ValueError):
        HardwareParams(**kwargs)
