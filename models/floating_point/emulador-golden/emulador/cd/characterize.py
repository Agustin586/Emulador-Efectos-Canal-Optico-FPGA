"""Caracterización de la CD (paso B4, PDF §6.2, Fig. 10 y Tabla 17).

Residuo de β₃: el compensador de dispersión (tesis de CDC) suele invertir solo
β₂. Lo que sobrevive es el término de β₃, como ISI residual que tiene que
absorber el ecualizador adaptativo. Se usa la referencia ideal y no el
overlap-save, porque a 200 GBd y 1000 km R ≈ 22 000 muestras.

Temporización (decisión 23): β₃ agrega un retardo global con parte fraccionaria
(τ(ω) = β₃ z ω²/2 tiene media no nula). Un receptor a 2 sps que solo elige
fases enteras ("grid") lo mezcla con la ISI residual; es como está calculada
la Tabla 17 del PDF. Con "ideal" se busca además el retardo fraccionario
continuo que minimiza la EVM (lo que haría una recuperación de temporización
perfecta) y queda solo la ISI de β₃.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ..config import Scenario
from ..rx import RxResult, recover
from ..stimuli import comm_signal
from .reference import cd_reference, cd_reference_pad


def beta3_residual(
    esc: Scenario,
    D: float,
    z: float,
    S: float,
    n_symbols: int = 4096,
    seed: int = 1,
    mod: str = "qam16",
    timing: str = "grid",
) -> RxResult:
    """ISI residual tras compensar solo β₂: canal (D, S) y compensación (-D, S = 0).

    - esc: escenario (Rs, sps, λ, roll-off).
    - D: coeficiente de dispersión, s/m².
    - z: distancia, m.
    - S: pendiente de dispersión del canal, s/m³ (con S = 0 es el control: solo
      queda el piso del RRC).
    - n_symbols, seed, mod: estímulo (``comm_signal``).
    - timing: "grid" (fases enteras de ``rx.recover``, como la Tabla 17) o
      "ideal" (además, el mejor retardo fraccionario, buscado en un período
      de una muestra con resolución de 1e-4).
    - return: ``rx.recover`` (fase de muestreo, retardo entero y ganancia
      compleja óptimos; EVM con ``evm_opt``, que no supera 100 %).

    Las dos etapas usan ``full=True``, así las colas del chirp no se pierden
    aunque R sea mayor que la señal.
    """
    sig = comm_signal(esc, mod=mod, n_symbols=n_symbols, seed=seed)
    x = sig.samples
    lam = esc.wavelength
    y = cd_reference(x, esc.fs, D, lam, z, S=S, full=True)
    back = cd_reference(y, esc.fs, -D, lam, z, S=0.0, full=True)
    off = cd_reference_pad(esc.fs, D, lam, z) + cd_reference_pad(esc.fs, -D, lam, z)
    rx = back[off : off + x.size]
    if timing == "grid":
        return recover(rx, sig)
    if timing != "ideal":
        raise ValueError(f'beta3_residual: timing tiene que ser "grid" o "ideal": {timing!r}')
    return _recover_ideal_timing(rx, sig)


def _recover_ideal_timing(rx: np.ndarray, sig) -> RxResult:
    """``recover`` después del retardo fraccionario (en muestras) de menor EVM."""
    X = np.fft.fft(rx)
    f = np.fft.fftfreq(rx.size)  # ciclos por muestra

    def at(tau: float) -> RxResult:
        return recover(np.fft.ifft(X * np.exp(2j * np.pi * f * tau)), sig)

    # recover ya absorbe los corrimientos enteros: la EVM es periódica en tau con
    # período 1 muestra, y con SNR alta el mínimo es muy angosto (~0,01 muestras)
    taus = np.linspace(-0.5, 0.5, 101)
    evms = [at(t).evm for t in taus]
    i = int(np.argmin(evms))
    fine = minimize_scalar(
        lambda t: at(t).evm, bounds=(taus[i] - 0.01, taus[i] + 0.01), method="bounded", options={"xatol": 1e-4}
    )
    return at(fine.x) if fine.fun < evms[i] else at(taus[i])
