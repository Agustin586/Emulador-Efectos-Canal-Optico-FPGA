"""Caracterización de la CD (paso B4, PDF §6.2, Fig. 10 y Tabla 17).

Residuo de β₃: el compensador de dispersión (tesis de CDC) suele invertir solo
β₂. Lo que sobrevive es el término de β₃, como ISI residual que tiene que
absorber el ecualizador adaptativo. Se usa la referencia ideal y no el
overlap-save, porque a 200 GBd y 1000 km R ≈ 22 000 muestras.
"""
from __future__ import annotations

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
) -> RxResult:
    """ISI residual tras compensar solo β₂: canal (D, S) y compensación (-D, S = 0).

    - esc: escenario (Rs, sps, λ, roll-off).
    - D: coeficiente de dispersión, s/m².
    - z: distancia, m.
    - S: pendiente de dispersión del canal, s/m³ (con S = 0 es el control: solo
      queda el piso del RRC).
    - n_symbols, seed, mod: estímulo (``comm_signal``).
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
    return recover(back[off : off + x.size], sig)
