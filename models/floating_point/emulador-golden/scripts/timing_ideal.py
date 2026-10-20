"""Retardo fraccionario ideal (FFT): qué le hace a la señal y al EVM.

Guarda una figura con tres paneles:
  1. Señal en el tiempo: las muestras retardadas y[n] caen sobre la curva continua
     de la entrada en t = n - mu.
  2. Constelación recibida con ese retardo (receptor que muestrea en el instante
     óptimo SIN retardo).
  3. EVM contra mu.

Uso (desde la raíz del repo):
    python -m scripts.timing_ideal
    python -m scripts.timing_ideal --mu 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample

from emulador.common import evm_percent, gen_signal, recover_symbols
from emulador.timing import ideal_delay


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mu", type=float, default=0.5, help="retardo, en muestras")
    ap.add_argument("--mod", default="qam16", choices=["pam4", "qam16"])
    ap.add_argument("--n-symbols", type=int, default=4096)
    ap.add_argument("--out", default="outputs", help="carpeta de salida para el gráfico")
    args = ap.parse_args()

    sig = gen_signal(args.mod, args.n_symbols, sps=2)
    x = sig.samples
    y = ideal_delay(x, args.mu)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

    # 1. Tiempo: curva continua de x (sobremuestreada x16) y muestras de y en n
    up, n0, n1 = 16, 200, 230
    x_cont = resample(x, len(x) * up)
    t_cont = np.arange(len(x_cont)) / up
    sel = (t_cont >= n0 - 1) & (t_cont <= n1)
    n = np.arange(n0, n1)
    axs[0].plot(t_cont[sel], x_cont[sel].real, color="0.6", lw=1, label="x(t) continua")
    axs[0].plot(n, x[n].real, "o", label="x[n]")
    axs[0].plot(n - args.mu, y[n].real, "s", mfc="none", label=f"y[n] = x(n - {args.mu:g}), dibujada en n - {args.mu:g}")
    axs[0].set_title("Parte real en el tiempo")
    axs[0].set_xlabel("t [muestras]")
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=8)

    # 2. Constelación con el retardo
    rx = recover_symbols(y, sig)
    evm = evm_percent(rx, sig.symbols)
    axs[1].plot(rx.real[32:-32], rx.imag[32:-32], ".", ms=2, alpha=0.5)
    axs[1].set_title(f"Constelación con mu = {args.mu:g} (EVM {evm:.2f} %)")
    axs[1].set_xlabel("I")
    axs[1].set_ylabel("Q")
    axs[1].set_aspect("equal")
    axs[1].grid(alpha=0.3)

    # 3. EVM contra mu
    mus = np.linspace(0, 2, 41)
    evms = [evm_percent(recover_symbols(ideal_delay(x, m), sig), sig.symbols) for m in mus]
    axs[2].plot(mus, evms, ".-")
    axs[2].axvline(args.mu, color="0.5", ls="--")
    axs[2].set_title("EVM contra retardo (sps = 2: mu = 2 es un símbolo)")
    axs[2].set_xlabel("mu [muestras]")
    axs[2].set_ylabel("EVM [%]")
    axs[2].grid(alpha=0.3)

    fig.suptitle("Error de temporización: retardo ideal (FFT)")
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "timing_ideal.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    print(f"mu = {args.mu:g}: EVM = {evm:.3f} %")
    print(f"Gráfico guardado en {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
