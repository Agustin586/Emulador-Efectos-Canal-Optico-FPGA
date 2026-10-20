"""Compara los métodos de retardo fraccionario: ideal (FFT), lineal y cúbico.

Guarda una figura con cuatro paneles:
  1. |H(w)| de cada método para un mu dado (el ideal vale 1 en todas las frecuencias).
  2. Retardo de grupo de cada método (el ideal vale mu en todas las frecuencias).
  3. Error de cada interpolador contra el ideal, en dB, en función de mu.
  4. EVM después del receptor, en función de mu.
La zona sombreada en 1 y 2 es la banda que ocupa la señal.

Uso (desde la raíz del repo):
    python -m scripts.timing_compare
    python -m scripts.timing_compare --mu 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana
import matplotlib.pyplot as plt
import numpy as np

from emulador.common import evm_percent, gen_signal, recover_symbols
from emulador.timing import cubic_delay, cubic_taps, delay_error_db, ideal_delay, linear_delay

METHODS = {"ideal": ideal_delay, "lineal": linear_delay, "cúbico": cubic_delay}
COLORS = {"ideal": "C0", "lineal": "C1", "cúbico": "C2"}


def freq_response(method: str, mu: float, w: np.ndarray) -> np.ndarray:
    """H(w) teórica de cada método, w en rad/muestra."""
    if method == "ideal":
        return np.exp(-1j * w * mu)
    if method == "lineal":
        return (1 - mu) + mu * np.exp(-1j * w)
    return sum(h * np.exp(-1j * w * d) for h, d in zip(cubic_taps(mu), (-1, 0, 1, 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mu", type=float, default=0.25, help="retardo para los paneles 1 y 2, en muestras")
    ap.add_argument("--mod", default="qam16", choices=["pam4", "qam16"])
    ap.add_argument("--n-symbols", type=int, default=4096)
    ap.add_argument("--out", default="outputs", help="carpeta de salida para el gráfico")
    args = ap.parse_args()

    sig = gen_signal(args.mod, args.n_symbols, sps=2)
    x = sig.samples
    band = (1 + sig.beta) / sig.sps  # borde de la banda de la señal, en unidades de pi rad/muestra

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    w = np.linspace(1e-3, np.pi, 1000)

    # 1 y 2. |H(w)| y retardo de grupo
    for method in METHODS:
        H = freq_response(method, args.mu, w)
        axs[0, 0].plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-12), color=COLORS[method], label=method)
        axs[0, 1].plot(w / np.pi, -np.gradient(np.unwrap(np.angle(H)), w), color=COLORS[method], label=method)
    axs[0, 0].set_ylim(-20, 2)
    axs[0, 1].set_ylim(-0.25, 1.25)
    axs[0, 0].set_title(f"|H(w)| con mu = {args.mu:g}")
    axs[0, 0].set_ylabel("dB")
    axs[0, 1].set_title(f"Retardo de grupo con mu = {args.mu:g}")
    axs[0, 1].set_ylabel("muestras")
    for ax in axs[0]:
        ax.axvspan(0, band, color="0.9", zorder=0, label="banda de la señal")
        ax.set_xlabel("w / pi  (1 = Nyquist)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # 3 y 4. Error contra el ideal y EVM, barriendo mu
    mus = np.linspace(0, 1, 21)
    refs = {mu: ideal_delay(x, mu) for mu in mus}
    for method, func in METHODS.items():
        ys = {mu: func(x, mu) for mu in mus}
        if method != "ideal":
            errs = [delay_error_db(ys[mu], refs[mu]) if mu not in (0, 1) else np.nan for mu in mus]
            axs[1, 0].plot(mus, errs, ".-", color=COLORS[method], label=method)
        evms = [evm_percent(recover_symbols(ys[mu], sig), sig.symbols) for mu in mus]
        axs[1, 1].plot(mus, evms, ".-", color=COLORS[method], label=method)
    axs[1, 0].set_title("Error del interpolador contra el ideal")
    axs[1, 0].set_ylabel("dB (más abajo = mejor)")
    axs[1, 1].set_title("EVM después del receptor")
    axs[1, 1].set_ylabel("EVM [%]")
    for ax in axs[1]:
        ax.set_xlabel("mu [muestras]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Error de temporización: ideal vs. lineal vs. cúbico ({args.mod}, sps = {sig.sps})")
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "timing_compare.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    print(f"Gráfico guardado en {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
