"""Corre un bloque sobre una señal de prueba y guarda un gráfico de validación.

Uso (desde la raíz del repo):
    python -m scripts.run_block phase_offset --param delta_theta_rad=0.3
    python -m scripts.run_block phase_offset --param delta_theta_rad=0.3 --mod pam4

El gráfico muestra la constelación antes y después del bloque, y el espectro.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: funciona dentro del contenedor Docker
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from emulador.blocks import REGISTRY
from emulador.common import evm_percent, gen_signal, recover_symbols


def parse_params(pairs: list[str]) -> dict:
    """Convierte ['a=1', 'b=0.5'] en {'a': 1, 'b': 0.5}."""
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def plot_before_after(sig, y: np.ndarray, title: str, path: Path) -> tuple[float, float]:
    """Guarda constelación y espectro (entrada vs salida). Devuelve (EVM entrada, EVM salida).

    Con dos polarizaciones (2, n) grafica la polarización X.
    """
    y_x = y[0] if y.ndim == 2 else y
    rx_in = recover_symbols(sig.samples, sig)
    rx_out = recover_symbols(y_x, sig)
    evm_in = evm_percent(rx_in, sig.symbols)
    evm_out = evm_percent(rx_out, sig.symbols)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, rx, label in ((axs[0], rx_in, f"Entrada (EVM {evm_in:.2f} %)"), (axs[1], rx_out, f"Salida (EVM {evm_out:.2f} %)")):
        ax.plot(rx.real[32:-32], rx.imag[32:-32], ".", ms=2, alpha=0.5)
        ax.set_title(label)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    for x_sig, lab in ((sig.samples, "entrada"), (y_x, "salida")):
        f, p = welch(x_sig, fs=sig.sps, nperseg=1024, return_onesided=False)
        order = np.argsort(f)
        axs[2].plot(f[order], 10 * np.log10(p[order] + 1e-20), label=lab)
    axs[2].set_title("Espectro (f normalizada a Rs)")
    axs[2].set_xlabel("f / Rs")
    axs[2].set_ylabel("dB")
    axs[2].grid(alpha=0.3)
    axs[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return evm_in, evm_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("block", choices=sorted(REGISTRY))
    ap.add_argument("--param", action="append", default=[], metavar="clave=valor")
    ap.add_argument("--mod", default="qam16", choices=["pam4", "qam16"])
    ap.add_argument("--n-symbols", type=int, default=4096)
    ap.add_argument("--sps", type=int, default=2)
    ap.add_argument("--out", default="outputs", help="carpeta de salida para el gráfico")
    args = ap.parse_args()

    sig = gen_signal(args.mod, args.n_symbols, args.sps)
    x = np.stack([sig.samples, sig.samples]) if args.block == "pmd" else sig.samples  # PMD: dos polarizaciones

    block = REGISTRY[args.block](**parse_params(args.param))
    try:
        y = block(x)
    except NotImplementedError as exc:
        print(f"[pendiente] {exc}")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.block}_validacion.png"
    evm_in, evm_out = plot_before_after(sig, y, f"{args.block} {block}", path)
    print(f"{args.block}: EVM entrada = {evm_in:.3f} %  ->  EVM salida = {evm_out:.3f} %")
    print(f"Gráfico guardado en {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
