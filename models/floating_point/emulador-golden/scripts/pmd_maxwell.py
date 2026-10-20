"""Histograma del DGD total del bloque PMD contra la Maxwelliana teórica.

Simula muchas fibras (una por semilla), mide el DGD total de cada una y guarda
el histograma junto con la curva teórica.

Uso (desde la raíz del repo):
    python -m scripts.pmd_maxwell
    python -m scripts.pmd_maxwell --n-sections 20 --dgd-ps 2 --n-fibras 5000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from emulador.blocks import PMD


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sections", type=int, default=20, help="con K <= 10 la cola se aparta de Maxwell")
    ap.add_argument("--dgd-ps", type=float, default=1.0, help="DGD por sección, en ps")
    ap.add_argument("--n-fibras", type=int, default=3000)
    ap.add_argument("--out", default="outputs", help="carpeta de salida para el gráfico")
    args = ap.parse_args()

    k, tau = args.n_sections, args.dgd_ps
    dgd = np.array([PMD(dgd_ps=tau, n_sections=k, seed=s).total_dgd_ps() for s in range(args.n_fibras)])

    maxwell = stats.maxwell(scale=tau * np.sqrt(k / 3))
    mean_teo = tau * np.sqrt(8 * k / (3 * np.pi))
    pvalue = stats.kstest(dgd, maxwell.cdf).pvalue

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(dgd, bins=50, density=True, alpha=0.6, label=f"simulado ({args.n_fibras} fibras)")
    t = np.linspace(0, dgd.max() * 1.1, 400)
    ax.plot(t, maxwell.pdf(t), lw=2, label="Maxwelliana teórica")
    ax.set_xlabel("DGD total [ps]")
    ax.set_ylabel("densidad de probabilidad")
    ax.set_title(f"PMD: K = {k} secciones de {tau:g} ps  |  media {dgd.mean():.2f} ps (teoría {mean_teo:.2f})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pmd_maxwell.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    print(f"DGD medio simulado = {dgd.mean():.3f} ps  |  teórico = {mean_teo:.3f} ps")
    print(f"Kolmogorov-Smirnov contra Maxwell: p = {pvalue:.3f}  (p > 0.01 -> compatible)")
    print(f"Gráfico guardado en {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
