"""Genera un gráfico detallado de la evolución temporal de theta[n] e incrementos delta_theta[n]
en el proceso de Wiener del ruido de fase del láser.

Uso:
    python -m scripts.plot_phase_noise_trajectory
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from emulador.blocks.phase_noise import PhaseNoise


def generate_phase_trajectory_plot(out_dirs: list[Path]):
    linewidth_hz = 100e3  # 100 kHz (valor típico de láser coherente ECL)
    fs = 2.0e9            # 2 GHz
    ts = 1.0 / fs         # 0.5 ns
    n_samples = 3000      # 1.5 microsegundos de observación
    t_ns = np.arange(n_samples) * ts * 1e9  # tiempo en nanosegundos

    sigma_increment = np.sqrt(2.0 * np.pi * linewidth_hz * ts)

    # 1. Simular un ensamble de realizaciones
    n_runs = 600
    phases = np.zeros((n_runs, n_samples))
    increments_sample = None

    for i in range(n_runs):
        rng = np.random.default_rng(seed=i)
        d_theta = rng.normal(loc=0.0, scale=sigma_increment, size=n_samples)
        if i == 0:
            increments_sample = d_theta
        phases[i, :] = np.cumsum(d_theta)

    # Varianza teórica y empírica
    t_sec = np.arange(n_samples) * ts
    sigma_theor_t = np.sqrt(2.0 * np.pi * linewidth_hz * t_sec)
    var_emp = np.var(phases, axis=0)
    var_theor = 2.0 * np.pi * linewidth_hz * t_sec

    # 2. Construir figura de 3 paneles
    fig = plt.figure(figsize=(15, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0])

    # Panel 1: Trayectorias temporales de theta(t) con conos de confianza
    ax1 = fig.add_subplot(gs[0, 0])
    # Graficar banda de confianza +/- 2 sigma
    ax1.fill_between(t_ns, -2 * sigma_theor_t, 2 * sigma_theor_t, color="#9467bd", alpha=0.15, label="Banda $\\pm 2\\sigma(t)$")
    ax1.plot(t_ns, 2 * sigma_theor_t, "--", color="#9467bd", lw=1.3)
    ax1.plot(t_ns, -2 * sigma_theor_t, "--", color="#9467bd", lw=1.3)

    # Graficar 6 realizaciones individuales
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#e377c2"]
    for k in range(6):
        ax1.plot(t_ns, phases[k, :], lw=1.4, color=colors[k], alpha=0.9, label=f"Corrida {k+1}" if k < 3 else None)

    ax1.set_title("Evolución Temporal de la Fase Acumulada $\\theta(t)$", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Tiempo $t$ (ns)", fontsize=10)
    ax1.set_ylabel("Fase acumulada $\\theta(t)$ (rad)", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Panel 2: Incrementos paso a paso Delta_theta[n] e Histograma
    ax2 = fig.add_subplot(gs[0, 1])
    # Histograma de los incrementos de una corrida
    count, bins, _ = ax2.hist(increments_sample, bins=45, density=True, alpha=0.65, color="#1f77b4", edgecolor="white", label="Histograma $\\Delta\\theta[n]$")
    # PDF Gaussiana teórica N(0, sigma^2)
    x_axis = np.linspace(-4 * sigma_increment, 4 * sigma_increment, 200)
    gauss_pdf = (1.0 / (sigma_increment * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_axis / sigma_increment) ** 2)
    ax2.plot(x_axis, gauss_pdf, "r-", lw=2, label="PDF $\\mathcal{N}(0, \\sigma_\\Delta^2)$")
    ax2.set_title("Distribución de Incrementos $\\Delta\\theta[n]$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Incremento de fase $\\Delta\\theta$ (rad)", fontsize=10)
    ax2.set_ylabel("Densidad de probabilidad", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)
    # Formato científico en x
    ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Panel 3: Crecimiento Lineal de la Varianza Var{theta(t)}
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t_ns, var_emp, lw=2, color="#2ca02c", label="Varianza empírica ($M=600$)")
    ax3.plot(t_ns, var_theor, "--", lw=1.8, color="#d62728", label="Ley de Wiener $2\\pi\\Delta\\nu t$")
    ax3.set_title("Crecimiento de Varianza $\\mathrm{Var}\\{\\theta(t)\\}$", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Tiempo $t$ (ns)", fontsize=10)
    ax3.set_ylabel("Varianza $\\mathrm{Var}\\{\\theta(t)\\}$ (rad$^2$)", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(f"Dinámica Estocástica del Ruido de Fase del Láser (Proceso de Wiener: $\\Delta\\nu = {linewidth_hz/1e3:.0f}\\text{{ kHz}}$, $f_s = {fs/1e9:.1f}\\text{{ GHz}}$)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "trayectoria_ruido_fase.png", dpi=140)
    plt.close(fig)
    print("  -> Guardado trayectoria_ruido_fase.png")


def main():
    root = Path(__file__).resolve().parent.parent
    dir1 = root / "outputs"
    dir2 = root.parent / "documentacion" / "figuras"
    generate_phase_trajectory_plot([dir1, dir2])


if __name__ == "__main__":
    main()
