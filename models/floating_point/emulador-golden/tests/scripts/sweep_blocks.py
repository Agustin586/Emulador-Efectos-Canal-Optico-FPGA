"""Genera barridos paramétricos realistas para cada deterioro y guarda figuras diagnósticas.

Uso (desde emulador-golden):
    python -m scripts.sweep_blocks
"""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from emulador.blocks.awgn import AWGN
from emulador.blocks.phase_noise import PhaseNoise
from emulador.blocks.phase_offset import PhaseOffset
from emulador.common import evm_percent, gen_signal, recover_symbols


def run_phase_offset_sweep(out_dirs: list[Path]):
    print("Ejecutando barrido de Offset de Fase...")
    sig = gen_signal("qam16", n_symbols=4096, sps=2, seed=42)
    x = sig.samples

    angles_const = [0.05, 0.15, 0.30, 0.50]  # rad
    angles_sweep = np.linspace(0.0, np.pi / 4, 30)

    # 1. Calcular EVM a lo largo del barrido
    evms = []
    for ang in angles_sweep:
        y = PhaseOffset(delta_theta_rad=ang)(x)
        rx = recover_symbols(y, sig)
        evms.append(evm_percent(rx, sig.symbols))

    # 2. Generar figura completa
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0])

    # Fila superior: 4 constelaciones
    for i, ang in enumerate(angles_const):
        ax = fig.add_subplot(gs[0, i])
        y = PhaseOffset(delta_theta_rad=ang)(x)
        rx = recover_symbols(y, sig)
        evm = evm_percent(rx, sig.symbols)
        ax.plot(rx.real[32:-32], rx.imag[32:-32], ".", ms=2.5, alpha=0.5, color="#1f77b4")
        ax.set_title(f"$\\Delta\\theta = {ang:.2f}$ rad ({np.degrees(ang):.1f}°)\nEVM: {evm:.2f} %", fontsize=10)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])

    # Fila inferior: Curva EVM vs Delta Theta
    ax_curve = fig.add_subplot(gs[1, :])
    ax_curve.plot(np.degrees(angles_sweep), evms, "o-", color="#d62728", lw=2, ms=4, label="EVM medido (%)")
    # Curva teórica aproximada para rotación pura: EVM = 2 * sin(delta_theta / 2) * 100%
    evm_theor = 2.0 * np.sin(angles_sweep / 2.0) * 100.0
    ax_curve.plot(np.degrees(angles_sweep), evm_theor, "--", color="#2ca02c", lw=1.8, label="Aproximación analítica $2\\sin(\\Delta\\theta/2)$")
    ax_curve.set_title("Curva de EVM vs. Ángulo de Offset de Fase (16-QAM)", fontsize=11, fontweight="bold")
    ax_curve.set_xlabel("Ángulo $\\Delta\\theta$ (grados)", fontsize=10)
    ax_curve.set_ylabel("EVM (%)", fontsize=10)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(fontsize=10)

    fig.suptitle("Barrido Paramétrico: Offset de Fase de Portadora (Carrier Phase Offset)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    for d in out_dirs:
        fig.savefig(d / "sweep_phase_offset.png", dpi=140)
    plt.close(fig)
    print("  -> Guardado sweep_phase_offset.png")


def run_awgn_sweep(out_dirs: list[Path]):
    print("Ejecutando barrido de AWGN...")
    sig = gen_signal("qam16", n_symbols=8192, sps=2, seed=42)
    x = sig.samples

    snrs_const = [12.0, 18.0, 24.0, 30.0]  # dB
    snrs_sweep = np.linspace(10.0, 32.0, 23)

    # 1. Curva EVM vs SNR
    evms = []
    for snr in snrs_sweep:
        y = AWGN(snr_db=snr, seed=123)(x)
        rx = recover_symbols(y, sig)
        evms.append(evm_percent(rx, sig.symbols))

    # 2. Generar figura
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0])

    # Fila superior: 4 constelaciones a distintas SNR
    for i, snr in enumerate(snrs_const):
        ax = fig.add_subplot(gs[0, i])
        y = AWGN(snr_db=snr, seed=123)(x)
        rx = recover_symbols(y, sig)
        evm = evm_percent(rx, sig.symbols)
        ax.plot(rx.real[32:-32], rx.imag[32:-32], ".", ms=2.5, alpha=0.45, color="#2ca02c")
        ax.set_title(f"SNR = {snr:.1f} dB\nEVM: {evm:.2f} %", fontsize=10)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_xlim([-1.4, 1.4])
        ax.set_ylim([-1.4, 1.4])

    # Fila inferior izquierda: EVM vs SNR
    ax_evm = fig.add_subplot(gs[1, :2])
    ax_evm.semilogy(snrs_sweep, evms, "s-", color="#1f77b4", lw=2, ms=4, label="EVM medido (%)")
    # Curva teórica en canal AWGN: EVM ≈ 10^(-SNR_sample/20) * 100% * sqrt(sps) corregido por filtro
    evm_theor = (10.0 ** (-snrs_sweep / 20.0)) * 100.0
    ax_evm.semilogy(snrs_sweep, evm_theor, "--", color="#d62728", lw=1.8, label="Teoría $10^{-\\mathrm{SNR}/20} \\times 100\\%$")
    ax_evm.set_title("Curva de EVM vs. SNR (dB)", fontsize=11, fontweight="bold")
    ax_evm.set_xlabel("SNR (dB)", fontsize=10)
    ax_evm.set_ylabel("EVM (%) [escala log]", fontsize=10)
    ax_evm.grid(True, which="both", alpha=0.3)
    ax_evm.legend(fontsize=9.5)

    # Fila inferior derecha: PSD comparativa
    ax_psd = fig.add_subplot(gs[1, 2:])
    f_in, p_in = welch(x, fs=sig.sps, nperseg=1024, return_onesided=False)
    order = np.argsort(f_in)
    ax_psd.plot(f_in[order], 10 * np.log10(p_in[order] + 1e-20), color="black", lw=1.5, label="TX (ideal)")

    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    for snr, col in zip(snrs_const, colors):
        y = AWGN(snr_db=snr, seed=123)(x)
        f_y, p_y = welch(y, fs=sig.sps, nperseg=1024, return_onesided=False)
        ax_psd.plot(f_y[order], 10 * np.log10(p_y[order] + 1e-20), color=col, lw=1.2, label=f"SNR={snr:.0f} dB")

    ax_psd.set_title("Densidad Espectral de Potencia (PSD)", fontsize=11, fontweight="bold")
    ax_psd.set_xlabel("Frecuencia normalizada ($f / R_s$)", fontsize=10)
    ax_psd.set_ylabel("Magnitud (dB)", fontsize=10)
    ax_psd.grid(True, alpha=0.3)
    ax_psd.set_ylim([-85, 8])
    ax_psd.legend(fontsize=9, loc="upper right")

    fig.suptitle("Barrido Paramétrico: Ruido Blanco Gaussiano Aditivo (AWGN)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    for d in out_dirs:
        fig.savefig(d / "sweep_awgn.png", dpi=140)
    plt.close(fig)
    print("  -> Guardado sweep_awgn.png")


def run_phase_noise_sweep(out_dirs: list[Path]):
    print("Ejecutando barrido de Ruido de Fase del Láser...")
    sig = gen_signal("qam16", n_symbols=4096, sps=2, seed=42)
    x = sig.samples
    fs = 2.0e9  # 2 GHz

    lws_const = [500.0, 2000.0, 5000.0, 15000.0]  # Hz
    lws_sweep = np.linspace(200.0, 25000.0, 25)

    # 1. Curva EVM vs Linewidth
    evms = []
    for lw in lws_sweep:
        y = PhaseNoise(linewidth_hz=lw, fs=fs, seed=42)(x)
        rx = recover_symbols(y, sig)
        evms.append(evm_percent(rx, sig.symbols))

    # 2. Generar figura
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0])

    # Fila superior: 4 constelaciones a distintos anchos de línea
    for i, lw in enumerate(lws_const):
        ax = fig.add_subplot(gs[0, i])
        y = PhaseNoise(linewidth_hz=lw, fs=fs, seed=42)(x)
        rx = recover_symbols(y, sig)
        evm = evm_percent(rx, sig.symbols)
        ax.plot(rx.real[32:-32], rx.imag[32:-32], ".", ms=2.5, alpha=0.45, color="#ff7f0e")
        label_lw = f"{lw/1e3:.1f} kHz" if lw >= 1e3 else f"{lw:.0f} Hz"
        ax.set_title(f"$\\Delta\\nu = {label_lw}$\nEVM: {evm:.2f} %", fontsize=10)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_xlim([-1.35, 1.35])
        ax.set_ylim([-1.35, 1.35])

    # Fila inferior izquierda: EVM vs Linewidth
    ax_evm = fig.add_subplot(gs[1, :2])
    ax_evm.plot(lws_sweep / 1e3, evms, "d-", color="#9467bd", lw=2, ms=4, label="EVM medido (%)")
    ax_evm.set_title("Curva de EVM vs. Ancho de Línea del Láser ($\\Delta\\nu$)", fontsize=11, fontweight="bold")
    ax_evm.set_xlabel("Ancho de línea $\\Delta\\nu$ (kHz)", fontsize=10)
    ax_evm.set_ylabel("EVM (%)", fontsize=10)
    ax_evm.grid(True, alpha=0.3)
    ax_evm.legend(fontsize=9.5)

    # Fila inferior derecha: PSD comparativa
    ax_psd = fig.add_subplot(gs[1, 2:])
    f_in, p_in = welch(x, fs=sig.sps, nperseg=1024, return_onesided=False)
    order = np.argsort(f_in)
    ax_psd.plot(f_in[order], 10 * np.log10(p_in[order] + 1e-20), color="black", lw=1.5, label="TX (ideal)")

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    for lw, col in zip(lws_const, colors):
        y = PhaseNoise(linewidth_hz=lw, fs=fs, seed=42)(x)
        f_y, p_y = welch(y, fs=sig.sps, nperseg=1024, return_onesided=False)
        label_lw = f"{lw/1e3:.1f} kHz" if lw >= 1e3 else f"{lw:.0f} Hz"
        ax_psd.plot(f_y[order], 10 * np.log10(p_y[order] + 1e-20), color=col, lw=1.2, label=f"$\\Delta\\nu={label_lw}$")

    ax_psd.set_title("Densidad Espectral de Potencia (PSD) - Faldas Lorentzianas", fontsize=11, fontweight="bold")
    ax_psd.set_xlabel("Frecuencia normalizada ($f / R_s$)", fontsize=10)
    ax_psd.set_ylabel("Magnitud (dB)", fontsize=10)
    ax_psd.grid(True, alpha=0.3)
    ax_psd.set_ylim([-90, 8])
    ax_psd.legend(fontsize=9, loc="upper right")

    fig.suptitle("Barrido Paramétrico: Ruido de Fase del Láser (Laser Phase Noise)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    for d in out_dirs:
        fig.savefig(d / "sweep_phase_noise.png", dpi=140)
    plt.close(fig)
    print("  -> Guardado sweep_phase_noise.png")


def main():
    root = Path(__file__).resolve().parent.parent
    dir1 = root / "outputs" / "barridos"
    dir2 = root.parent / "documentacion" / "figuras" / "barridos"
    dir1.mkdir(parents=True, exist_ok=True)
    dir2.mkdir(parents=True, exist_ok=True)

    out_dirs = [dir1, dir2]

    run_phase_offset_sweep(out_dirs)
    run_awgn_sweep(out_dirs)
    run_phase_noise_sweep(out_dirs)
    print("¡Barridos completados y figuras guardadas exitosamente!")


if __name__ == "__main__":
    main()
