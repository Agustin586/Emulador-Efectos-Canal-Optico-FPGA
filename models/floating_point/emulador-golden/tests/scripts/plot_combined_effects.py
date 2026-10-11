"""Genera figuras diagnósticas de validación para combinaciones de deterioros de canal.

Combinaciones simuladas:
1. Offset de Fase + AWGN
2. Offset de Fase + Ruido de Fase
3. Ruido de Fase + AWGN
4. Cadena Integral Completa (Offset + Ruido de Fase + AWGN)

Uso:
    python -m scripts.plot_combined_effects
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


def generate_combined_figures():
    print("Iniciando simulación de efectos combinados...")
    sig = gen_signal("qam16", n_symbols=4096, sps=2, seed=42)
    x = sig.samples

    # Parámetros estándar consistentes
    delta_theta = 0.30  # rad (~17.2 deg)
    linewidth = 3.0e3   # Hz (3 kHz a fs=2 GHz)
    snr_val = 22.0      # dB

    b_po = PhaseOffset(delta_theta_rad=delta_theta)
    b_pn = PhaseNoise(linewidth_hz=linewidth, fs=2.0e9, seed=123)
    b_awgn = AWGN(snr_db=snr_val, seed=456)

    # 1. Recuperación entrada ideal
    rx_in = recover_symbols(x, sig)
    evm_in = evm_percent(rx_in, sig.symbols)

    # Combinación 1: Offset + AWGN
    y_combo1 = b_awgn(b_po(x))
    rx_c1 = recover_symbols(y_combo1, sig)
    evm_c1 = evm_percent(rx_c1, sig.symbols)

    # Combinación 2: Offset + Ruido de Fase
    y_combo2 = b_pn(b_po(x))
    rx_c2 = recover_symbols(y_combo2, sig)
    evm_c2 = evm_percent(rx_c2, sig.symbols)
    max_env_err_c2 = np.max(np.abs(np.abs(y_combo2) - np.abs(x)))

    # Combinación 3: Ruido de Fase + AWGN
    y_combo3 = b_awgn(b_pn(x))
    rx_c3 = recover_symbols(y_combo3, sig)
    evm_c3 = evm_percent(rx_c3, sig.symbols)

    # Combinación 4: Cadena Integral Completa
    y_combo4 = b_awgn(b_pn(b_po(x)))
    rx_c4 = recover_symbols(y_combo4, sig)
    evm_c4 = evm_percent(rx_c4, sig.symbols)

    print(f"EVM Entrada: {evm_in:.2f} %")
    print(f"Combo 1 (Offset + AWGN): {evm_c1:.2f} %")
    print(f"Combo 2 (Offset + Ruido Fase): {evm_c2:.2f} % (Max diff envolvente: {max_env_err_c2:.2e})")
    print(f"Combo 3 (Ruido Fase + AWGN): {evm_c3:.2f} %")
    print(f"Combo 4 (Cadena Completa): {evm_c4:.2f} %")

    # Destinos de salida
    out_dirs = [
        Path("outputs"),
        Path("../documentacion/figuras"),
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # FIGURA 1: Combinaciones Par a Par (3 subplots horizontales)
    # =========================================================================
    fig1, axs1 = plt.subplots(1, 3, figsize=(14, 3.8), dpi=150)
    combos_pairwise = [
        (
            axs1[0],
            rx_c1,
            "Offset de Fase + AWGN",
            f"$\\Delta\\theta = {delta_theta:.2f}$ rad, SNR = {snr_val:.0f} dB\nEVM = {evm_c1:.2f} %",
            "#1f77b4",
            "Giro rígido + Nubes gaussianas",
        ),
        (
            axs1[1],
            rx_c2,
            "Offset de Fase + Ruido de Fase",
            f"$\\Delta\\theta = {delta_theta:.2f}$ rad, $\\Delta\\nu = {linewidth/1e3:.1f}$ kHz\nEVM = {evm_c2:.2f} % ($|y|=|x|$)",
            "#ff7f0e",
            "Giro estático + Arcos brownianos",
        ),
        (
            axs1[2],
            rx_c3,
            "Ruido de Fase + AWGN",
            f"$\\Delta\\nu = {linewidth/1e3:.1f}$ kHz, SNR = {snr_val:.0f} dB\nEVM = {evm_c3:.2f} %",
            "#2ca02c",
            "Arcos concéntricos ensanchados",
        ),
    ]

    for ax, rx_data, title, subtitle, color, note in combos_pairwise:
        ax.plot(rx_data.real[32:-32], rx_data.imag[32:-32], ".", ms=2.5, alpha=0.5, color=color)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("I (En fase)", fontsize=9.5)
        ax.set_ylabel("Q (Cuadratura)", fontsize=9.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim([-1.35, 1.35])
        ax.set_ylim([-1.35, 1.35])
        # Cuadro de anotación inferior
        ax.text(
            0.0,
            -1.18,
            note,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
        )

    fig1.tight_layout()

    for d in out_dirs:
        p1 = d / "combinaciones_par_a_par.png"
        fig1.savefig(p1, bbox_inches="tight")
        print(f"Guardado: {p1}")
    plt.close(fig1)

    # =========================================================================
    # FIGURA 2: Cadena Integral Coherente (Entrada vs Salida vs Espectro)
    # =========================================================================
    fig2, axs2 = plt.subplots(1, 3, figsize=(14, 3.8), dpi=150)

    # Subplot 1: Constelación TX
    axs2[0].plot(rx_in.real[32:-32], rx_in.imag[32:-32], ".", ms=2.5, alpha=0.6, color="#1f77b4")
    axs2[0].set_title(f"Entrada TX: 16-QAM Ideal\nEVM = {evm_in:.2f} %", fontsize=10.5, fontweight="bold")
    axs2[0].set_xlabel("I (En fase)", fontsize=9.5)
    axs2[0].set_ylabel("Q (Cuadratura)", fontsize=9.5)
    axs2[0].set_aspect("equal")
    axs2[0].grid(True, alpha=0.3, linestyle="--")
    axs2[0].set_xlim([-1.35, 1.35])
    axs2[0].set_ylim([-1.35, 1.35])

    # Subplot 2: Constelación RX Cadena Completa
    axs2[1].plot(rx_c4.real[32:-32], rx_c4.imag[32:-32], ".", ms=2.5, alpha=0.5, color="#d62728")
    axs2[1].set_title(
        f"Salida: Cadena Coherente Completa\n$\\Delta\\theta={delta_theta:.2f}$ rad, $\\Delta\\nu={linewidth/1e3:.1f}$ kHz, SNR={snr_val:.0f} dB\nEVM = {evm_c4:.2f} %",
        fontsize=10,
        fontweight="bold",
    )
    axs2[1].set_xlabel("I (En fase)", fontsize=9.5)
    axs2[1].set_ylabel("Q (Cuadratura)", fontsize=9.5)
    axs2[1].set_aspect("equal")
    axs2[1].grid(True, alpha=0.3, linestyle="--")
    axs2[1].set_xlim([-1.35, 1.35])
    axs2[1].set_ylim([-1.35, 1.35])
    axs2[1].text(
        0.0,
        -1.18,
        "Giro + Arcos Wiener + Nubes AWGN",
        ha="center",
        va="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    # Subplot 3: Espectros Welch
    f_in, p_in = welch(x, fs=sig.sps, nperseg=1024, return_onesided=False)
    f_out, p_out = welch(y_combo4, fs=sig.sps, nperseg=1024, return_onesided=False)
    ord_in = np.argsort(f_in)
    ord_out = np.argsort(f_out)

    axs2[2].plot(f_in[ord_in], 10 * np.log10(p_in[ord_in] + 1e-20), label="TX Limpia", color="#1f77b4", lw=1.5)
    axs2[2].plot(f_out[ord_out], 10 * np.log10(p_out[ord_out] + 1e-20), label="RX Cadena Completa", color="#d62728", lw=1.5, alpha=0.85)
    axs2[2].set_title("Espectro Welch (f normalizada a Rs)\nPiso AWGN y faldas de láser", fontsize=10.5, fontweight="bold")
    axs2[2].set_xlabel("f / Rs", fontsize=9.5)
    axs2[2].set_ylabel("Densidad espectral (dB)", fontsize=9.5)
    axs2[2].grid(True, alpha=0.3, linestyle="--")
    axs2[2].legend(loc="lower center", fontsize=8.5)
    axs2[2].set_ylim([-80, 5])

    fig2.tight_layout()

    for d in out_dirs:
        p2 = d / "combinacion_cadena_completa.png"
        fig2.savefig(p2, bbox_inches="tight")
        print(f"Guardado: {p2}")
    plt.close(fig2)

    # =========================================================================
    # FIGURA 3: Matriz 2x2 de Todas las Combinaciones Posibles
    # =========================================================================
    fig3, axs3 = plt.subplots(2, 2, figsize=(10, 9.5), dpi=150)
    matrix_data = [
        (axs3[0, 0], rx_c1, "1. Offset de Fase + AWGN", f"$\\Delta\\theta={delta_theta:.2f}$ rad, SNR={snr_val:.0f} dB\nEVM = {evm_c1:.2f} %", "#1f77b4"),
        (axs3[0, 1], rx_c2, "2. Offset + Ruido de Fase", f"$\\Delta\\theta={delta_theta:.2f}$ rad, $\\Delta\\nu={linewidth/1e3:.1f}$ kHz\nEVM = {evm_c2:.2f} % ($|y|=|x|$)", "#ff7f0e"),
        (axs3[1, 0], rx_c3, "3. Ruido de Fase + AWGN", f"$\\Delta\\nu={linewidth/1e3:.1f}$ kHz, SNR={snr_val:.0f} dB\nEVM = {evm_c3:.2f} %", "#2ca02c"),
        (axs3[1, 1], rx_c4, "4. Cadena Óptica Completa", f"$\\Delta\\theta={delta_theta:.2f}$, $\\Delta\\nu={linewidth/1e3:.1f}$ kHz, SNR={snr_val:.0f} dB\nEVM = {evm_c4:.2f} %", "#d62728"),
    ]

    for ax, rx_data, title, subtitle, color in matrix_data:
        ax.plot(rx_data.real[32:-32], rx_data.imag[32:-32], ".", ms=2.5, alpha=0.5, color=color)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
        ax.set_xlabel("I", fontsize=10)
        ax.set_ylabel("Q", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim([-1.4, 1.4])
        ax.set_ylim([-1.4, 1.4])

    fig3.suptitle("Matriz de Combinaciones Posibles del Canal Óptico", fontsize=14, fontweight="bold", y=0.99)
    fig3.tight_layout()

    for d in out_dirs:
        p3 = d / "matriz_4_combinaciones.png"
        fig3.savefig(p3, bbox_inches="tight")
        print(f"Guardado: {p3}")
    plt.close(fig3)

    print("¡Todas las figuras de efectos combinados se han generado exitosamente!")


if __name__ == "__main__":
    generate_combined_figures()
