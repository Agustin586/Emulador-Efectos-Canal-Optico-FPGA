"""Gráficos comunes de los notebooks: constelación, ojo, espectro y guardado.

Cada función recibe un ``Axes`` y dibuja sobre él, así se combinan en subplots.
Las figuras se guardan en docs/golden_model/figuras/ con ``save_fig``.
Paleta del proyecto: #14213D, #B35418, #2B5C9E (``style.mplstyle``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

STYLE = Path(__file__).with_name("style.mplstyle")
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "golden_model" / "figuras"


def use_style() -> None:
    """Activa el estilo del proyecto para las figuras que se creen después."""
    plt.style.use(STYLE)


def constellation(ax, s: np.ndarray, ref: np.ndarray | None = None, title: str | None = None):
    """Constelación de los símbolos ``s``; ``ref`` (opcional) marca los puntos ideales."""
    s = np.asarray(s)
    if ref is not None:
        pts = np.unique(np.round(np.asarray(ref), 9))
        ax.plot(pts.real, pts.imag, "+", color="C1", ms=14, mew=1, zorder=1)
    ax.plot(s.real, s.imag, ".", ms=2, alpha=0.4, zorder=2)
    # mismos límites en I y Q: con PAM4 (Q ~ 0) el eje Q no colapsa
    lim = 1.15 * np.max(np.abs(np.concatenate([s.real, s.imag])))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    if title:
        ax.set_title(title)
    return ax


def eye(ax, x: np.ndarray, sps: int, n_traces: int = 500, offset: int = 0, title: str | None = None):
    """Diagrama de ojo de la parte real: ventanas de 2 símbolos superpuestas.

    offset : muestra donde empieza la primera ventana. Para centrar el ojo en
             el instante de muestreo k·sps, usar offset = sps // 2 (el ojo queda
             abierto en t = 0,5 y 1,5 símbolos). Descartar los transitorios de
             los bordes con un offset múltiplo de sps.
    """
    x = np.asarray(x).real
    win = 2 * sps + 1
    starts = offset + sps * np.arange(n_traces)
    starts = starts[starts + win <= len(x)]
    t = np.arange(win) / sps
    traces = x[starts[:, None] + np.arange(win)]
    ax.plot(t, traces.T, color="C0", lw=0.6, alpha=0.15)
    ax.set_xlim(0, 2)
    ax.set_xlabel("t / T (símbolos)")
    ax.set_ylabel("amplitud")
    if title:
        ax.set_title(title)
    return ax


def spectrum(ax, x: np.ndarray, fs: float, nperseg: int = 1024, label: str | None = None,
             f_unit: float = 1e9, title: str | None = None):
    """Densidad espectral de potencia de dos lados (Welch), en dB, f ordenada.

    f_unit : escala del eje de frecuencias (1e9 -> GHz; Rs -> f/Rs).
    """
    f, p = welch(np.asarray(x), fs=fs, nperseg=nperseg, return_onesided=False)
    order = np.argsort(f)
    ax.plot(f[order] / f_unit, 10 * np.log10(p[order] + 1e-300), label=label)
    ax.set_xlabel("f [GHz]" if f_unit == 1e9 else "f")
    ax.set_ylabel("PSD [dB/Hz]")
    if title:
        ax.set_title(title)
    return ax


def save_fig(fig, name: str, pdf: bool = True, outdir: Path | str | None = None,
             dpi: float | None = None) -> Path:
    """Guarda ``fig`` como <outdir>/<name>.png (y .pdf para presentaciones).

    outdir por defecto: docs/golden_model/figuras/. Devuelve la ruta del .png.
    dpi    : resolución del .png (None -> la del estilo, 200). Los ojos, con
             cientos de trazas semitransparentes, comprimen mal: usar ~110.
    """
    outdir = Path(outdir) if outdir is not None else FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{name}.png"
    if dpi is None:
        fig.savefig(png)
    else:
        fig.savefig(png, dpi=dpi)
    if pdf:
        fig.savefig(outdir / f"{name}.pdf")
    return png
