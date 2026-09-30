"""Gráficos (paso P4 de GUIA_ISI_CD.md): pruebas de humo, sin comparar imágenes."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from emulador import plots  # noqa: E402
from emulador.config import Scenario  # noqa: E402
from emulador.stimuli import comm_signal  # noqa: E402


def test_style_file_loads():
    with plt.style.context(plots.STYLE):
        assert matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][:3] == [
            "#14213D",
            "#B35418",
            "#2B5C9E",
        ]


def test_plots_draw_and_save(tmp_path):
    esc = Scenario()
    sig = comm_signal(esc, mod="pam4", n_symbols=512)
    with plt.style.context(plots.STYLE):
        fig, axs = plt.subplots(1, 3)
        plots.constellation(axs[0], sig.symbols, ref=sig.symbols)
        plots.eye(axs[1], sig.samples, esc.sps, n_traces=100, offset=64 * esc.sps)
        plots.spectrum(axs[2], sig.samples, esc.fs, nperseg=256)
        png = plots.save_fig(fig, "prueba", outdir=tmp_path)
        plt.close(fig)
    assert png.exists() and (tmp_path / "prueba.pdf").exists()
    assert len(axs[1].lines) == 100


def test_fig_dir_is_in_docs():
    assert plots.FIG_DIR.parts[-3:] == ("docs", "golden_model", "figuras")
    assert plots.STYLE.exists()
