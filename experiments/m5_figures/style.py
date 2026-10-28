"""House style for the M5 figures: the data-viz method's reference palette on a light surface.

Every figure module uses these roles rather than raw hex, so a palette change
is one edit here. The three QAOA backends always take the first three
categorical slots, validated all-pairs; the classical context series are
de-emphasis grays told apart by marker shape.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"  # titles
INK_2 = "#52514e"  # axis labels, legend text, subtitles
MUTED = "#898781"  # tick labels, source line
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots in their fixed order; the first three validate all-pairs.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DEVICE, NOISE_MODEL, NOISELESS = BLUE, ORANGE, AQUA
# Classical samplers and references: grays, identity carried by marker shape and the legend.
CLASSICAL = "#898781"
CLASSICAL_LIGHT = "#b3b1aa"
# The sequential/ordinal blue ramp for magnitudes (light surface: start no lighter than 250).
BLUE_RAMP = {250: "#86b6ef", 400: "#3987e5", 550: "#1c5cab", 700: "#0d366b"}
STATUS_CRITICAL = "#d03b3b"  # a cap or a limit line, with a label, never a series

DPI = 200
MARKERS = {"greedy": "o", "simulated_annealing": "s", "tabu": "^", "exact_pwl_milp": "D",
           "cpsat": "P"}
NAMES = {"greedy": "greedy", "simulated_annealing": "simulated annealing", "tabu": "tabu",
         "exact_pwl_milp": "HiGHS (proof)", "cpsat": "CP-SAT (sound)",
         "qaoa_ideal": "QAOA, noiseless", "qaoa_noisy": "QAOA, noise model",
         "qaoa_ibm": "QAOA on ibm_fez"}
COLORS = {"qaoa_ideal": NOISELESS, "qaoa_noisy": NOISE_MODEL, "qaoa_ibm": DEVICE,
          "greedy": CLASSICAL, "simulated_annealing": CLASSICAL, "tabu": CLASSICAL,
          "exact_pwl_milp": CLASSICAL, "cpsat": CLASSICAL_LIGHT}


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.labelsize": 9, "axes.labelcolor": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 4, "ytick.major.pad": 4,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "lines.linewidth": 1.5, "lines.solid_joinstyle": "round", "lines.solid_capstyle": "round",
        "lines.markersize": 6, "lines.markeredgewidth": 0.8, "lines.markeredgecolor": SURFACE,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK_2,
        "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })


def new_figure(width: float = 8.0, height: float = 4.5, nrows: int = 1, ncols: int = 1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kw)
    return fig, axes


def finish(ax, *, grid: str | None = "y", baseline: bool = True) -> None:
    """Recessive chrome: one hairline grid direction, a baseline, nothing else."""
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom" and baseline)
        spine.set_color(BASELINE)
        spine.set_linewidth(0.6)
    ax.tick_params(length=0, colors=MUTED)


def marker_kw(color: str, marker: str = "o", size: float = 6.0, filled: bool = True) -> dict:
    """A >= 8 px marker with a surface ring, filled with the series color (or hollow)."""
    return {"marker": marker, "markersize": size, "linestyle": "none",
            "markerfacecolor": color if filled else SURFACE,
            "markeredgecolor": SURFACE if filled else color,
            "markeredgewidth": 0.8 if filled else 1.2, "color": color}


def legend(ax, *, loc: str = "upper center", ncol: int = 3, above: bool = True, **kw):
    if above:
        return ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=ncol,
                         frameon=False, borderaxespad=0.0, **kw)
    return ax.legend(loc=loc, ncol=ncol, frameon=False, **kw)


def heading(ax, title: str, subtitle: str, *, legend_row: bool = True,
            extra_pt: float = 0.0) -> None:
    """The title, then one line in secondary ink saying what is plotted, above the axes.

    With ``legend_row`` the band leaves room for a legend placed by ``legend`` on the
    axes' top edge; ``extra_pt`` raises both lines further, for a taller legend.
    """
    sub_pt, title_pt = (22.0, 38.0) if legend_row else (6.0, 22.0)
    ax.set_title(title, loc="left", pad=title_pt + extra_pt)
    ax.annotate(subtitle, (0.0, 1.0), xycoords="axes fraction", xytext=(0, sub_pt + extra_pt),
                textcoords="offset points", ha="left", va="baseline", fontsize=9,
                color=INK_2)


def source(fig, text: str) -> None:
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7, color=MUTED,
             transform=fig.transFigure)


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path
