"""Reps 1 against reps 2, instance by instance: the noiseless circuit, the noise model, the device.

Caption. One row per binary-encoded T0 instance that ran on ibm_fez at either depth (31:
the 30 of reps = 2 and ``D2+D4-2030``, whose reps = 2 gate run was killed, so it ran on the
device at reps = 1 only), labelled dials-seed and sorted by the routed CZ count of its
reps = 2 device circuit, shallowest at the top (``D2+D4-2030`` last). In each of the three
panels the x axis is success at epsilon = 1%, the share of runs whose best feasible sample
is within 1% of the proven optimum; the filled marker is reps = 2
(``m5-p2-qaoa_noise.csv``), the hollow one reps = 1 (``m5-p1-qaoa_noise.csv``), joined by
a hairline when both exist. Noiseless: 20 runs per instance at either depth. Noise model:
5 runs, for instances of at most 20 qubits whose runs finished (reps = 2: 14 instances;
reps = 1: 16; both: 13). Device: three 1,000-shot jobs per instance at reps = 2, one at
reps = 1, so the markers sit at 0, 1/3, 2/3, 1 and at 0 or 1. The panel titles give the
mean success over the instances present at both depths. The right-hand column gives the
routed CZ count of each device circuit, reps = 2 then reps = 1. How to read it: halving the
depth halves the CZ count (735-2,044 to 313-952) and costs the noiseless circuit about a
quarter of its success (mean 0.45 to 0.35 on the 31 instances), while on the device
the reps = 1 successes sit on four instances of 313-325 CZ gates and one of 378, where
reps = 2 had succeeded on 8 instances of 735 to 1,233 gates; the device's mean success
over the 30 shared instances goes from 0.12 (three jobs each) to 0.17 (one job each).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from matplotlib.lines import Line2D

from m5_figures import style

NAME = "reps_compare"
TITLE = "Reps 1 against reps 2, instance by instance"
SUBTITLE = "Success at ε = 1% per instance on each backend; filled: reps = 2, hollow: reps = 1"
SOURCE = ("Source: docs/figures/m5/data/m5-p2-qaoa_noise.csv and m5-p1-qaoa_noise.csv "
          "(p_success_0.01, feasibility, routed_2q; binary). "
          "Regenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")
PANELS = [("qaoa_ideal", style.NOISELESS, "noiseless"), ("qaoa_noisy", style.NOISE_MODEL,
                                                          "noise model"),
          ("qaoa_ibm", style.DEVICE, "device, ibm_fez")]
SIZE = 5.5
RING = SIZE + 3.0  # reps = 1 is a ring around where a reps = 2 dot would sit
ROW_IN = 0.19  # figure inches per instance row
HEAD_IN = 1.5  # title, subtitle, legend and panel titles above the axes


def _success(noise: pl.DataFrame) -> pl.DataFrame:
    """(dials, instance_seed, solver) -> success, dropping noise-model rows whose runs died."""
    rows = (noise.filter(pl.col("encoding") == "binary")
            .with_columns(pl.col("feasibility").fill_nan(None)))
    rows = rows.filter(~((pl.col("solver") == "qaoa_noisy") & pl.col("feasibility").is_null()))
    return rows.select("dials", "instance_seed", "solver", "p_success_0.01", "routed_2q")


def _cz(rows: pl.DataFrame) -> dict[tuple[str, int], float]:
    device = rows.filter(pl.col("solver") == "qaoa_ibm")
    return {(r["dials"], r["instance_seed"]): r["routed_2q"] for r in device.iter_rows(named=True)}


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    if "m5-p1-qaoa_noise" not in data:
        return []
    two, one = _success(data["m5-p2-qaoa_noise"]), _success(data["m5-p1-qaoa_noise"])
    cz2, cz1 = _cz(two), _cz(one)
    instances = sorted(set(cz2) | set(cz1),
                       key=lambda k: (k not in cz2, cz2.get(k, 0.0), cz1.get(k, 0.0)))
    y_of = {k: i for i, k in enumerate(instances)}
    n = len(instances)

    height = HEAD_IN + 0.55 + ROW_IN * n
    fig, axes = style.new_figure(8.0, height, 1, 3, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.86, top=1 - HEAD_IN / height, bottom=0.55 / height,
                        wspace=0.08)
    for ax, (solver, color, label) in zip(axes, PANELS, strict=True):
        s2 = {(r["dials"], r["instance_seed"]): r["p_success_0.01"]
              for r in two.filter(pl.col("solver") == solver).iter_rows(named=True)}
        s1 = {(r["dials"], r["instance_seed"]): r["p_success_0.01"]
              for r in one.filter(pl.col("solver") == solver).iter_rows(named=True)}
        both = [k for k in instances if k in s2 and k in s1]
        for k in instances:
            y = y_of[k]
            if k in s2 and k in s1:
                ax.plot([s2[k], s1[k]], [y, y], color=style.BASELINE, linewidth=0.8, zorder=1)
            if k in s1:
                ax.plot([s1[k]], [y], zorder=3, **style.marker_kw(color, "o", RING, filled=False))
            if k in s2:
                ax.plot([s2[k]], [y], zorder=4, **style.marker_kw(color, "o", SIZE))
        m2 = sum(s2[k] for k in both) / len(both)
        m1 = sum(s1[k] for k in both) / len(both)
        ax.set_title(f"{label}\nmean over {len(both)}: {m2:.2f} → {m1:.2f}", loc="left",
                     fontsize=9, color=style.INK, pad=6)
        ax.set_xlim(-0.08, 1.08)
        ax.set_xticks([0.0, 0.5, 1.0], ["0", "50%", "100%"])
        style.finish(ax, grid="x")

    axes[0].set_yticks(range(n), [f"{d}-{s}" for d, s in instances])
    axes[0].set_ylim(n - 0.4, -0.6)
    for label in axes[0].get_yticklabels():
        label.set_color(style.MUTED)
        label.set_fontsize(7.5)
    axes[1].set_xlabel("runs within 1% of the optimum")
    # The CZ counts of the device circuits, right of the device panel: reps 2 -> reps 1.
    for k, y in y_of.items():
        a = f"{cz2[k]:,.0f}" if k in cz2 else "-"
        b = f"{cz1[k]:,.0f}" if k in cz1 else "-"
        axes[2].annotate(f"{a} → {b}", (1.0, y), xycoords=("axes fraction", "data"),
                         xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                         fontsize=7, color=style.MUTED)
    axes[2].annotate("CZ, reps 2 → 1", (1.0, 1.0), xycoords="axes fraction", xytext=(8, 6),
                     textcoords="offset points", ha="left", va="bottom", fontsize=7.5,
                     color=style.INK_2)

    handles = [Line2D([], [], **style.marker_kw(style.CLASSICAL, "o", SIZE)),
               Line2D([], [], **style.marker_kw(style.CLASSICAL, "o", RING, filled=False))]
    fig.text(0.15, 1 - 0.12 / height, TITLE, ha="left", va="top", fontsize=11,
             fontweight="semibold", color=style.INK)
    fig.text(0.15, 1 - 0.42 / height, SUBTITLE, ha="left", va="top", fontsize=9,
             color=style.INK_2)
    fig.legend(handles, ["reps = 2", "reps = 1"], loc="upper left", ncol=2, frameon=False,
               bbox_to_anchor=(0.143, 1 - 0.62 / height), fontsize=8, borderaxespad=0.0)
    style.source(fig, SOURCE)
    return [style.save(fig, out, NAME)]
