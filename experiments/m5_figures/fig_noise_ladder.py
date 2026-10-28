"""Success and feasibility against circuit depth, per instance: noiseless, noise model, device.

Caption. One point per instance and backend, for the binary-encoded T0 instances that ran
on ibm_fez: 30 at reps = 2 (left column, ``m5-p2-qaoa_noise.csv``) and 31 at reps = 1
(right column, ``m5-p1-qaoa_noise.csv``, when present). The x axis, shared by every panel,
is the routed two-qubit (CZ) gate count of the instance's device circuit (``routed_2q`` of
its ``qaoa_ibm`` row: the live backend's routing), used for all three backends of that
instance. Top row: success at epsilon = 1%, the share of runs whose best feasible sample is
within 1% of the proven optimum. Bottom row: the feasible share of all samples, on a log
axis; an instance whose runs returned no feasible sample is drawn hollow at the floor of
0.01%. Aqua: the noiseless simulation, 20 runs of 1,000 shots per instance. Orange: the
fake_fez noise model, 5 runs, only for instances of at most 20 qubits and only where the
runs finished (reps = 2: 3 of the 17 such instances had every run killed at the 2 h limit;
reps = 1: 1 of 17), so 14 and 16 orange points. Blue: the device, the mean over the
instance's 1,000-shot jobs, three at reps = 2 and one at reps = 1, so a device success
reads 0, 1/3, 2/3 or 1 on the left and 0 or 1 on the right. The vertical hairline in each
column marks the deepest circuit on which any device job found the optimum. How to read
it: noiselessly, success falls with depth because depth tracks the qubit count; on the
device, every success sits at or left of the hairline (1,233 CZ at reps = 2, 378 at
reps = 1), and the reps = 1 circuits sit about half as deep, 313 to 952 CZ against 735 to
2,044. The noise model lies between the noiseless circuit and the device in both columns.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from m5_figures import style

NAME = "noise_ladder"
TITLE = "Success and feasibility against circuit depth, per instance, on each backend"
SUBTITLE = ("Binary T0 instances that ran on ibm_fez; x = routed CZ gates of the device "
            "circuit; left reps = 2, right reps = 1")
SOURCE = ("Source: docs/figures/m5/data/m5-p2-qaoa_noise.csv and m5-p1-qaoa_noise.csv "
          "(p_success_0.01, feasibility, routed_2q). "
          "Regenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")
BACKENDS = [("qaoa_ideal", style.NOISELESS), ("qaoa_noisy", style.NOISE_MODEL),
            ("qaoa_ibm", style.DEVICE)]
FLOOR = 1e-4  # where a zero feasibility is drawn, hollow
X_LIM = (200, 2150)
SIZE = 5.5


def _ladder(noise: pl.DataFrame) -> pl.DataFrame:
    """Per instance and backend: success, feasibility, and the device circuit's CZ count."""
    rows = noise.filter(pl.col("encoding") == "binary")
    cz = (rows.filter(pl.col("solver") == "qaoa_ibm")
          .select("dials", "instance_seed", pl.col("routed_2q").alias("cz")))
    return (rows.join(cz, on=["dials", "instance_seed"])  # instances that ran on the device
            .with_columns(pl.col("feasibility").fill_nan(None))
            .filter(~((pl.col("solver") == "qaoa_noisy") & pl.col("feasibility").is_null())))


def _column(ax_s, ax_f, frame: pl.DataFrame, label: str) -> None:
    for solver, color in BACKENDS:
        rows = frame.filter(pl.col("solver") == solver)
        name = style.NAMES[solver]
        ax_s.plot(rows["cz"].to_list(), rows["p_success_0.01"].to_list(), label=name,
                  **style.marker_kw(color, "o", SIZE))
        some = rows.filter(pl.col("feasibility") > 0)
        none = rows.filter(pl.col("feasibility") == 0)
        ax_f.plot(some["cz"].to_list(), some["feasibility"].to_list(),
                  **style.marker_kw(color, "o", SIZE))
        ax_f.plot(none["cz"].to_list(), [FLOOR] * none.height,
                  **style.marker_kw(color, "o", SIZE, filled=False))
    device = frame.filter((pl.col("solver") == "qaoa_ibm") & (pl.col("p_success_0.01") > 0))
    if device.height:
        last = float(device["cz"].max())
        for ax in (ax_s, ax_f):
            ax.axvline(last, color=style.BASELINE, linewidth=0.8, zorder=1)
        # Left of the line where the column has room (reps = 2), else right of it (reps = 1):
        # the two spots that stay clear of the points.
        left = last - X_LIM[0] > 700
        ax_s.annotate(f"deepest device success:\n{last:,.0f} CZ", (last, 0.86 if left else 0.62),
                      xytext=(-5 if left else 5, 0), textcoords="offset points",
                      ha="right" if left else "left", va="center", fontsize=7.5,
                      color=style.MUTED, linespacing=1.2)
    ax_s.annotate(label, (0.0, 1.0), xycoords="axes fraction", xytext=(0, 5),
                  textcoords="offset points", ha="left", va="bottom", fontsize=9,
                  color=style.INK, fontweight="semibold")


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    columns = [(_ladder(data["m5-p2-qaoa_noise"]), "reps = 2: three jobs per instance")]
    if "m5-p1-qaoa_noise" in data:
        columns.append((_ladder(data["m5-p1-qaoa_noise"]), "reps = 1: one job per instance"))

    fig, axes = style.new_figure(8.0, 6.2, 2, len(columns), sharex=True, sharey="row",
                                 squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.11, hspace=0.12, wspace=0.08)
    for j, (frame, label) in enumerate(columns):
        _column(axes[0][j], axes[1][j], frame, label)

    for ax in axes[0]:
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_locator(FixedLocator([0.0, 0.25, 0.5, 0.75, 1.0]))
        ax.yaxis.set_major_formatter(FixedFormatter(["0", "25%", "50%", "75%", "100%"]))
        style.finish(ax, grid="y")
    for ax in axes[1]:
        ax.set_yscale("log")
        ax.set_ylim(FLOOR / 10**0.25, 1.0)
        ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1.0]))
        ax.yaxis.set_major_formatter(FixedFormatter(["0.01%", "0.1%", "1%", "10%", "100%"]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(*X_LIM)
        ax.set_xticks([250, 500, 1000, 1500, 2000], ["250", "500", "1,000", "1,500", "2,000"])
        ax.set_xlabel("routed CZ gates of the device circuit")
        style.finish(ax, grid="y")
    axes[0][0].set_ylabel("runs within 1% of the optimum")
    axes[1][0].set_ylabel("feasible share of samples (log)")
    axes[1][0].annotate("hollow: no feasible sample", (0.0, 0.0), xycoords="axes fraction",
                        xytext=(4, 4), textcoords="offset points", ha="left", va="bottom",
                        fontsize=7.5, color=style.MUTED)

    # Above the left column, bottom to top: the column label, the legend, subtitle, title.
    style.heading(axes[0][0], TITLE, SUBTITLE, extra_pt=18)
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[0][0].legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
                      frameon=False, borderaxespad=2.0)
    style.source(fig, SOURCE)
    return [style.save(fig, out, NAME)]
