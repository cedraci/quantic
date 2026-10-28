"""At 1,000 reads: how often the optimum is found, and how many reads are feasible.

Caption. Two dot-plot panels share their rows: the eight T0 dial combinations
of the binary encoding (none, D1, D4, D2, D1+D4, D1+D2, D2+D4, D1+D2+D4, top
to bottom), read from ``docs/figures/m5/data/m5-p2-matched_reads.csv`` with
every solver at max_samples = 1,000. Left: ``p_success_0.01``, the share of
runs whose best feasible sample lies within epsilon = 1% of the optimum (a run
succeeds when its relative gap to the exact optimum is at most 1%). Right:
``feasibility``, the share of the 1,000 samples of a run that satisfy every
constraint, on a log axis from 0.05% to 100%. Each classical sampler (greedy,
simulated annealing, tabu; gray, told apart by marker shape) averages 100 runs
per cell = 5 instances x 20 seeds. QAOA noiseless (aqua) averages 20 runs per
instance, the fake_fez noise model (orange) 5 runs on each instance of at most
20 qubits, so it has no marker for the 24-qubit D1 cell, and the device (blue)
n jobs per instance: three 1,000-shot jobs on each of the 30 instances ibm_fez
ran, so the "k of n jobs" label next to each device marker counts the jobs
within 1% of the optimum. The D2+D4 noise-model cell had all 5 of its runs
killed (``n_killed == n_runs`` in ``m5-p2-cells.csv``), so it is drawn hollow at
zero success and at the floor of the feasibility axis; the D1+D2 noise-model
cell lost 10 of its 20 runs the same way and is plotted as reported. Inside a
row the six markers are spread vertically in legend order (device, noise model,
noiseless, greedy, simulated annealing, tabu) so that ties stay visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from m5_figures import style

NAME = "matched_reads"
ROWS = ["none", "D1", "D4", "D2", "D1+D4", "D1+D2", "D2+D4", "D1+D2+D4"]
# Legend order, and the top-to-bottom order of the markers inside a row.
SERIES = [
    ("qaoa_ibm", style.DEVICE, "o"),
    ("qaoa_noisy", style.NOISE_MODEL, "o"),
    ("qaoa_ideal", style.NOISELESS, "o"),
    ("greedy", style.CLASSICAL, style.MARKERS["greedy"]),
    ("simulated_annealing", style.CLASSICAL, style.MARKERS["simulated_annealing"]),
    ("tabu", style.CLASSICAL, style.MARKERS["tabu"]),
]
FLOOR = 5e-4  # the feasibility axis starts here; a killed cell sits on it
MARKER_SIZE = 6.5
SLOT_PT = 6.0  # vertical spacing between the six markers of a row, in points
ROW_PAD = 0.6  # data units above the first row and below the last
WIDTH, HEIGHT = 8.0, 6.4
LEFT, RIGHT, TOP, BOTTOM = 0.10, 0.985, 0.87, 0.085
TITLE = "At 1,000 reads: how often the optimum is found, and how many reads are feasible"
SUBTITLE = "T0, binary encoding, 1,000 reads per run; one row per dial combination, six solvers"
SOURCE = ("Source: docs/figures/m5/data/m5-p2-matched_reads.csv (killed runs from m5-p2-cells.csv)."
          "  Regenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")


def _killed_cells(cells: pl.DataFrame) -> dict[str, int]:
    """dials -> n_runs of the binary qaoa_noisy cells whose every run was killed."""
    dead = cells.filter((pl.col("encoding") == "binary") & (pl.col("solver") == "qaoa_noisy")
                        & (pl.col("n_killed") == pl.col("n_runs")))
    return dict(zip(dead["dials"].to_list(), dead["n_runs"].to_list(), strict=True))


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    reads = data["m5-p2-matched_reads"].filter(pl.col("encoding") == "binary")
    table = {(r["solver"], r["dials"]): r for r in reads.iter_rows(named=True)}
    killed = _killed_cells(data["m5-p2-cells"])

    fig, (ax_p, ax_f) = style.new_figure(WIDTH, HEIGHT, 1, 2, sharey=True)
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM, wspace=0.14)
    y_of = {dials: -i for i, dials in enumerate(ROWS)}
    span = len(ROWS) - 1 + 2 * ROW_PAD
    slot = SLOT_PT * span / (HEIGHT * (TOP - BOTTOM) * 72)
    note_kw = {"textcoords": "offset points", "va": "center", "fontsize": 6.5,
               "color": style.MUTED}

    handles: dict[str, plt.Line2D] = {}
    for j, (solver, color, marker) in reversed(list(enumerate(SERIES))):  # QAOA drawn last
        dy = (len(SERIES) - 1) / 2 * slot - j * slot
        xs_p: list[float] = []
        xs_f: list[float] = []
        ys: list[float] = []
        for dials in ROWS:
            row = table.get((solver, dials))
            if row is None:  # no noise-model run for the 24-qubit D1 cell
                continue
            y = y_of[dials] + dy
            if solver == "qaoa_noisy" and dials in killed:
                hollow = style.marker_kw(color, marker, MARKER_SIZE, filled=False)
                ax_p.plot([0.0], [y], **hollow)
                ax_f.plot([FLOOR], [y], **hollow)
                ax_f.annotate(f"all {killed[dials]} runs killed", (FLOOR, y), xytext=(7, 0),
                              **note_kw)
                continue
            xs_p.append(row["p_success_0.01"])
            xs_f.append(row["feasibility"])
            ys.append(y)
            if solver == "qaoa_ibm":
                n = row["n_runs"]
                k = round(row["p_success_0.01"] * n)
                ax_p.annotate(f"{k} of {n} jobs", (row["p_success_0.01"], y), xytext=(7, 1),
                              **note_kw)
        kw = style.marker_kw(color, marker, MARKER_SIZE)
        (handles[solver],) = ax_p.plot(xs_p, ys, label=style.NAMES[solver], **kw)
        ax_f.plot(xs_f, ys, **kw)

    ax_p.set_xlim(-0.04, 1.04)
    ax_p.xaxis.set_major_locator(FixedLocator([0.0, 0.25, 0.5, 0.75, 1.0]))
    ax_p.xaxis.set_major_formatter(FixedFormatter(["0", "25%", "50%", "75%", "100%"]))
    ax_p.set_xlabel("runs within 1% of the optimum")
    ax_f.set_xscale("log")
    ax_f.set_xlim(FLOOR / 10**0.1, 10**0.1)
    ax_f.xaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1.0]))
    ax_f.xaxis.set_major_formatter(FixedFormatter(["0.1%", "1%", "10%", "100%"]))
    ax_f.xaxis.set_minor_locator(NullLocator())
    ax_f.set_xlabel("feasible share of the 1,000 samples (log)")
    ax_p.set_ylim(-(len(ROWS) - 1) - ROW_PAD, ROW_PAD)
    ax_p.set_yticks([y_of[d] for d in ROWS])
    ax_p.set_yticklabels(ROWS)
    ax_p.set_ylabel("dials")
    for ax in (ax_p, ax_f):
        style.finish(ax, grid="x")

    style.heading(ax_p, TITLE, SUBTITLE)
    style.legend(ax_p, ncol=len(SERIES), handles=[handles[s] for s, _, _ in SERIES],
                 handletextpad=0.3, columnspacing=1.4)
    style.source(fig, SOURCE)
    return [style.save(fig, out, NAME)]
