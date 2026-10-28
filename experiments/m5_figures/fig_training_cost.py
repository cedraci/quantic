"""Training the angles: seconds per QUBO against qubit count.

Caption. Wall-clock seconds to train the QAOA angles of one QUBO (``train_s``) against
its qubit count (``n_vars``), from ``m5-p2-qaoa_training.csv``. Every training is a blue
point: the 47 reps = 2 QUBOs of the M5 study, 31 binary and 16 domain-wall, drawn with
the same marker and spread sideways where several share a qubit count and a similar
time; the line joins the median at each qubit count. The y axis is logarithmic, labelled
in seconds, minutes, hours and days. Training is COBYLA with 200 evaluations of an
exact-statevector CVaR at alpha = 0.25; 45 of the 47 trainings used all 200 evaluations.
Read it as an exponential: the median grows about 2.3x per added qubit, from 6 s at 16
qubits to 66 min at 24. The hollow marker at 28 qubits is the study's dry run, not a
training: 413 s per evaluation x 200 evaluations = 23 h and 20 GB of statevector, so it
was not run. The gray hairline at 1 s is the classical reference: HiGHS, at its 1 s
budget, has every one of these instances within 1% of the optimum in at most 1.0 s (cell
means 0.24-0.40 s; one run stopped at the budget without a proof), and every training
takes longer than HiGHS on the same instance, at least 8 times longer at reps = 2 and 2
times at reps = 1, each against its own store's HiGHS (1.2 against the reps = 2 store's).
The reps = 1 rung's 47 trainings (``m5-p1-qaoa_training.csv``) are
drawn the same way in orange: 3.6 h in all against 17.3 h at reps = 2, a median of 0.8 s
at 16 qubits and 16 min at 24, and COBYLA stopped before its 200 evaluations in 46 of
the 47 (31 to 146 evaluations), where at reps = 2 it ran out of evaluations in 45.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from m5_figures import style

NAME = "training_cost"
TITLE = "Training the angles: seconds per QUBO against qubit count"
SOURCE = ("Source: docs/figures/m5/data/m5-p2-qaoa_training.csv and m5-p1-qaoa_training.csv. "
          "Regenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")

# The study's 28-qubit dry run, which is in no CSV: one COBYLA evaluation of the
# exact statevector took 413 s and the statevector needed 20 GB, so the full
# 200-evaluation training (82,600 s, about 23 h) was not run.
DRY_RUN_QUBITS = 28
DRY_RUN_S_PER_EVAL = 413.0
DRY_RUN_EVALS = 200
DRY_RUN_GB = 20
DRY_RUN_S = DRY_RUN_S_PER_EVAL * DRY_RUN_EVALS  # 82,600 s

# HiGHS at its 1 s budget has every M5 instance within 1% of the optimum within this bound
# (slowest run 1.005 s; m5-p2-cells.csv and the stores).
HIGHS_PROOF_S = 1.0

X_LIM = (14.8, 28.9)
Y_LIM = (0.2, 2.5e5)
X_TICKS = [16, 18, 19, 20, 21, 22, 23, 24, DRY_RUN_QUBITS]
Y_TICKS = [(1, "1 s"), (10, "10 s"), (60, "1 min"), (600, "10 min"), (3600, "1 h"),
           (36_000, "10 h"), (86_400, "1 d")]
MARKER_PT = 6.0  # 8 px at 96 dpi
SWARM_CAP = 0.45  # the widest sideways offset of a point, in qubits
SWARM_CLEARANCE = 0.8  # the closest two centres come, in marker diameters; the ring does the rest
LABEL_GAP = 0.12  # qubits between the edge of a swarm and its direct label
HALO = {"boxstyle": "square,pad=0.15", "facecolor": style.SURFACE, "edgecolor": "none"}


def _medians(df: pl.DataFrame) -> pl.DataFrame:
    return (df.group_by("n_vars").agg(pl.col("train_s").median().alias("median_s"))
            .sort("n_vars"))


def _inch_scales(fig, ax) -> tuple[float, float]:
    """Inches per qubit on x and per decade on y, from the laid-out axes box."""
    pos = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ux = pos.width * fig_w / (X_LIM[1] - X_LIM[0])
    uy = pos.height * fig_h / (math.log10(Y_LIM[1]) - math.log10(Y_LIM[0]))
    return ux, uy


def _swarm(ys: list[float], ux: float, uy: float) -> list[float]:
    """Sideways offsets, in qubits, that keep same-x points from hiding one another.

    Points go in from the smallest value up, each to the first offset in
    0, +s, -s, +2s, -2s, ... (out to the cap) that clears every placed point by
    SWARM_CLEARANCE diameters; when none does, to the offset with the most room.
    """
    clearance = SWARM_CLEARANCE * MARKER_PT / 72.0
    step = clearance / 4
    n_steps = int(SWARM_CAP * ux / step)
    candidates = [0.0] + [sign * k * step for k in range(1, n_steps + 1) for sign in (1, -1)]
    placed: list[tuple[float, float]] = []
    offsets = [0.0] * len(ys)
    for i in sorted(range(len(ys)), key=lambda i: ys[i]):
        y_in = math.log10(ys[i]) * uy
        room = [(min((math.hypot(dx - px, y_in - py) for px, py in placed), default=math.inf),
                 dx) for dx in candidates]
        best = next((dx for gap, dx in room if gap >= clearance), max(room)[1])
        placed.append((best, y_in))
        offsets[i] = best / ux
    return offsets


def _points(ax, df: pl.DataFrame, color: str, ux: float, uy: float
            ) -> dict[int, tuple[float, float]]:
    """Draw one series' trainings as a swarm per qubit count; return each swarm's x extent."""
    extents: dict[int, tuple[float, float]] = {}
    for (n_vars,), group in df.group_by("n_vars"):
        ys = group["train_s"].to_list()
        xs = [n_vars + dx for dx in _swarm(ys, ux, uy)]
        ax.plot(xs, ys, zorder=3, **style.marker_kw(color, "o", MARKER_PT))
        extents[n_vars] = (min(xs), max(xs))
    return extents


def _human(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    series = [(data["m5-p2-qaoa_training"], style.BLUE, "reps = 2")]
    if "m5-p1-qaoa_training" in data:
        series.append((data["m5-p1-qaoa_training"], style.ORANGE, "reps = 1"))
    two_series = len(series) > 1
    trainings = series[0][0]

    fig, ax = style.new_figure(8.0, 5.0)
    ax.set_yscale("log")
    ax.set_ylim(*Y_LIM)
    ax.yaxis.set_major_locator(plt.FixedLocator([v for v, _ in Y_TICKS]))
    ax.yaxis.set_major_formatter(plt.FixedFormatter([t for _, t in Y_TICKS]))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(*X_LIM)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel("Qubits (QUBO variables)")
    ax.set_ylabel("Training time per QUBO (log scale)")

    # The median lines carry the legend handles; the points go on top of them later.
    for df, color, label in series:
        med = _medians(df)
        ax.plot(med["n_vars"].to_list(), med["median_s"].to_list(), color=color,
                linewidth=1.5, zorder=2, label=label)

    # The classical reference: a gray hairline with a muted label above it, starting right of
    # the 16-qubit swarm, which reaches down to this line at reps = 1.
    ax.axhline(HIGHS_PROOF_S, color=style.CLASSICAL, linewidth=0.8, zorder=1)
    ax.text(16.75, HIGHS_PROOF_S * 1.15,
            f"HiGHS, 1 s budget: every instance within 1% of its optimum in at most "
            f"{HIGHS_PROOF_S:.1f} s",
            ha="left", va="bottom", fontsize=8, color=style.MUTED, zorder=4)

    # The dry run at 28 qubits: a hollow marker, annotated to its left.
    ax.plot([DRY_RUN_QUBITS], [DRY_RUN_S], zorder=3,
            **style.marker_kw(style.BLUE, "o", MARKER_PT, filled=False))
    ax.text(DRY_RUN_QUBITS - 0.45, DRY_RUN_S,
            f"{DRY_RUN_QUBITS} qubits, dry run: {DRY_RUN_S_PER_EVAL:.0f} s per evaluation"
            f" × {DRY_RUN_EVALS} = {DRY_RUN_S / 3600:.0f} h\nand {DRY_RUN_GB} GB; not run",
            ha="right", va="center", fontsize=8, color=style.INK_2, linespacing=1.3,
            bbox=HALO, zorder=4)

    n_binary = trainings.filter(pl.col("encoding") == "binary").height
    n_domain_wall = trainings.filter(pl.col("encoding") == "domain_wall").height
    what = (f"Each reps = 2 training ({n_binary} binary, {n_domain_wall} domain-wall QUBOs)"
            " and the median per qubit count")
    if two_series:
        what = "Each training (points) and the median per qubit count, by QAOA depth"
    style.heading(ax, TITLE, what, legend_row=two_series)
    style.finish(ax, grid="y")
    if two_series:
        style.legend(ax, ncol=2)
    style.source(fig, SOURCE)
    fig.tight_layout()

    # The points last: the swarm needs the laid-out axes size in inches.
    ux, uy = _inch_scales(fig, ax)
    extents = [_points(ax, df, color, ux, uy) for df, color, _ in series]

    # Direct labels: the two reps = 2 medians the story quotes, nothing else.
    medians = _medians(trainings)
    first, last = medians.row(0, named=True), medians.row(-1, named=True)
    label_kw = {"va": "center", "fontsize": 8, "color": style.INK_2, "fontweight": "semibold",
                "bbox": HALO, "zorder": 4}
    ax.text(extents[0][first["n_vars"]][0] - LABEL_GAP, first["median_s"],
            _human(first["median_s"]), ha="right", **label_kw)
    ax.text(extents[0][last["n_vars"]][1] + LABEL_GAP, last["median_s"],
            _human(last["median_s"]), ha="left", **label_kw)
    return [style.save(fig, out, NAME)]
