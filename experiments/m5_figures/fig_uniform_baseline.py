"""Figure ``uniform_baseline``: feasible samples, the device against uniform random bitstrings.

Caption. One point per instance and backend for the 30 binary instances (at most 24
qubits) that ran on ibm_fez at reps = 2, from ``docs/figures/m5/data/m5-p2-uniform.csv``.
The x axis is ``uniform_frac``, the share of uniform random bitstrings that decode to a
feasible schedule: exact at 16 qubits (every one of the 65,536 bitstrings was decoded) and
estimated from 100,000 draws above that. The y axis is the feasible share of the samples
a backend returned, on the same log scale, with tick labels in percent. Blue: QAOA on
ibm_fez, the mean over the instance's three jobs of (feasible samples / 1,000 shots),
parsing ``hw_feasible`` such as "26+20+29" as the per-job counts; an instance with no
feasible sample in any job (3 of the 30) is drawn hollow at the floor of 0.001%. Aqua: the
same circuit simulated without noise, ``ratio_ideal`` times ``uniform_frac``. The diagonal
``y = x`` is what uniform random bitstrings give; the two fainter diagonals are 10 and 100
times that. Read it as: the device points hug the uniform line -- pooled over the 90 jobs,
462 feasible samples in 90,000 shots against 233.3 expected from uniform bitstrings, a
ratio of 1.98 -- while the noiseless circuit sits about 50 times above it (median ratio
50.3 over the 30 instances, from 1.8 on D1+D2-2030 to 4,515 on D4-2026, where a single
feasible device sample in 3,000 shots is already 33 times the uniform rate).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from m5_figures import style

NAME = "uniform_baseline"
TITLE = "Feasible samples: the device against uniform random bitstrings"
SUBTITLE = "Feasible share of samples against that of uniform random bitstrings, per instance"
SOURCE = ("Source: m5-p2-uniform.csv (uniform_frac, hw_feasible, hw_jobs, ratio_ideal)  ·  "
          "regenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")
SHOTS = 1_000  # shots per hardware job
FLOOR = 1e-5  # where a device mean of zero is drawn, hollow
X_LIM = (5e-6, 0.12)
Y_LIM = (5e-6, 0.5)
TICKS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
TICK_LABELS = ["0.001%", "0.01%", "0.1%", "1%", "10%"]
# The reference diagonals y = k x: (k, label, color, line width).
DIAGONALS = ((1, "uniform", style.BASELINE, 1.2), (10, "10×", style.GRID, 0.9),
             (100, "100×", style.GRID, 0.9))


def _rows(data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The 30 device instances with both feasible shares, parsed from the uniform baseline."""
    parts = pl.col("hw_feasible").cast(pl.Utf8).str.split("+")
    counts = parts.list.eval(pl.element().cast(pl.Float64))
    rows = (data["m5-p2-uniform"]
            .with_columns(feasible=counts.list.sum(), n_parts=parts.list.len(),
                          device=counts.list.mean() / SHOTS,
                          noiseless=pl.col("ratio_ideal") * pl.col("uniform_frac"),
                          label=pl.format("{}-{}", "dials", "instance_seed"))
            .sort("uniform_frac", "label"))
    if not (rows["n_parts"] == rows["hw_jobs"]).all():
        raise ValueError("hw_feasible does not list one feasible count per job")
    return rows


def _label_diagonal(ax, k: float, text: str) -> None:
    """Name y = k x just before it leaves the axes, hanging under the line in muted text."""
    (x0, x1), (_, y1) = ax.get_xlim(), ax.get_ylim()
    end = (x1, k * x1) if k * x1 <= y1 else (y1 / k, y1)
    anchor = (end[0] / 10 ** 0.08, end[1] / 10 ** 0.08)
    (px0, py0), (px1, py1) = ax.transData.transform([(x0, k * x0), end])
    angle = math.degrees(math.atan2(py1 - py0, px1 - px0))
    ax.annotate(text, xy=anchor, xytext=(-2, -4), textcoords="offset points",
                ha="right", va="top", rotation=angle, rotation_mode="anchor",
                fontsize=7.5, color=style.MUTED)


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    rows = _rows(data)
    by_label = {r["label"]: r for r in rows.iter_rows(named=True)}
    xs = rows["uniform_frac"].to_list()
    device = rows["device"].to_list()
    noiseless = rows["noiseless"].to_list()
    hollow = [x for x, y in zip(xs, device, strict=True) if y == 0]
    filled = [(x, y) for x, y in zip(xs, device, strict=True) if y > 0]
    jobs = sorted(rows["hw_jobs"].unique().to_list())
    n_jobs = str(jobs[0]) if len(jobs) == 1 else f"{jobs[0]}–{jobs[-1]}"
    feasible = int(rows["feasible"].sum())
    shots = int((rows["hw_jobs"] * SHOTS).sum())
    expected = float(rows["expected_feasible"].sum())
    median_ideal = float(rows["ratio_ideal"].median())

    fig, ax = style.new_figure(8.0, 6.0)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.835, bottom=0.085)
    ax.set_xscale("log")
    ax.set_yscale("log")

    for k, _, color, width in DIAGONALS:
        ax.plot(X_LIM, [k * X_LIM[0], k * X_LIM[1]], color=color, linewidth=width, zorder=1)

    ax.plot(xs, noiseless, zorder=3, label="QAOA, noiseless simulation of the same circuit",
            **style.marker_kw(style.NOISELESS, "o", 6.5))
    ax.plot([x for x, _ in filled], [y for _, y in filled], zorder=4,
            label=f"QAOA on ibm_fez, mean of its {n_jobs} jobs of {SHOTS:,} shots",
            **style.marker_kw(style.DEVICE, "o", 6.5))
    ax.plot(hollow, [FLOOR] * len(hollow), zorder=4,
            **style.marker_kw(style.DEVICE, "o", 6.5, filled=False))

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    for k, text, _, _ in DIAGONALS:
        _label_diagonal(ax, k, text)

    # Four direct labels: the device on the uniform line and at the uniform floor, the
    # noiseless circuit at its highest and lowest ratio.
    on_line, at_floor = by_label["D1+D2-2030"], by_label["D4-2026"]
    found = sum(int(c) for c in str(at_floor["hw_feasible"]).split("+"))
    notes = (
        ((on_line["uniform_frac"], on_line["device"]),
         f"D1+D2-2030\n{on_line['ratio_hw']:.1f}× uniform", (7, -5), "left", "top"),
        ((at_floor["uniform_frac"], at_floor["device"] if found else FLOOR),
         f"D4-2026, uniform {100 * at_floor['uniform_frac']:g}%\n"
         f"{found or 'none'} feasible in {at_floor['hw_jobs'] * SHOTS:,} shots", (8, 0), "left",
         "center"),
        ((at_floor["uniform_frac"], at_floor["noiseless"]),
         f"D4-2026, {at_floor['ratio_ideal']:,.0f}× uniform", (8, 0), "left", "center"),
        ((on_line["uniform_frac"], on_line["noiseless"]),
         f"D1+D2-2030, {on_line['ratio_ideal']:.1f}× uniform", (-8, 0), "right", "center"),
    )
    for xy, text, offset, ha, va in notes:
        ax.annotate(text, xy=xy, xytext=offset, textcoords="offset points", ha=ha, va=va,
                    fontsize=7.5, color=style.MUTED, linespacing=1.15)
    ax.text(3e-4, 2.6e-5, f"hollow: no feasible sample in any job ({len(hollow)} instances)",
            ha="center", va="bottom", fontsize=7.5, color=style.MUTED)
    ax.text(0.11, 1.2e-4,
            f"device, pooled over {int(rows['hw_jobs'].sum())} jobs: "
            f"{feasible / expected:.2f}× uniform\n({feasible:,} feasible of {shots:,} "
            f"shots; {expected:.1f} expected)\nnoiseless, median over {rows.height} "
            f"instances: {median_ideal:.0f}× uniform",
            ha="right", va="center", fontsize=7.5, color=style.INK_2, linespacing=1.3)

    ax.set_xticks(TICKS, TICK_LABELS)
    ax.set_yticks(TICKS, TICK_LABELS)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Share of uniform random bitstrings that are feasible "
                  "(every bitstring at 16 qubits; 100,000 draws above)")
    ax.set_ylabel("Feasible share of the backend's samples")
    style.finish(ax, grid="both")

    style.heading(ax, TITLE, SUBTITLE)
    style.legend(ax, ncol=2)
    style.source(fig, SOURCE)
    return [style.save(fig, out, NAME)]
