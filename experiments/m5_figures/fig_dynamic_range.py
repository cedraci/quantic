"""Dynamic range and qubit count against the noiseless success.

Caption. One point per binary-encoded T0 instance (31 points). The x axis is
the QUBO's dynamic range, the ratio of its largest to its smallest non-zero
coefficient magnitude (``dynamic_range`` in ``m5-p2-qaoa_training.csv``; the
constant is excluded), on a log scale. The y axis is the noiseless QAOA's
success at epsilon = 1 %: the share of its 20 runs on that instance whose
best sample is feasible without repair and within 1 % of the proven optimum
(``p_success_0.01`` of the ``qaoa_ideal`` rows of ``m5-p2-qaoa_noise.csv``).
The two tables are joined on (dials, instance_seed) for ``encoding ==
binary``. The color is the qubit count (``n_vars``) in four ordinal bins,
light to dark: 16, 18-20, 21-22, 23-24 qubits. How to read it: the dark
23-24 qubit points sit at or near zero whatever their dynamic range, the
light 16 qubit points at or near one; the note gives the Spearman rank
correlation of the success with each quantity, computed from these 31 points
(``scipy.stats.spearmanr``): the qubit count tracks the noiseless success
(-0.77, p = 4e-7), the dynamic range does not (-0.16, p = 0.38). The two
labelled points are the extremes that make the point: D1+D4-2030 has one of
the widest ranges and a success of 1.0 on 19 qubits; D1-2029, a similar
range, a success of 0.1 on 24 qubits.
"""

import math
from pathlib import Path

import polars as pl
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter
from scipy.stats import spearmanr

from m5_figures import style

NAME = "dynamic_range"
TITLE = "Dynamic range and qubit count against the noiseless success"
SUBTITLE = "Noiseless QAOA success at ε = 1 % per binary instance, against QUBO dynamic range"
SOURCE = ("Source: m5-p2-qaoa_training.csv and m5-p2-qaoa_noise.csv (qaoa_ideal rows), "
          "docs/figures/m5/data. "
          "Regenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")
# Ordinal qubit-count bins: (legend label, lowest n_vars, highest n_vars, BLUE_RAMP step).
BINS = [("16 qubits", 16, 16, 250), ("18-20", 18, 20, 400),
        ("21-22", 21, 22, 550), ("23-24", 23, 24, 700)]
# Direct labels, keyed by (dials, instance_seed): x offset in points and the text anchor.
LABELLED = {("D1+D4", 2030): (-8, "right"), ("D1", 2029): (8, "left")}


def points(data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The 31 binary instances: training-time QUBO diagnostics joined with the noiseless success."""
    training = (data["m5-p2-qaoa_training"]
                .filter(pl.col("encoding") == "binary")
                .select("dials", "instance_seed", "n_vars", "dynamic_range"))
    noiseless = (data["m5-p2-qaoa_noise"]
                 .filter((pl.col("solver") == "qaoa_ideal") & (pl.col("encoding") == "binary"))
                 .select("dials", "instance_seed", "p_success_0.01"))
    return training.join(noiseless, on=["dials", "instance_seed"], how="inner")


def p_text(p: float) -> str:
    """A p-value as "0.38" or, below 0.01, as "4e-7"."""
    if p >= 0.01:
        return f"{p:.2f}"
    exponent = math.floor(math.log10(p))
    mantissa = round(p / 10**exponent)
    if mantissa == 10:
        mantissa, exponent = 1, exponent + 1
    return f"{mantissa}e{exponent}"


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    pts = points(data)
    success = pts["p_success_0.01"].to_numpy()
    fig, ax = style.new_figure(8.0, 5.0)

    plotted = 0
    for label, lowest, highest, step in BINS:
        sub = pts.filter(pl.col("n_vars").is_between(lowest, highest))
        plotted += sub.height
        ax.plot(sub["dynamic_range"].to_numpy(), sub["p_success_0.01"].to_numpy(),
                label=label, **style.marker_kw(style.BLUE_RAMP[step], "o", 8))
    if plotted != pts.height:
        raise ValueError(f"{pts.height - plotted} instances fall outside the qubit bins")

    for row in pts.iter_rows(named=True):
        offset = LABELLED.get((row["dials"], row["instance_seed"]))
        if offset is None:
            continue
        dx, anchor = offset
        ax.annotate(f"{row['dials']}-{row['instance_seed']}",
                    (row["dynamic_range"], row["p_success_0.01"]),
                    xytext=(dx, 0), textcoords="offset points", ha=anchor, va="center",
                    fontsize=8, color=style.MUTED)

    rho_range, p_range = spearmanr(pts["dynamic_range"].to_numpy(), success)
    rho_qubits, p_qubits = spearmanr(pts["n_vars"].to_numpy(), success)
    ax.text(0.015, 0.06,
            f"Spearman: range {rho_range:.2f} (p = {p_text(p_range)}),\n"
            f"qubits {rho_qubits:.2f} (p = {p_text(p_qubits)})",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=style.MUTED,
            linespacing=1.4)

    ax.set_xscale("log")
    ax.set_xlim(8e3, 1.1e7)
    ax.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.06, 1.06)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("QUBO dynamic range: largest / smallest non-zero coefficient (log scale)")
    ax.set_ylabel("Noiseless QAOA success, ε = 1 % (share of 20 runs)")
    style.finish(ax, grid="both")

    style.legend(ax, ncol=4)
    style.heading(ax, TITLE, SUBTITLE)
    style.source(fig, SOURCE)
    return [style.save(fig, out, NAME)]
