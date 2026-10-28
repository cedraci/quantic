"""The error-free fraction of a shot against two-qubit gates, at ibm_fez's median CZ error.

Caption. The curve is (1 - 0.0025)^n: the probability that a shot passes n two-qubit (CZ)
gates without a two-qubit error, at 0.25% per gate -- the median CZ error of ibm_fez's
calibration read after the first pass (properties last updated 2026-10-27 16:32); the next
morning's calibration (last updated 2026-10-28 09:49) has the same median. The ticks along
the bottom mark the routed two-qubit gate count (``routed_2q``) of each of the 30 device
circuits, the ``qaoa_ibm`` binary-encoding rows of
``docs/figures/m5/data/m5-p2-qaoa_noise.csv`` (reps = 2); circuits with the same count
overlap, so fewer than 30 ticks show. The two labelled points are the shallowest and the
deepest of them: 735 CZ gates leave 16% of shots error-free, 2,044 leave 0.6%, so the
study's circuits span exactly the stretch where the curve falls from about one shot in six
to one in 170. The curve ignores single-qubit and readout errors (readout error is 1.1% per
qubit on that calibration, so a 16-24 qubit readout is itself wrong 16-23% of the time), so
it is an upper bound on the shots that survive. The reps = 1 rung's 31 device circuits
(``m5-p1-qaoa_noise``) form a second, orange rug row: 313 to 952 CZ gates, where the curve
gives 46% down to 9% of shots error-free, against 16% down to 0.6% for the reps = 2
circuits.
"""

from pathlib import Path

import polars as pl

from m5_figures import style

NAME = "error_free"
TITLE = ("The error-free fraction of a shot against two-qubit gates, "
         "at ibm_fez's median CZ error")
# Median CZ error of ibm_fez's calibration of 2026-10-27 (last updated 16:32);
# the calibration of 2026-10-28 (last updated 09:49) has the same median.
P_CZ = 0.0025
X_MAX = 2_600
STRIP = 10.0  # percentage points below 0% reserved for the rug


def _error_free(n_gates: float) -> float:
    """The percentage of shots that pass n_gates CZ gates without a two-qubit error."""
    return 100.0 * (1.0 - P_CZ) ** n_gates


def _pct(value: float) -> str:
    return f"{value:.0f}%" if value >= 10 else f"{value:.1f}%"


def _routed_2q(frame: pl.DataFrame) -> list[float]:
    """The routed two-qubit gate count of every device circuit of a study table."""
    rows = frame.filter((pl.col("solver") == "qaoa_ibm") & (pl.col("encoding") == "binary"))
    return sorted(rows["routed_2q"].drop_nulls().to_list())


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    rugs = [(_routed_2q(data["m5-p2-qaoa_noise"]), style.MUTED, "reps = 2")]
    if "m5-p1-qaoa_noise" in data:
        rugs.append((_routed_2q(data["m5-p1-qaoa_noise"]), style.ORANGE, "reps = 1"))
    gates = rugs[0][0]

    fig, ax = style.new_figure(8.0, 4.4)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.14)

    xs = list(range(0, X_MAX + 1, 5))
    ax.plot(xs, [_error_free(x) for x in xs], color=style.DEVICE, linewidth=2,
            label=f"(1 - {P_CZ:.2%})^n")

    # The rug: one short tick per device circuit, in a strip under the 0% line.
    levels = [0.045] if len(rugs) == 1 else [0.066, 0.024]
    size = 9 if len(rugs) == 1 else 7
    for (values, color, label), level in zip(rugs, levels, strict=True):
        ax.plot(values, [level] * len(values), transform=ax.get_xaxis_transform(),
                linestyle="none", marker="|", markersize=size, markeredgewidth=0.8,
                markeredgecolor=color, color=color,
                label=f"{len(values)} device circuits, {label}")
    if len(rugs) == 1:
        ax.text(20, levels[0], f"{len(gates)} device circuits, reps = 2",
                transform=ax.get_xaxis_transform(), ha="left", va="center",
                fontsize=7.5, color=style.MUTED)
        rug_note = f"ticks: routed CZ count of the {len(gates)} device circuits (reps = 2)"
    else:
        style.legend(ax, ncol=1, above=False, loc="upper right")
        rug_note = "ticks: routed CZ count of each device circuit, one row per reps"

    # The shallowest and the deepest device circuit, read off the curve.
    for n_gates, dy in ((min(gates), 14.0), (max(gates), 13.0)):
        p = _error_free(n_gates)
        ax.plot([n_gates], [p], **style.marker_kw(style.DEVICE, "o", 6.5))
        ax.annotate(f"{n_gates:,.0f} CZ: {_pct(p)}", xy=(n_gates, p),
                    xytext=(n_gates + 90, p + dy), ha="left", va="center",
                    fontsize=8.5, color=style.MUTED,
                    arrowprops={"arrowstyle": "-", "color": style.MUTED,
                                "linewidth": 0.6, "shrinkB": 4})

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-STRIP, 100)
    ax.set_xticks(range(0, X_MAX + 1, 500), [f"{v:,}" for v in range(0, X_MAX + 1, 500)])
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.set_xlabel("Two-qubit (CZ) gates in the routed circuit")
    ax.set_ylabel("Shots without a two-qubit error")
    style.heading(ax, TITLE, f"(1 - {P_CZ:.2%})^n for n CZ gates; {rug_note}",
                  legend_row=False)
    style.finish(ax, grid="y")
    tables = "m5-p2-qaoa_noise.csv" + (" and m5-p1-qaoa_noise.csv" if len(rugs) > 1 else "")
    style.source(fig, f"Source: docs/figures/m5/data/{tables} (qaoa_ibm, binary, routed_2q). "
                      "Regenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")
    return [style.save(fig, out, NAME)]
