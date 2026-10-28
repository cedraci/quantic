"""Time to solution at T0: the classical methods against QAOA on the device.

Caption. A dot plot of TTS99 at epsilon = 1% on the eight T0 cells without CVaR
(rows, top to bottom: none, D1, D2, D4, D1+D2, D1+D4, D2+D4, D1+D2+D4; T0 is
3 assets x 4 buckets, seeds 2026-2030), x in seconds on a log scale. TTS99 is
t_run x ln 0.01 / ln(1 - p): the time to reach a schedule within 1% of the
proven optimum with 99% confidence, from the mean run time t_run and the share
p of runs that got there; it equals t_run when p = 1 and is censored (no value)
when p = 0. The six series: HiGHS proof (exact_pwl_milp) and the sound CP-SAT
re-run from the M4 study (m4-cells.csv, m4-cpsat-cells.csv), each run at
budgets of 0.1, 1, 10 and 60 s, the dot being the smallest TTS99 over those
budgets; greedy, simulated annealing and tabu from the M5 store
(m5-p2-cells.csv), 1,000-read runs with no clock, matched to the device's
1,000 shots, the better of the binary and domain-wall encodings; and QAOA on
ibm_fez (m5-p2-cells.csv, solver qaoa_ibm, reps = 2 at its trained angles),
where t_run is the QPU time of one 1,000-shot job (2.0-2.5 s) and p the share
of the cell's 3 to 15 jobs within 1%. Only the device's values are labelled. A
hollow device marker in the "censored" column beyond the end of the axis marks
a cell where no job reached 1%, so TTS99 does not exist. How to read it: the
classical times are wall-clock; the device's are QPU seconds only, with no
training, transpilation or queue time, so the gap understates the real one.
Where the device succeeds at all it needs 22.7 s (D1+D2+D4 and D2+D4), 28.4 s
(D1+D2), 63.1 s (D2) and 133.5 s (D1+D4 and D4); none and D1 are censored. Greedy
finds the optimum in 0.16-0.44 s on every cell, simulated annealing in
0.57-6.6 s, tabu in 42-74 s: 1,000 reads are slow for tabu, whose marker sits right
of the device's on D2, D1+D2, D2+D4 and D1+D2+D4. HiGHS's best-over-budgets
TTS99 is 0.09-0.79 s (above 0.5 s on none, D1+D2 and D2), and the sound CP-SAT
is faster than HiGHS on all eight cells, at 0.04-0.16 s.
"""

from pathlib import Path

import polars as pl

from m5_figures import style

NAME = "tts_t0"
TTS = "tts99_s_0.01"
CELLS = ["none", "D1", "D2", "D4", "D1+D2", "D1+D4", "D2+D4", "D1+D2+D4"]
X_MIN, X_MAX = 0.03, 200.0  # the axis; 0.03 rather than 0.1 so CP-SAT's 0.04 s cells fit
X_CENSORED = 300.0  # the column just beyond the axis end
TICKS = {0.1: "0.1 s", 1.0: "1 s", 10.0: "10 s", 60.0: "1 min", 120.0: "2 min"}
# Vertical offsets inside a row, in row units, so that two classical marks at the same
# time stay visible (HiGHS and annealing on D2); the device sits on the row line.
DODGE = {"exact_pwl_milp": 0.2, "cpsat": 0.1, "greedy": 0.0, "simulated_annealing": -0.1,
         "tabu": -0.2, "qaoa_ibm": 0.0}
TITLE = "Time to solution at T0: the classical methods against QAOA on the device"
SUBTITLE = "TTS99 at ε = 1%. Classical: wall-clock. Device: QPU seconds only, no training, no queue"
SOURCE = ("Source: m4-cells.csv, m4-cpsat-cells.csv, m5-p2-cells.csv. "
          "Regenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")


def _t0(df: pl.DataFrame, solver: str) -> pl.DataFrame:
    return df.filter((pl.col("tier") == "T0") & (pl.col("solver") == solver))


def _min_by_cell(df: pl.DataFrame) -> dict[str, float | None]:
    """The smallest non-null TTS99 of each dial combination; None when all are censored."""
    rows = df.group_by("dials").agg(pl.col(TTS).min())
    return dict(zip(rows["dials"].to_list(), rows[TTS].to_list(), strict=True))


def _series(data: dict[str, pl.DataFrame]) -> list[tuple[str, dict[str, float | None]]]:
    """(solver, {dials: TTS99}) in legend order; the device last so it draws on top."""
    m4, cpsat, m5 = data["m4-cells"], data["m4-cpsat-cells"], data["m5-p2-cells"]
    reads = m5.filter(pl.col("max_samples") == 1000)  # the matched 1,000-read sampler runs
    return [
        ("exact_pwl_milp", _min_by_cell(_t0(m4, "exact_pwl_milp"))),  # best over 4 budgets
        ("cpsat", _min_by_cell(_t0(cpsat, "cpsat"))),
        ("greedy", _min_by_cell(_t0(reads, "greedy"))),  # best of the two encodings
        ("simulated_annealing", _min_by_cell(_t0(reads, "simulated_annealing"))),
        ("tabu", _min_by_cell(_t0(reads, "tabu"))),
        ("qaoa_ibm", _min_by_cell(_t0(m5, "qaoa_ibm"))),  # one binary row per cell
    ]


def _label_device(ax, device: dict[str, float | None], classical: list[dict], y_of: dict) -> None:
    """The device's value beside each marker, on the left when a classical mark is close right."""
    for cell, value in device.items():
        if value is None or cell not in y_of:
            continue
        crowded = any(value < (x or 0.0) < value * 2.6
                      for values in classical for c, x in values.items() if c == cell)
        side = -1 if crowded else 1
        ax.annotate(f"{value:.1f} s", (value, y_of[cell]), xytext=(8 * side, 0),
                    textcoords="offset points", ha="left" if side > 0 else "right",
                    va="center", fontsize=8, color=style.INK_2,
                    bbox={"boxstyle": "square,pad=0.15", "facecolor": style.SURFACE,
                          "edgecolor": "none"})


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    series = _series(data)
    y_of = {cell: len(CELLS) - 1 - i for i, cell in enumerate(CELLS)}  # "none" on top

    fig, ax = style.new_figure(8.0, 5.0)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.13)
    for solver, values in series:
        size = 7.5 if solver == "qaoa_ibm" else 6.0
        points = [(x, y_of[c] + DODGE[solver]) for c, x in values.items()
                  if c in y_of and x is not None]
        ax.plot([x for x, _ in points], [y for _, y in points], label=style.NAMES[solver],
                **style.marker_kw(style.COLORS[solver], style.MARKERS.get(solver, "o"), size))

    device = series[-1][1]
    censored = [y_of[c] for c in CELLS if device.get(c) is None]
    ax.plot([X_CENSORED] * len(censored), censored,
            **style.marker_kw(style.DEVICE, "o", 7.5, filled=False))
    _label_device(ax, device, [values for _, values in series[:-1]], y_of)

    ax.set_xscale("log")
    ax.set_xlim(X_MIN, X_CENSORED * 1.45)
    ax.set_xticks(list(TICKS), labels=list(TICKS.values()))
    ax.minorticks_off()
    ax.set_yticks([y_of[c] for c in CELLS], labels=CELLS)
    ax.set_ylim(-0.7, len(CELLS) - 0.3)
    ax.set_xlabel("TTS99 at ε = 1%, log scale")
    style.finish(ax, grid="x")
    ax.spines["bottom"].set_bounds(X_MIN, X_MAX)
    ax.annotate("censored", (X_CENSORED, 0), xycoords=("data", "axes fraction"), xytext=(0, -4),
                textcoords="offset points", ha="center", va="top", fontsize=8, color=style.MUTED)

    style.heading(ax, TITLE, SUBTITLE)
    style.legend(ax, ncol=6)
    style.source(fig, SOURCE)
    return [style.save(fig, out, NAME)]
