"""QPU seconds spent against the 300 s cap: the quota ledger as a running total.

Caption. The cumulative QPU seconds of the 122 ``ibm_fez`` jobs of the M5 studies, in
submission order, from ``m5-hardware-jobs.csv`` (the ledger ``~/.quantic/ibm-quota.json``
joined with the stores' hardware rows by ``experiments/m5-hardware-jobs.py``, which names
each job's lane and pass). ``usage_s`` is IBM's QPU charge for a job, the quantum seconds
it counts against the account; the step rises by that charge at each job. Every job cost
2 or 3 s against a per-job limit (``max_execution_time``) of 60 s, 20 s for the smoke job
and 57 down to 40 s for the last nine jobs of pass 3, as the cap's remainder shrank.
The brackets under the axis mark the five runs of the hardware lane: the smoke job of
2026-10-14 (job 1), the three reps = 2 passes over the 30 runnable instances (two on
2026-10-27, one on 2026-10-28) and the reps = 1 lane over its 31 instances on 2026-10-28,
whose segment of the step is orange. The red hairline is the study's rule, 300 s per
rolling 28 days; the gray one IBM's Open-plan allowance of 600 s. The step ends at 262 s:
the cap never bound, and IBM's own counter read 262 s after the last job. The ranges and
their times are derived from ``submitted_at`` (local time of the machine that
regenerates the figure).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from m5_figures import style

NAME = "qpu_ledger"
TITLE = "QPU seconds spent against the 300 s cap"
STUDY_CAP_S = 300.0  # quantic.solvers.quantum.quota.QUOTA_CEILING_S, per rolling 28 days
OPEN_PLAN_S = 600.0  # IBM's Open plan allowance over the same window
Y_TOP = 650.0
LANE_COLOR = {"reps2": style.BLUE, "smoke": style.BLUE, "reps1": style.ORANGE}


def _local(epoch_s: float) -> datetime:
    return datetime.fromtimestamp(epoch_s)


def _hm(epoch_s: float) -> str:
    """Local wall-clock time rounded to the nearest minute, as 16:52."""
    return f"{_local(round(epoch_s / 60) * 60):%H:%M}"


def _ledger(jobs: pl.DataFrame) -> pl.DataFrame:
    """The jobs in submission order, numbered from 1, with their running total."""
    return (jobs.sort("submitted_at")
            .with_columns(pl.int_range(1, pl.len() + 1).alias("job"),
                          pl.col("usage_s").cum_sum().alias("cum_s")))


def _runs(jobs: pl.DataFrame) -> list[dict]:
    """Each run of the lane: a maximal block of consecutive jobs of one lane and pass."""
    block = ((pl.col("lane") != pl.col("lane").shift()) | (pl.col("pass") != pl.col("pass").shift())
             ).fill_null(True).cum_sum()
    runs = (jobs.with_columns(block.alias("block"))
            .group_by("block", maintain_order=True)
            .agg(pl.col("lane").first(), pl.col("pass").first(), pl.col("job").min().alias("first"),
                 pl.col("job").max().alias("last"), pl.col("submitted_at").min().alias("t0"),
                 pl.col("submitted_at").max().alias("t1")))
    return list(runs.iter_rows(named=True))


def _label(run: dict, previous_day: str | None) -> str:
    day = f"{_local(run['t0']):%Y-%m-%d}"
    when = f"{day} " if day != previous_day else ""
    if run["lane"] == "smoke":
        return f"smoke job\n{day}"
    rung = "reps = 1" if run["lane"] == "reps1" else f"reps = 2, pass {run['pass']}"
    return f"{rung}\n{when}{_hm(run['t0'])}–{_hm(run['t1'])}"


def _mark_ranges(ax, ranges: list[tuple[int, int, str]]) -> None:
    """Extent brackets under the baseline, in data x, with muted text below each."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    axes_per_pt = 1 / (72 * ax.figure.get_figheight() * ax.get_position().height)
    bar, tick, text = -22 * axes_per_pt, 4 * axes_per_pt, -27 * axes_per_pt
    below = -26 * axes_per_pt  # a single-job range is labelled one row lower, clear of the next
    for first, last, label in ranges:
        x0, x1 = first - 0.4, last + 0.4
        ax.plot([x0, x0, x1, x1], [bar + tick, bar, bar, bar + tick], transform=trans,
                color=style.BASELINE, linewidth=0.6, clip_on=False, solid_capstyle="butt")
        single = first == last
        ax.text(x0 if single else (x0 + x1) / 2, text + (below if single else 0), label,
                transform=trans,
                ha="left" if single else "center", va="top", fontsize=7.5, color=style.MUTED,
                linespacing=1.3)


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    jobs = _ledger(data["m5-hardware-jobs"])
    n_jobs = jobs.height
    total = float(jobs["cum_s"][-1])
    runs = _runs(jobs)
    ranges, day = [], None
    for run in runs:
        ranges.append((run["first"], run["last"], _label(run, day)))
        day = f"{_local(run['t0']):%Y-%m-%d}"

    fig, ax = style.new_figure(8.0, 4.9)
    fig.subplots_adjust(left=0.085, right=0.965, top=0.82, bottom=0.24)

    # The running total: a step that rises by each job's charge at that job, coloured by lane.
    xs = [1, *jobs["job"].to_list()]
    ys = [0.0, *jobs["cum_s"].to_list()]
    for run in runs:
        # From the previous total at this run's first job, then each of its jobs' totals.
        first, last = run["first"], run["last"]
        # The last point carries the run's total across to where the next run starts.
        ax.step([first, *xs[first:last + 1], last + 1],
                [ys[first - 1], *ys[first:last + 1], ys[last]], where="post",
                color=LANE_COLOR[run["lane"]], linewidth=2, solid_joinstyle="miter",
                solid_capstyle="butt", zorder=3)
    ax.text(n_jobs + 2.0, total, f"{total:.0f} s", ha="left", va="center", fontsize=9,
            color=style.INK, fontweight="semibold")

    # The two limits, labelled on the line.
    for level, color, label in ((STUDY_CAP_S, style.STATUS_CRITICAL,
                                 "cap: 300 s per rolling 28 days (study rule)"),
                                (OPEN_PLAN_S, style.MUTED, "IBM Open plan: 600 s")):
        ax.axhline(level, color=color, linewidth=0.8, zorder=2)
        ax.text(1.0, level + 7, label, ha="left", va="bottom", fontsize=8, color=style.INK_2)

    ax.set_xlim(0, n_jobs + 9)
    ax.set_ylim(0, Y_TOP)
    xticks = [1, *range(20, n_jobs - 5, 20), n_jobs]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"job {xticks[0]}", *(str(t) for t in xticks[1:])])
    yticks = list(range(0, int(Y_TOP) + 1, 100))
    ax.set_yticks(yticks)
    ax.set_yticklabels(["0", *(f"{t} s" for t in yticks[1:])])
    ax.set_ylabel("QPU seconds, cumulative")
    _mark_ranges(ax, ranges)

    style.heading(ax, TITLE, f"Cumulative QPU seconds of the {n_jobs} QAOA jobs on ibm_fez, "
                             "in submission order")
    handles = [Line2D([], [], color=style.BLUE, linewidth=2),
               Line2D([], [], color=style.ORANGE, linewidth=2)]
    style.legend(ax, ncol=2, handles=handles, labels=["reps = 2 jobs (and the smoke job)",
                                                       "reps = 1 jobs"])
    style.finish(ax, grid="y")
    style.source(fig, "Source: docs/figures/m5/data/m5-hardware-jobs.csv (the ledger "
                      "~/.quantic/ibm-quota.json joined with the stores' hardware rows). "
                      "Regenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")
    return [style.save(fig, out, NAME)]
