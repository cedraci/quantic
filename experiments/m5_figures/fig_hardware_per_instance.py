"""Figure ``hardware_per_instance``: feasible samples per 1,000 shots, instance by instance.

Caption. Each row is one of the 30 binary-encoded T0 instances of at most 24 qubits
that ran on ibm_fez (reps = 2), labelled dials-seed and sorted by the routed CZ count
of its device circuit, shallowest at the top (735 CZ gates, D2+D4-2029) and deepest at
the bottom (2,044, none-2029). The bar is the mean number of feasible samples per 1,000
shots over the instance's 3 jobs of 1,000 shots: ``hw_feasible`` of
``docs/figures/m5/data/m5-p2-uniform.csv``, a string such as "26+20+29" with one count
per job, averaged over its ``hw_jobs`` parts. The hollow marker is ``exp_per_1000``, the
number of feasible bitstrings that 1,000 uniform random draws would give: exact (the
feasible share of all 2^16 bitstrings) for the six 16-qubit instances, estimated from
100,000 draws for the others. The "CZ" figure to the right of each bar is
``routed_2q``, the two-qubit gate count of the routed device circuit, from the
``qaoa_ibm`` binary rows of ``m5-p2-qaoa_noise.csv`` joined on (dials, instance_seed).
"optimum found in k of n jobs" marks the rows where k of the n jobs returned at least
one feasible sample within 1 % of the proven optimum (``hw_success``); the other 22
rows found none. How to read it: feasible samples are a few per thousand everywhere,
about twice what uniform random bitstrings would give (mean 5.13 per 1,000 against a
uniform expectation of 2.59; the largest bar, 25.0 on D1+D2-2030, sits at its uniform
expectation of 25.3), and the eight instances with a success lie at 735 to 1,233 CZ
gates, seven of them in the ten shallowest rows. From 1,261 CZ gates down no job found
the optimum, and three of those 16 instances returned no feasible sample in any job.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Patch

from m5_figures import style

NAME = "hardware_per_instance"
TITLE = ("Each instance on the device: feasible samples per 1,000 shots,\n"
         "and whether the optimum was found")
SOURCE = ("Source: docs/figures/m5/data/m5-p2-uniform.csv (hw_feasible, hw_jobs, hw_success, "
          "exp_per_1000) joined with m5-p2-qaoa_noise.csv (routed_2q of the qaoa_ibm, binary "
          "rows).\nRegenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")
BAR_HEIGHT = 0.7  # of the row pitch: about 23 px at 200 dpi with the layout below
MARKER_SIZE = 6.5  # points: an 18 px hollow disc, inside the bar's height
LABEL_SIZE = 7  # points, the CZ count and the "optimum found" label


def _rows(data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The device instances: the uniform baseline joined with the device circuit's CZ count.

    ``hw_feasible`` lists the feasible samples of each 1,000-shot job as "7+4"; its mean
    over the jobs is the bar. Rows come back sorted by routed CZ count, shallowest first.
    """
    circuits = (data["m5-p2-qaoa_noise"]
                .filter((pl.col("solver") == "qaoa_ibm") & (pl.col("encoding") == "binary"))
                .select("dials", "instance_seed", "routed_2q"))
    parts = pl.col("hw_feasible").cast(pl.Utf8).str.split("+")
    rows = (data["m5-p2-uniform"]
            .join(circuits, on=["dials", "instance_seed"], how="inner")
            .with_columns(feasible_mean=parts.list.eval(pl.element().cast(pl.Float64)).list.mean(),
                          n_parts=parts.list.len())
            .sort("routed_2q", "n_vars", "dials", "instance_seed"))
    if rows.height != data["m5-p2-uniform"].height:
        raise ValueError("some uniform-baseline instances have no qaoa_ibm circuit")
    if not (rows["n_parts"] == rows["hw_jobs"]).all():
        raise ValueError("hw_feasible does not list one feasible count per job")
    return rows


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    rows = _rows(data)
    n = rows.height
    ys = list(range(n))
    means = rows["feasible_mean"].to_list()
    expected = rows["exp_per_1000"].to_list()
    labels = [f"{d}-{s}" for d, s in zip(rows["dials"], rows["instance_seed"], strict=True)]
    jobs = sorted(rows["hw_jobs"].unique().to_list())
    n_jobs = str(jobs[0]) if len(jobs) == 1 else f"{jobs[0]}–{jobs[-1]}"
    xmax = int(max(*means, *expected) // 5) * 5 + 5  # the next multiple of 5 above the data

    fig, ax = style.new_figure(8.0, 6.4)
    fig.subplots_adjust(left=0.125, right=0.81, top=0.845, bottom=0.065)
    ax.barh(ys, means, height=BAR_HEIGHT, color=style.DEVICE, linewidth=0, zorder=2)
    hollow = style.marker_kw(style.CLASSICAL, "o", MARKER_SIZE, filled=False)
    ax.plot(expected, ys, zorder=3, clip_on=False, **hollow)  # whole discs at x = 0
    ax.axvline(0, color=style.BASELINE, linewidth=0.6, zorder=1)  # the bars' baseline

    # Right of each bar (or of the marker when it sits beyond the bar): the routed CZ count in
    # muted text and, only where a job found the optimum, the direct label in secondary ink.
    columns = (ys, means, expected, rows["routed_2q"], rows["hw_success"], rows["hw_jobs"])
    for y, mean, exp, cz, k, nj in zip(*columns, strict=True):
        texts = [TextArea(f"CZ {int(cz)}", textprops={"size": LABEL_SIZE, "color": style.MUTED})]
        if k >= 1:
            texts.append(TextArea(f"optimum found in {k} of {nj} jobs",
                                  textprops={"size": LABEL_SIZE, "color": style.INK_2}))
        box = HPacker(children=texts, pad=0, sep=6, align="baseline")
        ax.add_artist(AnnotationBbox(box, (max(mean, exp), y), xybox=(7, 0), xycoords="data",
                                     boxcoords="offset points", box_alignment=(0, 0.5),
                                     frameon=False, pad=0, annotation_clip=False))

    ax.set_yticks(ys, labels)
    plt.setp(ax.get_yticklabels(), color=style.MUTED)
    ax.set_ylim(n - 0.5, -0.5)  # shallowest circuit at the top
    ax.set_xlim(0, xmax)
    ax.set_xticks(range(0, xmax + 1, 5))
    ax.set_xlabel("feasible samples per 1,000 shots")
    style.finish(ax, grid="x")
    ax.tick_params(axis="y", pad=7)  # clears the unclipped markers that sit at x = 0

    style.heading(ax, TITLE, f"Mean of {n_jobs} jobs × 1,000 shots on ibm_fez; "
                             "rows sorted by routed CZ count, shallowest first")
    handles = [Patch(facecolor=style.DEVICE, edgecolor="none"), Line2D([], [], **hollow)]
    style.legend(ax, ncol=2, handles=handles,
                 labels=[f"device, mean of {n_jobs} jobs", "uniform random, expected"])
    style.source(fig, SOURCE)
    return [style.save(fig, out, NAME)]
