"""The M4 crossover: the method with the lowest TTS99 on each tier x dials cell.

Caption. A tile grid of the 32 M4 cells: rows are the 16 dial combinations (D1 square-root
impact, D2 cardinality cap + minimum lot, D3 CVaR risk, D4 all-or-nothing blocks), columns
the tiers T0 (3 assets x 4 buckets) and T1 (5 assets x 8 buckets). Each tile names the
method with the lowest TTS99 at epsilon = 1% on that cell and, on a second line, that
TTS99, taking for every solver its best (smallest) TTS99 over the four budgets (0.1, 1, 10,
60 s) and, for the samplers, the three encodings (binary, domain-wall, one-hot), which is
written in parentheses; a solver with no finite TTS99 on a cell is out, and a tie goes to
the exact method. A blue wash marks a cell an exact method wins, an orange wash a cell a
QUBO sampler wins. HiGHS is ``exact_pwl_milp`` and the samplers (greedy, tabu, SA =
simulated annealing) come from ``m4-cells.csv``; CP-SAT is the sound re-run (``feb00ff``)
in ``m4-cpsat-cells.csv``, and the ``cpsat`` rows of ``m4-cells.csv`` (the original,
unsound run) are ignored.

The data reproduce the M4 result: a sampler wins 5 of the 32 cells (CP-SAT wins 20 and
HiGHS 7), all five at the 0.1 s budget and none with the cardinality (D2) or CVaR (D3)
dial on: T1 none, D1 and D4 with tabu on domain-wall (0.27, 0.19 and 0.47 s against 0.66,
0.76 and 0.51 s for the best exact method, CP-SAT) and T0 none (greedy on domain-wall,
0.11 s against 0.16 s) and D4 (greedy on binary, 0.089 s against 0.094 s). Every cell with
D2 or D3 on goes to CP-SAT or HiGHS, as do T0 D1 and both D1+D4 cells, the other three
cells without D2 and D3. How to read it: the exact methods' times are times to a proven
optimum; a sampler's TTS99 is the time to be within 1% of it with 99% confidence, and on
the 27 cells it does not win the gap is one to four orders of magnitude (T0 D3: HiGHS
0.060 s, the best sampler 114 s).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from m5_figures import style

NAME = "m4_crossover"
TTS = "tts99_s_0.01"
TIERS = [("T0", "3 assets × 4 buckets"), ("T1", "5 assets × 8 buckets")]
DIALS = ["none", "D1", "D2", "D3", "D4", "D1+D2", "D1+D3", "D1+D4", "D2+D3", "D2+D4", "D3+D4",
         "D1+D2+D3", "D1+D2+D4", "D1+D3+D4", "D2+D3+D4", "D1+D2+D3+D4"]
EXACT = ["exact_pwl_milp", "cpsat"]
SAMPLERS = ["greedy", "tabu", "simulated_annealing"]
SHORT = {"exact_pwl_milp": "HiGHS", "cpsat": "CP-SAT", "greedy": "greedy", "tabu": "tabu",
         "simulated_annealing": "SA"}
ENCODING = {"binary": "binary", "domain_wall": "domain-wall", "one_hot": "one-hot"}
WASH = {"exact": style.BLUE, "sampler": style.ORANGE}
WASH_ALPHA = 0.25
HEADER = 0.9  # the column-label band above the 16 rows, in row units

TITLE = "The M4 crossover: the method with the lowest TTS99 on each cell"
SUBTITLE = ("Each tile: the method with the lowest TTS99 at ε = 1%, best over four budgets "
            "and three encodings, and that time")
SOURCE = ("D1 square-root impact · D2 cardinality cap + minimum lot · D3 CVaR risk · "
          "D4 all-or-nothing blocks. Budgets 0.1, 1, 10, 60 s; encodings binary, domain-wall, "
          "one-hot.\n"
          "Source: docs/figures/m5/data/m4-cells.csv (HiGHS = exact_pwl_milp, greedy, tabu, SA) "
          "and m4-cpsat-cells.csv (the sound CP-SAT re-run, feb00ff).\n"
          "Regenerate: uv run --no-sync --with matplotlib experiments/m5-figures.py")


def winners(data: dict[str, pl.DataFrame]) -> dict[tuple[str, str], dict]:
    """The winner of each (tier, dials) cell by best TTS99 over budgets and encodings."""
    cols = ["tier", "dials", "solver", pl.col("encoding").cast(pl.String), "budget_s", TTS]
    cells = pl.concat([
        data["m4-cells"].filter(pl.col("solver").is_in(EXACT[:1] + SAMPLERS)).select(cols),
        data["m4-cpsat-cells"].filter(pl.col("solver") == "cpsat").select(cols),
    ])
    best = (cells.filter(pl.col(TTS).is_not_null() & pl.col(TTS).is_finite())
            .sort(TTS)
            .group_by(["tier", "dials", "solver"], maintain_order=True)
            .agg(pl.col(TTS).first().alias("tts"), pl.col("encoding").first(),
                 pl.col("budget_s").first())
            .with_columns((~pl.col("solver").is_in(EXACT)).cast(pl.Int8).alias("sampler"))
            .sort(["tts", "sampler"])  # a tie goes to the exact method
            .group_by(["tier", "dials"], maintain_order=True)
            .first())
    return {(row["tier"], row["dials"]): row for row in best.to_dicts()}


def seconds(value: float) -> str:
    """Fixed-point seconds at a precision that follows the magnitude: 0.023 s, 0.27 s, 1.4 s."""
    if value < 0.1:
        return f"{value:.3f} s"
    if value < 1.0:
        return f"{value:.2f} s"
    if value < 10.0:
        return f"{value:.1f} s"
    return f"{value:.0f} s"


def tile_text(win: dict) -> tuple[str, str]:
    name = SHORT[win["solver"]]
    if win["sampler"]:
        name += f" ({ENCODING[win['encoding']]})"
    return name, seconds(win["tts"])


def make(data: dict[str, pl.DataFrame], out: Path) -> list[Path]:
    wins = winners(data)
    n_rows, n_cols = len(DIALS), len(TIERS)
    fig, ax = style.new_figure(8.0, 6.5)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.89, bottom=0.03)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + HEADER)

    # A 2 px surface gap between tiles: every tile is inset by 1 px on each side.
    box = ax.get_position()
    inset_x = n_cols / (box.width * fig.get_figwidth() * style.DPI)
    inset_y = (n_rows + HEADER) / (box.height * fig.get_figheight() * style.DPI)

    for c, (tier, _) in enumerate(TIERS):
        for r, dials in enumerate(DIALS):
            win = wins[(tier, dials)]
            family = "sampler" if win["sampler"] else "exact"
            x, y = c, n_rows - r - 1
            ax.add_patch(plt.Rectangle((x + inset_x, y + inset_y), 1 - 2 * inset_x,
                                       1 - 2 * inset_y, facecolor=WASH[family],
                                       alpha=WASH_ALPHA, edgecolor="none", linewidth=0))
            name, tts = tile_text(win)
            ax.text(x + 0.5, y + 0.71, name, ha="center", va="center", fontsize=8.5,
                    color=style.INK)
            ax.text(x + 0.5, y + 0.29, tts, ha="center", va="center", fontsize=7.5,
                    color=style.INK)

    for c, (tier, size) in enumerate(TIERS):
        ax.text(c + 0.5, n_rows + HEADER / 2, f"{tier}  ·  {size}", ha="center", va="center",
                fontsize=8.5, color=style.MUTED)

    ax.set_xticks([])
    ax.set_yticks([n_rows - r - 0.5 for r in range(n_rows)], DIALS)
    style.finish(ax, grid=None, baseline=False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=WASH[family], alpha=WASH_ALPHA,
                             edgecolor="none", label=label)
               for family, label in (("exact", "exact method wins"),
                                     ("sampler", "a QUBO sampler wins"))]
    style.legend(ax, ncol=2, handles=handles)

    style.heading(ax, TITLE, SUBTITLE)
    style.source(fig, SOURCE)
    return [style.save(fig, out, NAME)]
