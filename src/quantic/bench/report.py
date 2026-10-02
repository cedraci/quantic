"""The crossover report: CSV for machines, Markdown for people.

Three products, each written as both:

* ``crossover_<budget>s``: tier x dial rows, one column per solver -- each
  sampler at its best encoding, named in the cell -- showing median gap,
  feasibility rate and TTS at 99% for eps = 1%.
* ``dial_effects``: each dial's main effect and every pairwise interaction on
  each solver's median gap, at the largest budget. Sixteen combinations is the
  full 2**4 factorial, so these are exact contrasts, not a screening estimate.
* ``bracket_check``: every instance where a CP-SAT bracket and the reference
  bracket do not overlap. Expected empty.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import polars as pl

from quantic.bench.keys import REF_BRACKET, RUN
from quantic.bench.metrics import cells
from quantic.bench.store import OK, Store

DIALS = ("D1", "D2", "D3", "D4")
_OVERLAP_RTOL = 1e-6
CENSORED = "censored"


def _has(label: str, dial: str) -> bool:
    return dial in label.split("+")


def best_encoding(table: pl.DataFrame) -> pl.DataFrame:
    """Per (tier, dials, solver, budget), the encoding that reaches the target soonest.

    Ranked by TTS at eps = 1%, censored cells after every uncensored one (by their
    censored bound), then by median gap.
    """
    return (
        table.with_columns(
            pl.col("tts99_s").is_null().alias("_censored"),
            pl.coalesce("tts99_s", "censored_above_s").fill_null(float("inf")).alias("_tts"),
            pl.col("gap_median").fill_null(float("inf")).alias("_gap"),
        )
        .sort(["_censored", "_tts", "_gap", "encoding"], nulls_last=True)
        .group_by(["tier", "dials", "solver", "budget_s"], maintain_order=True)
        .first()
        .drop("_censored", "_tts", "_gap")
    )


def _cell_text(row: dict) -> str:
    if row["gap_median"] is not None:
        gap = f"{100 * row['gap_median']:.2f}%"
    elif row["ref_width_max"] is not None:
        gap = f"open, width {row['ref_width_max']:.3g}"
    else:
        gap = "-"
    if row["tts99_s"] is not None:
        tts = f"{row['tts99_s']:.3g}s"
    else:
        tts = f"{CENSORED}, > {row['censored_above_s']:.3g} s"
    enc = f" ({row['encoding']})" if row["encoding"] else ""
    return f"{gap} / {100 * row['feasibility']:.0f}% / {tts}{enc}"


def crossover(best: pl.DataFrame, budget: float) -> pl.DataFrame:
    at = best.filter(pl.col("budget_s") == budget)
    texts = at.with_columns(
        pl.Series("cell", [_cell_text(r) for r in at.iter_rows(named=True)])
    )
    return (
        texts.pivot(on="solver", index=["tier", "dials"], values="cell")
        .sort(["tier", "dials"])
    )


def dial_effects(best: pl.DataFrame) -> pl.DataFrame:
    top = best.filter(pl.col("budget_s") == best["budget_s"].max())
    out = []
    for (tier, solver), group in top.group_by(["tier", "solver"], maintain_order=True):
        gaps = {r["dials"]: r["gap_median"] for r in group.iter_rows(named=True)}
        if len(gaps) < 16 or any(g is None for g in gaps.values()):
            continue

        def mean(test, gaps=gaps) -> float:
            values = [g for label, g in gaps.items() if test(label)]
            return sum(values) / len(values)

        for d in DIALS:
            out.append({"tier": tier, "solver": solver, "effect": d,
                        "value": mean(lambda x, d=d: _has(x, d))
                        - mean(lambda x, d=d: not _has(x, d))})
        for a, b in itertools.combinations(DIALS, 2):
            same = mean(lambda x, a=a, b=b: _has(x, a) == _has(x, b))
            differ = mean(lambda x, a=a, b=b: _has(x, a) != _has(x, b))
            out.append({"tier": tier, "solver": solver, "effect": f"{a}x{b}",
                        "value": same - differ})
    return pl.DataFrame(out, schema={"tier": pl.String, "solver": pl.String,
                                     "effect": pl.String, "value": pl.Float64})


def bracket_check(runs: pl.DataFrame) -> pl.DataFrame:
    ref = runs.filter((pl.col("solver") == REF_BRACKET) & (pl.col("store_status") == OK))
    cp = runs.filter(
        (pl.col("kind") == RUN) & (pl.col("solver") == "cpsat")
        & pl.col("lower").is_not_null()
    )
    joined = cp.join(
        ref.select("instance_hash", pl.col("lower").alias("ref_lower"),
                   pl.col("upper").alias("ref_upper")),
        on="instance_hash",
    )
    slack = _OVERLAP_RTOL * pl.max_horizontal(pl.lit(1.0), pl.col("ref_upper").abs())
    apart = (pl.col("lower") > pl.col("ref_upper") + slack) | (
        pl.col("ref_lower") > pl.col("upper") + slack
    )
    return joined.filter(apart).select(
        "tier", "dials", "instance_seed", "solver_seed", "budget_s",
        "lower", "upper", "ref_lower", "ref_upper",
    )


def _markdown(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "_none_\n"
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([head, rule, *body]) + "\n"


def _write(frame: pl.DataFrame, out: Path, stem: str) -> list[Path]:
    csv, md = out / f"{stem}.csv", out / f"{stem}.md"
    frame.write_csv(csv)
    md.write_text(_markdown(frame), encoding="utf-8")
    return [csv, md]


def report(store: Path, out: Path, *, commit: str | None = None) -> list[Path]:
    with Store(store, read_only=True) as s:
        runs = s.runs()
    if commit is not None:
        runs = runs.filter(pl.col("commit") == commit)
    commits = runs["commit"].unique().to_list()
    if len(commits) > 1:
        raise ValueError(
            f"the store holds rows from {len(commits)} commits; pass --commit to "
            "choose one, since rows from different code do not belong in one table"
        )
    out.mkdir(parents=True, exist_ok=True)
    table = cells(runs)
    best = best_encoding(table)
    written = _write(table, out, "cells")
    for budget in sorted(best["budget_s"].unique().to_list()):
        written += _write(crossover(best, budget), out, f"crossover_{budget:g}s")
    written += _write(dial_effects(best), out, "dial_effects")
    written += _write(bracket_check(runs), out, "bracket_check")
    return written
