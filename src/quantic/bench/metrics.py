"""From rows to the numbers of parent spec section 9.1.

A run succeeds at tolerance ``eps`` when its schedule is feasible without
repair and its total is within ``eps`` of the instance's optimum, measured
against the proven lower bound. ``eps = 0`` means the optimum to the
round-off. A reference bracket wider than the solvers' own feasibility tolerance
is open: it has no gap and reports its width instead. Success never depends on
the bracket being closed: it is read against ``upper``.

Pooling is over solver seeds and instance seeds together, per cell of tier x
dial combination x solver x encoding x budget.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import polars as pl

from quantic.bench.keys import REF_BRACKET, REF_QUBO_MINIMUM, RUN
from quantic.bench.store import OK, PENDING
from quantic.solvers.base import _BRACKET_RTOL

EPSILONS = (0.0, 0.001, 0.01, 0.05)
# Two evaluator calls at one schedule agree to round-off; this is not a tolerance.
_SAME = 1e-9
_BOOTSTRAP = 1000
CELL = ["tier", "dials", "solver", "encoding", "budget_s", "max_samples"]
INSTANCE = ["tier", "dials", "instance_seed"]


def tts99(t_run: float, p: float) -> float | None:
    """Time to reach the target with 99% confidence; ``None`` when censored."""
    if p >= 1.0:
        return t_run
    if p <= 0.0:
        return None
    return t_run * math.log(0.01) / math.log(1.0 - p)


def _rounded(relative: pl.Expr) -> pl.Expr:
    """Round-off either side of zero is zero; anything larger is kept, sign and all."""
    return pl.when(relative.abs() <= _SAME).then(0.0).otherwise(relative)


def references(runs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per instance: the proven bracket, and per encoding the QUBO's minimum."""
    bracket = runs.filter(
        (pl.col("solver") == REF_BRACKET) & (pl.col("store_status") == OK)
    ).select(
        "instance_hash",
        pl.col("lower").alias("ref_lower"),
        pl.col("upper").alias("ref_upper"),
    )
    qubo = runs.filter(
        (pl.col("solver") == REF_QUBO_MINIMUM) & (pl.col("store_status") == OK)
    ).select(
        "instance_hash", "encoding", "zeta_bits",
        pl.col("qubo_energy").alias("ref_qubo_energy"),
        pl.col("qubo_total").alias("ref_qubo_total"),
    )
    return bracket, qubo


def _has_reference(runs: pl.DataFrame) -> pl.DataFrame:
    """The coordinates of every instance with an ``ok`` reference bracket.

    By coordinates, not by hash: a killed run's row has no instance hash, and
    its instance still has, or lacks, a reference.
    """
    return (
        runs.filter((pl.col("solver") == REF_BRACKET) & (pl.col("store_status") == OK))
        .select(*INSTANCE, pl.lit(True).alias("has_ref"))
        .unique()
    )


def scored(runs: pl.DataFrame) -> pl.DataFrame:
    """Every solver run with its gap, its success at each eps, and its regret."""
    bracket, qubo = references(runs)
    out = (
        runs.filter(pl.col("kind") == RUN)
        .join(bracket, on="instance_hash", how="left")
        .join(qubo, on=["instance_hash", "encoding", "zeta_bits"], how="left")
        .join(_has_reference(runs), on=INSTANCE, how="left")
        .with_columns(pl.col("has_ref").fill_null(False))
    )
    # No reference row is no success and no gap -- never a null that turns a
    # success rate into NaN. "Closed" decides only gap versus width.
    closed = (
        (pl.col("ref_upper") - pl.col("ref_lower"))
        <= _BRACKET_RTOL * pl.max_horizontal(pl.lit(1.0), pl.col("ref_upper").abs())
    ).fill_null(False)
    feasible = (pl.col("store_status") == OK) & pl.col("feasible").fill_null(False)
    out = out.with_columns(
        closed.alias("ref_closed"),
        (pl.col("ref_upper") - pl.col("ref_lower")).alias("ref_width"),
        pl.when(feasible & closed)
        .then(_rounded((pl.col("total") - pl.col("ref_lower")) / pl.col("ref_lower").abs()))
        .alias("gap"),
        _rounded(
            (pl.col("best_energy") - pl.col("ref_qubo_energy"))
            / pl.max_horizontal(pl.lit(1.0), pl.col("ref_qubo_energy").abs())
        ).alias("energy_gap"),
        _rounded(
            (pl.col("ref_qubo_total") - pl.col("ref_lower")) / pl.col("ref_lower").abs()
        ).alias("encoding_regret"),
    )
    for eps in EPSILONS:
        target = pl.col("ref_upper") + pl.max_horizontal(
            eps * pl.col("ref_lower").abs(), _SAME * pl.col("ref_upper").abs()
        )
        out = out.with_columns(
            (feasible & (pl.col("total") <= target)).fill_null(False).alias(f"ok_{eps}")
        )
    return out


def _bootstrap_tts(ok: np.ndarray, t_run: float, rng: np.random.Generator):
    if ok.size == 0:
        return None, None
    draws = rng.choice(ok, size=(_BOOTSTRAP, ok.size), replace=True).mean(axis=1)
    values = np.array([np.inf if (v := tts99(t_run, p)) is None else v for p in draws])
    # inverted_cdf returns observed values, so a censored draw stays infinite
    # instead of turning the interpolation into inf - inf.
    lo, hi = np.quantile(values, [0.025, 0.975], method="inverted_cdf")
    return (None if np.isinf(lo) else float(lo)), (None if np.isinf(hi) else float(hi))


def _cell_rng(seed: int, keys: tuple) -> np.random.Generator:
    """One generator per cell, so an interval does not depend on the other cells."""
    digest = hashlib.sha256(repr(keys).encode("utf-8")).digest()
    return np.random.default_rng([seed, int.from_bytes(digest[:8], "big")])


def _success(group: pl.DataFrame, eps: float, t_run: float, rng) -> dict:
    ok = group[f"ok_{eps}"].to_numpy().astype(float)
    p = float(ok.mean()) if ok.size else 0.0
    lo, hi = _bootstrap_tts(ok, t_run, rng)
    return {
        "p_success": p,
        "tts99_s": tts99(t_run, p),
        "tts99_lo_s": lo,
        "tts99_hi_s": hi,
        "censored_above_s": None if p > 0 else (group["budget_s"][0] or 0) * group.height,
    }


def cells(runs: pl.DataFrame, *, eps: float = 0.01, seed: int = 0) -> pl.DataFrame:
    """One row per cell: gap, feasibility, success, TTS with a bootstrap interval.

    The unsuffixed success columns are at ``eps``; every eps in ``EPSILONS`` also
    has its own ``<name>_<eps>`` columns.
    """
    runs = runs.filter(pl.col("store_status") != PENDING)
    rows = scored(runs)
    out = []
    for keys, group in rows.group_by(CELL, maintain_order=True):
        rng = _cell_rng(seed, keys)
        t_run = float(group["total_s"].drop_nulls().mean() or 0.0)
        by_eps = {e: _success(group, e, t_run, rng) for e in sorted({*EPSILONS, eps})}
        sampled = group["n_samples"].drop_nulls()
        if sampled.len():
            feasibility = float(group["n_feasible_samples"].sum() / sampled.sum())
        else:
            feasibility = float(group["feasible"].fill_null(False).mean())
        gaps = group["gap"].drop_nulls()
        open_refs = group.filter(~pl.col("ref_closed") & pl.col("ref_width").is_not_null())
        out.append({
            **dict(zip(CELL, keys, strict=True)),
            "n_runs": group.height,
            "n_killed": int((group["store_status"] == "killed").sum()),
            "n_errors": int((group["store_status"] == "error").sum()),
            # Instances with no ok reference: their runs cannot succeed, which
            # is the reference's failure, not the solver's.
            "n_missing_refs": group.filter(~pl.col("has_ref"))["instance_seed"].n_unique(),
            "gap_median": float(gaps.median()) if gaps.len() else None,
            "gap_q1": float(gaps.quantile(0.25)) if gaps.len() else None,
            "gap_q3": float(gaps.quantile(0.75)) if gaps.len() else None,
            "ref_width_max": open_refs["ref_width"].max() if open_refs.height else None,
            "n_open_refs": open_refs["instance_hash"].n_unique(),
            "feasibility": feasibility,
            "t_run_s": t_run,
            **by_eps[eps],
            **{f"{name}_{e}": v for e, d in by_eps.items() for name, v in d.items()},
            "bps_best": group.filter(pl.col("feasible").fill_null(False))["total_bps"].min(),
            "energy_gap_median": group["energy_gap"].drop_nulls().median(),
            "encoding_regret": group["encoding_regret"].drop_nulls().median(),
            "dynamic_range": group["dynamic_range"].drop_nulls().median(),
        })
    return pl.DataFrame(out, infer_schema_length=None)
