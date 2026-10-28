"""Every IBM hardware job of the M5 studies, one row each -- the M4/M5 comparison's per-job table.

Joins the quota ledger with the ``qaoa_ibm`` rows of the study stores by job id: the
lane a job belongs to (the label given to its store), its pass within that lane, its
instance, IBM's QPU charge and the queue, the routed circuit, and what it returned,
scored by ``bench.metrics.scored`` exactly as the report scores it. A job of the ledger
that no store names is kept, with only the ledger's columns.

    .venv/Scripts/python.exe experiments/m5-hardware-jobs.py <out.csv> --ledger <path> \\
        reps2=<store> reps1=<store> smoke=<store>

A repeat of a stored hardware task is keyed ``<key>#r<n>`` (``bench.hardware``), so a
row's pass is 1 + n, and 1 for a key without the suffix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from quantic.bench.keys import REFERENCE
from quantic.bench.metrics import scored
from quantic.bench.store import Store
from quantic.solvers.quantum.quota import Ledger


def one_reference_each(runs: pl.DataFrame) -> pl.DataFrame:
    """A store run at several commits holds each reference once per commit: keep one.

    ``scored`` joins references by instance hash, so repeats would repeat every run.
    The repeats must agree, or there is no single optimum to score against.
    """
    ident = ["instance_hash", "solver", "encoding", "zeta_bits"]
    refs = runs.filter(pl.col("kind") == REFERENCE)
    spread = refs.group_by(ident).agg(
        (pl.col("lower").max() - pl.col("lower").min()).alias("d_lower"),
        (pl.col("upper").max() - pl.col("upper").min()).alias("d_upper"))
    if spread.filter((pl.col("d_lower") > 0) | (pl.col("d_upper") > 0)).height:
        raise ValueError("a reference differs between the commits of one store")
    return pl.concat([runs.filter(pl.col("kind") != REFERENCE),
                      refs.unique(subset=ident, keep="first", maintain_order=True)])


def store_jobs(label: str, path: Path) -> pl.DataFrame:
    with Store(path, read_only=True) as s:
        runs = s.runs()
    rows = scored(one_reference_each(runs)).filter(pl.col("solver") == "qaoa_ibm")
    return rows.select(
        "job_id",
        pl.lit(label).alias("lane"),
        "reps",
        (pl.col("key").str.extract(r"#r(\d+)$").cast(pl.Int64).fill_null(0) + 1).alias("pass"),
        "tier", "dials", "instance_seed", "encoding",
        "qpu_s", "queue_s", "routed_2q", "routed_depth",
        "n_samples", "n_feasible_samples",
        pl.col("total").alias("best_total"),
        pl.col("ref_lower").alias("lower"),
        pl.col("ok_0.01").alias("success_0.01"),
        "store_status",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("stores", nargs="+", help="label=path, one per store")
    args = ap.parse_args()

    stored = pl.concat(
        [store_jobs(*spec.split("=", 1)) for spec in args.stores], how="diagonal_relaxed")
    ledger = pl.DataFrame([vars(e) for e in Ledger(args.ledger).entries()])
    jobs = (ledger.select("job_id", "submitted_at", "device", "state", "usage_s",
                          "max_execution_time_s", "task_identity")
            .join(stored, on="job_id", how="left")
            .sort("submitted_at")
            .with_row_index("job", offset=1))
    jobs.write_csv(args.out)

    print(f"{jobs.height} ledger jobs, {jobs['usage_s'].sum():.0f} s; "
          f"{jobs['lane'].null_count()} without a store row")
    summary = (jobs.group_by("lane", "reps", "pass", maintain_order=True)
               .agg(pl.len().alias("jobs"), pl.col("usage_s").sum().alias("usage_s"),
                    pl.col("success_0.01").sum().alias("within_1pct"),
                    pl.col("n_feasible_samples").sum().alias("feasible"),
                    pl.col("n_samples").sum().alias("shots")))
    for row in summary.iter_rows(named=True):
        print(row)


if __name__ == "__main__":
    main()
