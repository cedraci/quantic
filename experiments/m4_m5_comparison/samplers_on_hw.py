"""The samplers at 1,000 reads restricted to the instances QAOA ran on the device (like for like).

Scored by quantic.bench.metrics.scored on the main M5 store, as the report scores them.
Writes samplers_on_hw.csv: per (dials, instance_seed, solver, encoding), success at 1% and
feasibility.
"""

import sys
from pathlib import Path

import polars as pl

from quantic.bench.metrics import scored
from quantic.bench.store import Store

sys.stdout.reconfigure(encoding="utf-8")
D = str(Path(__file__).resolve().parents[2] / "docs" / "figures" / "m5" / "data") + "/"

STORE, OUT = sys.argv[1], sys.argv[2]
with Store(STORE, read_only=True) as s:
    runs = s.runs()
rows = scored(runs).filter(
    pl.col("solver").is_in(["greedy", "simulated_annealing", "tabu"])
    & (pl.col("max_samples") == 1000)
)
per = (
    rows.group_by("dials", "instance_seed", "solver", "encoding")
    .agg(
        pl.len().alias("n_runs"),
        pl.col("ok_0.01").mean().alias("p_success_0.01"),
        (pl.col("n_feasible_samples").sum() / pl.col("n_samples").sum()).alias("feasibility"),
        pl.col("total_s").mean().alias("t_run_s"),
    )
    .sort("dials", "instance_seed", "solver", "encoding")
)
per.write_csv(OUT)
hw = (
    pl.read_csv(D + "m5-p2-qaoa_noise.csv")
    .filter(pl.col("solver") == "qaoa_ibm")
    .select("dials", "instance_seed")
)
sub = per.join(hw, on=["dials", "instance_seed"]).filter(pl.col("encoding") == "binary")
cells = (
    sub.group_by("dials", "solver")
    .agg(
        pl.len().alias("instances"),
        pl.col("p_success_0.01").mean().alias("success"),
        pl.col("feasibility").mean().alias("feasibility"),
    )
    .sort("dials", "solver")
)
pl.Config.set_tbl_rows(40)
print(cells)
full = (
    per.filter(pl.col("encoding") == "binary")
    .group_by("solver", maintain_order=True)
    .agg(
        pl.col("p_success_0.01").mean().alias("success_all40"),
        pl.col("feasibility").mean().alias("feas_all40"),
    )
)
print(full)
print(
    sub.group_by("solver", maintain_order=True).agg(
        pl.col("p_success_0.01").mean().alias("success_hw30"),
        pl.col("p_success_0.01").min().alias("min_inst"),
        pl.col("feasibility").mean().alias("feas_hw30"),
    )
)
