"""A parameter-free model of the device's feasibility: the noiseless circuit's output diluted by
its error-free fraction, the rest uniform random bits.

    device_feasible ~= f * noiseless_feasible + (1 - f) * uniform_feasible
    f = (1 - p_cz)^CZ * (1 - p_ro)^qubits      (calibration medians; no fitted parameter)

Prints, per rung, the pooled prediction against the device and the per-instance rank correlation.
"""

import sys
from pathlib import Path

import polars as pl
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")

D = str(Path(__file__).resolve().parents[2] / "docs" / "figures" / "m5" / "data") + "/"
K = ["dials", "instance_seed"]
P_CZ = 0.0025  # median CZ error, ibm_fez, 2026-10-27 and 2026-10-28
for rung in ("p2", "p1"):
    n = pl.read_csv(D + f"m5-{rung}-qaoa_noise.csv").filter(pl.col("encoding") == "binary")
    hw = n.filter(pl.col("solver") == "qaoa_ibm").select(
        K + ["routed_2q", pl.col("feasibility").alias("device"), pl.col("n_runs").alias("jobs")]
    )
    ideal = n.filter(pl.col("solver") == "qaoa_ideal").select(
        K + [pl.col("feasibility").alias("noiseless")]
    )
    uni = pl.read_csv(D + f"m5-{rung}-uniform.csv").select(K + ["uniform_frac", "n_vars"])
    t = hw.join(ideal, on=K).join(uni, on=K)
    for p_ro in (0.0, 0.010):
        f = (1 - P_CZ) ** pl.col("routed_2q") * (1 - p_ro) ** pl.col("n_vars")
        m = t.with_columns(
            (f * pl.col("noiseless") + (1 - f) * pl.col("uniform_frac")).alias("model")
        )
        pooled_model = float((m["model"] * m["jobs"]).sum() / m["jobs"].sum())
        pooled_dev = float((m["device"] * m["jobs"]).sum() / m["jobs"].sum())
        rho = spearmanr(m["model"], m["device"])
        print(
            f"{rung} readout {p_ro:.3f}: model {100 * pooled_model:.3f}% device "
            f"{100 * pooled_dev:.3f}% "
            f"(device/model {pooled_dev / pooled_model:.2f}); Spearman {rho.statistic:.2f} "
            f"(p = {rho.pvalue:.1g}); {m.height} instances"
        )
    kept = t.with_columns(
        (pl.col("device") / pl.col("noiseless")).alias("kept"),
        ((1 - P_CZ) ** pl.col("routed_2q")).alias("ef"),
    )
    print(
        f"   device/noiseless feasibility, median {kept['kept'].median():.3f}; "
        f"error-free fraction at the median CZ ({kept['routed_2q'].median():.0f}) "
        f"{(1 - P_CZ) ** kept['routed_2q'].median():.3f}"
    )
