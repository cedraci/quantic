"""The numbers that qualify the comparison's findings: confounds, significance and clocks.

Prints, in order: HiGHS per instance in both M5 stores and each training against the HiGHS run
on its own instance; the device's successes split by circuit depth, against chance and the
noiseless circuit; rank correlations of the dynamic range, the CZ count and the qubit count with
success, with the qubit count held fixed; the significance of the reps = 1 rung's changes and of
the third pass's drift; the noise model's bias against the device; how much of the error-fraction
model's ranking the uniform baseline already gives; the device's QPU TTS99 against every
classical clock; and the hardware lane's per-job limits, empty jobs and wall time per job.

Usage: qualifications.py <reps-2 store> <reps-1 store>  (DuckDB, opened read-only)
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy.stats import binomtest, chi2_contingency, fisher_exact, rankdata, spearmanr
from scipy.stats import t as student_t

sys.stdout.reconfigure(encoding="utf-8")
D = str(Path(__file__).resolve().parents[2] / "docs" / "figures" / "m5" / "data") + "/"
K = ["dials", "instance_seed"]
P_CZ, P_RO = 0.0025, 0.01  # calibration medians, ibm_fez
STORES = {"p2": sys.argv[1], "p1": sys.argv[2]}
CELLS = ["none", "D1", "D4", "D1+D4", "D2", "D1+D2", "D2+D4", "D1+D2+D4"]


def binary(frame: pl.DataFrame, solver: str) -> pl.DataFrame:
    return frame.filter((pl.col("solver") == solver) & (pl.col("encoding") == "binary"))


def partial_spearman(x, y, z) -> tuple[float, float]:
    """Rank correlation of x and y with the rank of z regressed out of both."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    a = np.column_stack([np.ones_like(rz), rz])
    ex = rx - a @ np.linalg.lstsq(a, rx, rcond=None)[0]
    ey = ry - a @ np.linalg.lstsq(a, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    df = len(x) - 3
    return r, float(2 * student_t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df))


print("== HiGHS per instance (exact_pwl_milp, 1 s budget)")
highs = {}
for rung, path in STORES.items():
    con = duckdb.connect(path, read_only=True)
    h = con.execute("select dials, instance_seed, total_s, status from runs "
                    "where solver = 'exact_pwl_milp'").pl()
    con.close()
    highs[rung] = h
    cell = h.group_by("dials").agg(pl.col("total_s").mean())["total_s"]
    print(f"{rung}: {h.height} runs, {h['total_s'].min():.3f} to {h['total_s'].max():.3f} s "
          f"(median {h['total_s'].median():.3f}; cell means {cell.min():.2f} to {cell.max():.2f}); "
          f"over 0.4 s: {h.filter(pl.col('total_s') > 0.4).height}; not optimal: "
          f"{h.filter(pl.col('status') != 'optimal').select(K + ['status']).rows()}")
    train = pl.read_csv(D + f"m5-{rung}-qaoa_training.csv").join(h, on=K)
    ratio = train["train_s"] / train["total_s"]
    print(f"   training / HiGHS on the same instance: {ratio.min():.1f} to {ratio.max():.2g} "
          f"over {train.height} QUBOs; trainings faster than HiGHS: {(ratio < 1).sum()}")

print("== The device's successes by circuit depth (reps = 2)")
noise2 = pl.read_csv(D + "m5-p2-qaoa_noise.csv")
uni2 = pl.read_csv(D + "m5-p2-uniform.csv")
dev = binary(noise2, "qaoa_ibm").select(K + ["routed_2q", pl.col("n_runs").alias("jobs")])
ideal = binary(noise2, "qaoa_ideal").select(K + [pl.col("p_success_0.01").alias("ideal")])
z = dev.join(uni2.select(K + ["n_vars", "hw_success", "p_success_uniform"]), on=K).join(ideal, on=K)
for name, cond in (("at most 1,233 CZ", pl.col("routed_2q") <= 1233),
                   ("1,261 CZ or more", pl.col("routed_2q") >= 1261)):
    g = z.filter(cond)
    print(f"{name}: {g.height} instances, {g['jobs'].sum()} jobs, {g['n_vars'].min()} to "
          f"{g['n_vars'].max()} qubits; successes {g['hw_success'].sum()}, chance "
          f"{(g['p_success_uniform'] * g['jobs']).sum():.2f}, noiseless "
          f"{(g['ideal'] * g['jobs']).sum():.1f}")

print("== Rank correlations, and with the qubit count held fixed")
for rung in ("p2", "p1"):
    n = pl.read_csv(D + f"m5-{rung}-qaoa_noise.csv")
    tr = pl.read_csv(D + f"m5-{rung}-qaoa_training.csv").filter(pl.col("encoding") == "binary")
    x = tr.select(K + ["dynamic_range", "n_vars"]).join(
        binary(n, "qaoa_ideal").select(K + [pl.col("p_success_0.01").alias("ideal")]), on=K)
    r, p = partial_spearman(x["dynamic_range"], x["ideal"], x["n_vars"])
    print(f"{rung}: range vs noiseless success {spearmanr(x['dynamic_range'], x['ideal'])[0]:+.2f}"
          f", given qubits {r:+.2f} (p = {p:.3f})")
    hw = binary(n, "qaoa_ibm").select(K + ["routed_2q", pl.col("p_success_0.01").alias("dev")])
    y = x.join(hw, on=K)
    for a, b in (("dynamic_range", "routed_2q"), ("routed_2q", "dev"), ("n_vars", "dev"),
                 ("dynamic_range", "dev")):
        s = spearmanr(y[a], y[b])
        print(f"   {a} vs {b}: {s[0]:+.2f} (p = {s[1]:.2g})")
    r, p = partial_spearman(y["routed_2q"], y["dev"], y["n_vars"])
    print(f"   routed_2q vs device success given qubits: {r:+.2f} (p = {p:.2f})")
widest = (pl.read_csv(D + "m5-p2-qaoa_training.csv").filter(pl.col("encoding") == "binary")
          .sort("dynamic_range", descending=True).head(7))
print("widest ranges:", [(r["dials"], r["instance_seed"], r["n_vars"], f"{r['dynamic_range']:.3g}")
                         for r in widest.iter_rows(named=True)])

print("== The reps = 1 rung and the third pass: shot noise or not")
jobs = pl.read_csv(D + "m5-hardware-jobs.csv")
two = jobs.filter(pl.col("lane") == "reps2")
one = jobs.filter((pl.col("lane") == "reps1")
                  & ~((pl.col("dials") == "D2+D4") & (pl.col("instance_seed") == 2030)))
s2, s1 = int(two["success_0.01"].sum()), int(one["success_0.01"].sum())
print(f"success, shared instances: {s2} of {two.height} against {s1} of {one.height}; Fisher "
      f"p = {fisher_exact([[s2, two.height - s2], [s1, one.height - s1]])[1]:.2f}")
f2, n2 = int(two["n_feasible_samples"].sum()), int(two["n_samples"].sum())
f1, n1 = int(one["n_feasible_samples"].sum()), int(one["n_samples"].sum())
print(f"feasible samples: {f2} of {n2} against {f1} of {n1}; chi-squared p = "
      f"{chi2_contingency([[f2, n2 - f2], [f1, n1 - f1]])[1]:.1g}")
per = (two.group_by(K).agg((pl.col("n_feasible_samples").sum() / pl.col("n_samples").sum())
                           .alias("r2"))
       .join(one.group_by(K).agg((pl.col("n_feasible_samples").sum() / pl.col("n_samples").sum())
                                 .alias("r1")), on=K))
more, fewer = (per.filter(pl.col("r1") > pl.col("r2")).height,
               per.filter(pl.col("r1") < pl.col("r2")).height)
print(f"instances with more feasible shots at reps = 1: {more}, fewer: {fewer}")
by_pass = two.group_by("pass").agg(pl.col("n_feasible_samples").sum(), pl.col("n_samples").sum()
                                   ).sort("pass")
k = by_pass["n_feasible_samples"].to_list()
print(f"feasible samples per pass: {k}; third pass against a shared rate: binomial p = "
      f"{binomtest(k[2], sum(k), 1 / 3).pvalue:.3f}, z = "
      f"{(k[2] - sum(k) / 3) / np.sqrt(sum(k) * 2 / 9):.1f}; chi-squared over the passes p = "
      f"{chi2_contingency([[f, 30000 - f] for f in k])[1]:.3f}")

print("== The noise model against the device, instances simulated in full")
for rung in ("p2", "p1"):
    n = pl.read_csv(D + f"m5-{rung}-qaoa_noise.csv")
    # An instance whose runs were all killed has no feasibility (NaN): it was not simulated in full.
    noisy = binary(n, "qaoa_noisy").filter(pl.col("feasibility").is_not_nan()).select(
        K + [pl.col("feasibility").alias("nf"), pl.col("p_success_0.01").alias("ns")])
    q = noisy.join(binary(n, "qaoa_ibm").select(
        K + [pl.col("feasibility").alias("df"), pl.col("p_success_0.01").alias("ds")]), on=K)
    pos = q.filter(pl.col("df") > 0)
    per_instance = pos["nf"] / pos["df"]
    print(f"{rung}: {q.height} instances; feasibility {100 * q['nf'].mean():.2f}% against "
          f"{100 * q['df'].mean():.2f}% ({q['nf'].mean() / q['df'].mean():.1f} times; per instance "
          f"median {per_instance.median():.1f}, max {per_instance.max():.1f}); "
          f"higher on {q.filter(pl.col('nf') > pl.col('df')).height}; device empty on "
          f"{q.filter(pl.col('df') == 0).height}; success {q['ns'].mean():.2f} against "
          f"{q['ds'].mean():.2f}")

print("== What ranks the device's feasibility")
for rung in ("p2", "p1"):
    n = pl.read_csv(D + f"m5-{rung}-qaoa_noise.csv")
    u = pl.read_csv(D + f"m5-{rung}-uniform.csv").select(K + ["uniform_frac", "n_vars"])
    t = (binary(n, "qaoa_ibm").select(K + ["routed_2q", pl.col("feasibility").alias("device"),
                                           pl.col("n_runs").alias("jobs")])
         .join(binary(n, "qaoa_ideal").select(K + [pl.col("feasibility").alias("noiseless")]),
               on=K).join(u, on=K))
    f = (1 - P_CZ) ** t["routed_2q"] * (1 - P_RO) ** t["n_vars"]
    model = f * t["noiseless"] + (1 - f) * t["uniform_frac"]
    pooled = float((model * t["jobs"]).sum() / (t["device"] * t["jobs"]).sum())
    rho = spearmanr(model, t["device"])[0]
    print(f"{rung}: Spearman with the device: model {rho:.2f}, uniform "
          f"fraction alone {spearmanr(t['uniform_frac'], t['device'])[0]:.2f}, error-free fraction "
          f"alone {spearmanr(f, t['device'])[0]:.2f}; pooled model / device {pooled:.2f}")

print("== The device's QPU TTS99 against every classical clock (device / classical)")
m4 = pl.read_csv(D + "m4-cells.csv", infer_schema_length=100000).filter(pl.col("tier") == "T0")
cps = pl.read_csv(D + "m4-cpsat-cells.csv", infer_schema_length=100000).filter(
    pl.col("tier") == "T0")
c2 = pl.read_csv(D + "m5-p2-cells.csv", infer_schema_length=100000)
c1 = pl.read_csv(D + "m5-p1-cells.csv", infer_schema_length=100000)


def best(frame: pl.DataFrame, solvers: list[str]) -> dict[str, float]:
    rows = frame.filter(pl.col("solver").is_in(solvers)).group_by("dials").agg(
        pl.col("tts99_s_0.01").min())
    return dict(rows.iter_rows())


clocks = {"CP-SAT (M4, sound)": best(cps, ["cpsat"]), "HiGHS (M4)": best(m4, ["exact_pwl_milp"]),
          "best M4 sampler": best(m4, ["greedy", "simulated_annealing", "tabu"]),
          "tabu (M4)": best(m4, ["tabu"])}
for solver in ("greedy", "simulated_annealing", "tabu"):
    clocks[f"{solver}, 1,000 reads"] = best(c2.filter(pl.col("max_samples") == 1000), [solver])
for rung, cells in (("reps 2", c2), ("reps 1", c1)):
    device = {d: v for d, v in best(cells, ["qaoa_ibm"]).items() if v is not None}
    for name, clock in clocks.items():
        f = [device[d] / clock[d] for d in CELLS if d in device]
        print(f"{rung}, {name}: {min(f):.2f} to {max(f):.0f} times"
              + (f" (device faster on {sum(v < 1 for v in f)} cells)" if min(f) < 1 else ""))

print("== The hardware lane")
for lane in ("reps2", "reps1"):
    jl = jobs.filter(pl.col("lane") == lane).sort("submitted_at")
    gaps = pl.concat([jl.filter(pl.col("pass") == p)["submitted_at"].diff().drop_nulls()
                      for p in sorted(set(jl["pass"].to_list()))])
    empty = jl.filter(pl.col("n_feasible_samples") == 0).height
    print(f"{lane}: {jl.height} jobs, {empty} with no "
          f"feasible sample; wall time per job median {gaps.median():.1f} s ({gaps.min():.0f} to "
          f"{gaps.max():.0f} s)")
print("max_execution_time below 60 s:", jobs.filter(pl.col("max_execution_time_s") < 60)
      .select("lane", "pass", "max_execution_time_s").rows())
