"""Every derived number the M5 result and the M4/M5 comparison quote, from the committed CSVs.

    .venv/Scripts/python.exe -I facts.py [p2|p1]

Reads docs/figures/m5/data/m5-<rung>-*.csv and m5-hardware-jobs.csv; prints labelled facts.
"""

import sys
from pathlib import Path

import polars as pl
from scipy.stats import binom, poisson, spearmanr

sys.stdout.reconfigure(encoding="utf-8")

D = str(Path(__file__).resolve().parents[2] / "docs" / "figures" / "m5" / "data") + "/"
rung = sys.argv[1] if len(sys.argv) > 1 else "p2"
REPS = {"p2": 2, "p1": 1}[rung]
P_CZ = 0.0025
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)


def read(name):
    return pl.read_csv(D + f"m5-{rung}-{name}.csv", infer_schema_length=10000)


def say(label, value):
    print(f"{label}: {value}")


uni = read("uniform")
noise = read("qaoa_noise")
cells = read("cells")
train = read("qaoa_training")
mr = read("matched_reads")
jobs_all = pl.read_csv(D + "m5-hardware-jobs.csv", infer_schema_length=10000)
lane = "reps2" if rung == "p2" else "reps1"
jobs = jobs_all.filter(pl.col("lane") == lane)

print(f"===== rung {rung} (reps = {REPS})")
# ---------------------------------------------------------------- hardware, pooled
say("hardware jobs", jobs.height)
say("passes", sorted(jobs["pass"].unique().to_list()))
say("instances on hardware", jobs.select("dials", "instance_seed").unique().height)
shots = int(jobs["n_samples"].sum())
feas = int(jobs["n_feasible_samples"].sum())
say("shots", shots)
say("feasible samples", feas)
say("pooled feasible share %", round(100 * feas / shots, 3))
say("jobs with no feasible sample", jobs.filter(pl.col("n_feasible_samples") == 0).height)
say("jobs within 1%", int(jobs["success_0.01"].sum()))
say("QPU seconds (ledger usage_s)", jobs["usage_s"].sum())
say("QPU s per job: min/max", (jobs["usage_s"].min(), jobs["usage_s"].max()))
say(
    "usage 2 s / 3 s jobs",
    (jobs.filter(pl.col("usage_s") == 2).height, jobs.filter(pl.col("usage_s") == 3).height),
)
say(
    "queue s: median/min/max",
    (
        round(jobs["queue_s"].median(), 1),
        round(jobs["queue_s"].min(), 1),
        round(jobs["queue_s"].max(), 1),
    ),
)
for p in sorted(jobs["pass"].unique().to_list()):
    j = jobs.filter(pl.col("pass") == p)
    say(
        f"  pass {p}: jobs, within 1%, feasible, usage, queue median/max",
        (
            j.height,
            int(j["success_0.01"].sum()),
            int(j["n_feasible_samples"].sum()),
            j["usage_s"].sum(),
            round(j["queue_s"].median(), 1),
            round(j["queue_s"].max(), 1),
        ),
    )

# per instance
inst = (
    jobs.group_by("dials", "instance_seed")
    .agg(
        pl.len().alias("jobs"),
        pl.col("success_0.01").sum().alias("k"),
        pl.col("n_feasible_samples").sum().alias("feas"),
        pl.col("routed_2q").first().alias("cz"),
        pl.col("routed_depth").first().alias("depth"),
    )
    .sort("cz", "dials", "instance_seed")
)
succ = inst.filter(pl.col("k") > 0)
say("instances with >= 1 success", f"{succ.height} of {inst.height}")
say("max CZ of an instance with a success", succ["cz"].max())
say(
    "successful instances (dials-seed cz k/jobs)",
    [
        f"{r['dials']}-{r['instance_seed']} {r['cz']} {r['k']}/{r['jobs']}"
        for r in succ.iter_rows(named=True)
    ],
)
thr = succ["cz"].max()
shallow = inst.filter(pl.col("cz") <= thr)
deep = inst.filter(pl.col("cz") > thr)
say("instances (jobs) at <= threshold CZ", (shallow.height, int(shallow["jobs"].sum())))
say(
    "instances (jobs) above threshold CZ, successes",
    (deep.height, int(deep["jobs"].sum()), int(deep["k"].sum())),
)
say("deep instances min CZ", deep["cz"].min() if deep.height else None)
say("CZ range of device circuits", (inst["cz"].min(), inst["cz"].max()))
say("depth range of device circuits", (inst["depth"].min(), inst["depth"].max()))
say(
    "error-free at min/max CZ %",
    (
        round(100 * (1 - P_CZ) ** inst["cz"].min(), 2),
        round(100 * (1 - P_CZ) ** inst["cz"].max(), 2),
    ),
)
say("instances whose jobs all returned no feasible sample", inst.filter(pl.col("feas") == 0).height)

# ---------------------------------------------------------------- uniform baseline
exp = float(uni["expected_feasible"].sum())
say("uniform: expected feasible", round(exp, 1))
say("uniform: ratio pooled", round(feas / exp, 3))
say("uniform: Poisson tail p", f"{poisson.sf(feas - 1, exp):.1e}")
say(
    "uniform: device ratio median, min, max",
    (
        round(uni["ratio_hw"].median(), 2),
        round(uni["ratio_hw"].min(), 2),
        round(uni["ratio_hw"].max(), 2),
    ),
)
say(
    "uniform: noiseless ratio median, min, max",
    (
        round(uni["ratio_ideal"].median(), 1),
        round(uni["ratio_ideal"].min(), 2),
        round(uni["ratio_ideal"].max(), 0),
    ),
)
chance = float((uni["p_success_uniform"] * uni["hw_jobs"]).sum())
say("uniform: successes expected by chance (all)", round(chance, 2))
ex = uni.filter(pl.col("exact"))
n_ex, s_ex = int(ex["hw_jobs"].sum()), int(ex["hw_success"].sum())
p_ex = float((ex["p_success_uniform"] * ex["hw_jobs"]).sum()) / n_ex
say(
    "uniform: 16-qubit jobs, successes, expected, binomial tail",
    (n_ex, s_ex, round(p_ex * n_ex, 2), round(binom.sf(s_ex - 1, n_ex, p_ex), 4)),
)
ne = uni.filter(~pl.col("exact"))
say(
    "uniform: >16-qubit jobs, successes, expected",
    (
        int(ne["hw_jobs"].sum()),
        int(ne["hw_success"].sum()),
        round(float((ne["p_success_uniform"] * ne["hw_jobs"]).sum()), 2),
    ),
)
say(
    "uniform: 16-qubit p(success) per job",
    sorted(set(round(x, 4) for x in ex["p_success_uniform"].to_list())),
)

# ---------------------------------------------------------------- the ladder
b = noise.filter(pl.col("encoding") == "binary")
wide = b.pivot(
    on="solver",
    index=["dials", "instance_seed"],
    values=["p_success_0.01", "feasibility", "n_runs"],
)
hw_inst = b.filter(pl.col("solver") == "qaoa_ibm").select("dials", "instance_seed", "routed_2q")
ideal = b.filter(pl.col("solver") == "qaoa_ideal").join(hw_inst, on=["dials", "instance_seed"])
say("noiseless over hardware instances: mean success", round(ideal["p_success_0.01"].mean(), 3))
hwrow = b.filter(pl.col("solver") == "qaoa_ibm")
say("device over hardware instances: mean success", round(hwrow["p_success_0.01"].mean(), 3))
# pooled noiseless feasibility over hardware instances:
# the n_runs-weighted mean of feasibility (each run is 1,000 shots)
say(
    "noiseless pooled feasibility over hw instances %",
    round(100 * float((ideal["feasibility"] * ideal["n_runs"]).sum() / ideal["n_runs"].sum()), 2),
)
noisy = b.filter(pl.col("solver") == "qaoa_noisy")
full = noisy.filter(pl.col("feasibility").is_not_nan() & pl.col("feasibility").is_not_null())
killed_all = noisy.filter(pl.col("feasibility").is_nan() | pl.col("feasibility").is_null())
say("noisy instances (binary)", noisy.height)
say(
    "noisy instances all killed",
    [f"{r['dials']}-{r['instance_seed']}" for r in killed_all.iter_rows(named=True)],
)
lad = (
    noisy.join(hw_inst, on=["dials", "instance_seed"])  # noise-model instances that also ran on hw
    .select(
        "dials",
        "instance_seed",
        pl.col("p_success_0.01").alias("noisy"),
        pl.col("feasibility").alias("noisy_f"),
    )
    .join(
        ideal.select(
            "dials",
            "instance_seed",
            pl.col("p_success_0.01").alias("ideal"),
            pl.col("feasibility").alias("ideal_f"),
        ),
        on=["dials", "instance_seed"],
    )
    .join(
        hwrow.select(
            "dials",
            "instance_seed",
            pl.col("p_success_0.01").alias("hw"),
            pl.col("feasibility").alias("hw_f"),
            "routed_2q",
            "n_runs",
        ),
        on=["dials", "instance_seed"],
    )
    .sort("dials", "instance_seed")
)
say("ladder instances (noise model x hardware)", lad.height)
say(
    "ladder means: noiseless, noisy (killed=0), device",
    (
        round(lad["ideal"].mean(), 3),
        round(lad["noisy"].fill_nan(0).mean(), 3),
        round(lad["hw"].mean(), 3),
    ),
)
okl = lad.filter(pl.col("noisy_f").is_not_nan())
say("ladder instances simulated in full", okl.height)
say(
    "noisy vs device success: above, below, equal",
    (
        okl.filter(pl.col("noisy") > pl.col("hw")).height,
        okl.filter(pl.col("noisy") < pl.col("hw")).height,
        okl.filter(pl.col("noisy") == pl.col("hw")).height,
    ),
)
say(
    "noisy feasible range %",
    (round(100 * okl["noisy_f"].min(), 2), round(100 * okl["noisy_f"].max(), 2)),
)
say(
    "device feasible range on ladder %",
    (round(100 * lad["hw_f"].min(), 2), round(100 * lad["hw_f"].max(), 2)),
)
print(lad)

# ---------------------------------------------------------------- matched reads, binary
mb = mr.filter(pl.col("encoding") == "binary")
for s in ["greedy", "simulated_annealing", "tabu", "qaoa_ideal", "qaoa_noisy", "qaoa_ibm"]:
    x = mb.filter(pl.col("solver") == s)
    if x.height:
        say(
            f"matched {s}: success range, feasibility range %",
            (
                (x["p_success_0.01"].min(), x["p_success_0.01"].max()),
                (round(100 * x["feasibility"].min(), 2), round(100 * x["feasibility"].max(), 2)),
            ),
        )
print(
    mb.select("dials", "solver", "n_runs", "p_success_0.01", "feasibility", "tts").sort(
        "dials", "solver"
    )
)

# ---------------------------------------------------------------- TTS99
tt = (
    cells.filter(pl.col("solver") == "qaoa_ibm")
    .select(
        "dials",
        "n_runs",
        "p_success_0.01",
        "t_run_s",
        "tts99_s_0.01",
        "tts99_lo_s_0.01",
        "tts99_hi_s_0.01",
    )
    .sort("dials")
)
print(tt)
gs = (
    cells.filter(pl.col("solver").is_in(["greedy", "simulated_annealing", "tabu"]))
    .group_by("solver", maintain_order=True)
    .agg(pl.col("tts99_s_0.01").min().alias("min"), pl.col("tts99_s_0.01").max().alias("max"))
)
print(gs)

# ---------------------------------------------------------------- training
say("trainings", train.height)
say("trainings by encoding", train.group_by("encoding").len().sort("encoding").rows())
say("training total hours", round(train["train_s"].sum() / 3600, 2))
say("trainings at maxiter (200)", train.filter(pl.col("n_evals") >= 200).height)
say("n_evals of the others", sorted(train.filter(pl.col("n_evals") < 200)["n_evals"].to_list()))
print(
    train.group_by("n_vars")
    .agg(
        pl.len().alias("n"),
        pl.col("train_s").median().alias("median"),
        pl.col("train_s").max().alias("max"),
    )
    .sort("n_vars")
)
med = train.group_by("n_vars").agg(pl.col("train_s").median()).sort("n_vars")
lo, hi = med.row(0), med.row(-1)
say("growth per qubit (median, first to last)", round((hi[1] / lo[1]) ** (1 / (hi[0] - lo[0])), 2))
say(
    "dynamic range min/max",
    (f"{train['dynamic_range'].min():.2e}", f"{train['dynamic_range'].max():.2e}"),
)
say(
    "ising dynamic range min/max",
    (f"{train['ising_dynamic_range'].min():.2e}", f"{train['ising_dynamic_range'].max():.2e}"),
)

# ---------------------------------------------------------------- dynamic range vs success
tb = train.filter(pl.col("encoding") == "binary").join(
    b.filter(pl.col("solver") == "qaoa_ideal").select("dials", "instance_seed", "p_success_0.01"),
    on=["dials", "instance_seed"],
)
r1 = spearmanr(tb["dynamic_range"].log(), tb["p_success_0.01"])
r2 = spearmanr(tb["n_vars"], tb["p_success_0.01"])
say("Spearman range vs noiseless success", (round(r1.statistic, 2), f"{r1.pvalue:.2g}"))
say("Spearman qubits vs noiseless success", (round(r2.statistic, 2), f"{r2.pvalue:.2g}"))
for q in sorted(tb["n_vars"].unique().to_list()):
    g = tb.filter(pl.col("n_vars") == q)
    if g.height >= 4:
        r = spearmanr(g["dynamic_range"].log(), g["p_success_0.01"])
        say(f"  within {q} qubits (n={g.height})", (round(r.statistic, 2), f"{r.pvalue:.2g}"))
th = train.filter(pl.col("encoding") == "binary").join(
    hwrow.select("dials", "instance_seed", "p_success_0.01"), on=["dials", "instance_seed"]
)
r3 = spearmanr(th["dynamic_range"].log(), th["p_success_0.01"])
say("Spearman range vs device success", (round(r3.statistic, 2), f"{r3.pvalue:.2g}"))
