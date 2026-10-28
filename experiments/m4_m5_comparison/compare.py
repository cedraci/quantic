"""Section 8's cross-study numbers: the eight shared T0 cells, every study, every clock.

.venv/Scripts/python.exe -I compare.py <part>
"""

import math
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

D = str(Path(__file__).resolve().parents[2] / "docs" / "figures" / "m5" / "data") + "/"
CELLS = ["none", "D1", "D4", "D1+D4", "D2", "D1+D2", "D2+D4", "D1+D2+D4"]
K = ["dials", "instance_seed"]
m4 = pl.read_csv(D + "m4-cells.csv", infer_schema_length=10000).filter(pl.col("tier") == "T0")
cps = pl.read_csv(D + "m4-cpsat-cells.csv", infer_schema_length=10000).filter(
    pl.col("tier") == "T0"
)
c2 = pl.read_csv(D + "m5-p2-cells.csv", infer_schema_length=10000)
c1 = pl.read_csv(D + "m5-p1-cells.csv", infer_schema_length=10000)
t2 = pl.read_csv(D + "m5-p2-qaoa_training.csv")
t1 = pl.read_csv(D + "m5-p1-qaoa_training.csv")
n2 = pl.read_csv(D + "m5-p2-qaoa_noise.csv")
n1 = pl.read_csv(D + "m5-p1-qaoa_noise.csv")
jobs = pl.read_csv(D + "m5-hardware-jobs.csv").sort("submitted_at")
TTS = "tts99_s_0.01"
P = "p_success_0.01"
part = sys.argv[1]


def best(frame, solver, d):
    """Smallest TTS99 over budgets and encodings: (value, budget, encoding) or None."""
    r = frame.filter(
        (pl.col("solver") == solver) & (pl.col("dials") == d) & pl.col(TTS).is_not_null()
    )
    if not r.height:
        return None
    r = r.sort(TTS).row(0, named=True)
    return r[TTS], r["budget_s"], r["encoding"]


def row(frame, solver, d, enc="binary"):
    r = frame.filter(
        (pl.col("solver") == solver)
        & (pl.col("dials") == d)
        & (pl.col("encoding") == enc if enc else pl.lit(True))
    )
    return r.row(0, named=True) if r.height else None


def g(x, d=3):
    return "—" if x is None else f"{x:.{d}g}"


# wall-clock per hardware job: median gap between consecutive submissions in a lane run
gaps = jobs.with_columns(
    (pl.col("submitted_at") - pl.col("submitted_at").shift()).over(["lane", "pass"]).alias("gap")
)
WALL2 = float(gaps.filter(pl.col("lane") == "reps2")["gap"].drop_nulls().median())
WALL1 = float(gaps.filter(pl.col("lane") == "reps1")["gap"].drop_nulls().median())


def train_mean(t, d):
    """Training seconds of the binary QUBOs of a cell whose instance ran on the device."""
    noise = n2 if t is t2 else n1
    hw = noise.filter(pl.col("solver") == "qaoa_ibm").select(K)
    return t.filter((pl.col("dials") == d) & (pl.col("encoding") == "binary")).join(hw, on=K)[
        "train_s"
    ]


if part == "time":
    print(f"wall per job: reps2 {WALL2:.1f} s, reps1 {WALL1:.1f} s")
    print(
        "| cell | CP-SAT, best of M4 budgets | HiGHS, best of M4 budgets | best M4 sampler "
        "(budget, encoding) | greedy, 1,000 reads | annealing, 1,000 reads | device, reps 2, "
        "QPU | device, reps 1, QPU | device, reps 2, end to end | device, reps 1, end to end |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in CELLS:
        cp = best(cps, "cpsat", d)
        hi = best(m4, "exact_pwl_milp", d)
        samp = [(best(m4, s, d), s) for s in ("greedy", "tabu", "simulated_annealing")]
        samp = [(b, s) for b, s in samp if b]
        bs, bsn = min(samp, key=lambda x: x[0][0])
        gr = min(
            r[TTS]
            for r in c2.filter((pl.col("solver") == "greedy") & (pl.col("dials") == d)).iter_rows(
                named=True
            )
        )
        sa = min(
            r[TTS]
            for r in c2.filter(
                (pl.col("solver") == "simulated_annealing") & (pl.col("dials") == d)
            ).iter_rows(named=True)
        )
        out = [
            f"`{d}`",
            f"{cp[0]:.3g} s",
            f"{hi[0]:.3g} s",
            f"{bs[0]:.3g} s ({bsn.replace('simulated_annealing', 'annealing')}, {bs[1]:g} s, "
            f"{bs[2].replace('_', '-')})",
            f"{gr:.3g} s",
            f"{sa:.3g} s",
        ]
        e2e = []
        for c, t, wall in ((c2, t2, WALL2), (c1, t1, WALL1)):
            r = row(c, "qaoa_ibm", d)
            p, n = r[P], r["n_runs"]
            out.append("censored" if r[TTS] is None else f"{r[TTS]:.3g} s")
            tr = float(train_mean(t, d).mean())
            if p <= 0:
                e2e.append(f"censored (training {tr / 60:.3g} min)")
            else:
                reps = 1.0 if p >= 1 else math.log(0.01) / math.log(1 - p)
                e2e.append(f"{(tr + wall * reps) / 60:.3g} min")
        out += e2e
        print("| " + " | ".join(out) + " |")

if part == "quality":
    print(
        "| cell | greedy | annealing | tabu | noiseless, reps 2 | noiseless, reps 1 | noise "
        "model, reps 2 | noise model, reps 1 | device, reps 2 | device, reps 1 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in CELLS:
        out = [f"`{d}`"]
        for s in ("greedy", "simulated_annealing", "tabu"):
            r = row(c2, s, d)
            out.append(f"{r[P]:.2f} / {100 * r['feasibility']:.0f}%")
        for c in (c2, c1):
            r = row(c, "qaoa_ideal", d)
            out.append(f"{r[P]:.2f} / {100 * r['feasibility']:.1f}%")
        for c in (c2, c1):
            r = row(c, "qaoa_noisy", d)
            if r is None:
                out.append("—")
            elif r["n_killed"] == r["n_runs"]:
                out.append("killed")
            else:
                out.append(f"{r[P]:.2f} / {100 * r['feasibility']:.1f}%")
        for c in (c2, c1):
            r = row(c, "qaoa_ibm", d)
            out.append(
                f"{round(r[P] * r['n_runs'])} of {r['n_runs']} / {100 * r['feasibility']:.2g}%"
            )
        print("| " + " | ".join(out) + " |")

if part == "factors":
    # pooled over the 8 cells (binary): mean success and pooled feasibility
    def agg(c, s):
        r = c.filter((pl.col("solver") == s) & (pl.col("encoding") == "binary"))
        return r[P].mean(), (r["feasibility"] * r["n_runs"]).sum() / r["n_runs"].sum()

    for name, c, s in (
        ("greedy", c2, "greedy"),
        ("annealing", c2, "simulated_annealing"),
        ("tabu", c2, "tabu"),
        ("noiseless r2", c2, "qaoa_ideal"),
        ("noiseless r1", c1, "qaoa_ideal"),
        ("noise r2", c2, "qaoa_noisy"),
        ("noise r1", c1, "qaoa_noisy"),
        ("device r2", c2, "qaoa_ibm"),
        ("device r1", c1, "qaoa_ibm"),
    ):
        m, f = agg(c, s)
        print(f"{name}: mean cell success {m:.3f}, run-weighted feasibility {100 * f:.2f}%")
    # time factors: device QPU TTS over the best classical (CP-SAT or greedy at 1,000 reads)
    for lab, c in (("r2", c2), ("r1", c1)):
        for d in CELLS:
            r = row(c, "qaoa_ibm", d)
            if r[TTS] is None:
                continue
            cp = best(cps, "cpsat", d)[0]
            gr = min(
                x[TTS]
                for x in c2.filter(
                    (pl.col("solver") == "greedy") & (pl.col("dials") == d)
                ).iter_rows(named=True)
            )
            print(
                f"{lab} {d}: device {r[TTS]:.3g} s; / CP-SAT {r[TTS] / cp:.0f}x; / greedy(1000) "
                f"{r[TTS] / gr:.0f}x"
            )

if part == "dials":
    # 2^3 factorial over D1, D2, D4 on the 8 cells: main effect = mean(with) - mean(without)
    def has(label, dial):
        return dial in label.split("+")

    def effects(values):
        out = {}
        for dial in ("D1", "D2", "D4"):
            w = [v for k, v in values.items() if has(k, dial)]
            wo = [v for k, v in values.items() if not has(k, dial)]
            out[dial] = sum(w) / len(w) - sum(wo) / len(wo)
        return out

    series = {}
    for name, c, s in (
        ("greedy", c2, "greedy"),
        ("annealing", c2, "simulated_annealing"),
        ("tabu", c2, "tabu"),
        ("noiseless r2", c2, "qaoa_ideal"),
        ("noiseless r1", c1, "qaoa_ideal"),
        ("device r2", c2, "qaoa_ibm"),
        ("device r1", c1, "qaoa_ibm"),
    ):
        succ = {d: row(c, s, d)[P] for d in CELLS}
        feas = {d: row(c, s, d)["feasibility"] for d in CELLS}
        e1, e2 = effects(succ), effects(feas)
        print(
            f"{name}: success effects "
            + ", ".join(f"{k} {v:+.2f}" for k, v in e1.items())
            + " | feasibility effects "
            + ", ".join(f"{k} {100 * v:+.1f} pts" for k, v in e2.items())
        )
    # qubits per cell
    tb = (
        t2.filter(pl.col("encoding") == "binary")
        .group_by("dials")
        .agg(
            pl.col("n_vars").min().alias("lo"),
            pl.col("n_vars").max().alias("hi"),
            pl.len().alias("n"),
        )
    )
    print(tb.sort("dials"))

if part == "encodings":
    for lab, c in (("reps2", c2), ("reps1", c1)):
        print(lab)
        for d in CELLS:
            b = row(c, "qaoa_ideal", d, "binary")
            w = row(c, "qaoa_ideal", d, "domain_wall")
            if w is None:
                print(
                    f"  {d}: binary {b[P]:.2f}/{100 * b['feasibility']:.1f}% ({b['n_runs']} "
                    f"runs); domain-wall: none"
                )
            else:
                print(
                    f"  {d}: binary {b[P]:.2f}/{100 * b['feasibility']:.1f}% ({b['n_runs']}); "
                    f"domain-wall {w[P]:.2f}/{100 * w['feasibility']:.1f}% ({w['n_runs']})"
                )
    for s in ("greedy", "simulated_annealing", "tabu"):
        print(
            s,
            [
                (
                    d,
                    round(row(c2, s, d, "binary")[P], 2),
                    round(row(c2, s, d, "domain_wall")[P], 2),
                    round(100 * row(c2, s, d, "binary")["feasibility"]),
                    round(100 * row(c2, s, d, "domain_wall")["feasibility"]),
                )
                for d in CELLS
            ],
        )

if part == "cost":
    for lab, t in (("reps2", t2), ("reps1", t1)):
        b = t.filter(pl.col("encoding") == "binary")
        print(
            lab,
            "binary trainings",
            b.height,
            "total h",
            round(b["train_s"].sum() / 3600, 2),
            "all trainings h",
            round(t["train_s"].sum() / 3600, 2),
        )
    hw2 = jobs.filter(pl.col("lane") == "reps2")
    print(
        "reps2 jobs",
        hw2.height,
        "qpu",
        hw2["usage_s"].sum(),
        "queue total s",
        round(hw2["queue_s"].sum()),
        "wall per job median",
        round(WALL2, 1),
    )
    hw1 = jobs.filter(pl.col("lane") == "reps1")
    print(
        "reps1 jobs",
        hw1.height,
        "qpu",
        hw1["usage_s"].sum(),
        "queue total s",
        round(hw1["queue_s"].sum()),
        "wall per job median",
        round(WALL1, 1),
    )
    hi = c2.filter(pl.col("solver") == "exact_pwl_milp")
    print(
        "HiGHS M5 store, cell-mean run time s: min",
        round(hi["t_run_s"].min(), 3),
        "max",
        round(hi["t_run_s"].max(), 3),
        "sum over 40 instances",
        round((hi["t_run_s"] * hi["n_runs"]).sum(), 1),
    )
    cp = cps.filter(pl.col("dials").is_in(CELLS) & (pl.col("budget_s") == 1.0))
    print("CP-SAT sound at 1 s, mean t_run over the 8 cells", round(cp["t_run_s"].mean(), 3))
