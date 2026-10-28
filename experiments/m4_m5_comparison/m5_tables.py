"""The M5 result document's in-text tables, from the committed CSVs (printed as Markdown)."""

import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

D = str(Path(__file__).resolve().parents[2] / "docs" / "figures" / "m5" / "data") + "/"
CELLS = ["D1", "D1+D2", "D1+D2+D4", "D1+D4", "D2", "D2+D4", "D4", "none"]
K = ["dials", "instance_seed"]


def pct(x, d=1):
    if x is None:
        return "—"
    v = 100 * x
    if v == 0:
        return "0%"
    if v < 0.1:
        return f"{v:.2f}%"
    if v < 1:
        return f"{v:.2f}%".rstrip("0").rstrip(".") + (
            "%" if not f"{v:.2f}".rstrip("0").rstrip(".").endswith("%") else ""
        )
    return f"{v:.{d}f}%"


def p2f(x):
    return f"{x:.2f}".rstrip("0").rstrip(".") if x not in (0, 1) else ("0" if x == 0 else "1.00")


def kofn(p, n):
    return f"{round(p * n)} of {n}"


which = sys.argv[1]
mr2 = pl.read_csv(D + "m5-p2-matched_reads.csv")
mr1 = pl.read_csv(D + "m5-p1-matched_reads.csv")
n2 = pl.read_csv(D + "m5-p2-qaoa_noise.csv").with_columns(pl.col("feasibility").fill_nan(None))
n1 = pl.read_csv(D + "m5-p1-qaoa_noise.csv").with_columns(pl.col("feasibility").fill_nan(None))
c2 = pl.read_csv(D + "m5-p2-cells.csv", infer_schema_length=10000)
c1 = pl.read_csv(D + "m5-p1-cells.csv", infer_schema_length=10000)


def cell(mr, d, s, e="binary"):
    r = mr.filter((pl.col("dials") == d) & (pl.col("solver") == s) & (pl.col("encoding") == e))
    return r.row(0, named=True) if r.height else None


def f1(x):
    """feasibility as a percent with 2 significant figures."""
    if x is None:
        return "—"
    v = 100 * x
    if v == 0:
        return "0%"
    if v >= 10:
        s = f"{v:.0f}"
    elif v >= 1:
        s = f"{v:.1f}"
    else:
        s = f"{v:.2g}"
    return s + "%"


if which == "matched":
    print(
        "| cell | greedy | simulated annealing | tabu | QAOA, noiseless | QAOA, noise model | "
        "QAOA, `ibm_fez` |"
    )
    print("|---|---|---|---|---|---|---|")
    for d in CELLS:
        out = [f"`{d}`"]
        for s in ("greedy", "simulated_annealing", "tabu", "qaoa_ideal"):
            r = cell(mr2, d, s)
            out.append(f"{r['p_success_0.01']:.2f} / {f1(r['feasibility'])}")
        r = cell(mr2, d, "qaoa_noisy")
        cr = c2.filter(
            (pl.col("dials") == d)
            & (pl.col("solver") == "qaoa_noisy")
            & (pl.col("encoding") == "binary")
        )
        if r is None:
            out.append("— (24 qubits)")
        else:
            nk = cr["n_killed"][0]
            if nk == r["n_runs"]:
                out.append(f"all {nk} runs killed")
            elif nk:
                out.append(
                    f"{r['p_success_0.01']:.2f} / {f1(r['feasibility'])} ({nk} of {r['n_runs']} "
                    f"runs killed)"
                )
            else:
                out.append(f"{r['p_success_0.01']:.2f} / {f1(r['feasibility'])}")
        r = cell(mr2, d, "qaoa_ibm")
        out.append(f"{kofn(r['p_success_0.01'], r['n_runs'])} / {f1(r['feasibility'])}")
        print("| " + " | ".join(out) + " |")

if which == "tts":
    print(
        "| cell | greedy | simulated annealing | tabu | HiGHS (proof, 1 s budget) | QAOA, "
        "`ibm_fez`, reps = 2 (QPU seconds) | QAOA, `ibm_fez`, reps = 1 (QPU seconds) |"
    )
    print("|---|---|---|---|---|---|---|")
    for d in CELLS:
        out = [f"`{d}`"]
        for s in ("greedy", "simulated_annealing", "tabu"):
            r = c2.filter(
                (pl.col("dials") == d) & (pl.col("solver") == s) & (pl.col("encoding") == "binary")
            )
            out.append(f"{r['tts99_s_0.01'][0]:.3g}")
        r = c2.filter((pl.col("dials") == d) & (pl.col("solver") == "exact_pwl_milp"))
        out.append(f"{r['tts99_s_0.01'][0]:.2f}")
        for c in (c2, c1):
            r = c.filter((pl.col("dials") == d) & (pl.col("solver") == "qaoa_ibm"))
            v = r["tts99_s_0.01"][0]
            n = r["n_runs"][0]
            out.append(
                "censored" + f" ({n} jobs)"
                if v is None
                else f"{v:.3g} ({round(r['p_success_0.01'][0] * n)} of {n} jobs)"
            )
        print("| " + " | ".join(out) + " |")

if which == "ladder":
    hw = n2.filter(pl.col("solver") == "qaoa_ibm").select(
        K
        + [
            "routed_2q",
            pl.col("p_success_0.01").alias("hw"),
            pl.col("feasibility").alias("hw_f"),
            pl.col("n_runs").alias("hw_n"),
        ]
    )
    tr = (
        pl.read_csv(D + "m5-p2-qaoa_training.csv")
        .filter(pl.col("encoding") == "binary")
        .select(K + ["n_vars"])
    )
    ide = n2.filter((pl.col("solver") == "qaoa_ideal") & (pl.col("encoding") == "binary")).select(
        K + [pl.col("p_success_0.01").alias("ideal"), pl.col("feasibility").alias("ideal_f")]
    )
    noi = n2.filter((pl.col("solver") == "qaoa_noisy") & (pl.col("encoding") == "binary")).select(
        K + [pl.col("p_success_0.01").alias("noisy"), pl.col("feasibility").alias("noisy_f")]
    )
    t = hw.join(noi, on=K).join(ide, on=K).join(tr, on=K).sort("dials", "instance_seed")
    print(
        "| instance | qubits | CZ, routed | noiseless (20 runs) | noise model (5 runs) | "
        "`ibm_fez` (3 jobs) |"
    )
    print("|---|---|---|---|---|---|")
    for r in t.iter_rows(named=True):
        noisy = (
            "0 (all killed)" if r["noisy_f"] is None else f"{r['noisy']:.2f} ({f1(r['noisy_f'])})"
        )
        hwk = f"{round(r['hw'] * r['hw_n'])} of {r['hw_n']}"
        print(
            f"| `{r['dials']}-{r['instance_seed']}` | {r['n_vars']} | {r['routed_2q']:,.0f} | "
            f"{r['ideal']:.2f} ({f1(r['ideal_f'])}) | {noisy} | {hwk} ({f1(r['hw_f'])}) |"
        )
    m_i, m_n, m_h = t["ideal"].mean(), t["noisy"].fill_null(0).mean(), t["hw"].mean()
    print(
        f"| **mean** | | | **{m_i:.2f}** | **{m_n:.2f}** (killed instances as 0) | **{m_h:.2f}** |"
    )

if which == "cellsrung":
    print(
        "| cell | noiseless success, reps 2 → 1 | noiseless feasible, reps 2 → 1 | noise model "
        "success, reps 2 → 1 | device, reps 2 (jobs within 1%) | device, reps 1 | device "
        "feasible, reps 2 → 1 |"
    )
    print("|---|---|---|---|---|---|---|")
    for d in CELLS:
        a2, a1 = cell(mr2, d, "qaoa_ideal"), cell(mr1, d, "qaoa_ideal")
        b2, b1 = cell(mr2, d, "qaoa_noisy"), cell(mr1, d, "qaoa_noisy")
        h2, h1 = cell(mr2, d, "qaoa_ibm"), cell(mr1, d, "qaoa_ibm")

        def ns(b, c, d=d):
            if b is None:
                return "—"
            cr = c.filter(
                (pl.col("dials") == d)
                & (pl.col("solver") == "qaoa_noisy")
                & (pl.col("encoding") == "binary")
            )
            nk = cr["n_killed"][0]
            if nk == b["n_runs"]:
                return "killed"
            return f"{b['p_success_0.01']:.2f}" + (f" ({nk} of {b['n_runs']} killed)" if nk else "")

        print(
            f"| `{d}` | {a2['p_success_0.01']:.2f} → {a1['p_success_0.01']:.2f} | "
            f"{f1(a2['feasibility'])} → {f1(a1['feasibility'])} | {ns(b2, c2)} → {ns(b1, c1)} | "
            f"{kofn(h2['p_success_0.01'], h2['n_runs'])} | "
            f"{kofn(h1['p_success_0.01'], h1['n_runs'])} | {f1(h2['feasibility'])} → "
            f"{f1(h1['feasibility'])} |"
        )

if which == "perinstance":

    def side(n, tag):
        b = n.filter(pl.col("encoding") == "binary")
        hw = b.filter(pl.col("solver") == "qaoa_ibm").select(
            K
            + [
                pl.col("routed_2q").alias(f"cz{tag}"),
                pl.col("p_success_0.01").alias(f"hw{tag}"),
                pl.col("n_runs").alias(f"hn{tag}"),
                pl.col("feasibility").alias(f"hf{tag}"),
            ]
        )
        ide = b.filter(pl.col("solver") == "qaoa_ideal").select(
            K + [pl.col("p_success_0.01").alias(f"i{tag}")]
        )
        noi = b.filter(pl.col("solver") == "qaoa_noisy").select(
            K + [pl.col("p_success_0.01").alias(f"n{tag}"), pl.col("feasibility").alias(f"nf{tag}")]
        )
        return ide.join(hw, on=K, how="left").join(noi, on=K, how="left")

    tr = (
        pl.read_csv(D + "m5-p2-qaoa_training.csv")
        .filter(pl.col("encoding") == "binary")
        .select(K + ["n_vars"])
    )
    t = side(n2, "2").join(side(n1, "1"), on=K, how="full", coalesce=True).join(tr, on=K)
    t = t.filter(pl.col("cz2").is_not_null() | pl.col("cz1").is_not_null()).sort(
        pl.col("cz2").fill_null(9999), "cz1"
    )
    print(
        "| instance | qubits | CZ, reps 2 → 1 | noiseless, 2 → 1 | noise model, 2 → 1 | device, "
        "reps 2 | device, reps 1 | device feasible, 2 → 1 |"
    )
    print("|---|---|---|---|---|---|---|---|")

    def nm(v, f):
        if v is None:
            return "—"
        return "killed" if f is None else f"{v:.2f}"

    for r in t.iter_rows(named=True):
        cz2 = "—" if r["cz2"] is None else f"{r['cz2']:,.0f}"
        dev2 = "—" if r["hw2"] is None else kofn(r["hw2"], r["hn2"])
        dev1 = "—" if r["hw1"] is None else kofn(r["hw1"], r["hn1"])
        print(
            f"| `{r['dials']}-{r['instance_seed']}` | {r['n_vars']} | {cz2} → {r['cz1']:,.0f} | "
            f"{r['i2']:.2f} → {r['i1']:.2f} | {nm(r['n2'], r['nf2'])} → {nm(r['n1'], r['nf1'])} | "
            f"{dev2} | {dev1} | {f1(r['hf2'])} → {f1(r['hf1'])} |"
        )

if which == "training":
    t2 = pl.read_csv(D + "m5-p2-qaoa_training.csv")
    t1 = pl.read_csv(D + "m5-p1-qaoa_training.csv")
    g2 = t2.group_by("n_vars").agg(
        pl.len().alias("n"),
        pl.col("train_s").median().alias("m2"),
        pl.col("train_s").max().alias("x2"),
    )
    g1 = t1.group_by("n_vars").agg(
        pl.col("train_s").median().alias("m1"),
        pl.col("train_s").max().alias("x1"),
        pl.col("n_evals").median().alias("e1"),
    )
    g = g2.join(g1, on="n_vars").sort("n_vars")

    def tm(s):
        return f"{s:,.0f} s" if s >= 10 else f"{s:.1f} s"

    print(
        "| qubits | trainings | reps 2: median | reps 2: max | reps 1: median | reps 1: max | "
        "reps 1: median evaluations |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in g.iter_rows(named=True):
        print(
            f"| {r['n_vars']} | {r['n']} | {tm(r['m2'])} | {tm(r['x2'])} | {tm(r['m1'])} | "
            f"{tm(r['x1'])} | {r['e1']:.0f} |"
        )
