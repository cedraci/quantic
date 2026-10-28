"""Section 5's T0 tables: the M4 crossover rows at 0.1 s and 60 s, sound CP-SAT, and the "
"best-over-budgets winners."""

import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
D = str(Path(__file__).resolve().parents[2] / "docs" / "figures" / "m5" / "data") + "/"
ORDER = [
    "none",
    "D1",
    "D2",
    "D3",
    "D4",
    "D1+D2",
    "D1+D3",
    "D1+D4",
    "D2+D3",
    "D2+D4",
    "D3+D4",
    "D1+D2+D3",
    "D1+D2+D4",
    "D1+D3+D4",
    "D2+D3+D4",
    "D1+D2+D3+D4",
]
for b in ("0.1", "60"):
    x = pl.read_csv(D + f"m4-crossover_{b}s.csv").filter(pl.col("tier") == "T0")
    cp = (
        pl.read_csv(D + f"m4-cpsat-crossover_{b}s.csv")
        .filter(pl.col("tier") == "T0")
        .rename({"cpsat": "cpsat_sound"})
    )
    x = x.join(cp, on=["tier", "dials"])
    print(f"\n#### {b} s budget\n")
    print("| dials | HiGHS | CP-SAT (sound) | greedy | tabu | simulated annealing |")
    print("|---|---|---|---|---|---|")
    for d in ORDER:
        r = x.filter(pl.col("dials") == d).row(0, named=True)

        def f(s):
            return s.replace("_", "-") if s else "-"

        print(
            f"| `{d}` | {f(r['exact_pwl_milp'])} | {f(r['cpsat_sound'])} | {f(r['greedy'])} | "
            f"{f(r['tabu'])} | {f(r['simulated_annealing'])} |"
        )

# Best over budgets and encodings, per T0 cell: who wins TTS99 at 1%
m4 = pl.read_csv(D + "m4-cells.csv", infer_schema_length=10000).filter(pl.col("tier") == "T0")
cps = pl.read_csv(D + "m4-cpsat-cells.csv", infer_schema_length=10000).filter(
    pl.col("tier") == "T0"
)
rows = pl.concat(
    [
        m4.filter(
            pl.col("solver").is_in(["exact_pwl_milp", "greedy", "tabu", "simulated_annealing"])
        ),
        cps.with_columns(pl.lit("cpsat_sound").alias("solver")),
    ],
    how="diagonal_relaxed",
)
best = (
    rows.filter(pl.col("tts99_s_0.01").is_not_null())
    .sort("tts99_s_0.01")
    .group_by("dials", "solver", maintain_order=True)
    .first()
    .select("dials", "solver", "encoding", "budget_s", "tts99_s_0.01")
)
print("\n#### best over budgets (and encodings), T0\n")
print("| dials | HiGHS | CP-SAT (sound) | greedy | tabu | simulated annealing | winner |")
print("|---|---|---|---|---|---|---|")
NAME = {
    "exact_pwl_milp": "HiGHS",
    "cpsat_sound": "CP-SAT",
    "greedy": "greedy",
    "tabu": "tabu",
    "simulated_annealing": "SA",
}
for d in ORDER:
    b = best.filter(pl.col("dials") == d)
    cells = {}
    for r in b.iter_rows(named=True):
        enc = f", {r['encoding'].replace('_', '-')}" if r["encoding"] else ""
        cells[r["solver"]] = (
            r["tts99_s_0.01"],
            f"{r['tts99_s_0.01']:.3g} s ({r['budget_s']:g} s{enc})",
        )
    win = min(cells.items(), key=lambda kv: kv[1][0])[0]

    def g(s, cells=cells):
        return cells[s][1] if s in cells else "censored"

    print(
        f"| `{d}` | {g('exact_pwl_milp')} | {g('cpsat_sound')} | {g('greedy')} | {g('tabu')} | "
        f"{g('simulated_annealing')} | {NAME[win]} |"
    )
