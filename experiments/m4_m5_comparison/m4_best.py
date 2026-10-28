"""Best TTS99 at 1% over budgets and encodings, per cell and solver, both tiers; winner and "
"margin."""

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
NAME = {
    "exact_pwl_milp": "HiGHS",
    "cpsat_sound": "CP-SAT",
    "greedy": "greedy",
    "tabu": "tabu",
    "simulated_annealing": "SA",
}
tier = sys.argv[1]
m4 = pl.read_csv(D + "m4-cells.csv", infer_schema_length=10000).filter(pl.col("tier") == tier)
cps = pl.read_csv(D + "m4-cpsat-cells.csv", infer_schema_length=10000).filter(
    pl.col("tier") == tier
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
)
print(
    "| dials | HiGHS | CP-SAT (sound) | greedy | tabu | simulated annealing | winner | best exact "
    "/ best sampler |"
)
print("|---|---|---|---|---|---|---|---|")
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
    ex = min(v[0] for k, v in cells.items() if k in ("exact_pwl_milp", "cpsat_sound"))
    sm = [v[0] for k, v in cells.items() if k not in ("exact_pwl_milp", "cpsat_sound")]
    ratio = f"{ex / min(sm):.3g}" if sm else "no sampler reaches 1%"

    def g(s, cells=cells):
        return cells[s][1] if s in cells else "censored"

    print(
        f"| `{d}` | {g('exact_pwl_milp')} | {g('cpsat_sound')} | {g('greedy')} | {g('tabu')} | "
        f"{g('simulated_annealing')} | {NAME[win]} | {ratio} |"
    )
