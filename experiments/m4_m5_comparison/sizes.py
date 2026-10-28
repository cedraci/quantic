"""How many variables, hence qubits, does each QUBO of the studies have?

Prints section 3.7's tables: for each tier, dial combination and encoding, the range of
``variable_count`` over the five seeds (2026 to 2030) at the studies' eight zeta bits.
"""

import sys

import polars as pl

from quantic.bench.instances import tier_instances
from quantic.encoding.diagnostics import variable_count
from quantic.encoding.integer import ENCODINGS
from quantic.solvers.qubo.sampler import STUDY_ZETA_BITS

sys.stdout.reconfigure(encoding="utf-8")

# The tables' row order: without CVaR first, then with it, each by number of dials.
ORDER = ["none", "D1", "D2", "D4", "D1+D2", "D1+D4", "D2+D4", "D1+D2+D4", "D3", "D1+D3", "D2+D3",
         "D3+D4", "D1+D2+D3", "D1+D3+D4", "D2+D3+D4", "D1+D2+D3+D4"]

rows = []
for tier in ("T0", "T1"):
    for seed in range(2026, 2031):
        for inst in tier_instances(tier, seed):
            for enc in ENCODINGS:
                rows.append({"tier": tier, "dials": inst.dials.label, "seed": seed,
                             "encoding": enc.name,
                             "n_vars": variable_count(inst, enc, n_zeta_bits=STUDY_ZETA_BITS)})
t = pl.DataFrame(rows)


def span(values: pl.Series) -> str:
    lo, hi = values.min(), values.max()
    return f"{lo}" if lo == hi else f"{lo} to {hi}"


for tier in ("T0", "T1"):
    print(f"\n{tier}\n| dials | binary | domain-wall | one-hot |\n|---|---|---|---|")
    for dials in ORDER:
        cells = [span(t.filter((pl.col("tier") == tier) & (pl.col("dials") == dials)
                               & (pl.col("encoding") == enc))["n_vars"])
                 for enc in ("binary", "domain_wall", "one_hot")]
        print(f"| `{dials}` | " + " | ".join(cells) + " |")
