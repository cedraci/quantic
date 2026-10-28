"""The M5 hardware output against uniform random bits -- M5 result, section 3.

For every instance with a ``qaoa_ibm`` row in the store: the share of all
bitstrings that decode to a feasible schedule (every one of the 2^n at 16
qubits, ``--draws`` uniform random ones above), how many sit within 1% of the
proven optimum, what the hardware jobs returned, and the noiseless circuit's
feasibility, all through the study's own decoder and scorer (``run_sampler``).

    .venv/Scripts/python.exe experiments/m5-uniform-baseline.py <store> <out.csv> [--draws 100000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy.stats import binom, poisson

from quantic.bench.instances import InstanceCache
from quantic.encoding.decode import FEASIBLE
from quantic.encoding.encode import encode
from quantic.solvers.base import Budget
from quantic.solvers.qubo.sampler import STUDY_ZETA_BITS, encoding_named, energies, run_sampler

SHOTS = 1000
EXACT_UP_TO = 16


def feasible_totals(enc, draws: int, seed: int) -> tuple[int, bool, list[tuple[float, int]]]:
    """(bitstrings examined, exact?, [(total, count)] of the feasible ones)."""
    n = enc.n_vars
    exact = n <= EXACT_UP_TO
    if exact:
        bits = ((np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    else:
        bits = np.random.default_rng(seed).integers(0, 2, size=(draws, n), dtype=np.int8)

    def draw(bqm, reads, s):
        return bits, energies(enc.qubo, bits)

    result = run_sampler("uniform", enc, Budget(max_samples=len(bits)), 0, draw, one_batch=True)
    found = [(float(s.total), int(s.count)) for s in result.detail["samples"]
             if s.classification == FEASIBLE]
    return len(bits), exact, found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--draws", type=int, default=100_000)
    args = ap.parse_args()

    con = duckdb.connect(str(args.store), read_only=True)
    hw = con.execute(
        "SELECT tier, dials, instance_seed, encoding, n_feasible_samples, n_samples, total "
        "FROM runs WHERE solver = 'qaoa_ibm' AND store_status = 'ok' "
        "ORDER BY instance_seed, dials, key").pl()
    ideal = con.execute(
        "SELECT tier, dials, instance_seed, encoding, "
        "avg(n_feasible_samples * 1.0 / n_samples) AS f_ideal FROM runs "
        "WHERE solver = 'qaoa_ideal' AND store_status = 'ok' GROUP BY ALL").pl()
    optimum = con.execute(
        "SELECT tier, dials, instance_seed, min(lower) AS opt FROM runs "
        "WHERE solver = 'ref:exact_pwl_milp' GROUP BY ALL").pl()
    keys = ["tier", "dials", "instance_seed", "encoding"]
    instances = hw.select(keys).unique().sort(keys)

    cache = InstanceCache()
    rows = []
    for tier, dials, seed, encoding in instances.iter_rows():
        enc = encode(cache.get(tier, dials, seed), encoding_named(encoding),
                     n_zeta_bits=STUDY_ZETA_BITS)
        examined, exact, found = feasible_totals(enc, args.draws, seed)
        opt = optimum.filter((pl.col("tier") == tier) & (pl.col("dials") == dials)
                             & (pl.col("instance_seed") == seed))["opt"][0]
        jobs = hw.filter((pl.col("tier") == tier) & (pl.col("dials") == dials)
                         & (pl.col("instance_seed") == seed) & (pl.col("encoding") == encoding))
        n_feasible = sum(c for _, c in found)
        k = sum(c for t, c in found if t - opt <= 0.01 * abs(opt))
        uniform_frac = n_feasible / examined
        shots = int(jobs["n_samples"].sum())
        fi = ideal.filter((pl.col("tier") == tier) & (pl.col("dials") == dials)
                          & (pl.col("instance_seed") == seed) & (pl.col("encoding") == encoding))
        f_ideal = float(fi["f_ideal"][0]) if fi.height else None
        rows.append({
            "tier": tier, "dials": dials, "instance_seed": seed, "encoding": encoding,
            "n_vars": enc.n_vars, "exact": exact, "uniform_frac": uniform_frac,
            "exp_per_1000": SHOTS * uniform_frac,
            "hw_jobs": jobs.height,
            "hw_feasible": "+".join(str(int(f)) for f in jobs["n_feasible_samples"]),
            "hw_success": int(sum(t is not None and t - opt <= 0.01 * abs(opt)
                                  for t in jobs["total"])),
            "expected_feasible": uniform_frac * shots,
            "ratio_hw": (float(jobs["n_feasible_samples"].sum()) / (uniform_frac * shots)
                         if uniform_frac else None),
            "ratio_ideal": f_ideal / uniform_frac if f_ideal is not None and uniform_frac else None,
            "k_within_1pct": k,
            "p_success_uniform": 1 - (1 - k / examined) ** SHOTS,
            "optimum": opt,
        })
    table = pl.DataFrame(rows).sort("n_vars", "dials", "instance_seed")
    table.write_csv(args.out)

    feasible = int(hw["n_feasible_samples"].sum())
    expected = float(table["expected_feasible"].sum())
    successes = int(table["hw_success"].sum())
    chance = float((table["p_success_uniform"] * table["hw_jobs"]).sum())
    exact = table.filter(pl.col("exact"))
    n_ex, s_ex = int(exact["hw_jobs"].sum()), int(exact["hw_success"].sum())
    p_ex = float((exact["p_success_uniform"] * exact["hw_jobs"]).sum()) / n_ex if n_ex else 0.0
    print(f"{len(rows)} instances, {hw.height} hardware jobs")
    print(f"feasible samples: {feasible} of {int(hw['n_samples'].sum())} shots; uniform expects "
          f"{expected:.1f} (ratio {feasible / expected:.2f}, Poisson tail "
          f"{poisson.sf(feasible - 1, expected):.1e})")
    print(f"feasible rate over uniform: device median {table['ratio_hw'].median():.2f}, "
          f"noiseless median {table['ratio_ideal'].median():.1f}")
    print(f"within 1% of the optimum: {successes} jobs; uniform expects about {chance:.1f} "
          f"(a lower bound where the draws saw no such bitstring)")
    print(f"exact instances: {s_ex} of {n_ex} jobs, uniform expects {p_ex * n_ex:.1f}, "
          f"binomial tail {binom.sf(s_ex - 1, n_ex, p_ex):.3f}")


if __name__ == "__main__":
    main()
