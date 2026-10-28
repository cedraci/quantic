"""How dense are the QUBOs, and what would a T1 circuit need?

Counts non-zero couplings (off-diagonal QUBO terms) at T0 and T1 with the study's encoder, and
relates the T0 counts to the routed CZ counts of the device circuits (routing overhead).
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

from quantic.bench.instances import tier_instances
from quantic.encoding.encode import encode
from quantic.solvers.qubo.sampler import STUDY_ZETA_BITS, encoding_named

sys.stdout.reconfigure(encoding="utf-8")
D = str(Path(__file__).resolve().parents[2] / "docs" / "figures" / "m5" / "data") + "/"

DIALS = ["none", "D1", "D2", "D4", "D1+D2", "D1+D4", "D2+D4", "D1+D2+D4"]


def couplings(enc):
    q = enc.qubo
    m = q.matrix if hasattr(q, "matrix") else None
    if m is None:
        for name in ("Q", "q", "dense", "to_dense"):
            if hasattr(q, name):
                m = getattr(q, name)
                m = m() if callable(m) else m
                break
    m = np.asarray(m)
    off = m - np.diag(np.diag(m))
    nz = np.count_nonzero(np.triu(off + off.T, 1))
    return m.shape[0], nz


rows = []
for tier, seeds in (("T0", range(2026, 2031)), ("T1", range(2026, 2031))):
    for seed in seeds:
        by = {i.dials.label: i for i in tier_instances(tier, seed)}
        for d in DIALS:
            enc = encode(by[d], encoding_named("binary"), n_zeta_bits=STUDY_ZETA_BITS)
            n, nz = couplings(enc)
            rows.append(
                {
                    "tier": tier,
                    "dials": d,
                    "instance_seed": seed,
                    "n_vars": n,
                    "couplings": nz,
                    "density": nz / (n * (n - 1) / 2),
                }
            )
t = pl.DataFrame(rows)
hw = (
    pl.read_csv(D + "m5-p2-qaoa_noise.csv")
    .filter(pl.col("solver") == "qaoa_ibm")
    .select("dials", "instance_seed", "routed_2q")
)
t0 = t.filter(pl.col("tier") == "T0").join(hw, on=["dials", "instance_seed"])
# logical CZ of a reps-2 QAOA: 2 CZ per ZZ term per layer
t0 = t0.with_columns(
    (4 * pl.col("couplings")).alias("logical_cz_r2"),
    (pl.col("routed_2q") / (4 * pl.col("couplings"))).alias("overhead"),
)
pl.Config.set_tbl_rows(50)
print(
    t0.select(
        "dials",
        "instance_seed",
        "n_vars",
        "couplings",
        "density",
        "logical_cz_r2",
        "routed_2q",
        "overhead",
    )
)
print(
    "T0 routing overhead median",
    round(t0["overhead"].median(), 2),
    "range",
    round(t0["overhead"].min(), 2),
    round(t0["overhead"].max(), 2),
)
print("T0 density median", round(t0["density"].median(), 2))
t1 = t.filter(pl.col("tier") == "T1")
print(t1.filter(pl.col("instance_seed") == 2026))
lo, hi = t1["couplings"].min(), t1["couplings"].max()
print("T1 couplings over the five seeds", lo, "to", hi, "; density", round(t1["density"].min(), 2),
      "to", round(t1["density"].max(), 2), "; smallest:",
      t1.sort("couplings").head(1).select("dials", "instance_seed", "n_vars").rows())
o_med = t0["overhead"].median()
for reps in (2, 1):
    cz_lo, cz_hi = 2 * reps * lo * o_med, 2 * reps * hi * o_med
    print(f"T1 routed CZ at reps = {reps}, median T0 overhead: {cz_lo:,.0f} to {cz_hi:,.0f}; "
          f"shots free of a CZ error "
          f"{0.9975 ** cz_lo:.1e} to {0.9975 ** cz_hi:.1e}")
