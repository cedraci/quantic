"""The ladder both gates are run on.

M3a's gate and M3b's build the same sixty-four instances, from the same
parameters and the same seed. They are built here so that "the same" is a
fact about one function and not about two copies of it.
"""

from __future__ import annotations

import functools

import numpy as np

from quantic.problem.generator import generate_ladder
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams

BUCKET_NS = 1800 * 1_000_000_000
SEED = 2026
SMALL = {"n_raw": 300, "n_target": 15}


def params_for(tier) -> MarketParams:
    rng = np.random.default_rng(tier.n_assets)
    a = rng.normal(size=(tier.n_assets, tier.n_assets))
    return MarketParams(
        assets=tuple(
            AssetParams(f"S{i:02d}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + i, 0.01)
            for i in range(tier.n_assets)
        ),
        covariance=(a @ a.T) / tier.n_assets * 1e-5,
        bucket_ns=BUCKET_NS,
    )


@functools.cache
def gate_ladder() -> tuple[Instance, ...]:
    """All sixty-four instances. Built once per process: generation is pure."""
    return generate_ladder(params_for, seed=SEED, include_cvar=True, **SMALL)


def tier(name: str) -> tuple[Instance, ...]:
    return tuple(i for i in gate_ladder() if i.tier == name)
