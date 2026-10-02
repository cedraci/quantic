"""The instances the bench runs on.

The same generator, parameters and scenario sizes as the M3 gates' ladder
(``tests/ladder.py`` imports from here), so that seed ``GATE_SEED`` gives
exactly the instances M3a and M3b were proven on. Other seeds give new draws
of the same ladder.

The parameters are synthetic. Calibrating them from a real bundle is
``problem.from_bundle``'s job, and M4 does not do it: the study's question is
how solvers behave across the hardness dials, which the synthetic ladder spans.
"""

from __future__ import annotations

import numpy as np

from quantic.problem.dials import Dials
from quantic.problem.generator import generate
from quantic.problem.instance import Instance
from quantic.problem.liquidation import TierSpec, by_name
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.scenarios import generate_scenarios

GATE_SEED = 2026
BUCKET_NS = 1800 * 1_000_000_000
N_RAW = 300
N_TARGET = 15


def synthetic_params(tier: TierSpec) -> MarketParams:
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


def tier_instances(tier_name: str, seed: int) -> tuple[Instance, ...]:
    """All sixteen dial combinations of one tier at one seed.

    One raw scenario set per tier and seed, shared by the CVaR instances, as
    ``generate_ladder`` does.
    """
    tier = by_name(tier_name)
    params = synthetic_params(tier)
    raw = generate_scenarios(params, tier.n_buckets, seed=seed, n_raw=N_RAW)
    return tuple(
        generate(
            tier, dials, params, seed=seed, n_raw=N_RAW, n_target=N_TARGET,
            scenarios=raw if dials.cvar_risk else None,
        )
        for dials in Dials.combinations(include_cvar=True)
    )


class InstanceCache:
    """One worker's instances, built once per (tier, seed)."""

    def __init__(self) -> None:
        self._tiers: dict[tuple[str, int], dict[str, Instance]] = {}

    def get(self, tier: str, dials: str, seed: int) -> Instance:
        key = (tier, seed)
        if key not in self._tiers:
            self._tiers[key] = {i.dials.label: i for i in tier_instances(tier, seed)}
        try:
            return self._tiers[key][dials]
        except KeyError:
            raise KeyError(
                f"no dial combination {dials!r}; known: {sorted(self._tiers[key])}"
            ) from None
