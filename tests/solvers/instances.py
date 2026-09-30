"""Small instances the solver tests share.

Each is small enough to enumerate by hand-rolled loops inside a test, and each
turns on one thing: a dial, a risk spec, or a way of being infeasible. Risk
aversion is set high enough that the risk term moves the optimum, because a
solver that ignored the risk term would otherwise pass.
"""

from __future__ import annotations

import numpy as np

from quantic.core.types import Asset
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def params() -> MarketParams:
    return MarketParams(
        assets=(
            AssetParams("A", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01),
            AssetParams("B", 0.6, 0.7, 0.1, 0.004, 300_000.0, 101.0, 0.02),
        ),
        covariance=np.array([[2.0e-5, 0.8e-5], [0.8e-5, 1.5e-5]]),
        bucket_ns=BUCKET_NS,
    )


def instance(**kw) -> Instance:
    base = dict(
        instance_id="small", tier="T0",
        assets=tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B")),
        n_buckets=3, initial_lots=(4, 3), max_lots_per_bucket=3, params=params(),
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=2e-4),
        witness=Schedule(lots=((2, 1, 1), (1, 1, 1))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def plain() -> Instance:
    return instance(instance_id="plain")


def concave() -> Instance:
    return instance(instance_id="concave", dials=Dials(concave_impact=True))


def discrete() -> Instance:
    """One name a bucket, and a venue minimum of two lots on the first name."""
    return instance(
        instance_id="discrete",
        dials=Dials(discrete_participation=True),
        constraints=(
            FullLiquidation(), Cardinality(k=1), MinParticipation(min_lots=(2, 1)),
        ),
        witness=Schedule(lots=((2, 2, 0), (0, 0, 3))),
    )


def blocks() -> Instance:
    """The first name's first bucket trades three lots or none."""
    return instance(
        instance_id="blocks",
        dials=Dials(block_trades=True),
        constraints=(
            FullLiquidation(), BlockTrades(blocks=(Block(asset=0, bucket=0, lots=3),)),
        ),
        witness=Schedule(lots=((3, 1, 0), (1, 1, 1))),
    )


def scenarios(n: int = 8, n_buckets: int = 3, seed: int = 11) -> ScenarioSet:
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.5, 1.5, size=n)
    return ScenarioSet(
        returns=rng.standard_t(df=3, size=(n, n_buckets, 2)) * 0.01,
        weights=weights / weights.sum(), seed=seed,
    )


def cvar(lam: float = 0.02) -> Instance:
    """Eight unevenly weighted fat-tailed scenarios, several of them in the tail."""
    return instance(
        instance_id="cvar",
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=lam, alpha=0.7, scenarios=scenarios()),
    )


def cvar_discrete(lam: float = 0.02) -> Instance:
    return instance(
        instance_id="cvar_discrete",
        dials=Dials(cvar_risk=True, discrete_participation=True),
        constraints=(
            FullLiquidation(), Cardinality(k=1), MinParticipation(min_lots=(2, 1)),
        ),
        risk=CVaRRisk(lam=lam, alpha=0.7, scenarios=scenarios()),
        witness=Schedule(lots=((2, 2, 0), (0, 0, 3))),
    )


def no_risk_aversion() -> Instance:
    return instance(instance_id="no_risk_aversion", risk=VarianceRisk(lam=0.0))


def cvar_without_risk_aversion() -> Instance:
    return instance(
        instance_id="cvar_without_risk_aversion",
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=0.0, alpha=0.7, scenarios=scenarios()),
    )


def nothing_to_sell_in_one_name() -> Instance:
    """The second name's position is zero, so its cells can only hold zero."""
    return instance(
        instance_id="nothing_to_sell_in_one_name",
        initial_lots=(4, 0),
        witness=Schedule(lots=((2, 1, 1), (0, 0, 0))),
    )


def infeasible() -> Instance:
    """Two names that each need two buckets, two buckets, one name a bucket.

    ``Instance`` checks the witness's shape and not its feasibility, so an
    instance nobody can solve is constructible -- from a bundle, for one.
    """
    return instance(
        instance_id="infeasible",
        n_buckets=2, initial_lots=(4, 4), max_lots_per_bucket=3,
        dials=Dials(discrete_participation=True),
        constraints=(
            FullLiquidation(), Cardinality(k=1), MinParticipation(min_lots=(1, 1)),
        ),
        witness=Schedule(lots=((3, 1), (1, 3))),
    )


EVERY_SOLVABLE = (
    plain, concave, discrete, blocks, cvar, cvar_discrete,
    no_risk_aversion, cvar_without_risk_aversion, nothing_to_sell_in_one_name,
)
