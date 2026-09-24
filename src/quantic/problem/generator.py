"""Generate instances that are feasible by construction.

Spec section 14 makes M2 done when every dial combination generates a *valid*
instance across every tier. Validity here means something a test can assert: a
feasible schedule provably exists, one is constructed, and it is attached to
the instance as its witness.

That witness is not decoration. It gives M3's decode/repair a known-good
target, M4 a guaranteed-feasible baseline to measure the optimality gap
against, and it means an infeasible result from any solver is unambiguously
the solver's fault rather than an empty feasible set.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np

from quantic.core.types import Asset
from quantic.problem.constraints.base import Constraint
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.liquidation import LADDER, TierSpec
from quantic.problem.objective import evaluate
from quantic.problem.params import MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

# Fraction of the witness's non-zero cells carved into indivisible blocks
# under D4. A quarter leaves enough free cells that the block structure
# constrains without dictating the whole schedule.
_BLOCK_FRACTION = 0.25

_DEFAULT_LOT_SIZE = 100
_DEFAULT_TICK_SIZE = 0.01


class InstanceGenerationError(RuntimeError):
    """Raised when no feasible instance could be built for a tier and dial set."""


def _distribute(total: int, buckets: list[int], max_lots: int, width: int) -> list[int]:
    """Spread ``total`` lots as evenly as possible over ``buckets`` of a row of ``width``."""
    row = [0] * width
    n = len(buckets)
    base, remainder = divmod(total, n)
    for j, t in enumerate(buckets):
        value = base + (1 if j < remainder else 0)
        if value > max_lots:
            raise InstanceGenerationError(
                f"cannot place {total} lots in {n} bucket(s) without exceeding the "
                f"{max_lots}-lot range"
            )
        row[t] = value
    return row


def _spread_schedule(positions: np.ndarray, tier: TierSpec) -> Schedule:
    """Every asset trades in every bucket. Used when D2 is off."""
    all_buckets = list(range(tier.n_buckets))
    return Schedule(
        lots=tuple(
            tuple(
                _distribute(
                    int(x), all_buckets, tier.max_lots_per_bucket, tier.n_buckets
                )
            )
            for x in positions
        )
    )


def _concentrated_schedule(
    positions: np.ndarray, tier: TierSpec, rng: np.random.Generator
) -> Schedule:
    """Each asset trades in a subset of buckets, so cardinality can bind.

    Assets are placed largest-need-first into the least-loaded buckets. The
    assignment exists because ``TierSpec`` caps each asset's bucket count so
    that total demand never exceeds ``n_buckets * active_per_bucket``.
    """
    max_lots = tier.max_lots_per_bucket
    need = [max(1, math.ceil(int(x) / max_lots)) for x in positions]
    load = [0] * tier.n_buckets
    rows: list[tuple[int, ...]] = [()] * tier.n_assets

    # Largest need first: the tightest assets get the pick of the capacity.
    # Ties broken by index so the result depends only on the seed via
    # `positions`, never on dict or set ordering.
    order = sorted(range(tier.n_assets), key=lambda i: (-need[i], i))
    for i in order:
        available = [t for t in range(tier.n_buckets) if load[t] < tier.active_per_bucket]
        if len(available) < need[i]:
            raise InstanceGenerationError(
                f"{tier.name}: asset {i} needs {need[i]} bucket(s) but only "
                f"{len(available)} have capacity below {tier.active_per_bucket} names. "
                "TierSpec.buckets_per_asset_cap should have prevented this"
            )
        chosen = sorted(sorted(available, key=lambda t: (load[t], t))[: need[i]])
        for t in chosen:
            load[t] += 1
        rows[i] = tuple(
            _distribute(int(positions[i]), chosen, max_lots, tier.n_buckets)
        )

    del rng  # placement is deterministic given the positions; kept for symmetry
    return Schedule(lots=tuple(rows))


def _carve_blocks(witness: Schedule, rng: np.random.Generator) -> tuple[Block, ...]:
    """Declare a subset of the witness's non-zero cells indivisible."""
    grid = witness.as_array()
    cells = [
        (i, t)
        for i in range(witness.n_assets)
        for t in range(witness.n_buckets)
        if grid[i, t] > 0
    ]
    if not cells:
        raise InstanceGenerationError("witness has no non-zero cell to carve a block from")
    count = max(1, int(len(cells) * _BLOCK_FRACTION))
    picked = rng.choice(len(cells), size=count, replace=False)
    return tuple(
        Block(asset=cells[int(j)][0], bucket=cells[int(j)][1], lots=int(grid[cells[int(j)]]))
        for j in sorted(int(j) for j in picked)
    )


def _balanced_lam(instance: Instance) -> float:
    """``lam`` making the risk term equal the cost terms at the witness.

    An arbitrary ``lam`` makes one side of the trade-off vanish -- either risk
    swamps execution cost or it is invisible -- and a problem where one side
    is irrelevant is not the problem the crossover study is about. M4 sweeps
    ``lam`` around this point.
    """
    zero = dataclasses.replace(instance, risk=VarianceRisk(lam=0.0))
    costs = evaluate(zero, instance.witness).total
    unit = dataclasses.replace(instance, risk=VarianceRisk(lam=1.0))
    raw_risk = evaluate(unit, instance.witness).risk_cost
    if raw_risk <= 0.0:
        return 0.0
    return costs / raw_risk


def generate(
    tier: TierSpec,
    dials: Dials,
    params: MarketParams,
    *,
    seed: int,
    lam: float | None = None,
) -> Instance:
    """Build one feasible instance for ``tier`` under ``dials``.

    ``lam`` defaults to the value that balances risk against execution cost at
    the witness. ``params`` may describe more assets than the tier needs, in
    which case the leading ones are used.
    """
    if params.n_assets < tier.n_assets:
        raise InstanceGenerationError(
            f"{tier.name} needs {tier.n_assets} assets but MarketParams describes "
            f"{params.n_assets}"
        )
    params = params.subset(tier.n_assets)

    # Seeded from the full configuration so that two tiers, or two dial
    # combinations, never share a position draw at the same seed.
    rng = np.random.default_rng(
        [seed, tier.n_assets, tier.n_buckets, tier.bits, _dial_code(dials)]
    )

    ceiling = tier.max_position_lots(discrete_participation=dials.discrete_participation)
    positions = rng.integers(1, ceiling + 1, size=tier.n_assets)

    if dials.discrete_participation:
        witness = _concentrated_schedule(positions, tier, rng)
    else:
        witness = _spread_schedule(positions, tier)

    constraints: list[Constraint] = [FullLiquidation()]

    if dials.discrete_participation:
        # Tightest k the witness satisfies. A looser one would make D2 a
        # no-op dial that still shows up in every results row.
        busiest = max(witness.active_names(t) for t in range(tier.n_buckets))
        constraints.append(Cardinality(k=busiest))

        grid = witness.as_array()
        min_lots = []
        for i in range(tier.n_assets):
            nonzero = grid[i][grid[i] > 0]
            min_lots.append(int(nonzero.min()) if nonzero.size else 1)
        constraints.append(MinParticipation(min_lots=tuple(min_lots)))

    if dials.block_trades:
        constraints.append(BlockTrades(blocks=_carve_blocks(witness, rng)))

    assets = tuple(
        Asset(symbol=a.symbol, lot_size=_DEFAULT_LOT_SIZE, tick_size=_DEFAULT_TICK_SIZE)
        for a in params.assets
    )

    instance = Instance(
        instance_id=f"{tier.name}-{dials.label}-{seed}",
        tier=tier.name,
        assets=assets,
        n_buckets=tier.n_buckets,
        initial_lots=tuple(int(x) for x in positions),
        max_lots_per_bucket=tier.max_lots_per_bucket,
        params=params,
        dials=dials,
        constraints=tuple(constraints),
        risk=VarianceRisk(lam=0.0),
        witness=witness,
        seed=seed,
    )

    chosen_lam = _balanced_lam(instance) if lam is None else lam
    instance = dataclasses.replace(instance, risk=VarianceRisk(lam=chosen_lam))

    report = classify(instance, witness)
    if not report.feasible:
        raise InstanceGenerationError(
            f"{tier.name}/{dials.label}: the constructed witness is infeasible, which "
            f"means the construction is wrong rather than the instance hard. "
            f"Violations: {dict(report.violations)}"
        )
    return instance


def _dial_code(dials: Dials) -> int:
    """A small integer identifying the dial combination, for seeding."""
    return (
        int(dials.concave_impact)
        | int(dials.discrete_participation) << 1
        | int(dials.cvar_risk) << 2
        | int(dials.block_trades) << 3
    )


def generate_ladder(
    params_for: Callable[[TierSpec], MarketParams],
    *,
    seed: int,
    include_cvar: bool = False,
) -> tuple[Instance, ...]:
    """Every tier crossed with every dial combination.

    ``include_cvar=False`` gives M2a's 32 instances. Once M2b lands, flipping
    it to ``True`` gives all 64 with no other change -- which is the whole
    point of ``RiskSpec`` being a variant point.

    ``params_for`` is a callable rather than a single ``MarketParams`` because
    each tier needs a different asset count, from 3 at T0 to 30 at T3.
    """
    out: list[Instance] = []
    for tier in LADDER:
        params = params_for(tier)
        for dials in Dials.combinations(include_cvar=include_cvar):
            out.append(generate(tier, dials, params, seed=seed))
    return tuple(out)
