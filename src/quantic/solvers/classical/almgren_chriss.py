"""What a desk does today -- parent spec section 7.2's reference.

The Almgren-Chriss trajectory solves the continuous problem one name at a
time: linear temporary impact, variance risk on that name's own holdings, no
lots, no venue minimum, no cardinality, no blocks. It is then rounded onto the
lot grid. Any result the study calls interesting has to beat this.

Holdings after bucket ``t`` follow ``X sinh(kappa (T - t)) / sinh(kappa T)``,
with ``cosh kappa = 1 + lam sigma**2 / (2 k)``: ``k`` is the temporary cost per
share squared of the linear arm, ``sigma**2`` the name's own price variance per
bucket. The formula knows variance only. Under CVaR it is given no risk
aversion and the trajectory is a straight line, which is what a desk without a
CVaR model would trade.

Rounding takes, per name, the admissible row nearest the trajectory in the sum
of absolute lot differences -- a small dynamic programme over the lot grid.
Rows are chosen name by name, so a cardinality limit can be breached; the
result then says so through its feasibility report. Nothing here is a proof,
so the status is always ``HEURISTIC``.
"""

from __future__ import annotations

import math
import time

import numpy as np

from quantic.micro.impact.almgren_chriss import AlmgrenChriss as LinearImpact
from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.base import HEURISTIC, Budget, Result
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.scoring import score

NAME = "almgren_chriss"


def holdings_fraction(n_buckets: int, kappa: float) -> np.ndarray:
    """``h_t / X`` for ``t = 0 .. T``: one at the start, zero at the end."""
    t = np.arange(n_buckets + 1, dtype=float)
    if kappa == 0.0:
        return (n_buckets - t) / n_buckets
    return np.sinh(kappa * (n_buckets - t)) / math.sinh(kappa * n_buckets)


def kappa_for(instance: Instance, i: int) -> float:
    """The urgency of name ``i``'s trajectory. Zero without variance risk."""
    risk = instance.risk
    if not isinstance(risk, VarianceRisk) or risk.lam == 0.0:
        return 0.0
    a = instance.params.assets[i]
    k = LinearImpact(eta=a.y_coef, gamma=a.gamma).temporary_cost(
        1.0, instance.params.impact_params(i)
    ) * a.price
    sigma2 = float(np.asarray(instance.params.price_covariance())[i, i])
    return math.acosh(1.0 + float(risk.lam) * sigma2 / (2.0 * k))


def nearest_row(
    values: tuple[tuple[int, ...], ...], position: int, target: np.ndarray
) -> tuple[int, ...] | None:
    """The row of admissible values summing to ``position`` nearest ``target``.

    Nearest in ``sum |v_t - target_t|``; ties go to the row that is
    lexicographically smallest, so the answer does not depend on float order.
    ``None`` when no row reaches the position.
    """
    n = len(values)
    # best[t][r]: (cost, row) for buckets t.. selling exactly r lots.
    best: list[dict[int, tuple[float, tuple[int, ...]]]] = [{} for _ in range(n + 1)]
    best[n][0] = (0.0, ())
    for t in range(n - 1, -1, -1):
        for r in range(position + 1):
            options = []
            for v in values[t]:
                if v <= r and (r - v) in best[t + 1]:
                    cost, rest = best[t + 1][r - v]
                    options.append((cost + abs(v - float(target[t])), (v, *rest)))
            if options:
                best[t][r] = min(options)
    found = best[0].get(position)
    return None if found is None else found[1]


def almgren_chriss(instance: Instance) -> Result:
    """The rounded Almgren-Chriss schedule, scored by the evaluator."""
    started = time.perf_counter()
    grid = lot_grid(instance)
    rows = []
    kappas = []
    for i in range(instance.n_assets):
        kappa = kappa_for(instance, i)
        kappas.append(kappa)
        held = holdings_fraction(instance.n_buckets, kappa) * instance.initial_lots[i]
        row = nearest_row(grid.values[i], grid.initial_lots[i], -np.diff(held))
        if row is None:
            return Result(
                solver=NAME, status=HEURISTIC, schedule=None, zeta=None,
                objective=None, bracket=None, feasibility=None,
                timings={"total_s": time.perf_counter() - started},
                detail={"kappa": kappas, "unreachable_asset": i},
            )
        rows.append(row)

    schedule = Schedule(lots=tuple(rows))
    breakdown, zeta = score(instance, schedule)
    return Result(
        solver=NAME, status=HEURISTIC, schedule=schedule, zeta=zeta,
        objective=breakdown, bracket=None,
        feasibility=classify(instance, schedule),
        timings={"total_s": time.perf_counter() - started},
        detail={"kappa": kappas},
    )


class AlmgrenChrissSolver:
    """``almgren_chriss`` behind the ``Solver`` protocol. The budget is not used."""

    name = NAME

    def solve(self, instance: Instance, budget: Budget) -> Result:
        return almgren_chriss(instance)
