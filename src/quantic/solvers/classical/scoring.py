"""What a schedule costs, for every solver in ``classical/``.

``problem.objective.evaluate`` is the only evaluator in the project. This
module is the one place the solvers call it from, so that the question "which
``zeta``, and which temporary impact" has one answer.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from quantic.problem.instance import Instance
from quantic.problem.objective import ObjectiveBreakdown, evaluate, optimal_zeta
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule

# (asset, bucket, lots) -> the temporary impact cost of that one cell.
CellCost = Callable[[int, int, int], float]


def score(
    instance: Instance, schedule: Schedule, *, cell_cost: CellCost | None = None
) -> tuple[ObjectiveBreakdown, float | None]:
    """The objective at ``schedule``, and the ``zeta`` it was scored at.

    Under CVaR the ``zeta`` is the minimising one, so the risk term is the
    true CVaR. With a ``cell_cost`` one field of the evaluator's output is
    replaced, as ``encoding.encode.reference_breakdown`` does, and nothing is
    recomputed.
    """
    zeta = (
        optimal_zeta(instance, schedule)
        if isinstance(instance.risk, CVaRRisk)
        else None
    )
    breakdown = evaluate(instance, schedule, zeta=zeta)
    if cell_cost is not None:
        breakdown = dataclasses.replace(
            breakdown,
            temporary_impact_cost=float(
                sum(
                    cell_cost(i, t, lots)
                    for i, row in enumerate(schedule.lots)
                    for t, lots in enumerate(row)
                )
            ),
        )
    return breakdown, zeta
