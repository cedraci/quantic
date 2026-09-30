"""The exact optimum by looking at every feasible schedule.

Nothing here is clever, which is the point. It shares the lot grid and the
evaluator with the MILP and nothing else: no bound, no relaxation, no model of
the objective. When the two agree, the MILP's model is right; when they do
not, this one is the reference.
"""

from __future__ import annotations

import time

from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.objective import ObjectiveBreakdown
from quantic.problem.schedule import Schedule
from quantic.solvers.base import INFEASIBLE, OPTIMAL, Bracket, Budget, Result
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.scoring import CellCost, score

NAME = "brute_force"
MAX_SCHEDULES = 10_000_000


def brute_force(
    instance: Instance,
    *,
    cell_cost: CellCost | None = None,
    max_schedules: int = MAX_SCHEDULES,
) -> Result:
    """Minimise the objective over every feasible schedule on the lot grid."""
    started = time.perf_counter()
    grid = lot_grid(instance)
    n_candidates = grid.n_candidates()
    if n_candidates > max_schedules:
        raise ValueError(
            f"{instance.instance_id} has {n_candidates} schedules on its lot grid, "
            f"above the limit of {max_schedules}. Enumeration is the reference at "
            "T0; above that the MILP carries the proof"
        )

    best: tuple[float, Schedule, ObjectiveBreakdown, float | None] | None = None
    n_feasible = 0
    for schedule in grid.schedules():
        if not classify(instance, schedule).feasible:
            continue
        n_feasible += 1
        breakdown, zeta = score(instance, schedule, cell_cost=cell_cost)
        if best is None or breakdown.total < best[0]:
            best = (breakdown.total, schedule, breakdown, zeta)

    timings = {"total_s": time.perf_counter() - started}
    detail = {"n_candidates": n_candidates, "n_feasible": n_feasible}
    if best is None:
        return Result(
            solver=NAME, status=INFEASIBLE, schedule=None, zeta=None,
            objective=None, bracket=None, feasibility=None,
            timings=timings, detail=detail,
        )
    total, schedule, breakdown, zeta = best
    return Result(
        solver=NAME, status=OPTIMAL, schedule=schedule, zeta=zeta,
        objective=breakdown, bracket=Bracket(lower=total, upper=total),
        feasibility=classify(instance, schedule),
        timings=timings, detail=detail,
    )


class BruteForce:
    """``brute_force`` behind the ``Solver`` protocol. The budget is not used."""

    name = NAME

    def solve(self, instance: Instance, budget: Budget) -> Result:
        return brute_force(instance)
