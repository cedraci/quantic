"""The exact optimum through a mixed-integer linear program -- spec section 7.3.

``milp_model`` builds the program's pieces; this module solves it and says
what was proven.

Three things in the objective are not per-cell. Permanent impact sums to half
the square of the position, so under full liquidation it is a constant. CVaR
is Rockafellar-Uryasev, which is a linear program as written. Variance is
convex quadratic, which HiGHS does not take, and goes through outer
approximation: a tangent cut at each schedule the program returns, until the
program's value for the risk meets the evaluator's. A cut is tight at the
schedule it was taken at, so no schedule is cut twice and the loop ends.

The program's own account of what its schedule costs is never reported.
``Bracket.upper`` is the evaluator's total at the schedule, and
``Bracket.lower`` is the program's dual bound.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.base import INFEASIBLE, OPTIMAL, TIME_LIMIT, Bracket, Budget, Result
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.milp_model import (
    cell_costs,
    columns_of,
    coupling_rows,
    loss_model,
    permanent_constant,
    variance_gradient,
)
from quantic.solvers.classical.scoring import CellCost, score

NAME = "exact_pwl_milp"

# The risk the program reports and the risk the evaluator reports, relative to
# the objective. Below this the outer approximation has converged.
_CUT_RTOL = 1e-9
_MAX_ROUNDS = 10_000

# scipy.optimize.milp's status codes.
_SCIPY_OPTIMAL = 0
_SCIPY_LIMIT = 1
_SCIPY_INFEASIBLE = 2
_SCIPY_ERROR = 4


def exact_pwl_milp(
    instance: Instance,
    *,
    budget: Budget | None = None,
    cell_cost: CellCost | None = None,
) -> Result:
    """Minimise the objective over the lot grid, with a proven lower bound."""
    started = time.perf_counter()
    limit = None if budget is None else budget.wall_clock_s
    risk = instance.risk
    if not isinstance(risk, (VarianceRisk, CVaRRisk)):
        raise NotImplementedError(
            f"exact_pwl_milp does not know risk spec {risk.name!r} "
            f"({type(risk).__name__}). Dispatch is on type, never on a lam attribute"
        )

    grid = lot_grid(instance)
    columns = columns_of(grid)
    n_y = columns.n
    lam = float(risk.lam)
    cvar = isinstance(risk, CVaRRisk) and lam > 0.0
    variance = isinstance(risk, VarianceRisk) and lam > 0.0

    n_scenarios = risk.scenarios.n_scenarios if cvar else 0
    # y, then zeta and one u per scenario, or one theta.
    n_vars = n_y + (1 + n_scenarios if cvar else 0) + (1 if variance else 0)

    cost = np.zeros(n_vars)
    cost[:n_y] = cell_costs(instance, columns, cell_cost=cell_cost)
    constant = permanent_constant(instance)

    lower_bounds = np.zeros(n_vars)
    upper_bounds = np.ones(n_vars)
    upper_bounds[n_y:] = np.inf
    integrality = np.zeros(n_vars)
    integrality[:n_y] = 1

    matrix, row_lower, row_upper = coupling_rows(columns, grid, n_vars)
    blocks = [matrix]
    row_lowers = [row_lower]
    row_uppers = [row_upper]

    if cvar:
        zeta_at = n_y
        lower_bounds[zeta_at] = -np.inf
        weights = np.asarray(risk.scenarios.weights, dtype=float)
        cost[zeta_at] = lam
        cost[zeta_at + 1:] = lam * weights / (1.0 - risk.alpha)
        base, tail = loss_model(instance)
        # u_s + zeta - L_s >= 0, with L_s = base_s - sum x * tail.
        per_column = tail[:, columns.asset, columns.bucket] * columns.lots[None, :]
        excess = np.zeros((n_scenarios, n_vars))
        excess[:, :n_y] = per_column
        excess[:, zeta_at] = 1.0
        excess[np.arange(n_scenarios), zeta_at + 1 + np.arange(n_scenarios)] = 1.0
        blocks.append(sparse.csr_array(excess))
        row_lowers.append(base)
        row_uppers.append(np.full(n_scenarios, np.inf))

    if variance:
        cost[n_y] = 1.0

    best: tuple[float, Schedule] | None = None
    proven = -np.inf
    rounds = 0
    n_cuts = 0
    presolve_retries = 0
    cut_at: set[Schedule] = set()
    status = OPTIMAL

    while True:
        rounds += 1
        options = {"mip_rel_gap": 0.0}
        if limit is not None:
            left = limit - (time.perf_counter() - started)
            if left <= 0.0:
                status = TIME_LIMIT
                break
            options["time_limit"] = left

        program = {
            "c": cost,
            "integrality": integrality,
            "bounds": Bounds(lower_bounds, upper_bounds),
            "constraints": LinearConstraint(
                sparse.vstack(blocks, format="csr"),
                np.concatenate(row_lowers),
                np.concatenate(row_uppers),
            ),
        }
        solved = milp(**program, options=options)
        if solved.status == _SCIPY_ERROR:
            # HiGHS's presolve fails outright on some small programs once they
            # carry cuts -- T0-D1+D2 at the fourth round is one. The program
            # itself is sound: without presolve the same solver solves it.
            presolve_retries += 1
            solved = milp(**program, options={**options, "presolve": False})

        if solved.status == _SCIPY_INFEASIBLE:
            return Result(
                solver=NAME, status=INFEASIBLE, schedule=None, zeta=None,
                objective=None, bracket=None, feasibility=None,
                timings={"total_s": time.perf_counter() - started},
                detail={
                    "rounds": rounds, "n_cuts": n_cuts, "n_binaries": n_y,
                    "presolve_retries": presolve_retries,
                },
            )
        if solved.status not in (_SCIPY_OPTIMAL, _SCIPY_LIMIT):
            raise RuntimeError(
                f"HiGHS stopped on {instance.instance_id} with status "
                f"{solved.status}: {solved.message}"
            )

        bound = getattr(solved, "mip_dual_bound", None)
        if bound is not None and np.isfinite(bound):
            proven = max(proven, float(bound) + constant)

        if solved.x is not None:
            schedule = columns.schedule(solved.x[:n_y])
            breakdown, _ = score(instance, schedule, cell_cost=cell_cost)
            if best is None or breakdown.total < best[0]:
                best = (breakdown.total, schedule)

        if solved.status == _SCIPY_LIMIT:
            status = TIME_LIMIT
            break
        if not variance:
            break

        reported = float(solved.x[n_y])
        gap = breakdown.risk_cost - reported
        if gap <= _CUT_RTOL * max(1.0, abs(breakdown.total)):
            break
        if schedule in cut_at:
            # The cut at this schedule is already in the program and is tight
            # there, so what is left of the gap is HiGHS's own feasibility
            # tolerance. Adding the same cut again would change nothing, and
            # the loop would not end.
            break
        if rounds >= _MAX_ROUNDS:
            raise RuntimeError(
                f"outer approximation on {instance.instance_id} has not converged "
                f"in {_MAX_ROUNDS} rounds; the last gap was {gap!r}"
            )

        # theta >= v(x^) + g . (x - x^), with x linear in the binaries.
        gradient = variance_gradient(instance, schedule)
        at = schedule.as_array().astype(float)
        cut = np.zeros((1, n_vars))
        cut[0, :n_y] = -gradient[columns.asset, columns.bucket] * columns.lots
        cut[0, n_y] = 1.0
        blocks.append(sparse.csr_array(cut))
        row_lowers.append(np.array([breakdown.risk_cost - float((gradient * at).sum())]))
        row_uppers.append(np.array([np.inf]))
        cut_at.add(schedule)
        n_cuts += 1

    timings = {"total_s": time.perf_counter() - started}
    detail = {
        "rounds": rounds, "n_cuts": n_cuts, "n_binaries": n_y,
        "presolve_retries": presolve_retries,
    }
    if best is None:
        return Result(
            solver=NAME, status=TIME_LIMIT, schedule=None, zeta=None,
            objective=None, bracket=None, feasibility=None,
            timings=timings, detail=detail,
        )

    total, schedule = best
    breakdown, zeta = score(instance, schedule, cell_cost=cell_cost)
    return Result(
        solver=NAME, status=status, schedule=schedule, zeta=zeta,
        objective=breakdown,
        bracket=Bracket(lower=min(proven, total) if status == TIME_LIMIT else proven,
                        upper=total),
        feasibility=classify(instance, schedule),
        timings=timings, detail=detail,
    )


class ExactPwlMilp:
    """``exact_pwl_milp`` behind the ``Solver`` protocol."""

    name = NAME

    def solve(self, instance: Instance, budget: Budget) -> Result:
        return exact_pwl_milp(instance, budget=budget)
