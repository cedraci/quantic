"""The exact optimum through OR-Tools CP-SAT -- parent spec section 7.2's opponent.

CP-SAT is the third exact method. It shares the lot grid and the program's
pieces with ``exact_pwl_milp`` -- one binary per admissible lot value of each
cell, the cell costs read off the evaluator -- and does not import that module
or ``brute_force``: a bracket that agrees with the MILP's is evidence only if
nothing but the shared pieces is shared.

CP-SAT takes integers only. Every real number is turned into one by rounding in
the direction that keeps the integer program a **relaxation**: a cost
coefficient is floored, the right-hand side of a ``>=`` constraint is floored,
a coefficient a variable is subtracted with is ceiled. Every feasible schedule
then has an integer point costing at most its true objective, in ticks, plus
the ceiling of each auxiliary -- one tick per auxiliary, ``slack`` below. So

    optimum >= (CP-SAT's bound - slack) / ticks_per_unit + constant

and ``Bracket.lower`` is that. ``Bracket.upper`` is the evaluator at the
schedule, as for every solver.

Under CVaR the weights ``w_s / (1 - alpha)`` are real and multiply ``zeta`` in
every scenario's row. They are moved into the rows, scaled by the largest power of two that fits
and ceiled there, so the objective's coefficients stay exactly one.
"""

from __future__ import annotations

import math
import time

import numpy as np
from ortools.sat.python import cp_model

from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.base import (
    _BRACKET_RTOL,
    HEURISTIC,
    INFEASIBLE,
    OPTIMAL,
    TIME_LIMIT,
    Bracket,
    Budget,
    Result,
)
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.milp_model import (
    cell_costs,
    columns_of,
    loss_model,
    permanent_constant,
    variance_gradient,
)
from quantic.solvers.classical.scoring import score

NAME = "cpsat"

# Ticks per unit of the objective's scale. One tick of slack per auxiliary is
# then about 1e-9 of the objective.
_TICKS = 2.0**30
# Each CVaR row is scaled by the largest power of two, at most 2**_ROW_BITS,
# that keeps it under _ROW_LIMIT; the weight on zeta is resolved to that.
_ROW_BITS = 40
_ROW_LIMIT = 2.0**60
# CP-SAT works in int64; stay well inside it.
_INT_LIMIT = 2**62
_CUT_RTOL = 1e-9
_MAX_ROUNDS = 10_000


class ScaleOverflow(ValueError):
    """The scaled program would not fit in CP-SAT's 64-bit integers."""


def _check(values, what: str) -> None:
    if any(abs(int(v)) >= _INT_LIMIT for v in values):
        raise ScaleOverflow(f"{what} exceeds 2**62 once scaled")


def cpsat(instance: Instance, *, budget: Budget | None = None, seed: int = 0) -> Result:
    """Minimise the objective over the lot grid with CP-SAT, with a proven bound."""
    started = time.perf_counter()
    limit = None if budget is None else budget.wall_clock_s
    risk = instance.risk
    if not isinstance(risk, (VarianceRisk, CVaRRisk)):
        raise NotImplementedError(
            f"cpsat does not know risk spec {risk.name!r} ({type(risk).__name__}). "
            "Dispatch is on type, never on a lam attribute"
        )

    grid = lot_grid(instance)
    columns = columns_of(grid)
    n_y = columns.n
    lam = float(risk.lam)
    cvar = isinstance(risk, CVaRRisk) and lam > 0.0
    variance = isinstance(risk, VarianceRisk) and lam > 0.0

    reference, _ = score(instance, instance.witness)
    ticks = _TICKS / max(1.0, abs(reference.total))
    constant = permanent_constant(instance)

    model = cp_model.CpModel()
    y = [model.NewBoolVar(f"y{c}") for c in range(n_y)]
    _add_coupling(model, y, columns, grid)

    cost = [math.floor(ticks * float(v)) for v in cell_costs(instance, columns)]
    _check(cost, "a cell cost")
    objective = [cost[c] * y[c] for c in range(n_y)]
    slack = 0

    theta = None
    if variance:
        top = _variance_ceiling(instance)
        theta = model.NewIntVar(0, math.ceil(ticks * top) + 1, "theta")
        objective.append(theta)
        slack = 1
    if cvar:
        constant += _add_cvar(model, instance, columns, y, ticks, objective)
        slack = 1 + risk.scenarios.n_scenarios

    model.Minimize(sum(objective))
    for c, value in _hint(instance, columns).items():
        model.AddHint(y[c], value)

    best: tuple[float, Schedule] | None = None
    proven = -math.inf
    rounds = 0
    n_cuts = 0
    cut_at: set[Schedule] = set()
    status = OPTIMAL
    if variance:
        _add_cut(model, instance, columns, y, theta, ticks, instance.witness)
        cut_at.add(instance.witness)
        n_cuts += 1

    while True:
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1
        solver.parameters.random_seed = int(seed)
        # A Ctrl-C ends the run, never a solve that then reports TIME_LIMIT.
        solver.parameters.catch_sigint_signal = False
        if limit is not None:
            left = limit - (time.perf_counter() - started)
            if left <= 0.0:
                status = TIME_LIMIT
                break
            solver.parameters.max_time_in_seconds = left
        rounds += 1
        outcome = solver.Solve(model)

        if outcome == cp_model.MODEL_INVALID:
            raise RuntimeError(
                f"CP-SAT rejected the model for {instance.instance_id}: "
                f"{model.Validate()}"
            )
        if outcome == cp_model.INFEASIBLE:
            if best is not None or rounds > 1:
                raise RuntimeError(
                    f"CP-SAT reported {instance.instance_id} infeasible in round "
                    f"{rounds} after a feasible round; cuts never remove a schedule"
                )
            return Result(
                solver=NAME, status=INFEASIBLE, schedule=None, zeta=None,
                objective=None, bracket=None, feasibility=None,
                timings={"total_s": time.perf_counter() - started},
                detail={"rounds": rounds, "n_cuts": n_cuts, "n_binaries": n_y},
            )

        bound = (solver.BestObjectiveBound() - slack) / ticks + constant
        proven = max(proven, bound)

        if outcome in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            chosen = np.array([solver.Value(v) for v in y], dtype=float)
            schedule = columns.schedule(chosen)
            breakdown, _ = score(instance, schedule)
            if best is None or breakdown.total < best[0]:
                best = (breakdown.total, schedule)

        if outcome != cp_model.OPTIMAL:
            status = TIME_LIMIT
            break
        if not variance:
            break

        reported = solver.Value(theta) / ticks
        gap = breakdown.risk_cost - reported
        if gap <= _CUT_RTOL * max(1.0, abs(breakdown.total)) or schedule in cut_at:
            break
        if rounds >= _MAX_ROUNDS:
            raise RuntimeError(
                f"outer approximation on {instance.instance_id} has not converged "
                f"in {_MAX_ROUNDS} rounds; the last gap was {gap!r}"
            )
        _add_cut(model, instance, columns, y, theta, ticks, schedule)
        model.ClearHints()
        for c, value in _hint_from(columns, schedule).items():
            model.AddHint(y[c], value)
        cut_at.add(schedule)
        n_cuts += 1

    timings = {"total_s": time.perf_counter() - started}
    detail = {
        "rounds": rounds, "n_cuts": n_cuts, "n_binaries": n_y,
        "ticks_per_unit": ticks, "slack_ticks": slack,
    }
    if best is None:
        return Result(
            solver=NAME, status=TIME_LIMIT, schedule=None, zeta=None,
            objective=None, bracket=None, feasibility=None,
            timings=timings, detail=detail,
        )

    total, schedule = best
    breakdown, zeta = score(instance, schedule)
    bracket = Bracket(lower=proven, upper=total)
    if status == OPTIMAL and bracket.width > _BRACKET_RTOL * max(1.0, abs(total)):
        # CP-SAT proved its integer program; the rounding that made it integer
        # left the bracket open. That is a schedule with a bound, not a proof.
        status = HEURISTIC
    return Result(
        solver=NAME, status=status, schedule=schedule, zeta=zeta,
        objective=breakdown, bracket=bracket,
        feasibility=classify(instance, schedule),
        timings=timings, detail=detail,
    )


def _add_coupling(model, y, columns, grid) -> None:
    """One value per cell, full liquidation per name, cardinality per bucket."""
    n_assets, n_buckets = columns.first.shape
    every = np.arange(columns.n)
    for i in range(n_assets):
        for t in range(n_buckets):
            mine = every[(columns.asset == i) & (columns.bucket == t)]
            model.AddExactlyOne(y[int(c)] for c in mine)
    for i in range(n_assets):
        mine = every[columns.asset == i]
        model.Add(
            sum(int(columns.lots[c]) * y[int(c)] for c in mine)
            == int(grid.initial_lots[i])
        )
    if grid.cardinality is not None:
        for t in range(n_buckets):
            mine = every[(columns.bucket == t) & (columns.lots > 0)]
            model.Add(sum(y[int(c)] for c in mine) <= int(grid.cardinality))


def _variance_ceiling(instance: Instance) -> float:
    """No schedule's variance term exceeds ``lam * T * lambda_max * |X|**2``."""
    sigma = np.asarray(instance.params.price_covariance(), dtype=float)
    top_eig = float(np.max(np.linalg.eigvalsh(sigma)))
    shares = np.asarray(instance.initial_lots, dtype=float) * instance.lot_sizes()
    return float(instance.risk.lam) * instance.n_buckets * max(top_eig, 0.0) * float(
        shares @ shares
    )


def _add_cut(model, instance, columns, y, theta, ticks, schedule: Schedule) -> None:
    """``theta >= v(x^) + g . (x - x^)``, floored so it stays valid."""
    breakdown, _ = score(instance, schedule)
    gradient = variance_gradient(instance, schedule)
    at = schedule.as_array().astype(float)
    rhs = breakdown.risk_cost - float((gradient * at).sum())
    coef = gradient[columns.asset, columns.bucket] * columns.lots
    rows = [math.floor(ticks * float(a)) for a in coef]
    floor_rhs = math.floor(ticks * rhs)
    _check([*rows, floor_rhs], "a variance cut")
    model.Add(theta >= floor_rhs + sum(rows[c] * y[c] for c in range(columns.n)))


def _add_cvar(model, instance, columns, y, ticks, objective) -> float:
    """Rockafellar-Uryasev in integers. Returns ``lam * ell_lo``, a constant.

    ``zeta = ell_lo + z`` with ``z >= 0``, where ``ell_lo`` is below every loss
    any schedule on the grid can take, so the minimising ``zeta`` (a loss) is
    never cut off. ``w`` carries ``lam * z`` and ``v_s`` carries
    ``lam * p_s * u_s``, both in objective ticks, each with coefficient one.
    """
    risk = instance.risk
    lam = float(risk.lam)
    p = np.asarray(risk.scenarios.weights, dtype=float) / (1.0 - risk.alpha)
    base, tail = loss_model(instance)
    # t[s, c]: what selling column c's lots takes off scenario s's loss.
    t = tail[:, columns.asset, columns.bucket] * columns.lots[None, :]
    n_assets, n_buckets = columns.first.shape
    lowest = base.copy()
    highest = base.copy()
    for i in range(n_assets):
        for tt in range(n_buckets):
            lo = int(columns.first[i, tt])
            hi = lo + int(columns.count[i, tt])
            lowest -= t[:, lo:hi].max(axis=1)
            highest -= t[:, lo:hi].min(axis=1)
    ell_lo = float(lowest.min())
    span = max(float(highest.max()) - ell_lo, 0.0)

    w_top = math.ceil(ticks * lam * span) + 1
    w = model.NewIntVar(0, w_top, "w")
    objective.append(w)
    for s in range(base.size):
        v_top = math.ceil(ticks * lam * p[s] * span) + 1
        v = model.NewIntVar(0, v_top, f"v{s}")
        objective.append(v)
        # K v_s + ceil(K p_s) w >= K ticks lam p_s (L_s - ell_lo), relaxed. The
        # weight on w is resolved to 1/K, so K is as large as the row allows.
        unit = ticks * lam * p[s]
        size = (
            unit * (abs(float(base[s]) - ell_lo) + float(np.abs(t[s]).sum()))
            + (p[s] + 1.0) * w_top + v_top
        )
        k = 2.0 ** min(_ROW_BITS, math.floor(math.log2(_ROW_LIMIT / max(size, 1.0))))
        if k < 1.0:
            raise ScaleOverflow(
                f"a CVaR row of {instance.instance_id} does not fit in 2**60 "
                "even unscaled"
            )
        rhs = math.floor(k * unit * (float(base[s]) - ell_lo))
        rows = [math.ceil(k * unit * float(a)) for a in t[s]]
        w_coef = math.ceil(k * p[s])
        _check([*rows, rhs, w_coef * w_top, int(k) * v_top], "a CVaR row")
        model.Add(
            int(k) * v + w_coef * w
            >= rhs - sum(rows[c] * y[c] for c in range(columns.n))
        )
    return lam * ell_lo


def _hint(instance: Instance, columns) -> dict[int, int]:
    return _hint_from(columns, instance.witness)


def _hint_from(columns, schedule: Schedule) -> dict[int, int]:
    """The binaries that select ``schedule``: CP-SAT starts from a feasible point."""
    lots = schedule.as_array()
    return {
        c: int(lots[int(columns.asset[c]), int(columns.bucket[c])] == int(columns.lots[c]))
        for c in range(columns.n)
    }


class CpSat:
    """``cpsat`` behind the ``Solver`` protocol."""

    name = NAME

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def solve(self, instance: Instance, budget: Budget) -> Result:
        return cpsat(instance, budget=budget, seed=self.seed)
