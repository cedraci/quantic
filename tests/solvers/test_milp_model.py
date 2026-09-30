import numpy as np
import pytest

from quantic.problem.objective import evaluate
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.milp_model import (
    cell_costs,
    columns_of,
    coupling_rows,
    loss_model,
    permanent_constant,
    variance_gradient,
)
from tests.solvers.instances import (
    EVERY_SOLVABLE,
    concave,
    cvar,
    discrete,
    plain,
)

REL = 1e-9


def _zeta(inst):
    return 0.0 if isinstance(inst.risk, CVaRRisk) else None


def _selected(columns, schedule):
    """The 0/1 vector that picks ``schedule``'s value in every cell."""
    y = np.zeros(columns.n)
    for c in range(columns.n):
        i, t = int(columns.asset[c]), int(columns.bucket[c])
        y[c] = float(schedule.lots[i][t] == int(columns.lots[c]))
    return y


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_every_cell_has_one_column_per_value_it_may_take(build):
    grid = lot_grid(build())
    columns = columns_of(grid)
    for i in range(grid.n_assets):
        for t in range(grid.n_buckets):
            mine = (columns.asset == i) & (columns.bucket == t)
            assert tuple(columns.lots[mine]) == grid.values[i][t]
            assert columns.count[i, t] == len(grid.values[i][t])
            assert tuple(
                columns.lots[columns.first[i, t]:columns.first[i, t] + columns.count[i, t]]
            ) == grid.values[i][t]


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_columns_read_back_the_schedule_that_selected_them(build):
    inst = build()
    columns = columns_of(lot_grid(inst))
    for schedule in lot_grid(inst).schedules():
        assert columns.schedule(_selected(columns, schedule)) == schedule


def test_a_solution_a_hair_off_integer_still_reads_back():
    """HiGHS returns 0.9999999 and 1e-9, not 1 and 0."""
    inst = plain()
    columns = columns_of(lot_grid(inst))
    y = _selected(columns, inst.witness)
    noisy = np.where(y > 0.5, y - 1e-7, y + 1e-9)
    assert columns.schedule(noisy) == inst.witness


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_costs_add_up_to_what_the_evaluator_charges(build):
    inst = build()
    columns = columns_of(lot_grid(inst))
    costs = cell_costs(inst, columns)
    for schedule in lot_grid(inst).schedules():
        scored = evaluate(inst, schedule, zeta=_zeta(inst))
        assert costs @ _selected(columns, schedule) == pytest.approx(
            scored.spread_cost + scored.temporary_impact_cost, rel=REL
        )


def test_a_cost_table_replaces_the_temporary_impact_in_the_columns():
    inst = concave()
    columns = columns_of(lot_grid(inst))
    plain_costs = cell_costs(inst, columns)
    free = cell_costs(inst, columns, cell_cost=lambda i, t, lots: 0.0)
    a = inst.params.assets
    spread = np.array(
        [
            a[i].half_spread * lots * inst.assets[i].lot_size
            for i, lots in zip(columns.asset, columns.lots, strict=True)
        ]
    )
    assert free == pytest.approx(spread, rel=REL)
    assert np.all(plain_costs >= free)


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_permanent_impact_is_the_same_at_every_schedule_that_liquidates(build):
    inst = build()
    constant = permanent_constant(inst)
    for schedule in lot_grid(inst).schedules():
        assert evaluate(
            inst, schedule, zeta=_zeta(inst)
        ).permanent_impact_cost == pytest.approx(constant, rel=REL)


def test_permanent_impact_is_not_constant_once_liquidation_is_partial():
    """The identity is about full liquidation; the constant must not be used without it."""
    inst = plain()
    short = Schedule(lots=((2, 1, 0), (1, 1, 1)))
    assert evaluate(inst, short).permanent_impact_cost != pytest.approx(
        permanent_constant(inst), rel=1e-3
    )


def test_the_loss_model_is_the_scenario_sets():
    """At the witness, at the schedule that sells nothing, and everywhere between.

    A loss is a coefficient times a holding, so a feasible schedule alone is
    blind to every cell it no longer holds.
    """
    inst = cvar()
    base, tail = loss_model(inst)
    unsold = Schedule.from_array(np.zeros((inst.n_assets, inst.n_buckets), dtype=int))
    for schedule in (inst.witness, unsold, *lot_grid(inst).schedules()):
        x = schedule.as_array().astype(float)
        mine = base - np.einsum("snt,nt->s", tail, x)
        theirs = inst.risk.scenarios.losses(inst, schedule)
        assert mine == pytest.approx(theirs, rel=REL, abs=REL * np.abs(theirs).max())


def test_the_gradient_is_the_slope_of_the_evaluators_variance():
    """A quadratic's central difference is exact, so this is an equality."""
    inst = plain()
    at = Schedule(lots=((2, 1, 1), (1, 1, 1)))
    gradient = variance_gradient(inst, at)
    for i in range(inst.n_assets):
        for t in range(inst.n_buckets):
            up = at.as_array().copy()
            down = at.as_array().copy()
            up[i, t] += 1
            down[i, t] -= 1
            slope = (
                evaluate(inst, Schedule.from_array(up)).risk_cost
                - evaluate(inst, Schedule.from_array(down)).risk_cost
            ) / 2.0
            assert gradient[i, t] == pytest.approx(slope, rel=1e-7)


def test_selling_sooner_never_adds_variance():
    """A lot sold in a bucket is a holding given up from that bucket on.

    The last bucket's entry is zero at a schedule that liquidates: nothing is
    held after it, so there is nothing left to give up.
    """
    gradient = variance_gradient(plain(), plain().witness)
    assert np.all(gradient <= 0.0)
    assert np.all(gradient[:, 0] < gradient[:, 1])
    assert np.all(gradient[:, -1] == 0.0)


def test_the_gradient_at_zero_holdings():
    """The gradient at the all-zero schedule, verified by exact quadratic formula.

    At x=0, where no lots are sold and all holdings equal initial position,
    the variance is quadratic in x. For a quadratic f(x) = c + g'x + 0.5 x'Hx,
    the Hessian can be computed from second differences, then used to verify g.
    """
    inst = plain()
    at = Schedule.from_array(np.zeros((inst.n_assets, inst.n_buckets), dtype=int))
    gradient = variance_gradient(inst, at)
    v_zero = evaluate(inst, at).risk_cost

    h = 1.0
    for i in range(inst.n_assets):
        for t in range(inst.n_buckets):
            # Evaluate at x + h*e_i
            perturbed1 = at.as_array().copy()
            perturbed1[i, t] = int(h)
            v_h = evaluate(inst, Schedule.from_array(perturbed1)).risk_cost

            # Evaluate at x + 2h*e_i
            perturbed2 = at.as_array().copy()
            perturbed2[i, t] = int(2 * h)
            v_2h = evaluate(inst, Schedule.from_array(perturbed2)).risk_cost

            # For a quadratic: H_ii = [v(2h) - 2v(h) + v(0)] / h²
            hessian_diag = (v_2h - 2 * v_h + v_zero) / (h * h)

            # Gradient from exact quadratic formula: g_i = [v(h) - v(0)] / h - (h/2) H_ii
            gradient_from_evaluator = (v_h - v_zero) / h - (h / 2) * hessian_diag

            assert gradient[i, t] == pytest.approx(gradient_from_evaluator, rel=1e-7)


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_rows_hold_at_every_schedule_on_the_grid_and_at_no_other(build):
    """One value a cell, the whole position sold, and cardinality where it binds."""
    inst = build()
    grid = lot_grid(inst)
    columns = columns_of(grid)
    matrix, lower, upper = coupling_rows(columns, grid, columns.n)
    for schedule in grid.schedules():
        held = matrix @ _selected(columns, schedule)
        inside = bool(np.all((held >= lower - 1e-9) & (held <= upper + 1e-9)))
        within = grid.cardinality is None or all(
            schedule.active_names(t) <= grid.cardinality
            for t in range(grid.n_buckets)
        )
        assert inside == within


def test_the_rows_refuse_a_schedule_that_leaves_a_lot_unsold():
    inst = plain()
    grid = lot_grid(inst)
    columns = columns_of(grid)
    matrix, lower, upper = coupling_rows(columns, grid, columns.n)
    short = Schedule(lots=((2, 1, 0), (1, 1, 1)))
    held = matrix @ _selected(columns, short)
    assert not np.all((held >= lower - 1e-9) & (held <= upper + 1e-9))


def _n_rows(grid):
    columns = columns_of(grid)
    return coupling_rows(columns, grid, columns.n)[0].shape[0]


def test_cardinality_adds_one_row_a_bucket():
    with_it = lot_grid(discrete())
    without = lot_grid(plain())
    cells = with_it.n_assets * with_it.n_buckets
    assert _n_rows(without) == cells + without.n_assets
    assert _n_rows(with_it) == cells + with_it.n_assets + with_it.n_buckets
