import itertools

import pytest

from quantic.encoding.encode import encode, reference_breakdown
from quantic.encoding.integer import BinaryEncoding
from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.objective import evaluate, optimal_zeta
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.base import INFEASIBLE, OPTIMAL, Budget, Solver
from quantic.solvers.classical.brute_force import BruteForce, brute_force
from quantic.solvers.classical.scoring import score
from tests.solvers.instances import (
    EVERY_SOLVABLE,
    blocks,
    concave,
    cvar,
    discrete,
    infeasible,
    instance,
    plain,
)


def _every_feasible(inst):
    """Every feasible schedule, found without the lot grid."""
    cap = inst.max_lots_per_bucket
    out = []
    for flat in itertools.product(range(cap + 1), repeat=inst.n_assets * inst.n_buckets):
        schedule = Schedule(
            lots=tuple(
                tuple(flat[i * inst.n_buckets:(i + 1) * inst.n_buckets])
                for i in range(inst.n_assets)
            )
        )
        if classify(inst, schedule).feasible:
            out.append(schedule)
    return out


def _true_total(inst, schedule):
    zeta = optimal_zeta(inst, schedule) if isinstance(inst.risk, CVaRRisk) else None
    return evaluate(inst, schedule, zeta=zeta).total


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_no_feasible_schedule_costs_less_than_the_one_returned(build):
    inst = build()
    result = brute_force(inst)
    cheapest = min(_true_total(inst, s) for s in _every_feasible(inst))
    assert result.status == OPTIMAL
    assert result.objective.total == cheapest
    assert result.feasibility.feasible


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_value_reported_is_the_evaluators_at_the_schedule_returned(build):
    inst = build()
    result = brute_force(inst)
    assert result.objective.total == _true_total(inst, result.schedule)
    assert result.bracket.lower == result.bracket.upper == result.objective.total


def test_the_risk_term_moves_the_optimum_on_the_instances_these_tests_lean_on():
    """Otherwise a solver that ignored risk would pass every test above."""
    for build in (plain, cvar):
        inst = build()
        with_risk = brute_force(inst).schedule
        costs_only = min(
            _every_feasible(inst),
            key=lambda s, inst=inst: _true_total(inst, s) - score(inst, s)[0].risk_cost,
        )
        assert with_risk != costs_only, build.__name__


def test_under_cvar_the_zeta_returned_is_the_minimising_one():
    inst = cvar()
    result = brute_force(inst)
    assert result.zeta == optimal_zeta(inst, result.schedule)


def test_without_cvar_there_is_no_zeta():
    assert brute_force(plain()).zeta is None


def test_it_says_how_many_schedules_it_looked_at():
    inst = discrete()
    result = brute_force(inst)
    assert result.detail["n_feasible"] == len(_every_feasible(inst))
    assert result.detail["n_candidates"] >= result.detail["n_feasible"]
    assert result.timings["total_s"] >= 0.0


def test_an_instance_nobody_can_solve_is_reported_and_not_raised():
    result = brute_force(infeasible())
    assert result.status == INFEASIBLE
    assert result.schedule is None
    assert result.bracket is None
    assert result.detail["n_feasible"] == 0


def test_a_grid_too_large_to_enumerate_is_refused_before_it_starts():
    with pytest.raises(ValueError, match="above the limit"):
        brute_force(plain(), max_schedules=10)


def test_a_cost_table_replaces_the_temporary_impact_and_nothing_else():
    inst = concave()
    true, _ = score(inst, inst.witness)
    doubled, _ = score(
        inst, inst.witness, cell_cost=lambda i, t, lots: 2.0 * _cell(inst, i, t, lots)
    )
    assert doubled.temporary_impact_cost == pytest.approx(
        2.0 * true.temporary_impact_cost, rel=1e-12
    )
    assert doubled.spread_cost == true.spread_cost
    assert doubled.permanent_impact_cost == true.permanent_impact_cost
    assert doubled.risk_cost == true.risk_cost


def _cell(inst, i, t, lots):
    grid = [[0] * inst.n_buckets for _ in range(inst.n_assets)]
    grid[i][t] = lots
    return evaluate(
        inst, Schedule(lots=tuple(tuple(r) for r in grid))
    ).temporary_impact_cost


def test_with_the_encoders_table_the_optimum_is_of_the_function_it_declares():
    """M3a's hand-off: the baseline for the log encoding is stated against the
    approximand, because that is what the QUBO minimises."""
    inst = concave()
    enc = encode(inst, BinaryEncoding())
    report = enc.breakdown.approximand
    assert report is not None

    result = brute_force(inst, cell_cost=report.cell_cost)
    declared = min(
        reference_breakdown(enc, s).total for s in _every_feasible(inst)
    )
    assert result.objective.total == pytest.approx(declared, rel=1e-12)
    assert result.objective.temporary_impact_cost == pytest.approx(
        report.temporary_cost(result.schedule), rel=1e-12
    )


def test_the_encoders_table_adds_up_to_its_own_total():
    inst = concave()
    report = encode(inst, BinaryEncoding()).breakdown.approximand
    for schedule in _every_feasible(inst):
        by_cell = sum(
            report.cell_cost(i, t, lots)
            for i, row in enumerate(schedule.lots)
            for t, lots in enumerate(row)
        )
        assert by_cell == pytest.approx(report.temporary_cost(schedule), rel=1e-12)


def test_the_encoders_table_refuses_a_value_the_cell_cannot_take():
    inst = instance(
        dials=Dials(concave_impact=True, block_trades=True),
        constraints=blocks().constraints, witness=blocks().witness,
    )
    report = encode(inst, BinaryEncoding()).breakdown.approximand
    with pytest.raises(ValueError, match="no cost"):
        report.cell_cost(0, 0, 2)          # the block cell trades 0 or 3
    with pytest.raises(KeyError, match="neither"):
        report.cell_cost(5, 5, 1)


def test_it_answers_to_the_solver_protocol():
    solver = BruteForce()
    assert isinstance(solver, Solver)
    inst = plain()
    assert solver.solve(inst, Budget()).schedule == brute_force(inst).schedule
