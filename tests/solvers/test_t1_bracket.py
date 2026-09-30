"""The exact baseline at T1 -- recorded, not gated.

Parent spec section 5.5 gives T1 an exact MILP bracket as its ground truth,
and M4's optimality gaps will be stated against it. Nothing can check the
program's answer here the way enumeration does at T0: the smallest T1 grid
holds ten million schedules and the largest 2e27. What is asserted is what
can be: the schedule is feasible, the upper end is the evaluator's, and the
lower end is below it.
"""

import pytest

from quantic.problem.objective import evaluate, optimal_zeta
from quantic.problem.risk import CVaRRisk
from quantic.solvers.base import OPTIMAL, TIME_LIMIT, Budget
from quantic.solvers.classical.exact_pwl_milp import exact_pwl_milp
from tests.ladder import tier

BUDGET_S = 120.0


@pytest.fixture(scope="module")
def results():
    return {
        instance.instance_id: (
            instance, exact_pwl_milp(instance, budget=Budget(wall_clock_s=BUDGET_S))
        )
        for instance in tier("T1")
    }


def test_the_tier_is_sixteen_combinations(results):
    assert len(results) == 16


def test_every_instance_ends_with_a_verdict(results):
    verdicts = {name: result.status for name, (_, result) in results.items()}
    assert set(verdicts.values()) <= {OPTIMAL, TIME_LIMIT}, verdicts


def test_every_schedule_returned_is_feasible(results):
    for name, (_, result) in results.items():
        assert result.schedule is not None, name
        assert result.feasibility.feasible, (name, dict(result.feasibility.violations))


def test_every_upper_bound_is_the_evaluators(results):
    for name, (instance, result) in results.items():
        zeta = (
            optimal_zeta(instance, result.schedule)
            if isinstance(instance.risk, CVaRRisk) else None
        )
        assert result.bracket.upper == evaluate(
            instance, result.schedule, zeta=zeta
        ).total, name


def test_no_lower_bound_is_above_its_upper(results):
    for name, (_, result) in results.items():
        bracket = result.bracket
        assert bracket.lower <= bracket.upper + 1e-6 * max(1.0, abs(bracket.upper)), name


def test_no_schedule_beats_the_witness_by_less_than_nothing(results):
    """The witness is feasible, so the optimum cannot cost more than it does."""
    for name, (instance, result) in results.items():
        if result.status != OPTIMAL:
            continue
        zeta = (
            optimal_zeta(instance, instance.witness)
            if isinstance(instance.risk, CVaRRisk) else None
        )
        at_witness = evaluate(instance, instance.witness, zeta=zeta).total
        assert result.bracket.upper <= at_witness * (1.0 + 1e-9), name
