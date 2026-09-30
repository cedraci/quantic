import dataclasses

import pytest

from quantic.problem.feasibility import classify
from quantic.problem.objective import evaluate
from quantic.solvers.base import (
    INFEASIBLE,
    OPTIMAL,
    TIME_LIMIT,
    Bracket,
    Budget,
    Result,
    Solver,
)
from tests.solvers.instances import plain


def _result(**kw) -> Result:
    inst = plain()
    base = dict(
        solver="s", status=OPTIMAL, schedule=inst.witness, zeta=None,
        objective=evaluate(inst, inst.witness),
        bracket=Bracket(lower=1.0, upper=1.0),
        feasibility=classify(inst, inst.witness),
    )
    base.update(kw)
    return Result(**base)


def test_a_bracket_reports_how_far_apart_its_ends_are():
    assert Bracket(lower=1.0, upper=1.5).width == 0.5


def test_a_bracket_refuses_a_lower_bound_above_the_value_attained():
    with pytest.raises(ValueError, match="one of the two is wrong"):
        Bracket(lower=2.0, upper=1.0)


def test_a_bracket_lets_a_solver_tolerance_through_and_reports_no_negative_width():
    """HiGHS proves its bound to about 1e-7, so the bound can sit a hair above."""
    bracket = Bracket(lower=30.0 + 1e-7, upper=30.0)
    assert bracket.width == 0.0


def test_a_bracket_does_not_let_more_than_a_tolerance_through():
    with pytest.raises(ValueError):
        Bracket(lower=30.0 + 1e-3, upper=30.0)


def test_a_bracket_with_nothing_proven_yet_is_infinitely_wide():
    assert Bracket(lower=float("-inf"), upper=3.0).width == float("inf")


@pytest.mark.parametrize("lower, upper", [(float("nan"), 3.0), (1.0, float("nan"))])
def test_a_bracket_refuses_a_bound_that_is_not_a_number(lower, upper):
    with pytest.raises(ValueError, match="NaN"):
        Bracket(lower=lower, upper=upper)


@pytest.mark.parametrize("field", ["wall_clock_s", "max_samples"])
@pytest.mark.parametrize("value", [0, -1])
def test_a_budget_refuses_a_limit_that_allows_nothing(field, value):
    with pytest.raises(ValueError, match=field):
        Budget(**{field: value})


def test_a_budget_with_no_limits_is_the_default():
    assert Budget() == Budget(wall_clock_s=None, target=None, max_samples=None)


def test_a_result_refuses_a_status_nobody_defined():
    with pytest.raises(ValueError, match="status"):
        _result(status="probably_fine")


@pytest.mark.parametrize("missing", ["schedule", "bracket"])
def test_an_optimal_result_has_to_carry_what_makes_it_optimal(missing):
    with pytest.raises(ValueError, match="optimal"):
        _result(**{missing: None})


@pytest.mark.parametrize("status", [TIME_LIMIT, INFEASIBLE])
def test_a_result_that_is_not_optimal_may_carry_nothing(status):
    result = _result(
        status=status, schedule=None, objective=None, bracket=None, feasibility=None
    )
    assert result.schedule is None


def test_a_result_cannot_be_edited_after_the_fact():
    result = _result(timings={"total_s": 1.0}, detail={"rounds": 3})
    with pytest.raises(TypeError):
        result.timings["total_s"] = 0.0
    with pytest.raises(TypeError):
        result.detail["rounds"] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = TIME_LIMIT


def test_a_result_does_not_share_its_mappings_with_the_caller():
    timings = {"total_s": 1.0}
    result = _result(timings=timings)
    timings["total_s"] = 99.0
    assert result.timings["total_s"] == 1.0


def test_anything_with_a_name_and_a_solve_method_is_a_solver():
    class Stub:
        name = "stub"

        def solve(self, instance, budget):
            return _result()

    assert isinstance(Stub(), Solver)
    assert not isinstance(object(), Solver)
