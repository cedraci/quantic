import types

import pytest

from quantic.encoding.encode import encode
from quantic.encoding.integer import BinaryEncoding
from quantic.problem.objective import evaluate, optimal_zeta
from quantic.problem.risk import CVaRRisk
from quantic.solvers.base import INFEASIBLE, OPTIMAL, TIME_LIMIT, Budget, Solver
from quantic.solvers.classical import exact_pwl_milp as module
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.exact_pwl_milp import ExactPwlMilp, exact_pwl_milp
from tests.solvers.instances import (
    EVERY_SOLVABLE,
    concave,
    cvar,
    cvar_without_risk_aversion,
    discrete,
    infeasible,
    no_risk_aversion,
    plain,
)

REL = 1e-9


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_it_finds_what_enumeration_finds(build):
    inst = build()
    by_milp = exact_pwl_milp(inst)
    by_hand = brute_force(inst)
    assert by_milp.status == OPTIMAL
    assert by_milp.feasibility.feasible
    assert by_milp.objective.total == pytest.approx(by_hand.objective.total, rel=REL)


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_bracket_closes_on_the_evaluators_value(build):
    inst = build()
    result = exact_pwl_milp(inst)
    zeta = optimal_zeta(inst, result.schedule) if isinstance(inst.risk, CVaRRisk) else None
    assert result.bracket.upper == evaluate(inst, result.schedule, zeta=zeta).total
    assert result.bracket.width <= 1e-6 * max(1.0, abs(result.bracket.upper))
    assert result.zeta == zeta


def test_variance_takes_more_than_one_round_and_cvar_takes_one():
    assert exact_pwl_milp(plain()).detail["rounds"] > 1
    assert exact_pwl_milp(plain()).detail["n_cuts"] == (
        exact_pwl_milp(plain()).detail["rounds"] - 1
    )
    assert exact_pwl_milp(cvar()).detail["rounds"] == 1


@pytest.mark.parametrize("build", [no_risk_aversion, cvar_without_risk_aversion])
def test_with_no_risk_aversion_there_is_no_risk_in_the_program(build):
    result = exact_pwl_milp(build())
    assert result.detail["rounds"] == 1
    assert result.detail["n_cuts"] == 0
    assert result.objective.risk_cost == 0.0


def test_an_instance_nobody_can_solve_is_reported_and_not_raised():
    result = exact_pwl_milp(infeasible())
    assert result.status == INFEASIBLE
    assert result.schedule is None
    assert result.bracket is None


def test_with_the_encoders_table_it_finds_the_optimum_enumeration_finds():
    inst = concave()
    report = encode(inst, BinaryEncoding()).breakdown.approximand
    by_milp = exact_pwl_milp(inst, cell_cost=report.cell_cost)
    by_hand = brute_force(inst, cell_cost=report.cell_cost)
    assert by_milp.objective.total == pytest.approx(by_hand.objective.total, rel=REL)
    assert by_milp.objective.temporary_impact_cost == pytest.approx(
        report.temporary_cost(by_milp.schedule), rel=REL
    )


def test_a_cost_table_that_punishes_a_cell_moves_the_optimum_off_it():
    inst = plain()
    usual = exact_pwl_milp(inst).schedule
    i, t = next(
        (i, t) for i in range(inst.n_assets) for t in range(inst.n_buckets)
        if usual.lots[i][t] > 0
    )
    punished = exact_pwl_milp(
        inst,
        cell_cost=lambda a, b, lots: 1e6 if (a, b) == (i, t) and lots else 0.0,
    ).schedule
    assert punished.lots[i][t] == 0


def test_a_budget_already_spent_returns_without_a_schedule():
    result = exact_pwl_milp(plain(), budget=Budget(wall_clock_s=1e-12))
    assert result.status == TIME_LIMIT
    assert result.schedule is None
    assert result.bracket is None


def test_running_out_of_time_keeps_the_best_schedule_and_an_honest_bracket(monkeypatch):
    """HiGHS stops at its limit with an incumbent and a bound that has not met it."""
    real = module.milp

    def out_of_time(**kwargs):
        solved = real(**kwargs)
        return types.SimpleNamespace(
            status=1, x=solved.x, fun=solved.fun, message="time limit",
            mip_dual_bound=solved.mip_dual_bound - 5.0,
        )

    monkeypatch.setattr(module, "milp", out_of_time)
    inst = plain()
    result = exact_pwl_milp(inst, budget=Budget(wall_clock_s=60.0))
    assert result.status == TIME_LIMIT
    assert result.feasibility.feasible
    assert result.bracket.upper == evaluate(inst, result.schedule).total
    assert result.bracket.lower < brute_force(inst).objective.total
    assert result.bracket.width > 0.0
    assert result.detail["rounds"] == 1


def test_when_presolve_fails_the_program_is_solved_without_it(monkeypatch):
    """HiGHS's presolve gives up on some small programs once they carry cuts."""
    real = module.milp
    seen = []

    def presolve_fails(**kwargs):
        seen.append(dict(kwargs["options"]))
        if kwargs["options"].get("presolve", True):
            return types.SimpleNamespace(
                status=4, x=None, fun=None, message="Solve error", mip_dual_bound=None
            )
        return real(**kwargs)

    monkeypatch.setattr(module, "milp", presolve_fails)
    inst = plain()
    result = exact_pwl_milp(inst)
    assert result.status == OPTIMAL
    assert result.objective.total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )
    assert result.detail["presolve_retries"] == result.detail["rounds"]
    assert {"presolve" in options for options in seen} == {True, False}


def test_a_program_highs_cannot_solve_either_way_is_raised(monkeypatch):
    monkeypatch.setattr(
        module, "milp",
        lambda **kwargs: types.SimpleNamespace(
            status=4, x=None, fun=None, message="Solve error", mip_dual_bound=None
        ),
    )
    with pytest.raises(RuntimeError, match="Solve error"):
        exact_pwl_milp(plain())


def test_infeasible_after_a_feasible_round_is_raised_and_not_reported(monkeypatch):
    """Cuts only bound theta from below, so this is numerical trouble, not the instance."""
    real = module.milp
    calls = []

    def infeasible_second_round(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return real(**kwargs)
        return types.SimpleNamespace(
            status=2, x=None, fun=None, message="infeasible", mip_dual_bound=None
        )

    monkeypatch.setattr(module, "milp", infeasible_second_round)
    with pytest.raises(RuntimeError, match="infeasible"):
        exact_pwl_milp(plain())
    assert len(calls) == 2


def test_a_cut_with_the_wrong_sign_cannot_pass_for_an_answer(monkeypatch):
    """The mutation the gate is asked to catch, at a size that runs in a test.

    An invalid cut either removes the optimum, and the value no longer
    matches enumeration, or leaves a bound above the value attained, and the
    bracket refuses to be built.
    """
    real = module.variance_gradient
    monkeypatch.setattr(module, "variance_gradient", lambda i, s: -real(i, s))
    inst = plain()
    try:
        result = exact_pwl_milp(inst)
    except (ValueError, RuntimeError):
        return
    assert result.status != OPTIMAL or result.objective.total != pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )


def test_cardinality_is_in_the_program():
    """Dropping it returns a schedule that trades two names in a bucket."""
    inst = discrete()
    result = exact_pwl_milp(inst)
    assert all(result.schedule.active_names(t) <= 1 for t in range(inst.n_buckets))


def test_it_answers_to_the_solver_protocol():
    solver = ExactPwlMilp()
    assert isinstance(solver, Solver)
    inst = plain()
    assert solver.solve(inst, Budget()).schedule == exact_pwl_milp(inst).schedule


def test_a_limit_nobody_set_is_raised_and_not_called_a_time_limit(monkeypatch):
    real = module.milp

    def iteration_limit(**kwargs):
        solved = real(**kwargs)
        return types.SimpleNamespace(
            status=1, x=solved.x, fun=solved.fun, message="iteration limit",
            mip_dual_bound=solved.mip_dual_bound,
        )

    monkeypatch.setattr(module, "milp", iteration_limit)
    with pytest.raises(RuntimeError, match="never set"):
        exact_pwl_milp(plain())


def test_a_bound_above_the_schedule_is_raised_not_clamped(monkeypatch):
    """The old min(proven, total) turned a wrong bound into a plausible one."""
    real = module.milp

    def wrong_bound(**kwargs):
        solved = real(**kwargs)
        return types.SimpleNamespace(
            status=1, x=solved.x, fun=solved.fun, message="time limit",
            mip_dual_bound=solved.mip_dual_bound + 1e3,
        )

    monkeypatch.setattr(module, "milp", wrong_bound)
    with pytest.raises(ValueError, match="one of the two is wrong"):
        exact_pwl_milp(plain(), budget=Budget(wall_clock_s=60.0))


def test_the_presolve_retry_gets_the_time_that_is_left(monkeypatch):
    real = module.milp
    limits = []
    clock = iter([0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock, 10.0))

    def presolve_fails(**kwargs):
        limits.append(kwargs["options"].get("time_limit"))
        if kwargs["options"].get("presolve", True):
            return types.SimpleNamespace(
                status=4, x=None, fun=None, message="Solve error", mip_dual_bound=None
            )
        return real(**kwargs)

    monkeypatch.setattr(module, "milp", presolve_fails)
    exact_pwl_milp(plain(), budget=Budget(wall_clock_s=60.0))
    assert limits[:2] == [60.0, 50.0]


def test_a_round_the_budget_never_let_run_is_not_counted(monkeypatch):
    """Variance needs a second round; the clock runs out before it starts."""
    calls = []
    real = module.milp
    clock = iter([0.0, 0.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock, 100.0))

    def counted(**kwargs):
        calls.append(1)
        return real(**kwargs)

    monkeypatch.setattr(module, "milp", counted)
    result = exact_pwl_milp(plain(), budget=Budget(wall_clock_s=60.0))
    assert result.status == TIME_LIMIT
    assert len(calls) == 1
    assert result.detail["rounds"] == 1
