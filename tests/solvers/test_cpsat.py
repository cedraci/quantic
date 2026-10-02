import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantic.problem.objective import evaluate, optimal_zeta
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.solvers.base import INFEASIBLE, OPTIMAL, TIME_LIMIT, Budget, Solver
from quantic.solvers.classical import cpsat as module
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.cpsat import CpSat, cpsat
from tests.solvers.instances import (
    EVERY_SOLVABLE,
    cvar,
    discrete,
    infeasible,
    plain,
)

REL = 1e-9


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_it_finds_what_enumeration_finds(build):
    inst = build()
    found = cpsat(inst)
    assert found.status == OPTIMAL
    assert found.feasibility.feasible
    assert found.objective.total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_bound_is_below_the_optimum_and_the_top_is_the_evaluators(build):
    inst = build()
    found = cpsat(inst)
    optimum = brute_force(inst).objective.total
    zeta = optimal_zeta(inst, found.schedule) if isinstance(inst.risk, CVaRRisk) else None
    assert found.bracket.upper == evaluate(inst, found.schedule, zeta=zeta).total
    assert found.bracket.lower <= optimum + REL * abs(optimum)


def test_an_instance_nobody_can_solve_is_reported_and_not_raised():
    found = cpsat(infeasible())
    assert found.status == INFEASIBLE
    assert found.schedule is None


@settings(max_examples=40, deadline=None)
@given(
    lam=st.floats(min_value=0.0, max_value=5e-3),
    use_cvar=st.booleans(),
    use_discrete=st.booleans(),
)
def test_the_scaled_bound_never_exceeds_the_true_optimum(lam, use_cvar, use_discrete):
    """The rounding to integers keeps a relaxation, so the bound stays a bound."""
    inst = discrete() if use_discrete else plain()
    if use_cvar:
        inst = dataclasses.replace(inst, risk=dataclasses.replace(cvar().risk, lam=lam))
    else:
        inst = dataclasses.replace(inst, risk=VarianceRisk(lam=lam))
    found = cpsat(inst)
    optimum = brute_force(inst).objective.total
    assert found.bracket.lower <= optimum + REL * max(1.0, abs(optimum))


def test_running_out_of_time_keeps_the_best_schedule_and_an_honest_bracket(monkeypatch):
    clock = iter([0.0, 0.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock, 100.0))
    inst = plain()
    found = cpsat(inst, budget=Budget(wall_clock_s=60.0))
    assert found.status == TIME_LIMIT
    assert found.feasibility.feasible
    assert found.bracket.lower <= brute_force(inst).objective.total
    assert found.detail["rounds"] == 1


def test_the_same_seed_gives_the_same_answer():
    first = cpsat(discrete(), seed=3)
    again = cpsat(discrete(), seed=3)
    assert first.schedule == again.schedule
    assert first.bracket == again.bracket


def test_it_is_a_solver():
    assert isinstance(CpSat(seed=1), Solver)
    assert CpSat(seed=1).solve(plain(), Budget()).status == OPTIMAL


def test_a_program_too_wide_for_64_bit_integers_is_refused_by_type(monkeypatch):
    monkeypatch.setattr(module, "_ROW_LIMIT", 1.0)
    with pytest.raises(module.ScaleOverflow, match=r"2\*\*60"):
        cpsat(cvar())
