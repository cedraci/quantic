import dataclasses

import numpy as np
import pytest

from quantic.micro.impact.almgren_chriss import AlmgrenChriss as LinearImpact
from quantic.problem.risk import VarianceRisk
from quantic.solvers.base import HEURISTIC, Budget, Solver
from quantic.solvers.classical.almgren_chriss import (
    AlmgrenChrissSolver,
    almgren_chriss,
    holdings_fraction,
    kappa_for,
    nearest_row,
)
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.scoring import score
from tests.ladder import tier
from tests.solvers.instances import EVERY_SOLVABLE, cvar, discrete, plain


def test_the_linear_arm_costs_a_constant_times_the_square_of_the_trade():
    inst = plain()
    a = inst.params.assets[0]
    model = LinearImpact(eta=a.y_coef, gamma=a.gamma)
    p = inst.params.impact_params(0)
    cost = [model.temporary_cost(q, p) * q * a.price for q in (100.0, 300.0)]
    assert cost[1] / cost[0] == pytest.approx(9.0, rel=1e-12)


def test_the_trajectory_minimises_the_continuous_problem():
    """Impact k q**2 plus lam sigma**2 h**2 per bucket: a quadratic, solved exactly."""
    inst = plain()
    a = inst.params.assets[0]
    k = LinearImpact(eta=a.y_coef, gamma=a.gamma).temporary_cost(
        1.0, inst.params.impact_params(0)
    ) * a.price
    sigma2 = float(np.asarray(inst.params.price_covariance())[0, 0])
    # Chosen so that kappa is 0.5: urgent enough to bend the line, not to empty it.
    lam = 2.0 * k * (np.cosh(0.5) - 1.0) / sigma2
    inst = dataclasses.replace(inst, risk=VarianceRisk(lam=lam))
    n, position = inst.n_buckets, 400.0

    # d/dh_t of sum k (h_{t-1} - h_t)**2 + lam sigma2 h_t**2 = 0 for t = 1 .. n-1.
    m = n - 1
    system = np.zeros((m, m))
    rhs = np.zeros(m)
    for t in range(m):
        system[t, t] = 2.0 * k + lam * sigma2
        if t > 0:
            system[t, t - 1] = -k
        if t < m - 1:
            system[t, t + 1] = -k
    rhs[0] = k * position
    exact = np.linalg.solve(system, rhs)

    closed = holdings_fraction(n, kappa_for(inst, 0))[1:-1] * position
    assert kappa_for(inst, 0) == pytest.approx(0.5, rel=1e-12)
    assert closed == pytest.approx(exact, rel=1e-12)


def test_without_variance_risk_the_trajectory_is_a_straight_line():
    assert kappa_for(cvar(), 0) == 0.0
    assert holdings_fraction(4, 0.0) == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])


@pytest.mark.parametrize("seed", range(5))
def test_the_nearest_row_is_the_nearest_of_every_row(seed):
    rng = np.random.default_rng(seed)
    inst = discrete()
    grid = lot_grid(inst)
    for i in range(inst.n_assets):
        target = rng.uniform(0, 3, size=inst.n_buckets)
        rows = grid.rows(i)
        def distance(row, target=target):
            return sum(abs(v - x) for v, x in zip(row, target, strict=True)), row

        by_hand = min(rows, key=distance)
        assert nearest_row(grid.values[i], grid.initial_lots[i], target) == by_hand


def test_a_position_no_row_reaches_has_no_nearest_row():
    assert nearest_row(((0, 1), (0, 1)), 3, np.zeros(2)) is None


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_result_is_the_evaluators_and_never_a_proof(build):
    inst = build()
    result = almgren_chriss(inst)
    assert result.status == HEURISTIC
    assert result.bracket is None
    assert result.objective == score(inst, result.schedule)[0]
    assert result.schedule.as_array().sum(axis=1).tolist() == list(inst.initial_lots)


def test_a_breached_cardinality_is_reported_not_repaired():
    """Rows are chosen name by name; somewhere on the ladder that breaches D2."""
    breached = [
        r for inst in tier("T1") if inst.dials.discrete_participation
        for r in [almgren_chriss(inst)] if not r.feasibility.feasible
    ]
    for r in breached:
        assert set(r.feasibility.violations) >= {"cardinality"}
        assert r.status == HEURISTIC


def test_it_is_a_solver():
    assert isinstance(AlmgrenChrissSolver(), Solver)
    assert AlmgrenChrissSolver().solve(plain(), Budget()).status == HEURISTIC


def test_rounding_is_deterministic():
    insts = [i for i in tier("T0")]
    first = [almgren_chriss(i).schedule for i in insts]
    again = [almgren_chriss(i).schedule for i in insts]
    assert first == again
    assert all(s is not None for s in first)
