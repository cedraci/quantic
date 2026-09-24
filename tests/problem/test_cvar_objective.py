"""The Rockafellar-Uryasev CVaR term.

Every claim here was verified numerically while the spec was written. The
tolerances are measured, not guessed: the variance correspondence holds only
in expectation, and R-U equals the sorted-tail mean exactly only when
alpha * S is an integer.
"""

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import cvar_of, evaluate, optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.scenarios import generate_scenarios
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
ALPHA = 0.95


def _params(n: int = 3) -> MarketParams:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(n, n))
    return MarketParams(
        assets=tuple(
            AssetParams(f"S{i}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01)
            for i in range(n)
        ),
        covariance=(a @ a.T) / n * 1e-4,
        bucket_ns=BUCKET_NS,
    )


def _instance(risk, n: int = 3, t: int = 4) -> Instance:
    params = _params(n)
    return Instance(
        instance_id="x", tier="T0",
        assets=tuple(Asset(f"S{i}", lot_size=100, tick_size=0.01) for i in range(n)),
        n_buckets=t, initial_lots=tuple(2 * t for _ in range(n)),
        max_lots_per_bucket=2, params=params, dials=Dials(),
        constraints=(FullLiquidation(),), risk=risk,
        witness=Schedule(lots=tuple(tuple(2 for _ in range(t)) for _ in range(n))),
        seed=0,
    )


def _cvar_instance(lam: float = 1.0, alpha: float = ALPHA, n_raw: int = 20_000):
    scenarios = generate_scenarios(_params(3), n_buckets=4, seed=11, n_raw=n_raw)
    return _instance(CVaRRisk(lam=lam, alpha=alpha, scenarios=scenarios))


# --- the variational property: the whole formulation in one test ------------


def test_rockafellar_uryasev_is_minimised_at_the_optimal_zeta():
    """For ANY zeta, CVaR(x, zeta) >= CVaR(x, optimal_zeta(x)).

    This single property validates the entire R-U formulation, and is exactly
    what M3's binary-expanded zeta trades against. Verified during design over
    a 400-point grid; 60 points is ample here.
    """
    inst = _cvar_instance()
    best = optimal_zeta(inst, inst.witness)
    at_best = evaluate(inst, inst.witness, zeta=best).risk_cost

    losses = inst.risk.scenarios.losses(inst, inst.witness)
    for zeta in np.linspace(losses.min(), losses.max(), 60):
        assert evaluate(inst, inst.witness, zeta=float(zeta)).risk_cost >= at_best - 1e-6


def test_at_the_optimal_zeta_ru_equals_the_sorted_tail_mean():
    """R-U attains the true CVaR at zeta = VaR.

    Exact equality only when alpha * S is an integer, so this is a tolerance,
    not an assertion of identity. Measured during design: rel diff 2.0e-5.
    """
    inst = _cvar_instance(lam=1.0)
    losses = inst.risk.scenarios.losses(inst, inst.witness)
    weights = inst.risk.scenarios.weights

    order = np.argsort(losses)
    cumulative = np.cumsum(weights[order])
    tail = cumulative >= ALPHA
    closed_form = float(
        np.sum(weights[order][tail] * losses[order][tail]) / np.sum(weights[order][tail])
    )

    best = optimal_zeta(inst, inst.witness)
    assert evaluate(inst, inst.witness, zeta=best).risk_cost == pytest.approx(
        closed_form, rel=1e-3
    )


# --- the correspondence with the variance term ------------------------------


def test_the_scenario_loss_variance_matches_the_variance_terms_quadratic_form():
    """THE TIE between the scenario machinery and the term CVaR replaces.

    Var(L_s) = sum_t h_t' Sigma_price h_t. Measured during design at
    S=200,000: ratio 0.9965. At S=20,000 a 5% tolerance is comfortable and
    still fails loudly on a wrong covariance or a missing lots-to-shares
    conversion.
    """
    inst = _cvar_instance(n_raw=20_000)
    losses = inst.risk.scenarios.losses(inst, inst.witness)

    holdings = inst.remaining_lots(inst.witness) * inst.lot_sizes()[:, None]
    sigma_price = inst.params.price_covariance()
    analytic = sum(
        float(holdings[:, t] @ sigma_price @ holdings[:, t])
        for t in range(inst.n_buckets)
    )
    assert float(losses.var()) == pytest.approx(analytic, rel=0.05)


# --- CVaR's basic shape -----------------------------------------------------


def test_cvar_exceeds_the_mean_loss():
    inst = _cvar_instance()
    losses = inst.risk.scenarios.losses(inst, inst.witness)
    mean = float(inst.risk.scenarios.weights @ losses)
    best = optimal_zeta(inst, inst.witness)
    assert evaluate(inst, inst.witness, zeta=best).risk_cost > mean


def test_cvar_is_non_decreasing_in_the_tail_level():
    values = []
    for alpha in (0.80, 0.90, 0.95, 0.99):
        inst = _cvar_instance(alpha=alpha)
        values.append(
            evaluate(inst, inst.witness, zeta=optimal_zeta(inst, inst.witness)).risk_cost
        )
    assert values == sorted(values)


def test_risk_cost_is_linear_in_the_risk_aversion():
    one, two = _cvar_instance(lam=1.0), _cvar_instance(lam=2.0)
    z = optimal_zeta(one, one.witness)
    assert evaluate(two, two.witness, zeta=z).risk_cost == pytest.approx(
        2.0 * evaluate(one, one.witness, zeta=z).risk_cost
    )


def test_zero_risk_aversion_removes_the_cvar_term():
    inst = _cvar_instance(lam=0.0)
    assert evaluate(inst, inst.witness, zeta=optimal_zeta(inst, inst.witness)).risk_cost == 0.0


def test_cvar_changes_no_other_objective_term():
    """D3 must swap the risk summary and nothing else."""
    scenarios = generate_scenarios(_params(3), n_buckets=4, seed=11, n_raw=2000)
    variance = _instance(VarianceRisk(lam=1e-9))
    cvar = _instance(CVaRRisk(lam=1e-9, alpha=ALPHA, scenarios=scenarios))

    a = evaluate(variance, variance.witness)
    b = evaluate(cvar, cvar.witness, zeta=optimal_zeta(cvar, cvar.witness))
    assert b.spread_cost == pytest.approx(a.spread_cost)
    assert b.temporary_impact_cost == pytest.approx(a.temporary_impact_cost)
    assert b.permanent_impact_cost == pytest.approx(a.permanent_impact_cost)


# --- cvar_of, the diagnostic used for reduction error -----------------------


def test_cvar_of_scores_an_arbitrary_scenario_set():
    inst = _cvar_instance()
    other = generate_scenarios(_params(3), n_buckets=4, seed=99, n_raw=2000)
    value = cvar_of(other, inst, inst.witness, alpha=ALPHA)
    assert np.isfinite(value) and value > 0.0


def test_cvar_of_at_an_explicit_zeta_is_at_least_its_optimum():
    inst = _cvar_instance()
    scenarios = inst.risk.scenarios
    best = cvar_of(scenarios, inst, inst.witness, alpha=ALPHA)
    worse = cvar_of(scenarios, inst, inst.witness, alpha=ALPHA, zeta=0.0)
    assert worse >= best - 1e-6


# --- the zeta contract ------------------------------------------------------


def test_omitting_zeta_under_cvar_raises():
    """No silent default: the call site must state which zeta it is scoring."""
    inst = _cvar_instance()
    with pytest.raises(ValueError, match="zeta"):
        evaluate(inst, inst.witness)


def test_supplying_zeta_under_variance_risk_raises():
    inst = _instance(VarianceRisk(lam=1e-6))
    with pytest.raises(ValueError, match="zeta"):
        evaluate(inst, inst.witness, zeta=0.0)


def test_optimal_zeta_requires_a_cvar_risk():
    inst = _instance(VarianceRisk(lam=1e-6))
    with pytest.raises(TypeError, match="CVaRRisk"):
        optimal_zeta(inst, inst.witness)


def test_optimal_zeta_lies_within_the_loss_range():
    inst = _cvar_instance()
    losses = inst.risk.scenarios.losses(inst, inst.witness)
    assert losses.min() <= optimal_zeta(inst, inst.witness) <= losses.max()


def test_the_breakdown_still_sums_to_the_total_under_cvar():
    inst = _cvar_instance()
    b = evaluate(inst, inst.witness, zeta=optimal_zeta(inst, inst.witness))
    assert b.total == pytest.approx(
        b.spread_cost + b.temporary_impact_cost + b.permanent_impact_cost + b.risk_cost
    )
