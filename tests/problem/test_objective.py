import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.micro.impact.almgren_chriss import AlmgrenChriss
from quantic.micro.impact.sqrt_law import PowerLawImpact
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import ObjectiveBreakdown, evaluate, impact_model_for
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(*, dials=None, lam=1e-6, lot_size=100, prices=(100.0, 100.0), **kw) -> Instance:
    assets = tuple(
        Asset(s, lot_size=lot_size, tick_size=0.01) for s in ("A", "B")
    )
    params = MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, p, 0.01)
            for s, p in zip(("A", "B"), prices, strict=True)
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    base = dict(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(4, 4), max_lots_per_bucket=4, params=params,
        dials=dials or Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=lam), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def test_the_breakdown_sums_to_the_total():
    b = evaluate(_instance(), _instance().witness)
    assert b.total == pytest.approx(
        b.spread_cost + b.temporary_impact_cost + b.permanent_impact_cost + b.risk_cost
    )


def test_every_term_is_finite_and_non_negative():
    b = evaluate(_instance(), _instance().witness)
    for f in dataclasses.fields(b):
        value = getattr(b, f.name)
        assert np.isfinite(value)
        assert value >= 0.0


def test_spread_cost_is_half_spread_times_shares():
    """4 lots x 100 shares x $0.01 per share, per asset, over the whole schedule."""
    inst = _instance()
    b = evaluate(inst, inst.witness)
    assert b.spread_cost == pytest.approx(2 * 4 * 100 * 0.01)


def test_an_empty_schedule_costs_nothing_but_risk():
    inst = _instance()
    empty = Schedule(lots=((0, 0), (0, 0)))
    b = evaluate(inst, empty)
    assert b.spread_cost == 0.0
    assert b.temporary_impact_cost == 0.0
    assert b.permanent_impact_cost == 0.0
    assert b.risk_cost > 0.0       # the position is still held


# --- dial D1 ---------------------------------------------------------------


def test_d1_off_selects_the_linear_almgren_chriss_model():
    assert isinstance(impact_model_for(_instance(), 0), AlmgrenChriss)


def test_d1_on_selects_the_calibrated_power_law():
    inst = _instance(dials=Dials(concave_impact=True))
    model = impact_model_for(inst, 0)
    assert isinstance(model, PowerLawImpact)
    assert model.delta == 0.5
    assert model.y_coef == 0.8


def test_d1_changes_the_temporary_impact_term_and_nothing_else():
    """If D1 moved any other term, the screening design could not attribute the effect."""
    off = _instance()
    on = _instance(dials=Dials(concave_impact=True))
    b_off, b_on = evaluate(off, off.witness), evaluate(on, on.witness)

    assert b_on.temporary_impact_cost != pytest.approx(b_off.temporary_impact_cost)
    assert b_on.spread_cost == pytest.approx(b_off.spread_cost)
    assert b_on.permanent_impact_cost == pytest.approx(b_off.permanent_impact_cost)
    assert b_on.risk_cost == pytest.approx(b_off.risk_cost)


# --- the unit-discipline tests, where this layer is most likely to be wrong -


def test_doubling_lot_size_and_halving_lots_leaves_every_term_unchanged():
    """The lots-to-shares conversion happens exactly once. If it happened twice,
    or not at all, this identity breaks."""
    small = _instance(lot_size=100)
    big = _instance(
        lot_size=200, initial_lots=(2, 2), max_lots_per_bucket=2,
        witness=Schedule(lots=((1, 1), (1, 1))),
    )
    a = evaluate(small, small.witness)
    b = evaluate(big, big.witness)
    assert a.spread_cost == pytest.approx(b.spread_cost)
    assert a.temporary_impact_cost == pytest.approx(b.temporary_impact_cost)
    assert a.permanent_impact_cost == pytest.approx(b.permanent_impact_cost)
    assert a.risk_cost == pytest.approx(b.risk_cost)


def test_permanent_impact_is_path_independent():
    """With linear permanent impact the term depends only on the total traded.

    Two different full-liquidation schedules must agree, which is a sharp test
    of the whole term including the cumulative-before convention.
    """
    inst = _instance()
    even = Schedule(lots=((2, 2), (2, 2)))
    front = Schedule(lots=((4, 0), (4, 0)))
    assert evaluate(inst, even).permanent_impact_cost == pytest.approx(
        evaluate(inst, front).permanent_impact_cost
    )


def test_permanent_impact_matches_the_closed_form():
    """0.5 * gamma * sigma * X_shares**2 / V * price, summed over assets."""
    inst = _instance()
    shares = 4 * 100
    expected = 2 * 0.5 * 0.1 * 0.005 * shares**2 / 400_000.0 * 100.0
    assert evaluate(inst, inst.witness).permanent_impact_cost == pytest.approx(expected)


def test_scaling_every_price_scales_the_risk_term_quadratically():
    """Sigma_price = diag(p) Sigma_returns diag(p) is quadratic in price.

    Getting this wrong -- using the return covariance directly -- is a silent
    error of price**2, four orders of magnitude at a $100 stock.
    """
    base = _instance(prices=(100.0, 100.0))
    scaled = _instance(prices=(200.0, 200.0))
    assert evaluate(scaled, scaled.witness).risk_cost == pytest.approx(
        4.0 * evaluate(base, base.witness).risk_cost
    )


def test_risk_is_linear_in_the_risk_aversion():
    a = _instance(lam=1e-6)
    b = _instance(lam=2e-6)
    assert evaluate(b, b.witness).risk_cost == pytest.approx(
        2.0 * evaluate(a, a.witness).risk_cost
    )


def test_zero_risk_aversion_removes_the_risk_term_entirely():
    inst = _instance(lam=0.0)
    assert evaluate(inst, inst.witness).risk_cost == 0.0


def test_the_final_bucket_carries_no_risk_under_full_liquidation():
    """h is the holding AFTER each bucket, so the last one is zero."""
    inst = _instance()
    h = inst.remaining_lots(inst.witness)
    assert h[:, -1].tolist() == [0, 0]


def test_front_loading_costs_more_impact_than_spreading():
    """Concave or linear, trading faster costs more. A sanity check on the sign."""
    inst = _instance(dials=Dials(concave_impact=True))
    even = evaluate(inst, Schedule(lots=((2, 2), (2, 2)))).temporary_impact_cost
    front = evaluate(inst, Schedule(lots=((4, 0), (4, 0)))).temporary_impact_cost
    assert front > even


# --- reporting -------------------------------------------------------------


def test_in_bps_rescales_every_term_against_notional():
    b = ObjectiveBreakdown(
        spread_cost=10.0, temporary_impact_cost=20.0,
        permanent_impact_cost=5.0, risk_cost=5.0,
    )
    bps = b.in_bps(notional=100_000.0)
    assert bps.spread_cost == pytest.approx(1.0)      # 10 / 1e5 * 1e4
    assert bps.total == pytest.approx(4.0)


def test_in_bps_rejects_a_non_positive_notional():
    b = ObjectiveBreakdown(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="notional"):
        b.in_bps(notional=0.0)


def test_evaluate_does_not_check_feasibility():
    """Scoring an infeasible schedule is how M3 calibrates penalty magnitudes.

    Conflating cost with feasibility is spec 6.5's named failure mode, so the
    two are deliberately separate calls.
    """
    inst = _instance()
    infeasible = Schedule(lots=((1, 1), (1, 1)))     # sells 2 of 4 on each asset
    assert np.isfinite(evaluate(inst, infeasible).total)


def test_evaluate_rejects_a_schedule_of_the_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        evaluate(_instance(), Schedule(lots=((1, 1, 1), (1, 1, 1))))
