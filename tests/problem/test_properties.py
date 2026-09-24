"""Properties that must hold for ANY schedule, not just the witness.

The parametrised gates in test_generator and test_m2a_gate cover the tier x
dial cross product exhaustively, but only ever evaluate the witness. A solver
returns arbitrary schedules, and M3 deliberately scores infeasible ones while
calibrating penalty magnitudes, so the invariants below are exercised over
generated grids instead.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.generator import generate
from quantic.problem.liquidation import T0
from quantic.problem.objective import evaluate
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
MAX_LOTS = 6


@st.composite
def grids(draw, n_assets=None, n_buckets=None, max_value=MAX_LOTS):
    """An arbitrary non-negative lot grid, every cell drawn from ``0..max_value``.

    ``max_value`` exists so the instance-based properties below can generate
    grids already inside a tier's per-bucket ceiling. Filtering afterwards with
    ``assume`` would be catastrophic at T0: the grid is 3x4 and the ceiling is
    3, so only ``(4/7)**12 = 0.12%`` of draws survive and Hypothesis's
    ``filter_too_much`` health check fires every run, deterministically.
    Generating in range is faster and samples the space that actually matters.
    """
    rows = n_assets if n_assets is not None else draw(st.integers(1, 4))
    cols = n_buckets if n_buckets is not None else draw(st.integers(1, 4))
    flat = draw(
        st.lists(
            st.integers(0, max_value), min_size=rows * cols, max_size=rows * cols
        )
    )
    return Schedule(
        lots=tuple(tuple(flat[r * cols : (r + 1) * cols]) for r in range(rows))
    )


def _params(n: int) -> MarketParams:
    assets = tuple(
        AssetParams(
            symbol=f"S{i:02d}", delta=0.5, y_coef=0.8, gamma=0.1,
            sigma_bucket=0.005, bucket_volume_shares=400_000.0,
            price=100.0 + i, half_spread=0.01,
        )
        for i in range(n)
    )
    return MarketParams(assets=assets, covariance=np.eye(n) * 1e-5, bucket_ns=BUCKET_NS)


# --- Schedule --------------------------------------------------------------


@given(grids())
def test_the_two_cumulative_conventions_always_differ_by_the_schedule(s):
    assert (s.cumulative() - s.cumulative_before()).tolist() == s.as_array().tolist()


@given(grids())
def test_cumulative_before_always_starts_at_zero(s):
    assert s.cumulative_before()[:, 0].tolist() == [0] * s.n_assets


@given(grids())
def test_a_schedule_round_trips_through_its_array(s):
    assert Schedule.from_array(s.as_array()) == s


@given(grids())
def test_active_names_never_exceeds_the_asset_count(s):
    assert all(0 <= s.active_names(t) <= s.n_assets for t in range(s.n_buckets))


# --- constraints -----------------------------------------------------------


@given(grids(n_assets=3, n_buckets=3), st.lists(st.integers(0, 12), min_size=3, max_size=3))
def test_violation_is_zero_exactly_when_satisfied_for_every_constraint(s, positions):
    x = tuple(positions)
    constraints = [
        FullLiquidation(),
        Cardinality(k=2),
        MinParticipation(min_lots=(2, 3, 2)),
        BlockTrades(blocks=(Block(asset=0, bucket=0, lots=4),)),
    ]
    for c in constraints:
        assert (c.violation(s, x) == 0.0) == c.is_satisfied(s, x), c.name


@given(grids(n_assets=3, n_buckets=3), st.lists(st.integers(0, 12), min_size=3, max_size=3))
def test_violations_are_always_non_negative_and_finite(s, positions):
    x = tuple(positions)
    for c in (
        FullLiquidation(),
        Cardinality(k=2),
        MinParticipation(min_lots=(2, 3, 2)),
        BlockTrades(blocks=(Block(asset=0, bucket=0, lots=4),)),
    ):
        v = c.violation(s, x)
        assert v >= 0.0 and np.isfinite(v), c.name


@given(grids(n_assets=3, n_buckets=3), st.integers(1, 3), st.integers(1, 3))
def test_a_tighter_cardinality_never_reports_a_smaller_violation(s, a, b):
    """Monotone in k: tightening a constraint cannot make a schedule more feasible."""
    tight, loose = min(a, b), max(a, b)
    x = (0, 0, 0)
    assert Cardinality(k=tight).violation(s, x) >= Cardinality(k=loose).violation(s, x)


@given(grids(n_assets=3, n_buckets=3), st.integers(1, 6), st.integers(1, 6))
def test_a_higher_minimum_lot_never_reports_a_smaller_violation(s, a, b):
    tight, loose = max(a, b), min(a, b)
    x = (0, 0, 0)
    assert MinParticipation(min_lots=(tight,) * 3).violation(s, x) >= MinParticipation(
        min_lots=(loose,) * 3
    ).violation(s, x)


# --- objective and feasibility, against a real instance --------------------


@settings(max_examples=50, deadline=None)
@given(grids(n_assets=T0.n_assets, n_buckets=T0.n_buckets,
             max_value=T0.max_lots_per_bucket))
def test_the_breakdown_always_sums_to_the_total(s):
    inst = generate(T0, Dials(concave_impact=True), _params(T0.n_assets), seed=0)
    b = evaluate(inst, s)
    assert b.total == float(
        b.spread_cost + b.temporary_impact_cost + b.permanent_impact_cost + b.risk_cost
    )


@settings(max_examples=50, deadline=None)
@given(grids(n_assets=T0.n_assets, n_buckets=T0.n_buckets,
             max_value=T0.max_lots_per_bucket))
def test_every_objective_term_is_finite_and_non_negative_for_any_schedule(s):
    """M3 scores infeasible schedules while calibrating penalties; none may be NaN."""
    inst = generate(T0, Dials(concave_impact=True), _params(T0.n_assets), seed=0)
    b = evaluate(inst, s)
    for value in (
        b.spread_cost, b.temporary_impact_cost, b.permanent_impact_cost, b.risk_cost
    ):
        assert np.isfinite(value) and value >= 0.0


@settings(max_examples=50, deadline=None)
@given(grids(n_assets=T0.n_assets, n_buckets=T0.n_buckets,
             max_value=T0.max_lots_per_bucket))
def test_the_total_violation_always_equals_the_sum_of_its_parts(s):
    inst = generate(T0, Dials(discrete_participation=True), _params(T0.n_assets), seed=0)
    report = classify(inst, s)
    assert report.total_violation == float(sum(report.violations.values()))
    assert report.feasible == (report.total_violation == 0.0)
