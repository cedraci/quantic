import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.expand import cell_temporary_cost, expand_objective
from quantic.encoding.integer import BinaryEncoding, DomainWallEncoding, OneHotEncoding
from quantic.encoding.layout import encode_schedule, plan_layout
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import evaluate
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
REL = 1e-9


def _instance(**kw) -> Instance:
    assets = tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B"))
    params = MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
            for j, s in enumerate(("A", "B"))
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    base = dict(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(4, 4), max_lots_per_bucket=4, params=params,
        dials=Dials(concave_impact=True), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=0.0), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def _all_schedules(cap=4):
    for a0 in range(cap + 1):
        for a1 in range(cap + 1):
            for b0 in range(cap + 1):
                for b1 in range(cap + 1):
                    yield Schedule(lots=((a0, a1), (b0, b1)))


def test_the_log_encoding_now_reports_an_approximand_instead_of_raising():
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    breakdown = expand_objective(inst, layout, layout.registry)
    assert breakdown.approximand is not None
    assert breakdown.approximand.max_abs_error > 0.0


@pytest.mark.parametrize("encoding", [OneHotEncoding(), DomainWallEncoding()])
def test_a_table_encoding_still_reports_no_approximand(encoding):
    """The absence of a report is the claim that the encoding was exact."""
    inst = _instance()
    layout = plan_layout(inst, encoding)
    assert expand_objective(inst, layout, layout.registry).approximand is None


def test_the_qubo_equals_the_objective_computed_with_the_substituted_term():
    """Assertion one of two: exact, no modelling tolerance."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    breakdown = expand_objective(inst, layout, layout.registry)
    report = breakdown.approximand
    for schedule in _all_schedules():
        bits = encode_schedule(layout, schedule, layout.registry)
        expected = dataclasses.replace(
            evaluate(inst, schedule),
            temporary_impact_cost=report.temporary_cost(schedule),
        )
        assert breakdown.total.energy(bits) == pytest.approx(expected.total, rel=REL)


def test_the_reported_error_is_the_true_maximum_over_the_lot_grid():
    """Assertion two of two: the report does not flatter itself."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    report = expand_objective(inst, layout, layout.registry).approximand
    worst = 0.0
    for plan in layout.cells:
        b, c = report.coefficients[(plan.asset, plan.bucket)]
        for v in plan.values:
            exact = cell_temporary_cost(inst, plan.asset, plan.bucket, v)
            worst = max(worst, abs(exact - (b * v + c * v * v)))
    assert report.max_abs_error == pytest.approx(worst, rel=1e-12)


def test_the_fit_is_exact_at_zero_lots():
    """f(0) == 0 exactly; a fitted constant would shift every energy."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    report = expand_objective(inst, layout, layout.registry).approximand
    empty = Schedule(lots=((0, 0), (0, 0)))
    assert report.temporary_cost(empty) == 0.0


def test_the_report_is_json_safe_for_the_results_row():
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    import json

    payload = expand_objective(inst, layout, layout.registry).approximand.describe()
    assert json.loads(json.dumps(payload))["max_abs_error"] >= 0.0


def test_no_approximand_appears_when_d1_is_off_under_the_log_encoding():
    inst = _instance(dials=Dials())
    layout = plan_layout(inst, BinaryEncoding())
    assert expand_objective(inst, layout, layout.registry).approximand is None


def _block_instance() -> Instance:
    return _instance(
        dials=Dials(concave_impact=True, block_trades=True),
        constraints=(
            FullLiquidation(),
            BlockTrades((Block(asset=0, bucket=0, lots=2),)),
        ),
    )


def test_the_substituted_term_still_charges_the_cells_it_did_not_fit():
    """A block cell is one bit and exact, so the fit never sees it.

    The M3a gate's first finding: the reference dropped those cells, and came
    out low by exactly what the blocks cost.
    """
    inst = _block_instance()
    layout = plan_layout(inst, BinaryEncoding())
    breakdown = expand_objective(inst, layout, layout.registry)
    report = breakdown.approximand
    assert (0, 0) not in report.coefficients
    for a1 in range(5):
        for b0 in range(5):
            for b1 in range(5):
                for a0 in (0, 2):
                    schedule = Schedule(lots=((a0, a1), (b0, b1)))
                    bits = encode_schedule(layout, schedule, layout.registry)
                    assert breakdown.temporary.energy(bits) == pytest.approx(
                        report.temporary_cost(schedule), rel=REL
                    )


def test_an_unfitted_cell_is_charged_what_the_evaluator_charges_it():
    inst = _block_instance()
    layout = plan_layout(inst, BinaryEncoding())
    report = expand_objective(inst, layout, layout.registry).approximand
    only_the_block = Schedule(lots=((2, 0), (0, 0)))
    assert report.temporary_cost(only_the_block) == pytest.approx(
        evaluate(inst, only_the_block).temporary_impact_cost, rel=REL
    )


def test_the_substituted_term_refuses_a_value_an_exact_cell_cannot_take():
    inst = _block_instance()
    layout = plan_layout(inst, BinaryEncoding())
    report = expand_objective(inst, layout, layout.registry).approximand
    with pytest.raises(ValueError, match="asset 0, bucket 0"):
        report.temporary_cost(Schedule(lots=((1, 0), (0, 0))))
