import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.feasibility import FeasibilityReport, classify
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(constraints=None, witness=None) -> Instance:
    assets = (Asset("A", lot_size=100, tick_size=0.01), Asset("B", lot_size=100, tick_size=0.01))
    params = MarketParams(
        assets=(
            AssetParams("A", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01),
            AssetParams("B", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01),
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    return Instance(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(3, 4), max_lots_per_bucket=3, params=params, dials=Dials(),
        constraints=constraints or (FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6),
        witness=witness or Schedule(lots=((2, 1), (3, 1))),
        seed=0,
    )


def test_the_witness_is_feasible():
    inst = _instance()
    report = classify(inst, inst.witness)
    assert isinstance(report, FeasibilityReport)
    assert report.feasible
    assert report.total_violation == 0.0


def test_every_constraint_appears_in_the_report_even_when_satisfied():
    """A metric that only lists breaches cannot distinguish 'clean' from 'unchecked'."""
    inst = _instance(constraints=(FullLiquidation(), Cardinality(k=2)))
    report = classify(inst, inst.witness)
    assert set(report.violations) == {"full_liquidation", "cardinality"}
    assert all(v == 0.0 for v in report.violations.values())


def test_a_breach_is_reported_against_the_constraint_that_caused_it():
    inst = _instance()
    bad = Schedule(lots=((1, 1), (3, 1)))       # asset 0 sells 2 of 3
    report = classify(inst, bad)
    assert not report.feasible
    assert report.violations["full_liquidation"] == 1.0
    assert report.total_violation == 1.0


def test_violations_from_several_constraints_are_summed():
    inst = _instance(constraints=(FullLiquidation(), Cardinality(k=1)))
    bad = Schedule(lots=((1, 1), (3, 1)))       # 1 short, and 2 names in both buckets
    report = classify(inst, bad)
    assert report.violations["full_liquidation"] == 1.0
    assert report.violations["cardinality"] == 2.0
    assert report.total_violation == 3.0


def test_feasible_is_exactly_total_violation_being_zero():
    inst = _instance(constraints=(FullLiquidation(), Cardinality(k=1)))
    for lots in (((2, 1), (3, 1)), ((1, 1), (3, 1)), ((3, 0), (0, 4))):
        report = classify(inst, Schedule(lots=lots))
        assert report.feasible == (report.total_violation == 0.0)


def test_a_schedule_of_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        classify(_instance(), Schedule(lots=((1, 1, 1), (1, 1, 1))))


def test_duplicate_constraint_names_are_rejected():
    """Two constraints sharing a name would overwrite each other in the report."""
    inst = _instance(constraints=(Cardinality(k=1), Cardinality(k=2)))
    with pytest.raises(ValueError, match="duplicate|cardinality"):
        classify(inst, inst.witness)


def test_the_report_is_hashable_and_frozen():
    inst = _instance()
    report = classify(inst, inst.witness)
    with pytest.raises(Exception):  # noqa: B017
        report.feasible = False       # type: ignore[misc]
