import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.expand import cell_temporary_cost, expand_objective
from quantic.encoding.integer import ENCODINGS, BinaryEncoding, OneHotEncoding
from quantic.encoding.layout import encode_schedule, plan_layout
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
        covariance=np.array([[1.0, 0.3], [0.3, 1.0]]) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    base = dict(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(4, 4), max_lots_per_bucket=4, params=params,
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def _all_schedules(instance):
    """Every schedule on the 2x2 grid within the cell bounds. Small by design."""
    cap = instance.max_lots_per_bucket
    for a0 in range(cap + 1):
        for a1 in range(cap + 1):
            for b0 in range(cap + 1):
                for b1 in range(cap + 1):
                    yield Schedule(lots=((a0, a1), (b0, b1)))


_BUILT: dict = {}


def _built(instance, encoding):
    """Build the layout and expansion once per (instance, encoding).

    The schedule loops below run to 625 grids each; rebuilding the expansion
    inside them would cost thousands of redundant builds per test.
    """
    key = (instance.content_hash, encoding.name)
    if key not in _BUILT:
        layout = plan_layout(instance, encoding)
        _BUILT[key] = (layout, expand_objective(instance, layout, layout.registry))
    return _BUILT[key]


def _energies(instance, encoding, schedule):
    layout, breakdown = _built(instance, encoding)
    return breakdown, encode_schedule(layout, schedule, layout.registry)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_spread_term_matches_the_evaluator_on_every_schedule(encoding):
    inst = _instance(risk=VarianceRisk(lam=0.0))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, encoding, schedule)
        assert breakdown.spread.energy(bits) == pytest.approx(
            evaluate(inst, schedule).spread_cost, rel=REL
        )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_linear_impact_term_matches_the_evaluator_on_every_schedule(encoding):
    inst = _instance(risk=VarianceRisk(lam=0.0))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, encoding, schedule)
        assert breakdown.temporary.energy(bits) == pytest.approx(
            evaluate(inst, schedule).temporary_impact_cost, rel=REL
        )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_permanent_term_matches_the_evaluator_on_every_schedule(encoding):
    """The cross-bucket term: a path-dependent quadratic, easiest to get wrong."""
    inst = _instance(risk=VarianceRisk(lam=0.0))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, encoding, schedule)
        assert breakdown.permanent.energy(bits) == pytest.approx(
            evaluate(inst, schedule).permanent_impact_cost, rel=REL
        )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_variance_term_matches_the_evaluator_on_every_schedule(encoding):
    inst = _instance(risk=VarianceRisk(lam=2.5e-6))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, encoding, schedule)
        assert breakdown.risk.energy(bits) == pytest.approx(
            evaluate(inst, schedule).risk_cost, rel=REL
        )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_total_matches_the_evaluator_on_every_schedule(encoding):
    inst = _instance(risk=VarianceRisk(lam=2.5e-6))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, encoding, schedule)
        assert breakdown.total.energy(bits) == pytest.approx(
            evaluate(inst, schedule).total, rel=REL
        )


def test_the_linear_impact_cost_is_exactly_quadratic_in_lots():
    """The identity the log expansion relies on: cost(x) == cost(1) * x**2."""
    inst = _instance()
    one = cell_temporary_cost(inst, 0, 0, 1)
    for x in range(5):
        assert cell_temporary_cost(inst, 0, 0, x) == pytest.approx(one * x * x, rel=1e-12)


def test_concave_impact_is_exact_under_a_table_encoding():
    inst = _instance(dials=Dials(concave_impact=True), risk=VarianceRisk(lam=0.0))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, OneHotEncoding(), schedule)
        assert breakdown.approximand is None
        assert breakdown.temporary.energy(bits) == pytest.approx(
            evaluate(inst, schedule).temporary_impact_cost, rel=REL
        )


def test_an_unknown_risk_spec_is_refused_even_carrying_a_lam():
    """Dispatch on type. The same impostor M2b caught in the evaluator."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class _NotVariance:
        lam: float = 1.0
        name: str = "not_variance"

        def describe(self) -> dict:
            return {"name": self.name}

    inst = dataclasses.replace(_instance(), risk=_NotVariance())
    layout = plan_layout(inst, BinaryEncoding())
    with pytest.raises(NotImplementedError, match="_NotVariance"):
        expand_objective(inst, layout, layout.registry)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_encode_schedule_produces_a_binary_vector_over_the_registry(encoding):
    inst = _instance()
    layout = plan_layout(inst, encoding)
    bits = encode_schedule(layout, inst.witness, layout.registry)
    assert bits.shape == (len(layout.registry),)
    assert set(np.unique(bits)) <= {0, 1}
