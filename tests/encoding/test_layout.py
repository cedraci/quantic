import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.integer import (
    ENCODINGS,
    BinaryEncoding,
    DomainWallEncoding,
    OneHotEncoding,
)
from quantic.encoding.layout import plan_layout
from quantic.encoding.qubo import QuboBuilder
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(**kw) -> Instance:
    assets = tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B"))
    params = MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01)
            for s in ("A", "B")
        ),
        covariance=np.eye(2) * 1e-5,
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


def _value_at(layout, plan, bits):
    builder = QuboBuilder(layout.registry)
    builder.add_form(plan.value_form)
    vector = [bits.get(k, 0) for k in layout.registry.keys]
    return builder.build().energy(vector)


def _indicator_at(layout, plan, bits):
    builder = QuboBuilder(layout.registry)
    builder.add_form(plan.indicator_form)
    vector = [bits.get(k, 0) for k in layout.registry.keys]
    return builder.build().energy(vector)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_every_cell_round_trips_every_admissible_value(encoding):
    layout = plan_layout(_instance(), encoding)
    for plan in layout.cells:
        for v in plan.values:
            assert plan.decode(plan.encode(v)) == v


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_value_form_agrees_with_the_encoded_bits(encoding):
    layout = plan_layout(_instance(), encoding)
    for plan in layout.cells:
        for v in plan.values:
            assert _value_at(layout, plan, plan.encode(v)) == pytest.approx(float(v))


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_cell_bound_is_the_tighter_of_the_cap_and_the_position(encoding):
    """initial_lots=(1, 4) with a cap of 4: asset 0 can never trade more than 1."""
    layout = plan_layout(_instance(initial_lots=(1, 4)), encoding)
    assert max(layout.cell(0, 0).values) == 1
    assert max(layout.cell(1, 0).values) == 4


@pytest.mark.parametrize(
    "encoding", [OneHotEncoding(), DomainWallEncoding()], ids=lambda e: e.name
)
def test_d2_is_free_under_a_table_encoding(encoding):
    """The forbidden values are simply absent; no indicator and no penalty."""
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=2), MinParticipation((3, 3))),
        witness=Schedule(lots=((4, 0), (0, 4))),
        initial_lots=(4, 4),
    )
    layout = plan_layout(inst, encoding)
    plan = layout.cell(0, 0)
    assert plan.values == (0, 3, 4)
    assert plan.kind == "semi_continuous"
    assert not any(k[0] == "y" for k in layout.registry.keys)


def test_d2_under_the_log_encoding_needs_an_indicator_and_a_residual():
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=2), MinParticipation((3, 3))),
        witness=Schedule(lots=((4, 0), (0, 4))),
        initial_lots=(4, 4),
    )
    layout = plan_layout(inst, BinaryEncoding())
    plan = layout.cell(0, 0)
    assert plan.values == (0, 3, 4)
    assert plan.indicator_key == ("y", 0, 0)
    assert plan.residual_keys  # x = 3*y + residual over [0, 1]
    for v in plan.values:
        assert _value_at(layout, plan, plan.encode(v)) == pytest.approx(float(v))


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_block_cell_costs_exactly_one_bit_under_every_encoding(encoding):
    inst = _instance(
        dials=Dials(block_trades=True),
        constraints=(FullLiquidation(), BlockTrades((Block(asset=0, bucket=0, lots=2),))),
    )
    layout = plan_layout(inst, encoding)
    plan = layout.cell(0, 0)
    assert plan.kind == "block"
    assert plan.bit_keys == (("block", 0, 0),)
    assert plan.values == (0, 2)
    assert _value_at(layout, plan, plan.encode(2)) == pytest.approx(2.0)
    assert _value_at(layout, plan, plan.encode(0)) == pytest.approx(0.0)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_block_cell_refuses_a_value_it_cannot_take(encoding):
    inst = _instance(
        dials=Dials(block_trades=True),
        constraints=(FullLiquidation(), BlockTrades((Block(asset=0, bucket=0, lots=2),))),
    )
    plan = plan_layout(inst, encoding).cell(0, 0)
    with pytest.raises(ValueError, match="not one of this cell"):
        plan.encode(1)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_every_cell_has_an_indicator_when_cardinality_is_present(encoding):
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=2), MinParticipation((3, 3))),
        witness=Schedule(lots=((4, 0), (0, 4))),
    )
    layout = plan_layout(inst, encoding)
    for plan in layout.cells:
        assert plan.indicator_form is not None


def test_cardinality_without_min_participation_is_refused():
    """The generator never emits this pairing; if it ever did, silence would be worse."""
    inst = _instance(constraints=(FullLiquidation(), Cardinality(k=2)))
    with pytest.raises(ValueError, match="indicator"):
        plan_layout(inst, BinaryEncoding())


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_cell_whose_minimum_exceeds_its_cap_can_only_be_zero(encoding):
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=2), MinParticipation((9, 1))),
        witness=Schedule(lots=((0, 0), (4, 4))),
        initial_lots=(0, 8),
        max_lots_per_bucket=4,
    )
    layout = plan_layout(inst, encoding)
    assert layout.cell(0, 0).values == (0,)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_registry_holds_every_cell_key_exactly_once(encoding):
    layout = plan_layout(_instance(), encoding)
    named = [k for plan in layout.cells for k in plan.bit_keys]
    assert sorted(named) == sorted(set(named))
    assert set(named) <= set(layout.registry.keys)


def test_indicator_is_forced_under_log_encoding_even_with_a_degenerate_minimum():
    """min_lots == 1 (PLAIN) must still get a real indicator when cardinality is
    present; the generator emits exactly this pairing (min_lots[i] == 1 for an
    asset that never trades or trades at most 1 lot)."""
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=2), MinParticipation((1, 3))),
    )
    layout = plan_layout(inst, BinaryEncoding())
    for plan in layout.cells:
        assert plan.indicator_form is not None
    plan = layout.cell(0, 0)
    assert plan.values == (0, 1, 2, 3, 4)
    for v in plan.values:
        assert plan.decode(plan.encode(v)) == v
        assert _value_at(layout, plan, plan.encode(v)) == pytest.approx(float(v))


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_cell_whose_minimum_exceeds_its_cap_still_has_a_zero_indicator_under_cardinality(
    encoding,
):
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=2), MinParticipation((9, 1))),
        witness=Schedule(lots=((0, 0), (4, 4))),
        initial_lots=(0, 8),
        max_lots_per_bucket=4,
    )
    layout = plan_layout(inst, encoding)
    plan = layout.cell(0, 0)
    assert plan.values == (0,)
    assert plan.indicator_form is not None
    assert _indicator_at(layout, plan, plan.encode(0)) == pytest.approx(0.0)


def test_an_unknown_constraint_type_is_refused_even_carrying_a_known_name():
    """Dispatch is on type. M2b's lesson, applied to constraints."""

    @dataclasses.dataclass(frozen=True)
    class _NotAConstraint:
        name: str = "min_participation"
        min_lots: tuple[int, ...] = (1, 1)

        def violation(self, schedule, initial_lots) -> float:
            return 0.0

        def is_satisfied(self, schedule, initial_lots) -> bool:
            return True

        def describe(self) -> dict:
            return {"name": self.name}

    inst = _instance(constraints=(FullLiquidation(), _NotAConstraint()))
    with pytest.raises(NotImplementedError, match="_NotAConstraint"):
        plan_layout(inst, BinaryEncoding())
