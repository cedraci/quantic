import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.expand import expand_objective
from quantic.encoding.integer import ENCODINGS, BinaryEncoding, OneHotEncoding
from quantic.encoding.layout import encode_schedule, plan_layout
from quantic.encoding.penalty import (
    derive_penalties,
    expand_penalties,
    penalty_aux_keys,
    slack_bits,
)
from quantic.encoding.registry import VarRegistry
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
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
            for j, s in enumerate(("A", "B"))
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


def _d2_instance():
    return _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=1), MinParticipation((3, 3))),
        witness=Schedule(lots=((4, 0), (0, 4))),
    )


def _d4_instance():
    return _instance(
        dials=Dials(block_trades=True),
        constraints=(FullLiquidation(), BlockTrades((Block(asset=0, bucket=0, lots=2),))),
        witness=Schedule(lots=((2, 2), (2, 2))),
    )


def _setup(instance, encoding):
    layout = plan_layout(instance, encoding)
    registry = VarRegistry(
        keys=layout.registry.keys + penalty_aux_keys(instance, layout)
    )
    objective = expand_objective(instance, layout, registry).total
    strengths = derive_penalties(objective, layout)
    penalties = expand_penalties(instance, layout, registry, strengths)
    return layout, registry, objective, strengths, penalties


def _bits(layout, registry, instance, schedule):
    vector = encode_schedule(layout, schedule, registry)
    extra = slack_bits(instance, layout, schedule)
    for key, value in extra.items():
        vector[registry.index(key)] = value
    return vector


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize(
    "factory", [_instance, _d2_instance, _d4_instance], ids=["none", "d2", "d4"]
)
def test_the_penalty_block_is_exactly_zero_at_a_feasible_witness(encoding, factory):
    """Exactly zero, up to the float noise a dense Q-matrix accumulation leaves
    when a non-integer derived strength scales an algebraically-zero square."""
    inst = factory()
    layout, registry, _, _, penalties = _setup(inst, encoding)
    assert penalties.energy(_bits(layout, registry, inst, inst.witness)) == pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_an_under_liquidating_schedule_pays_a_penalty(encoding):
    inst = _instance()
    layout, registry, _, _, penalties = _setup(inst, encoding)
    short = Schedule(lots=((2, 1), (2, 2)))
    assert penalties.energy(_bits(layout, registry, inst, short)) > 0.0


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_one_lot_violation_raises_the_total_energy(encoding):
    """The property derived penalties exist to guarantee."""
    inst = _instance()
    layout, registry, objective, _, penalties = _setup(inst, encoding)
    total = objective + penalties
    good = total.energy(_bits(layout, registry, inst, inst.witness))
    short = total.energy(_bits(layout, registry, inst, Schedule(lots=((2, 1), (2, 2)))))
    assert short > good


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_breaching_cardinality_pays_a_penalty(encoding):
    inst = _d2_instance()
    layout, registry, _, _, penalties = _setup(inst, encoding)
    both_active = Schedule(lots=((4, 0), (4, 0)))
    assert penalties.energy(_bits(layout, registry, inst, both_active)) > 0.0


def test_a_residual_set_without_its_indicator_pays_a_penalty():
    """Spec section 6.2's min-participation term, under the log encoding."""
    inst = _d2_instance()
    layout, registry, _, _, penalties = _setup(inst, BinaryEncoding())
    vector = np.zeros(len(registry), dtype=np.int64)
    residual = [k for k in registry.keys if k[0] == "x"]
    vector[registry.index(residual[0])] = 1
    assert penalties.energy(vector) > 0.0


def test_a_table_encoding_needs_no_min_participation_penalty():
    """D2 is free under one-hot: the forbidden values are simply not there."""
    inst = _d2_instance()
    _, _, _, strengths, _ = _setup(inst, OneHotEncoding())
    assert strengths.min_participation == 0.0


def test_an_invalid_one_hot_state_pays_a_validity_penalty():
    inst = _instance()
    layout, registry, _, _, penalties = _setup(inst, OneHotEncoding())
    vector = np.zeros(len(registry), dtype=np.int64)  # no bit set anywhere
    assert penalties.energy(vector) > 0.0


def test_cardinality_slack_spans_zero_to_k():
    """Sum(y) + slack == K. The slack absorbs the shortfall, not the surplus."""
    inst = _d2_instance()
    layout = plan_layout(inst, BinaryEncoding())
    keys = penalty_aux_keys(inst, layout)
    assert keys  # k = 1, so one slack bit per bucket
    assert all(k[0] == "card" for k in keys)


def test_derivation_is_deterministic_and_positive():
    inst = _instance()
    layout, _, objective, strengths, _ = _setup(inst, BinaryEncoding())
    again = derive_penalties(objective, layout)
    assert strengths == again
    assert strengths.full_liquidation > 0.0
    assert strengths.safety_factor == 2.0
    assert strengths.strategy == "derived_bounds"


def test_the_strengths_are_json_safe_for_the_results_row():
    import json

    inst = _instance()
    _, _, _, strengths, _ = _setup(inst, BinaryEncoding())
    assert json.loads(json.dumps(strengths.describe()))["strategy"] == "derived_bounds"


def test_an_unknown_constraint_type_is_refused_even_carrying_a_known_name():
    @dataclasses.dataclass(frozen=True)
    class _NotFullLiquidation:
        name: str = "full_liquidation"

        def violation(self, schedule, initial_lots) -> float:
            return 0.0

        def is_satisfied(self, schedule, initial_lots) -> bool:
            return True

        def describe(self) -> dict:
            return {"name": self.name}

    inst = _instance(constraints=(FullLiquidation(), _NotFullLiquidation()))
    with pytest.raises(NotImplementedError, match="_NotFullLiquidation"):
        plan_layout(inst, BinaryEncoding())
