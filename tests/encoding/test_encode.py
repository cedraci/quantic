import json

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.encode import encode, encode_solution, reference_breakdown
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import cvar_of, optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
REL = 1e-9


def _params():
    return MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
            for j, s in enumerate(("A", "B"))
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )


def _instance(**kw) -> Instance:
    base = dict(
        instance_id="x", tier="T0",
        assets=tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B")),
        n_buckets=2, initial_lots=(4, 4), max_lots_per_bucket=4, params=_params(),
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def _cvar_instance():
    rng = np.random.default_rng(7)
    scenarios = ScenarioSet(
        returns=rng.normal(scale=0.01, size=(4, 2, 2)),
        weights=np.full(4, 0.25), seed=7,
    )
    return _instance(
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=1e-6, alpha=0.95, scenarios=scenarios),
    )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_energy_at_the_witness_equals_the_reference_objective(encoding):
    inst = _instance()
    enc = encode(inst, encoding)
    bits = encode_solution(enc, inst.witness)
    assert enc.qubo.energy(bits) == pytest.approx(
        reference_breakdown(enc, inst.witness).total, rel=REL
    )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_penalty_block_is_exactly_zero_at_the_witness(encoding):
    """Exactly zero, up to the float noise a dense Q-matrix accumulation leaves
    when a non-integer derived strength scales an algebraically-zero square --
    see tests/encoding/test_penalty.py for the full justification of the bound."""
    inst = _instance()
    enc = encode(inst, encoding)
    floor = 1e-9 * max(enc.strengths.full_liquidation, 1.0)
    assert abs(enc.penalties.energy(encode_solution(enc, inst.witness))) < floor


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_cvar_instance_encodes_and_scores_at_a_snapped_zeta(encoding):
    inst = _cvar_instance()
    enc = encode(inst, encoding)
    zeta = enc.gadget.zeta.snap(optimal_zeta(inst, inst.witness))
    bits = encode_solution(enc, inst.witness, zeta=zeta)
    losses = enc.gadget.losses_from(inst, enc.layout, inst.witness)
    aux = enc.gadget.optimal_aux(losses, zeta)
    continuous = inst.risk.lam * cvar_of(
        inst.risk.scenarios, inst, inst.witness, alpha=inst.risk.alpha, zeta=zeta
    )
    expected = (
        reference_breakdown(enc, inst.witness, zeta=zeta).total
        - continuous
        + enc.gadget.encoded_cvar(losses, zeta, aux)
    )
    assert enc.qubo.energy(bits) == pytest.approx(expected, rel=REL)


def test_encoding_a_cvar_instance_without_a_zeta_is_refused():
    enc = encode(_cvar_instance(), BinaryEncoding())
    with pytest.raises(ValueError, match="zeta"):
        encode_solution(enc, enc.instance.witness)


def test_passing_a_zeta_to_a_variance_instance_is_refused():
    enc = encode(_instance(), BinaryEncoding())
    with pytest.raises(ValueError, match="no zeta"):
        encode_solution(enc, enc.instance.witness, zeta=0.0)


def test_the_derived_strength_does_not_depend_on_its_own_penalty_block():
    """The two-pass derivation must not feed the CVaR penalty back into the bound."""
    inst = _cvar_instance()
    first = encode(inst, BinaryEncoding())
    second = encode(inst, BinaryEncoding())
    assert first.strengths == second.strengths
    assert first.gadget.penalty == first.strengths.cvar


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_encoding_is_deterministic(encoding):
    inst = _instance()
    a, b = encode(inst, encoding), encode(inst, encoding)
    assert np.array_equal(a.qubo.Q, b.qubo.Q)
    assert a.qubo.constant == b.qubo.constant
    assert a.registry.keys == b.registry.keys


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_registry_covers_every_bit_any_stage_named(encoding):
    enc = encode(_cvar_instance(), encoding)
    named = {k for plan in enc.layout.cells for k in plan.bit_keys}
    named |= set(enc.gadget.keys)
    assert named <= set(enc.registry.keys)
    assert enc.qubo.n_vars == len(enc.registry)


def test_the_description_is_json_safe_and_names_the_encoding():
    enc = encode(_instance(), BinaryEncoding())
    payload = json.loads(json.dumps(enc.describe()))
    assert payload["encoding"] == "binary"
    assert payload["n_vars"] == enc.n_vars
    assert payload["penalties"]["strategy"] == "derived_bounds"


def test_the_reference_breakdown_swaps_the_impact_term_when_approximated():
    import dataclasses

    from quantic.problem.objective import evaluate

    inst = _instance(dials=Dials(concave_impact=True))
    enc = encode(inst, BinaryEncoding())
    assert enc.breakdown.approximand is not None
    expected = dataclasses.replace(
        evaluate(inst, inst.witness),
        temporary_impact_cost=enc.breakdown.approximand.temporary_cost(inst.witness),
    )
    assert reference_breakdown(enc, inst.witness) == expected


def test_the_reference_breakdown_is_untouched_when_nothing_was_approximated():
    from quantic.problem.objective import evaluate

    inst = _instance()
    enc = encode(inst, BinaryEncoding())
    assert enc.breakdown.approximand is None
    assert reference_breakdown(enc, inst.witness) == evaluate(inst, inst.witness)
