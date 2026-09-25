import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.decode import decode
from quantic.encoding.encode import encode, encode_solution
from quantic.encoding.integer import (
    ENCODINGS,
    BinaryEncoding,
    DomainWallEncoding,
    OneHotEncoding,
)
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(**kw) -> Instance:
    base = dict(
        instance_id="x", tier="T0",
        assets=tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B")),
        n_buckets=2, initial_lots=(4, 4), max_lots_per_bucket=4,
        params=MarketParams(
            assets=tuple(
                AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
                for j, s in enumerate(("A", "B"))
            ),
            covariance=np.eye(2) * 1e-5, bucket_ns=BUCKET_NS,
        ),
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_feasible_schedule_round_trips(encoding):
    inst = _instance()
    enc = encode(inst, encoding)
    sample = decode(enc, encode_solution(enc, inst.witness))
    assert sample.schedule == inst.witness
    assert sample.classification == "feasible"
    assert sample.report.feasible


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_every_representable_schedule_round_trips(encoding):
    inst = _instance()
    enc = encode(inst, encoding)
    for a0 in range(5):
        for b0 in range(5):
            schedule = Schedule(lots=((a0, 4 - a0), (b0, 4 - b0)))
            assert decode(enc, encode_solution(enc, schedule)).schedule == schedule


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_decodable_but_infeasible_schedule_is_repairable(encoding):
    inst = _instance()
    enc = encode(inst, encoding)
    short = Schedule(lots=((2, 1), (2, 2)))
    sample = decode(enc, encode_solution(enc, short))
    assert sample.schedule == short
    assert sample.classification == "repairable"
    assert sample.report.total_violation > 0.0


def test_an_empty_one_hot_cell_is_rejected_rather_than_repaired():
    """No schedule to score, so there is nothing to repair."""
    enc = encode(_instance(), OneHotEncoding())
    sample = decode(enc, np.zeros(enc.n_vars, dtype=np.int64))
    assert sample.classification == "rejected"
    assert sample.schedule is None
    assert "one" in sample.reason


def test_a_doubly_set_one_hot_cell_is_rejected():
    enc = encode(_instance(), OneHotEncoding())
    bits = encode_solution(enc, enc.instance.witness)
    first = next(i for i, k in enumerate(enc.registry.keys) if k[0] == "x")
    bits[first] = 1
    bits[first + 1] = 1
    assert decode(enc, bits).classification == "rejected"


def test_a_broken_domain_wall_is_rejected():
    enc = encode(_instance(), DomainWallEncoding())
    bits = np.zeros(enc.n_vars, dtype=np.int64)
    walls = [i for i, k in enumerate(enc.registry.keys) if k[0] == "x"]
    bits[walls[1]] = 1  # a hole: w2 set while w1 is clear
    sample = decode(enc, bits)
    assert sample.classification == "rejected"
    assert "monotone" in sample.reason


def test_a_residual_without_its_indicator_is_rejected():
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=1), MinParticipation((3, 3))),
        witness=Schedule(lots=((4, 0), (0, 4))),
    )
    enc = encode(inst, BinaryEncoding())
    bits = np.zeros(enc.n_vars, dtype=np.int64)
    residual = next(i for i, k in enumerate(enc.registry.keys) if k[0] == "x")
    bits[residual] = 1
    assert decode(enc, bits).classification == "rejected"


def test_a_cvar_sample_reports_the_zeta_it_decoded():
    rng = np.random.default_rng(3)
    scenarios = ScenarioSet(
        returns=rng.normal(scale=0.01, size=(4, 2, 2)),
        weights=np.full(4, 0.25), seed=3,
    )
    inst = _instance(
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=1e-6, alpha=0.95, scenarios=scenarios),
    )
    enc = encode(inst, BinaryEncoding())
    zeta = enc.gadget.zeta.snap(optimal_zeta(inst, inst.witness))
    sample = decode(enc, encode_solution(enc, inst.witness, zeta=zeta))
    assert sample.zeta == pytest.approx(zeta, rel=1e-12)
    assert sample.classification == "feasible"


def test_a_variance_sample_reports_no_zeta():
    enc = encode(_instance(), BinaryEncoding())
    assert decode(enc, encode_solution(enc, enc.instance.witness)).zeta is None


def test_decoding_a_wrong_length_vector_is_refused():
    enc = encode(_instance(), BinaryEncoding())
    with pytest.raises(ValueError, match="variables"):
        decode(enc, np.zeros(enc.n_vars - 1, dtype=np.int64))
