import json

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.diagnostics import (
    DENSE_EMBEDDING_CEILING,
    dynamic_range_ratio,
    variable_budget,
    variable_count,
)
from quantic.encoding.encode import encode
from quantic.encoding.integer import ENCODINGS, BinaryEncoding, OneHotEncoding
from quantic.encoding.qubo import QuboBuilder
from quantic.encoding.registry import RegistryBuilder, VarRegistry
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
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


def _cvar_instance(n=6):
    rng = np.random.default_rng(11)
    scenarios = ScenarioSet(
        returns=rng.normal(scale=0.01, size=(n, 2, 2)),
        weights=np.full(n, 1.0 / n), seed=11,
    )
    return _instance(
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=1e-6, alpha=0.95, scenarios=scenarios),
    )


def test_the_ratio_is_the_largest_coefficient_over_the_smallest_non_zero_one():
    b = RegistryBuilder()
    b.add(("a",))
    b.add(("b",))
    builder = QuboBuilder(b.freeze())
    builder.add_linear(("a",), 8.0)
    builder.add_quadratic(("a",), ("b",), 0.5)
    assert dynamic_range_ratio(builder.build()) == pytest.approx(16.0)


def test_an_all_zero_qubo_has_no_meaningful_ratio():
    reg = VarRegistry(keys=(("a",),))
    assert dynamic_range_ratio(QuboBuilder(reg).build()) == 0.0


def test_the_constant_does_not_enter_the_ratio():
    """A constant shifts every energy equally; no coupler has to resolve it."""
    reg = VarRegistry(keys=(("a",),))
    builder = QuboBuilder(reg)
    builder.add_linear(("a",), 2.0)
    builder.add_constant(1e12)
    assert dynamic_range_ratio(builder.build()) == pytest.approx(1.0)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_count_agrees_with_what_encode_actually_builds(encoding):
    for inst in (_instance(), _cvar_instance()):
        assert variable_count(inst, encoding) == encode(inst, encoding).n_vars


def test_one_hot_costs_more_variables_than_the_log_encoding():
    inst = _instance()
    assert variable_count(inst, OneHotEncoding()) > variable_count(inst, BinaryEncoding())


def test_cvar_auxiliaries_dominate_the_count():
    """Why scenario reduction is mandatory rather than an optimisation."""
    plain = variable_count(_instance(), BinaryEncoding())
    cvar = variable_count(_cvar_instance(n=20), BinaryEncoding())
    assert cvar > 5 * plain


def test_the_budget_rows_flag_what_fits_a_dense_embedding():
    rows = variable_budget([_instance(), _cvar_instance(n=20)])
    assert len(rows) == 2 * len(ENCODINGS)
    small = next(r for r in rows if r.dials == "none" and r.encoding == "binary")
    assert small.fits_dense_embedding is (small.n_vars <= DENSE_EMBEDDING_CEILING)
    assert any(not r.fits_dense_embedding for r in rows)


def test_the_budget_is_json_safe_for_the_results_row():
    rows = variable_budget([_instance()])
    payload = json.loads(json.dumps([r.describe() for r in rows]))
    assert payload[0]["tier"] == "T0"


def test_counting_stays_cheap_on_a_layout_no_matrix_could_hold():
    """A dense T3-sized one-hot term is hundreds of megabytes; counting keys is free.

    The instance below names more bits than a dense matrix of that size would
    fit in memory, so the fact that this returns at all is the assertion.
    """
    wide = _instance(
        assets=tuple(
            Asset(f"S{i:02d}", lot_size=100, tick_size=0.01) for i in range(30)
        ),
        n_buckets=12,
        initial_lots=(15,) * 30,
        max_lots_per_bucket=15,
        params=MarketParams(
            assets=tuple(
                AssetParams(f"S{i:02d}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01)
                for i in range(30)
            ),
            covariance=np.eye(30) * 1e-5,
            bucket_ns=BUCKET_NS,
        ),
        witness=Schedule(lots=tuple((15,) + (0,) * 11 for _ in range(30))),
    )
    assert variable_count(wide, OneHotEncoding()) == 30 * 12 * 16
