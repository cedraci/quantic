import numpy as np
import pytest

from quantic.problem.constraints.block_trades import BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.generator import InstanceGenerationError, generate
from quantic.problem.liquidation import LADDER, T0, T1, T3
from quantic.problem.objective import evaluate
from quantic.problem.params import AssetParams, MarketParams

BUCKET_NS = 1800 * 1_000_000_000


def _params(n: int) -> MarketParams:
    rng = np.random.default_rng(0)
    assets = tuple(
        AssetParams(
            symbol=f"S{i:02d}", delta=0.5, y_coef=0.8, gamma=0.1,
            sigma_bucket=0.005, bucket_volume_shares=400_000.0,
            price=100.0 + i, half_spread=0.01,
        )
        for i in range(n)
    )
    a = rng.normal(size=(n, n))
    cov = (a @ a.T) / n * 1e-5           # PSD by construction
    return MarketParams(assets=assets, covariance=cov, bucket_ns=BUCKET_NS)


def _gen(tier, dials, seed=0, **kw):
    return generate(tier, dials, _params(tier.n_assets), seed=seed, **kw)


# --- the core promise ------------------------------------------------------


@pytest.mark.parametrize("tier", LADDER, ids=lambda t: t.name)
@pytest.mark.parametrize("dials", Dials.combinations(), ids=lambda d: d.label)
def test_the_witness_is_feasible_for_every_tier_and_dial_combination(tier, dials):
    """THE M2a PROMISE. If this fails, the done-criterion is unreachable."""
    inst = _gen(tier, dials)
    report = classify(inst, inst.witness)
    assert report.feasible, f"{tier.name}/{dials.label}: {dict(report.violations)}"


@pytest.mark.parametrize("tier", LADDER, ids=lambda t: t.name)
@pytest.mark.parametrize("dials", Dials.combinations(), ids=lambda d: d.label)
def test_every_generated_instance_evaluates_finitely(tier, dials):
    inst = _gen(tier, dials)
    assert np.isfinite(evaluate(inst, inst.witness).total)


def test_the_witness_fully_liquidates():
    inst = _gen(T1, Dials())
    assert [inst.witness.sold(i) for i in range(inst.n_assets)] == list(inst.initial_lots)


def test_no_witness_cell_exceeds_the_tier_range():
    inst = _gen(T3, Dials(discrete_participation=True))
    assert inst.witness.as_array().max() <= T3.max_lots_per_bucket


# --- dial wiring -----------------------------------------------------------


def test_full_liquidation_is_present_under_every_dial_combination():
    for dials in Dials.combinations():
        inst = _gen(T0, dials)
        assert any(isinstance(c, FullLiquidation) for c in inst.constraints)


def test_d2_off_adds_neither_cardinality_nor_minimum_lot():
    inst = _gen(T1, Dials())
    assert not any(isinstance(c, Cardinality | MinParticipation) for c in inst.constraints)


def test_d2_on_adds_both_constraints_together():
    """Spec 4: cardinality without a minimum lot is not meaningful on a desk."""
    inst = _gen(T1, Dials(discrete_participation=True))
    assert any(isinstance(c, Cardinality) for c in inst.constraints)
    assert any(isinstance(c, MinParticipation) for c in inst.constraints)


def test_d4_on_adds_block_trades():
    inst = _gen(T1, Dials(block_trades=True))
    assert any(isinstance(c, BlockTrades) for c in inst.constraints)


def test_d4_carves_at_least_one_block():
    inst = _gen(T1, Dials(block_trades=True))
    blocks = next(c for c in inst.constraints if isinstance(c, BlockTrades))
    assert len(blocks.blocks) >= 1


def test_d1_changes_no_constraint_only_the_objective():
    """D1 lives in the objective; a constraint difference would confound the design."""
    off = _gen(T1, Dials())
    on = _gen(T1, Dials(concave_impact=True))
    assert [c.describe() for c in off.constraints] == [c.describe() for c in on.constraints]


def test_the_dials_are_recorded_on_the_instance():
    d = Dials(concave_impact=True, block_trades=True)
    assert _gen(T1, d).dials == d


# --- the binding tests: a dial that does not bind is decoration -------------


@pytest.mark.parametrize("tier", LADDER, ids=lambda t: t.name)
def test_cardinality_binds_strictly_below_the_asset_count(tier):
    """k == n_assets constrains nothing, so D2 would contribute no hardness
    while still appearing in every results row as though it did."""
    inst = _gen(tier, Dials(discrete_participation=True))
    card = next(c for c in inst.constraints if isinstance(c, Cardinality))
    assert 1 <= card.k < tier.n_assets


@pytest.mark.parametrize("tier", LADDER, ids=lambda t: t.name)
def test_cardinality_is_tight_against_the_witness(tier):
    """Tightest k the witness satisfies: one lower and the witness would break."""
    inst = _gen(tier, Dials(discrete_participation=True))
    card = next(c for c in inst.constraints if isinstance(c, Cardinality))
    busiest = max(inst.witness.active_names(t) for t in range(inst.n_buckets))
    assert card.k == busiest
    assert Cardinality(k=card.k - 1).violation(inst.witness, inst.initial_lots) > 0


@pytest.mark.parametrize("tier", LADDER, ids=lambda t: t.name)
def test_the_minimum_lot_is_positive_and_tight(tier):
    inst = _gen(tier, Dials(discrete_participation=True))
    mp = next(c for c in inst.constraints if isinstance(c, MinParticipation))
    grid = inst.witness.as_array()
    for i, m in enumerate(mp.min_lots):
        assert m >= 1
        nonzero = grid[i][grid[i] > 0]
        if nonzero.size:
            assert m == int(nonzero.min())


# --- risk scaling ----------------------------------------------------------


def test_lam_is_auto_scaled_so_risk_balances_cost_at_the_witness():
    """An unscaled lam makes one side of the trade-off vanish, and a problem
    where risk or cost is irrelevant is not the problem being studied."""
    inst = _gen(T1, Dials())
    b = evaluate(inst, inst.witness)
    costs = b.spread_cost + b.temporary_impact_cost + b.permanent_impact_cost
    assert b.risk_cost == pytest.approx(costs, rel=1e-6)


def test_an_explicit_lam_is_respected():
    inst = _gen(T1, Dials(), lam=3e-7)
    assert inst.risk.lam == 3e-7


# --- determinism -----------------------------------------------------------


def test_the_same_seed_gives_an_identical_content_hash():
    assert _gen(T1, Dials(), seed=7).content_hash == _gen(T1, Dials(), seed=7).content_hash


def test_a_different_seed_gives_a_different_instance():
    assert _gen(T1, Dials(), seed=7).content_hash != _gen(T1, Dials(), seed=8).content_hash


def test_the_seed_is_recorded_on_the_instance():
    assert _gen(T1, Dials(), seed=11).seed == 11


def test_different_dials_give_different_instances_at_the_same_seed():
    a = _gen(T1, Dials(), seed=3)
    b = _gen(T1, Dials(discrete_participation=True), seed=3)
    assert a.content_hash != b.content_hash


# --- rejections ------------------------------------------------------------


def test_too_few_assets_in_params_is_rejected():
    with pytest.raises(InstanceGenerationError, match="30 assets|asset"):
        generate(T3, Dials(), _params(5), seed=0)


def test_surplus_assets_in_params_are_narrowed_to_the_tier():
    inst = generate(T0, Dials(), _params(10), seed=0)
    assert inst.n_assets == T0.n_assets
    assert inst.params.n_assets == T0.n_assets
