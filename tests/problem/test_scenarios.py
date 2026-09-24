import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.scenarios import (
    ReductionReport,
    ScenarioSet,
    generate_scenarios,
    reduce_scenarios,
)
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(n_assets: int = 2, n_buckets: int = 2) -> Instance:
    assets = tuple(
        Asset(f"S{i}", lot_size=100, tick_size=0.01) for i in range(n_assets)
    )
    params = MarketParams(
        assets=tuple(
            AssetParams(f"S{i}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01)
            for i in range(n_assets)
        ),
        covariance=np.eye(n_assets) * 1e-4,
        bucket_ns=BUCKET_NS,
    )
    lots = tuple(tuple(2 for _ in range(n_buckets)) for _ in range(n_assets))
    return Instance(
        instance_id="x", tier="T0", assets=assets, n_buckets=n_buckets,
        initial_lots=tuple(2 * n_buckets for _ in range(n_assets)),
        max_lots_per_bucket=2, params=params, dials=Dials(),
        constraints=(FullLiquidation(),), risk=VarianceRisk(lam=1e-6),
        witness=Schedule(lots=lots), seed=0,
    )


def _scenarios(s=4, t=2, n=2, seed=0) -> ScenarioSet:
    returns = np.zeros((s, t, n))
    return ScenarioSet(returns=returns, weights=np.full(s, 1.0 / s), seed=seed)


def test_shape_accessors_read_the_returns_array():
    ss = _scenarios(s=7, t=3, n=5)
    assert (ss.n_scenarios, ss.n_buckets, ss.n_assets) == (7, 3, 5)


def test_a_zero_shock_scenario_produces_zero_loss():
    inst = _instance()
    losses = _scenarios(t=inst.n_buckets, n=inst.n_assets).losses(inst, inst.witness)
    assert losses.shape == (4,)
    assert np.allclose(losses, 0.0)


def test_a_negative_return_on_a_held_position_is_a_positive_loss():
    """The sign convention: we are long and liquidating, so a price fall hurts."""
    inst = _instance()
    returns = np.zeros((1, inst.n_buckets, inst.n_assets))
    returns[0, 0, 0] = -0.01                      # asset 0 falls 1% in bucket 0
    ss = ScenarioSet(returns=returns, weights=np.array([1.0]), seed=0)
    # After bucket 0, asset 0 still holds 2 lots x 100 shares at $100 = $20,000.
    assert ss.losses(inst, inst.witness)[0] == pytest.approx(200.0)


def test_loss_is_linear_in_the_shock():
    inst = _instance()
    base = np.zeros((1, inst.n_buckets, inst.n_assets))
    base[0, 0, 0] = -0.01
    one = ScenarioSet(returns=base, weights=np.array([1.0]), seed=0)
    two = ScenarioSet(returns=base * 2.0, weights=np.array([1.0]), seed=0)
    assert two.losses(inst, inst.witness)[0] == pytest.approx(
        2.0 * one.losses(inst, inst.witness)[0]
    )


def test_a_fully_liquidated_final_bucket_contributes_no_loss():
    """h is the holding AFTER each bucket, so the last one is zero."""
    inst = _instance()
    returns = np.zeros((1, inst.n_buckets, inst.n_assets))
    returns[0, -1, :] = -0.5                      # a huge shock in the last bucket
    ss = ScenarioSet(returns=returns, weights=np.array([1.0]), seed=0)
    assert ss.losses(inst, inst.witness)[0] == pytest.approx(0.0)


def test_losses_reject_a_scenario_set_of_the_wrong_shape():
    inst = _instance()
    with pytest.raises(ValueError, match="shape|buckets|assets"):
        _scenarios(t=99, n=inst.n_assets).losses(inst, inst.witness)


# --- validation -------------------------------------------------------------


def test_weights_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        ScenarioSet(returns=np.zeros((2, 1, 1)), weights=np.array([0.3, 0.3]), seed=0)


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ScenarioSet(returns=np.zeros((2, 1, 1)), weights=np.array([1.5, -0.5]), seed=0)


def test_a_weight_vector_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="weights"):
        ScenarioSet(returns=np.zeros((3, 1, 1)), weights=np.array([0.5, 0.5]), seed=0)


def test_non_finite_returns_are_rejected():
    bad = np.zeros((2, 1, 1))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ScenarioSet(returns=bad, weights=np.full(2, 0.5), seed=0)


def test_a_returns_array_that_is_not_three_dimensional_is_rejected():
    with pytest.raises(ValueError, match="3-D|three"):
        ScenarioSet(returns=np.zeros((2, 2)), weights=np.full(2, 0.5), seed=0)


def test_an_empty_scenario_set_is_rejected():
    with pytest.raises(ValueError, match="at least one scenario"):
        ScenarioSet(returns=np.zeros((0, 1, 1)), weights=np.zeros(0), seed=0)


# --- equality and hashing, per the M2a array-dataclass rule -----------------


def test_identical_scenario_sets_compare_and_hash_equal():
    """A frozen dataclass holding an ndarray needs explicit eq/hash.

    The generated __eq__ raises ValueError on a 2+ element array; the
    generated __hash__ raises TypeError. MarketParams and Instance were fixed
    this way in M2a and this type must follow.
    """
    a, b = _scenarios(s=5), _scenarios(s=5)
    assert a == b
    assert hash(a) == hash(b)


def test_scenario_sets_differing_in_returns_compare_unequal():
    a = _scenarios(s=3)
    other = np.zeros((3, 2, 2))
    other[0, 0, 0] = 0.01
    b = ScenarioSet(returns=other, weights=np.full(3, 1 / 3), seed=0)
    assert a != b


def test_scenario_sets_differing_in_seed_compare_unequal():
    assert _scenarios(seed=1) != _scenarios(seed=2)


def test_a_scenario_set_compares_false_against_an_unrelated_object():
    assert _scenarios() != "not a scenario set"


def test_scenario_sets_deduplicate_in_a_set():
    assert len({_scenarios(s=5), _scenarios(s=5)}) == 1


def test_dataclasses_replace_works_on_a_scenario_set():
    ss = _scenarios()
    report = ReductionReport(n_raw=10, n_reduced=4, wasserstein=0.5)
    assert dataclasses.replace(ss, reduction=report).reduction is report


# --- ReductionReport --------------------------------------------------------


def test_reduction_report_leaves_the_cvar_fields_unset_until_a_witness_exists():
    r = ReductionReport(n_raw=2000, n_reduced=25, wasserstein=0.4)
    assert (r.cvar_raw, r.cvar_reduced, r.cvar_discrepancy) == (None, None, None)


def test_reduction_report_rejects_a_reduction_that_does_not_reduce():
    with pytest.raises(ValueError, match="n_reduced"):
        ReductionReport(n_raw=10, n_reduced=10, wasserstein=0.0)


def test_reduction_report_rejects_a_negative_distance():
    with pytest.raises(ValueError, match="wasserstein"):
        ReductionReport(n_raw=10, n_reduced=4, wasserstein=-1.0)


def _params(n: int, scale: float = 1e-4) -> MarketParams:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(n, n))
    return MarketParams(
        assets=tuple(
            AssetParams(f"S{i}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01)
            for i in range(n)
        ),
        covariance=(a @ a.T) / n * scale,
        bucket_ns=BUCKET_NS,
    )


def test_generated_scenarios_have_the_requested_shape():
    ss = generate_scenarios(_params(3), n_buckets=4, seed=1, n_raw=500)
    assert (ss.n_scenarios, ss.n_buckets, ss.n_assets) == (500, 4, 3)


def test_generated_weights_are_uniform_and_sum_to_one():
    ss = generate_scenarios(_params(2), n_buckets=3, seed=1, n_raw=250)
    assert np.allclose(ss.weights, 1 / 250)
    assert ss.weights.sum() == pytest.approx(1.0)


def test_an_unreduced_set_carries_no_reduction_report():
    assert generate_scenarios(_params(2), n_buckets=2, seed=1, n_raw=50).reduction is None


def test_the_sample_covariance_matches_the_parameter_covariance():
    """The scenario machinery must describe the same risk the variance term does.

    params.covariance is already at the bucket horizon, so generation must not
    rescale it -- a second rescale is exactly the class of silent error the
    units discipline exists to prevent.
    """
    params = _params(3)
    ss = generate_scenarios(params, n_buckets=4, seed=5, n_raw=40_000)
    flat = ss.returns.reshape(-1, 3)
    empirical = np.cov(flat, rowvar=False)
    assert np.allclose(empirical, params.covariance, rtol=0.10, atol=1e-8)


def test_shocks_are_independent_across_buckets():
    """Independence across buckets is what makes Var(L) equal the variance term."""
    ss = generate_scenarios(_params(2), n_buckets=2, seed=9, n_raw=40_000)
    first = ss.returns[:, 0, 0]
    second = ss.returns[:, 1, 0]
    assert abs(float(np.corrcoef(first, second)[0, 1])) < 0.05


def test_generation_is_deterministic_in_its_seed():
    a = generate_scenarios(_params(2), n_buckets=3, seed=7, n_raw=100)
    b = generate_scenarios(_params(2), n_buckets=3, seed=7, n_raw=100)
    assert a == b


def test_a_different_seed_gives_different_scenarios():
    a = generate_scenarios(_params(2), n_buckets=3, seed=7, n_raw=100)
    b = generate_scenarios(_params(2), n_buckets=3, seed=8, n_raw=100)
    assert a != b


def test_the_seed_is_recorded_on_the_set():
    assert generate_scenarios(_params(2), n_buckets=2, seed=11, n_raw=10).seed == 11


def test_generation_emits_no_warnings():
    """Test output must be pristine; a PSD-boundary covariance must not warn."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generate_scenarios(_params(3), n_buckets=3, seed=1, n_raw=200)


@pytest.mark.parametrize("bad,match", [(0, "n_raw"), (-5, "n_raw")])
def test_a_non_positive_scenario_count_is_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        generate_scenarios(_params(2), n_buckets=2, seed=1, n_raw=bad)


def test_a_non_positive_bucket_count_is_rejected():
    with pytest.raises(ValueError, match="n_buckets"):
        generate_scenarios(_params(2), n_buckets=0, seed=1, n_raw=10)


def test_reduction_keeps_the_requested_number_of_scenarios():
    raw = generate_scenarios(_params(2), n_buckets=3, seed=1, n_raw=200)
    assert reduce_scenarios(raw, n_target=10).n_scenarios == 10


def test_reduced_weights_are_a_probability_distribution():
    raw = generate_scenarios(_params(2), n_buckets=3, seed=1, n_raw=200)
    reduced = reduce_scenarios(raw, n_target=10)
    assert reduced.weights.sum() == pytest.approx(1.0)
    assert (reduced.weights >= 0).all()


def test_every_representative_is_an_actual_raw_scenario():
    """Not a centroid. CVaR is a tail statistic and averaging thins the tail.

    A centroid path is smoother than any real path, so a reduced set built
    from centroids would systematically understate the very quantity CVaR
    measures.
    """
    raw = generate_scenarios(_params(2), n_buckets=3, seed=1, n_raw=200)
    reduced = reduce_scenarios(raw, n_target=8)
    for kept in reduced.returns:
        assert any(np.array_equal(kept, original) for original in raw.returns)


def test_the_reduction_report_records_the_counts_and_the_distance():
    raw = generate_scenarios(_params(2), n_buckets=3, seed=1, n_raw=200)
    report = reduce_scenarios(raw, n_target=12).reduction
    assert report is not None
    assert (report.n_raw, report.n_reduced) == (200, 12)
    assert report.wasserstein > 0.0
    assert report.cvar_raw is None      # needs a witness; the generator fills these


def test_wasserstein_distance_falls_as_more_scenarios_are_kept():
    """The reduction error must be a real measurement, not a decoration."""
    raw = generate_scenarios(_params(2), n_buckets=3, seed=4, n_raw=300)
    distances = [
        reduce_scenarios(raw, n_target=n).reduction.wasserstein for n in (5, 15, 40)
    ]
    assert distances[0] > distances[1] > distances[2]


def test_reduction_is_deterministic():
    raw = generate_scenarios(_params(2), n_buckets=3, seed=1, n_raw=200)
    assert reduce_scenarios(raw, n_target=10) == reduce_scenarios(raw, n_target=10)


def test_the_seed_is_carried_through_reduction():
    raw = generate_scenarios(_params(2), n_buckets=3, seed=17, n_raw=100)
    assert reduce_scenarios(raw, n_target=5).seed == 17


def test_reducing_to_at_least_the_original_count_is_rejected():
    raw = generate_scenarios(_params(2), n_buckets=2, seed=1, n_raw=50)
    with pytest.raises(ValueError, match="n_target"):
        reduce_scenarios(raw, n_target=50)


def test_a_non_positive_target_is_rejected():
    raw = generate_scenarios(_params(2), n_buckets=2, seed=1, n_raw=50)
    with pytest.raises(ValueError, match="n_target"):
        reduce_scenarios(raw, n_target=0)


def test_reducing_an_already_reduced_set_is_rejected():
    """Reducing twice would report a Wasserstein distance against the wrong base."""
    raw = generate_scenarios(_params(2), n_buckets=2, seed=1, n_raw=60)
    once = reduce_scenarios(raw, n_target=20)
    with pytest.raises(ValueError, match="already reduced"):
        reduce_scenarios(once, n_target=5)
