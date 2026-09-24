import numpy as np
import pytest

from quantic.core.session import SYNTH_SESSION, Boundary
from quantic.data.synth import SynthConfig, generate_bundle
from quantic.micro.impact.calibrate import CalibrationError
from quantic.problem.from_bundle import market_params_from_bundle
from quantic.problem.params import MarketParams

BUCKET_NS = SYNTH_SESSION.length_ns // 13

CFG = SynthConfig(
    symbols=("SYNA", "SYNB", "SYNC"), n_days=40, buckets_per_day=13,
    seed=17, base_price=1000.0, noise_frac=0.05, depth_levels=4,
)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return generate_bundle(tmp_path_factory.mktemp("fb") / "b", CFG)


def _build(bundle, **kw):
    base = dict(
        session=SYNTH_SESSION,
        bucket_ns=BUCKET_NS,
        spread_boundary=Boundary.CLOSING,   # the generator stamps L1 at bucket end
        assume_own_participation=True,
    )
    base.update(kw)
    return market_params_from_bundle(bundle, **base)


def test_it_builds_market_params_for_every_symbol(bundle):
    params = _build(bundle)
    assert isinstance(params, MarketParams)
    assert params.symbols == ("SYNA", "SYNB", "SYNC")
    assert params.bucket_ns == BUCKET_NS


def test_the_calibrated_exponent_reaches_the_params(bundle):
    """The M1-to-M2 contract, exercised end to end."""
    gt = bundle.manifest.extra["ground_truth"]
    params = _build(bundle)
    for a in params.assets:
        assert a.delta == pytest.approx(gt["impact_delta"][a.symbol], abs=0.05)


def test_the_covariance_is_rescaled_to_the_bucket_horizon(bundle):
    """estimate_covariance is annualised; the risk term needs per-bucket."""
    from quantic.micro.covariance import estimate_covariance

    annual = estimate_covariance(bundle.daily())
    params = _build(bundle)
    ratio = BUCKET_NS / SYNTH_SESSION.length_ns / 252.0
    assert np.allclose(params.covariance, annual.matrix * ratio, rtol=1e-9)


def test_the_covariance_rows_are_ordered_to_match_the_assets(bundle):
    """A mismatched ordering would misprice every cross-asset risk term."""
    params = _build(bundle, symbols=("SYNC", "SYNA"))
    assert params.symbols == ("SYNC", "SYNA")
    assert params.covariance.shape == (2, 2)
    # Diagonal entries must follow the requested order, not the bundle's.
    full = _build(bundle)
    idx = {s: i for i, s in enumerate(full.symbols)}
    assert params.covariance[0, 0] == pytest.approx(full.covariance[idx["SYNC"], idx["SYNC"]])
    assert params.covariance[1, 1] == pytest.approx(full.covariance[idx["SYNA"], idx["SYNA"]])


def test_half_spread_and_price_come_from_l1(bundle):
    params = _build(bundle)
    for a in params.assets:
        assert a.half_spread > 0.0
        assert a.price == pytest.approx(CFG.base_price, rel=0.5)


def test_bucket_volume_is_adv_divided_by_the_buckets_in_a_session(bundle):
    params = _build(bundle)
    for a in params.assets:
        assert a.bucket_volume_shares == pytest.approx(
            CFG.adv_shares / CFG.buckets_per_day, rel=0.4
        )


def test_gamma_defaults_to_zero_and_can_be_supplied(bundle):
    assert all(a.gamma == 0.0 for a in _build(bundle).assets)
    supplied = _build(bundle, gamma={"SYNA": 0.2, "SYNB": 0.3, "SYNC": 0.4})
    assert [a.gamma for a in supplied.assets] == [0.2, 0.3, 0.4]


def test_a_subset_of_symbols_can_be_requested(bundle):
    assert _build(bundle, symbols=("SYNB",)).symbols == ("SYNB",)


def test_an_unknown_symbol_is_named(bundle):
    with pytest.raises(KeyError, match="ZZZ"):
        _build(bundle, symbols=("ZZZ",))


def test_the_identification_assumption_is_not_assumed_by_default(bundle):
    """M2's objective evaluates own participation; the fit is on net imbalance.

    Defaulting this to True would bury the assumption at exactly the point a
    human should be making it.
    """
    with pytest.raises(CalibrationError, match="own participation|identification"):
        market_params_from_bundle(
            bundle, session=SYNTH_SESSION, bucket_ns=BUCKET_NS,
            spread_boundary=Boundary.CLOSING,
        )


def test_the_result_feeds_the_generator(bundle):
    """The point of the adapter: real parameters into the ladder."""
    from quantic.problem.dials import Dials
    from quantic.problem.feasibility import classify
    from quantic.problem.generator import generate
    from quantic.problem.liquidation import T0

    params = _build(bundle)
    inst = generate(T0, Dials(concave_impact=True), params, seed=0)
    assert classify(inst, inst.witness).feasible
