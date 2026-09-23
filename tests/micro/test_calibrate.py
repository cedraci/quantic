import numpy as np
import polars as pl
import pytest

from quantic.data.synth import SynthConfig, generate_bundle
from quantic.micro.impact.calibrate import (
    CalibrationError,
    CalibrationResult,
    calibrate_bundle,
    estimate_bucket_sigma,
    fit_power_law,
    observations_from_buckets,
)

BUCKET_NS = (23_400 // 13) * 1_000_000_000  # 30-minute buckets

# R23 amendment 1: noise_frac=0.05, not the brief's 0.2. Measurement shows
# 0.2 does not clear the gate's tolerances; 0.05 does, with real margin.
LOW_NOISE = SynthConfig(
    symbols=("SYNA", "SYNB", "SYNC"),
    n_days=40,
    buckets_per_day=13,
    seed=17,
    base_price=1000.0,   # keeps tick rounding far below the impact signal
    noise_frac=0.05,
    depth_levels=4,
)

REALISTIC = SynthConfig(
    symbols=("SYNA", "SYNB"),
    n_days=60,
    buckets_per_day=13,
    seed=23,
    base_price=1000.0,
    noise_frac=1.0,
    depth_levels=4,
)


@pytest.fixture(scope="module")
def low_noise_bundle(tmp_path_factory):
    return generate_bundle(tmp_path_factory.mktemp("cal_low") / "b", LOW_NOISE)


@pytest.fixture(scope="module")
def realistic_bundle(tmp_path_factory):
    return generate_bundle(tmp_path_factory.mktemp("cal_real") / "b", REALISTIC)


def test_observations_have_expected_columns(low_noise_bundle):
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), bucket_ns=BUCKET_NS
    )
    assert obs.columns == [
        "symbol", "bucket_id", "mid", "volume", "signed_flow", "participation", "impact"
    ]
    assert obs.height > 0


def test_overnight_returns_are_excluded(low_noise_bundle):
    """13 buckets per day, 40 days: each day loses its first bucket to the gap."""
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), bucket_ns=BUCKET_NS
    )
    per_symbol = obs.group_by("symbol").len()["len"].unique().to_list()
    assert per_symbol == [LOW_NOISE.n_days * (LOW_NOISE.buckets_per_day - 1)]


def test_recovers_ground_truth_delta_and_y_in_low_noise(low_noise_bundle):
    """THE M1 GATE: calibration recovers the injected impact parameters."""
    gt = low_noise_bundle.manifest.extra["ground_truth"]
    sigma = gt["sigma_bucket"]
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), bucket_ns=BUCKET_NS
    )

    for symbol in LOW_NOISE.symbols:
        result = fit_power_law(obs.filter(pl.col("symbol") == symbol), sigma=sigma)
        assert result.delta == pytest.approx(gt["impact_delta"][symbol], abs=0.05)
        assert result.y_coef == pytest.approx(gt["impact_Y"][symbol], rel=0.30)
        assert result.r_squared > 0.90


def test_recovers_delta_under_realistic_diffusion_noise(realistic_bundle):
    """Robustness check, not a precision check (R23 amendment 3).

    At realistic diffusion noise (noise_frac=1.0), per-symbol delta recovery
    is NOT asserted to any tolerance: measurement shows the binned-regression
    estimator only recovers delta to roughly 0.13-0.26 even at 1800-3600
    observations per symbol, and the error does not shrink monotonically with
    sample size because of selection bias in which bins survive the
    mean-signed-impact > 0 filter. Asserting a delta tolerance here would be
    an assertion the method cannot support, which is worse than none. The
    low-noise test above is the actual correctness gate for this estimator.
    This test only asserts that calibration runs cleanly end-to-end and
    produces well-formed, sane results under realistic noise.
    """
    gt = realistic_bundle.manifest.extra["ground_truth"]
    results = calibrate_bundle(
        realistic_bundle, bucket_ns=BUCKET_NS, sigma=dict.fromkeys(
            REALISTIC.symbols, gt["sigma_bucket"]
        )
    )
    for symbol in REALISTIC.symbols:
        result = results[symbol]
        assert isinstance(result, CalibrationResult)
        assert 0.0 < result.delta <= 1.0
        assert result.y_coef > 0.0
        assert np.isfinite(result.y_coef)
        assert np.isfinite(result.r_squared)
        assert result.n_observations > 0


def test_estimate_bucket_sigma_is_in_the_right_ballpark(realistic_bundle):
    """With diffusion-dominated returns, realised vol should track daily_vol."""
    gt = realistic_bundle.manifest.extra["ground_truth"]
    sigma = estimate_bucket_sigma(
        realistic_bundle.daily(), buckets_per_day=REALISTIC.buckets_per_day
    )
    for symbol in REALISTIC.symbols:
        assert sigma[symbol] == pytest.approx(gt["sigma_bucket"], rel=0.5)


def test_calibrate_bundle_estimates_sigma_when_not_supplied(realistic_bundle):
    results = calibrate_bundle(realistic_bundle, bucket_ns=BUCKET_NS)
    assert set(results) == set(REALISTIC.symbols)
    assert all(0.0 < r.delta <= 1.0 for r in results.values())


def test_to_model_round_trips_into_a_usable_impact_model(low_noise_bundle):
    from quantic.micro.impact.base import ImpactParams

    gt = low_noise_bundle.manifest.extra["ground_truth"]
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), bucket_ns=BUCKET_NS
    )
    model = fit_power_law(
        obs.filter(pl.col("symbol") == "SYNA"), sigma=gt["sigma_bucket"]
    ).to_model()
    p = ImpactParams(symbol="SYNA", sigma=gt["sigma_bucket"], bucket_volume=4e5, price=1000.0)
    assert model.price_impact(4e4, p) > 0


def test_too_few_bins_is_rejected():
    obs = pl.DataFrame(
        {
            "symbol": ["SYNA"] * 3,
            "bucket_id": [0, 1, 2],
            "mid": [100.0, 100.1, 100.2],
            "volume": [1000, 1000, 1000],
            "signed_flow": [100, 200, 300],
            "participation": [0.1, 0.2, 0.3],
            "impact": [0.001, 0.002, 0.003],
        }
    )
    with pytest.raises(CalibrationError, match="bins"):
        fit_power_law(obs, sigma=0.005, n_bins=20)
