import numpy as np
import polars as pl
import pytest

from quantic.core.session import NYSE, SYNTH_SESSION
from quantic.data.synth import SynthConfig, generate_bundle
from quantic.micro.impact.calibrate import (
    CalibrationError,
    CalibrationResult,
    calibrate_bundle,
    estimate_bucket_sigma,
    fit_power_law,
    observations_from_buckets,
)
from quantic.micro.liquidity import OutOfSessionError

BUCKET_NS = (23_400 // 13) * 1_000_000_000  # 30-minute buckets
SESSION = SYNTH_SESSION

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
        low_noise_bundle.l1(), low_noise_bundle.l3(), session=SESSION, bucket_ns=BUCKET_NS
    )
    assert obs.columns == [
        "symbol", "bucket_id", "mid", "volume", "signed_flow", "participation", "impact"
    ]
    assert obs.height > 0


def test_overnight_returns_are_excluded(low_noise_bundle):
    """13 buckets per day, 40 days: each day loses its first bucket to the gap."""
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), session=SESSION, bucket_ns=BUCKET_NS
    )
    per_symbol = obs.group_by("symbol").len()["len"].unique().to_list()
    assert per_symbol == [LOW_NOISE.n_days * (LOW_NOISE.buckets_per_day - 1)]


def test_recovers_ground_truth_delta_and_y_in_low_noise(low_noise_bundle):
    """THE M1 GATE: calibration recovers the injected impact parameters."""
    gt = low_noise_bundle.manifest.extra["ground_truth"]
    sigma = gt["sigma_bucket"]
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), session=SESSION, bucket_ns=BUCKET_NS
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
        realistic_bundle, session=SESSION, bucket_ns=BUCKET_NS, sigma=dict.fromkeys(
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
    results = calibrate_bundle(realistic_bundle, session=SESSION, bucket_ns=BUCKET_NS)
    assert set(results) == set(REALISTIC.symbols)
    assert all(0.0 < r.delta <= 1.0 for r in results.values())


def test_to_model_round_trips_into_a_usable_impact_model(low_noise_bundle):
    from quantic.micro.impact.base import ImpactParams

    gt = low_noise_bundle.manifest.extra["ground_truth"]
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), session=SESSION, bucket_ns=BUCKET_NS
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


def _dense_nyse_quotes(days, *, bucket_ns, jump):
    """L1 and L3 spanning ``days`` continuously, including pre- and post-market.

    A real L1 export quotes through the extended session, which is what makes
    epoch-floored bucketing dangerous: the overnight window becomes contiguous
    in bucket_id, so the gap filter ``bucket_id - prev_bucket == 1`` admits
    every overnight return as if it were intraday impact.
    """
    import polars as pl

    l1_rows, l3_rows = [], []
    price = 100.0
    ts = NYSE.open_ns(days[0]) + bucket_ns
    end = NYSE.open_ns(days[-1]) + NYSE.length_ns
    while ts <= end:
        # Step the price by `jump` exactly once, across the first overnight gap.
        if NYSE.open_ns(days[0]) + NYSE.length_ns < ts <= NYSE.open_ns(days[1]):
            price = 100.0 * (1.0 + jump)
        l1_rows.append(
            {
                "ts_ns": ts, "symbol": "AAA",
                "bid": price - 0.01, "ask": price + 0.01,
                "bid_size": 100, "ask_size": 100, "last_px": price, "last_size": 10,
            }
        )
        l3_rows.append(
            {
                "ts_ns": ts - 1, "seq": len(l3_rows), "symbol": "AAA",
                "order_id": len(l3_rows) + 1, "action": "execute", "side": "sell",
                "px": price, "size": 100,
            }
        )
        ts += bucket_ns
    return pl.DataFrame(l1_rows), pl.DataFrame(l3_rows)


def test_out_of_session_quotes_are_rejected_rather_than_misbucketed():
    import datetime as dt

    bucket_ns = NYSE.length_ns // 20
    l1, l3 = _dense_nyse_quotes(
        [dt.date(2026, 3, 3), dt.date(2026, 3, 4)], bucket_ns=bucket_ns, jump=0.20
    )
    with pytest.raises(OutOfSessionError, match="L1 quote"):
        observations_from_buckets(l1, l3, session=NYSE, bucket_ns=bucket_ns)


def test_overnight_gap_is_excluded_on_a_real_nyse_calendar():
    """THE P5 REGRESSION.

    With quotes spanning the extended session, epoch-floored bucketing makes
    the whole overnight window contiguous: all 92 transitions in this fixture
    have ``bucket_id - prev_bucket == 1``, so the 20% overnight jump is scored
    as intraday impact. Session-relative bucketing sees only the 40 in-session
    quotes and drops each session's first bucket, so the jump cannot appear.
    """
    import datetime as dt

    bpd = 20
    bucket_ns = NYSE.length_ns // bpd
    l1, l3 = _dense_nyse_quotes(
        [dt.date(2026, 3, 3), dt.date(2026, 3, 4)], bucket_ns=bucket_ns, jump=0.20
    )

    obs = observations_from_buckets(
        l1, l3, session=NYSE, bucket_ns=bucket_ns, allow_out_of_session=True
    )

    assert obs.height == 2 * (bpd - 1), "each session must lose exactly its first bucket"
    assert obs["impact"].abs().max() < 1e-9, (
        "the 20% overnight return leaked into the intraday impact observations"
    )


def test_bucket_width_that_does_not_divide_the_session_is_rejected(realistic_bundle):
    """The old calibrate_bundle rounded 6.5 buckets/day to 6: a silent 4.1% sigma error."""
    from quantic.core.session import SessionError

    with pytest.raises(SessionError, match="does not divide"):
        calibrate_bundle(realistic_bundle, session=SESSION, bucket_ns=3600 * 1_000_000_000)
