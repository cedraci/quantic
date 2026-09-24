"""The impact estimator, in isolation from the synthetic generator.

Review finding 2.3. The binned log-log fit dropped any bin whose mean signed
impact was non-positive, because it then took a logarithm. That conditions on
the outcome variable: surviving bins are biased upward and *which* bins
survive is draw-dependent, so the estimator is not consistent. Measured
consequence was that delta error did not fall monotonically with sample size.

Fitting ``mean_impact = Y * sigma * f**delta`` directly by non-linear least
squares needs no logarithm, therefore no positivity filter, therefore no
selection mechanism.
"""

import numpy as np
import polars as pl
import pytest

from quantic.micro.impact.calibrate import (
    CalibrationError,
    ParticipationBasis,
    fit_power_law,
)

BUCKET_NS = 1800 * 1_000_000_000
SIGMA = 0.005


def _observations(*, n, delta, y_coef, noise_sd, seed, flow_sd=0.08):
    """Observations drawn straight from the impact law, with no generator in the way."""
    rng = np.random.default_rng(seed)
    f = np.clip(rng.normal(0.0, flow_sd, n), -0.3, 0.3)
    f[np.abs(f) < 1e-4] = 1e-4
    impact = y_coef * SIGMA * np.sign(f) * np.abs(f) ** delta
    impact = impact + rng.normal(0.0, noise_sd, n)
    return pl.DataFrame(
        {
            "symbol": ["SYNA"] * n,
            "bucket_id": list(range(n)),
            "mid": [100.0] * n,
            "volume": [10_000] * n,
            "signed_flow": (f * 10_000).tolist(),
            "participation": f.tolist(),
            "impact": impact.tolist(),
        }
    )


def test_every_bin_is_used_no_matter_the_sign_of_its_mean_impact():
    """THE 2.3 REGRESSION: no bin is dropped for having negative mean impact.

    At this noise level several bins have negative mean signed impact. The old
    log-log fit silently discarded them, which is selection on the outcome
    variable. Direct least squares keeps all of them.
    """
    obs = _observations(n=240, delta=0.5, y_coef=0.8, noise_sd=SIGMA, seed=3)

    result = fit_power_law(obs, sigma=SIGMA, bucket_ns=BUCKET_NS, n_bins=6)

    # Confirm the fixture really does contain negative-mean bins, or the test
    # proves nothing.
    f = np.abs(obs["participation"].to_numpy())
    signed = obs["impact"].to_numpy() * np.sign(obs["participation"].to_numpy())
    order = np.argsort(f)
    means = [signed[g].mean() for g in np.array_split(order, 6)]
    assert any(m <= 0 for m in means), "fixture has no negative bin; it tests nothing"

    assert result.n_bins == 6


def test_delta_error_falls_with_sample_size():
    """Consistency. The filtered log-log estimator did not have this property."""

    def mean_error(n):
        errors = []
        for seed in range(12):
            obs = _observations(
                n=n, delta=0.5, y_coef=0.8, noise_sd=SIGMA * 0.25, seed=seed
            )
            errors.append(abs(fit_power_law(
                obs, sigma=SIGMA, bucket_ns=BUCKET_NS, n_bins=6
            ).delta - 0.5))
        return float(np.mean(errors))

    assert mean_error(4000) < mean_error(400)


def test_recovers_delta_and_y_from_a_clean_draw():
    obs = _observations(n=4000, delta=0.5, y_coef=0.8, noise_sd=SIGMA * 0.05, seed=11)
    result = fit_power_law(obs, sigma=SIGMA, bucket_ns=BUCKET_NS, n_bins=6)
    assert result.delta == pytest.approx(0.5, abs=0.05)
    assert result.y_coef == pytest.approx(0.8, rel=0.15)
    assert result.basis is ParticipationBasis.NET_IMBALANCE


def test_a_convex_fit_is_reported_rather_than_returned():
    """Finding I6 / section 4.2: delta=2.26 built a CalibrationResult happily.

    `to_model()` then raised, far from the fit that produced it. A delta
    outside PowerLawImpact's valid (0, 1] is a failed calibration and must be
    named where it happens.
    """
    obs = _observations(n=4000, delta=1.9, y_coef=0.8, noise_sd=SIGMA * 0.02, seed=5)
    with pytest.raises(CalibrationError, match=r"delta"):
        fit_power_law(obs, sigma=SIGMA, bucket_ns=BUCKET_NS, n_bins=6)


def test_impact_moving_against_flow_is_reported_rather_than_clamped():
    """A negative fitted Y means the data does not show impact. Say so."""
    obs = _observations(n=4000, delta=0.5, y_coef=-0.8, noise_sd=SIGMA * 0.05, seed=7)
    with pytest.raises(CalibrationError, match=r"y_coef|negative"):
        fit_power_law(obs, sigma=SIGMA, bucket_ns=BUCKET_NS, n_bins=6)
