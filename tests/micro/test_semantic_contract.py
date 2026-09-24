"""The M1 -> M2 contract: every quantity crossing the boundary names its units.

Review finding 2.2. Three volatility time bases lived on this branch with
none of them named in a type, ``bucket_volume`` was shares while the spec
defines participation in notional, and the calibrated ``Y`` was fitted on
aggregate order-flow imbalance while M2's objective evaluates our own
participation. Each is one keyword away from a wrong-but-plausible objective.
"""

import numpy as np
import polars as pl
import pytest

from quantic.micro.covariance import CovarianceEstimate, estimate_covariance
from quantic.micro.impact.base import ImpactParams
from quantic.micro.impact.calibrate import (
    CalibrationError,
    CalibrationResult,
    ParticipationBasis,
)
from quantic.micro.impact.sqrt_law import PowerLawImpact

BUCKET_NS = 1800 * 1_000_000_000


def _params(**kw):
    base = dict(
        symbol="SYNA",
        sigma_bucket=0.005,
        bucket_ns=BUCKET_NS,
        bucket_volume_shares=400_000.0,
        price=100.0,
    )
    base.update(kw)
    return ImpactParams(**base)


# --- 2.2a: the volatility time base is in the type -------------------------


def test_impact_params_names_its_volatility_time_base():
    p = _params()
    assert p.sigma_bucket == 0.005
    assert p.bucket_ns == BUCKET_NS
    assert not hasattr(p, "sigma"), "the unqualified name is what made this ambiguous"


def test_impact_params_rescales_volatility_by_square_root_of_time():
    p = _params()
    half = p.rescale_to(bucket_ns=BUCKET_NS // 2)
    assert half.bucket_ns == BUCKET_NS // 2
    assert half.sigma_bucket == pytest.approx(0.005 / np.sqrt(2.0))
    # Volume is a flow, so it scales linearly, not by sqrt.
    assert half.bucket_volume_shares == pytest.approx(200_000.0)


def test_covariance_estimate_names_its_horizon():
    daily = pl.DataFrame(
        {
            "date": pl.date_range(
                __import__("datetime").date(2026, 1, 5),
                __import__("datetime").date(2026, 3, 5),
                eager=True,
            ),
            "symbol": "AAA",
        }
    ).with_columns(
        (100.0 + pl.int_range(pl.len()).cast(pl.Float64) * 0.1).alias("close")
    )
    daily = daily.select("date", "symbol", "close")
    est = estimate_covariance(daily)
    assert est.horizon_days == 252.0


def test_covariance_can_be_rescaled_to_the_bucket_horizon():
    cov = CovarianceEstimate(
        symbols=("A", "B"),
        matrix=np.eye(2) * 0.04,  # 20% annual vol
        n_observations=100,
        shrinkage=0.0,
        horizon_days=252.0,
    )
    daily_cov = cov.at_horizon(days=1.0)
    assert daily_cov.horizon_days == 1.0
    assert daily_cov.matrix[0, 0] == pytest.approx(0.04 / 252.0)
    # Round trip.
    assert daily_cov.at_horizon(days=252.0).matrix[0, 0] == pytest.approx(0.04)


# --- 2.2b: the volume unit is pinned to shares -----------------------------


def test_bucket_volume_is_named_in_shares():
    p = _params()
    assert p.bucket_volume_shares == 400_000.0
    assert not hasattr(p, "bucket_volume")


def test_participation_is_computed_from_shares_by_the_type_itself():
    """A caller cannot mix a notional q with a share V if it never divides by hand."""
    p = _params()
    assert p.participation(40_000) == pytest.approx(0.1)
    assert p.participation(-40_000) == pytest.approx(0.1)


def test_notional_conversion_is_explicit():
    p = _params()
    assert p.notional(40_000) == pytest.approx(40_000 * 100.0)
    assert p.bucket_volume_notional == pytest.approx(400_000.0 * 100.0)


# --- 2.2c: the participation basis travels with the fit --------------------


def _result(basis):
    return CalibrationResult(
        symbol="SYNA", delta=0.5, y_coef=0.8, r_squared=0.99,
        n_observations=500, n_bins=6, sigma=0.005, bucket_ns=BUCKET_NS, basis=basis,
    )


def test_calibration_result_records_the_regressor_it_was_fitted_on():
    assert _result(ParticipationBasis.NET_IMBALANCE).basis is ParticipationBasis.NET_IMBALANCE


def test_to_model_refuses_to_reinterpret_an_imbalance_fit_silently():
    """A Y fitted on market imbalance is not a Y for our own participation."""
    with pytest.raises(CalibrationError, match="own participation|identification"):
        _result(ParticipationBasis.NET_IMBALANCE).to_model()


def test_the_identification_assumption_can_be_adopted_explicitly():
    model = _result(ParticipationBasis.NET_IMBALANCE).to_model(
        assume_own_participation=True
    )
    assert isinstance(model, PowerLawImpact)
    assert model.delta == 0.5


def test_a_fit_already_on_own_participation_needs_no_assumption():
    assert isinstance(
        _result(ParticipationBasis.OWN_PARTICIPATION).to_model(), PowerLawImpact
    )
