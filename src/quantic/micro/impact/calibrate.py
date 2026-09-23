"""Calibrate the power-law impact exponent from bucket-level order flow.

Binned log-log regression (Almgren et al.; Toth et al.): per-bucket impact is
dominated by diffusion, so observations are grouped into equal-count bins by
absolute participation and the *signed* impact is averaged within each bin,
which cancels the noise. Averaging absolute impact instead would bias Y upward.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl

from quantic.data.bundle import DatasetBundle
from quantic.micro.covariance import log_returns
from quantic.micro.impact.sqrt_law import PowerLawImpact
from quantic.micro.liquidity import signed_order_flow

MIN_BINS = 4

# R23 amendment 2: 6 bins, not 20, is the default for both fit_power_law and
# calibrate_bundle. Chosen empirically: with more bins each bin holds too few
# observations for the mean-signed-impact averaging to cancel diffusion
# noise, and the fit degrades (delta error becomes larger and erratic).
DEFAULT_N_BINS = 6


class CalibrationError(ValueError):
    """Raised when there is not enough signal to fit an impact law."""


@dataclass(frozen=True)
class CalibrationResult:
    symbol: str
    delta: float
    y_coef: float
    r_squared: float
    n_observations: int
    n_bins: int

    def to_model(self, gamma: float = 0.0) -> PowerLawImpact:
        return PowerLawImpact(delta=self.delta, y_coef=self.y_coef, gamma=gamma)


def observations_from_buckets(
    l1: pl.DataFrame, l3: pl.DataFrame, *, bucket_ns: int
) -> pl.DataFrame:
    """Join bucket-level signed flow to the mid return realised over that bucket."""
    # A quote stamped exactly on a boundary belongs to the closing bucket.
    mids = (
        l1.with_columns(
            ((pl.col("ts_ns") - 1) // bucket_ns).alias("bucket_id"),
            ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
        )
        .sort(["symbol", "bucket_id", "ts_ns"])
        .group_by(["symbol", "bucket_id"])
        .agg(pl.col("mid").last())
        .sort(["symbol", "bucket_id"])
        .with_columns(
            pl.col("mid").shift(1).over("symbol").alias("prev_mid"),
            pl.col("bucket_id").shift(1).over("symbol").alias("prev_bucket"),
        )
        # Drop gaps, which are overnight returns rather than intraday impact.
        .filter(pl.col("bucket_id") - pl.col("prev_bucket") == 1)
        .with_columns((pl.col("mid") / pl.col("prev_mid") - 1.0).alias("impact"))
    )

    flow = signed_order_flow(l3, bucket_ns=bucket_ns)

    return (
        mids.join(flow, on=["symbol", "bucket_id"], how="inner")
        .select(
            "symbol", "bucket_id", "mid", "volume", "signed_flow", "participation", "impact"
        )
        .sort(["symbol", "bucket_id"])
    )


def fit_power_law(
    observations: pl.DataFrame,
    *,
    sigma: float,
    n_bins: int = DEFAULT_N_BINS,
    min_participation: float = 1e-4,
) -> CalibrationResult:
    if sigma <= 0:
        raise CalibrationError(f"sigma must be positive, got {sigma}")

    symbols = observations["symbol"].unique().to_list()
    if len(symbols) != 1:
        raise CalibrationError(f"fit one symbol at a time, got {sorted(symbols)}")

    usable = observations.filter(pl.col("participation").abs() >= min_participation)
    if usable.height < n_bins * 2:
        raise CalibrationError(
            f"need at least {n_bins * 2} observations for {n_bins} bins, got {usable.height}"
        )

    abs_f = usable["participation"].abs().to_numpy()
    signed_impact = (
        usable["impact"].to_numpy() * np.sign(usable["participation"].to_numpy())
    )

    order = np.argsort(abs_f)
    groups = [g for g in np.array_split(order, n_bins) if g.size > 0]

    x_vals: list[float] = []
    y_vals: list[float] = []
    for g in groups:
        mean_f = float(abs_f[g].mean())
        mean_impact = float(signed_impact[g].mean())
        if mean_f <= 0 or mean_impact <= 0:
            continue  # a bin whose mean impact is negative carries no power-law signal
        x_vals.append(np.log(mean_f))
        y_vals.append(np.log(mean_impact / sigma))

    if len(x_vals) < MIN_BINS:
        raise CalibrationError(
            f"only {len(x_vals)} usable bins after filtering; need at least {MIN_BINS}"
        )

    x = np.asarray(x_vals)
    y = np.asarray(y_vals)
    slope, intercept = np.polyfit(x, y, 1)

    residuals = y - (slope * x + intercept)
    ss_res = float((residuals**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return CalibrationResult(
        symbol=symbols[0],
        delta=float(slope),
        y_coef=float(np.exp(intercept)),
        r_squared=r_squared,
        n_observations=usable.height,
        n_bins=len(x_vals),
    )


def estimate_bucket_sigma(daily: pl.DataFrame, *, buckets_per_day: int) -> dict[str, float]:
    """Per-bucket volatility from daily close-to-close returns.

    Independent of the returns being regressed, which is what keeps the fitted
    Y coefficient meaningful.
    """
    returns = log_returns(daily)
    out: dict[str, float] = {}
    for symbol in (c for c in returns.columns if c != "date"):
        daily_vol = float(returns[symbol].std(ddof=1))
        out[symbol] = daily_vol / np.sqrt(buckets_per_day)
    return out


def calibrate_bundle(
    bundle: DatasetBundle,
    *,
    bucket_ns: int,
    sigma: Mapping[str, float] | None = None,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, CalibrationResult]:
    session_ns = 23_400 * 1_000_000_000
    buckets_per_day = max(int(round(session_ns / bucket_ns)), 1)
    sigmas = dict(sigma) if sigma is not None else estimate_bucket_sigma(
        bundle.daily(), buckets_per_day=buckets_per_day
    )

    observations = observations_from_buckets(bundle.l1(), bundle.l3(), bucket_ns=bucket_ns)
    results: dict[str, CalibrationResult] = {}
    for symbol in sorted(observations["symbol"].unique().to_list()):
        if symbol not in sigmas:
            raise CalibrationError(f"no sigma supplied for {symbol}")
        results[symbol] = fit_power_law(
            observations.filter(pl.col("symbol") == symbol),
            sigma=sigmas[symbol],
            n_bins=n_bins,
        )
    return results
