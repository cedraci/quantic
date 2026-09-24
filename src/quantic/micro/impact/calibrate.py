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

from quantic.core.session import Boundary, TradingSession
from quantic.data.bundle import DatasetBundle
from quantic.micro.bucketing import with_session_buckets
from quantic.micro.covariance import log_returns
from quantic.micro.impact.sqrt_law import PowerLawImpact
from quantic.micro.liquidity import OutOfSessionError, signed_order_flow

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
    l1: pl.DataFrame,
    l3: pl.DataFrame,
    *,
    session: TradingSession,
    bucket_ns: int,
    allow_out_of_session: bool = False,
) -> pl.DataFrame:
    """Join bucket-level signed flow to the mid return realised over that bucket.

    Both sides are bucketed session-relative (:mod:`quantic.core.session`).
    The L1 mid series uses :attr:`Boundary.CLOSING` because a quote stamped on
    a bucket boundary describes the bucket that just ended; the L3 flow uses
    :attr:`Boundary.OPENING` because a message on the boundary is part of the
    new bucket's flow. Those are the project's only two conventions and both
    are half-open, so nothing is double-counted.

    Contiguity is tested on ``session_index``, never on ``bucket_id``.
    Consecutive sessions are adjacent in ``bucket_id``, so a ``bucket_id``
    difference of one does not mean two buckets are contiguous in market time
    -- that was the mechanism by which overnight returns were being scored as
    intraday impact.
    """
    session.buckets_per_day(bucket_ns)  # reject a width that does not divide the session

    quotes = with_session_buckets(
        l1, session=session, bucket_ns=bucket_ns, boundary=Boundary.CLOSING
    )
    outside = quotes.filter(pl.col("bucket_id").is_null())
    if outside.height and not allow_out_of_session:
        raise OutOfSessionError(
            f"{outside.height} of {quotes.height} L1 quote(s) fall outside the "
            f"{session.open_sec}s+{session.length_sec}s {session.tz} session "
            f"(example ts_ns={outside['ts_ns'][0]}); pass allow_out_of_session=True "
            "to drop them"
        )

    # A crossed or non-positive quote makes (bid + ask) / 2 meaningless: an
    # injected crossed quote produces a bucket "impact" of several percent
    # against a normal range well under one percent, which then dominates the
    # bin it lands in. BookSnapshot.mid already refuses this; the calibration
    # path must not be more permissive than the type it mirrors.
    bad = quotes.filter(
        pl.col("bucket_id").is_not_null()
        & (
            (pl.col("bid") >= pl.col("ask"))
            | (pl.col("bid") <= 0)
            | (pl.col("ask") <= 0)
            | pl.col("bid").is_null()
            | pl.col("ask").is_null()
        )
    )
    if bad.height:
        row = bad.row(0, named=True)
        raise CalibrationError(
            f"{bad.height} L1 quote(s) are crossed, locked or non-positive and cannot "
            f"yield a mid (example: symbol={row['symbol']} ts_ns={row['ts_ns']} "
            f"bid={row['bid']} ask={row['ask']})"
        )

    mids = (
        quotes.filter(pl.col("bucket_id").is_not_null())
        .with_columns(((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"))
        .sort(["symbol", "bucket_id", "ts_ns"])
        .group_by(["symbol", "bucket_id"], maintain_order=True)
        .agg(
            pl.col("mid").last(),
            pl.col("session_day").last(),
            pl.col("session_index").last(),
        )
        .sort(["symbol", "bucket_id"])
        .with_columns(
            pl.col("mid").shift(1).over("symbol").alias("prev_mid"),
            pl.col("session_day").shift(1).over("symbol").alias("prev_session_day"),
            pl.col("session_index").shift(1).over("symbol").alias("prev_session_index"),
        )
        # Keep only buckets contiguous *within* a session. This drops each
        # session's first bucket, whose predecessor is the previous close.
        .filter(
            (pl.col("session_day") == pl.col("prev_session_day"))
            & (pl.col("session_index") - pl.col("prev_session_index") == 1)
        )
        .with_columns((pl.col("mid") / pl.col("prev_mid") - 1.0).alias("impact"))
    )

    flow = signed_order_flow(
        l3,
        session=session,
        bucket_ns=bucket_ns,
        allow_out_of_session=allow_out_of_session,
    )

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
    session: TradingSession,
    bucket_ns: int,
    sigma: Mapping[str, float] | None = None,
    n_bins: int = DEFAULT_N_BINS,
    allow_out_of_session: bool = False,
) -> dict[str, CalibrationResult]:
    """Calibrate every symbol in ``bundle`` against ``session``.

    ``session`` is explicit rather than assumed. The previous version hardcoded
    a second copy of the session length and derived the bucket count by
    rounding, which silently reported 6 buckets per day for a 1-hour bucket in
    a 6.5-hour session -- a 4.1% error in every sigma derived from it.
    :meth:`TradingSession.buckets_per_day` is exact and rejects that width.
    """
    buckets_per_day = session.buckets_per_day(bucket_ns)
    sigmas = dict(sigma) if sigma is not None else estimate_bucket_sigma(
        bundle.daily(), buckets_per_day=buckets_per_day
    )

    observations = observations_from_buckets(
        bundle.l1(),
        bundle.l3(),
        session=session,
        bucket_ns=bucket_ns,
        allow_out_of_session=allow_out_of_session,
    )
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
