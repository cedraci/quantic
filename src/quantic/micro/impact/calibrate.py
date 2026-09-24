"""Calibrate the power-law impact exponent from bucket-level order flow.

Per-bucket impact is dominated by diffusion, so observations are grouped into
equal-count bins by absolute participation and the *signed* impact is averaged
within each bin, which cancels the noise (Almgren et al.; Toth et al.).
Averaging absolute impact instead would bias Y upward.

The bin means are then fitted **directly** by non-linear least squares:

    mean_impact / sigma = Y * f ** delta

rather than by regressing ``log(mean_impact / sigma)`` on ``log f``. The
logarithm is why the previous estimator had to discard every bin whose mean
signed impact was non-positive, and discarding on the sign of the outcome
variable is selection on the outcome variable: surviving bins are biased
upward and *which* bins survive is draw-dependent. The measured symptom was
that delta error did not fall monotonically with sample size -- 4800
observations per symbol did worse than 3600. Without a logarithm there is no
positivity requirement, so there is no filter and no selection mechanism.

A fit that lands outside ``PowerLawImpact``'s valid ``(0, 1]``, or that finds
impact moving against flow, is a *failed* calibration. It is raised here,
where the evidence is, rather than returned as a well-formed result that
explodes later in ``to_model()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from quantic.core.session import Boundary, TradingSession
from quantic.data.bundle import DatasetBundle
from quantic.micro.bucketing import with_session_buckets
from quantic.micro.covariance import log_returns
from quantic.micro.impact.sqrt_law import PowerLawImpact
from quantic.micro.liquidity import OutOfSessionError, signed_order_flow

MIN_BINS = 4

# The fit is allowed to wander well outside the physically valid range so that
# a convex or super-linear result is *observed* and reported, rather than
# clamped into looking plausible.
DELTA_FIT_BOUNDS = (0.01, 3.0)

# R23 amendment 2: 6 bins, not 20, is the default for both fit_power_law and
# calibrate_bundle. Chosen empirically: with more bins each bin holds too few
# observations for the mean-signed-impact averaging to cancel diffusion
# noise, and the fit degrades (delta error becomes larger and erratic).
DEFAULT_N_BINS = 6


class CalibrationError(ValueError):
    """Raised when there is not enough signal to fit an impact law."""


class ParticipationBasis(StrEnum):
    """Which participation variable a power law was regressed on.

    ``NET_IMBALANCE`` is what L3 data can actually observe: ``signed_flow /
    volume``, the market's *aggregate* net order-flow imbalance in
    ``[-1, 1]``. Nothing in a public message stream isolates one
    participant's orders, so this is the only regressor available from real
    data.

    ``OWN_PARTICIPATION`` is what M2's objective evaluates: ``x[i,t] / V_i``,
    our own one-sided share of bucket volume in ``[0, 1]``.

    They are not the same variable, and a ``Y`` fitted on one is not a ``Y``
    for the other without an assumption. Carrying the basis on the result --
    and refusing to build a model across it silently -- is what keeps that
    assumption visible instead of buried.
    """

    NET_IMBALANCE = "net_imbalance"
    OWN_PARTICIPATION = "own_participation"


@dataclass(frozen=True)
class CalibrationResult:
    """A fitted power law, with everything needed to interpret it.

    ``sigma`` and ``bucket_ns`` are recorded because ``y_coef`` is only
    meaningful relative to the volatility it was divided by and the bucket
    that volatility was measured over.
    """

    symbol: str
    delta: float
    y_coef: float
    r_squared: float
    n_observations: int
    n_bins: int
    sigma: float
    bucket_ns: int
    basis: ParticipationBasis

    def to_model(
        self,
        gamma: float = 0.0,
        *,
        assume_own_participation: bool = False,
    ) -> PowerLawImpact:
        """Build an impact model M2 can evaluate on its own decision variables.

        Refuses by default when the fit was made on net order-flow imbalance,
        because :class:`PowerLawImpact` is evaluated on
        :meth:`ImpactParams.participation` -- our own one-sided share of
        bucket volume. Reusing an imbalance-fitted ``Y`` there is the standard
        identification assumption in this literature (the impact of net signed
        volume does not depend on whose order supplied it), and it may well be
        right, but it is an assumption and the call site should say so.
        """
        if (
            self.basis is not ParticipationBasis.OWN_PARTICIPATION
            and not assume_own_participation
        ):
            raise CalibrationError(
                f"{self.symbol}: y_coef={self.y_coef:.4f} was fitted on "
                f"{self.basis.value} (aggregate signed order-flow imbalance in "
                "[-1, 1]), but PowerLawImpact is evaluated on own participation "
                "|q|/V in [0, 1]. Treating one as the other is an identification "
                "assumption -- that impact depends on net signed volume and not on "
                "whose order supplied it. Pass assume_own_participation=True to "
                "adopt it deliberately"
            )
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


def _power_law(f: np.ndarray, y_coef: float, delta: float) -> np.ndarray:
    return y_coef * f**delta


def fit_power_law(
    observations: pl.DataFrame,
    *,
    sigma: float,
    bucket_ns: int,
    n_bins: int = DEFAULT_N_BINS,
    min_participation: float = 1e-4,
) -> CalibrationResult:
    """Fit ``impact = Y * sigma * |f| ** delta`` to binned bucket observations.

    ``f`` is ``participation`` as produced by
    :func:`quantic.micro.liquidity.signed_order_flow`, i.e. aggregate net
    order-flow imbalance -- see :class:`ParticipationBasis`.
    """
    if sigma <= 0:
        raise CalibrationError(f"sigma must be positive, got {sigma}")

    symbols = observations["symbol"].unique().to_list()
    if len(symbols) != 1:
        raise CalibrationError(f"fit one symbol at a time, got {sorted(symbols)}")
    symbol = symbols[0]

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

    # Every bin is kept. A bin whose mean signed impact came out negative is a
    # draw of a noisy quantity, not evidence to be excluded -- excluding it is
    # what biased the old estimator.
    x = np.array([abs_f[g].mean() for g in groups], dtype=float)
    y = np.array([signed_impact[g].mean() for g in groups], dtype=float) / sigma

    if len(x) < MIN_BINS:
        raise CalibrationError(
            f"{symbol}: only {len(x)} bins available; need at least {MIN_BINS}"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise CalibrationError(f"{symbol}: non-finite bin means; cannot fit")
    if (x <= 0).any():
        raise CalibrationError(f"{symbol}: a bin has non-positive mean participation")

    # Seed from a coarse guess so the optimiser starts somewhere sensible; the
    # square-root law is the natural prior for delta.
    y0 = float(np.mean(np.abs(y) / x**0.5)) or 1.0
    try:
        (y_coef, delta), _ = curve_fit(
            _power_law,
            x,
            y,
            p0=(y0, 0.5),
            bounds=(
                (-np.inf, DELTA_FIT_BOUNDS[0]),
                (np.inf, DELTA_FIT_BOUNDS[1]),
            ),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError) as exc:
        raise CalibrationError(f"{symbol}: impact fit did not converge: {exc}") from exc

    predicted = _power_law(x, y_coef, delta)
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if y_coef <= 0:
        raise CalibrationError(
            f"{symbol}: fitted y_coef={y_coef:.4f} is not positive, so the data show "
            "price moving against net order flow over the fitted buckets. This is a "
            "failed calibration, not a small impact coefficient; it is reported rather "
            "than clamped to zero"
        )
    if not 0.0 < delta <= 1.0:
        raise CalibrationError(
            f"{symbol}: fitted delta={delta:.4f} lies outside the valid (0, 1] range "
            f"that PowerLawImpact accepts (r_squared={r_squared:.4f}, "
            f"n_observations={usable.height}). A delta above 1 is a convex impact law, "
            "which the spec's hardness dial D1 does not model. Reported at the fit "
            "rather than deferred to to_model()"
        )

    return CalibrationResult(
        symbol=symbol,
        delta=float(delta),
        y_coef=float(y_coef),
        r_squared=r_squared,
        n_observations=usable.height,
        n_bins=len(x),
        sigma=sigma,
        bucket_ns=bucket_ns,
        basis=ParticipationBasis.NET_IMBALANCE,
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
            bucket_ns=bucket_ns,
            n_bins=n_bins,
        )
    return results
