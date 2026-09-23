"""Liquidity metrics derived from books, message streams and daily bars."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from quantic.core.types import BookSnapshot, EmptyBookError


class InsufficientDataError(ValueError):
    """Raised when an estimator has too few observations to be meaningful."""


def quoted_spread(book: BookSnapshot) -> float:
    spread = book.spread
    if spread is None:
        raise EmptyBookError(f"one-sided book for {book.symbol} at ts_ns={book.ts_ns}")
    return spread


def relative_spread(book: BookSnapshot) -> float:
    mid = book.mid
    if mid is None:
        raise EmptyBookError(f"one-sided book for {book.symbol} at ts_ns={book.ts_ns}")
    return quoted_spread(book) / mid


def depth_shares(book: BookSnapshot, levels: int) -> tuple[int, int]:
    return (
        sum(lvl.size for lvl in book.bids[:levels]),
        sum(lvl.size for lvl in book.asks[:levels]),
    )


def depth_notional(book: BookSnapshot, levels: int) -> tuple[float, float]:
    return (
        sum(lvl.price * lvl.size for lvl in book.bids[:levels]),
        sum(lvl.price * lvl.size for lvl in book.asks[:levels]),
    )


def _executions(l3: pl.DataFrame, bucket_ns: int) -> pl.DataFrame:
    return l3.filter(pl.col("action") == "execute").with_columns(
        (pl.col("ts_ns") // bucket_ns).alias("bucket_id")
    )


def bucket_volume(l3: pl.DataFrame, *, bucket_ns: int) -> pl.DataFrame:
    return (
        _executions(l3, bucket_ns)
        .group_by(["symbol", "bucket_id"])
        .agg(pl.col("size").sum().alias("volume"))
        .sort(["symbol", "bucket_id"])
    )


def signed_order_flow(l3: pl.DataFrame, *, bucket_ns: int) -> pl.DataFrame:
    """Aggregate executions into signed flow per bucket.

    An execution resting on the sell side means a buyer lifted the offer and is
    counted positive; one resting on the buy side is counted negative. The
    synthetic generator uses the same convention, so calibration recovers the
    injected parameters rather than their mirror image.
    """
    return (
        _executions(l3, bucket_ns)
        .with_columns(
            pl.when(pl.col("side") == "sell")
            .then(pl.col("size"))
            .otherwise(-pl.col("size"))
            .alias("signed")
        )
        .group_by(["symbol", "bucket_id"])
        .agg(
            pl.col("size").sum().alias("volume"),
            pl.col("signed").sum().alias("signed_flow"),
        )
        .with_columns((pl.col("signed_flow") / pl.col("volume")).alias("participation"))
        .sort(["symbol", "bucket_id"])
    )


def adv(daily: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    return (
        daily.sort(["symbol", "date"])
        .with_columns(
            pl.col("volume")
            .cast(pl.Float64)
            .rolling_mean(window_size=window, min_samples=1)
            .over("symbol")
            .alias("adv")
        )
        .select("symbol", "date", "adv")
    )


def _decay(t: np.ndarray, s_inf: float, amplitude: float, tau: float) -> np.ndarray:
    return s_inf + amplitude * np.exp(-t / tau)


def resiliency_halflife(
    elapsed_ns: np.ndarray, spreads: np.ndarray, *, min_r_squared: float = 0.5
) -> float:
    """Half-life, in nanoseconds, of spread decay back towards its floor."""
    elapsed_ns = np.asarray(elapsed_ns, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    if elapsed_ns.size < 5:
        raise InsufficientDataError(
            f"need at least 5 observations to fit a decay, got {elapsed_ns.size}"
        )

    # Check if series is constant (no variation to fit)
    total_ss = np.sum((spreads - np.mean(spreads)) ** 2)
    if total_ss < np.finfo(float).eps:
        raise InsufficientDataError("spread series is constant and there is no decay to fit")

    span = max(elapsed_ns.max() - elapsed_ns.min(), 1.0)
    guess = (float(spreads.min()), float(spreads.max() - spreads.min()), span / 4.0)
    try:
        params, _ = curve_fit(
            _decay, elapsed_ns, spreads, p0=guess, maxfev=10_000,
            bounds=([-np.inf, 0.0, 1e-9], [np.inf, np.inf, np.inf]),
        )
    except RuntimeError as exc:
        raise InsufficientDataError(f"spread decay fit did not converge: {exc}") from exc

    # Compute R² to validate fit quality
    predicted = _decay(elapsed_ns, *params)
    residuals = spreads - predicted
    residual_ss = np.sum(residuals ** 2)
    r_squared = 1.0 - (residual_ss / total_ss)
    if r_squared < min_r_squared:
        raise InsufficientDataError(
            f"spread decay fit achieved R²={r_squared:.4f}, below threshold {min_r_squared}"
        )

    return float(params[2] * np.log(2.0))
