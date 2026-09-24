"""Cross-asset covariance estimation from daily bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf

MIN_OBSERVATIONS = 5

# Ceiling on the share of dates a ragged cross-section may cost, before the
# estimate is refused rather than quietly computed on what is left.
DEFAULT_MAX_DROPPED_FRACTION = 0.10


class InsufficientHistoryError(ValueError):
    """Raised when there are too few return observations to estimate covariance."""


class DuplicateObservationError(ValueError):
    """Raised when a (date, symbol) pair appears more than once in daily bars."""


@dataclass(frozen=True)
class CovarianceEstimate:
    """A covariance matrix that knows what horizon it is stated over.

    ``horizon_days`` is the number of trading days the ``matrix`` covers: 252
    for the annualised default, 1 for daily. It exists because the matrix was
    previously annualised silently while :class:`ImpactParams` carried a
    per-bucket sigma, so an M2 risk term combining them would have been wrong
    by ~sqrt(3276) and entirely plausible.
    """

    symbols: tuple[str, ...]
    matrix: np.ndarray
    n_observations: int
    shrinkage: float
    horizon_days: float = 252.0

    def at_horizon(self, *, days: float) -> CovarianceEstimate:
        """The same estimate restated over ``days`` trading days.

        Covariance is linear in time, so this scales the matrix by the ratio
        of horizons -- unlike a volatility, which goes as the square root.
        Mixing those two up is precisely the failure this method exists to
        prevent, so neither is ever done at a call site.
        """
        if days <= 0:
            raise ValueError(f"days must be positive, got {days}")
        from dataclasses import replace

        return replace(
            self,
            matrix=self.matrix * (days / self.horizon_days),
            horizon_days=float(days),
        )

    def to_frame(self) -> pl.DataFrame:
        data = {"symbol": list(self.symbols)}
        for j, symbol in enumerate(self.symbols):
            data[symbol] = self.matrix[:, j].tolist()
        return pl.DataFrame(data)


def log_returns(
    daily: pl.DataFrame,
    *,
    max_dropped_fraction: float = DEFAULT_MAX_DROPPED_FRACTION,
) -> pl.DataFrame:
    """Wide log returns, one column per symbol, with ragged coverage bounded.

    The pivot-then-``drop_nulls`` shape is inherently all-or-nothing across
    the cross-section: a null for any one symbol removes that date for *every*
    symbol, including symbols with complete coverage. Measured on 5 symbols
    over 60 days with one symbol missing 10 scattered days, that cost 18 of 59
    rows -- 30% of the cross-section.

    Real coverage is ragged through halts, staggered listings and venue
    holidays, so this cannot be prevented here; it can only be *bounded and
    reported*. Beyond ``max_dropped_fraction`` the estimate is refused, naming
    the count and the symbols responsible, rather than silently computed on a
    third less data than the caller believes they supplied.
    """
    if not 0.0 <= max_dropped_fraction <= 1.0:
        raise ValueError(
            f"max_dropped_fraction must lie in [0, 1], got {max_dropped_fraction}"
        )

    duplicates = (
        daily.group_by(["date", "symbol"])
        .len()
        .filter(pl.col("len") > 1)
        .sort(["date", "symbol"])
    )
    if duplicates.height:
        row = duplicates.row(0, named=True)
        symbols = sorted(set(duplicates["symbol"].to_list()))
        raise DuplicateObservationError(
            f"{duplicates.height} duplicated (date, symbol) pair(s) in daily bars for "
            f"{symbols} (example: symbol={row['symbol']} date={row['date']} appears "
            f"{row['len']} times). A pivot cannot resolve which close is authoritative; "
            "de-duplicate at ingest"
        )

    wide = (
        daily.select("date", "symbol", "close")
        .sort(["date", "symbol"])
        .pivot(on="symbol", index="date", values="close")
        .sort("date")
    )
    symbols = [c for c in wide.columns if c != "date"]
    returns = wide.with_columns(
        [(pl.col(s) / pl.col(s).shift(1)).log().alias(s) for s in symbols]
    )

    # Row 0 is null by construction (no prior close), so it is not a coverage
    # loss and must not count against the budget.
    candidate = returns.slice(1)
    kept = candidate.drop_nulls()
    dropped = candidate.height - kept.height

    if candidate.height and dropped / candidate.height > max_dropped_fraction:
        culprits = {
            s: int(candidate[s].is_null().sum())
            for s in symbols
            if candidate[s].is_null().any()
        }
        detail = ", ".join(
            f"{sym}: {n} missing" for sym, n in sorted(culprits.items(), key=lambda kv: -kv[1])
        )
        raise InsufficientHistoryError(
            f"ragged coverage dropped {dropped} of {candidate.height} dates "
            f"({dropped / candidate.height:.1%}), above the "
            f"{max_dropped_fraction:.0%} limit. A null for any one symbol removes that "
            f"date for the whole cross-section. Per-symbol gaps: {detail}. Raise "
            "max_dropped_fraction to accept the loss deliberately, or narrow the "
            "symbol set"
        )

    return kept


def estimate_covariance(
    daily: pl.DataFrame,
    *,
    method: str = "ledoit_wolf",
    trading_days: int = 252,
) -> CovarianceEstimate:
    if method not in {"ledoit_wolf", "sample"}:
        raise ValueError(f"unknown method {method!r}; expected 'ledoit_wolf' or 'sample'")

    returns = log_returns(daily)
    symbols = tuple(c for c in returns.columns if c != "date")
    values = returns.select(symbols).to_numpy()

    if not np.isfinite(values).all():
        bad_counts = {
            symbol: int((~np.isfinite(values[:, j])).sum())
            for j, symbol in enumerate(symbols)
            if not np.isfinite(values[:, j]).all()
        }
        detail = ", ".join(f"{sym}: {n} non-finite" for sym, n in bad_counts.items())
        raise InsufficientHistoryError(
            f"non-finite log returns for {detail} (out of {values.shape[0]} observations); "
            "a zero or negative close in the underlying daily bars produces -inf/NaN "
            "returns that must not be silently fed into covariance estimation"
        )

    if values.shape[0] < MIN_OBSERVATIONS:
        raise InsufficientHistoryError(
            f"need at least {MIN_OBSERVATIONS} return observations, got {values.shape[0]}"
        )

    if method == "ledoit_wolf":
        estimator = LedoitWolf().fit(values)
        cov = np.asarray(estimator.covariance_)
        shrinkage = float(estimator.shrinkage_)
    else:
        cov = np.cov(values, rowvar=False, ddof=1)
        shrinkage = 0.0

    return CovarianceEstimate(
        symbols=symbols,
        matrix=cov * trading_days,
        n_observations=values.shape[0],
        shrinkage=shrinkage,
        horizon_days=float(trading_days),
    )
