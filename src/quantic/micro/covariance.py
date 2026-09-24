"""Cross-asset covariance estimation from daily bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf

MIN_OBSERVATIONS = 5


class InsufficientHistoryError(ValueError):
    """Raised when there are too few return observations to estimate covariance."""


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


def log_returns(daily: pl.DataFrame) -> pl.DataFrame:
    wide = (
        daily.select("date", "symbol", "close")
        .sort(["date", "symbol"])
        .pivot(on="symbol", index="date", values="close")
        .sort("date")
    )
    symbols = [c for c in wide.columns if c != "date"]
    return wide.with_columns(
        [(pl.col(s) / pl.col(s).shift(1)).log().alias(s) for s in symbols]
    ).drop_nulls()


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
