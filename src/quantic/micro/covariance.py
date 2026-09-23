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
    symbols: tuple[str, ...]
    matrix: np.ndarray
    n_observations: int
    shrinkage: float

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
    )
