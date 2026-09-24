"""Build :class:`MarketParams` from a :class:`DatasetBundle`.

The only module in ``problem/`` that imports ``micro/``, and deliberately
separate from the generator. Measured on 2026-09-24, calibration **raises**
for 8 of 25 symbols at spec section 8.4's 20-day data shape and returns a
median delta error of 0.34 for the rest. Calling it inside instance
generation would make a third of the ladder fail to generate for reasons that
have nothing to do with the problem model, and would stop instance identity
being reproducible from a seed alone.

Here, a calibration failure surfaces where a human asked for real data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from quantic.core.session import Boundary, TradingSession
from quantic.data.bundle import DatasetBundle
from quantic.micro.books import bucket_spreads
from quantic.micro.covariance import estimate_covariance
from quantic.micro.impact.calibrate import calibrate_bundle
from quantic.micro.liquidity import adv
from quantic.problem.params import AssetParams, MarketParams


def market_params_from_bundle(
    bundle: DatasetBundle,
    *,
    session: TradingSession,
    bucket_ns: int,
    symbols: Sequence[str] | None = None,
    gamma: Mapping[str, float] | None = None,
    spread_boundary: Boundary = Boundary.OPENING,
    assume_own_participation: bool = False,
) -> MarketParams:
    """Compose M1's estimators into the M2 parameter set.

    ``assume_own_participation`` is passed straight through to
    ``CalibrationResult.to_model`` and is **not** defaulted to ``True``: the
    M2 objective evaluates our own participation while the fit was made on
    aggregate net order-flow imbalance, and that identification assumption
    belongs at the call site where a human chose it.

    ``spread_boundary`` defaults to ``OPENING``, which is right for real L1.
    The synthetic generator stamps one quote per bucket at that bucket's end
    and needs ``CLOSING``.
    """
    buckets_per_day = session.buckets_per_day(bucket_ns)

    calibrations = calibrate_bundle(bundle, session=session, bucket_ns=bucket_ns)
    wanted = tuple(symbols) if symbols is not None else tuple(sorted(calibrations))
    missing = [s for s in wanted if s not in calibrations]
    if missing:
        raise KeyError(
            f"no calibration for symbol(s) {missing}; the bundle calibrated "
            f"{sorted(calibrations)}"
        )

    # to_model() enforces the participation-basis contract. Its return value is
    # discarded -- delta and y_coef are read from the result -- but the check
    # is the point.
    for s in wanted:
        calibrations[s].to_model(assume_own_participation=assume_own_participation)

    spreads = (
        bucket_spreads(
            bundle.l1(), session=session, bucket_ns=bucket_ns, boundary=spread_boundary
        )
        .group_by("symbol")
        .agg(pl.col("half_spread").mean(), pl.col("mid").mean())
    )
    spread_by_symbol = {
        row["symbol"]: (row["half_spread"], row["mid"])
        for row in spreads.iter_rows(named=True)
    }

    adv_by_symbol = {
        row["symbol"]: row["adv"]
        for row in adv(bundle.daily()).group_by("symbol").agg(pl.col("adv").last()).iter_rows(
            named=True
        )
    }

    assets: list[AssetParams] = []
    for symbol in wanted:
        if symbol not in spread_by_symbol:
            raise KeyError(f"no L1 spread observations for {symbol}")
        if symbol not in adv_by_symbol:
            raise KeyError(f"no daily bars for {symbol}")
        half_spread, mid = spread_by_symbol[symbol]
        result = calibrations[symbol]
        assets.append(
            AssetParams(
                symbol=symbol,
                delta=result.delta,
                y_coef=result.y_coef,
                gamma=float(gamma[symbol]) if gamma else 0.0,
                sigma_bucket=result.sigma,
                bucket_volume_shares=float(adv_by_symbol[symbol]) / buckets_per_day,
                price=float(mid),
                half_spread=float(half_spread),
            )
        )

    return MarketParams(
        assets=tuple(assets),
        covariance=_bucket_covariance(bundle, session, bucket_ns, wanted),
        bucket_ns=bucket_ns,
    )


def _bucket_covariance(
    bundle: DatasetBundle,
    session: TradingSession,
    bucket_ns: int,
    wanted: tuple[str, ...],
) -> np.ndarray:
    """Annualised covariance rescaled to one bucket, reordered to ``wanted``.

    ``estimate_covariance`` returns its own symbol ordering. Reindexing here
    rather than assuming the two agree is what stops a silently transposed
    risk term.
    """
    estimate = estimate_covariance(bundle.daily())
    # at_horizon takes a number of trading days; one bucket is a fraction of one.
    bucket_fraction_of_a_day = bucket_ns / session.length_ns
    rescaled = estimate.at_horizon(days=bucket_fraction_of_a_day)

    index = {s: i for i, s in enumerate(estimate.symbols)}
    missing = [s for s in wanted if s not in index]
    if missing:
        raise KeyError(f"no daily returns for symbol(s) {missing}")
    order = [index[s] for s in wanted]
    return np.asarray(rescaled.matrix, dtype=float)[np.ix_(order, order)]
