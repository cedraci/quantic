import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantic.micro.covariance import (
    DuplicateObservationError,
    InsufficientHistoryError,
    estimate_covariance,
    log_returns,
)


def _daily(n_days: int = 60, n_symbols: int = 4, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = [f"S{i}" for i in range(n_symbols)]
    dates = [dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(n_days)]
    rows = []
    for s_idx, symbol in enumerate(symbols):
        price = 100.0
        for d in dates:
            price *= float(np.exp(rng.normal(0.0, 0.01 * (1 + s_idx * 0.2))))
            rows.append({"date": d, "symbol": symbol, "close": price})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_log_returns_shape_and_columns():
    r = log_returns(_daily(n_days=10, n_symbols=3))
    assert r.height == 9  # first row dropped
    assert r.columns == ["date", "S0", "S1", "S2"]


def test_estimate_is_symmetric_and_positive_semidefinite():
    est = estimate_covariance(_daily())
    assert est.symbols == ("S0", "S1", "S2", "S3")
    assert np.allclose(est.matrix, est.matrix.T)
    assert np.linalg.eigvalsh(est.matrix).min() > -1e-12


def test_shrinkage_is_a_valid_intensity():
    est = estimate_covariance(_daily())
    assert 0.0 <= est.shrinkage <= 1.0


def test_annualisation_scales_the_matrix():
    daily = _daily()
    a = estimate_covariance(daily, trading_days=252)
    b = estimate_covariance(daily, trading_days=1)
    assert np.allclose(a.matrix, b.matrix * 252)


def test_shrinkage_conditions_the_matrix_in_the_short_sample_regime():
    """30 assets on 20 days: the sample estimator is singular, shrinkage is not."""
    daily = _daily(n_days=21, n_symbols=30, seed=3)
    lw = estimate_covariance(daily, method="ledoit_wolf")
    sample = estimate_covariance(daily, method="sample")
    assert np.linalg.cond(lw.matrix) < np.linalg.cond(sample.matrix)
    assert np.linalg.eigvalsh(lw.matrix).min() > 0


def test_to_frame_is_labelled():
    frame = estimate_covariance(_daily(n_symbols=2)).to_frame()
    assert frame.columns == ["symbol", "S0", "S1"]
    assert frame["symbol"].to_list() == ["S0", "S1"]


def test_short_history_is_rejected():
    with pytest.raises(InsufficientHistoryError, match="at least"):
        estimate_covariance(_daily(n_days=2, n_symbols=3))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        estimate_covariance(_daily(), method="wishful")


def _daily_with_bad_close(bad_value: float, n_days: int = 30, n_symbols: int = 2) -> pl.DataFrame:
    """A clean frame with one symbol's close overwritten mid-series.

    Overwriting a middle date (not the first or last) guarantees the bad value
    poisons two consecutive log returns for that symbol, not zero.
    """
    daily = _daily(n_days=n_days, n_symbols=n_symbols, seed=1)
    dates = sorted(daily["date"].unique().to_list())
    target_date = dates[n_days // 2]
    return daily.with_columns(
        pl.when((pl.col("symbol") == "S0") & (pl.col("date") == target_date))
        .then(pl.lit(bad_value))
        .otherwise(pl.col("close"))
        .alias("close")
    )


@pytest.mark.parametrize("method", ["ledoit_wolf", "sample"])
def test_zeroed_close_raises_with_named_symbol(method: str):
    daily = _daily_with_bad_close(0.0)
    with pytest.raises(InsufficientHistoryError, match="S0"):
        estimate_covariance(daily, method=method)


@pytest.mark.parametrize("method", ["ledoit_wolf", "sample"])
def test_negative_close_raises_with_named_symbol(method: str):
    daily = _daily_with_bad_close(-5.0)
    with pytest.raises(InsufficientHistoryError, match="S0"):
        estimate_covariance(daily, method=method)


@pytest.mark.parametrize("method", ["ledoit_wolf", "sample"])
def test_clean_frame_still_works_for_both_methods(method: str):
    est = estimate_covariance(_daily(n_days=30, n_symbols=2), method=method)
    assert np.isfinite(est.matrix).all()


# --- P4: ragged real-world coverage ----------------------------------------


def _ragged_daily(n_days=60, n_symbols=5, missing_for="SYM0", n_missing=10):
    """Daily bars where one symbol is missing scattered days.

    Real coverage is ragged through halts, staggered listings and venue
    holidays. `log_returns` pivots wide and then calls `drop_nulls()`, so one
    symbol's gap removes that date for *every* symbol -- including symbols
    with complete coverage.
    """
    import datetime as dt

    rng = np.random.default_rng(0)
    start = dt.date(2026, 1, 5)
    dates = [start + dt.timedelta(days=i) for i in range(n_days)]
    gaps = set(rng.choice(range(1, n_days), size=n_missing, replace=False).tolist())

    rows = []
    for j in range(n_symbols):
        symbol = f"SYM{j}"
        price = 100.0
        for i, date in enumerate(dates):
            price *= float(np.exp(rng.normal(0.0, 0.01)))
            if symbol == missing_for and i in gaps:
                continue
            rows.append({"date": date, "symbol": symbol, "close": price})
    return pl.DataFrame(rows)


def test_ragged_coverage_raises_rather_than_silently_shrinking_the_cross_section():
    daily = _ragged_daily()
    with pytest.raises(InsufficientHistoryError, match="dropped|coverage"):
        log_returns(daily)


def test_the_dropped_count_and_the_responsible_symbol_are_named():
    daily = _ragged_daily()
    try:
        log_returns(daily)
    except InsufficientHistoryError as exc:
        message = str(exc)
    assert "SYM0" in message, "the message must name which symbol caused the loss"
    assert "10" in message or "18" in message, "the message must quantify the loss"


def test_a_caller_can_widen_the_tolerance_deliberately():
    daily = _ragged_daily()
    returns = log_returns(daily, max_dropped_fraction=0.5)
    assert returns.height > 0


def test_small_ragged_gaps_are_within_the_default_tolerance():
    daily = _ragged_daily(n_days=100, n_missing=2)
    assert log_returns(daily).height > 0


def test_duplicate_date_symbol_rows_raise_a_typed_error():
    """A duplicated (date, symbol) row is the most common real-export defect.

    It previously surfaced as a raw polars ComputeError from the pivot, which
    a caller cannot catch alongside the module's own errors.
    """
    import datetime as dt

    daily = pl.DataFrame(
        {
            "date": [dt.date(2026, 1, 5), dt.date(2026, 1, 5), dt.date(2026, 1, 6)],
            "symbol": ["AAA", "AAA", "AAA"],
            "close": [100.0, 101.0, 102.0],
        }
    )
    with pytest.raises(DuplicateObservationError, match="AAA"):
        log_returns(daily)
