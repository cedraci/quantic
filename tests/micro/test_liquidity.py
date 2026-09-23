import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantic.core.types import BookSnapshot, PriceLevel
from quantic.micro.liquidity import (
    InsufficientDataError,
    adv,
    bucket_volume,
    depth_notional,
    depth_shares,
    quoted_spread,
    relative_spread,
    resiliency_halflife,
    signed_order_flow,
)


def _book() -> BookSnapshot:
    return BookSnapshot(
        ts_ns=1,
        symbol="SYNA",
        bids=(PriceLevel(99.99, 500), PriceLevel(99.98, 300)),
        asks=(PriceLevel(100.01, 400), PriceLevel(100.02, 600)),
    )


def test_quoted_and_relative_spread():
    assert quoted_spread(_book()) == pytest.approx(0.02)
    assert relative_spread(_book()) == pytest.approx(0.02 / 100.00)


def test_depth_shares_and_notional():
    assert depth_shares(_book(), levels=2) == (800, 1000)
    bid_notional, ask_notional = depth_notional(_book(), levels=2)
    assert bid_notional == pytest.approx(99.99 * 500 + 99.98 * 300)
    assert ask_notional == pytest.approx(100.01 * 400 + 100.02 * 600)


def test_depth_truncates_to_requested_levels():
    assert depth_shares(_book(), levels=1) == (500, 400)


def _l3() -> pl.DataFrame:
    rows = [
        # bucket 0: buyer lifts 300 on the offer, seller hits 100 on the bid -> +200
        (0, 0, "SYNA", 1, "execute", "sell", 100.01, 300),
        (1, 1, "SYNA", 2, "execute", "buy", 99.99, 100),
        (2, 2, "SYNA", 3, "add", "buy", 99.98, 900),  # not traded volume
        # bucket 1: seller hits 400 on the bid -> -400
        (1_000, 3, "SYNA", 4, "execute", "buy", 99.97, 400),
    ]
    return pl.DataFrame(
        rows,
        schema=["ts_ns", "seq", "symbol", "order_id", "action", "side", "px", "size"],
        orient="row",
    ).with_columns(
        pl.col("ts_ns").cast(pl.Int64),
        pl.col("seq").cast(pl.Int64),
        pl.col("order_id").cast(pl.Int64),
        pl.col("px").cast(pl.Float64),
        pl.col("size").cast(pl.Int64),
    )


def test_bucket_volume_counts_only_executions():
    out = bucket_volume(_l3(), bucket_ns=1_000).sort("bucket_id")
    assert out["volume"].to_list() == [400, 400]


def test_signed_order_flow_uses_the_documented_convention():
    out = signed_order_flow(_l3(), bucket_ns=1_000).sort("bucket_id")
    assert out["signed_flow"].to_list() == [200, -400]
    assert out["participation"].to_list() == pytest.approx([0.5, -1.0])


def test_adv_is_a_trailing_mean():
    daily = pl.DataFrame(
        {
            "date": [dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 7)],
            "symbol": ["SYNA"] * 3,
            "volume": [100, 200, 300],
        }
    )
    out = adv(daily, window=2).sort("date")
    assert out["adv"].to_list() == pytest.approx([100.0, 150.0, 250.0])


def test_resiliency_halflife_recovers_a_known_decay():
    tau = 5.0e9  # 5 seconds in nanoseconds
    elapsed = np.linspace(0.0, 4 * tau, 60)
    spreads = 0.01 + 0.03 * np.exp(-elapsed / tau)
    assert resiliency_halflife(elapsed, spreads) == pytest.approx(tau * np.log(2), rel=0.02)


def test_resiliency_halflife_needs_enough_points():
    with pytest.raises(InsufficientDataError):
        resiliency_halflife(np.array([0.0, 1.0]), np.array([0.02, 0.01]))
