import pytest

from quantic.core.types import BookSnapshot, PriceLevel, Side
from quantic.micro.impact.depth_walk import (
    InsufficientDepthError,
    empirical_impact_curve,
    walk,
)


def _book() -> BookSnapshot:
    return BookSnapshot(
        ts_ns=1,
        symbol="SYNA",
        bids=(PriceLevel(99.99, 100), PriceLevel(99.98, 200), PriceLevel(99.97, 300)),
        asks=(PriceLevel(100.01, 100), PriceLevel(100.02, 200), PriceLevel(100.03, 300)),
    )


def test_buy_within_the_touch_pays_half_the_spread():
    r = walk(_book(), Side.BUY, 100)
    assert r.avg_price == pytest.approx(100.01)
    assert r.levels_consumed == 1
    assert r.fractional_cost == pytest.approx(0.01 / 100.0)


def test_buy_across_levels_averages_correctly():
    r = walk(_book(), Side.BUY, 250)
    expected = (100.01 * 100 + 100.02 * 150) / 250
    assert r.avg_price == pytest.approx(expected)
    assert r.levels_consumed == 2
    assert r.notional == pytest.approx(expected * 250)
    assert r.fractional_cost == pytest.approx((expected - 100.0) / 100.0)


def test_sell_walks_the_bid_side():
    r = walk(_book(), Side.SELL, 250)
    expected = (99.99 * 100 + 99.98 * 150) / 250
    assert r.avg_price == pytest.approx(expected)
    assert r.fractional_cost == pytest.approx((100.0 - expected) / 100.0)


def test_cost_is_monotone_in_size():
    costs = [walk(_book(), Side.BUY, q).fractional_cost for q in (100, 250, 500)]
    assert costs == sorted(costs)


def test_exhausting_the_book_raises():
    with pytest.raises(InsufficientDepthError, match="600"):
        walk(_book(), Side.BUY, 700)


def test_non_positive_size_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        walk(_book(), Side.BUY, 0)


def test_empirical_curve_shape():
    curve = empirical_impact_curve(_book(), Side.BUY, [100, 250, 600])
    assert curve.columns == ["shares", "participation_of_depth", "fractional_cost"]
    assert curve.height == 3
    assert curve["participation_of_depth"].to_list() == pytest.approx([100 / 600, 250 / 600, 1.0])


def test_empirical_curve_skips_sizes_beyond_depth():
    curve = empirical_impact_curve(_book(), Side.BUY, [100, 700])
    assert curve.height == 1
