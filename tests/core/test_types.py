import pytest

from quantic.core.types import Asset, BookSnapshot, EmptyBookError, PriceLevel, Side


def _book() -> BookSnapshot:
    return BookSnapshot(
        ts_ns=1_000,
        symbol="SYNA",
        bids=(PriceLevel(99.99, 500), PriceLevel(99.98, 800)),
        asks=(PriceLevel(100.01, 400), PriceLevel(100.02, 900)),
    )


def test_side_values():
    assert Side.BUY.value == "buy"
    assert Side.SELL.value == "sell"


def test_asset_is_frozen():
    a = Asset(symbol="SYNA", lot_size=100, tick_size=0.01)
    assert a.currency == "USD"
    with pytest.raises(AttributeError):
        a.symbol = "OTHER"  # type: ignore[misc]


def test_book_touch_and_derived_prices():
    b = _book()
    assert b.best_bid == pytest.approx(99.99)
    assert b.best_ask == pytest.approx(100.01)
    assert b.mid == pytest.approx(100.00)
    assert b.spread == pytest.approx(0.02)


def test_empty_side_yields_none_not_crash():
    b = BookSnapshot(ts_ns=1, symbol="SYNA", bids=(), asks=(PriceLevel(100.01, 10),))
    assert b.best_bid is None
    assert b.mid is None
    assert b.spread is None


def test_bid_ordering_is_validated():
    with pytest.raises(ValueError, match="descending"):
        BookSnapshot(
            ts_ns=1,
            symbol="SYNA",
            bids=(PriceLevel(99.98, 10), PriceLevel(99.99, 10)),
            asks=(),
        )


def test_ask_ordering_is_validated():
    with pytest.raises(ValueError, match="ascending"):
        BookSnapshot(
            ts_ns=1,
            symbol="SYNA",
            bids=(),
            asks=(PriceLevel(100.02, 10), PriceLevel(100.01, 10)),
        )


def test_crossed_book_is_rejected():
    with pytest.raises(EmptyBookError) as exc:
        b = BookSnapshot(
            ts_ns=1,
            symbol="SYNA",
            bids=(PriceLevel(100.05, 10),),
            asks=(PriceLevel(100.01, 10),),
        )
        _ = b.mid
    assert "crossed" in str(exc.value)
