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


def test_aggressor_is_a_distinct_type_from_resting_side():
    """`Side` means the resting order's side; `Aggressor` means who crossed.

    Both spell their members "buy"/"sell" but they mean opposite things about
    the same trade: a buying aggressor consumes *sell*-side resting liquidity.
    Before they were separated, `micro.liquidity.signed_order_flow` read
    `side` as resting while `micro.impact.depth_walk.walk` read the same enum
    as aggressor, and nothing could catch a caller passing one for the other.
    """
    from quantic.core.types import Aggressor, Side

    assert Aggressor.BUY != Side.BUY
    assert not isinstance(Side.BUY, Aggressor)


def test_aggressor_names_the_resting_side_it_consumes():
    from quantic.core.types import Aggressor, Side

    assert Aggressor.BUY.consumes is Side.SELL
    assert Aggressor.SELL.consumes is Side.BUY


def test_aggressor_can_be_derived_from_the_resting_side_that_traded():
    """An execution resting on the sell side means a buyer lifted the offer."""
    from quantic.core.types import Aggressor, Side

    assert Aggressor.from_resting_side(Side.SELL) is Aggressor.BUY
    assert Aggressor.from_resting_side(Side.BUY) is Aggressor.SELL
