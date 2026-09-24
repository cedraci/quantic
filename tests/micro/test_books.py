"""Books from L1 and L2, not only from L3 replay.

Review finding: no BookSnapshot could be built from `l1_taq` or `l2_depth`,
only by replaying `l3_messages`. Every book-based liquidity metric therefore
reached only the 3-5 symbols that have L3. Under spec section 8.4's data shape
-- ~25 names with L1+L2, 3-5 with L3 -- no liquidity metric was computable for
~20 of 25 symbols, and spec section 5.2's spread-cost term had no route from
`l1_taq` to a number at all.
"""

import datetime as dt

import polars as pl
import pytest

from quantic.core.session import NYSE, TradingSession
from quantic.core.types import BookSnapshot, EmptyBookError
from quantic.micro.books import (
    BookSourceError,
    books_from_l1,
    books_from_l2,
    bucket_spreads,
    spread_series,
)
from quantic.micro.liquidity import quoted_spread, relative_spread

_NS = 1_000_000_000
SESSION = TradingSession(open_sec=0, length_sec=3600, tz="UTC")
BUCKET_NS = 900 * _NS


def _l1(rows=None):
    rows = rows or [
        (0, "AAA", 99.99, 100.01, 500, 400, 100.0, 10),
        (1 * _NS, "AAA", 99.98, 100.02, 600, 300, 100.0, 20),
        (0, "BBB", 49.99, 50.01, 100, 100, 50.0, 5),
    ]
    return pl.DataFrame(
        rows,
        schema=["ts_ns", "symbol", "bid", "ask", "bid_size", "ask_size",
                "last_px", "last_size"],
        orient="row",
    )


def _l2():
    rows = []
    for level, (bid, ask) in enumerate([(99.99, 100.01), (99.98, 100.02), (99.97, 100.03)]):
        rows.append((0, "AAA", "buy", level, bid, 500 - level * 100))
        rows.append((0, "AAA", "sell", level, ask, 400 - level * 100))
    return pl.DataFrame(
        rows, schema=["ts_ns", "symbol", "side", "level", "px", "size"], orient="row"
    ).with_columns(pl.col("level").cast(pl.Int32))


def test_a_book_is_built_from_each_l1_quote():
    books = list(books_from_l1(_l1()))
    assert len(books) == 3
    assert all(isinstance(b, BookSnapshot) for b in books)


def test_an_l1_book_has_exactly_one_level_a_side():
    book = next(b for b in books_from_l1(_l1()) if b.symbol == "AAA" and b.ts_ns == 0)
    assert len(book.bids) == 1
    assert len(book.asks) == 1
    assert book.best_bid == 99.99
    assert book.best_ask == 100.01
    assert book.mid == pytest.approx(100.0)


def test_existing_liquidity_metrics_now_work_on_an_l1_book():
    """This is the point: spec 5.2's spread cost gets a route from l1_taq."""
    book = next(b for b in books_from_l1(_l1()) if b.symbol == "AAA" and b.ts_ns == 0)
    assert quoted_spread(book) == pytest.approx(0.02)
    assert relative_spread(book) == pytest.approx(0.0002)


def test_an_l2_book_keeps_every_level_in_the_required_order():
    book = next(iter(books_from_l2(_l2())))
    assert [lvl.price for lvl in book.bids] == [99.99, 99.98, 99.97]
    assert [lvl.price for lvl in book.asks] == [100.01, 100.02, 100.03]
    assert [lvl.size for lvl in book.bids] == [500, 400, 300]


def test_l2_levels_out_of_order_are_sorted_not_trusted():
    """A real export's `level` column is not always authoritative."""
    scrambled = _l2().sample(fraction=1.0, shuffle=True, seed=1)
    book = next(iter(books_from_l2(scrambled)))
    assert [lvl.price for lvl in book.bids] == [99.99, 99.98, 99.97]


def test_a_crossed_l1_quote_is_refused_by_the_book_it_builds():
    crossed = _l1([(0, "AAA", 100.5, 100.01, 1, 1, 100.0, 1)])
    book = next(iter(books_from_l1(crossed)))
    with pytest.raises(EmptyBookError, match="crossed"):
        _ = book.mid


def test_a_zero_size_l2_level_is_dropped_rather_than_resting_at_zero():
    empty_level = _l2().with_columns(
        pl.when((pl.col("side") == "buy") & (pl.col("level") == 1))
        .then(0)
        .otherwise(pl.col("size"))
        .alias("size")
    )
    book = next(iter(books_from_l2(empty_level)))
    assert [lvl.price for lvl in book.bids] == [99.99, 99.97]


# --- spec 5.2 term 1: a half-spread per symbol and bucket ------------------


def test_bucket_spreads_gives_the_spread_cost_term_a_route_from_l1():
    out = bucket_spreads(_l1(), session=SESSION, bucket_ns=BUCKET_NS)
    assert set(out.columns) >= {
        "symbol", "bucket_id", "quoted_spread", "relative_spread", "half_spread", "mid"
    }
    aaa = out.filter(pl.col("symbol") == "AAA").row(0, named=True)
    assert aaa["quoted_spread"] == pytest.approx(0.03, abs=0.011)
    assert aaa["half_spread"] == pytest.approx(aaa["quoted_spread"] / 2.0)


def test_bucket_spreads_covers_every_symbol_with_l1_not_only_those_with_l3():
    out = bucket_spreads(_l1(), session=SESSION, bucket_ns=BUCKET_NS)
    assert set(out["symbol"].to_list()) == {"AAA", "BBB"}


def test_bucket_spreads_rejects_quotes_outside_the_session():
    late = _l1([(7200 * _NS, "AAA", 99.99, 100.01, 1, 1, 100.0, 1)])
    from quantic.micro.liquidity import OutOfSessionError

    with pytest.raises(OutOfSessionError):
        bucket_spreads(late, session=SESSION, bucket_ns=BUCKET_NS)


def test_bucket_spreads_rejects_a_crossed_quote():
    from quantic.data.schemas import SchemaError

    crossed = _l1([(0, "AAA", 100.5, 100.01, 1, 1, 100.0, 1)])
    with pytest.raises((SchemaError, ValueError), match="cross|lock"):
        bucket_spreads(crossed, session=SESSION, bucket_ns=BUCKET_NS)


def test_bucket_spreads_works_on_a_real_calendar():
    ts = NYSE.open_ns(dt.date(2026, 3, 3))
    rows = [(ts + i * _NS, "AAA", 99.99, 100.01, 1, 1, 100.0, 1) for i in range(5)]
    out = bucket_spreads(_l1(rows), session=NYSE, bucket_ns=NYSE.length_ns // 13)
    assert out.height == 1


def test_the_closing_convention_is_available_for_generator_stamped_quotes():
    """The synthetic generator stamps one quote per bucket at the bucket's end."""
    from quantic.core.session import Boundary

    at_bucket_end = _l1([(BUCKET_NS, "AAA", 99.99, 100.01, 1, 1, 100.0, 1)])

    closing = bucket_spreads(
        at_bucket_end, session=SESSION, bucket_ns=BUCKET_NS, boundary=Boundary.CLOSING
    )
    opening = bucket_spreads(
        at_bucket_end, session=SESSION, bucket_ns=BUCKET_NS, boundary=Boundary.OPENING
    )
    assert closing["bucket_id"][0] == opening["bucket_id"][0] - 1


# --- a route from a bundle to resiliency_halflife --------------------------


def test_spread_series_feeds_resiliency_halflife_directly():
    """`resiliency_halflife` had no caller and no path from data.

    Nothing extracted an `(elapsed_ns, spread)` series from a bundle, so a
    metric with its own tests could not be computed from any real input.
    """
    import numpy as np

    from quantic.micro.liquidity import resiliency_halflife

    # A spread shock decaying back towards a 0.02 floor.
    rows = []
    for i in range(12):
        spread = 0.02 + 0.10 * float(np.exp(-i / 3.0))
        rows.append((i * _NS, "AAA", 100.0 - spread / 2, 100.0 + spread / 2, 1, 1, 100.0, 1))

    elapsed, spreads = spread_series(_l1(rows), symbol="AAA")

    assert elapsed[0] == 0
    assert len(elapsed) == len(spreads) == 12
    halflife = resiliency_halflife(elapsed, spreads)
    assert halflife == pytest.approx(3.0 * np.log(2.0) * _NS, rel=0.05)


def test_spread_series_can_start_from_an_event():
    elapsed, _ = spread_series(_l1(), symbol="AAA", since_ns=1 * _NS)
    assert elapsed.tolist() == [0]


def test_spread_series_names_an_absent_symbol():
    with pytest.raises(BookSourceError, match="ZZZ"):
        spread_series(_l1(), symbol="ZZZ")
