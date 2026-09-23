import polars as pl
import pytest

from quantic.micro.book_reconstruct import (
    BookBuilder,
    BookReconstructionError,
    final_book,
    snapshots_at,
)

COLUMNS = ["ts_ns", "seq", "symbol", "order_id", "action", "side", "px", "size"]


def _messages(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=COLUMNS, orient="row").with_columns(
        pl.col("ts_ns").cast(pl.Int64),
        pl.col("seq").cast(pl.Int64),
        pl.col("order_id").cast(pl.Int64),
        pl.col("px").cast(pl.Float64),
        pl.col("size").cast(pl.Int64),
    )


def test_add_builds_both_sides():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 500)
    b.apply("add", 2, "sell", 100.01, 400)
    snap = b.snapshot(10, levels=5)
    assert snap.best_bid == pytest.approx(99.99)
    assert snap.best_ask == pytest.approx(100.01)
    assert snap.bids[0].size == 500


def test_sizes_aggregate_across_orders_at_one_price():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 500)
    b.apply("add", 2, "buy", 99.99, 300)
    assert b.snapshot(10, levels=5).bids[0].size == 800


def test_partial_execute_reduces_then_full_execute_removes():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 500)
    b.apply("execute", 1, "buy", 99.99, 200)
    assert b.snapshot(10, levels=5).bids[0].size == 300
    b.apply("execute", 1, "buy", 99.99, 300)
    assert b.snapshot(11, levels=5).bids == ()
    assert b.open_orders == 0


def test_cancel_removes_the_order():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "sell", 100.01, 400)
    b.apply("cancel", 1, "sell", 100.01, 400)
    assert b.snapshot(10, levels=5).asks == ()


def test_replace_moves_price_and_size_under_same_id():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 500)
    b.apply("replace", 1, "buy", 99.97, 700)
    snap = b.snapshot(10, levels=5)
    assert [lvl.price for lvl in snap.bids] == pytest.approx([99.97])
    assert snap.bids[0].size == 700


def test_levels_are_returned_in_book_order_and_truncated():
    b = BookBuilder("SYNA")
    for i, px in enumerate([99.95, 99.99, 99.97]):
        b.apply("add", i + 1, "buy", px, 100)
    for i, px in enumerate([100.05, 100.01, 100.03]):
        b.apply("add", i + 10, "sell", px, 100)
    snap = b.snapshot(10, levels=2)
    assert [lvl.price for lvl in snap.bids] == pytest.approx([99.99, 99.97])
    assert [lvl.price for lvl in snap.asks] == pytest.approx([100.01, 100.03])


def test_duplicate_order_id_is_rejected():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 500)
    with pytest.raises(BookReconstructionError, match="duplicate"):
        b.apply("add", 1, "buy", 99.98, 100)


def test_unknown_order_id_is_rejected_when_strict():
    b = BookBuilder("SYNA")
    with pytest.raises(BookReconstructionError, match="unknown order"):
        b.apply("cancel", 42, "buy", 99.99, 100)


def test_unknown_order_id_is_tolerated_when_not_strict():
    b = BookBuilder("SYNA", strict=False)
    b.apply("cancel", 42, "buy", 99.99, 100)
    assert b.snapshot(10, levels=5).bids == ()


def test_overcancel_is_rejected():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 100)
    with pytest.raises(BookReconstructionError, match="exceeds"):
        b.apply("cancel", 1, "buy", 99.99, 500)


def test_unknown_action_is_rejected():
    b = BookBuilder("SYNA")
    with pytest.raises(BookReconstructionError, match="unknown action"):
        b.apply("teleport", 1, "buy", 99.99, 100)


def test_snapshots_at_reflects_state_at_each_timestamp():
    msgs = _messages(
        [
            (100, 0, "SYNA", 1, "add", "buy", 99.99, 500),
            (100, 1, "SYNA", 2, "add", "sell", 100.01, 400),
            (200, 2, "SYNA", 1, "execute", "buy", 99.99, 200),
            (300, 3, "SYNA", 3, "add", "buy", 99.98, 900),
        ]
    )
    snaps = snapshots_at(msgs, [150, 250, 350], levels=5)
    assert [s.ts_ns for s in snaps] == [150, 250, 350]
    assert snaps[0].bids[0].size == 500
    assert snaps[1].bids[0].size == 300
    assert len(snaps[2].bids) == 2


def test_snapshots_at_requires_a_single_symbol():
    msgs = _messages(
        [
            (100, 0, "SYNA", 1, "add", "buy", 99.99, 500),
            (100, 1, "SYNB", 2, "add", "buy", 99.99, 500),
        ]
    )
    with pytest.raises(ValueError, match="single symbol"):
        snapshots_at(msgs, [150], levels=5)


def test_final_book_applies_every_message():
    msgs = _messages(
        [
            (100, 0, "SYNA", 1, "add", "buy", 99.99, 500),
            (200, 1, "SYNA", 1, "cancel", "buy", 99.99, 500),
        ]
    )
    assert final_book(msgs, levels=5).bids == ()
