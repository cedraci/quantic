"""The vectorised bucketing path must agree with core.session row for row."""

import datetime as dt

import polars as pl
import pytest

from quantic.core.session import NYSE, SYNTH_SESSION, Boundary
from quantic.micro.bucketing import with_session_buckets

_NS = 1_000_000_000
_BUCKET = 1800 * _NS


def _ts(session, date: dt.date, seconds: int) -> int:
    return session.open_ns(date) + seconds * _NS


def test_matches_the_scalar_locator_row_for_row():
    session = NYSE
    stamps = [
        _ts(session, dt.date(2026, 3, 3), s)
        for s in (0, 1, 1799, 1800, 23_399, 23_400)
    ] + [
        _ts(session, dt.date(2026, 7, 1), s) for s in (0, 5400, 23_399)
    ]
    df = pl.DataFrame({"ts_ns": stamps})

    out = with_session_buckets(df, session=session, bucket_ns=_BUCKET)

    for row in out.iter_rows(named=True):
        located = session.locate(row["ts_ns"], _BUCKET)
        if located is None:
            assert row["session_index"] is None
            assert row["bucket_id"] is None
        else:
            assert row["session_day"] == located.session_day
            assert row["session_index"] == located.session_index
            assert row["bucket_id"] == located.bucket_id


def test_out_of_session_rows_are_null_not_dropped():
    date = dt.date(2026, 3, 3)
    df = pl.DataFrame(
        {"ts_ns": [_ts(NYSE, date, -1), _ts(NYSE, date, 0), _ts(NYSE, date, 23_400)]}
    )

    out = with_session_buckets(df, session=NYSE, bucket_ns=_BUCKET)

    assert out.height == 3, "rows outside the session must survive, visibly, as nulls"
    assert out["session_index"].to_list() == [None, 0, None]


def test_closing_boundary_shifts_a_boundary_stamp_into_the_bucket_it_ends():
    date = dt.date(2026, 3, 3)
    df = pl.DataFrame({"ts_ns": [_ts(NYSE, date, 1800)]})

    opening = with_session_buckets(df, session=NYSE, bucket_ns=_BUCKET)
    closing = with_session_buckets(
        df, session=NYSE, bucket_ns=_BUCKET, boundary=Boundary.CLOSING
    )

    assert opening["session_index"][0] == 1
    assert closing["session_index"][0] == 0


def test_daylight_saving_does_not_shift_the_index():
    before = pl.DataFrame({"ts_ns": [_ts(NYSE, dt.date(2026, 3, 6), 0)]})
    after = pl.DataFrame({"ts_ns": [_ts(NYSE, dt.date(2026, 3, 9), 0)]})
    kw = {"session": NYSE, "bucket_ns": _BUCKET}

    assert with_session_buckets(before, **kw)["session_index"][0] == 0
    assert with_session_buckets(after, **kw)["session_index"][0] == 0


def test_rejects_a_bucket_width_that_does_not_divide_the_session():
    df = pl.DataFrame({"ts_ns": [_ts(SYNTH_SESSION, dt.date(2026, 1, 5), 0)]})
    with pytest.raises(Exception, match="does not divide"):
        with_session_buckets(df, session=SYNTH_SESSION, bucket_ns=3600 * _NS)


def test_preserves_existing_columns_and_row_order():
    date = dt.date(2026, 3, 3)
    df = pl.DataFrame(
        {
            "ts_ns": [_ts(NYSE, date, 0), _ts(NYSE, date, 1800)],
            "symbol": ["AAA", "BBB"],
        }
    )
    out = with_session_buckets(df, session=NYSE, bucket_ns=_BUCKET)
    assert out["symbol"].to_list() == ["AAA", "BBB"]
    assert out["ts_ns"].to_list() == df["ts_ns"].to_list()
