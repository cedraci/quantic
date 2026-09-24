"""Vectorised session-relative bucketing for polars frames.

The scalar authority is :mod:`quantic.core.session`; this module is the same
arithmetic expressed as polars expressions so it can run over a whole table.
:func:`tests.micro.test_bucketing` pins the two implementations together row
for row, which is what stops them drifting apart the way the two epoch-floor
expressions in ``liquidity`` and ``calibrate`` did.
"""

from __future__ import annotations

import polars as pl

from quantic.core.session import Boundary, TradingSession

_NS = 1_000_000_000

SESSION_COLUMNS = ("session_day", "session_index", "bucket_id")


def with_session_buckets(
    df: pl.DataFrame,
    *,
    session: TradingSession,
    bucket_ns: int,
    boundary: Boundary = Boundary.OPENING,
    ts_col: str = "ts_ns",
) -> pl.DataFrame:
    """Add ``session_day``, ``session_index`` and ``bucket_id`` to ``df``.

    Rows whose timestamp falls outside a session -- pre-market, post-market,
    or a timestamp on a non-session day -- get nulls in all three columns and
    are **kept**. Dropping them here would be a silent data loss at exactly the
    point where a real export differs most from synthetic data; callers filter
    them explicitly so the loss is visible at the call site.
    """
    buckets = session.buckets_per_day(bucket_ns)

    local = (
        pl.from_epoch(pl.col(ts_col), time_unit="ns")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(session.tz)
    )
    # Truncating a tz-aware series truncates in local wall time, so this is the
    # local midnight of the local date, resolved across DST correctly.
    local_midnight_ns = local.dt.truncate("1d").dt.epoch(time_unit="ns")
    session_day = local.dt.date().to_physical().cast(pl.Int64)

    offset = pl.col(ts_col) - (local_midnight_ns + session.open_sec * _NS)
    if boundary is Boundary.CLOSING:
        offset = offset - 1

    in_session = (offset >= 0) & (offset < session.length_ns)
    index = pl.when(in_session).then(offset // bucket_ns).otherwise(None).cast(pl.Int64)

    return df.with_columns(
        pl.when(in_session).then(session_day).otherwise(None).cast(pl.Int64).alias("session_day"),
        index.alias("session_index"),
        pl.when(in_session)
        .then(session_day * buckets + (offset // bucket_ns))
        .otherwise(None)
        .cast(pl.Int64)
        .alias("bucket_id"),
    )
