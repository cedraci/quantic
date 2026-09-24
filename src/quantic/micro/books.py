"""Build :class:`BookSnapshot` values from ``l1_taq`` and ``l2_depth``.

Until this module existed, a ``BookSnapshot`` could only be produced by
replaying ``l3_messages``. Every book-based liquidity metric therefore reached
only the 3-5 symbols that have L3. Under spec section 8.4's data shape --
~25 names with L1 and L2, 3-5 of them with L3 -- no liquidity metric was
computable for roughly 20 of 25 symbols, and spec section 5.2's spread-cost
term, ``sum over (i,t) of (s_i / 2) * x[i,t]``, had no route from ``l1_taq``
to a number at all. That term is the first of M2's four objective components.

L1 gives a one-level book, L2 gives the published depth. Both go through the
same :class:`BookSnapshot` type as L3 replay, so ``quoted_spread``,
``relative_spread``, ``depth_shares``, ``depth_notional`` and ``depth_walk``
work unchanged on all three sources.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import polars as pl

from quantic.core.session import Boundary, TradingSession
from quantic.core.types import BookSnapshot, PriceLevel
from quantic.micro.bucketing import with_session_buckets
from quantic.micro.liquidity import OutOfSessionError


class BookSourceError(ValueError):
    """Raised when a quote table cannot yield a well-formed book."""


def books_from_l1(l1: pl.DataFrame) -> Iterator[BookSnapshot]:
    """One single-level book per L1 row, in ``(symbol, ts_ns)`` order.

    A crossed or locked quote is *not* rejected here: it is a real thing a
    feed publishes, and :attr:`BookSnapshot.mid` already refuses to derive a
    mid from one. Passing it through keeps the refusal at the point where a
    price is actually needed.
    """
    columns = (
        l1.sort(["symbol", "ts_ns"])
        .select("ts_ns", "symbol", "bid", "ask", "bid_size", "ask_size")
        .to_dict(as_series=False)
    )
    for ts_ns, symbol, bid, ask, bid_size, ask_size in zip(
        columns["ts_ns"],
        columns["symbol"],
        columns["bid"],
        columns["ask"],
        columns["bid_size"],
        columns["ask_size"],
        strict=True,
    ):
        yield BookSnapshot(
            ts_ns=ts_ns,
            symbol=symbol,
            bids=(PriceLevel(bid, int(bid_size)),) if bid_size is not None else (),
            asks=(PriceLevel(ask, int(ask_size)),) if ask_size is not None else (),
        )


def books_from_l2(l2: pl.DataFrame, *, levels: int | None = None) -> Iterator[BookSnapshot]:
    """One book per ``(symbol, ts_ns)`` in ``l2_depth``.

    Levels are ordered by price, not by the ``level`` column. A real export's
    ``level`` is not reliably authoritative, and :class:`BookSnapshot` refuses
    a mis-ordered book on construction, so trusting it would turn a vendor
    quirk into an exception far from its cause. Zero-size levels are dropped:
    a level with no size is not resting liquidity, and leaving it in would
    make ``depth_shares`` count a level that cannot be traded.
    """
    ordered = l2.sort(["symbol", "ts_ns", "side", "px"])
    for (symbol, ts_ns), group in ordered.group_by(
        ["symbol", "ts_ns"], maintain_order=True
    ):
        rows = group.filter(pl.col("size") > 0)
        bids = (
            rows.filter(pl.col("side") == "buy")
            .sort("px", descending=True)
            .select("px", "size")
            .rows()
        )
        asks = (
            rows.filter(pl.col("side") == "sell").sort("px").select("px", "size").rows()
        )
        if levels is not None:
            bids, asks = bids[:levels], asks[:levels]
        yield BookSnapshot(
            ts_ns=ts_ns,
            symbol=symbol,
            bids=tuple(PriceLevel(px, int(size)) for px, size in bids),
            asks=tuple(PriceLevel(px, int(size)) for px, size in asks),
        )


def bucket_spreads(
    l1: pl.DataFrame,
    *,
    session: TradingSession,
    bucket_ns: int,
    boundary: Boundary = Boundary.OPENING,
    allow_out_of_session: bool = False,
) -> pl.DataFrame:
    """Per symbol and session bucket, the quoted spread and half-spread.

    ``half_spread`` is ``s_i / 2`` from spec section 5.2's spread-cost term,
    so M2 can form ``(s_i / 2) * x[i,t]`` directly. Spreads are averaged
    across the quotes in the bucket, which is the quantity a schedule
    executing through that bucket actually pays against.

    ``boundary`` defaults to :attr:`Boundary.OPENING`, which is right for real
    L1: a quote is stamped when it is observed, so it describes the bucket it
    falls in, and a quote at the session open belongs to the first bucket.
    The synthetic generator instead stamps one quote per bucket at that
    bucket's *end*, so a fixture built from it needs
    :attr:`Boundary.CLOSING`. The two are named rather than assumed precisely
    because this choice is not obvious from either side.

    Computed from the columns rather than by iterating books: it runs over
    every L1 row in the bundle, and a ``BookSnapshot`` per quote would cost
    more than the metric is worth.
    """
    session.buckets_per_day(bucket_ns)

    bucketed = with_session_buckets(
        l1, session=session, bucket_ns=bucket_ns, boundary=boundary
    )
    outside = bucketed.filter(pl.col("bucket_id").is_null())
    if outside.height and not allow_out_of_session:
        raise OutOfSessionError(
            f"{outside.height} of {bucketed.height} L1 quote(s) fall outside the "
            f"{session.open_sec}s+{session.length_sec}s {session.tz} session "
            f"(example ts_ns={outside['ts_ns'][0]}); pass allow_out_of_session=True "
            "to drop them"
        )

    usable = bucketed.filter(pl.col("bucket_id").is_not_null())
    crossed = usable.filter(pl.col("bid") >= pl.col("ask"))
    if crossed.height:
        row = crossed.row(0, named=True)
        raise BookSourceError(
            f"{crossed.height} L1 quote(s) are crossed or locked, so they have no "
            f"meaningful spread (example: symbol={row['symbol']} ts_ns={row['ts_ns']} "
            f"bid={row['bid']} ask={row['ask']})"
        )

    return (
        usable.with_columns(
            (pl.col("ask") - pl.col("bid")).alias("quoted_spread"),
            ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
        )
        .with_columns((pl.col("quoted_spread") / pl.col("mid")).alias("relative_spread"))
        .group_by(["symbol", "bucket_id"])
        .agg(
            pl.col("quoted_spread").mean(),
            pl.col("relative_spread").mean(),
            pl.col("mid").mean(),
            pl.len().alias("n_quotes"),
        )
        .with_columns((pl.col("quoted_spread") / 2.0).alias("half_spread"))
        .select(
            "symbol", "bucket_id", "quoted_spread", "half_spread",
            "relative_spread", "mid", "n_quotes",
        )
        .sort(["symbol", "bucket_id"])
    )


def spread_series(
    l1: pl.DataFrame, *, symbol: str, since_ns: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(elapsed_ns, spreads)`` for one symbol, ready for ``resiliency_halflife``.

    ``resiliency_halflife`` had its own tests but no caller and no route from
    a bundle: nothing extracted an elapsed-time-and-spread series from
    ``l1_taq``. This is that route. ``since_ns`` anchors the clock at an
    event -- a large print, a depth depletion -- so the decay is measured from
    it rather than from the start of the day.
    """
    rows = l1.filter(pl.col("symbol") == symbol).sort("ts_ns")
    if rows.height == 0:
        present = sorted(l1["symbol"].unique().to_list())
        raise BookSourceError(f"symbol {symbol!r} is absent from l1; present: {present}")
    if since_ns is not None:
        rows = rows.filter(pl.col("ts_ns") >= since_ns)
        if rows.height == 0:
            raise BookSourceError(
                f"symbol {symbol!r} has no l1 quotes at or after ts_ns={since_ns}"
            )

    ts = rows["ts_ns"].to_numpy()
    spreads = (rows["ask"] - rows["bid"]).to_numpy()
    return (ts - ts[0]).astype(float), spreads.astype(float)
