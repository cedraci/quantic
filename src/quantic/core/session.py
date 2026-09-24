"""Trading sessions and session-relative bucketing.

This module is the single source of truth for "what a bucket is". Before it
existed the project carried two half-open conventions in one ``bucket_id``
namespace -- ``ts_ns // bucket_ns`` in ``micro.liquidity`` and
``(ts_ns - 1) // bucket_ns`` in ``micro.impact.calibrate`` -- reconciled only
by a comment, and both were floored against the Unix epoch rather than against
the session open.

Epoch-floored bucketing is wrong for any real calendar. At 09:30
America/New_York with 20-minute buckets the session open falls mid-bucket, so
the first bucket of every day straddles the overnight gap and the overnight
return is silently attributed to the day's order flow. Bucketing here is
measured from the session open, so that cannot happen by construction, and the
overnight gap is visible to consumers as ``session_index == 0``.

Like the rest of ``core``, this module imports nothing from ``quantic``.
"""

from __future__ import annotations

import datetime as dt
import zoneinfo
from dataclasses import dataclass
from enum import StrEnum

_NS = 1_000_000_000
_EPOCH_ORDINAL = dt.date(1970, 1, 1).toordinal()


class SessionError(ValueError):
    """Raised when a session definition or a bucket width is not usable."""


class Boundary(StrEnum):
    """Which side of a bucket a timestamp landing exactly on a boundary belongs to.

    ``OPENING`` is the half-open interval ``[start, end)``: a stamp on a
    boundary opens the new bucket. This is the default and the right choice
    for event streams such as ``l3_messages``, where a message at the open is
    part of the first bucket's flow.

    ``CLOSING`` is the half-open interval ``(start, end]``: a stamp on a
    boundary closes the bucket that is ending. This is the right choice for
    state series such as ``l1_taq``, which the synthetic generator stamps at
    ``ts_end_ns`` precisely because the quote describes the bucket that just
    finished.

    Both conventions are half-open, so neither can double-count. Naming them
    is what stops the two from being chosen by accident in different modules.
    """

    OPENING = "opening"
    CLOSING = "closing"


@dataclass(frozen=True, slots=True)
class SessionBucket:
    """Where a timestamp falls within a trading session.

    ``session_day`` is the *local* session date as days since 1970-01-01 -- the
    same physical representation polars uses for ``pl.Date``, so the scalar and
    vectorised bucketing paths produce identical values. It is stable across
    DST transitions.
    ``session_index`` runs ``0 .. buckets_per_day - 1``.

    ``bucket_id`` is a single strictly-increasing key over the two. Consumers
    that need to know whether two buckets are contiguous *within a session*
    must test ``session_index``, not ``bucket_id``: consecutive days are
    adjacent in ``bucket_id`` but separated by the overnight gap.
    """

    session_day: int
    session_index: int
    bucket_id: int


@dataclass(frozen=True, slots=True)
class TradingSession:
    """A continuous daily trading session in a named timezone.

    ``open_sec`` is seconds after local midnight; ``length_sec`` is the
    session's length. Both are wall-clock quantities in ``tz``, so a session
    keeps its local open and its length across daylight-saving transitions.
    """

    open_sec: int
    length_sec: int
    tz: str = "UTC"

    def __post_init__(self) -> None:
        if self.open_sec < 0:
            raise SessionError(f"open_sec must be non-negative, got {self.open_sec}")
        if self.length_sec <= 0:
            raise SessionError(f"length_sec must be positive, got {self.length_sec}")
        if self.open_sec + self.length_sec > 24 * 3600:
            raise SessionError(
                f"session {self.open_sec}s + {self.length_sec}s runs past local midnight; "
                "overnight sessions are not supported"
            )
        try:
            zoneinfo.ZoneInfo(self.tz)
        except Exception as exc:  # noqa: BLE001 - re-raised as the module's own error
            raise SessionError(f"unknown timezone {self.tz!r}") from exc

    @property
    def zone(self) -> zoneinfo.ZoneInfo:
        return zoneinfo.ZoneInfo(self.tz)

    @property
    def length_ns(self) -> int:
        return self.length_sec * _NS

    def buckets_per_day(self, bucket_ns: int) -> int:
        """Whole buckets in one session, or raise.

        Deliberately exact. The previous code derived this by rounding, which
        turned a 1-hour bucket in a 6.5-hour session into 6 rather than 6.5 --
        a silent 4.1% error in every per-bucket volatility derived from it.
        Rather than pick a rounding direction, a bucket width that does not
        divide the session is rejected.
        """
        if bucket_ns <= 0:
            raise SessionError(f"bucket_ns must be positive, got {bucket_ns}")
        if self.length_ns % bucket_ns != 0:
            raise SessionError(
                f"bucket_ns={bucket_ns} does not divide the {self.length_sec}s session "
                f"({self.length_ns / bucket_ns:.4f} buckets per day); choose a width that "
                "divides the session exactly rather than accepting a rounded bucket count"
            )
        return self.length_ns // bucket_ns

    def open_ns(self, date: dt.date) -> int:
        """Epoch ns of ``date``'s session open, resolved in the session's timezone."""
        midnight = dt.datetime(date.year, date.month, date.day, tzinfo=self.zone)
        return int(midnight.timestamp()) * _NS + self.open_sec * _NS

    def locate(
        self,
        ts_ns: int,
        bucket_ns: int,
        *,
        boundary: Boundary = Boundary.OPENING,
    ) -> SessionBucket | None:
        """Locate ``ts_ns`` within the session, or ``None`` if it falls outside one.

        Returning ``None`` rather than raising is deliberate: pre-market,
        post-market and holiday timestamps are ordinary in real exports, and a
        caller filtering them is doing so visibly. Nothing is silently
        reassigned to a neighbouring bucket.
        """
        buckets = self.buckets_per_day(bucket_ns)

        # The local date of ts_ns is the only candidate session, because a
        # session never crosses local midnight (enforced in __post_init__).
        local = dt.datetime.fromtimestamp(ts_ns // _NS, tz=self.zone)
        date = local.date()
        offset = ts_ns - self.open_ns(date)

        if boundary is Boundary.CLOSING:
            # (start, end] -- shift so the arithmetic below stays a single floor.
            offset -= 1

        if offset < 0 or offset >= self.length_ns:
            return None

        index = offset // bucket_ns
        day = date.toordinal() - _EPOCH_ORDINAL
        return SessionBucket(
            session_day=day,
            session_index=int(index),
            bucket_id=day * buckets + int(index),
        )


# 09:30-16:00 America/New_York. The calendar every real single-market equity
# bundle in this project is expected to arrive on.
NYSE = TradingSession(open_sec=9 * 3600 + 30 * 60, length_sec=23_400, tz="America/New_York")

# The synthetic generator stamps its sessions at 09:30 *UTC* with the same
# length. Kept separate from NYSE so that a test passing on synthetic data is
# never mistaken for a test passing on a real calendar.
SYNTH_SESSION = TradingSession(open_sec=9 * 3600 + 30 * 60, length_sec=23_400, tz="UTC")
