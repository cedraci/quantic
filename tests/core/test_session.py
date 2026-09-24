"""Session-aware bucketing: one half-open convention, one source of session length."""

import datetime as dt

import pytest

from quantic.core.session import (
    NYSE,
    SYNTH_SESSION,
    Boundary,
    SessionError,
    TradingSession,
)

_NS = 1_000_000_000


def _local_ts(session: TradingSession, date: dt.date, seconds_into_session: int) -> int:
    """Epoch ns for a wall-clock offset from ``date``'s session open."""
    import zoneinfo

    tz = zoneinfo.ZoneInfo(session.tz)
    midnight = dt.datetime(date.year, date.month, date.day, tzinfo=tz)
    open_at = midnight + dt.timedelta(seconds=session.open_sec)
    return int((open_at + dt.timedelta(seconds=seconds_into_session)).timestamp()) * _NS


def test_buckets_per_day_is_exact():
    assert NYSE.buckets_per_day(1800 * _NS) == 13
    assert NYSE.buckets_per_day(60 * _NS) == 390


def test_buckets_per_day_rejects_a_bucket_that_does_not_divide_the_session():
    # 23400s / 3600s = 6.5. The old code rounded this to 6, a silent 4.1% error
    # in every sigma derived from it.
    with pytest.raises(SessionError, match="does not divide"):
        NYSE.buckets_per_day(3600 * _NS)


def test_session_open_starts_bucket_zero():
    ts = _local_ts(NYSE, dt.date(2026, 3, 3), 0)
    located = NYSE.locate(ts, 1800 * _NS)
    assert located is not None
    assert located.session_index == 0


def test_buckets_are_measured_from_the_session_open_not_from_the_epoch():
    """09:30 New York is not on an epoch boundary for a 1170s (19.5-minute) bucket.

    1170s is chosen because it divides the 23400s session exactly (20 buckets)
    but divides neither 14:30 UTC (EST open) nor 13:30 UTC (EDT open), so an
    epoch-floored bucket would straddle the session open -- the exact failure
    this module exists to make impossible.
    """
    bucket_ns = 1170 * _NS
    date = dt.date(2026, 3, 3)
    open_ts = _local_ts(NYSE, date, 0)

    assert NYSE.locate(open_ts, bucket_ns).session_index == 0
    assert NYSE.locate(open_ts + bucket_ns - 1, bucket_ns).session_index == 0
    assert NYSE.locate(open_ts + bucket_ns, bucket_ns).session_index == 1


def test_first_bucket_of_a_day_is_not_contiguous_with_the_last_of_the_previous_day():
    """The overnight gap must be visible to a consumer, not hidden by bucket_id arithmetic."""
    bucket_ns = 1800 * _NS
    bpd = NYSE.buckets_per_day(bucket_ns)

    last = NYSE.locate(_local_ts(NYSE, dt.date(2026, 3, 3), NYSE.length_sec - 1), bucket_ns)
    first = NYSE.locate(_local_ts(NYSE, dt.date(2026, 3, 4), 0), bucket_ns)

    assert last.session_index == bpd - 1
    assert first.session_index == 0
    assert first.session_day == last.session_day + 1


def test_timestamps_outside_the_session_are_located_as_none():
    bucket_ns = 1800 * _NS
    date = dt.date(2026, 3, 3)
    assert NYSE.locate(_local_ts(NYSE, date, -1), bucket_ns) is None
    assert NYSE.locate(_local_ts(NYSE, date, NYSE.length_sec), bucket_ns) is None


def test_closing_boundary_assigns_a_session_close_stamp_to_the_final_bucket():
    """L1 quotes are stamped at the bucket's end, so they close the bucket they end."""
    bucket_ns = 1800 * _NS
    bpd = NYSE.buckets_per_day(bucket_ns)
    date = dt.date(2026, 3, 3)
    close_ts = _local_ts(NYSE, date, NYSE.length_sec)

    assert NYSE.locate(close_ts, bucket_ns, boundary=Boundary.CLOSING).session_index == bpd - 1
    assert NYSE.locate(close_ts, bucket_ns, boundary=Boundary.OPENING) is None

    open_ts = _local_ts(NYSE, date, 0)
    assert NYSE.locate(open_ts, bucket_ns, boundary=Boundary.CLOSING) is None
    assert NYSE.locate(open_ts + 1, bucket_ns, boundary=Boundary.CLOSING).session_index == 0


def test_bucket_id_is_strictly_increasing_across_days():
    bucket_ns = 1800 * _NS
    a = NYSE.locate(_local_ts(NYSE, dt.date(2026, 3, 3), 0), bucket_ns)
    b = NYSE.locate(_local_ts(NYSE, dt.date(2026, 3, 3), 1800), bucket_ns)
    c = NYSE.locate(_local_ts(NYSE, dt.date(2026, 3, 4), 0), bucket_ns)
    assert a.bucket_id < b.bucket_id < c.bucket_id


def test_daylight_saving_transition_does_not_shift_the_session_index():
    """2026-03-08 is the US spring-forward; the session is still 09:30-16:00 local."""
    bucket_ns = 1800 * _NS
    before = NYSE.locate(_local_ts(NYSE, dt.date(2026, 3, 6), 0), bucket_ns)
    after = NYSE.locate(_local_ts(NYSE, dt.date(2026, 3, 9), 0), bucket_ns)
    assert before.session_index == 0
    assert after.session_index == 0


def test_synth_session_matches_the_generator_calendar():
    assert SYNTH_SESSION.tz == "UTC"
    assert SYNTH_SESSION.open_sec == 9 * 3600 + 30 * 60
    assert SYNTH_SESSION.length_sec == 23_400


def test_session_rejects_a_nonsensical_definition():
    with pytest.raises(SessionError):
        TradingSession(open_sec=-1, length_sec=100, tz="UTC")
    with pytest.raises(SessionError):
        TradingSession(open_sec=0, length_sec=0, tz="UTC")
    with pytest.raises(SessionError):
        TradingSession(open_sec=0, length_sec=100, tz="Not/AZone")


# --- parsing a session from a CLI-friendly spec ----------------------------


def test_a_named_calendar_can_be_looked_up():
    assert TradingSession.parse("nyse") == NYSE
    assert TradingSession.parse("NYSE") == NYSE
    assert TradingSession.parse("synth") == SYNTH_SESSION


def test_an_explicit_spec_can_be_parsed():
    session = TradingSession.parse("09:30-16:00@America/New_York")
    assert session.open_sec == 9 * 3600 + 30 * 60
    assert session.length_sec == 23_400
    assert session.tz == "America/New_York"


def test_an_explicit_spec_handles_a_non_round_close():
    session = TradingSession.parse("09:00-17:30@Europe/Paris")
    assert session.length_sec == (17 * 3600 + 30 * 60) - 9 * 3600


def test_an_unparseable_spec_lists_the_named_calendars():
    with pytest.raises(SessionError, match="nyse"):
        TradingSession.parse("not-a-session")


def test_a_close_before_the_open_is_rejected():
    with pytest.raises(SessionError):
        TradingSession.parse("16:00-09:30@America/New_York")
