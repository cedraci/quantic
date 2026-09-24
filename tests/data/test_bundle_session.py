"""A bundle must carry the trading calendar its timestamps belong to."""
import datetime as dt

import pytest

from quantic.core.session import SYNTH_SESSION, TradingSession
from quantic.data.bundle import DatasetBundle
from quantic.data.synth import SynthConfig, generate_bundle

CFG = SynthConfig(symbols=("SYNA",), n_days=2, buckets_per_day=3, seed=5, depth_levels=3)


def test_synth_bundle_records_its_session(tmp_path):
    bundle = generate_bundle(tmp_path / "b", CFG)
    assert bundle.session == SYNTH_SESSION


def test_session_survives_a_reload(tmp_path):
    generate_bundle(tmp_path / "b", CFG)
    assert DatasetBundle.load(tmp_path / "b").session == SYNTH_SESSION


def test_a_bundle_without_a_recorded_session_says_so(tmp_path):
    import polars as pl

    daily = pl.DataFrame(
        {
            "date": [dt.date(2026, 1, 5)], "symbol": ["AAA"],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1], "adv": [1.0],
        }
    )
    bundle = DatasetBundle.write(
        tmp_path / "b", {"daily_bars": daily}, bundle_id="x", provenance="test"
    )
    with pytest.raises(KeyError, match="session"):
        _ = bundle.session


def test_a_session_can_be_declared_at_write_time(tmp_path):
    import polars as pl

    nyse_like = TradingSession(open_sec=34_200, length_sec=23_400, tz="America/New_York")
    daily = pl.DataFrame(
        {
            "date": [dt.date(2026, 1, 5)], "symbol": ["AAA"],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1], "adv": [1.0],
        }
    )
    DatasetBundle.write(
        tmp_path / "b", {"daily_bars": daily},
        bundle_id="x", provenance="test", session=nyse_like,
    )
    assert DatasetBundle.load(tmp_path / "b").session == nyse_like


def test_generator_rejects_a_bucket_count_that_does_not_divide_the_session():
    with pytest.raises(Exception, match="does not divide|divide"):
        SynthConfig(symbols=("SYNA",), n_days=1, buckets_per_day=7)
