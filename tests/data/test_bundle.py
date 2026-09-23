import datetime as dt

import polars as pl
import pytest

from quantic.data.bundle import BundleIntegrityError, DatasetBundle


def _daily() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": pl.Series(
                [dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 5)],
                dtype=pl.Date,
            ),
            "symbol": pl.Series(["SYNB", "SYNB", "SYNA"], dtype=pl.Utf8),
            "open": pl.Series([100.0, 101.0, 50.0], dtype=pl.Float64),
            "high": pl.Series([102.0, 103.0, 51.0], dtype=pl.Float64),
            "low": pl.Series([99.0, 100.0, 49.0], dtype=pl.Float64),
            "close": pl.Series([101.0, 102.0, 50.5], dtype=pl.Float64),
            "volume": pl.Series([1000, 1100, 900], dtype=pl.Int64),
            "adv": pl.Series([1000.0, 1050.0, 900.0], dtype=pl.Float64),
        }
    )


def _l1() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_ns": pl.Series([200, 100], dtype=pl.Int64),
            "symbol": pl.Series(["SYNA", "SYNA"], dtype=pl.Utf8),
            "bid": pl.Series([49.99, 49.98], dtype=pl.Float64),
            "ask": pl.Series([50.01, 50.00], dtype=pl.Float64),
            "bid_size": pl.Series([500, 500], dtype=pl.Int64),
            "ask_size": pl.Series([500, 500], dtype=pl.Int64),
            "last_px": pl.Series([50.0, 49.99], dtype=pl.Float64),
            "last_size": pl.Series([100, 100], dtype=pl.Int64),
        }
    )


def _write(tmp_path) -> DatasetBundle:
    return DatasetBundle.write(
        tmp_path / "bundle",
        {"daily_bars": _daily(), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
        extra={"ground_truth": {"impact_delta": {"SYNA": 0.5}}},
    )


def test_write_then_load_roundtrips(tmp_path):
    written = _write(tmp_path)
    loaded = DatasetBundle.load(tmp_path / "bundle")
    assert loaded.content_hash == written.content_hash
    assert loaded.symbols == ("SYNA", "SYNB")
    assert loaded.manifest.extra["ground_truth"]["impact_delta"]["SYNA"] == 0.5


def test_tables_are_returned_in_canonical_sorted_order(tmp_path):
    _write(tmp_path)
    l1 = DatasetBundle.load(tmp_path / "bundle").l1()
    assert l1["ts_ns"].to_list() == [100, 200]


def test_date_range_is_derived_from_daily_bars(tmp_path):
    b = _write(tmp_path)
    assert b.manifest.start_date == "2026-01-05"
    assert b.manifest.end_date == "2026-01-06"


def test_content_hash_is_stable_across_identical_writes(tmp_path):
    a = _write(tmp_path)
    b = DatasetBundle.write(
        tmp_path / "bundle2",
        {"daily_bars": _daily(), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
        extra={"ground_truth": {"impact_delta": {"SYNA": 0.5}}},
    )
    assert a.content_hash == b.content_hash


def test_content_hash_changes_when_data_changes(tmp_path):
    a = _write(tmp_path)
    changed = _daily().with_columns(pl.col("close") + 1.0)
    b = DatasetBundle.write(
        tmp_path / "bundle3",
        {"daily_bars": changed, "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )
    assert a.content_hash != b.content_hash


def test_validate_detects_tampering(tmp_path):
    _write(tmp_path)
    bundle = DatasetBundle.load(tmp_path / "bundle")
    bundle.validate()  # clean

    victim = next((tmp_path / "bundle" / "l1_taq").rglob("*.parquet"))
    victim.write_bytes(victim.read_bytes() + b"tampered")

    with pytest.raises(BundleIntegrityError, match="hash mismatch"):
        DatasetBundle.load(tmp_path / "bundle").validate()


def test_validate_detects_missing_file(tmp_path):
    _write(tmp_path)
    next((tmp_path / "bundle" / "l1_taq").rglob("*.parquet")).unlink()
    with pytest.raises(BundleIntegrityError, match="missing"):
        DatasetBundle.load(tmp_path / "bundle").validate()


def test_write_rejects_nonconforming_table(tmp_path):
    from quantic.data.schemas import SchemaError

    bad = _l1().drop("last_size")
    with pytest.raises(SchemaError):
        DatasetBundle.write(
            tmp_path / "bad",
            {"daily_bars": _daily(), "l1_taq": bad},
            bundle_id="bad",
            provenance="unit-test",
        )


def test_write_requires_daily_bars(tmp_path):
    with pytest.raises(ValueError, match="daily_bars"):
        DatasetBundle.write(
            tmp_path / "nodaily",
            {"l1_taq": _l1()},
            bundle_id="nodaily",
            provenance="unit-test",
        )
