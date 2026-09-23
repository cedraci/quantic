import datetime as dt
import json

import polars as pl
import pytest

from quantic.data.bundle import BundleIntegrityError, DatasetBundle
from quantic.data.manifest import MANIFEST_FILENAME


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


def _l1_two_symbols() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_ns": pl.Series([200, 100, 150], dtype=pl.Int64),
            "symbol": pl.Series(["SYNA", "SYNA", "SYNB"], dtype=pl.Utf8),
            "bid": pl.Series([49.99, 49.98, 60.0], dtype=pl.Float64),
            "ask": pl.Series([50.01, 50.00, 60.02], dtype=pl.Float64),
            "bid_size": pl.Series([500, 500, 500], dtype=pl.Int64),
            "ask_size": pl.Series([500, 500, 500], dtype=pl.Int64),
            "last_px": pl.Series([50.0, 49.99, 60.01], dtype=pl.Float64),
            "last_size": pl.Series([100, 100, 100], dtype=pl.Int64),
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
    # +0.1, not +1.0: a larger bump would push some rows' close above their
    # high, tripping validate_values' max(open, close) <= high invariant.
    changed = _daily().with_columns(pl.col("close") + 0.1)
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


def test_write_rejects_value_invalid_table(tmp_path):
    from quantic.data.schemas import SchemaError

    bad = _l1().with_columns(pl.col("bid_size").mul(0).sub(5))  # -5
    with pytest.raises(SchemaError):
        DatasetBundle.write(
            tmp_path / "bad_values",
            {"daily_bars": _daily(), "l1_taq": bad},
            bundle_id="bad_values",
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


def _inject_untracked_l1_partition(root, symbol: str = "ZZZZ") -> None:
    """Drop a schema-conforming l1_taq partition directly onto disk, bypassing
    ``DatasetBundle.write`` so it is never recorded in the manifest."""
    df = pl.DataFrame(
        {
            "ts_ns": pl.Series([999], dtype=pl.Int64),
            "symbol": pl.Series([symbol], dtype=pl.Utf8),
            "bid": pl.Series([1.0], dtype=pl.Float64),
            "ask": pl.Series([1.1], dtype=pl.Float64),
            "bid_size": pl.Series([1], dtype=pl.Int64),
            "ask_size": pl.Series([1], dtype=pl.Int64),
            "last_px": pl.Series([1.05], dtype=pl.Float64),
            "last_size": pl.Series([1], dtype=pl.Int64),
        }
    )
    dest_dir = root / "l1_taq" / f"symbol={symbol}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest_dir / "part.parquet", compression="zstd")


def test_validate_detects_untracked_file(tmp_path):
    _write(tmp_path)
    root = tmp_path / "bundle"
    _inject_untracked_l1_partition(root)

    with pytest.raises(BundleIntegrityError, match="untracked"):
        DatasetBundle.load(root).validate()


def test_table_ignores_untracked_file(tmp_path):
    _write(tmp_path)
    root = tmp_path / "bundle"
    _inject_untracked_l1_partition(root)

    l1 = DatasetBundle.load(root).l1()
    assert set(l1["symbol"].to_list()) == {"SYNA"}
    assert "ZZZZ" not in l1["symbol"].to_list()


def test_rewriting_bundle_with_fewer_symbols_does_not_leak_stale_partitions(tmp_path):
    root = tmp_path / "bundle"
    DatasetBundle.write(
        root,
        {"daily_bars": _daily(), "l1_taq": _l1_two_symbols()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )
    DatasetBundle.write(
        root,
        {"daily_bars": _daily().filter(pl.col("symbol") == "SYNA"), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )

    loaded = DatasetBundle.load(root)
    assert loaded.symbols == ("SYNA",)
    assert set(loaded.l1()["symbol"].to_list()) == {"SYNA"}


def test_rewrite_leaves_bundle_validatable(tmp_path):
    root = tmp_path / "bundle"
    DatasetBundle.write(
        root,
        {"daily_bars": _daily(), "l1_taq": _l1_two_symbols()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )
    DatasetBundle.write(
        root,
        {"daily_bars": _daily().filter(pl.col("symbol") == "SYNA"), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )

    loaded = DatasetBundle.load(root)
    loaded.validate()  # must not raise
    assert loaded.symbols == ("SYNA",)
    assert not (root / "l1_taq" / "symbol=SYNB" / "part.parquet").exists()


def test_rewrite_dropping_a_granularity_leaves_bundle_validatable(tmp_path):
    root = tmp_path / "bundle"
    DatasetBundle.write(
        root,
        {"daily_bars": _daily(), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )
    DatasetBundle.write(
        root,
        {"daily_bars": _daily()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )

    loaded = DatasetBundle.load(root)
    loaded.validate()  # must not raise
    assert not (root / "l1_taq").exists()


def test_write_preserves_unrelated_files_in_target_directory(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    (root / "README.txt").write_text("keep me")
    (root / "notes").mkdir()
    (root / "notes" / "keep.md").write_text("keep me too")

    DatasetBundle.write(
        root,
        {"daily_bars": _daily(), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )

    assert (root / "README.txt").read_text() == "keep me"
    assert (root / "notes" / "keep.md").read_text() == "keep me too"


def test_failed_write_does_not_destroy_existing_bundle(tmp_path):
    from quantic.data.schemas import SchemaError

    root = tmp_path / "bundle"
    DatasetBundle.write(
        root,
        {"daily_bars": _daily(), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )

    bad = _l1().drop("last_size")
    with pytest.raises(SchemaError):
        DatasetBundle.write(
            root,
            {"daily_bars": _daily(), "l1_taq": bad},
            bundle_id="test-bundle",
            provenance="unit-test",
        )

    loaded = DatasetBundle.load(root)
    loaded.validate()  # must not raise: the original good bundle survives
    assert loaded.l1()["ts_ns"].to_list() == [100, 200]


def test_failed_write_with_value_invalid_table_does_not_destroy_existing_bundle(tmp_path):
    from quantic.data.schemas import SchemaError

    root = tmp_path / "bundle"
    DatasetBundle.write(
        root,
        {"daily_bars": _daily(), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )

    bad = _l1().with_columns(pl.col("bid_size").mul(0).sub(5))  # -5
    with pytest.raises(SchemaError):
        DatasetBundle.write(
            root,
            {"daily_bars": _daily(), "l1_taq": bad},
            bundle_id="test-bundle",
            provenance="unit-test",
        )

    loaded = DatasetBundle.load(root)
    loaded.validate()  # must not raise: the original good bundle survives
    assert loaded.l1()["ts_ns"].to_list() == [100, 200]


def test_load_accepts_current_schema_version(tmp_path):
    _write(tmp_path)
    loaded = DatasetBundle.load(tmp_path / "bundle")
    assert loaded.manifest.schema_version == "1"


def test_load_rejects_mismatched_schema_version(tmp_path):
    _write(tmp_path)
    manifest_path = tmp_path / "bundle" / MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "2"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BundleIntegrityError) as excinfo:
        DatasetBundle.load(tmp_path / "bundle")
    message = str(excinfo.value)
    assert "'2'" in message
    assert "'1'" in message


def test_failed_write_with_unknown_granularity_does_not_destroy_existing_bundle(tmp_path):
    root = tmp_path / "bundle"
    DatasetBundle.write(
        root,
        {"daily_bars": _daily(), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )

    with pytest.raises(KeyError):
        DatasetBundle.write(
            root,
            {"daily_bars": _daily(), "l4_telepathy": _l1()},
            bundle_id="test-bundle",
            provenance="unit-test",
        )

    loaded = DatasetBundle.load(root)
    loaded.validate()  # must not raise: the original good bundle survives
    assert loaded.l1()["ts_ns"].to_list() == [100, 200]
