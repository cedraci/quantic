"""Building a DatasetBundle from a raw export directory.

Review finding 2.1: there was no code path from a real data export to a
DatasetBundle. `quantic data ingest` called `DatasetBundle.load`, which needs
a manifest.json that only `DatasetBundle.write` produces, and the only CLI
route to `write` was `synth`. Pointed at a directory of exchange parquet,
ingest raised FileNotFoundError.
"""

import datetime as dt

import polars as pl
import pytest

from quantic.core.session import NYSE, TradingSession
from quantic.data.bundle import DatasetBundle
from quantic.data.ingest import (
    ExportLayoutError,
    build_bundle_from_export,
)
from quantic.data.schemas import SchemaError


def _daily(symbols=("AAA", "BBB"), n=5):
    rows = []
    for s in symbols:
        px = 100.0
        for i in range(n):
            rows.append(
                {
                    "date": dt.date(2026, 1, 5) + dt.timedelta(days=i),
                    "symbol": s,
                    "open": px, "high": px + 1, "low": px - 1, "close": px + 0.5,
                    "volume": 1_000 + i, "adv": 1_000.0,
                }
            )
            px += 0.5
    return pl.DataFrame(rows)


def _l1(symbols=("AAA", "BBB"), n=5):
    rows = []
    for s in symbols:
        for i in range(n):
            rows.append(
                {
                    "ts_ns": NYSE.open_ns(dt.date(2026, 1, 5)) + i * 10**9,
                    "symbol": s,
                    "bid": 99.99, "ask": 100.01,
                    "bid_size": 100, "ask_size": 100,
                    "last_px": 100.0, "last_size": 10,
                }
            )
    return pl.DataFrame(rows)


def _write_export(root, *, tables, fmt="parquet"):
    root.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if fmt == "parquet":
            df.write_parquet(root / f"{name}.parquet")
        else:
            df.write_csv(root / f"{name}.csv")
    return root


def test_builds_a_valid_bundle_from_a_flat_parquet_export(tmp_path):
    export = _write_export(
        tmp_path / "export", tables={"daily_bars": _daily(), "l1_taq": _l1()}
    )

    report = build_bundle_from_export(
        export, tmp_path / "bundle", bundle_id="real-1", session=NYSE
    )

    bundle = DatasetBundle.load(tmp_path / "bundle")
    bundle.validate()
    assert bundle.session == NYSE
    assert report.rows["daily_bars"] == 10
    assert set(bundle.manifest.granularities) == {"daily_bars", "l1_taq"}


def test_reads_a_csv_export(tmp_path):
    export = _write_export(
        tmp_path / "export", tables={"daily_bars": _daily()}, fmt="csv"
    )
    build_bundle_from_export(
        export, tmp_path / "bundle", bundle_id="real-csv", session=NYSE
    )
    assert DatasetBundle.load(tmp_path / "bundle").daily().height == 10


def test_reads_a_per_symbol_directory_layout(tmp_path):
    """Exchange exports are routinely partitioned per symbol."""
    export = tmp_path / "export"
    for symbol in ("AAA", "BBB"):
        d = export / "daily_bars" / f"symbol={symbol}"
        d.mkdir(parents=True)
        _daily(symbols=(symbol,)).write_parquet(d / "part.parquet")

    build_bundle_from_export(
        export, tmp_path / "bundle", bundle_id="real-part", session=NYSE
    )
    assert set(DatasetBundle.load(tmp_path / "bundle").symbols) == {"AAA", "BBB"}


def test_column_order_is_normalised_not_required(tmp_path):
    scrambled = _daily().select("volume", "symbol", "close", "date", "adv", "high", "low", "open")
    _write_export(tmp_path / "export", tables={"daily_bars": scrambled})

    build_bundle_from_export(
        tmp_path / "export", tmp_path / "bundle", bundle_id="x", session=NYSE
    )
    assert DatasetBundle.load(tmp_path / "bundle").daily().columns[0] == "date"


def test_dtypes_are_coerced_to_the_schema(tmp_path):
    loose = _daily().with_columns(
        pl.col("volume").cast(pl.Int32),
        pl.col("close").cast(pl.Float32),
    )
    _write_export(tmp_path / "export", tables={"daily_bars": loose})

    build_bundle_from_export(
        tmp_path / "export", tmp_path / "bundle", bundle_id="x", session=NYSE
    )
    daily = DatasetBundle.load(tmp_path / "bundle").daily()
    assert daily.schema["volume"] == pl.Int64
    assert daily.schema["close"] == pl.Float64


def test_extra_columns_are_dropped_and_reported(tmp_path):
    extra = _daily().with_columns(
        pl.lit("XNYS").alias("venue"), pl.lit(1).alias("vendor_flag")
    )
    _write_export(tmp_path / "export", tables={"daily_bars": extra})

    report = build_bundle_from_export(
        tmp_path / "export", tmp_path / "bundle", bundle_id="x", session=NYSE
    )
    assert set(report.dropped_columns["daily_bars"]) == {"venue", "vendor_flag"}


def test_a_missing_required_column_names_it(tmp_path):
    incomplete = _daily().drop("adv")
    _write_export(tmp_path / "export", tables={"daily_bars": incomplete})

    with pytest.raises(ExportLayoutError, match="adv"):
        build_bundle_from_export(
            tmp_path / "export", tmp_path / "bundle", bundle_id="x", session=NYSE
        )


def test_value_level_validation_runs_on_the_ingest_path(tmp_path):
    """Blocker C4's validate_values is the foundation of this path (finding 2.1)."""
    crossed = _l1().with_columns(pl.lit(100.5).alias("bid"))  # bid > ask
    _write_export(
        tmp_path / "export", tables={"daily_bars": _daily(), "l1_taq": crossed}
    )

    with pytest.raises(SchemaError, match="crossed or locked"):
        build_bundle_from_export(
            tmp_path / "export", tmp_path / "bundle", bundle_id="x", session=NYSE
        )


def test_an_export_without_daily_bars_is_rejected(tmp_path):
    _write_export(tmp_path / "export", tables={"l1_taq": _l1()})

    with pytest.raises(ExportLayoutError, match="daily_bars"):
        build_bundle_from_export(
            tmp_path / "export", tmp_path / "bundle", bundle_id="x", session=NYSE
        )


def test_an_empty_export_directory_says_what_it_looked_for(tmp_path):
    (tmp_path / "export").mkdir()
    with pytest.raises(ExportLayoutError, match="daily_bars"):
        build_bundle_from_export(
            tmp_path / "export", tmp_path / "bundle", bundle_id="x", session=NYSE
        )


def test_the_recorded_session_round_trips(tmp_path):
    _write_export(tmp_path / "export", tables={"daily_bars": _daily()})
    session = TradingSession(open_sec=32_400, length_sec=20_700, tz="Europe/Paris")

    build_bundle_from_export(
        tmp_path / "export", tmp_path / "bundle", bundle_id="x", session=session
    )
    assert DatasetBundle.load(tmp_path / "bundle").session == session


def test_provenance_records_where_the_data_came_from(tmp_path):
    _write_export(tmp_path / "export", tables={"daily_bars": _daily()})
    build_bundle_from_export(
        tmp_path / "export", tmp_path / "bundle", bundle_id="x", session=NYSE,
        provenance="market-data-machine/2026-09-24",
    )
    bundle = DatasetBundle.load(tmp_path / "bundle")
    assert bundle.manifest.provenance == "market-data-machine/2026-09-24"
    assert bundle.manifest.extra["export"]["source"].endswith("export")


def test_a_granularity_with_no_rows_reads_back_as_an_empty_typed_frame(tmp_path):
    """A real export routinely has a granularity with zero rows.

    `write()` succeeded and `validate()` reported clean, but `table()` then
    raised a raw polars ComputeError because no partition files existed to
    read. It must come back as an empty frame with the right schema.
    """
    empty_l1 = _l1().clear()
    _write_export(tmp_path / "export", tables={"daily_bars": _daily(), "l1_taq": empty_l1})

    build_bundle_from_export(
        tmp_path / "export", tmp_path / "bundle", bundle_id="x", session=NYSE
    )

    bundle = DatasetBundle.load(tmp_path / "bundle")
    bundle.validate()
    l1 = bundle.l1()
    assert l1.height == 0
    assert l1.columns == ["ts_ns", "symbol", "bid", "ask", "bid_size", "ask_size",
                          "last_px", "last_size"]
    assert l1.schema["ts_ns"] == pl.Int64
