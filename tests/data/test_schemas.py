import datetime as dt

import polars as pl
import pyarrow as pa
import pytest

from quantic.data.schemas import SCHEMAS, L3Action, SchemaError, validate


def test_all_four_granularities_present():
    assert set(SCHEMAS) == {"l1_taq", "l2_depth", "l3_messages", "daily_bars"}


def test_l3_carries_sequence_number_for_deterministic_replay():
    assert "seq" in SCHEMAS["l3_messages"].names


def test_l3_action_values():
    assert {a.value for a in L3Action} == {"add", "cancel", "execute", "replace"}


def test_validate_accepts_conforming_table():
    df = pl.DataFrame(
        {
            "ts_ns": pl.Series([1], dtype=pl.Int64),
            "symbol": pl.Series(["SYNA"], dtype=pl.Utf8),
            "bid": pl.Series([99.99], dtype=pl.Float64),
            "ask": pl.Series([100.01], dtype=pl.Float64),
            "bid_size": pl.Series([500], dtype=pl.Int64),
            "ask_size": pl.Series([400], dtype=pl.Int64),
            "last_px": pl.Series([100.0], dtype=pl.Float64),
            "last_size": pl.Series([100], dtype=pl.Int64),
        }
    )
    validate("l1_taq", df.to_arrow())


def test_validate_rejects_unknown_granularity():
    with pytest.raises(SchemaError, match="unknown granularity"):
        validate("l4_telepathy", pa.table({"x": [1]}))


def test_validate_rejects_wrong_column_set():
    with pytest.raises(SchemaError, match="column mismatch"):
        validate("l1_taq", pa.table({"ts_ns": pa.array([1], type=pa.int64())}))


def test_validate_rejects_wrong_dtype():
    df = pl.DataFrame(
        {
            "ts_ns": pl.Series([1], dtype=pl.Int32),  # wrong: must be Int64
            "symbol": pl.Series(["SYNA"], dtype=pl.Utf8),
            "bid": pl.Series([99.99], dtype=pl.Float64),
            "ask": pl.Series([100.01], dtype=pl.Float64),
            "bid_size": pl.Series([500], dtype=pl.Int64),
            "ask_size": pl.Series([400], dtype=pl.Int64),
            "last_px": pl.Series([100.0], dtype=pl.Float64),
            "last_size": pl.Series([100], dtype=pl.Int64),
        }
    )
    with pytest.raises(SchemaError, match="dtype mismatch"):
        validate("l1_taq", df.to_arrow())


def test_validate_accepts_polars_large_string_encoding():
    """Polars emits large_string for Utf8; it must validate against pa.string()."""
    df = pl.DataFrame({"symbol": pl.Series(["SYNA"], dtype=pl.Utf8)})
    observed = df.to_arrow().schema.field("symbol").type
    assert observed in (pa.string(), pa.large_string())
    table = pa.table(
        {
            "date": pa.array([dt.date(2026, 1, 5)], type=pa.date32()),
            "symbol": df.to_arrow().column("symbol"),
            "open": pa.array([1.0], type=pa.float64()),
            "high": pa.array([1.0], type=pa.float64()),
            "low": pa.array([1.0], type=pa.float64()),
            "close": pa.array([1.0], type=pa.float64()),
            "volume": pa.array([1], type=pa.int64()),
            "adv": pa.array([1.0], type=pa.float64()),
        }
    )
    validate("daily_bars", table)
