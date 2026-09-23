import datetime as dt

import polars as pl
import pyarrow as pa
import pytest

from quantic.data.schemas import SCHEMAS, L3Action, SchemaError, validate, validate_values


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


def _l1_table(**overrides) -> pa.Table:
    data = {
        "ts_ns": [100, 200],
        "symbol": ["SYNA", "SYNA"],
        "bid": [99.99, 100.0],
        "ask": [100.01, 100.02],
        "bid_size": [500, 500],
        "ask_size": [400, 400],
        "last_px": [100.0, 100.0],
        "last_size": [100, 100],
    }
    data.update(overrides)
    return pa.table(
        {
            "ts_ns": pa.array(data["ts_ns"], type=pa.int64()),
            "symbol": pa.array(data["symbol"], type=pa.string()),
            "bid": pa.array(data["bid"], type=pa.float64()),
            "ask": pa.array(data["ask"], type=pa.float64()),
            "bid_size": pa.array(data["bid_size"], type=pa.int64()),
            "ask_size": pa.array(data["ask_size"], type=pa.int64()),
            "last_px": pa.array(data["last_px"], type=pa.float64()),
            "last_size": pa.array(data["last_size"], type=pa.int64()),
        }
    )


def _l2_table(**overrides) -> pa.Table:
    data = {
        "ts_ns": [100, 200],
        "symbol": ["SYNA", "SYNA"],
        "side": ["buy", "sell"],
        "level": [0, 0],
        "px": [99.99, 100.01],
        "size": [500, 500],
    }
    data.update(overrides)
    return pa.table(
        {
            "ts_ns": pa.array(data["ts_ns"], type=pa.int64()),
            "symbol": pa.array(data["symbol"], type=pa.string()),
            "side": pa.array(data["side"], type=pa.string()),
            "level": pa.array(data["level"], type=pa.int32()),
            "px": pa.array(data["px"], type=pa.float64()),
            "size": pa.array(data["size"], type=pa.int64()),
        }
    )


def _l3_table(**overrides) -> pa.Table:
    data = {
        "ts_ns": [100, 200],
        "seq": [0, 1],
        "symbol": ["SYNA", "SYNA"],
        "order_id": [1, 2],
        "action": ["add", "execute"],
        "side": ["buy", "sell"],
        "px": [99.99, 100.01],
        "size": [500, 100],
    }
    data.update(overrides)
    return pa.table(
        {
            "ts_ns": pa.array(data["ts_ns"], type=pa.int64()),
            "seq": pa.array(data["seq"], type=pa.int64()),
            "symbol": pa.array(data["symbol"], type=pa.string()),
            "order_id": pa.array(data["order_id"], type=pa.int64()),
            "action": pa.array(data["action"], type=pa.string()),
            "side": pa.array(data["side"], type=pa.string()),
            "px": pa.array(data["px"], type=pa.float64()),
            "size": pa.array(data["size"], type=pa.int64()),
        }
    )


def _daily_bars_table(**overrides) -> pa.Table:
    data = {
        "date": [dt.date(2026, 1, 5), dt.date(2026, 1, 6)],
        "symbol": ["SYNA", "SYNA"],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1100],
        "adv": [1000.0, 1050.0],
    }
    data.update(overrides)
    return pa.table(
        {
            "date": pa.array(data["date"], type=pa.date32()),
            "symbol": pa.array(data["symbol"], type=pa.string()),
            "open": pa.array(data["open"], type=pa.float64()),
            "high": pa.array(data["high"], type=pa.float64()),
            "low": pa.array(data["low"], type=pa.float64()),
            "close": pa.array(data["close"], type=pa.float64()),
            "volume": pa.array(data["volume"], type=pa.int64()),
            "adv": pa.array(data["adv"], type=pa.float64()),
        }
    )


def test_validate_values_accepts_conforming_tables():
    validate_values("l1_taq", _l1_table())
    validate_values("l2_depth", _l2_table())
    validate_values("l3_messages", _l3_table())
    validate_values("daily_bars", _daily_bars_table())


def test_l1_rejects_crossed_book():
    with pytest.raises(SchemaError, match="bid"):
        validate_values("l1_taq", _l1_table(bid=[101.0, 100.0], ask=[100.0, 100.02]))


def test_l1_rejects_non_positive_bid():
    with pytest.raises(SchemaError, match="bid"):
        validate_values("l1_taq", _l1_table(bid=[-3.0, 100.0]))


def test_l1_rejects_negative_bid_size():
    with pytest.raises(SchemaError, match="bid_size"):
        validate_values("l1_taq", _l1_table(bid_size=[-5, 500]))


def test_l1_rejects_nan_last_px():
    with pytest.raises(SchemaError, match="last_px"):
        validate_values("l1_taq", _l1_table(last_px=[float("nan"), 100.0]))


def test_l1_rejects_negative_last_size():
    with pytest.raises(SchemaError, match="last_size"):
        validate_values("l1_taq", _l1_table(last_size=[-1, 100]))


def test_l2_rejects_non_positive_px():
    with pytest.raises(SchemaError, match="px"):
        validate_values("l2_depth", _l2_table(px=[0.0, 100.01]))


def test_l2_rejects_negative_size():
    with pytest.raises(SchemaError, match="size"):
        validate_values("l2_depth", _l2_table(size=[-5, 500]))


def test_l2_rejects_negative_level():
    with pytest.raises(SchemaError, match="level"):
        validate_values("l2_depth", _l2_table(level=[-1, 0]))


def test_l2_rejects_invalid_side():
    with pytest.raises(SchemaError, match="side"):
        validate_values("l2_depth", _l2_table(side=["sideways", "sell"]))


def test_l3_rejects_invalid_action():
    with pytest.raises(SchemaError, match="action"):
        validate_values("l3_messages", _l3_table(action=["frobnicate", "execute"]))


def test_l3_rejects_invalid_side():
    with pytest.raises(SchemaError, match="side"):
        validate_values("l3_messages", _l3_table(side=["sideways", "sell"]))


def test_l3_rejects_negative_size():
    with pytest.raises(SchemaError, match="size"):
        validate_values("l3_messages", _l3_table(size=[-5, 100]))


def test_l3_rejects_non_positive_px():
    with pytest.raises(SchemaError, match="px"):
        validate_values("l3_messages", _l3_table(px=[-1.0, 100.01]))


def test_daily_bars_rejects_high_below_close():
    with pytest.raises(SchemaError, match="high"):
        validate_values("daily_bars", _daily_bars_table(high=[1.0, 103.0], low=[99.0, 100.0]))


def test_daily_bars_rejects_low_above_open_and_close():
    with pytest.raises(SchemaError, match="low"):
        validate_values("daily_bars", _daily_bars_table(low=[100.5, 100.0]))


def test_daily_bars_rejects_negative_volume():
    with pytest.raises(SchemaError, match="volume"):
        validate_values("daily_bars", _daily_bars_table(volume=[-7, 1100]))


def test_daily_bars_rejects_negative_adv():
    with pytest.raises(SchemaError, match="adv"):
        validate_values("daily_bars", _daily_bars_table(adv=[-1.0, 1050.0]))


def test_daily_bars_rejects_non_positive_close():
    with pytest.raises(SchemaError, match="close"):
        validate_values("daily_bars", _daily_bars_table(close=[0.0, 102.0]))


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
