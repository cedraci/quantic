"""PyArrow schemas for the four market data granularities, plus validation."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pyarrow as pa

from quantic.core.types import Side

SCHEMA_VERSION = "1"


class L3Action(StrEnum):
    ADD = "add"
    CANCEL = "cancel"
    EXECUTE = "execute"
    REPLACE = "replace"


class SchemaError(ValueError):
    """Raised when a table does not conform to its declared granularity schema."""


L1_TAQ = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("symbol", pa.string()),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("bid_size", pa.int64()),
        ("ask_size", pa.int64()),
        ("last_px", pa.float64()),
        ("last_size", pa.int64()),
    ]
)

L2_DEPTH = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("symbol", pa.string()),
        ("side", pa.string()),
        ("level", pa.int32()),
        ("px", pa.float64()),
        ("size", pa.int64()),
    ]
)

# ``seq`` disambiguates messages sharing a timestamp so replay is deterministic.
L3_MESSAGES = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("seq", pa.int64()),
        ("symbol", pa.string()),
        ("order_id", pa.int64()),
        ("action", pa.string()),
        ("side", pa.string()),
        ("px", pa.float64()),
        ("size", pa.int64()),
    ]
)

DAILY_BARS = pa.schema(
    [
        ("date", pa.date32()),
        ("symbol", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
        ("adv", pa.float64()),
    ]
)

SCHEMAS: dict[str, pa.Schema] = {
    "l1_taq": L1_TAQ,
    "l2_depth": L2_DEPTH,
    "l3_messages": L3_MESSAGES,
    "daily_bars": DAILY_BARS,
}

# Polars' ``to_arrow()`` emits large_string/large_binary for Utf8/Binary. Those
# encode identical values and differ only in offset width, so validation treats
# them as equivalent to their 32-bit counterparts.
_CANONICAL_TYPES: dict[pa.DataType, pa.DataType] = {
    pa.large_string(): pa.string(),
    pa.large_binary(): pa.binary(),
}


def _canonical(dtype: pa.DataType) -> pa.DataType:
    return _CANONICAL_TYPES.get(dtype, dtype)


def validate(name: str, table: pa.Table) -> None:
    """Raise :class:`SchemaError` unless ``table`` conforms exactly to ``name``'s schema.

    Column order is part of the contract: bundles are content-hashed, and a
    reordered table would hash differently while being semantically identical,
    which would silently break reproducibility claims.
    """
    expected = SCHEMAS.get(name)
    if expected is None:
        raise SchemaError(f"unknown granularity {name!r}; expected one of {sorted(SCHEMAS)}")

    if list(table.schema.names) != list(expected.names):
        raise SchemaError(
            f"column mismatch for {name!r}: "
            f"expected {list(expected.names)}, got {list(table.schema.names)}"
        )

    for field in expected:
        actual = table.schema.field(field.name).type
        if _canonical(actual) != _canonical(field.type):
            raise SchemaError(
                f"dtype mismatch for {name!r}.{field.name}: expected {field.type}, got {actual}"
            )


def _column(table: pa.Table, column: str) -> np.ndarray:
    return table.column(column).to_numpy(zero_copy_only=False)


def _require_finite_positive(name: str, table: pa.Table, column: str) -> None:
    values = _column(table, column)
    bad = ~np.isfinite(values) | (values <= 0)
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        example = values[bad][0]
        raise SchemaError(
            f"{name!r}.{column!r}: {n_bad} row(s) are not finite and > 0 "
            f"(example value: {example!r})"
        )


def _require_nonnegative(name: str, table: pa.Table, column: str) -> None:
    values = _column(table, column)
    bad = values < 0
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        example = values[bad][0]
        raise SchemaError(
            f"{name!r}.{column!r}: {n_bad} row(s) are negative (example value: {example!r})"
        )


def _require_vocabulary(name: str, table: pa.Table, column: str, vocabulary: set[str]) -> None:
    values = table.column(column).to_pylist()
    bad_values = [v for v in values if v not in vocabulary]
    n_bad = len(bad_values)
    if n_bad:
        raise SchemaError(
            f"{name!r}.{column!r}: {n_bad} row(s) outside the allowed vocabulary "
            f"{sorted(vocabulary)} (example value: {bad_values[0]!r})"
        )


def validate_values(name: str, table: pa.Table) -> None:
    """Raise :class:`SchemaError` unless every row of ``table`` is physically sensible.

    :func:`validate` only checks that a table is *shaped* like ``name``'s
    schema (columns, order, dtypes). A table can pass that check and still
    hold a crossed book, a negative price, or an action string that is not
    in :class:`L3Action`'s vocabulary. This function catches those, since
    silently accepting them would violate the project's "nothing is
    silently dropped or patched" constraint just as badly as a shape defect.
    """
    if name not in SCHEMAS:
        raise SchemaError(f"unknown granularity {name!r}; expected one of {sorted(SCHEMAS)}")

    side_vocabulary = {s.value for s in Side}

    if name == "l1_taq":
        _require_finite_positive(name, table, "bid")
        _require_finite_positive(name, table, "ask")
        _require_finite_positive(name, table, "last_px")

        bid = _column(table, "bid")
        ask = _column(table, "ask")
        bad = bid >= ask
        n_bad = int(np.count_nonzero(bad))
        if n_bad:
            raise SchemaError(
                f"{name!r}: {n_bad} row(s) have bid >= ask (crossed or locked book) "
                f"(example: bid={bid[bad][0]!r}, ask={ask[bad][0]!r})"
            )

        _require_nonnegative(name, table, "bid_size")
        _require_nonnegative(name, table, "ask_size")
        _require_nonnegative(name, table, "last_size")

    elif name == "l2_depth":
        _require_finite_positive(name, table, "px")
        _require_nonnegative(name, table, "size")
        _require_nonnegative(name, table, "level")
        _require_vocabulary(name, table, "side", side_vocabulary)

    elif name == "l3_messages":
        _require_finite_positive(name, table, "px")
        _require_nonnegative(name, table, "size")
        _require_vocabulary(name, table, "action", {a.value for a in L3Action})
        _require_vocabulary(name, table, "side", side_vocabulary)

    elif name == "daily_bars":
        for column in ("open", "high", "low", "close"):
            _require_finite_positive(name, table, column)

        open_ = _column(table, "open")
        high = _column(table, "high")
        low = _column(table, "low")
        close = _column(table, "close")
        min_oc = np.minimum(open_, close)
        max_oc = np.maximum(open_, close)

        bad_low = low > min_oc
        n_bad = int(np.count_nonzero(bad_low))
        if n_bad:
            raise SchemaError(
                f"{name!r}: {n_bad} row(s) have low > min(open, close) "
                f"(example: low={low[bad_low][0]!r}, open={open_[bad_low][0]!r}, "
                f"close={close[bad_low][0]!r})"
            )

        bad_high = max_oc > high
        n_bad = int(np.count_nonzero(bad_high))
        if n_bad:
            raise SchemaError(
                f"{name!r}: {n_bad} row(s) have max(open, close) > high "
                f"(example: high={high[bad_high][0]!r}, open={open_[bad_high][0]!r}, "
                f"close={close[bad_high][0]!r})"
            )

        _require_nonnegative(name, table, "volume")
        _require_nonnegative(name, table, "adv")
