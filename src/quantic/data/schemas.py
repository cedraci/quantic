"""PyArrow schemas for the four market data granularities, plus validation."""

from __future__ import annotations

from enum import StrEnum

import pyarrow as pa

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
