"""Turn a raw market-data export into a content-hashed :class:`DatasetBundle`.

Spec section 8.3 defines ``quantic data ingest <path>`` as the verb that
"validates against schema, computes hashes, registers in the local catalog".
Until this module existed that verb only worked on directories a *previous*
``DatasetBundle.write`` had produced: it called :meth:`DatasetBundle.load`,
which requires a ``manifest.json``, and the only route to ``write`` was the
synthetic generator. Pointed at a directory of exchange parquet it raised
``FileNotFoundError``. Spec section 15's own risk table says real data is
needed first at M1 validation, so that gap blocked M1's real-data half.

What arrives from a market-data machine is not bundle-shaped. It has extra
vendor columns, its own column order, narrower dtypes, and it may be one file
per granularity or a per-symbol directory tree. This module normalises those
differences and nothing else: it will reorder columns, widen dtypes and drop
columns the schema does not define, but it will not invent a missing column,
coerce a value, or repair a row. Everything it keeps goes through both
:func:`validate` and :func:`validate_values` before a single byte is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from quantic.core.session import TradingSession
from quantic.data.bundle import DatasetBundle
from quantic.data.schemas import SCHEMAS

# Extensions tried for each granularity, in order. Parquet first: it carries
# dtypes, so it needs less coercion and cannot silently reinterpret a value.
_SUFFIXES = (".parquet", ".csv", ".tsv")

_POLARS_TYPES: dict[str, pl.DataType] = {
    "int64": pl.Int64,
    "int32": pl.Int32,
    "double": pl.Float64,
    "string": pl.Utf8,
    "large_string": pl.Utf8,
    "date32[day]": pl.Date,
}


class ExportLayoutError(ValueError):
    """Raised when an export directory is not shaped like a market-data export."""


@dataclass(frozen=True)
class IngestReport:
    """What ingest found, kept and discarded. Returned so it can be printed."""

    bundle_id: str
    content_hash: str
    rows: dict[str, int] = field(default_factory=dict)
    sources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    dropped_columns: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _target_dtype(name: str, column: str) -> pl.DataType:
    arrow_type = str(SCHEMAS[name].field(column).type)
    try:
        return _POLARS_TYPES[arrow_type]
    except KeyError as exc:  # pragma: no cover - guards a schema change
        raise ExportLayoutError(
            f"no polars dtype known for {name!r}.{column!r} ({arrow_type})"
        ) from exc


def discover(export: Path, name: str) -> list[Path]:
    """Files holding granularity ``name``, in deterministic order.

    Accepts a flat ``<export>/<name>.parquet`` and a partitioned
    ``<export>/<name>/**/*.parquet`` tree, which is how exchange exports
    usually arrive.
    """
    found: list[Path] = []
    for suffix in _SUFFIXES:
        flat = export / f"{name}{suffix}"
        if flat.is_file():
            found.append(flat)
    directory = export / name
    if directory.is_dir():
        for suffix in _SUFFIXES:
            found.extend(sorted(directory.rglob(f"*{suffix}")))
    return sorted(set(found))


def _read(paths: list[Path]) -> pl.DataFrame:
    frames = []
    for path in paths:
        if path.suffix == ".parquet":
            frames.append(pl.read_parquet(path))
        elif path.suffix == ".tsv":
            frames.append(pl.read_csv(path, separator="\t", try_parse_dates=True))
        else:
            frames.append(pl.read_csv(path, try_parse_dates=True))
    return pl.concat(frames, how="vertical_relaxed")


def _normalise(name: str, df: pl.DataFrame) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Project ``df`` onto ``name``'s schema: required columns, order, dtypes."""
    expected = list(SCHEMAS[name].names)

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ExportLayoutError(
            f"{name!r} export is missing required column(s) {missing}; it has "
            f"{sorted(df.columns)}. Ingest will reorder, widen and drop columns, but "
            "it will not invent one -- request the field from the data machine"
        )

    dropped = tuple(sorted(set(df.columns) - set(expected)))

    casts = []
    for column in expected:
        target = _target_dtype(name, column)
        if df.schema[column] != target:
            casts.append(pl.col(column).cast(target, strict=True).alias(column))
    if casts:
        try:
            df = df.with_columns(casts)
        except Exception as exc:  # noqa: BLE001 - re-raised with the granularity named
            raise ExportLayoutError(
                f"{name!r} export has a column that cannot be cast to its schema "
                f"dtype without loss: {exc}"
            ) from exc

    return df.select(expected), dropped


def build_bundle_from_export(
    export: Path,
    out: Path,
    *,
    bundle_id: str,
    session: TradingSession,
    provenance: str = "export",
) -> IngestReport:
    """Read a raw export directory and write a validated bundle to ``out``.

    ``session`` is required, not inferred. Every intraday metric is bucketed
    session-relative, and bucketing timestamps stamped in one calendar against
    another misattributes flow invisibly.
    """
    export = Path(export)
    if not export.is_dir():
        raise ExportLayoutError(f"export path {export} is not a directory")

    tables: dict[str, pl.DataFrame] = {}
    sources: dict[str, tuple[str, ...]] = {}
    dropped: dict[str, tuple[str, ...]] = {}

    for name in sorted(SCHEMAS):
        paths = discover(export, name)
        if not paths:
            continue
        frame, dropped_columns = _normalise(name, _read(paths))
        tables[name] = frame
        sources[name] = tuple(p.relative_to(export).as_posix() for p in paths)
        dropped[name] = dropped_columns

    if "daily_bars" not in tables:
        looked_for = ", ".join(
            f"{name}{{{'|'.join(_SUFFIXES)}}} or {name}/" for name in sorted(SCHEMAS)
        )
        raise ExportLayoutError(
            f"no daily_bars found under {export}. A bundle's date range and all "
            f"covariance estimation derive from it. Looked for: {looked_for}"
        )

    # DatasetBundle.write runs validate() and validate_values() on every table
    # before it touches the filesystem, so a bad export cannot leave a
    # half-written bundle behind.
    bundle = DatasetBundle.write(
        out,
        tables,
        bundle_id=bundle_id,
        provenance=provenance,
        session=session,
        extra={
            "export": {
                "source": str(export),
                "files": {name: list(paths) for name, paths in sorted(sources.items())},
                "dropped_columns": {
                    name: list(cols) for name, cols in sorted(dropped.items()) if cols
                },
            }
        },
    )

    return IngestReport(
        bundle_id=bundle.manifest.bundle_id,
        content_hash=bundle.content_hash,
        rows={name: df.height for name, df in sorted(tables.items())},
        sources=sources,
        dropped_columns=dropped,
    )
