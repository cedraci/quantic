"""The ingestion boundary: a content-hashed, on-disk market data bundle.

Layout::

    root/manifest.json
    root/l1_taq/symbol=SYNA/part.parquet
    root/l2_depth/symbol=SYNA/part.parquet
    root/l3_messages/symbol=SYNA/part.parquet
    root/daily_bars/part.parquet
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from quantic.data.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    compute_content_hash,
    sha256_file,
)
from quantic.data.schemas import SCHEMA_VERSION, SCHEMAS, validate

# Canonical row ordering. Bundles are content-hashed, so ordering is part of
# the data's identity and must not depend on how a caller happened to build it.
SORT_KEYS: dict[str, list[str]] = {
    "l1_taq": ["symbol", "ts_ns"],
    "l2_depth": ["symbol", "ts_ns", "side", "level"],
    "l3_messages": ["symbol", "ts_ns", "seq"],
    "daily_bars": ["symbol", "date"],
}

_PARTITIONED = ("l1_taq", "l2_depth", "l3_messages")


class BundleIntegrityError(RuntimeError):
    """Raised when on-disk bytes do not match the manifest."""


class DatasetBundle:
    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = Path(root)
        self.manifest = manifest

    @property
    def content_hash(self) -> str:
        return self.manifest.content_hash

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.manifest.symbols

    @classmethod
    def write(
        cls,
        root: Path,
        tables: Mapping[str, pl.DataFrame],
        *,
        bundle_id: str,
        provenance: str,
        extra: dict[str, Any] | None = None,
    ) -> DatasetBundle:
        root = Path(root)
        if "daily_bars" not in tables:
            raise ValueError(
                "a bundle must include daily_bars: the manifest date range and all "
                "covariance estimation derive from it"
            )

        root.mkdir(parents=True, exist_ok=True)
        file_hashes: dict[str, str] = {}
        symbols: set[str] = set()

        for name in sorted(tables):
            df = tables[name].sort(SORT_KEYS[name])
            validate(name, df.to_arrow())
            symbols.update(df["symbol"].unique().to_list())

            if name in _PARTITIONED:
                for symbol in sorted(df["symbol"].unique().to_list()):
                    rel = f"{name}/symbol={symbol}/part.parquet"
                    dest = root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    df.filter(pl.col("symbol") == symbol).write_parquet(
                        dest, compression="zstd"
                    )
                    file_hashes[rel] = sha256_file(dest)
            else:
                rel = f"{name}/part.parquet"
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(dest, compression="zstd")
                file_hashes[rel] = sha256_file(dest)

        daily = tables["daily_bars"]
        start: dt.date = daily["date"].min()  # type: ignore[assignment]
        end: dt.date = daily["date"].max()  # type: ignore[assignment]

        manifest = Manifest(
            schema_version=SCHEMA_VERSION,
            bundle_id=bundle_id,
            created_utc=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
            provenance=provenance,
            symbols=tuple(sorted(symbols)),
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            granularities=tuple(sorted(tables)),
            files=file_hashes,
            content_hash=compute_content_hash(file_hashes),
            extra=dict(extra or {}),
        )
        manifest.write(root / MANIFEST_FILENAME)
        return cls(root, manifest)

    @classmethod
    def load(cls, root: Path) -> DatasetBundle:
        root = Path(root)
        return cls(root, Manifest.read(root / MANIFEST_FILENAME))

    def validate(self) -> None:
        for rel, expected in sorted(self.manifest.files.items()):
            path = self.root / rel
            if not path.exists():
                raise BundleIntegrityError(f"missing file {rel} in bundle {self.root}")
            actual = sha256_file(path)
            if actual != expected:
                raise BundleIntegrityError(
                    f"hash mismatch for {rel}: manifest {expected}, on disk {actual}"
                )
        recomputed = compute_content_hash(self.manifest.files)
        if recomputed != self.manifest.content_hash:
            raise BundleIntegrityError(
                f"content_hash mismatch: manifest {self.manifest.content_hash}, "
                f"recomputed {recomputed}"
            )

    def table(self, name: str) -> pl.DataFrame:
        if name not in SCHEMAS:
            raise KeyError(f"unknown granularity {name!r}")
        if name not in self.manifest.granularities:
            raise KeyError(f"bundle {self.manifest.bundle_id} has no {name!r} table")
        files = sorted((self.root / name).rglob("*.parquet"))
        return pl.read_parquet(files).sort(SORT_KEYS[name])

    def l1(self) -> pl.DataFrame:
        return self.table("l1_taq")

    def l2(self) -> pl.DataFrame:
        return self.table("l2_depth")

    def l3(self) -> pl.DataFrame:
        return self.table("l3_messages")

    def daily(self) -> pl.DataFrame:
        return self.table("daily_bars")
