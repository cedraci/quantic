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
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from quantic.core.session import TradingSession
from quantic.data.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    compute_content_hash,
    sha256_file,
)
from quantic.data.schemas import SCHEMA_VERSION, SCHEMAS, validate, validate_values

# Canonical row ordering. Bundles are content-hashed, so ordering is part of
# the data's identity and must not depend on how a caller happened to build it.
SORT_KEYS: dict[str, list[str]] = {
    "l1_taq": ["symbol", "ts_ns"],
    "l2_depth": ["symbol", "ts_ns", "side", "level"],
    "l3_messages": ["symbol", "ts_ns", "seq"],
    "daily_bars": ["symbol", "date"],
}

_PARTITIONED = ("l1_taq", "l2_depth", "l3_messages")

# Well-known key in Manifest.extra. A bundle's trading calendar is part of its
# identity: every intraday metric derived from it is bucketed session-relative,
# and bucketing L3 stamped in one calendar against another silently
# misattributes flow. Stored in `extra` rather than as a manifest field so
# existing bundles stay readable without a schema_version bump.
SESSION_KEY = "session"


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

    @property
    def session(self) -> TradingSession:
        """The trading calendar these timestamps belong to.

        Raises rather than defaulting. Guessing a calendar is how L3 stamped
        in one market ends up bucketed against another, and the resulting
        misattribution is invisible in the output.
        """
        payload = self.manifest.extra.get(SESSION_KEY)
        if payload is None:
            raise KeyError(
                f"bundle {self.manifest.bundle_id} records no trading session; it was "
                "written before sessions were tracked, or by a writer that did not "
                "declare one. Pass session= to DatasetBundle.write, or supply the "
                "session explicitly at the call site"
            )
        return TradingSession(
            open_sec=int(payload["open_sec"]),
            length_sec=int(payload["length_sec"]),
            tz=str(payload["tz"]),
        )

    @classmethod
    def write(
        cls,
        root: Path,
        tables: Mapping[str, pl.DataFrame],
        *,
        bundle_id: str,
        provenance: str,
        session: TradingSession | None = None,
        extra: dict[str, Any] | None = None,
    ) -> DatasetBundle:
        root = Path(root)
        if "daily_bars" not in tables:
            raise ValueError(
                "a bundle must include daily_bars: the manifest date range and all "
                "covariance estimation derive from it"
            )

        # Nothing is deleted until every table has been validated. write() can
        # reject input in two ways: a table that fails schema validation, or a
        # table under an unknown granularity name (SORT_KEYS[name] raises
        # KeyError for that case). Both must be caught before any destructive
        # filesystem call, or a caller who passes one bad table alongside
        # otherwise-good data would destroy an existing good bundle and leave
        # a stale manifest.json pointing at files that no longer exist.
        prepared: dict[str, pl.DataFrame] = {}
        for name in sorted(tables):
            df = tables[name].sort(SORT_KEYS[name])
            arrow = df.to_arrow()
            validate(name, arrow)
            validate_values(name, arrow)
            prepared[name] = df

        # A bundle directory must contain exactly what its manifest describes,
        # because validate() treats any untracked parquet as an integrity
        # failure. Clear every granularity subdirectory (not just the ones in
        # `tables`) before writing, so a rewrite that drops a symbol or an
        # entire granularity can never leave a stale, untracked partition
        # behind. This only runs after every table above has been validated.
        root.mkdir(parents=True, exist_ok=True)
        for name in SCHEMAS:
            stale = root / name
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=False)

        file_hashes: dict[str, str] = {}
        symbols: set[str] = set()

        for name, df in prepared.items():
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

        daily = prepared["daily_bars"]
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
            granularities=tuple(sorted(prepared)),
            files=file_hashes,
            content_hash=compute_content_hash(file_hashes),
            extra=_with_session(dict(extra or {}), session),
        )
        manifest.write(root / MANIFEST_FILENAME)
        return cls(root, manifest)

    @classmethod
    def load(cls, root: Path) -> DatasetBundle:
        root = Path(root)
        manifest = Manifest.read(root / MANIFEST_FILENAME)
        if manifest.schema_version != SCHEMA_VERSION:
            raise BundleIntegrityError(
                f"bundle {root} has schema_version {manifest.schema_version!r}, but this "
                f"build only supports schema_version {SCHEMA_VERSION!r}: the manifest may "
                "describe a data layout this code does not understand"
            )
        return cls(root, manifest)

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

        tracked = set(self.manifest.files)
        on_disk = {p.relative_to(self.root).as_posix() for p in self.root.rglob("*.parquet")}
        untracked = sorted(on_disk - tracked)
        if untracked:
            raise BundleIntegrityError(
                f"untracked file(s) found in bundle {self.root} "
                f"(present on disk but not in manifest): {', '.join(untracked)}"
            )

    def table(self, name: str) -> pl.DataFrame:
        if name not in SCHEMAS:
            raise KeyError(f"unknown granularity {name!r}")
        if name not in self.manifest.granularities:
            raise KeyError(f"bundle {self.manifest.bundle_id} has no {name!r} table")
        rels = sorted(rel for rel in self.manifest.files if rel.startswith(f"{name}/"))
        files = [self.root / rel for rel in rels]
        return pl.read_parquet(files).sort(SORT_KEYS[name])

    def l1(self) -> pl.DataFrame:
        return self.table("l1_taq")

    def l2(self) -> pl.DataFrame:
        return self.table("l2_depth")

    def l3(self) -> pl.DataFrame:
        return self.table("l3_messages")

    def daily(self) -> pl.DataFrame:
        return self.table("daily_bars")


def _with_session(
    extra: dict[str, Any], session: TradingSession | None
) -> dict[str, Any]:
    if session is not None:
        extra[SESSION_KEY] = {
            "open_sec": session.open_sec,
            "length_sec": session.length_sec,
            "tz": session.tz,
        }
    return extra
