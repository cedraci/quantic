"""A local registry mapping bundle IDs to paths and content hashes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from quantic.data.bundle import DatasetBundle

DEFAULT_CATALOG_PATH = Path("data/catalog.json")


class DuplicateBundleError(ValueError):
    """Raised when a bundle_id is reused for different content."""


@dataclass(frozen=True)
class CatalogEntry:
    bundle_id: str
    path: str
    content_hash: str
    provenance: str
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    granularities: tuple[str, ...]


class Catalog:
    def __init__(self, path: Path = DEFAULT_CATALOG_PATH) -> None:
        self.path = Path(path)

    def entries(self) -> list[CatalogEntry]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            CatalogEntry(
                **{**r, "symbols": tuple(r["symbols"]), "granularities": tuple(r["granularities"])}
            )
            for r in raw
        ]

    def get(self, bundle_id: str) -> CatalogEntry:
        for entry in self.entries():
            if entry.bundle_id == bundle_id:
                return entry
        raise KeyError(f"no bundle {bundle_id!r} in catalog {self.path}")

    def register(self, bundle: DatasetBundle) -> CatalogEntry:
        m = bundle.manifest
        entry = CatalogEntry(
            bundle_id=m.bundle_id,
            path=str(Path(bundle.root).resolve()),
            content_hash=m.content_hash,
            provenance=m.provenance,
            symbols=m.symbols,
            start_date=m.start_date,
            end_date=m.end_date,
            granularities=m.granularities,
        )
        existing = self.entries()
        for prior in existing:
            if prior.bundle_id != entry.bundle_id:
                continue
            if prior.content_hash == entry.content_hash:
                return prior
            raise DuplicateBundleError(
                f"bundle_id {entry.bundle_id!r} already registered with content_hash "
                f"{prior.content_hash}, refusing to overwrite with {entry.content_hash}"
            )

        existing.append(entry)
        payload = [
            {**asdict(e), "symbols": list(e.symbols), "granularities": list(e.granularities)}
            for e in sorted(existing, key=lambda e: e.bundle_id)
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return entry
