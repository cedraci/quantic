"""Bundle manifests and deterministic content hashing.

Every benchmark result pins a bundle's ``content_hash``, so this module is
load-bearing for the reproducibility claims in spec section 9.3.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "manifest.json"
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def compute_content_hash(file_hashes: Mapping[str, str]) -> str:
    """Aggregate per-file hashes into one order-independent digest.

    Paths are sorted, and each path and hash is NUL-terminated so that no two
    distinct mappings can serialise to the same byte stream.
    """
    h = hashlib.sha256()
    for path in sorted(file_hashes):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(file_hashes[path].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass(frozen=True)
class Manifest:
    schema_version: str
    bundle_id: str
    created_utc: str
    provenance: str
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    granularities: tuple[str, ...]
    files: dict[str, str]
    content_hash: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = compute_content_hash(self.files)
        if self.content_hash != expected:
            raise ValueError(
                f"content_hash does not match files: expected {expected}, got {self.content_hash}"
            )

    def to_json(self) -> str:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["granularities"] = list(self.granularities)
        return json.dumps(payload, indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Manifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["symbols"] = tuple(payload["symbols"])
        payload["granularities"] = tuple(payload["granularities"])
        return cls(**payload)
