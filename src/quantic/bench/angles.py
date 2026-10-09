"""Trained QAOA angles, cached in files beside the store -- M5 design section 5.4.

The store has one writer, the main process; workers that train or read angles
use this directory instead. A file is written whole or not at all.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib import metadata
from pathlib import Path

from quantic.solvers.quantum.train import ANGLE_RULE, Angles


class MissingAngles(RuntimeError):
    """A QAOA run's angles were never trained, or their training failed."""


def angle_key(qubo_hash: str, reps: int, alpha: float, maxiter: int) -> str:
    body = {"qubo": qubo_hash, "reps": reps, "alpha": alpha, "maxiter": maxiter,
            "rule": ANGLE_RULE, "numpy": metadata.version("numpy"),
            "scipy": metadata.version("scipy")}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def angles_dir_for(store: Path) -> Path:
    store = Path(store)
    return store.with_name(store.stem + ".angles")


class AngleCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Angles | None:
        path = self._path(key)
        if not path.exists():
            return None
        return Angles.from_json(json.loads(path.read_text(encoding="utf-8"))["angles"])

    def put(self, key: str, angles: Angles, meta: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"angles": angles.to_json(), **meta}, indent=1),
                       encoding="utf-8")
        os.replace(tmp, path)
