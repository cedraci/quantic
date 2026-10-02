"""What a run is, and the key that names it.

A key is a SHA-256 over the run's coordinates and the code that runs it: the
git commit and the hash of ``uv.lock``. The same key always means the same
computation, so a store can be resumed by skipping the keys it holds, and rows
from different code never share a key.

The instance enters through its coordinates -- tier, dial combination, seed --
which the commit pins to one instance; its content hash is written on the row.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from quantic.bench.config import Experiment

RUN = "run"
REFERENCE = "reference"

# Reference solvers: computed once per instance, never budgeted.
REF_BRACKET = "ref:exact_pwl_milp"
REF_BRUTE_FORCE = "ref:brute_force"
REF_QUBO_MINIMUM = "ref:qubo_minimum"


@dataclass(frozen=True)
class CodeVersion:
    commit: str
    lock_hash: str
    dirty: bool


@dataclass(frozen=True)
class Task:
    kind: str
    tier: str
    dials: str
    instance_seed: int
    solver: str
    encoding: str | None = None
    zeta_bits: int | None = None
    solver_seed: int | None = None
    budget_s: float | None = None
    max_samples: int | None = None

    def key(self, code: CodeVersion) -> str:
        payload = {**asdict(self), "commit": code.commit, "lock_hash": code.lock_hash}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_version(root: Path) -> CodeVersion:
    """The commit, the lock file's hash, and whether tracked files differ."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    lock = (Path(root) / "uv.lock").read_bytes()
    return CodeVersion(
        commit=git("rev-parse", "HEAD"),
        lock_hash=hashlib.sha256(lock).hexdigest(),
        dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
    )


def expand(experiment: Experiment) -> list[Task]:
    """Every task the experiment asks for: references first, then runs."""
    references: list[Task] = []
    runs: list[Task] = []
    qubo_encodings = sorted({
        (e, p.zeta_bits) for p in experiment.solvers for e in p.encodings if e is not None
    })
    for tier in experiment.tiers:
        for seed in experiment.seeds:
            for dials in experiment.dials:
                at = {"tier": tier, "dials": dials, "instance_seed": seed}
                references.append(Task(kind=REFERENCE, solver=REF_BRACKET, **at))
                if tier == "T0":
                    references.append(Task(kind=REFERENCE, solver=REF_BRUTE_FORCE, **at))
                if tier in experiment.qubo_minimum_tiers:
                    for encoding, bits in qubo_encodings:
                        references.append(Task(
                            kind=REFERENCE, solver=REF_QUBO_MINIMUM,
                            encoding=encoding, zeta_bits=bits, **at,
                        ))
                for plan in experiment.solvers:
                    for encoding in plan.encodings:
                        for solver_seed in plan.seeds:
                            for budget in plan.budgets_s:
                                runs.append(Task(
                                    kind=RUN, solver=plan.name, encoding=encoding,
                                    zeta_bits=plan.zeta_bits, solver_seed=solver_seed,
                                    budget_s=budget, max_samples=plan.max_samples, **at,
                                ))
    return references + runs
