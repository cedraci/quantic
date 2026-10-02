"""Hand-built store rows, for tests that need a store without running anything."""

from __future__ import annotations

from pathlib import Path

from quantic.bench.keys import REF_BRACKET, REFERENCE, RUN, CodeVersion, Task
from quantic.bench.store import OK, Store, blank_row

CODE = CodeVersion(commit="c0ffee", lock_hash="10c4", dirty=False)


def row(task: Task, **values) -> dict:
    out = blank_row(task.key(CODE), task, CODE, "here", "{}")
    out.update(store_status=OK, instance_hash=f"{task.tier}/{task.dials}", **values)
    return out


def ref(dials: str, lower: float, upper: float, tier: str = "T0") -> dict:
    task = Task(kind=REFERENCE, tier=tier, dials=dials, instance_seed=1, solver=REF_BRACKET)
    return row(task, status="optimal", lower=lower, upper=upper, total=upper)


def run_row(dials: str, solver: str, seed: int, total: float | None, *, budget=1.0,
            feasible=True, encoding=None, tier="T0", **extra) -> dict:
    task = Task(kind=RUN, tier=tier, dials=dials, instance_seed=1, solver=solver,
                encoding=encoding, solver_seed=seed, budget_s=budget)
    return row(task, status="heuristic", total=total, feasible=feasible,
               total_s=0.5, total_bps=total, **extra)


def write(path: Path, rows: list[dict]) -> Path:
    with Store(path) as s:
        for r in rows:
            s.write(r)
    return path
