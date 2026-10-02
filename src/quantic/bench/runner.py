"""Plan and run an experiment into a store, resumably.

``plan`` says how many tasks an experiment holds, how many a store already
has, and roughly how long the rest will take. ``run`` executes the rest. Each
finished task is written as it arrives; a run interrupted for any reason is
finished by running the same command again.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from quantic.bench.config import Experiment
from quantic.bench.execute import machine, packages
from quantic.bench.keys import REFERENCE, CodeVersion, Task, code_version, expand
from quantic.bench.pool import Died, Finished, Late, Outcome, run_tasks
from quantic.bench.store import ERROR, KILLED, OK, Store, blank_row

# A budgeted run is killed at KILL_FACTOR * budget + KILL_MARGIN_S seconds.
KILL_FACTOR = 3.0
KILL_MARGIN_S = 30.0
# References are unbudgeted proofs; this only stops one that has hung.
REFERENCE_DEADLINE_S = 3600.0
# For the estimate only: a reference's typical cost.
_REFERENCE_ESTIMATE_S = 1.0


class DirtyTree(RuntimeError):
    """Tracked files differ from the commit the rows would name."""


@dataclass(frozen=True)
class Plan:
    n_tasks: int
    n_done: int
    n_todo: int
    estimate_s: float
    n_workers: int


@dataclass(frozen=True)
class Summary:
    n_written: int
    n_killed: int
    n_errors: int


def deadline_s(task: Task, margin_s: float = KILL_MARGIN_S) -> float:
    if task.kind == REFERENCE or task.budget_s is None:
        return REFERENCE_DEADLINE_S
    return KILL_FACTOR * task.budget_s + margin_s


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def _todo(experiment: Experiment, code: CodeVersion, store: Path | None):
    tasks = [(t.key(code), t) for t in expand(experiment)]
    done: set[str] = set()
    if store is not None and Path(store).exists():
        with Store(store, read_only=True) as s:
            done = s.done_keys()
    return tasks, [(k, t) for k, t in tasks if k not in done]


def plan(
    experiment: Experiment, *, root: Path, store: Path | None = None,
    workers: int | None = None,
) -> Plan:
    code = code_version(root)
    n = workers or default_workers()
    tasks, todo = _todo(experiment, code, store)
    seconds = sum(
        _REFERENCE_ESTIMATE_S if t.kind == REFERENCE else t.budget_s for _, t in todo
    )
    return Plan(
        n_tasks=len(tasks), n_done=len(tasks) - len(todo), n_todo=len(todo),
        estimate_s=seconds / n, n_workers=n,
    )


def run(
    experiment: Experiment,
    *,
    root: Path,
    store: Path,
    workers: int | None = None,
    allow_dirty: bool = False,
    stop_after: int | None = None,
    kill_margin_s: float = KILL_MARGIN_S,
) -> Summary:
    """Run every task the store does not hold yet.

    ``stop_after`` and ``kill_margin_s`` exist for the tests: the first
    interrupts a run partway, the second makes a kill take seconds, not 30.
    """
    code = code_version(root)
    if code.dirty and not allow_dirty:
        raise DirtyTree(
            "tracked files differ from HEAD, so rows would name a commit that did "
            "not produce them. Commit, or pass --allow-dirty (recorded on every row)"
        )
    _, todo = _todo(experiment, code, store)
    if stop_after is not None:
        todo = todo[:stop_after]

    where = machine()
    versions = json.dumps(packages(), sort_keys=True)
    counts = {"written": 0, "killed": 0, "errors": 0}

    with Store(store) as out:
        def on_done(key: str, task: Task, outcome: Outcome) -> None:
            if isinstance(outcome, Finished):
                row, samples, trace = outcome.payload
                out.write(row, samples, trace)
                if row["store_status"] != OK:
                    counts["errors"] += 1
            elif isinstance(outcome, Late):
                row = blank_row(key, task, code, where, versions)
                row["store_status"] = KILLED
                row["error"] = f"killed after {deadline_s(task, kill_margin_s):.1f} s"
                out.write(row)
                counts["killed"] += 1
            elif isinstance(outcome, Died):
                # A crash, not the clock: no answer, and no kill either.
                row = blank_row(key, task, code, where, versions)
                row["store_status"] = ERROR
                row["error"] = f"WorkerDied: exit code {outcome.exitcode}"
                out.write(row)
                counts["errors"] += 1
            else:
                raise TypeError(f"unknown outcome {outcome!r}")
            counts["written"] += 1

        run_tasks(
            todo, code=code, root=root, n_workers=workers or default_workers(),
            deadline_s=lambda task: deadline_s(task, kill_margin_s), on_done=on_done,
        )
    return Summary(
        n_written=counts["written"], n_killed=counts["killed"], n_errors=counts["errors"]
    )
