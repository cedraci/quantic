"""Plan and run an experiment into a store, resumably.

``plan`` says how many tasks an experiment holds, how many a store already
has, and roughly how long the rest will take. ``run`` executes the rest. Each
finished task is written as it arrives; a run interrupted for any reason is
finished by running the same command again.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from quantic.bench.angles import angles_dir_for
from quantic.bench.config import QUANTUM_SOLVERS, Experiment, QuantumSettings
from quantic.bench.execute import machine, packages
from quantic.bench.keys import (
    REF_QAOA_ANGLES,
    REFERENCE,
    CodeVersion,
    Task,
    code_version,
    expand,
)
from quantic.bench.pool import Died, Finished, Late, Outcome, run_tasks
from quantic.bench.quantum_plan import QuboSizes, cap, is_hardware, is_quantum
from quantic.bench.store import ERROR, KILLED, OK, Store, blank_row

# A budgeted run is killed at KILL_FACTOR * budget + KILL_MARGIN_S seconds.
KILL_FACTOR = 3.0
KILL_MARGIN_S = 30.0
# References are unbudgeted proofs; this only stops one that has hung.
REFERENCE_DEADLINE_S = 3600.0
# For the estimate only: a reference's typical cost.
_REFERENCE_ESTIMATE_S = 1.0
# A progress line every this many written tasks, or this many seconds, whichever first.
PROGRESS_EVERY = 100
PROGRESS_S = 300.0


class DirtyTree(RuntimeError):
    """Tracked files differ from the commit the rows would name."""


@dataclass(frozen=True)
class Plan:
    n_tasks: int
    n_done: int
    n_todo: int
    estimate_s: float
    n_workers: int
    n_over_max: int = 0
    n_over_noisy: int = 0
    n_hardware: int = 0


@dataclass(frozen=True)
class Summary:
    n_written: int
    n_killed: int
    n_errors: int
    hardware_stop: str | None = None


def deadline_s(task: Task, margin_s: float = KILL_MARGIN_S,
               settings: QuantumSettings | None = None) -> float:
    if task.solver == REF_QAOA_ANGLES:
        return (settings or QuantumSettings()).train_deadline_s
    if task.solver in QUANTUM_SOLVERS:
        return (settings or QuantumSettings()).run_deadline_s
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
    capped = cap(tasks, experiment.quantum, QuboSizes())
    todo = [(k, t) for k, t in capped.kept if k not in done]
    return capped, todo


def plan(
    experiment: Experiment, *, root: Path, store: Path | None = None,
    workers: int | None = None,
) -> Plan:
    code = code_version(root)
    n = workers or default_workers()
    capped, todo = _todo(experiment, code, store)
    q = experiment.quantum or QuantumSettings()

    def estimate(t: Task) -> float:
        if t.solver in QUANTUM_SOLVERS:
            return q.run_deadline_s / 10
        return _REFERENCE_ESTIMATE_S if t.kind == REFERENCE else t.budget_s

    seconds = sum(estimate(t) for _, t in todo)
    return Plan(
        n_tasks=len(capped.kept), n_done=len(capped.kept) - len(todo), n_todo=len(todo),
        estimate_s=seconds / n, n_workers=n, n_over_max=capped.over_max,
        n_over_noisy=capped.over_noisy,
        n_hardware=sum(is_hardware(t) for _, t in todo),
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
    progress_every: int = PROGRESS_EVERY,
    progress_s: float = PROGRESS_S,
    no_hardware: bool = False,
    repeat_hardware: bool = False,
    hardware_only: bool = False,
    client=None,
    ledger=None,
) -> Summary:
    """Run every task the store does not hold yet, printing progress as it goes.

    ``hardware_only`` runs the IBM lane alone: no classical, training or
    sampling task, so a commit made after a ``no_hardware`` run does not redo
    the simulations first. The lane matches gates by identity and angles by
    cache key, neither of which names the commit.

    ``stop_after`` and ``kill_margin_s`` exist for the tests: the first
    interrupts a run partway, the second makes a kill take seconds, not 30.
    """
    if hardware_only and no_hardware:
        raise ValueError("hardware_only and no_hardware together would run nothing")
    code = code_version(root)
    if code.dirty and not allow_dirty:
        raise DirtyTree(
            "tracked files differ from HEAD, so rows would name a commit that did "
            "not produce them. Commit, or pass --allow-dirty (recorded on every row)"
        )
    capped, todo = _todo(experiment, code, store)
    if stop_after is not None:
        todo = todo[:stop_after]
    if hardware_only:
        todo = [(k, t) for k, t in todo if is_hardware(t)]
    classical = [(k, t) for k, t in todo if not is_quantum(t)]
    training = [(k, t) for k, t in todo if t.solver == REF_QAOA_ANGLES]
    sampling = [(k, t) for k, t in todo
                if is_quantum(t) and t.solver != REF_QAOA_ANGLES and not is_hardware(t)]
    # --repeat-hardware: every planned hardware task, stored or not.
    hardware = [(k, t) for k, t in (capped.kept if repeat_hardware else todo)
                if is_hardware(t)]

    where = machine()
    versions = json.dumps(packages(), sort_keys=True)
    counts = {"written": 0, "killed": 0, "errors": 0}
    started = time.monotonic()
    last = {"at": started, "written": 0}
    hardware_stop: str | None = None

    def progress() -> None:
        now = time.monotonic()
        if (counts["written"] - last["written"] < progress_every
                and now - last["at"] < progress_s):
            return
        last.update(at=now, written=counts["written"])
        per_min = counts["written"] / max((now - started) / 60, 1e-9)
        left = len(todo) - counts["written"]
        print(
            f"[bench] {dt.datetime.now():%H:%M:%S} written {counts['written']} "
            f"(killed {counts['killed']}, errors {counts['errors']}), todo {left}, "
            f"rate {per_min:.1f}/min, eta {left / per_min / 60:.1f} h",
            flush=True,
        )

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
                limit = deadline_s(task, kill_margin_s, experiment.quantum)
                row["error"] = f"killed after {limit:.1f} s"
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
            progress()

        context = {"angles_dir": str(angles_dir_for(store))}
        q = experiment.quantum
        for jobs, n in ((classical, workers or default_workers()),
                        (training, q.workers if q else 1),
                        (sampling, q.workers if q else 1)):
            if jobs:
                run_tasks(
                    jobs, code=code, root=root, n_workers=n,
                    deadline_s=lambda task: deadline_s(task, kill_margin_s, q),
                    on_done=on_done, context=context,
                )
        if (hardware or out.pending_hardware().height) and not no_hardware:
            from quantic.bench.hardware import run_hardware

            lane = run_hardware(
                hardware, store=out, settings=q, code=code, where=where,
                versions=versions, angles_dir=angles_dir_for(store),
                client=client, ledger=ledger, repeat=repeat_hardware,
            )
            counts["written"] += lane.n_written
            counts["errors"] += lane.n_errors
            hardware_stop = lane.stopped
    return Summary(
        n_written=counts["written"], n_killed=counts["killed"], n_errors=counts["errors"],
        hardware_stop=hardware_stop,
    )
