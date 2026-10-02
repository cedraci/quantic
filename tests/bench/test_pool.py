import time
from pathlib import Path

import pytest

from quantic.bench import pool as pool_module
from quantic.bench.keys import RUN, CodeVersion, Task
from quantic.bench.pool import Finished, WorkerStartError, _StartFailures, run_tasks
from quantic.bench.store import OK

ROOT = Path(__file__).resolve().parents[2]
CODE = CodeVersion(commit="c0ffee", lock_hash="10c4", dirty=False)
STAND_INS = "tests.bench.stand_ins"


def _task(solver: str, seed: int = 0) -> Task:
    return Task(kind=RUN, tier="T0", dials="none", instance_seed=2026, solver=solver,
                solver_seed=seed, budget_s=1.0)


def test_a_result_that_arrives_while_another_is_written_is_credited_not_killed():
    """The quick task's writer sleeps past the slow task's deadline; the slow
    task answered in time, so it is done, whatever the clock says on reading."""
    quick, slow = _task("almgren_chriss"), _task(f"{STAND_INS}:Nap", seed=20)
    outcomes = {}

    def on_done(key, task, outcome):
        outcomes[key] = outcome
        if key == "quick":
            time.sleep(8.0)

    run_tasks([("quick", quick), ("slow", slow)], code=CODE, n_workers=2,
              deadline_s=lambda t: 5.0, on_done=on_done)
    assert isinstance(outcomes["slow"], Finished)
    assert outcomes["slow"].payload[0]["store_status"] == OK


def test_a_task_whose_worker_died_idle_is_sent_to_a_new_worker():
    first, second = _task(f"{STAND_INS}:DieWhenIdle"), _task("almgren_chriss")
    outcomes = {}

    def on_done(key, task, outcome):
        outcomes[key] = outcome
        if key == "first":
            time.sleep(2.0)  # the worker dies meanwhile, idle

    run_tasks([("first", first), ("second", second)], code=CODE, n_workers=1,
              deadline_s=lambda t: 30.0, on_done=on_done)
    assert isinstance(outcomes["second"], Finished)
    assert outcomes["second"].payload[0]["store_status"] == OK


class _RefusesSend:
    """A worker's pipe whose far end has gone: ``send`` fails as on a dead worker."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def send(self, item) -> None:
        raise BrokenPipeError("the worker is gone")

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_a_task_that_cannot_be_sent_goes_to_a_new_worker(monkeypatch):
    made = []

    class FirstSlotIsDead(pool_module._Slot):
        def __init__(self, *args) -> None:
            super().__init__(*args)
            made.append(self)
            if len(made) == 1:
                self.conn = _RefusesSend(self.conn)

    monkeypatch.setattr(pool_module, "_Slot", FirstSlotIsDead)
    outcomes = {}
    run_tasks([("only", _task("almgren_chriss"))], code=CODE, n_workers=1,
              deadline_s=lambda t: 30.0,
              on_done=lambda key, task, outcome: outcomes.update({key: outcome}))
    assert len(made) == 2
    assert outcomes["only"].payload[0]["store_status"] == OK


def test_only_consecutive_start_failures_stop_a_run():
    starts = _StartFailures(limit=3)
    for _ in range(10):  # a worker that fails now and then, over days
        starts.failed(1)
        starts.failed(1)
        starts.ready()
    for _ in range(3):
        starts.failed(1)
    with pytest.raises(WorkerStartError, match="4 workers in a row"):
        starts.failed(7)
