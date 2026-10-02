"""Worker processes that can be killed one at a time.

``concurrent.futures`` cannot stop one task without stopping the pool, and a
sampler written in Python can ignore its budget. Each worker here is a
long-lived process with its own pair of pipes: the main process sends it one
task at a time, notes the deadline, and terminates and replaces it if the
deadline passes. A terminated process can corrupt the pipe it was writing to,
so every worker's pipes are its own and are thrown away with it.

Workers start with ``spawn`` on every platform, so a test on Linux sees what a
run on Windows sees.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing.connection import wait

from quantic.bench.keys import CodeVersion, Task

_READY = "ready"
_POLL_S = 0.2
_MAX_FAILED_STARTS = 3


class WorkerStartError(RuntimeError):
    """Workers keep dying before they can take a task."""


def _worker(conn, code: CodeVersion) -> None:
    from quantic.bench.execute import Executor

    executor = Executor(code)
    conn.send(_READY)
    while True:
        item = conn.recv()
        if item is None:
            return
        key, task = item
        conn.send((key, executor.run(key, task)))


class _Slot:
    def __init__(self, ctx, code: CodeVersion) -> None:
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, code), daemon=True)
        self.process.start()
        child.close()
        self.ready = False
        self.job: tuple[str, Task, float] | None = None

    def messages(self) -> list:
        """Everything the worker has sent and the main process has not read."""
        out = []
        try:
            while self.conn.poll():
                out.append(self.conn.recv())
        except (EOFError, OSError):
            pass  # the worker is gone; what it sent before going was read
        return out

    def stop(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join()
        self.conn.close()


@dataclass(frozen=True)
class Finished:
    """The worker answered: ``payload`` is what ``Executor.run`` returned."""

    payload: tuple


@dataclass(frozen=True)
class Late:
    """The deadline passed with no answer, so the worker was terminated."""


@dataclass(frozen=True)
class Died:
    """The worker process ended on its own, before its deadline and with no answer."""

    exitcode: int | None


Outcome = Finished | Late | Died
OnDone = Callable[[str, Task, Outcome], None]


class _StartFailures:
    """Consecutive workers that died before they were ready; a ready one resets it.

    Over a multi-day run, a worker now and then dies at start for reasons of
    its own. Only an unbroken series says that no worker can start.
    """

    def __init__(self, limit: int = _MAX_FAILED_STARTS) -> None:
        self.limit = limit
        self.count = 0

    def ready(self) -> None:
        self.count = 0

    def failed(self, exitcode: int | None) -> None:
        self.count += 1
        if self.count > self.limit:
            raise WorkerStartError(
                f"{self.count} workers in a row died before they were ready; "
                f"the last exited with code {exitcode}"
            )


def run_tasks(
    jobs: Iterable[tuple[str, Task]],
    *,
    code: CodeVersion,
    n_workers: int,
    deadline_s: Callable[[Task], float],
    on_done: OnDone,
) -> None:
    """Run every job; call ``on_done`` in the main process as each one ends."""
    ctx = mp.get_context("spawn")
    pending = deque(jobs)
    slots = [_Slot(ctx, code) for _ in range(n_workers)]
    starts = _StartFailures()

    def take(slot: _Slot, message) -> None:
        if message == _READY:
            slot.ready = True
            starts.ready()
        elif slot.job is not None and message[0] == slot.job[0]:
            key, task, _ = slot.job
            slot.job = None
            on_done(key, task, Finished(message[1]))

    def drain(slot: _Slot) -> None:
        for message in slot.messages():
            take(slot, message)

    try:
        while pending or any(s.job for s in slots):
            for i, slot in enumerate(slots):
                if slot.ready and slot.job is None and pending:
                    key, task = pending.popleft()
                    try:
                        slot.conn.send((key, task))
                    except OSError:
                        # The worker died while idle. The task never started:
                        # it goes back to the front, and no row is written.
                        pending.appendleft((key, task))
                        slot.stop()
                        slots[i] = _Slot(ctx, code)
                        continue
                    slot.job = (key, task, time.monotonic() + deadline_s(task))

            by_conn = {s.conn: s for s in slots}
            for conn in wait(list(by_conn), timeout=_POLL_S):
                drain(by_conn[conn])

            for i, slot in enumerate(slots):
                died = not slot.process.is_alive()
                late = slot.job is not None and time.monotonic() > slot.job[2]
                if not (died or late):
                    continue
                # While on_done wrote another task, this one may have answered
                # in time. A result that arrived counts, whatever the clock says.
                drain(slot)
                if not died and slot.job is None:
                    continue
                if died and not slot.ready:
                    starts.failed(slot.process.exitcode)
                job, exitcode = slot.job, slot.process.exitcode
                slot.stop()
                slots[i] = _Slot(ctx, code)
                if job is not None:
                    on_done(job[0], job[1], Died(exitcode) if died else Late())
    finally:
        for slot in slots:
            if slot.ready and slot.job is None and slot.process.is_alive():
                with contextlib.suppress(OSError):
                    slot.conn.send(None)
            slot.stop()
