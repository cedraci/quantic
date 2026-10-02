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

    def stop(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join()
        self.conn.close()


# (key, task, payload or None, killed)
OnDone = Callable[[str, Task, tuple | None, bool], None]


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
    failed_starts = 0
    try:
        while pending or any(s.job for s in slots):
            for slot in slots:
                if slot.ready and slot.job is None and pending:
                    key, task = pending.popleft()
                    slot.job = (key, task, time.monotonic() + deadline_s(task))
                    slot.conn.send((key, task))

            by_conn = {s.conn: s for s in slots}
            for conn in wait(list(by_conn), timeout=_POLL_S):
                slot = by_conn[conn]
                try:
                    message = conn.recv()
                except (EOFError, OSError):
                    message = None
                if message == _READY:
                    slot.ready = True
                elif message is not None and slot.job is not None:
                    key, task, _ = slot.job
                    sent_key, payload = message
                    if sent_key == key:
                        slot.job = None
                        on_done(key, task, payload, False)

            now = time.monotonic()
            for i, slot in enumerate(slots):
                died = not slot.process.is_alive()
                late = slot.job is not None and now > slot.job[2]
                if died and not slot.ready:
                    failed_starts += 1
                    if failed_starts > _MAX_FAILED_STARTS:
                        raise WorkerStartError(
                            f"{failed_starts} workers died before they were ready; "
                            f"the last exited with code {slot.process.exitcode}"
                        )
                if late or died:
                    job = slot.job
                    slot.stop()
                    slots[i] = _Slot(ctx, code)
                    if job is not None:
                        on_done(job[0], job[1], None, True)
    finally:
        for slot in slots:
            if slot.ready and slot.job is None and slot.process.is_alive():
                with contextlib.suppress(OSError):
                    slot.conn.send(None)
            slot.stop()
