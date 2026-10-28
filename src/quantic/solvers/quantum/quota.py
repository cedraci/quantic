"""The IBM quota ledger -- M5 design section 7.1, parent spec section 7.5.

The quota belongs to the account, not to a study, so the ledger lives outside
every store. A pending job counts at its ``max_execution_time``, which IBM
enforces, so even a runaway job cannot take the window past the cap. IBM's own
figure, when it can be read, is trusted only when it is the larger one.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

QUOTA_CEILING_S = 300.0
WINDOW_S = 28 * 24 * 3600
PENDING = "pending"
DONE = "done"


class QuotaExhausted(RuntimeError):
    """Not enough of the window is left for one more job."""


class QuotaBusy(RuntimeError):
    """The ledger is locked by another hardware run."""


@dataclass(frozen=True)
class Entry:
    job_id: str
    device: str
    task_key: str
    submitted_at: float
    max_execution_time_s: int
    state: str
    usage_s: float | None
    # The task's identity, which names no commit: a job reserved under one commit
    # is still that task's job under the next. None in ledgers written before it.
    task_identity: str | None = None


def default_path() -> Path:
    override = os.environ.get("QUANTIC_QUOTA_PATH")
    return Path(override) if override else Path.home() / ".quantic" / "ibm-quota.json"


class Ledger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()

    def entries(self) -> list[Entry]:
        if not self.path.exists():
            return []
        body = json.loads(self.path.read_text(encoding="utf-8"))
        return [Entry(**job) for job in body["jobs"]]

    def pending(self) -> list[Entry]:
        return [e for e in self.entries() if e.state == PENDING]

    def _save(self, entries: list[Entry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"jobs": [asdict(e) for e in entries]}, indent=1),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    @contextmanager
    def lock(self):
        """Acquire an exclusive lock on the ledger, held for the duration of the context.

        Raises QuotaBusy if another process holds the lock.
        """
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
        except FileExistsError:
            raise QuotaBusy(
                f"another hardware run holds {lock_path}; if none does, delete the file"
            ) from None

        try:
            yield
        finally:
            lock_path.unlink(missing_ok=True)

    def used_s(self, now: float) -> float:
        total = 0.0
        for e in self.entries():
            if now - e.submitted_at > WINDOW_S:
                continue
            if e.state == PENDING:
                total += e.max_execution_time_s
            else:  # DONE
                total += e.usage_s if e.usage_s is not None else e.max_execution_time_s
        return total

    def allowance(self, *, cap_s: float, per_job_cap_s: float, min_job_s: float,
                  ibm_consumed_s: float | None, now: float) -> int:
        # Validate inputs
        if not math.isfinite(cap_s) or cap_s < 0:
            raise ValueError(f"cap_s must be finite and non-negative, got {cap_s}")
        if cap_s > QUOTA_CEILING_S:
            raise ValueError(
                f"a cap of {cap_s} s is above the ceiling of {QUOTA_CEILING_S:g} s "
                "(M5 decision D4); change QUOTA_CEILING_S in code, not in config"
            )
        if not math.isfinite(per_job_cap_s) or per_job_cap_s < 0:
            raise ValueError(
                f"per_job_cap_s must be finite and non-negative, got {per_job_cap_s}"
            )
        if not math.isfinite(min_job_s) or min_job_s < 1:
            raise ValueError(
                f"min_job_s must be finite and >= 1, got {min_job_s}"
            )
        if ibm_consumed_s is not None and (
            not math.isfinite(ibm_consumed_s) or ibm_consumed_s < 0
        ):
            raise ValueError(
                f"ibm_consumed_s must be finite and non-negative, got {ibm_consumed_s}"
            )

        used = max(self.used_s(now), ibm_consumed_s or 0.0)
        left = cap_s - used
        grant = int(math.floor(min(per_job_cap_s, left)))

        if grant < min_job_s:
            raise QuotaExhausted(
                f"{used:.1f} s of the {cap_s:g} s window are used and {max(left, 0):.1f} s "
                f"are left, less than one job's minimum of {min_job_s:g} s"
            )
        return grant

    def add_pending(self, job_id: str, device: str, task_key: str,
                    max_execution_time_s: int, now: float,
                    task_identity: str | None = None) -> None:
        max_time_float = float(max_execution_time_s)
        if not math.isfinite(max_time_float) or max_time_float < 0:
            raise ValueError(
                f"max_execution_time_s must be finite and non-negative, "
                f"got {max_time_float}"
            )
        max_time_int = int(max_time_float)
        if max_time_int < 1:
            raise ValueError(f"max_execution_time_s must be >= 1, got {max_time_int}")
        entry = Entry(job_id=job_id, device=device, task_key=task_key, submitted_at=now,
                      max_execution_time_s=max_time_int, state=PENDING,
                      usage_s=None, task_identity=task_identity)
        self._save([*self.entries(), entry])

    def mark_done(self, job_id: str, usage_s: float | None) -> None:
        """``None`` is IBM never giving the figure: the job then counts at its limit."""
        usage_float = None if usage_s is None else float(usage_s)
        if usage_float is not None and (not math.isfinite(usage_float) or usage_float < 0):
            raise ValueError(f"usage_s must be finite and non-negative, got {usage_float}")
        entries = self.entries()
        for i, e in enumerate(entries):
            if e.job_id == job_id:
                entries[i] = replace(e, state=DONE, usage_s=usage_float)
                self._save(entries)
                return
        raise KeyError(f"no job {job_id!r} in the ledger at {self.path}")

    def confirm(self, reservation_id: str, job_id: str) -> None:
        """Give a reservation made before submitting its real job id."""
        entries = self.entries()
        for i in range(len(entries) - 1, -1, -1):
            if entries[i].job_id == reservation_id:
                entries[i] = replace(entries[i], job_id=job_id)
                self._save(entries)
                return
        raise KeyError(f"no reservation {reservation_id!r} in the ledger at {self.path}")
