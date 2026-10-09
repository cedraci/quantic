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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

QUOTA_CEILING_S = 300.0
WINDOW_S = 28 * 24 * 3600
PENDING = "pending"
DONE = "done"


class QuotaExhausted(RuntimeError):
    """Not enough of the window is left for one more job."""


@dataclass(frozen=True)
class Entry:
    job_id: str
    device: str
    task_key: str
    submitted_at: float
    max_execution_time_s: int
    state: str
    usage_s: float | None


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
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"jobs": [asdict(e) for e in entries]}, indent=1),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    def used_s(self, now: float) -> float:
        total = 0.0
        for e in self.entries():
            if now - e.submitted_at > WINDOW_S:
                continue
            total += e.max_execution_time_s if e.state == PENDING else (e.usage_s or 0.0)
        return total

    def allowance(self, *, cap_s: float, per_job_cap_s: float, min_job_s: float,
                  ibm_consumed_s: float | None, now: float) -> int:
        if cap_s > QUOTA_CEILING_S:
            raise ValueError(
                f"a cap of {cap_s} s is above the ceiling of {QUOTA_CEILING_S:g} s "
                "(M5 decision D4); change QUOTA_CEILING_S in code, not in config"
            )
        used = max(self.used_s(now), ibm_consumed_s or 0.0)
        left = cap_s - used
        if left < min_job_s:
            raise QuotaExhausted(
                f"{used:.1f} s of the {cap_s:g} s window are used and {max(left, 0):.1f} s "
                f"are left, less than one job's minimum of {min_job_s:g} s"
            )
        return int(math.floor(min(per_job_cap_s, left)))

    def add_pending(self, job_id: str, device: str, task_key: str,
                    max_execution_time_s: int, now: float) -> None:
        entry = Entry(job_id=job_id, device=device, task_key=task_key, submitted_at=now,
                      max_execution_time_s=int(max_execution_time_s), state=PENDING,
                      usage_s=None)
        self._save([*self.entries(), entry])

    def mark_done(self, job_id: str, usage_s: float) -> None:
        entries = self.entries()
        for i, e in enumerate(entries):
            if e.job_id == job_id:
                entries[i] = replace(e, state=DONE, usage_s=float(usage_s))
                self._save(entries)
                return
        raise KeyError(f"no job {job_id!r} in the ledger at {self.path}")
