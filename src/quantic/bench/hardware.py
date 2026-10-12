"""The hardware lane -- M5 design section 7.

One IBM job at a time, in the main process, which also holds the store. For
each task: skip it if it is stored under any commit; refuse it without a
passed gate; ask the ledger for an allowance; route; submit; record the job id
in the ledger and the store; then wait, collect, and replace the pending row
with the result. A run that dies while waiting leaves a pending row and a
pending ledger entry, and the next run collects that job instead of paying for
another one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from quantic.bench.angles import AngleCache, MissingAngles, angle_key
from quantic.bench.execute import fill_row
from quantic.bench.instances import InstanceCache
from quantic.bench.keys import Task, gate_for
from quantic.bench.store import ERROR, OK, PENDING, blank_row
from quantic.encoding.diagnostics import dynamic_range_ratio
from quantic.encoding.encode import encode
from quantic.solvers.quantum.backends import bits_from_counts, route, routed_stats
from quantic.solvers.quantum.ibm import JobFailed, RuntimeClient
from quantic.solvers.quantum.ising import qubo_hash, to_ising
from quantic.solvers.quantum.qaoa import QaoaCollected
from quantic.solvers.quantum.quota import Ledger, QuotaExhausted
from quantic.solvers.qubo.sampler import encoding_named


@dataclass
class HardwareSummary:
    n_written: int = 0
    n_errors: int = 0
    n_collected: int = 0
    n_gate_refused: int = 0
    n_skipped_done: int = 0
    stopped: str | None = None


class _Lane:
    def __init__(self, *, store, settings, code, where, versions, angles_dir, client,
                 ledger, now):
        self.store, self.settings, self.code = store, settings, code
        self.where, self.versions, self.now = where, versions, now
        self.angles = AngleCache(Path(angles_dir))
        self.client, self.ledger = client, ledger
        self.instances = InstanceCache()
        self.summary = HardwareSummary()

    def encoded(self, task: Task):
        instance = self.instances.get(task.tier, task.dials, task.instance_seed)
        enc = encode(instance, encoding_named(task.encoding), n_zeta_bits=task.zeta_bits)
        return instance, enc

    def angles_for(self, task: Task, enc):
        params = json.loads(task.params)
        key = angle_key(qubo_hash(enc.qubo), params["reps"], params["alpha"],
                        params["maxiter"])
        angles = self.angles.get(key)
        if angles is None:
            raise MissingAngles(f"no trained angles {key[:12]} for {task.tier}-"
                                f"{task.dials}-{task.instance_seed}")
        return key, params, angles

    def finish(self, task: Task, row: dict, job_id: str) -> None:
        """Collect ``job_id`` and replace the pending row with what it returned."""
        instance, enc = self.encoded(task)
        key, params, angles = self.angles_for(task, enc)
        started = time.perf_counter()
        try:
            counts = self.client.counts(job_id)
        except JobFailed as exc:
            # None when IBM never gives the figure: the row says unknown, not 0 s,
            # and the ledger counts the job at its limit.
            usage = self.client.usage_s(job_id)
            row.update(store_status=ERROR, error=f"JobFailed: {exc}", qpu_s=usage,
                       total_s=usage)
            self.store.replace(row)
            self.ledger.mark_done(job_id, usage)
            self.summary.n_errors += 1
            self.summary.n_written += 1
            return
        bits = bits_from_counts(counts, enc.n_vars)
        result = QaoaCollected(task.encoding, seed=task.solver_seed or 0,
                               zeta_bits=task.zeta_bits, angles=angles,
                               shots=len(bits), bits=bits,
                               device=params["device"]).solve_encoded(enc)
        out, samples, trace = fill_row(row, result, instance)
        usage = self.client.usage_s(job_id)
        out.update(store_status=OK, qpu_s=usage, total_s=usage,
                   queue_s=self.client.queue_s(job_id), angle_key=key,
                   reps=params["reps"], shots=len(bits), backend="ibm",
                   device=params["device"], job_id=job_id, n_vars=enc.n_vars,
                   dynamic_range=dynamic_range_ratio(enc.qubo),
                   timings={"total_s": usage,
                            "collect_wall_s": time.perf_counter() - started})
        self.store.replace(out, samples, trace)
        self.ledger.mark_done(job_id, usage)
        self.summary.n_written += 1

    def submit(self, key: str, task: Task) -> None:
        instance, enc = self.encoded(task)
        angle, params, angles = self.angles_for(task, enc)
        q = self.settings
        allowance = self.ledger.allowance(
            cap_s=q.quota_cap_s, per_job_cap_s=q.per_job_cap_s, min_job_s=q.min_job_s,
            ibm_consumed_s=self.client.consumed_s(), now=self.now(),
        )
        backend = self.client.backend(params["device"])
        self.client.use(backend)
        routed = route(to_ising(enc.qubo), angles, backend)
        depth, two = routed_stats(routed)
        # Reserve first: if the process dies inside submit, the allowance still
        # counts against the window. confirm() swaps in the real job id after.
        reservation = f"reserved:{key}"
        self.ledger.add_pending(reservation, params["device"], key, allowance,
                                now=self.now(), task_identity=task.identity())
        job_id = self.client.submit(routed, q.shots, allowance)
        self.ledger.confirm(reservation, job_id)
        row = self.pending_row(key, task, job_id, instance, angle, params, (depth, two))
        # replace, not write: under --repeat-hardware, or after a killed row, the
        # key may already be stored, and the new job's row must take its place.
        self.store.replace(row)
        self.finish(task, row, job_id)

    def pending_row(self, key, task, job_id, instance, angle, params, routed):
        depth, two = routed
        row = blank_row(key, task, self.code, self.where, self.versions)
        row.update(store_status=PENDING, job_id=job_id, device=params["device"],
                   backend="ibm", angle_key=angle, reps=params["reps"],
                   shots=self.settings.shots, routed_depth=depth, routed_2q=two,
                   instance_hash=instance.content_hash, notional=instance.notional())
        return row

    def reconcile(self, entry, key: str, task: Task) -> None:
        """A job the ledger knows but the store never recorded: record it, collect it."""
        instance, enc = self.encoded(task)
        angle, params, _ = self.angles_for(task, enc)
        row = self.pending_row(key, task, entry.job_id, instance, angle, params,
                               (None, None))
        self.store.replace(row)
        self.finish(task, row, entry.job_id)


def _task_of(row: dict) -> Task:
    fields = ("kind", "tier", "dials", "instance_seed", "solver", "encoding", "zeta_bits",
              "solver_seed", "budget_s", "max_samples", "params")
    return Task(**{f: row[f] for f in fields})


def run_hardware(tasks, *, store, settings, code, where, versions, angles_dir,
                 client=None, ledger=None, repeat: bool = False,
                 now=time.time) -> HardwareSummary:
    lane = _Lane(store=store, settings=settings, code=code, where=where,
                 versions=versions, angles_dir=angles_dir,
                 client=client if client is not None else RuntimeClient(),
                 ledger=ledger if ledger is not None else Ledger(), now=now)
    with lane.ledger.lock():
        handled: set[str] = set()
        for row in lane.store.pending_hardware().iter_rows(named=True):
            lane.finish(_task_of(row), dict(row), row["job_id"])
            lane.summary.n_collected += 1
            handled.add(row["identity"])
        by_key = {key: task for key, task in tasks}
        by_identity = {task.identity(): task for _, task in tasks}
        for entry in lane.ledger.pending():
            if entry.job_id.startswith("reserved:") or lane.store.has_job(entry.job_id):
                continue
            # By identity when the entry has one: a job confirmed under an older
            # commit is still this task's job, to be adopted, not paid for again.
            if entry.task_identity is not None:
                task = by_identity.get(entry.task_identity)
            else:
                task = by_key.get(entry.task_key.split("#r")[0])
            if task is None:
                continue
            lane.reconcile(entry, entry.task_key, task)
            lane.summary.n_collected += 1
            handled.add(task.identity())
        done = set() if repeat else lane.store.identities("qaoa_ibm")
        for key, task in tasks:
            if task.identity() in done or task.identity() in handled:
                lane.summary.n_skipped_done += 1
                continue
            if not lane.store.gate_passed(gate_for(task, settings).identity()):
                lane.summary.n_gate_refused += 1
                print(f"[hardware] no passed gate for {task.tier}-{task.dials}-"
                      f"{task.instance_seed} {task.encoding}: run the qaoa_noisy gate first",
                      flush=True)
                continue
            if repeat:
                key = lane.store.repeat_key(key, task.identity())
            try:
                lane.submit(key, task)
            except QuotaExhausted as exc:
                lane.summary.stopped = str(exc)
                break
    return lane.summary
