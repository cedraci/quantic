"""IBM Quantum behind a small interface -- M5 design section 7.3.

``RuntimeClient`` is the account; ``LocalClient`` is the same calls in Qiskit
Runtime's local mode against a fake device, for the tests and the gate. Nothing
here decides whether a job may run: that is the ledger's and the gate's job.
"""

from __future__ import annotations

from typing import Protocol

from quantic.solvers.quantum.backends import fake_backend

_USAGE_KEYS = ("usage_consumed_seconds",)


class NoAccount(RuntimeError):
    """No IBM Quantum account is saved on this machine."""


class JobFailed(RuntimeError):
    """IBM reports the job failed or was cancelled."""


def consumed_from_usage(answer) -> float | None:
    """Seconds consumed in the window, from ``service.usage()``, or ``None``.

    The shape of that answer is IBM's; a key that is missing or unreadable is
    ``None``, so the ledger falls back to its own count and never to zero.
    """
    if not isinstance(answer, dict):
        return None
    for key in _USAGE_KEYS:
        if key in answer:
            try:
                return float(answer[key])
            except (TypeError, ValueError):
                return None
    return None


class IbmClient(Protocol):
    def consumed_s(self) -> float | None: ...
    def backend(self, device: str): ...
    def submit(self, routed, shots: int, max_execution_time_s: int) -> str: ...
    def counts(self, job_id: str) -> dict[str, int]: ...
    def usage_s(self, job_id: str) -> float: ...
    def queue_s(self, job_id: str) -> float | None: ...


def _sampler(mode, max_execution_time_s: int):
    from qiskit_ibm_runtime.executor_sampler import Sampler

    return Sampler(mode=mode, options={"max_execution_time": int(max_execution_time_s)})


def _status_name(job) -> str:
    try:
        status = job.status()
    except Exception:
        return ""
    return str(getattr(status, "name", status)).split(".")[-1].upper()


class RuntimeClient:
    def __init__(self) -> None:
        from qiskit_ibm_runtime import QiskitRuntimeService

        try:
            self.service = QiskitRuntimeService()
        except Exception as exc:  # AccountNotFoundError and its relatives
            raise NoAccount(
                "no IBM Quantum account is saved. Save one once with "
                "QiskitRuntimeService.save_account(channel='ibm_quantum_platform', "
                f"token=...). Qiskit said: {exc}"
            ) from exc
        self._backend = None

    def consumed_s(self) -> float | None:
        try:
            return consumed_from_usage(self.service.usage())
        except Exception:
            return None

    def backend(self, device: str):
        return self.service.backend(device)

    def use(self, backend) -> None:
        """The backend the next submissions go to; the lane sets it before each job."""
        self._backend = backend

    def submit(self, routed, shots: int, max_execution_time_s: int) -> str:
        if self._backend is None:
            raise RuntimeError("call use(backend) before submit")
        job = _sampler(self._backend, max_execution_time_s).run([routed], shots=shots)
        return job.job_id()

    def counts(self, job_id: str) -> dict[str, int]:
        job = self.service.job(job_id)
        try:
            result = job.result()
        except Exception as exc:
            # Only a job IBM reports as errored or cancelled has failed; a dropped
            # connection must leave the job pending to be collected later.
            if _status_name(job) in ("ERROR", "CANCELLED"):
                raise JobFailed(f"job {job_id} failed: {exc}") from exc
            raise
        return dict(result[0].data.meas.get_counts())

    def usage_s(self, job_id: str) -> float:
        return float(self.service.job(job_id).usage())

    def queue_s(self, job_id: str) -> float | None:
        try:
            stamps = self.service.job(job_id).metrics().get("timestamps", {})
            from datetime import datetime

            created = datetime.fromisoformat(stamps["created"].replace("Z", "+00:00"))
            running = datetime.fromisoformat(stamps["running"].replace("Z", "+00:00"))
            return (running - created).total_seconds()
        except Exception:
            return None


class LocalClient:
    def __init__(self, *, usage_per_job_s: float = 2.0, consumed: float | None = None,
                 fail: tuple[str, ...] = ()) -> None:
        self.usage_per_job_s, self.consumed, self.fail = usage_per_job_s, consumed, set(fail)
        self.submitted: list[tuple[str, int]] = []
        self._jobs: dict[str, object] = {}
        self._backend = None

    def consumed_s(self) -> float | None:
        """Grows with the jobs run, as IBM's figure would; ``None`` stays ``None``."""
        if self.consumed is None:
            return None
        return self.consumed + self.usage_per_job_s * len(self.submitted)

    def backend(self, device: str):
        self._backend = fake_backend(device)
        return self._backend

    def use(self, backend) -> None:
        self._backend = backend

    def submit(self, routed, shots: int, max_execution_time_s: int) -> str:
        job_id = f"job-{len(self.submitted)}"
        self._jobs[job_id] = _sampler(self._backend, max_execution_time_s).run(
            [routed], shots=shots)
        self.submitted.append((job_id, int(max_execution_time_s)))
        return job_id

    def counts(self, job_id: str) -> dict[str, int]:
        if job_id in self.fail:
            raise JobFailed(f"job {job_id} failed: told to fail")
        return dict(self._jobs[job_id].result()[0].data.meas.get_counts())

    def usage_s(self, job_id: str) -> float:
        return self.usage_per_job_s

    def queue_s(self, job_id: str) -> float | None:
        return 0.0
