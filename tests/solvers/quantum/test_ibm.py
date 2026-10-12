import pytest

from quantic.solvers.quantum.backends import route
from quantic.solvers.quantum.ibm import JobFailed, LocalClient, consumed_from_usage
from quantic.solvers.quantum.ising import to_ising
from quantic.solvers.quantum.train import train
from tests.solvers.quantum.test_ising import random_qubo


def _routed(client):
    qubo = random_qubo(4, 1)
    angles = train(qubo, reps=1, alpha=0.25, maxiter=5)
    return route(to_ising(qubo), angles, client.backend("ibm_fez"))


def test_a_local_job_returns_counts_usage_and_its_limit():
    client = LocalClient(usage_per_job_s=2.5)
    job = client.submit(_routed(client), shots=50, max_execution_time_s=30)
    counts = client.counts(job)
    assert sum(counts.values()) == 50
    assert all(len(k.replace(" ", "")) == 4 for k in counts)
    assert client.usage_s(job) == 2.5
    assert client.submitted == [(job, 30)]


def test_a_failed_job_raises_with_its_message():
    client = LocalClient(fail=("job-0",))
    job = client.submit(_routed(client), shots=10, max_execution_time_s=10)
    assert job == "job-0"
    with pytest.raises(JobFailed, match="job-0"):
        client.counts(job)


@pytest.mark.parametrize("answer,expected", [
    ({"usage_consumed_seconds": 12.5, "usage_limit_seconds": 600}, 12.5),
    ({"usage_consumed_seconds": "40"}, 40.0),
    ({"something_else": 3}, None),
    (None, None),
    ("not a dict", None),
])
def test_an_unreadable_usage_answer_is_none_not_zero(answer, expected):
    assert consumed_from_usage(answer) == expected


class _StubJob:
    def __init__(self, status):
        self._status = status

    def result(self):
        raise ConnectionError("the network went away")

    def status(self):
        return self._status


class _StubService:
    def __init__(self, status):
        self._job = _StubJob(status)

    def job(self, job_id):
        return self._job


def _runtime_client(status):
    from quantic.solvers.quantum.ibm import RuntimeClient

    client = RuntimeClient.__new__(RuntimeClient)
    client.service = _StubService(status)
    return client


class _MetricsJob:
    def __init__(self, answers):
        self.answers, self.asked = list(answers), 0

    def metrics(self):
        self.asked += 1
        answer = self.answers[min(self.asked, len(self.answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer


class _MetricsService:
    def __init__(self, job):
        self._job = job

    def job(self, job_id):
        return self._job


def _usage_client(answers, deadline_s=0.05):
    from quantic.solvers.quantum.ibm import RuntimeClient

    client = RuntimeClient.__new__(RuntimeClient)
    client.service = _MetricsService(_MetricsJob(answers))
    client.usage_poll_s, client.usage_deadline_s = 0.001, deadline_s
    return client


_PENDING_USAGE = {"usage": {"status": "pending", "qpu_charge_time_seconds": 0}}


def test_a_usage_that_stays_pending_is_none_not_zero():
    client = _usage_client([_PENDING_USAGE])
    assert client.usage_s("j") is None
    assert client.service.job("j").asked > 1          # it was asked again


def test_a_usage_is_read_once_ibm_has_computed_it():
    final = {"usage": {"status": "completed", "qpu_charge_time_seconds": 3.25}}
    client = _usage_client([_PENDING_USAGE, ConnectionError("blip"), final], deadline_s=60)
    assert client.usage_s("j") == 3.25


@pytest.mark.parametrize("usage", [
    {"status": "completed"},
    {"status": "completed", "qpu_charge_time_seconds": None},
    {"status": "completed", "qpu_charge_time_seconds": float("nan")},
    {"status": "completed", "qpu_charge_time_seconds": -1},
])
def test_a_final_usage_that_is_missing_or_unreadable_is_none(usage):
    assert _usage_client([{"usage": usage}]).usage_s("j") is None


def test_a_local_client_can_leave_usage_pending():
    assert LocalClient(usage_pending=True).usage_s("job-0") is None


@pytest.mark.parametrize("status", ["ERROR", "CANCELLED"])
def test_a_job_that_errored_or_was_cancelled_is_a_failed_job(status):
    with pytest.raises(JobFailed):
        _runtime_client(status).counts("j")


@pytest.mark.parametrize("status", ["RUNNING", "QUEUED"])
def test_any_other_exception_leaves_the_job_pending(status):
    with pytest.raises(ConnectionError):
        _runtime_client(status).counts("j")
