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
