import json

import pytest

from quantic.solvers.quantum.quota import (
    QUOTA_CEILING_S,
    WINDOW_S,
    Ledger,
    QuotaBusy,
    QuotaExhausted,
    default_path,
)

NOW = 1_800_000_000.0


def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "q.json")


def test_an_empty_ledger_grants_the_per_job_cap(tmp_path):
    got = ledger(tmp_path).allowance(cap_s=300, per_job_cap_s=60, min_job_s=10,
                                     ibm_consumed_s=None, now=NOW)
    assert got == 60


def test_pending_jobs_count_at_their_limit_and_done_jobs_at_their_usage(tmp_path):
    led = ledger(tmp_path)
    led.add_pending("a", "ibm_fez", "k1", 60, now=NOW - 100)
    led.add_pending("b", "ibm_fez", "k2", 60, now=NOW - 50)
    led.mark_done("a", 3.5)
    assert led.used_s(NOW) == pytest.approx(63.5)
    assert [e.job_id for e in led.pending()] == ["b"]


def test_jobs_older_than_the_window_do_not_count(tmp_path):
    led = ledger(tmp_path)
    led.add_pending("old", "ibm_fez", "k", 60, now=NOW - WINDOW_S - 1)
    led.mark_done("old", 50.0)
    assert led.used_s(NOW) == 0.0


def test_the_larger_of_ibm_and_the_ledger_is_used(tmp_path):
    led = ledger(tmp_path)
    led.add_pending("a", "ibm_fez", "k", 20, now=NOW)
    led.mark_done("a", 20.0)
    assert led.allowance(cap_s=300, per_job_cap_s=500, min_job_s=10,
                         ibm_consumed_s=250.0, now=NOW) == 50
    assert led.allowance(cap_s=300, per_job_cap_s=500, min_job_s=10,
                         ibm_consumed_s=5.0, now=NOW) == 280


def test_the_allowance_never_takes_used_past_the_cap(tmp_path):
    led = ledger(tmp_path)
    led.add_pending("a", "ibm_fez", "k", 60, now=NOW)
    led.mark_done("a", 271.4)
    got = led.allowance(cap_s=300, per_job_cap_s=60, min_job_s=10,
                        ibm_consumed_s=None, now=NOW)
    assert got == 28
    assert led.used_s(NOW) + got <= 300


def test_below_the_minimum_job_the_ledger_refuses(tmp_path):
    led = ledger(tmp_path)
    led.add_pending("a", "ibm_fez", "k", 60, now=NOW)
    led.mark_done("a", 295.0)
    with pytest.raises(QuotaExhausted, match="295"):
        led.allowance(cap_s=300, per_job_cap_s=60, min_job_s=10, ibm_consumed_s=None, now=NOW)


def test_a_cap_above_the_ceiling_is_refused(tmp_path):
    with pytest.raises(ValueError, match="300"):
        ledger(tmp_path).allowance(cap_s=QUOTA_CEILING_S + 1, per_job_cap_s=60,
                                   min_job_s=10, ibm_consumed_s=None, now=NOW)


def test_the_file_survives_a_reopen_and_is_plain_json(tmp_path):
    led = ledger(tmp_path)
    led.add_pending("a", "ibm_fez", "k", 60, now=NOW)
    again = Ledger(tmp_path / "q.json")
    assert again.pending()[0].job_id == "a"
    body = json.loads((tmp_path / "q.json").read_text(encoding="utf-8"))
    assert body["jobs"][0]["state"] == "pending"
    assert not list(tmp_path.glob("*.tmp"))


def test_marking_an_unknown_job_is_refused(tmp_path):
    with pytest.raises(KeyError, match="nope"):
        ledger(tmp_path).mark_done("nope", 1.0)


def test_the_default_path_honours_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTIC_QUOTA_PATH", str(tmp_path / "x.json"))
    assert default_path() == tmp_path / "x.json"


# Concurrency tests (a)
def test_the_lock_is_held_inside_with_and_released_after(tmp_path):
    led = ledger(tmp_path)
    lock_path = tmp_path / "q.json.lock"

    assert not lock_path.exists()
    with led.lock():
        assert lock_path.exists()
    assert not lock_path.exists()


def test_a_second_lock_while_the_first_is_held_raises_quota_busy(tmp_path):
    led1 = ledger(tmp_path)
    led2 = ledger(tmp_path)

    with led1.lock(), pytest.raises(QuotaBusy), led2.lock():
        pass


def test_the_lock_is_released_even_when_the_body_raises(tmp_path):
    led = ledger(tmp_path)
    lock_path = tmp_path / "q.json.lock"

    try:
        with led.lock():
            raise RuntimeError("test error")
    except RuntimeError:
        pass

    assert not lock_path.exists()


# Non-finite and negative inputs tests (b)
@pytest.mark.parametrize("bad_cap", [float("nan"), float("inf"), -1.0])
def test_invalid_cap_s_raises_value_error(tmp_path, bad_cap):
    with pytest.raises(ValueError, match="cap_s"):
        ledger(tmp_path).allowance(cap_s=bad_cap, per_job_cap_s=60, min_job_s=10,
                                   ibm_consumed_s=None, now=NOW)


@pytest.mark.parametrize("bad_per_job", [float("nan"), float("inf"), -1.0])
def test_invalid_per_job_cap_s_raises_value_error(tmp_path, bad_per_job):
    with pytest.raises(ValueError, match="per_job_cap_s"):
        ledger(tmp_path).allowance(cap_s=300, per_job_cap_s=bad_per_job, min_job_s=10,
                                   ibm_consumed_s=None, now=NOW)


@pytest.mark.parametrize("bad_min", [float("nan"), float("inf"), -1.0, 0.5])
def test_invalid_min_job_s_raises_value_error(tmp_path, bad_min):
    with pytest.raises(ValueError, match="min_job_s"):
        ledger(tmp_path).allowance(cap_s=300, per_job_cap_s=60, min_job_s=bad_min,
                                   ibm_consumed_s=None, now=NOW)


@pytest.mark.parametrize("bad_ibm", [float("nan"), float("inf"), -1.0])
def test_invalid_ibm_consumed_s_raises_value_error(tmp_path, bad_ibm):
    with pytest.raises(ValueError, match="ibm_consumed_s"):
        ledger(tmp_path).allowance(cap_s=300, per_job_cap_s=60, min_job_s=10,
                                   ibm_consumed_s=bad_ibm, now=NOW)


@pytest.mark.parametrize("bad_usage", [float("nan"), float("inf"), -1.0])
def test_invalid_usage_s_in_mark_done_raises_value_error(tmp_path, bad_usage):
    led = ledger(tmp_path)
    led.add_pending("a", "ibm_fez", "k", 60, now=NOW)
    with pytest.raises(ValueError, match="usage_s"):
        led.mark_done("a", bad_usage)


@pytest.mark.parametrize("bad_max_time", [0, -1, float("nan")])
def test_invalid_max_execution_time_s_in_add_pending_raises_value_error(tmp_path, bad_max_time):
    with pytest.raises(ValueError, match="max_execution_time_s"):
        ledger(tmp_path).add_pending("a", "ibm_fez", "k", bad_max_time, now=NOW)


# Done entries with None usage test (c)
def test_a_done_entry_with_none_usage_counts_at_its_limit(tmp_path):
    led = ledger(tmp_path)
    # Write an entry directly to JSON with usage_s: null and state: done
    led.path.parent.mkdir(parents=True, exist_ok=True)
    entry_data = {
        "job_id": "test_job",
        "device": "ibm_fez",
        "task_key": "k",
        "submitted_at": NOW,
        "max_execution_time_s": 42,
        "state": "done",
        "usage_s": None,
    }
    led.path.write_text(
        json.dumps({"jobs": [entry_data]}, indent=1),
        encoding="utf-8"
    )

    # Should count at max_execution_time_s (42), not 0
    assert led.used_s(NOW) == pytest.approx(42.0)


# Grant calculation test (d)
def test_grant_below_minimum_is_refused(tmp_path):
    led = ledger(tmp_path)
    led.add_pending("a", "ibm_fez", "k", 60, now=NOW)
    led.mark_done("a", 289.1)
    # left = 300 - 289.1 = 10.9, min_job_s = 10.5
    # grant = floor(min(60, 10.9)) = 10 < 10.5
    with pytest.raises(QuotaExhausted, match="10.9"):
        led.allowance(cap_s=300, per_job_cap_s=60, min_job_s=10.5,
                      ibm_consumed_s=None, now=NOW)
