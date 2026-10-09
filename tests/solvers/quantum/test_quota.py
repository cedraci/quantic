import json

import pytest

from quantic.solvers.quantum.quota import (
    QUOTA_CEILING_S,
    WINDOW_S,
    Ledger,
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
