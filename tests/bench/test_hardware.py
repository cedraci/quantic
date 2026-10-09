from pathlib import Path

import pytest

from quantic.bench.config import load
from quantic.bench.hardware import run_hardware
from quantic.bench.keys import code_version, expand, gate_for
from quantic.bench.quantum_plan import is_hardware
from quantic.bench.runner import run
from quantic.bench.store import ERROR, OK, PENDING, Store
from quantic.solvers.quantum.ibm import LocalClient
from quantic.solvers.quantum.quota import Ledger, QuotaBusy

ROOT = Path(__file__).resolve().parents[2]

HW = """
[instances]
tiers = ["T0"]
dials = ["D2"]
seeds = [2026]

[quantum]
reps = 1
maxiter = 10
shots = 20
max_qubits = 18
gate_shots = 4
device = "ibm_fez"
workers = 1
per_job_cap_s = 30
min_job_s = 5

[solvers.qaoa_ibm]
encodings = ["binary"]
seeds = "0..1"
"""


@pytest.fixture
def study(tmp_path):
    path = tmp_path / "e.toml"
    path.write_text(HW, encoding="utf-8")
    exp = load(path)
    store = tmp_path / "s.duckdb"
    # Train and gate, no hardware yet.
    run(exp, root=ROOT, store=store, workers=1, allow_dirty=True, no_hardware=True)
    return exp, store, Ledger(tmp_path / "q.json")


def _lane(exp, store, ledger, client, **kw):
    code = code_version(ROOT)
    tasks = [(t.key(code), t) for t in expand(exp) if is_hardware(t)]
    from quantic.bench.angles import angles_dir_for

    with Store(store) as s:
        return run_hardware(tasks, store=s, settings=exp.quantum, code=code, where="here",
                            versions="{}", angles_dir=angles_dir_for(store),
                            client=client, ledger=ledger, **kw)


def _hw_rows(store):
    with Store(store, read_only=True) as s:
        runs = s.runs()
    return runs.filter(runs["solver"] == "qaoa_ibm")


def test_the_lane_runs_each_task_once_within_the_cap(study):
    exp, store, ledger = study
    client = LocalClient(usage_per_job_s=2.0)
    got = _lane(exp, store, ledger, client)
    assert got.n_written == 2 and got.stopped is None
    rows = _hw_rows(store)
    assert (rows["store_status"] == OK).all()
    assert rows["qpu_s"].to_list() == [2.0, 2.0] and (rows["total_s"] == 2.0).all()
    assert rows["job_id"].n_unique() == 2 and (rows["shots"] == 20).all()
    assert all(limit <= 30 for _, limit in client.submitted)
    assert [e.state for e in ledger.entries()] == ["done", "done"]
    again = _lane(exp, store, ledger, LocalClient())
    assert again.n_written == 0 and again.n_skipped_done == 2


def test_no_gate_no_hardware(study, tmp_path):
    exp, store, ledger = study
    with Store(store) as s:
        s.con.execute("DELETE FROM runs WHERE kind = 'gate'")
    client = LocalClient()
    got = _lane(exp, store, ledger, client)
    assert got.n_gate_refused == 2 and client.submitted == []


def test_the_ledger_stops_the_lane_and_keeps_what_is_done(study):
    exp, store, ledger = study
    # 6 s left: one job of at most 6 s; it uses 2 s, leaving 4 s, under min_job_s = 5.
    client = LocalClient(consumed=294.0, usage_per_job_s=2.0)
    got = _lane(exp, store, ledger, client)
    assert got.n_written == 1 and got.stopped and "left" in got.stopped
    assert client.submitted == [("job-0", 6)]


def test_a_job_submitted_before_a_crash_is_collected_not_resubmitted(study):
    exp, store, ledger = study

    class Crash(LocalClient):
        def counts(self, job_id):
            raise ConnectionError("the network went away")

    first = Crash()
    with pytest.raises(ConnectionError):
        _lane(exp, store, ledger, first)
    rows = _hw_rows(store)
    assert rows.height == 1 and rows["store_status"][0] == PENDING
    assert ledger.pending()[0].job_id == rows["job_id"][0]
    # The same session's client still holds the job: the resume collects it by id.
    first.counts = LocalClient.counts.__get__(first)
    got = _lane(exp, store, ledger, first)
    assert got.n_collected == 1
    assert len(first.submitted) == 2                 # the crashed job, then the second task
    assert (_hw_rows(store)["store_status"] == OK).all()


def test_a_failed_job_is_an_error_row_and_its_usage_is_kept(study):
    exp, store, ledger = study
    client = LocalClient(fail=("job-0",), usage_per_job_s=1.5)
    got = _lane(exp, store, ledger, client)
    rows = _hw_rows(store).sort("job_id")
    assert got.n_errors == 1
    assert rows["store_status"].to_list() == [ERROR, OK]
    assert "job-0" in rows["error"][0] and rows["qpu_s"][0] == 1.5


def test_repeat_hardware_resubmits_stored_tasks(study):
    exp, store, ledger = study
    _lane(exp, store, ledger, LocalClient())
    client = LocalClient()
    got = _lane(exp, store, ledger, client, repeat=True)
    assert len(client.submitted) == 2 and got.n_written == 2


def test_the_gate_names_the_same_coordinates(study):
    exp, _, _ = study
    hw = next(t for t in expand(exp) if is_hardware(t))
    gate = gate_for(hw, exp.quantum)
    assert (gate.tier, gate.dials, gate.instance_seed, gate.encoding) == \
        (hw.tier, hw.dials, hw.instance_seed, hw.encoding)


def test_a_locked_ledger_stops_the_lane_before_anything_is_submitted(study):
    exp, store, ledger = study
    lock = ledger.path.with_suffix(ledger.path.suffix + ".lock")
    lock.write_text("1", encoding="utf-8")
    client = LocalClient()
    with pytest.raises(QuotaBusy):
        _lane(exp, store, ledger, client)
    assert client.submitted == []
