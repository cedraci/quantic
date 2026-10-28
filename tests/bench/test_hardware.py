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


def _study(tmp_path, text=HW):
    path = tmp_path / "e.toml"
    path.write_text(text, encoding="utf-8")
    exp = load(path)
    store = tmp_path / "s.duckdb"
    # Train and gate, no hardware yet.
    run(exp, root=ROOT, store=store, workers=1, allow_dirty=True, no_hardware=True)
    return exp, store, Ledger(tmp_path / "q.json")


@pytest.fixture
def study(tmp_path):
    return _study(tmp_path)


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


def test_a_usage_ibm_never_gives_is_unknown_and_counts_at_its_limit(study):
    exp, store, ledger = study
    client = LocalClient(usage_pending=True)
    got = _lane(exp, store, ledger, client)
    assert got.n_written == 2 and got.n_errors == 0
    rows = _hw_rows(store)
    assert (rows["store_status"] == OK).all()
    assert rows["qpu_s"].to_list() == [None, None]
    assert rows["total_s"].to_list() == [None, None]
    entries = ledger.entries()
    assert [(e.state, e.usage_s) for e in entries] == [("done", None), ("done", None)]
    limits = sum(e.max_execution_time_s for e in entries)
    assert limits == 60 and ledger.used_s(entries[-1].submitted_at) == limits


def test_a_failed_job_whose_usage_ibm_never_gives_counts_at_its_limit(study):
    exp, store, ledger = study
    client = LocalClient(fail=("job-0",), usage_pending=True)
    got = _lane(exp, store, ledger, client)
    rows = _hw_rows(store).sort("job_id")
    assert got.n_errors == 1 and rows["store_status"].to_list() == [ERROR, OK]
    assert rows["qpu_s"].to_list() == [None, None]
    assert [e.usage_s for e in ledger.entries()] == [None, None]


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


def _again(exp, store, ledger, client, **kw):
    return run(exp, root=ROOT, store=store, workers=1, allow_dirty=True,
               client=client, ledger=ledger, **kw)


def test_bench_run_collects_a_pending_job_instead_of_submitting_another(tmp_path):
    exp, store, ledger = _study(tmp_path, HW.replace('"0..1"', '"0..0"'))

    class Crash(LocalClient):
        def counts(self, job_id):
            raise ConnectionError("the network went away")

    client = Crash()
    with pytest.raises(ConnectionError):
        _again(exp, store, ledger, client)
    assert _hw_rows(store)["store_status"].to_list() == [PENDING]
    client.counts = LocalClient.counts.__get__(client)
    got = _again(exp, store, ledger, client)
    assert got.n_written == 1
    assert _hw_rows(store)["store_status"].to_list() == [OK]
    assert len(client.submitted) == 1


def test_hardware_only_runs_the_lane_alone_under_a_later_commit(study, monkeypatch):
    from quantic.bench import runner as runner_module
    from quantic.bench.keys import CodeVersion

    exp, store, ledger = study
    with Store(store, read_only=True) as s:
        before = s.runs()
    later = CodeVersion(commit="f" * 40, lock_hash="e" * 64, dirty=False)
    monkeypatch.setattr(runner_module, "code_version", lambda root: later)
    client = LocalClient()
    got = _again(exp, store, ledger, client, hardware_only=True)
    assert got.n_written == 2 and len(client.submitted) == 2
    with Store(store, read_only=True) as s:
        after = s.runs()
    hw = after.filter(after["solver"] == "qaoa_ibm")
    assert (hw["store_status"] == OK).all() and (hw["commit"] == later.commit).all()
    others = after.filter(after["solver"] != "qaoa_ibm")
    assert others.height == before.height                 # no simulator row added
    assert later.commit not in others["commit"].to_list()


def test_hardware_only_with_no_hardware_is_refused(study):
    exp, store, ledger = study
    with pytest.raises(ValueError, match="no_hardware"):
        _again(exp, store, ledger, LocalClient(), hardware_only=True, no_hardware=True)


def test_repeat_hardware_keeps_the_earlier_row_under_the_same_commit(study):
    exp, store, ledger = study
    client = LocalClient()
    _again(exp, store, ledger, client)
    first = _hw_rows(store).sort("key")
    _again(exp, store, ledger, client, repeat_hardware=True)
    rows = _hw_rows(store)
    assert rows.height == 4 and (rows["store_status"] == OK).all()
    assert rows.group_by("identity").len()["len"].to_list() == [2, 2]
    assert len(client.submitted) == 4
    kept = rows.filter(~rows["key"].str.contains("#r")).sort("key")
    assert kept["job_id"].to_list() == first["job_id"].to_list()


def test_the_allowance_is_reserved_before_the_job_is_submitted(study):
    exp, store, ledger = study

    class Refuse(LocalClient):
        def submit(self, routed, shots, max_execution_time_s):
            raise RuntimeError("submit died")

    with pytest.raises(RuntimeError):
        _lane(exp, store, ledger, Refuse())
    [entry] = ledger.pending()
    assert entry.job_id.startswith("reserved:") and entry.max_execution_time_s == 30
    assert ledger.used_s(entry.submitted_at) == 30
    # A normal job afterwards: real ids only, none of its own reservations left.
    _lane(exp, store, ledger, LocalClient())
    ids = [e.job_id for e in ledger.entries()]
    assert sum(i.startswith("reserved:") for i in ids) == 1 and len(ids) == 3


def test_a_ledger_job_with_no_store_row_is_collected_not_resubmitted(study):
    exp, store, ledger = study
    client = LocalClient()

    class Flaky:
        def __init__(self, inner):
            self.inner, self.armed = inner, True

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def replace(self, *a, **k):
            if self.armed:
                self.armed = False
                raise RuntimeError("died before the store row")
            return self.inner.replace(*a, **k)

    code = code_version(ROOT)
    tasks = [(t.key(code), t) for t in expand(exp) if is_hardware(t)]
    from quantic.bench.angles import angles_dir_for

    with Store(store) as s, pytest.raises(RuntimeError):
        run_hardware(tasks, store=Flaky(s), settings=exp.quantum, code=code, where="here",
                     versions="{}", angles_dir=angles_dir_for(store), client=client,
                     ledger=ledger)
    assert _hw_rows(store).height == 0 and len(ledger.pending()) == 1
    got = _lane(exp, store, ledger, client)
    assert got.n_collected == 1
    assert len(client.submitted) == 2
    rows = _hw_rows(store)
    assert (rows["store_status"] == OK).all() and rows["job_id"].n_unique() == 2


def test_a_job_confirmed_under_an_older_commit_is_adopted_by_identity(study):
    from dataclasses import replace

    exp, store, ledger = study
    client = LocalClient()

    class Flaky:
        def __init__(self, inner):
            self.inner, self.armed = inner, True

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def replace(self, *a, **k):
            if self.armed:
                self.armed = False
                raise RuntimeError("died before the store row")
            return self.inner.replace(*a, **k)

    code = code_version(ROOT)
    tasks = [(t.key(code), t) for t in expand(exp) if is_hardware(t)]
    from quantic.bench.angles import angles_dir_for

    with Store(store) as s, pytest.raises(RuntimeError):
        run_hardware(tasks, store=Flaky(s), settings=exp.quantum, code=code, where="here",
                     versions="{}", angles_dir=angles_dir_for(store), client=client,
                     ledger=ledger)
    [entry] = ledger.pending()
    assert entry.task_identity == tasks[0][1].identity()
    # As if confirmed under commit A: another key, the same identity.
    ledger._save([replace(entry, task_key="a" * 64)])
    got = _lane(exp, store, ledger, client)
    assert got.n_collected == 1
    assert len(client.submitted) == 2                 # nothing new for the adopted task
    rows = _hw_rows(store)
    assert (rows["store_status"] == OK).all() and rows.height == 2
    assert entry.job_id in rows["job_id"].to_list()


def test_a_ledger_written_before_task_identity_still_loads(tmp_path):
    import json

    path = tmp_path / "q.json"
    path.write_text(json.dumps({"jobs": [{
        "job_id": "j", "device": "ibm_fez", "task_key": "k", "submitted_at": 1.0,
        "max_execution_time_s": 30, "state": "pending", "usage_s": None}]}),
        encoding="utf-8")
    [entry] = Ledger(path).entries()
    assert entry.task_identity is None and entry.task_key == "k"
