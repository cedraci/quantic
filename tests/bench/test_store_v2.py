import duckdb
import pytest

from quantic.bench.keys import GATE, RUN, Task
from quantic.bench.store import ERROR, OK, PENDING, RUN_COLUMNS, Store, blank_row
from tests.bench.rows import CODE

V1_RUNS = RUN_COLUMNS[: RUN_COLUMNS.index("params")]


def _task(**kw):
    base = dict(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="qaoa_ibm",
                encoding="binary", zeta_bits=8, solver_seed=0, max_samples=10, params="{}")
    return Task(**{**base, **kw})


def _row(task, **values):
    row = blank_row(task.key(CODE), task, CODE, "here", "{}")
    row.update({"store_status": OK, **values})
    return row


def _v1_store(path):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
    con.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    cols = ", ".join(f"{c} VARCHAR" if c not in ("instance_seed",) else f"{c} BIGINT"
                     for c in V1_RUNS)
    con.execute(f"CREATE TABLE runs ({cols})")
    con.execute("CREATE TABLE samples (key VARCHAR, digest VARCHAR, count BIGINT, "
                "energy DOUBLE, classification VARCHAR, total DOUBLE)")
    con.execute("CREATE TABLE traces (key VARCHAR, seconds DOUBLE, best_energy DOUBLE)")
    values = {c: None for c in V1_RUNS}
    values.update(key="old", kind="run", tier="T0", dials="none", instance_seed=1,
                  solver="cpsat", store_status="ok")
    con.execute(f"INSERT INTO runs VALUES ({', '.join('?' for _ in V1_RUNS)})",
                [values[c] for c in V1_RUNS])
    con.close()


def test_a_schema_1_store_is_read_and_migrated(tmp_path):
    path = tmp_path / "v1.duckdb"
    _v1_store(path)
    with Store(path, read_only=True) as s:
        runs = s.runs()
    assert runs.height == 1 and "identity" in runs.columns and runs["identity"][0] is None
    with Store(path) as s:
        s.write(_row(_task()))
        assert s.runs().height == 2
    with Store(path, read_only=True) as s:
        assert s.con.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "2"


def test_rows_carry_their_identity_and_parameters(tmp_path):
    task = _task()
    with Store(tmp_path / "s.duckdb") as s:
        s.write(_row(task))
        got = s.runs().row(0, named=True)
    assert got["identity"] == task.identity() and got["params"] == "{}"


def test_replace_swaps_a_pending_row_for_its_result(tmp_path):
    task = _task()
    with Store(tmp_path / "s.duckdb") as s:
        s.write(_row(task, store_status=PENDING, job_id="j1"))
        s.replace(_row(task, job_id="j1", qpu_s=2.0),
                  [("d1", 3, -1.0, "feasible", 5.0)], [(0.0, -1.0)])
        runs, samples = s.runs(), s.samples()
    assert runs.height == 1 and runs["store_status"][0] == OK and runs["qpu_s"][0] == 2.0
    assert samples.height == 1


def test_identities_count_ok_pending_and_error_rows(tmp_path):
    a, b, c, d = (_task(solver_seed=i) for i in range(4))
    with Store(tmp_path / "s.duckdb") as s:
        s.write(_row(a))
        s.write(_row(b, store_status=PENDING))
        s.write(_row(c, store_status=ERROR))
        s.write(_row(d, store_status="killed"))
        assert s.identities("qaoa_ibm") == {a.identity(), b.identity(), c.identity()}
        assert s.pending_hardware()["identity"].to_list() == [b.identity()]


def test_a_gate_passes_only_when_ok_and_heuristic(tmp_path):
    good = _task(kind=GATE, solver="qaoa_noisy", max_samples=32)
    bad = _task(kind=GATE, solver="qaoa_noisy", max_samples=32, dials="D4")
    with Store(tmp_path / "s.duckdb") as s:
        s.write(_row(good, status="heuristic"))
        s.write(_row(bad, store_status=ERROR, error="boom"))
        assert s.gate_passed(good.identity())
        assert not s.gate_passed(bad.identity())
        assert not s.gate_passed("nothing")


def test_a_newer_schema_is_still_refused(tmp_path):
    from quantic.bench.store import SchemaMismatch

    path = tmp_path / "v9.duckdb"
    with Store(path) as s:
        s.con.execute("UPDATE meta SET value = '9' WHERE key = 'schema_version'")
    with pytest.raises(SchemaMismatch):
        Store(path)
