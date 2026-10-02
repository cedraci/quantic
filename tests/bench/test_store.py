import multiprocessing as mp
import time

import duckdb
import pytest

from quantic.bench.store import SCHEMA_VERSION, SchemaMismatch, Store, StoreBusy
from tests.bench.rows import run_row, write


def test_a_row_reads_back_as_written_with_its_samples_and_trace(tmp_path):
    row = run_row("none", "tabu", 0, 10.0, encoding="binary", n_samples=3,
                  violations={"cardinality": 0.0}, schedule=[[1, 2], [0, 3]])
    with Store(tmp_path / "s.duckdb") as s:
        s.write(row, [("ab12", 2, -1.5, "feasible", 10.0)], [(0.1, -1.0), (0.2, -1.5)])
        back = s.runs().row(0, named=True)
        assert s.samples()["count"].to_list() == [2]
        assert s.con.execute("SELECT count(*) FROM traces").fetchone() == (2,)
        assert s.done_keys() == {row["key"]}
    assert back["total"] == 10.0
    assert back["schedule"] == "[[1, 2], [0, 3]]"
    assert back["violations"] == '{"cardinality": 0.0}'


def test_a_row_with_a_missing_or_unknown_column_is_refused(tmp_path):
    row = run_row("none", "tabu", 0, 10.0)
    with Store(tmp_path / "s.duckdb") as s:
        with pytest.raises(ValueError, match="unknown"):
            s.write({**row, "colour": "blue"})
        assert s.runs().is_empty()


def test_a_failed_write_leaves_nothing_behind(tmp_path):
    first = run_row("none", "tabu", 0, 10.0)
    second = run_row("none", "tabu", 1, 11.0)
    with Store(tmp_path / "s.duckdb") as s:
        s.write(first, [("ab12", 1, 0.0, "feasible", 1.0)], [(0.1, -1.0)])
        before = (s.runs(), s.samples(), s.con.execute("SELECT * FROM traces").pl())
        # The runs and samples inserts succeed; the trace tuple is one value short.
        with pytest.raises(duckdb.Error):
            s.write(second, [("cd34", 1, 0.0, "feasible", 1.0)], [(0.2,)])
        after = (s.runs(), s.samples(), s.con.execute("SELECT * FROM traces").pl())
        assert s.done_keys() == {first["key"]}
    for was, now in zip(before, after, strict=True):
        assert was.equals(now)


def test_a_store_of_another_schema_version_is_refused(tmp_path):
    path = write(tmp_path / "s.duckdb", [])
    con = duckdb.connect(str(path))
    con.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'",
                [str(SCHEMA_VERSION + 1)])
    con.close()
    with pytest.raises(SchemaMismatch):
        Store(path)


def _hold_open(path, opened, release):
    with Store(path):
        opened.set()
        release.wait(30)


def test_a_store_another_process_is_writing_is_refused_with_a_reason(tmp_path):
    path = write(tmp_path / "s.duckdb", [])
    ctx = mp.get_context("spawn")
    opened, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_open, args=(path, opened, release))
    child.start()
    try:
        assert opened.wait(30)
        with pytest.raises(StoreBusy, match="bench run"):
            Store(path, read_only=True)
    finally:
        release.set()
        child.join()


def test_a_store_in_a_directory_that_does_not_exist_yet_is_created(tmp_path):
    path = tmp_path / "studies" / "m4" / "s.duckdb"
    with Store(path) as s:
        assert s.done_keys() == set()
    assert path.exists()


def test_a_task_with_fifty_thousand_samples_is_written_in_seconds(tmp_path):
    """A 60 s greedy run keeps ~120k distinct samples; the writer must keep up.

    Row-at-a-time inserts ran at ~2,500 rows/s (20 s here); a frame insert runs
    at ~300,000 rows/s. The bound leaves room for a loaded machine.
    """
    row = run_row("none", "greedy", 0, 10.0, encoding="binary")
    samples = [(f"{i:016x}", 1 + i % 3, -float(i), "feasible", 10.0 + i)
               for i in range(50_000)]
    samples[7] = (samples[7][0], 2, -7.0, "infeasible", None)
    trace = [(i / 1000, -float(i)) for i in range(5_000)]
    with Store(tmp_path / "s.duckdb") as s:
        started = time.perf_counter()
        s.write(row, samples, trace)
        seconds = time.perf_counter() - started
        back = s.con.execute(
            "SELECT digest, count, energy, classification, total FROM samples "
            "ORDER BY digest"
        ).fetchall()
        assert s.con.execute("SELECT seconds, best_energy FROM traces "
                             "ORDER BY seconds").fetchall() == trace
    assert back == sorted(samples)
    assert seconds < 5.0, seconds
