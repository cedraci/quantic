import time
from pathlib import Path

import polars as pl
import pytest

from quantic.bench import runner as runner_module
from quantic.bench.config import load
from quantic.bench.keys import CodeVersion
from quantic.bench.runner import DirtyTree, plan, run
from quantic.bench.store import ERROR, KILLED, OK, Store

ROOT = Path(__file__).resolve().parents[2]
TIMING = ["total_s", "timings"]

SMALL = """
[instances]
tiers = ["T0"]
dials = ["none", "D2+D3"]
seeds = [2026]

[references]
qubo_minimum_tiers = ["T0"]

[solvers.almgren_chriss]
budgets_s = [1]

[solvers.cpsat]
budgets_s = [30]

[solvers.simulated_annealing]
encodings = ["binary", "one_hot"]
seeds = "0..1"
budgets_s = [600]
max_samples = 15
"""


def _experiment(tmp_path: Path, text: str, name: str = "exp"):
    path = tmp_path / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return load(path)


def _frame(store: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    with Store(store, read_only=True) as s:
        return s.runs().drop(TIMING), s.samples()


def test_a_stopped_run_resumes_to_the_same_store_as_an_unbroken_one(tmp_path):
    exp = _experiment(tmp_path, SMALL)
    whole, broken = tmp_path / "whole.duckdb", tmp_path / "broken.duckdb"
    run(exp, root=ROOT, store=whole, workers=3, allow_dirty=True)
    first = run(exp, root=ROOT, store=broken, workers=3, allow_dirty=True, stop_after=5)
    assert first.n_written == 5
    assert plan(exp, root=ROOT, store=broken).n_todo == plan(exp, root=ROOT).n_tasks - 5
    run(exp, root=ROOT, store=broken, workers=3, allow_dirty=True)

    runs_whole, samples_whole = _frame(whole)
    runs_broken, samples_broken = _frame(broken)
    assert runs_whole.height == plan(exp, root=ROOT).n_tasks
    assert set(runs_whole["store_status"]) == {OK}
    assert runs_broken.equals(runs_whole)
    assert samples_broken.equals(samples_whole)


def test_running_again_does_nothing_once_the_store_is_complete(tmp_path):
    exp = _experiment(tmp_path, SMALL)
    store = tmp_path / "s.duckdb"
    run(exp, root=ROOT, store=store, workers=3, allow_dirty=True)
    again = run(exp, root=ROOT, store=store, workers=3, allow_dirty=True)
    assert again.n_written == 0


def test_a_hanging_solver_is_killed_and_the_run_carries_on(tmp_path):
    exp = _experiment(tmp_path, """
[instances]
tiers = ["T0"]
dials = ["none"]
seeds = [2026]

[solvers."tests.bench.stand_ins:Hang"]
budgets_s = [0.1]

[solvers.almgren_chriss]
budgets_s = [1]
""")
    store = tmp_path / "s.duckdb"
    summary = run(
        exp, root=ROOT, store=store, workers=2, allow_dirty=True, kill_margin_s=2.0
    )
    rows, _ = _frame(store)
    hung = rows.filter(pl.col("solver") == "tests.bench.stand_ins:Hang")
    assert summary.n_killed == 1
    assert hung["store_status"].to_list() == [KILLED]
    assert hung["error"][0].startswith("killed after")
    assert rows.filter(pl.col("solver") == "almgren_chriss")["store_status"].to_list() == [OK]


def test_a_solver_that_raises_is_a_row_that_says_so(tmp_path):
    exp = _experiment(tmp_path, """
[instances]
tiers = ["T0"]
dials = ["none"]
seeds = [2026]

[solvers."tests.bench.stand_ins:Boom"]
budgets_s = [1]
""")
    store = tmp_path / "s.duckdb"
    summary = run(exp, root=ROOT, store=store, workers=1, allow_dirty=True)
    rows, _ = _frame(store)
    boom = rows.filter(pl.col("solver") == "tests.bench.stand_ins:Boom")
    assert summary.n_errors == 1
    assert boom["store_status"].to_list() == [ERROR]
    assert boom["error"][0] == "ArithmeticError: this solver always fails"


def test_a_dirty_tree_is_refused_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner_module, "code_version",
        lambda root: CodeVersion(commit="abc", lock_hash="def", dirty=True),
    )
    exp = _experiment(tmp_path, SMALL)
    with pytest.raises(DirtyTree, match="allow-dirty"):
        run(exp, root=ROOT, store=tmp_path / "s.duckdb")


def test_the_plan_counts_tasks_and_estimates_from_their_budgets(tmp_path):
    exp = _experiment(tmp_path, SMALL)
    p = plan(exp, root=ROOT, workers=2)
    # 2 instances x (bracket + brute force + 2 QUBO minima) references,
    # and 2 x (1 + 1 + 2 encodings x 2 seeds) runs.
    assert p.n_tasks == 2 * 4 + 2 * 6
    assert p.estimate_s == pytest.approx((8 * 1.0 + 2 * (1 + 30 + 4 * 600)) / 2)


def test_an_instance_that_cannot_be_built_is_an_error_row_not_a_crash():
    from quantic.bench.execute import Executor
    from quantic.bench.keys import RUN, Task

    code = CodeVersion(commit="abc", lock_hash="def", dirty=False)
    task = Task(kind=RUN, tier="T0", dials="D9", instance_seed=2026,
                solver="almgren_chriss", budget_s=1.0)
    row, samples, trace = Executor(code).run(task.key(code), task)
    assert row["store_status"] == ERROR
    assert row["error"].startswith("KeyError")
    assert samples == [] and trace == []


def test_a_worker_that_dies_is_an_error_row_not_a_kill(tmp_path):
    exp = _experiment(tmp_path, """
[instances]
tiers = ["T0"]
dials = ["none"]
seeds = [2026]

[solvers."tests.bench.stand_ins:Die"]
budgets_s = [60]

[solvers.almgren_chriss]
budgets_s = [1]
""")
    store = tmp_path / "s.duckdb"
    started = time.monotonic()
    summary = run(exp, root=ROOT, store=store, workers=2, allow_dirty=True)
    assert time.monotonic() - started < 60  # not left to its deadline
    rows, _ = _frame(store)
    died = rows.filter(pl.col("solver") == "tests.bench.stand_ins:Die")
    assert (summary.n_killed, summary.n_errors) == (0, 1)
    assert died["store_status"].to_list() == [ERROR]
    assert died["error"].to_list() == ["WorkerDied: exit code 3"]
    assert rows.filter(pl.col("solver") == "almgren_chriss")["store_status"].to_list() == [OK]
