import duckdb
from typer.testing import CliRunner

from quantic.bench.keys import RUN, CodeVersion, Task
from quantic.bench.store import OK, Store, blank_row
from quantic.cli import app
from tests.bench.rows import ref, run_row, write


def test_the_cli_plans_without_running(tmp_path):
    config = tmp_path / "exp.toml"
    config.write_text(
        "[instances]\ntiers = ['T0']\ndials = ['none']\nseeds = [2026]\n"
        "[solvers.almgren_chriss]\nbudgets_s = [1]\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["bench", "plan", str(config), "--workers", "2"])
    assert result.exit_code == 0, result.output
    assert "3 tasks, 0 already in the store, 3 to run" in result.output


def test_the_cli_refuses_a_bad_config(tmp_path):
    config = tmp_path / "exp.toml"
    config.write_text("[instances]\ncolour = 'blue'\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["bench", "plan", str(config)])
    assert result.exit_code != 0
    assert "unknown key" in result.output


def test_the_cli_reports_a_store(tmp_path):
    store = write(tmp_path / "s.duckdb", [ref("none", 10.0, 10.0),
                                         run_row("none", "sa", 0, 10.0)])
    result = CliRunner().invoke(app, ["bench", "report", str(store), "--out",
                                      str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "crossover_1s.md").exists()


def _two_commits(tmp_path):
    store = write(tmp_path / "s.duckdb", [ref("none", 10.0, 10.0),
                                          run_row("none", "sa", 0, 10.0)])
    other = CodeVersion(commit="beef", lock_hash="10c4", dirty=False)
    task = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="sa",
                solver_seed=9, budget_s=1.0)
    with Store(store) as s:
        s.write({**blank_row(task.key(other), task, other, "here", "{}"),
                 "store_status": OK})
    return store


def test_the_cli_refuses_a_store_of_two_commits_and_reports_the_one_named(tmp_path):
    store = _two_commits(tmp_path)
    refused = CliRunner().invoke(app, ["bench", "report", str(store), "--out",
                                       str(tmp_path / "a")])
    assert refused.exit_code == 1
    assert "--commit" in refused.output
    assert isinstance(refused.exception, SystemExit)

    chosen = CliRunner().invoke(app, ["bench", "report", str(store), "--out",
                                      str(tmp_path / "b"), "--commit", "c0ffee"])
    assert chosen.exit_code == 0, chosen.output
    assert (tmp_path / "b" / "crossover_1s.md").exists()

    unknown = CliRunner().invoke(app, ["bench", "report", str(store), "--out",
                                       str(tmp_path / "c"), "--commit", "deadbeef"])
    assert unknown.exit_code == 1
    assert "'beef'" in unknown.output and "'c0ffee'" in unknown.output


def test_the_cli_says_when_there_is_no_store(tmp_path):
    result = CliRunner().invoke(app, ["bench", "report", str(tmp_path / "none.duckdb"),
                                      "--out", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "no store" in result.output


def test_the_cli_run_refuses_a_store_of_another_schema(tmp_path):
    store = write(tmp_path / "s.duckdb", [])
    con = duckdb.connect(str(store))
    con.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    con.close()
    config = tmp_path / "exp.toml"
    config.write_text("[instances]\ndials = ['none']\n[solvers.almgren_chriss]\n"
                      "budgets_s = [1]\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["bench", "run", str(config), "--store", str(store),
                                      "--allow-dirty", "--workers", "1"])
    assert result.exit_code == 1
    assert "schema version 999" in result.output
