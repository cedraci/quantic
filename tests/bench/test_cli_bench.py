from typer.testing import CliRunner

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
