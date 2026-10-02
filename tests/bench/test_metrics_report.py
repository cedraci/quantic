import math

import polars as pl
import pytest

from quantic.bench.keys import REF_QUBO_MINIMUM, REFERENCE, RUN, CodeVersion, Task
from quantic.bench.metrics import cells, scored, tts99
from quantic.bench.report import bracket_check, dial_effects, report
from quantic.bench.store import OK, Store, blank_row
from quantic.problem.dials import Dials
from tests.bench.rows import ref, row, run_row, write


@pytest.mark.parametrize(
    "p, expected",
    [(1.0, 2.0), (0.0, None), (0.5, 2.0 * math.log(0.01) / math.log(0.5))],
)
def test_time_to_solution(p, expected):
    got = tts99(2.0, p)
    assert got == expected if expected is None else got == pytest.approx(expected)


def test_tts_on_a_known_success_rate_from_runs(tmp_path):
    """Two of four seeds reach the optimum: p = 0.5."""
    rows = [ref("none", 10.0, 10.0)] + [
        run_row("none", "sa", s, total) for s, total in enumerate([10.0, 10.0, 11.0, 12.0])
    ]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        table = cells(s.runs(), eps=0.0)
    got = table.row(0, named=True)
    assert got["p_success"] == 0.5
    assert got["tts99_s"] == pytest.approx(0.5 * math.log(0.01) / math.log(0.5))
    assert got["gap_median"] == pytest.approx(0.05)
    assert got["censored_above_s"] is None


def test_an_infeasible_run_never_succeeds_and_a_cell_with_no_success_is_censored(tmp_path):
    rows = [ref("none", 10.0, 10.0),
            run_row("none", "sa", 0, 9.0, feasible=False),
            run_row("none", "sa", 1, None, feasible=None)]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        runs = s.runs()
    assert not scored(runs)["ok_0.05"].any()
    got = cells(runs).row(0, named=True)
    assert got["tts99_s"] is None
    assert got["censored_above_s"] == 2.0
    assert got["gap_median"] is None


def test_an_open_reference_gives_no_gap_and_no_success(tmp_path):
    rows = [ref("none", 9.0, 10.0), run_row("none", "sa", 0, 10.0)]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        out = scored(s.runs())
    assert out["gap"].to_list() == [None]
    assert not out["ok_0.0"].any()


def test_regret_is_read_against_the_qubos_minimum(tmp_path):
    qmin = Task(kind=REFERENCE, tier="T0", dials="none", instance_seed=1,
                solver=REF_QUBO_MINIMUM, encoding="binary", zeta_bits=8)
    run = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="sa",
               encoding="binary", zeta_bits=8, solver_seed=0, budget_s=1.0)
    rows = [
        ref("none", 10.0, 10.0),
        row(qmin, qubo_energy=20.0, qubo_total=10.5),
        row(run, total=11.0, feasible=True, best_energy=21.0, total_s=1.0),
    ]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        out = scored(s.runs()).row(0, named=True)
    assert out["energy_gap"] == pytest.approx(0.05)
    assert out["encoding_regret"] == pytest.approx(0.05)


def test_dial_effects_are_the_factorial_contrasts():
    """A gap that is D1's indicator gives D1 an effect of one and nothing else."""
    labels = [d.label for d in Dials.combinations(include_cvar=True)]
    best = pl.DataFrame({
        "tier": ["T0"] * 16, "dials": labels, "solver": ["sa"] * 16,
        "budget_s": [1.0] * 16,
        "gap_median": [1.0 if "D1" in label.split("+") else 0.0 for label in labels],
    })
    effects = {r["effect"]: r["value"] for r in dial_effects(best).iter_rows(named=True)}
    assert effects["D1"] == pytest.approx(1.0)
    assert all(abs(v) < 1e-12 for k, v in effects.items() if k != "D1")


def test_brackets_that_do_not_meet_are_listed(tmp_path):
    rows = [
        ref("none", 10.0, 10.0), ref("D1", 20.0, 20.0),
        run_row("none", "cpsat", 0, 10.0, lower=9.99, upper=10.0),
        run_row("D1", "cpsat", 0, 20.5, lower=20.4, upper=20.5),
    ]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        apart = bracket_check(s.runs())
    assert apart["dials"].to_list() == ["D1"]


def test_the_report_writes_every_table_and_refuses_mixed_commits(tmp_path):
    rows = [ref("none", 10.0, 10.0), run_row("none", "sa", 0, 10.0),
            run_row("none", "cpsat", 0, 10.0, lower=10.0, upper=10.0)]
    store = write(tmp_path / "s.duckdb", rows)
    written = report(store, tmp_path / "out")
    names = {p.name for p in written}
    assert {"cells.csv", "crossover_1s.md", "dial_effects.csv", "bracket_check.md"} <= names
    assert "_none_" in (tmp_path / "out" / "bracket_check.md").read_text(encoding="utf-8")

    other = CodeVersion(commit="beef", lock_hash="10c4", dirty=False)
    task = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="sa",
                solver_seed=9, budget_s=1.0)
    with Store(store) as s:
        s.write({**blank_row(task.key(other), task, other, "here", "{}"),
                 "store_status": OK})
    with pytest.raises(ValueError, match="--commit"):
        report(store, tmp_path / "out2")
    assert report(store, tmp_path / "out3", commit="c0ffee")


def test_a_report_on_a_store_with_no_references_says_so_cell_by_cell(tmp_path):
    rows = [run_row("none", "sa", 0, 10.0), run_row("none", "sa", 1, 11.0)]
    store = write(tmp_path / "s.duckdb", rows)
    report(store, tmp_path / "out")
    text = (tmp_path / "out" / "crossover_1s.md").read_text(encoding="utf-8")
    assert "- / 100% / censored" in text
