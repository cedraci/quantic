import math

import polars as pl
import pytest

from quantic.bench import metrics
from quantic.bench.keys import REF_QUBO_MINIMUM, REFERENCE, RUN, CodeVersion, Task
from quantic.bench.metrics import cells, scored, tts99
from quantic.bench.report import bracket_check, dial_effects, report
from quantic.bench.store import ERROR, KILLED, OK, Store, blank_row
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


def test_an_open_reference_gives_no_gap_but_success_is_read_against_upper(tmp_path):
    rows = [ref("none", 9.0, 10.0), run_row("none", "sa", 0, 10.0)]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        out = scored(s.runs())
    assert out["gap"].to_list() == [None]
    assert out["ok_0.0"].to_list() == [True]


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


def test_a_bracket_open_by_less_than_the_solver_tolerance_counts_as_closed(tmp_path):
    rows = [ref("none", 10.0, 10.0 + 1e-7), run_row("none", "sa", 0, 10.0)]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        out = scored(s.runs())
    assert out["ref_closed"].to_list() == [True]
    assert out["gap"].to_list() == [pytest.approx(0.0)]


def test_a_run_at_the_upper_bound_succeeds_even_when_the_reference_is_open(tmp_path):
    rows = [ref("none", 9.9, 10.0), run_row("none", "sa", 0, 10.0)]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        out = scored(s.runs())
    assert out["ok_0.0"].to_list() == [True]
    assert out["gap"].to_list() == [None]


def test_an_open_reference_reports_its_width_in_the_cell_and_the_report(tmp_path):
    rows = [ref("none", 9.0, 10.0), run_row("none", "sa", 0, 10.0)]
    store = write(tmp_path / "s.duckdb", rows)
    with Store(store, read_only=True) as s:
        got = cells(s.runs()).row(0, named=True)
    assert got["ref_width_max"] == pytest.approx(1.0)
    assert got["n_open_refs"] == 1
    report(store, tmp_path / "out")
    text = (tmp_path / "out" / "crossover_1s.md").read_text(encoding="utf-8")
    assert "open, width 1 /" in text


def test_cells_carry_success_and_tts_at_every_epsilon(tmp_path):
    totals = [10.0, 10.0005, 10.05, 10.4, 12.0]
    rows = [ref("none", 10.0, 10.0)] + [
        run_row("none", "sa", s, t) for s, t in enumerate(totals)
    ]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        got = cells(s.runs()).row(0, named=True)
    assert [got[f"p_success_{e}"] for e in (0.0, 0.001, 0.01, 0.05)] == [0.2, 0.4, 0.6, 0.8]
    assert got["tts99_s_0.0"] > got["tts99_s_0.05"]
    for e in (0.0, 0.001, 0.01, 0.05):
        for name in ("tts99_lo_s", "tts99_hi_s", "censored_above_s"):
            assert f"{name}_{e}" in got
    assert got["p_success"] == got["p_success_0.01"]


def test_best_encoding_prefers_reliable_success_to_a_lucky_zero_gap():
    from quantic.bench.report import best_encoding

    table = pl.DataFrame({
        "tier": ["T0"] * 3, "dials": ["none"] * 3, "solver": ["sa"] * 3,
        "budget_s": [1.0] * 3, "encoding": ["lucky", "steady", "never"],
        "gap_median": [0.0, 0.005, None],
        "tts99_s": [500.0, 0.5, None],
        "censored_above_s": [None, None, 40.0],
    })
    assert best_encoding(table)["encoding"].to_list() == ["steady"]
    assert best_encoding(table.filter(pl.col("encoding") != "steady"))[
        "encoding"].to_list() == ["lucky"]


def test_a_censored_cell_prints_its_bound(tmp_path):
    rows = [ref("none", 10.0, 10.0), run_row("none", "sa", 0, 11.0)]
    store = write(tmp_path / "s.duckdb", rows)
    report(store, tmp_path / "out")
    text = (tmp_path / "out" / "crossover_1s.md").read_text(encoding="utf-8")
    assert "censored, > 1 s" in text


def test_a_cells_interval_does_not_depend_on_the_other_cells(tmp_path):
    labels = ("none", "D1", "D2", "D3")
    rows = [ref(d, 20.0, 20.0) for d in labels]
    rows += [run_row(d, "sa", s, t) for d in labels
             for s, t in enumerate([20.0, 21.0, 20.0, 22.0, 23.0, 21.0, 20.0, 24.0])]
    with Store(write(tmp_path / "all.duckdb", rows), read_only=True) as s:
        everything = cells(s.runs())
    for d in labels:
        mine = [r for r in rows if r["dials"] == d]
        with Store(write(tmp_path / f"{d}.duckdb", mine), read_only=True) as s:
            alone = cells(s.runs()).row(0, named=True)
        both = everything.filter(pl.col("dials") == d).row(0, named=True)
        assert alone["tts99_lo_s"] == both["tts99_lo_s"]
        assert alone["tts99_hi_s"] == both["tts99_hi_s"]


def test_a_cell_without_its_reference_says_so_and_the_failed_reference_is_listed(tmp_path):
    broken = ref("none", 10.0, 10.0)
    broken.update(store_status=ERROR, status=None, lower=None, upper=None, total=None,
                  error="ArithmeticError: the reference failed")
    rows = [broken, ref("D1", 10.0, 10.0),
            run_row("none", "sa", 0, 10.0), run_row("D1", "sa", 0, 10.0)]
    store = write(tmp_path / "s.duckdb", rows)
    with Store(store, read_only=True) as s:
        table = cells(s.runs())
    missing = dict(zip(table["dials"], table["n_missing_refs"], strict=True))
    assert missing == {"none": 1, "D1": 0}

    report(store, tmp_path / "out")
    listed = pl.read_csv(tmp_path / "out" / "references.csv")
    assert listed["key"].to_list() == [broken["key"]]
    assert listed["store_status"].to_list() == [ERROR]
    assert listed["error"].to_list() == ["ArithmeticError: the reference failed"]
    text = (tmp_path / "out" / "references.md").read_text(encoding="utf-8")
    assert "ArithmeticError: the reference failed" in text


def test_dial_effects_say_which_solver_they_skipped_and_why():
    labels = [d.label for d in Dials.combinations(include_cvar=True)]
    whole = pl.DataFrame({
        "tier": ["T0"] * 16, "dials": labels, "solver": ["sa"] * 16,
        "budget_s": [1.0] * 16, "gap_median": [0.0] * 16,
    })
    short = whole.filter(pl.col("dials") != "D1+D2").with_columns(solver=pl.lit("tabu"))
    gapless = whole.with_columns(
        solver=pl.lit("greedy"),
        gap_median=pl.when(pl.col("dials") == "D3").then(None).otherwise(0.0),
    )
    effects = dial_effects(pl.concat([whole, short, gapless]))
    skipped = effects.filter(pl.col("effect") == "skipped")
    notes = dict(zip(skipped["solver"], skipped["note"], strict=True))
    assert set(notes) == {"tabu", "greedy"}
    assert "D1+D2" in notes["tabu"] and "no cell" in notes["tabu"]
    assert "D3" in notes["greedy"] and "no gap" in notes["greedy"]
    assert effects.filter(pl.col("solver") == "sa").height == 4 + 6


def test_the_bracket_check_says_how_much_it_compared(tmp_path):
    compared = write(tmp_path / "a.duckdb", [
        ref("none", 10.0, 10.0), run_row("none", "cpsat", 0, 10.0, lower=10.0, upper=10.0),
    ])
    report(compared, tmp_path / "a")
    text = (tmp_path / "a" / "bracket_check.md").read_text(encoding="utf-8")
    assert "1 CP-SAT run on 1 instance" in text and "_none_" in text

    nothing = write(tmp_path / "b.duckdb", [ref("none", 10.0, 10.0),
                                            run_row("none", "sa", 0, 10.0)])
    report(nothing, tmp_path / "b")
    text = (tmp_path / "b" / "bracket_check.md").read_text(encoding="utf-8")
    assert "Nothing was compared" in text and "_none_" not in text


def test_a_commit_the_store_does_not_hold_is_refused_by_name(tmp_path):
    store = write(tmp_path / "s.duckdb", [ref("none", 10.0, 10.0),
                                          run_row("none", "sa", 0, 10.0)])
    with pytest.raises(ValueError, match="deadbeef.*c0ffee"):
        report(store, tmp_path / "out", commit="deadbeef")


def test_killed_and_failed_runs_count_against_success_and_never_succeed(tmp_path):
    killed = run_row("none", "sa", 2, None, feasible=None)
    killed.update(store_status=KILLED, instance_hash=None, status=None, total_s=None,
                  total_bps=None, error="killed after 33.0 s")
    # A failed run that somehow carries a total at the optimum still never succeeds.
    failed = run_row("none", "sa", 3, 10.0, feasible=True)
    failed.update(store_status=ERROR, status=None, error="WorkerDied: exit code 3")
    rows = [ref("none", 10.0, 10.0), run_row("none", "sa", 0, 10.0),
            run_row("none", "sa", 1, 10.0), killed, failed]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        runs = s.runs()
    out = scored(runs)
    for status in (KILLED, ERROR):
        mine = out.filter(pl.col("store_status") == status)
        assert not any(mine[f"ok_{e}"].any() for e in (0.0, 0.001, 0.01, 0.05)), status
    got = cells(runs).row(0, named=True)
    assert (got["n_runs"], got["n_killed"], got["n_errors"]) == (4, 1, 1)
    assert got["p_success"] == 0.5
    assert got["n_missing_refs"] == 0


def test_a_cell_without_a_sample_cap_keeps_its_m4_bootstrap_seed(tmp_path):
    """max_samples joined CELL, but M4 intervals were seeded from five fields."""
    totals = [10.0 if i % 5 in (0, 2, 3) else 12.0 for i in range(47)]
    rows = [ref("none", 10.0, 10.0)] + [
        run_row("none", "sa", s, t) for s, t in enumerate(totals)
    ]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        runs = s.runs()
    got = cells(runs).row(0, named=True)
    scores = scored(runs)
    rng = metrics._cell_rng(0, ("T0", "none", "sa", None, 1.0))
    for eps in sorted(metrics.EPSILONS):           # one generator, drawn in eps order
        ok = scores[f"ok_{eps}"].to_numpy().astype(float)
        lo, hi = metrics._bootstrap_tts(ok, 0.5, rng)
        assert (lo, hi) == (got[f"tts99_lo_s_{eps}"], got[f"tts99_hi_s_{eps}"])
    assert got["tts99_lo_s"] is not None
