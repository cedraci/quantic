import polars as pl

from quantic.bench.keys import GATE, REF_QAOA_ANGLES, REFERENCE, RUN, Task
from quantic.bench.report import report
from quantic.bench.store import PENDING
from quantic.solvers.quantum.quota import Ledger
from tests.bench.rows import ref, row, write


def _q(solver, seed, total, *, kind=RUN, commit=None, store_status=None, **extra):
    task = Task(kind=kind, tier="T0", dials="D2", instance_seed=1, solver=solver,
                encoding="binary", zeta_bits=8, solver_seed=seed, max_samples=20,
                params="{}")
    r = row(task, status="heuristic", total=total, feasible=total is not None,
            total_s=extra.pop("total_s", 0.5), n_samples=20, n_feasible_samples=10,
            total_bps=total, shots=20, **extra)
    if commit:
        r["commit"] = commit
    if store_status:                 # row() sets store_status itself; override after
        r["store_status"] = store_status
    return r


def _store(tmp_path, extra=()):
    train = Task(kind=REFERENCE, tier="T0", dials="D2", instance_seed=1,
                 solver=REF_QAOA_ANGLES, encoding="binary", zeta_bits=8, params="{}")
    rows = [
        ref("D2", 10.0, 10.0),
        row(train, status="trained", n_vars=16, dynamic_range=5e5, total_s=12.0,
            detail={"cvar": 11.0, "n_evals": 40, "ising_dynamic_range": 9e4}),
        _q("qaoa_ideal", 0, 10.0), _q("qaoa_ideal", 1, 12.0),
        _q("qaoa_noisy", 0, 13.0, routed_depth=900, routed_2q=800),
        _q("qaoa_noisy", 0, 10.0, kind=GATE),
        _q("qaoa_ibm", 0, 10.0, commit="older", qpu_s=2.0, total_s=2.0,
           routed_depth=950, routed_2q=850),
        *extra,
    ]
    return write(tmp_path / "s.duckdb", rows)


def _read(out, stem):
    return pl.read_csv(out / f"{stem}.csv")


def test_hardware_rows_from_another_commit_stay_in_the_report(tmp_path):
    out = tmp_path / "out"
    report(_store(tmp_path), out)                    # no CommitChoice although two commits
    matched = _read(out, "matched_reads")
    assert set(matched["solver"].to_list()) == {"qaoa_ideal", "qaoa_noisy", "qaoa_ibm"}
    tts = dict(zip(matched["solver"], matched["tts"], strict=True))
    assert tts["qaoa_ideal"] == "simulated" and tts["qaoa_noisy"] == "simulated"
    assert tts["qaoa_ibm"].endswith("s")


def test_gates_and_pending_rows_are_not_measurements(tmp_path):
    pending = _q("qaoa_ibm", 5, None, store_status=PENDING, job_id="j")
    out = tmp_path / "out"
    report(_store(tmp_path, [pending]), out)
    cells = _read(out, "cells")
    noisy = cells.filter(pl.col("solver") == "qaoa_noisy")
    assert noisy["n_runs"].to_list() == [1]          # the gate row is not counted
    ibm = cells.filter(pl.col("solver") == "qaoa_ibm")
    assert ibm["n_runs"].to_list() == [1]            # the pending row is not counted


def test_noise_and_training_tables(tmp_path):
    out = tmp_path / "out"
    report(_store(tmp_path), out)
    noise = _read(out, "qaoa_noise")
    assert noise.filter(pl.col("solver") == "qaoa_ibm")["qpu_s"].to_list() == [2.0]
    assert noise.filter(pl.col("solver") == "qaoa_noisy")["routed_2q"].to_list() == [800]
    training = _read(out, "qaoa_training")
    assert training["n_vars"].to_list() == [16] and training["n_evals"].to_list() == [40]
    assert not (out / "crossover_nones.csv").exists()


def test_simulators_show_no_qpu_seconds_and_no_tts(tmp_path):
    out = tmp_path / "out"
    report(_store(tmp_path), out)
    noise = _read(out, "qaoa_noise")
    for solver in ("qaoa_ideal", "qaoa_noisy"):
        assert noise.filter(pl.col("solver") == solver)["qpu_s"].to_list() == [None]
    cells = _read(out, "cells")
    tts = [c for c in cells.columns if c.startswith("tts99")]
    assert "tts99_s" in tts and "tts99_lo_s_0.01" in tts
    simulated = cells.filter(pl.col("solver").is_in(["qaoa_ideal", "qaoa_noisy"]))
    assert simulated.height == 2
    assert all(simulated[c].null_count() == simulated.height for c in tts)
    assert cells.filter(pl.col("solver") == "qaoa_ibm")["tts99_s"].to_list() == [2.0]


def test_the_quota_table_reads_the_ledger(tmp_path):
    ledger = Ledger(tmp_path / "q.json")
    ledger.add_pending("j1", "ibm_fez", "k", 30, now=1_800_000_000.0)
    ledger.mark_done("j1", 4.0)
    out = tmp_path / "out"
    report(_store(tmp_path), out, ledger_path=tmp_path / "q.json")
    quota = _read(out, "quota")
    assert quota["job_id"].to_list() == ["j1"] and quota["usage_s"].to_list() == [4.0]
