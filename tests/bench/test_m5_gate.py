"""The M5 gate, end to end, with no network -- M5 design section 13.

A small study: two T0 cells of 16 binary qubits at seed 2026 (D2 and D1+D2), a
few seeds, IBM in local mode against the fake Fez. Every phase runs: training,
noiseless and noisy runs, gates, the hardware lane, and the report.
"""

from pathlib import Path

import polars as pl

from quantic.bench.config import load
from quantic.bench.keys import GATE as GATE_KIND
from quantic.bench.keys import REF_QAOA_ANGLES, REFERENCE, RUN
from quantic.bench.report import report
from quantic.bench.runner import plan, run
from quantic.bench.store import OK, Store
from quantic.solvers.quantum.ibm import LocalClient
from quantic.solvers.quantum.quota import Ledger

ROOT = Path(__file__).resolve().parents[2]

GATE = """
[instances]
tiers = ["T0"]
dials = ["D2", "D1+D2"]
seeds = [2026]

[references]
qubo_minimum_tiers = ["T0"]

[quantum]
reps = 1
maxiter = 20
shots = 64
max_qubits = 18
noisy_max_qubits = 18
gate_shots = 8
device = "ibm_fez"
workers = 2

[solvers.qaoa_ideal]
encodings = ["binary"]
seeds = "0..2"

[solvers.qaoa_noisy]
encodings = ["binary"]
seeds = "0..1"

[solvers.qaoa_ibm]
encodings = ["binary"]
seeds = "0..0"

[solvers.simulated_annealing]
encodings = ["binary"]
seeds = "0..2"
max_samples = 64
budgets_s = [600]
"""


def test_the_m5_pipeline_runs_end_to_end_within_the_quota(tmp_path):
    path = tmp_path / "gate.toml"
    path.write_text(GATE, encoding="utf-8")
    exp = load(path)
    store = tmp_path / "gate.duckdb"
    ledger = Ledger(tmp_path / "q.json")
    client = LocalClient(usage_per_job_s=3.0)
    summary = run(exp, root=ROOT, store=store, workers=2, allow_dirty=True,
                  client=client, ledger=ledger)
    assert summary.n_errors == 0 and summary.hardware_stop is None
    with Store(store, read_only=True) as s:
        runs = s.runs()
    hw = runs.filter(pl.col("solver") == "qaoa_ibm")
    assert hw.height == 2 and (hw["store_status"] == OK).all()
    assert (runs["store_status"] == OK).all()

    def count(**where):
        return runs.filter(*(pl.col(k) == v for k, v in where.items())).height

    assert count(kind=RUN, solver="qaoa_ideal") == 6
    assert count(kind=RUN, solver="qaoa_noisy") == 4
    assert count(kind=GATE_KIND) == 2
    assert count(kind=REFERENCE, solver=REF_QAOA_ANGLES) == 2
    first = ledger.entries()
    assert [e.state for e in first] == ["done", "done"]
    assert sum(e.usage_s for e in ledger.entries()) == 6.0
    assert plan(exp, root=ROOT, store=store).n_todo == 0
    client2 = LocalClient()
    again = run(exp, root=ROOT, store=store, workers=2, allow_dirty=True,
                client=client2, ledger=ledger)
    assert again.n_written == 0                       # nothing re-run, nothing re-paid
    assert client2.submitted == []
    assert ledger.entries() == first
    assert [e.state for e in ledger.entries()] == ["done", "done"]
    assert sum(e.usage_s for e in ledger.entries()) == 6.0
    out = tmp_path / "report"
    written = {p.name for p in report(store, out, ledger_path=tmp_path / "q.json")}
    assert {"matched_reads.csv", "qaoa_noise.csv", "qaoa_training.csv",
            "quota.csv"} <= written
    assert pl.read_csv(out / "qaoa_training.csv").height == 2
    quota = pl.read_csv(out / "quota.csv")
    assert quota.height == 2
    assert sorted(quota["job_id"].to_list()) == sorted(e.job_id for e in first)
    assert quota["usage_s"].sum() == 6.0
    noise = pl.read_csv(out / "qaoa_noise.csv")
    assert set(noise["solver"].to_list()) == {"qaoa_ideal", "qaoa_noisy", "qaoa_ibm"}
