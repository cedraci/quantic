"""The M4 gate -- M4 design section 9.

The harness is correct and complete on the M3 ladder's 32 T0-T1 instances.
Every check is about correctness or integrity; none says a solver is good.
A slow sampler is a finding for the result doc, not a failure here.

The experiment runs once per session into a temporary store
(``experiments/m4-gate.toml``, 1,248 tasks, about 13.5 minutes on 14 workers
in the dry run, spec section 9); every check reads that store. Checks 6 and 7
run small experiments of their own.

Never weaken a check to make it pass. If one fails, stop and report which
check, which instance, which solver and which encoding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantic.bench.config import load
from quantic.bench.execute import Executor
from quantic.bench.instances import InstanceCache
from quantic.bench.keys import (
    REF_BRACKET,
    REF_BRUTE_FORCE,
    REF_QUBO_MINIMUM,
    RUN,
    code_version,
    expand,
)
from quantic.bench.metrics import cells
from quantic.bench.report import CENSORED, best_encoding, bracket_check, crossover
from quantic.bench.runner import plan, run
from quantic.bench.store import KILLED, OK, Store
from quantic.encoding.encode import encode
from quantic.problem.schedule import Schedule
from quantic.solvers.base import _BRACKET_RTOL, OPTIMAL, Budget
from quantic.solvers.classical.qubo_minimum import qubo_minimum
from quantic.solvers.classical.scoring import score
from quantic.solvers.qubo.sampler import encoding_named, energies, run_sampler

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "experiments" / "m4-gate.toml"
# Two evaluator calls at one schedule: round-off only.
SAME = 1e-9
# The same energy accumulated in two orders at dynamic ranges up to 1e10: the
# tolerance M3b set for the closed form against the dense z'Qz.
DENSE = 1e-6
SOLVERS = ("almgren_chriss", "exact_pwl_milp", "cpsat", "simulated_annealing", "tabu",
           "greedy")
SAMPLERS = ("simulated_annealing", "tabu", "greedy")
# The crossover tables' solver columns, one table per budget in the gate file.
COLUMNS = {
    1.0: {"almgren_chriss"},
    60.0: {"exact_pwl_milp", "cpsat"},
    600.0: set(SAMPLERS),
}
# A crossover cell ends in a TTS or the censored marker, then the encoding if any.
TTS_OR_CENSORED = re.compile(
    rf" / (?:[-+.0-9e]+s|{CENSORED}, > [-+.0-9e]+ s)(?: \([a-z_]+\))?$"
)


@pytest.fixture(scope="module")
def gate_store(tmp_path_factory) -> Path:
    store = tmp_path_factory.mktemp("gate") / "gate.duckdb"
    summary = run(load(GATE), root=ROOT, store=store, allow_dirty=True)
    assert summary.n_errors == 0 and summary.n_killed == 0, summary
    return store


@pytest.fixture(scope="module")
def gate(gate_store) -> pl.DataFrame:
    with Store(gate_store, read_only=True) as s:
        return s.runs()


@pytest.fixture(scope="module")
def gate_samples(gate_store) -> pl.DataFrame:
    with Store(gate_store, read_only=True) as s:
        return s.samples()


def _schedule(row) -> Schedule:
    return Schedule(lots=tuple(tuple(r) for r in json.loads(row["schedule"])))


def test_the_store_holds_every_task_and_nothing_failed(gate):
    assert gate.height == plan(load(GATE), root=ROOT).n_tasks
    assert set(gate["store_status"]) == {OK}
    assert set(gate.filter(pl.col("kind") == RUN)["solver"]) == set(SOLVERS)
    assert gate["instance_hash"].n_unique() == 32


def test_check_1_every_objective_is_the_evaluators_and_every_optimum_is_proven(gate):
    cache = InstanceCache()
    wrong = []
    for row in gate.filter(pl.col("schedule").is_not_null()).iter_rows(named=True):
        inst = cache.get(row["tier"], row["dials"], row["instance_seed"])
        total = score(inst, _schedule(row))[0].total
        if abs(total - row["total"]) > SAME * max(1.0, abs(total)):
            wrong.append((row["tier"], row["dials"], row["solver"], row["encoding"]))
        if row["status"] == OPTIMAL:
            width = row["upper"] - row["lower"]
            assert row["feasible"], row["key"]
            assert width <= _BRACKET_RTOL * max(1.0, abs(row["upper"])), row["key"]
    assert not wrong


def test_check_2_cpsat_meets_the_reference_bracket_and_t0_meets_brute_force(gate):
    assert bracket_check(gate).is_empty()
    cp = gate.filter((pl.col("solver") == "cpsat") & pl.col("lower").is_not_null())
    assert cp["instance_hash"].n_unique() == 32

    # The reference, and the same MILP run as a budgeted solver, both answer
    # to brute force at T0, as CP-SAT does.
    compared = dict.fromkeys((REF_BRACKET, "cpsat", "exact_pwl_milp"), 0)
    by = {
        (r["instance_hash"], r["solver"]): r
        for r in gate.filter(pl.col("tier") == "T0").iter_rows(named=True)
        if r["solver"] in (REF_BRUTE_FORCE, *compared)
    }
    brute = {i: row for (i, solver), row in by.items() if solver == REF_BRUTE_FORCE}
    assert len(brute) == 16
    for instance, row in brute.items():
        optimum = row["total"]
        assert by[(instance, REF_BRACKET)]["status"] == OPTIMAL, instance
        for other in compared:
            mine = by[(instance, other)]
            if mine["status"] == OPTIMAL:
                assert mine["total"] == pytest.approx(optimum, rel=SAME), (instance, other)
                compared[other] += 1
    assert all(n > 0 for n in compared.values()), compared


def test_check_3_every_sampler_reports_the_qubos_own_energy(gate):
    """A sampler that read the bits in another order would report other energies."""
    sampled = gate.filter(pl.col("n_samples").is_not_null())
    assert sampled.height == 32 * 3 * (5 + 3 + 3)
    assert sampled["energy_mismatch"].max() <= DENSE
    assert (sampled["n_samples"] == sampled["max_samples"]).all()


def test_check_4_the_sampler_pipeline_returns_the_qubos_minimum_intact(gate):
    """Fed the enumerated minimum, the pipeline reports its energy and its total.

    Whether a sampler *finds* the minimum is a finding for the result doc, not a
    gate condition: on the CVaR instances the QUBO has 260-296 variables and a
    dynamic range near 1e7, and the dry run's annealer ended 12-45 times above
    it. What the gate can demand is that a sampler which does return the
    minimum is credited with it -- the right energy, the right decoded total.
    """
    cache = InstanceCache()
    refs = gate.filter(pl.col("solver") == REF_QUBO_MINIMUM)
    assert refs.height == 16 * 3
    for row in refs.iter_rows(named=True):
        inst = cache.get(row["tier"], row["dials"], row["instance_seed"])
        enc = encode(inst, encoding_named(row["encoding"]), n_zeta_bits=row["zeta_bits"])
        lowest = qubo_minimum(enc)

        def draw(bqm, reads, seed, bits=lowest.bits, enc=enc):
            stack = np.tile(bits, (reads, 1))
            return stack, energies(enc.qubo, stack)

        result = run_sampler("stand_in", enc, Budget(max_samples=1), 0, draw)
        where = (row["dials"], row["encoding"])
        assert result.detail["best_energy"] == pytest.approx(
            row["qubo_energy"], rel=DENSE
        ), where
        if row["qubo_total"] is None:
            assert result.schedule is None, where
        else:
            assert result.objective.total == pytest.approx(row["qubo_total"], rel=SAME), where


def test_check_5_rerunning_a_task_reproduces_its_row(gate, gate_samples):
    code = code_version(ROOT)
    executor = Executor(code)
    picked = (
        gate.filter(pl.col("kind") == RUN)
        .group_by(["tier", "solver"]).agg(pl.col("key").min())
    )
    tasks = {t.key(code): t for t in expand(load(GATE))}
    ignore = {"total_s", "timings"}
    for key in picked["key"]:
        again, samples, _ = executor.run(key, tasks[key])
        row = gate.filter(pl.col("key") == key).row(0, named=True)
        # The samples it keeps are its row too. (A trace is a timing.)
        stored = gate_samples.filter(pl.col("key") == key).drop("key").rows()
        assert stored == sorted(samples), key
        assert bool(samples) == (row["solver"] in SAMPLERS), key
        for column, value in again.items():
            if column in ignore:
                continue
            stored = row[column]
            if isinstance(value, (dict, list)):
                value = json.loads(json.dumps(value))
                stored = json.loads(stored)
            assert stored == value, (key, column)


SMALL = """
[instances]
tiers = ["T0"]
dials = ["none", "D2+D3"]
seeds = [2026]

[solvers.cpsat]
budgets_s = [30]

[solvers.simulated_annealing]
encodings = ["binary", "one_hot"]
seeds = "0..1"
budgets_s = [600]
max_samples = 15
"""


def test_check_6_a_stopped_run_resumes_to_the_same_store(tmp_path):
    path = tmp_path / "small.toml"
    path.write_text(SMALL, encoding="utf-8")
    exp = load(path)
    whole, broken = tmp_path / "whole.duckdb", tmp_path / "broken.duckdb"
    run(exp, root=ROOT, store=whole, workers=3, allow_dirty=True)
    run(exp, root=ROOT, store=broken, workers=3, allow_dirty=True, stop_after=4)
    run(exp, root=ROOT, store=broken, workers=3, allow_dirty=True)
    with Store(whole, read_only=True) as a, Store(broken, read_only=True) as b:
        assert b.runs().drop("total_s", "timings").equals(a.runs().drop("total_s", "timings"))
        assert b.samples().equals(a.samples())


def test_check_7_a_hanging_solver_is_killed_and_the_run_carries_on(tmp_path):
    path = tmp_path / "hang.toml"
    path.write_text(
        "[instances]\ntiers = ['T0']\ndials = ['none']\nseeds = [2026]\n"
        "[solvers.'tests.bench.stand_ins:Hang']\nbudgets_s = [0.1]\n"
        "[solvers.almgren_chriss]\nbudgets_s = [1]\n",
        encoding="utf-8",
    )
    store = tmp_path / "s.duckdb"
    run(load(path), root=ROOT, store=store, workers=2, allow_dirty=True, kill_margin_s=2.0)
    with Store(store, read_only=True) as s:
        status = dict(s.runs().select("solver", "store_status").iter_rows())
    assert status["tests.bench.stand_ins:Hang"] == KILLED
    assert status["almgren_chriss"] == OK


def test_check_8_the_report_fills_every_cell(gate):
    declared = {}
    for p in load(GATE).solvers:
        for budget in p.budgets_s:
            declared.setdefault(budget, set()).add(p.name)
    assert declared == COLUMNS
    best = best_encoding(cells(gate))
    assert set(best["budget_s"].unique().to_list()) == set(COLUMNS)
    for budget, expected in COLUMNS.items():
        table = crossover(best, budget)
        solvers = [c for c in table.columns if c not in ("tier", "dials")]
        assert set(solvers) == expected, budget
        assert table.height == 32, budget
        for column in solvers:
            assert table[column].null_count() == 0, (budget, column)
            for text in table[column]:
                assert TTS_OR_CENSORED.search(text), (budget, column, text)
