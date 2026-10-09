from pathlib import Path

from quantic.bench.config import QuantumSettings, load
from quantic.bench.keys import GATE, REF_QAOA_ANGLES, RUN, Task, expand
from quantic.bench.quantum_plan import QuboSizes, cap, is_hardware, is_quantum
from quantic.bench.runner import deadline_s, plan, run
from quantic.bench.store import OK, Store

ROOT = Path(__file__).resolve().parents[2]

SMALL = """
[instances]
tiers = ["T0"]
dials = ["D2", "D4", "D1+D2+D3"]
seeds = [2026]

[quantum]
reps = 1
maxiter = 10
shots = 20
max_qubits = 18
noisy_max_qubits = 16
gate_shots = 4
device = "ibm_fez"
workers = 2

[solvers.qaoa_ideal]
encodings = ["binary"]
seeds = "0..1"

[solvers.qaoa_noisy]
encodings = ["binary"]
seeds = "0..0"
"""


def _exp(tmp_path, text=SMALL):
    path = tmp_path / "e.toml"
    path.write_text(text, encoding="utf-8")
    return load(path)


def test_quantum_and_hardware_tasks_are_told_apart():
    hw = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="qaoa_ibm")
    assert is_quantum(hw) and is_hardware(hw)
    angles = Task(kind="reference", tier="T0", dials="none", instance_seed=1,
                  solver=REF_QAOA_ANGLES)
    assert is_quantum(angles) and not is_hardware(angles)
    assert not is_quantum(Task(kind=RUN, tier="T0", dials="none", instance_seed=1,
                               solver="cpsat"))


def test_tasks_over_the_qubit_caps_are_dropped_and_counted(tmp_path):
    exp = _exp(tmp_path)
    sizes = QuboSizes()
    jobs = [(str(i), t) for i, t in enumerate(expand(exp))]
    capped = cap(jobs, exp.quantum, sizes)
    kept = [t for _, t in capped.kept]
    for t in kept:
        if is_quantum(t):
            assert sizes.n_vars(t) <= 18
        if t.kind == RUN and t.solver == "qaoa_noisy":
            assert sizes.n_vars(t) <= 16
    assert capped.over_max > 0                       # the CVaR cell is hundreds of qubits
    assert not any(t.dials == "D1+D2+D3" and is_quantum(t) for t in kept)


def test_deadlines_follow_the_quantum_settings():
    q = QuantumSettings(train_deadline_s=111.0, run_deadline_s=222.0)
    train = Task(kind="reference", tier="T0", dials="D2", instance_seed=1,
                 solver=REF_QAOA_ANGLES)
    gate = Task(kind=GATE, tier="T0", dials="D2", instance_seed=1, solver="qaoa_noisy")
    assert deadline_s(train, 30.0, q) == 111.0
    assert deadline_s(gate, 30.0, q) == 222.0


def test_a_quantum_study_trains_before_it_samples(tmp_path):
    exp = _exp(tmp_path)
    store = tmp_path / "s.duckdb"
    run(exp, root=ROOT, store=store, workers=2, allow_dirty=True, no_hardware=True)
    with Store(store, read_only=True) as s:
        runs = s.runs()
    trained = runs.filter(runs["solver"] == REF_QAOA_ANGLES)
    sampled = runs.filter(runs["solver"].is_in(["qaoa_ideal", "qaoa_noisy"]))
    assert trained.height > 0 and (trained["store_status"] == OK).all()
    assert sampled.height > 0 and (sampled["store_status"] == OK).all()
    assert set(sampled["angle_key"].to_list()) <= set(trained["angle_key"].to_list())
    p = plan(exp, root=ROOT, store=store)
    assert p.n_todo == 0 and p.n_over_max > 0
