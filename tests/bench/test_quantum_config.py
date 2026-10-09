import json

import pytest

from quantic.bench.config import ConfigError, QuantumSettings, load
from quantic.bench.keys import GATE, REF_QAOA_ANGLES, RUN, Task, expand, gate_for

BASE = """
[instances]
tiers = ["T0"]
dials = ["none", "D4"]
seeds = [2026]
"""

QAOA = BASE + """
[quantum]
reps = 2
shots = 100
device = "ibm_fez"

[solvers.qaoa_ideal]
encodings = ["binary"]
seeds = "0..1"

[solvers.qaoa_ibm]
encodings = ["binary"]
seeds = "0..1"
"""


def _load(tmp_path, text):
    path = tmp_path / "e.toml"
    path.write_text(text, encoding="utf-8")
    return load(path)


def test_the_quantum_section_has_the_spec_defaults(tmp_path):
    exp = _load(tmp_path, QAOA)
    q = exp.quantum
    assert (q.reps, q.alpha, q.maxiter, q.shots, q.max_qubits, q.noisy_max_qubits,
            q.gate_shots, q.device, q.quota_cap_s) == (2, 0.25, 200, 100, 28, 20, 32,
                                                       "ibm_fez", 300.0)


def test_qaoa_tasks_have_no_budget_and_carry_their_parameters(tmp_path):
    tasks = expand(_load(tmp_path, QAOA))
    ideal = [t for t in tasks if t.kind == RUN and t.solver == "qaoa_ideal"]
    assert len(ideal) == 2 * 2                      # two dials, two seeds
    assert all(t.budget_s is None and t.max_samples == 100 for t in ideal)
    assert json.loads(ideal[0].params) == {"alpha": 0.25, "maxiter": 200, "reps": 2}
    ibm = [t for t in tasks if t.kind == RUN and t.solver == "qaoa_ibm"]
    assert json.loads(ibm[0].params)["device"] == "ibm_fez"


def test_every_qaoa_instance_and_encoding_gets_one_training_reference(tmp_path):
    tasks = expand(_load(tmp_path, QAOA))
    trains = [t for t in tasks if t.solver == REF_QAOA_ANGLES]
    assert len(trains) == 2
    assert {t.encoding for t in trains} == {"binary"}
    assert tasks.index(trains[0]) < min(i for i, t in enumerate(tasks) if t.kind == RUN)


def test_one_gate_per_hardware_coordinate_not_per_seed(tmp_path):
    exp = _load(tmp_path, QAOA)
    tasks = expand(exp)
    gates = [t for t in tasks if t.kind == GATE]
    assert len(gates) == 2                          # two dials; seeds 0 and 1 share one
    ibm = next(t for t in tasks if t.solver == "qaoa_ibm")
    gate = gate_for(ibm, exp.quantum)
    assert gate in gates
    assert gate.solver == "qaoa_noisy" and gate.max_samples == 32


def test_identity_ignores_the_code_and_keys_do_not():
    from quantic.bench.keys import CodeVersion

    task = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="qaoa_ibm",
                params="{}")
    a, b = CodeVersion("c1", "l", False), CodeVersion("c2", "l", False)
    assert task.key(a) != task.key(b)
    assert len(task.identity()) == 64 and task.identity() == task.identity()


@pytest.mark.parametrize("extra,match", [
    ("[solvers.qaoa_ideal]\nencodings = [\"binary\"]\n", "needs a \\[quantum\\]"),
    ("[quantum]\nquota_cap_s = 301\n[solvers.qaoa_ideal]\nencodings = [\"binary\"]\n", "300"),
    ("[quantum]\ndevice = \"ibm_nowhere\"\n[solvers.qaoa_ideal]\nencodings = [\"binary\"]\n",
     "fake twin"),
    ("[quantum]\nalpha = 0\n[solvers.qaoa_ideal]\nencodings = [\"binary\"]\n", "alpha"),
    ("[quantum]\n[solvers.qaoa_ideal]\nencodings = [\"binary\"]\nbudgets_s = [1]\n",
     "budgets_s"),
    ("[quantum]\nbogus = 1\n[solvers.qaoa_ideal]\nencodings = [\"binary\"]\n", "bogus"),
])
def test_a_quantum_config_that_does_not_say_what_it_means_is_refused(tmp_path, extra, match):
    with pytest.raises(ConfigError, match=match):
        _load(tmp_path, BASE + extra)


def test_an_m4_config_still_loads_and_has_no_quantum(tmp_path):
    exp = _load(tmp_path, BASE + "[solvers.cpsat]\nbudgets_s = [1]\n")
    assert exp.quantum is None
    assert QuantumSettings().quota_cap_s == 300.0
