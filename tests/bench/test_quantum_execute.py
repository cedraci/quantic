import json

from quantic.bench.angles import AngleCache, angle_key, angles_dir_for
from quantic.bench.execute import Executor
from quantic.bench.keys import GATE, REF_QAOA_ANGLES, REFERENCE, RUN, Task
from quantic.bench.store import ERROR, OK
from quantic.solvers.quantum.train import Angles
from tests.bench.rows import CODE

PARAMS = json.dumps({"alpha": 0.25, "maxiter": 15, "reps": 1}, sort_keys=True,
                    separators=(",", ":"))
DEVICE = json.dumps({"alpha": 0.25, "device": "ibm_fez", "maxiter": 15, "reps": 1},
                    sort_keys=True, separators=(",", ":"))
AT = dict(tier="T0", dials="D2", instance_seed=2026, encoding="binary", zeta_bits=8)


def _ex(tmp_path):
    return Executor(CODE, {"angles_dir": str(tmp_path / "a")})


def test_the_cache_round_trips_and_writes_atomically(tmp_path):
    cache = AngleCache(tmp_path / "a")
    a = Angles((0.1,), (0.2,), -1.0, 5, 0.3)
    assert cache.get("k") is None
    cache.put("k", a, {"qubo_hash": "h"})
    assert cache.get("k") == a
    assert not list((tmp_path / "a").glob("*.tmp"))
    assert angles_dir_for(tmp_path / "s.duckdb") == tmp_path / "s.angles"
    assert angle_key("h", 1, 0.25, 15) != angle_key("h", 2, 0.25, 15)


def test_a_training_reference_trains_once_then_reads_the_cache(tmp_path):
    ex = _ex(tmp_path)
    task = Task(kind=REFERENCE, solver=REF_QAOA_ANGLES, params=PARAMS, **AT)
    row, _, _ = ex.run("k1", task)
    assert row["store_status"] == OK and row["status"] == "trained"
    assert row["angle_key"] and row["reps"] == 1 and row["n_vars"] == 16
    assert row["detail"]["cached"] is False and row["total_s"] > 0
    again, _, _ = ex.run("k2", task)
    assert again["detail"]["cached"] is True
    assert again["angle_key"] == row["angle_key"] and again["total_s"] == row["total_s"]


def test_a_qaoa_run_without_angles_is_an_error_row(tmp_path):
    task = Task(kind=RUN, solver="qaoa_ideal", solver_seed=0, max_samples=20,
                params=PARAMS, **AT)
    row, _, _ = _ex(tmp_path).run("k", task)
    assert row["store_status"] == ERROR and "MissingAngles" in row["error"]


def test_an_ideal_run_and_a_gate_after_training(tmp_path):
    ex = _ex(tmp_path)
    ex.run("t", Task(kind=REFERENCE, solver=REF_QAOA_ANGLES, params=PARAMS, **AT))
    run = Task(kind=RUN, solver="qaoa_ideal", solver_seed=3, max_samples=20,
               params=PARAMS, **AT)
    row, samples, _ = ex.run("r", run)
    assert row["store_status"] == OK and row["status"] == "heuristic"
    assert row["n_samples"] == 20 and row["shots"] == 20 and row["backend"] == "ideal"
    assert sum(s[1] for s in samples) == 20
    gate = Task(kind=GATE, solver="qaoa_noisy", solver_seed=0, max_samples=8,
                params=DEVICE, **AT)
    grow, _, _ = ex.run("g", gate)
    assert grow["store_status"] == OK and grow["backend"] == "noisy"
    assert grow["device"] == "ibm_fez" and grow["routed_2q"] > 0
