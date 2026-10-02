import numpy as np
import pytest

from quantic.encoding.decode import FEASIBLE
from quantic.encoding.encode import encode
from quantic.encoding.integer import BinaryEncoding, OneHotEncoding
from quantic.encoding.qubo import Qubo
from quantic.encoding.registry import VarRegistry
from quantic.solvers.base import HEURISTIC, Budget, Solver
from quantic.solvers.classical.qubo_minimum import qubo_minimum
from quantic.solvers.classical.scoring import score
from quantic.solvers.qubo.greedy import Greedy
from quantic.solvers.qubo.sampler import energies, to_bqm
from quantic.solvers.qubo.simulated_annealing import SimulatedAnnealing
from quantic.solvers.qubo.tabu import Tabu
from tests.solvers.instances import cvar, discrete, plain

SAMPLERS = (SimulatedAnnealing, Tabu, Greedy)


def test_the_bqm_has_every_variable_and_the_same_energy():
    registry = VarRegistry(keys=tuple(("x", i) for i in range(4)))
    q = np.zeros((4, 4))
    q[0, 0], q[0, 2], q[2, 2] = 1.5, -2.0, 0.25          # bits 1 and 3 appear nowhere
    qubo = Qubo(Q=q, constant=3.0, registry=registry)
    bqm = to_bqm(qubo)
    assert bqm.num_variables == 4
    for bits in np.ndindex(2, 2, 2, 2):
        assert bqm.energy(dict(enumerate(bits))) == pytest.approx(qubo.energy(bits))
    every = np.array(list(np.ndindex(2, 2, 2, 2)))
    assert energies(qubo, every) == pytest.approx([qubo.energy(b) for b in every])


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda s: s.name)
@pytest.mark.parametrize("build", [plain, discrete, cvar], ids=lambda b: b.__name__)
def test_every_sample_is_counted_judged_and_scored_by_the_evaluator(sampler, build):
    inst = build()
    result = sampler("binary", seed=1).solve(inst, Budget(max_samples=20))
    detail = result.detail
    assert result.status == HEURISTIC
    assert detail["n_samples"] == 20
    assert sum(s.count for s in detail["samples"]) == 20
    assert detail["energy_mismatch"] <= 1e-9
    for s in detail["samples"]:
        assert (s.total is not None) == (s.classification == FEASIBLE)
    if result.schedule is not None:
        assert result.feasibility.feasible
        assert result.objective == score(inst, result.schedule)[0]
        assert result.objective.total == min(
            s.total for s in detail["samples"] if s.total is not None
        )


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda s: s.name)
def test_the_same_seed_draws_the_same_samples_and_a_shorter_run_is_a_prefix(sampler):
    inst = discrete()
    short = sampler("one_hot", seed=4).solve(inst, Budget(max_samples=7))
    again = sampler("one_hot", seed=4).solve(inst, Budget(max_samples=7))
    longer = sampler("one_hot", seed=4).solve(inst, Budget(max_samples=15))
    assert short.detail["samples"] == again.detail["samples"]
    assert {s.digest for s in short.detail["samples"]} <= {
        s.digest for s in longer.detail["samples"]
    }


def test_the_trace_only_ever_goes_down():
    result = SimulatedAnnealing("binary", seed=2).solve(cvar(), Budget(max_samples=31))
    trace = result.detail["trace"]
    assert trace
    assert all(b[1] < a[1] for a, b in zip(trace, trace[1:], strict=False))
    assert trace[-1][1] == result.detail["best_energy"]


def test_annealing_finds_the_qubos_minimum_on_a_small_instance():
    enc = encode(plain(), BinaryEncoding(), n_zeta_bits=8)
    found = SimulatedAnnealing("binary", seed=0).solve_encoded(enc, Budget(max_samples=63))
    lowest = qubo_minimum(enc).energy
    assert found.detail["best_energy"] == pytest.approx(lowest, rel=1e-9)


def test_a_wall_clock_run_draws_at_least_one_batch():
    result = Greedy("binary", seed=0).solve(plain(), Budget(wall_clock_s=1e-6))
    assert result.detail["n_batches"] == 1
    assert result.detail["n_samples"] == 1


def test_a_sampler_with_no_stopping_rule_is_refused():
    with pytest.raises(ValueError, match="sample cap"):
        Greedy("binary").solve(plain(), Budget())


def test_an_instance_encoded_another_way_is_refused():
    enc = encode(plain(), OneHotEncoding())
    with pytest.raises(ValueError, match="one_hot"):
        Greedy("binary").solve_encoded(enc, Budget(max_samples=1))


def test_an_unknown_encoding_is_refused():
    with pytest.raises(KeyError, match="gray"):
        Greedy("gray")


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda s: s.name)
def test_each_is_a_solver(sampler):
    assert isinstance(sampler("binary"), Solver)
