import numpy as np
import pytest
from qiskit_aer import AerSimulator

from quantic.solvers.quantum.circuit import qaoa_circuit
from quantic.solvers.quantum.ising import basis_energies, phases, to_ising
from quantic.solvers.quantum.statevector import Landscape, qaoa_state
from quantic.solvers.quantum.train import Angles, initial_angles, train
from tests.solvers.quantum.test_ising import random_qubo


def test_the_numpy_state_is_the_gate_circuits_state():
    qubo = random_qubo(6, 7, scale=3.0)
    ising = to_ising(qubo)
    gammas, betas = (0.41, 0.17, 0.9), (0.33, 0.62, 0.05)
    mine = qaoa_state(phases(qubo, ising), gammas, betas)
    circuit = qaoa_circuit(ising, gammas, betas, measure=False)
    circuit.save_statevector()
    aer = AerSimulator(method="statevector").run(circuit).result().get_statevector().data
    assert abs(np.vdot(mine, aer)) == pytest.approx(1.0, abs=1e-10)


def test_the_exact_cvar_matches_a_brute_force_one():
    qubo = random_qubo(5, 3)
    ising = to_ising(qubo)
    land = Landscape(qubo, ising)
    probs = np.abs(qaoa_state(land.phase, (0.3,), (0.4,))) ** 2
    energy = basis_energies(qubo)
    for alpha in (0.05, 0.25, 0.5, 1.0):
        order = np.argsort(energy, kind="stable")
        mass, total = 0.0, 0.0
        for k in order:
            take = min(probs[k], alpha - mass)
            if take <= 0:
                break
            total += take * energy[k]
            mass += take
        assert land.cvar(probs, alpha) == pytest.approx(total / alpha, rel=1e-9)
    assert land.cvar(probs, 1.0) == pytest.approx(float(probs @ energy), rel=1e-9)


def test_the_ramp_starts_low_gamma_high_beta():
    gammas, betas = initial_angles(3)
    assert gammas.tolist() == pytest.approx([0.125, 0.375, 0.625])
    assert betas.tolist() == pytest.approx([0.625, 0.375, 0.125])


def test_training_is_deterministic_and_improves_on_the_start():
    qubo = random_qubo(6, 11)
    a = train(qubo, reps=2, alpha=0.25, maxiter=60)
    b = train(qubo, reps=2, alpha=0.25, maxiter=60)
    assert (a.gammas, a.betas, a.cvar) == (b.gammas, b.betas, b.cvar)
    land = Landscape(qubo, to_ising(qubo))
    g0, b0 = initial_angles(2)
    start = land.cvar(np.abs(qaoa_state(land.phase, g0, b0)) ** 2, 0.25)
    assert a.cvar <= start
    assert 1 <= a.n_evals <= 60
    assert Angles.from_json(a.to_json()) == a


def test_trained_angles_put_the_minimum_among_the_likeliest_states():
    qubo = random_qubo(4, 2)
    land = Landscape(qubo, to_ising(qubo))
    angles = train(qubo, reps=3, alpha=0.25, maxiter=200)
    probs = np.abs(qaoa_state(land.phase, angles.gammas, angles.betas)) ** 2
    best = int(np.argmin(basis_energies(qubo)))
    assert probs[best] > 1.0 / 2**4


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_alpha_outside_zero_one_is_refused(alpha):
    with pytest.raises(ValueError, match="alpha"):
        train(random_qubo(3, 0), reps=1, alpha=alpha, maxiter=5)
