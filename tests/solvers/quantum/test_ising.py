import itertools

import numpy as np
import pytest

from quantic.encoding.qubo import Qubo
from quantic.encoding.registry import VarRegistry
from quantic.solvers.quantum.ising import basis_energies, phases, qubo_hash, to_ising


def random_qubo(n: int, seed: int, *, scale: float = 1.0) -> Qubo:
    rng = np.random.default_rng(seed)
    q = np.triu(rng.normal(size=(n, n)) * scale)
    q[rng.random((n, n)) < 0.3] = 0.0
    q = np.triu(q)
    return Qubo(Q=q, constant=float(rng.normal()),
                registry=VarRegistry(keys=tuple(("x", i) for i in range(n))))


@pytest.mark.parametrize("n,seed", [(1, 0), (2, 1), (5, 2), (8, 3), (10, 4)])
def test_the_ising_form_gives_back_the_qubo_energy_on_every_bitstring(n, seed):
    qubo = random_qubo(n, seed, scale=1e3)
    ising = to_ising(qubo)
    for bits in itertools.product((0, 1), repeat=n):
        assert ising.energy(bits) == pytest.approx(qubo.energy(bits), rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("n,seed", [(1, 0), (3, 1), (7, 2), (10, 3)])
def test_basis_energies_are_little_endian_and_exact(n, seed):
    qubo = random_qubo(n, seed)
    table = basis_energies(qubo)
    assert table.shape == (2**n,)
    for k in range(2**n):
        bits = [(k >> i) & 1 for i in range(n)]
        assert table[k] == pytest.approx(qubo.energy(bits), rel=1e-12, abs=1e-12)


def test_the_largest_scaled_coefficient_is_one_and_phases_undo_the_scale():
    qubo = random_qubo(6, 5, scale=250.0)
    ising = to_ising(qubo)
    top = max(np.abs(ising.h).max(), np.abs(ising.J).max())
    assert top == pytest.approx(1.0)
    assert np.all(np.tril(ising.J) == 0.0)
    assert ising.dynamic_range >= 1.0
    table = basis_energies(qubo)
    assert phases(qubo, ising) * ising.scale + ising.offset == pytest.approx(table)


def test_a_qubo_with_no_coefficient_has_scale_one():
    q = np.zeros((3, 3))
    qubo = Qubo(Q=q, constant=2.0, registry=VarRegistry(keys=tuple(("x", i) for i in range(3))))
    ising = to_ising(qubo)
    assert ising.scale == 1.0 and ising.dynamic_range == 1.0
    assert ising.energy([1, 0, 1]) == pytest.approx(2.0)


def test_the_hash_names_the_content():
    a, b = random_qubo(5, 1), random_qubo(5, 1)
    c = random_qubo(5, 2)
    assert qubo_hash(a) == qubo_hash(b)
    assert qubo_hash(a) != qubo_hash(c)
    assert len(qubo_hash(a)) == 64
