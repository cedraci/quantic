"""Where a QAOA circuit's shots come from -- M5 design section 6."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from qiskit import QuantumCircuit, transpile

from quantic.encoding.qubo import Qubo
from quantic.solvers.quantum.circuit import qaoa_circuit
from quantic.solvers.quantum.ising import Ising, phases, to_ising
from quantic.solvers.quantum.statevector import qaoa_state
from quantic.solvers.quantum.train import Angles

OPTIMIZATION_LEVEL = 3


def bits_from_counts(counts: Mapping[str, int], n: int) -> np.ndarray:
    """One row per shot, variable ``i`` in column ``i``.

    Qiskit writes clbit 0 rightmost, and ``measure_all`` puts qubit ``i`` on
    clbit ``i``.
    """
    rows = []
    for key, count in sorted(counts.items()):
        text = key.replace(" ", "")
        if len(text) != n:
            raise ValueError(f"a count key {key!r} does not have {n} bits")
        row = np.array([int(text[n - 1 - i]) for i in range(n)], dtype=np.int8)
        rows.append(np.repeat(row[None, :], int(count), axis=0))
    return np.concatenate(rows) if rows else np.zeros((0, n), dtype=np.int8)


def sample_ideal(qubo: Qubo, angles: Angles, shots: int, seed: int) -> np.ndarray:
    ising = to_ising(qubo)
    probs = np.abs(qaoa_state(phases(qubo, ising), angles.gammas, angles.betas)) ** 2
    probs /= probs.sum()
    picks = np.random.default_rng(seed).choice(probs.size, size=shots, p=probs)
    return ((picks[:, None] >> np.arange(ising.n)[None, :]) & 1).astype(np.int8)


def fake_backend(device: str):
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    provider = FakeProviderForBackendV2()
    name = "fake_" + device.removeprefix("ibm_")
    try:
        return provider.backend(name)
    except Exception as exc:  # QiskitBackendNotFoundError, whatever its module
        known = sorted(b.name for b in provider.backends())
        raise KeyError(f"no fake twin {name!r} for device {device!r}; known: {known}") from exc


def route(ising: Ising, angles: Angles, backend, *, seed_transpiler: int = 0) -> QuantumCircuit:
    circuit = qaoa_circuit(ising, angles.gammas, angles.betas, measure=True)
    return transpile(circuit, backend, optimization_level=OPTIMIZATION_LEVEL,
                     seed_transpiler=seed_transpiler)


def routed_stats(routed: QuantumCircuit) -> tuple[int, int]:
    two = sum(1 for inst in routed.data
              if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")
    return int(routed.depth()), int(two)


def sample_noisy(routed: QuantumCircuit, backend, shots: int, seed: int) -> np.ndarray:
    from qiskit_aer import AerSimulator

    simulator = AerSimulator.from_backend(backend)
    counts = simulator.run(routed, shots=shots, seed_simulator=seed).result().get_counts()
    return bits_from_counts(counts, routed.num_clbits)
