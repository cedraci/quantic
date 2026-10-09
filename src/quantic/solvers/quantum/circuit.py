"""The QAOA circuit at given angles, for routing and for real or noisy sampling.

``RZ(2 gamma h_i)`` is ``exp(-i gamma h_i Z_i)``, ``RZZ(2 gamma J_ij)`` is
``exp(-i gamma J_ij Z_i Z_j)`` and ``RX(2 beta)`` is ``exp(-i beta X)``: the same
unitary as ``statevector.qaoa_state``.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from quantic.solvers.quantum.ising import Ising


def qaoa_circuit(ising: Ising, gammas, betas, *, measure: bool = True) -> QuantumCircuit:
    n = ising.n
    if n == 0:
        raise ValueError("a QUBO with no variables has no circuit")
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    rows, cols = np.nonzero(ising.J)
    for gamma, beta in zip(gammas, betas, strict=True):
        for i in range(n):
            if ising.h[i] != 0.0:
                circuit.rz(2.0 * float(gamma) * float(ising.h[i]), i)
        for i, j in zip(rows, cols, strict=True):
            circuit.rzz(2.0 * float(gamma) * float(ising.J[i, j]), int(i), int(j))
        circuit.rx(2.0 * float(beta), range(n))
    if measure:
        circuit.measure_all()
    return circuit
