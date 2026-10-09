"""Exact QAOA on a numpy statevector -- M5 design section 5.3.

The cost layer ``exp(-i gamma H)`` is diagonal in the computational basis, so
it is one elementwise phase from the precomputed Ising energies. The mixer is
``RX(2 beta)`` on each qubit, one pass over the state per qubit. That costs
``O(p n 2**n)``, where simulating the circuit gate by gate costs a pass per
ZZ term, ``O(p n**2 2**n)``. ``tests/solvers/quantum/test_train.py`` checks
the two agree.
"""

from __future__ import annotations

import numpy as np

from quantic.encoding.qubo import Qubo
from quantic.solvers.quantum.ising import Ising, basis_energies


def _mix(state: np.ndarray, beta: float, n: int) -> None:
    c, s = np.cos(beta), np.sin(beta)
    for i in range(n):
        view = state.reshape(-1, 2, 2**i)
        a0 = view[:, 0, :].copy()
        a1 = view[:, 1, :].copy()
        view[:, 0, :] = c * a0 - 1j * s * a1
        view[:, 1, :] = c * a1 - 1j * s * a0


def qaoa_state(phase: np.ndarray, gammas, betas) -> np.ndarray:
    size = int(phase.size)
    n = size.bit_length() - 1
    if size != 2**n:
        raise ValueError(f"the phase table has {size} entries, not a power of two")
    state = np.full(size, 1.0 / np.sqrt(size), dtype=np.complex128)
    for gamma, beta in zip(gammas, betas, strict=True):
        state *= np.exp(-1j * float(gamma) * phase)
        _mix(state, float(beta), n)
    return state


class Landscape:
    """One QUBO's energies, sorted once, for the exact CVaR of any state."""

    def __init__(self, qubo: Qubo, ising: Ising) -> None:
        energy = basis_energies(qubo)
        self.phase = (energy - ising.offset) / ising.scale
        self.order = np.argsort(energy, kind="stable")
        self.sorted = energy[self.order]
        self.minimum = float(self.sorted[0])

    def cvar(self, probs: np.ndarray, alpha: float) -> float:
        """Mean energy of the lowest ``alpha`` of the probability mass."""
        mass = probs[self.order]
        cum = np.cumsum(mass)
        m = min(int(np.searchsorted(cum, alpha)), mass.size - 1)
        head = float(mass[:m] @ self.sorted[:m])
        before = float(cum[m - 1]) if m else 0.0
        return (head + (alpha - before) * float(self.sorted[m])) / alpha
