"""The QUBO as an Ising operator, for a gate-model circuit -- M5 design section 5.1.

``x_i = (1 - z_i) / 2`` turns ``x'Qx + c`` into ``sum h_i z_i + sum J_ij z_i z_j
+ offset``. A measured bit ``b_i`` is ``x_i``: ``|1>`` is ``z = -1``. The
coefficients are divided by ``scale``, the largest of them, so the circuit's
angles stay in a fixed range, and ``scale * ising + offset`` gives back the
QUBO's energy exactly.

Qiskit and Aer are little-endian: basis state ``k`` has bit ``(k >> i) & 1`` on
qubit ``i``, which is variable ``i`` here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from quantic.encoding.qubo import Qubo


@dataclass(frozen=True, eq=False)
class Ising:
    h: np.ndarray
    J: np.ndarray
    scale: float
    offset: float
    dynamic_range: float

    @property
    def n(self) -> int:
        return int(self.h.size)

    def energy(self, bits) -> float:
        """The QUBO's energy at ``bits``, computed through the Ising form."""
        z = 1.0 - 2.0 * np.asarray(bits, dtype=float)
        return float(self.scale * (self.h @ z + z @ self.J @ z) + self.offset)


def to_ising(qubo: Qubo) -> Ising:
    q = qubo.Q
    n = qubo.n_vars
    diag = np.diag(q).copy()
    upper = np.triu(q, 1)
    h = -diag / 2.0 - (upper + upper.T).sum(axis=1) / 4.0
    j = upper / 4.0
    offset = float(qubo.constant + diag.sum() / 2.0 + upper.sum() / 4.0)
    coefs = np.concatenate([np.abs(h), np.abs(j[np.triu_indices(n, 1)])])
    nonzero = coefs[coefs > 0.0]
    scale = float(nonzero.max()) if nonzero.size else 1.0
    dynamic_range = float(nonzero.max() / nonzero.min()) if nonzero.size else 1.0
    return Ising(h=h / scale, J=j / scale, scale=scale, offset=offset,
                 dynamic_range=dynamic_range)


def basis_energies(qubo: Qubo) -> np.ndarray:
    """The QUBO energy of every basis state, built one variable at a time.

    Adding variable ``i`` doubles the table: a state with bit ``i`` set costs the
    same state without it, plus ``Q_ii``, plus ``Q_ji`` for each lower bit ``j``
    it has set. That field is itself built by doubling, so the whole table costs
    about ``2**(n+1)`` additions, not ``n**2 * 2**n``.
    """
    q = qubo.Q
    table = np.array([float(qubo.constant)])
    for i in range(qubo.n_vars):
        field = np.zeros(1)
        for j in range(i):
            field = np.concatenate([field, field + q[j, i]])
        table = np.concatenate([table, table + q[i, i] + field])
    return table


def phases(qubo: Qubo, ising: Ising) -> np.ndarray:
    """Each basis state's scaled Ising energy: the diagonal of the cost operator."""
    return (basis_energies(qubo) - ising.offset) / ising.scale


def qubo_hash(qubo: Qubo) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(qubo.Q, dtype=np.float64).tobytes())
    digest.update(repr(float(qubo.constant)).encode("utf-8"))
    digest.update(repr(qubo.registry.keys).encode("utf-8"))
    return digest.hexdigest()
