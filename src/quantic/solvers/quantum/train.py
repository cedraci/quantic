"""CVaR-QAOA angles, trained on the noiseless statevector -- M5 design 5.2-5.3.

The start is the linear ramp and COBYLA is deterministic, so the same QUBO,
depth, alpha and iteration cap always give the same angles. They are trained
once and shared by every backend (decision D3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quantic.encoding.qubo import Qubo
from quantic.solvers.quantum.ising import to_ising
from quantic.solvers.quantum.statevector import Landscape, qaoa_state

DELTA = 0.75
ANGLE_RULE = f"linear-ramp-{DELTA}"
# Part of every angle cache key. Bump it whenever the trainer, the statevector
# or the Ising map changes what angles a QUBO trains to, or cached angles from
# the old code would be reused as if the new code had trained them.
TRAINER_VERSION = 1
_RHOBEG = 0.25


@dataclass(frozen=True)
class Angles:
    gammas: tuple[float, ...]
    betas: tuple[float, ...]
    cvar: float
    n_evals: int
    train_s: float

    def to_json(self) -> dict:
        return {"gammas": list(self.gammas), "betas": list(self.betas), "cvar": self.cvar,
                "n_evals": self.n_evals, "train_s": self.train_s}

    @classmethod
    def from_json(cls, body: dict) -> Angles:
        return cls(gammas=tuple(float(g) for g in body["gammas"]),
                   betas=tuple(float(b) for b in body["betas"]),
                   cvar=float(body["cvar"]), n_evals=int(body["n_evals"]),
                   train_s=float(body["train_s"]))


def initial_angles(reps: int) -> tuple[np.ndarray, np.ndarray]:
    k = (np.arange(reps) + 0.5) / reps
    return k * DELTA, (1.0 - k) * DELTA


def train(qubo: Qubo, *, reps: int, alpha: float, maxiter: int) -> Angles:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
    if reps < 1 or maxiter < 1:
        raise ValueError(f"reps and maxiter must be at least 1, got {reps}, {maxiter}")
    started = time.perf_counter()
    land = Landscape(qubo, to_ising(qubo))

    def objective(x: np.ndarray) -> float:
        state = qaoa_state(land.phase, x[:reps], x[reps:])
        return land.cvar(np.abs(state) ** 2, alpha)

    gammas, betas = initial_angles(reps)
    found = minimize(objective, np.concatenate([gammas, betas]), method="COBYLA",
                     options={"maxiter": maxiter, "rhobeg": _RHOBEG})
    return Angles(
        gammas=tuple(float(v) for v in found.x[:reps]),
        betas=tuple(float(v) for v in found.x[reps:]),
        cvar=float(found.fun), n_evals=int(found.nfev),
        train_s=time.perf_counter() - started,
    )
