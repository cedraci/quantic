"""The ladder both gates are run on.

M3a's gate and M3b's build the same sixty-four instances, from the same
parameters and the same seed. They are built here so that "the same" is a
fact about one function and not about two copies of it.
"""

from __future__ import annotations

import functools

from quantic.bench.instances import GATE_SEED, N_RAW, N_TARGET, synthetic_params
from quantic.problem.generator import generate_ladder
from quantic.problem.instance import Instance

SEED = GATE_SEED
SMALL = {"n_raw": N_RAW, "n_target": N_TARGET}
params_for = synthetic_params


@functools.cache
def gate_ladder() -> tuple[Instance, ...]:
    """All sixty-four instances. Built once per process: generation is pure."""
    return generate_ladder(params_for, seed=SEED, include_cvar=True, **SMALL)


def tier(name: str) -> tuple[Instance, ...]:
    return tuple(i for i in gate_ladder() if i.tier == name)
