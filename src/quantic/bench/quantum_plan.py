"""Which tasks are quantum, which need hardware, and which are too big to plan.

M5 design sections 2 and 6: a QUBO above ``max_qubits`` is never planned, and a
full noisy run above ``noisy_max_qubits`` is never planned. The size is read by
encoding the instance in the main process, which costs milliseconds at T0.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantic.bench.config import QUANTUM_SOLVERS, QuantumSettings
from quantic.bench.instances import InstanceCache
from quantic.bench.keys import REF_QAOA_ANGLES, RUN, Task
from quantic.encoding.encode import encode
from quantic.solvers.qubo.sampler import encoding_named

_QUANTUM = (REF_QAOA_ANGLES, *QUANTUM_SOLVERS)


def is_quantum(task: Task) -> bool:
    return task.solver in _QUANTUM


def is_hardware(task: Task) -> bool:
    return task.kind == RUN and task.solver == "qaoa_ibm"


class QuboSizes:
    def __init__(self) -> None:
        self.instances = InstanceCache()
        self._sizes: dict[tuple, int] = {}

    def n_vars(self, task: Task) -> int:
        key = (task.tier, task.dials, task.instance_seed, task.encoding, task.zeta_bits)
        if key not in self._sizes:
            instance = self.instances.get(task.tier, task.dials, task.instance_seed)
            self._sizes[key] = encode(instance, encoding_named(task.encoding),
                                      n_zeta_bits=task.zeta_bits).n_vars
        return self._sizes[key]


@dataclass(frozen=True)
class Capped:
    kept: list
    over_max: int
    over_noisy: int


def cap(jobs, settings: QuantumSettings | None, sizes: QuboSizes) -> Capped:
    if settings is None:
        return Capped(kept=list(jobs), over_max=0, over_noisy=0)
    kept, over_max, over_noisy = [], 0, 0
    for key, task in jobs:
        if is_quantum(task):
            n = sizes.n_vars(task)
            if n > settings.max_qubits:
                over_max += 1
                continue
            if task.kind == RUN and task.solver == "qaoa_noisy" and n > settings.noisy_max_qubits:
                over_noisy += 1
                continue
        kept.append((key, task))
    return Capped(kept=kept, over_max=over_max, over_noisy=over_noisy)
