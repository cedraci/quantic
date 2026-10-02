"""Tabu search on the QUBO (``dwave.samplers``, formerly ``dwave-tabu``).

Tabu's own stopping rule is a wall clock per read, which would make a run's
samples depend on the machine's load. Each read is bounded by restarts
instead, with the timeout set out of reach; the bench's per-run kill is the
backstop.
"""

from __future__ import annotations

from dwave.samplers import TabuSampler

from quantic.solvers.qubo.sampler import QuboSampler, in_variable_order

NAME = "tabu"
RESTARTS_PER_READ = 10
_NEVER_MS = 10**9


class Tabu(QuboSampler):
    name = NAME

    def draw(self, bqm, reads, seed):
        found = TabuSampler().sample(
            bqm, num_reads=reads, seed=seed,
            num_restarts=RESTARTS_PER_READ, timeout=_NEVER_MS,
        )
        return in_variable_order(found, bqm.num_variables)
