"""Simulated annealing on the QUBO (``dwave.samplers``, formerly ``neal``)."""

from __future__ import annotations

from dwave.samplers import SimulatedAnnealingSampler

from quantic.solvers.qubo.sampler import QuboSampler, in_variable_order

NAME = "simulated_annealing"


class SimulatedAnnealing(QuboSampler):
    name = NAME

    def draw(self, bqm, reads, seed):
        found = SimulatedAnnealingSampler().sample(bqm, num_reads=reads, seed=seed)
        return in_variable_order(found, bqm.num_variables)
