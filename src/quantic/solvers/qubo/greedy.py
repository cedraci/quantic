"""Steepest descent from uniformly random starts: the cheap floor."""

from __future__ import annotations

from dwave.samplers import SteepestDescentSampler

from quantic.solvers.qubo.sampler import QuboSampler, in_variable_order

NAME = "greedy"


class Greedy(QuboSampler):
    name = NAME

    def draw(self, bqm, reads, seed):
        found = SteepestDescentSampler().sample(bqm, num_reads=reads, seed=seed)
        return in_variable_order(found, bqm.num_variables)
