"""Solvers that misbehave on purpose, loaded by import path in a worker."""

from __future__ import annotations

import time


class Hang:
    name = "hang"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def solve(self, instance, budget):
        time.sleep(3600)


class Boom:
    name = "boom"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def solve(self, instance, budget):
        raise ArithmeticError("this solver always fails")
