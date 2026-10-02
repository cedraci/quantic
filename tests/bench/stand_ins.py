"""Solvers that misbehave on purpose, loaded by import path in a worker."""

from __future__ import annotations

import os
import threading
import time

from quantic.solvers.classical.almgren_chriss import AlmgrenChrissSolver


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


class Nap:
    """Answers correctly, after ``seed`` tenths of a second."""

    name = "nap"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def solve(self, instance, budget):
        time.sleep(self.seed / 10)
        return AlmgrenChrissSolver().solve(instance, budget)


class Die:
    """Takes its worker down mid-task, as a native crash would."""

    name = "die"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def solve(self, instance, budget):
        os._exit(3)


class DieWhenIdle:
    """Answers, then takes its worker down half a second later, while it is idle."""

    name = "die_when_idle"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def solve(self, instance, budget):
        threading.Timer(0.5, os._exit, args=(4,)).start()
        return AlmgrenChrissSolver().solve(instance, budget)
