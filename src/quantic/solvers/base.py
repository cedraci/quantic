"""The solver interface -- spec section 7.1, at the size the exact solvers need.

Every solver answers ``solve(instance, budget) -> Result``. What comes back is
a schedule and the evaluator's verdict on it, never the solver's own account
of what its schedule costs: ``Result.objective`` is
``problem.objective.evaluate`` at ``Result.schedule``, and a solver that
modelled the objective wrongly shows up as a bracket that does not close
rather than as a better number.

A sampler's energy trace travels in ``Result.detail["trace"]`` as
``(seconds_since_start, best_energy)`` pairs; only samplers have one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from quantic.problem.feasibility import FeasibilityReport
from quantic.problem.instance import Instance
from quantic.problem.objective import ObjectiveBreakdown
from quantic.problem.schedule import Schedule

OPTIMAL = "optimal"
TIME_LIMIT = "time_limit"
INFEASIBLE = "infeasible"
# A schedule from a method that never proves anything: a sampler, a rounding.
# Not TIME_LIMIT, which says that more time could have produced a proof.
HEURISTIC = "heuristic"

_STATUSES = (OPTIMAL, TIME_LIMIT, INFEASIBLE, HEURISTIC)

# A proven lower bound comes out of a solver working to its own tolerance, so
# it can sit a hair above the value it bounds. Beyond this it is not round-off.
_BRACKET_RTOL = 1e-6


@dataclass(frozen=True, slots=True)
class Budget:
    """What a solver may spend. ``None`` means no limit."""

    wall_clock_s: float | None = None
    target: float | None = None
    max_samples: int | None = None

    def __post_init__(self) -> None:
        if self.wall_clock_s is not None and self.wall_clock_s <= 0:
            raise ValueError(f"wall_clock_s must be positive, got {self.wall_clock_s}")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError(f"max_samples must be at least 1, got {self.max_samples}")


@dataclass(frozen=True, slots=True)
class Bracket:
    """The optimum lies in ``[lower, upper]``.

    ``lower`` is proven: no feasible schedule costs less. ``upper`` is
    attained: it is the evaluator's total at a feasible schedule.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper):
            raise ValueError(
                f"a bracket cannot have a NaN bound: lower={self.lower!r}, "
                f"upper={self.upper!r}"
            )
        slack = _BRACKET_RTOL * max(1.0, abs(self.upper))
        if self.lower > self.upper + slack:
            raise ValueError(
                f"lower bound {self.lower!r} is above the attained value "
                f"{self.upper!r}: a schedule cannot cost less than the proven "
                "minimum, so one of the two is wrong"
            )

    @property
    def width(self) -> float:
        return max(self.upper - self.lower, 0.0)


@dataclass(frozen=True)
class Result:
    """One solver's answer on one instance."""

    solver: str
    status: str
    schedule: Schedule | None
    zeta: float | None
    objective: ObjectiveBreakdown | None
    bracket: Bracket | None
    feasibility: FeasibilityReport | None
    timings: Mapping[str, float] = field(default_factory=dict)
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(
                f"status must be one of {_STATUSES}, got {self.status!r}"
            )
        if self.status == OPTIMAL:
            self._check_optimal()
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))

    def _check_optimal(self) -> None:
        """``OPTIMAL`` is a claim, and the result carries its evidence."""
        if self.schedule is None or self.bracket is None or self.feasibility is None:
            raise ValueError(
                "an optimal result carries the schedule, the bracket and the "
                "feasibility report that make it optimal"
            )
        if not self.feasibility.feasible:
            raise ValueError(
                f"an optimal result's schedule must be feasible; {self.solver} "
                f"returned one that breaches {dict(self.feasibility.violations)}"
            )
        allowed = _BRACKET_RTOL * max(1.0, abs(self.bracket.upper))
        if self.bracket.width > allowed:
            raise ValueError(
                f"an optimal result's bracket must be closed; {self.solver} "
                f"returned [{self.bracket.lower!r}, {self.bracket.upper!r}], "
                f"{self.bracket.width!r} wide against {allowed!r}"
            )


@runtime_checkable
class Solver(Protocol):
    name: str

    def solve(self, instance: Instance, budget: Budget) -> Result: ...
