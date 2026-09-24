"""Dial D2, part 1: touch at most ``k`` names per bucket.

Standard desk practice, and combinatorial with no convex relaxation -- the
semi-continuous structure forces big-M binaries in any MIQP formulation (spec
section 4). Spec section 6.2 encodes it with a binary slack:
``(sum over i of y[i,t] + sum over j of 2^j s_j - k)^2``.

Paired with :class:`MinParticipation` under one dial, because cardinality
without a minimum lot size is not meaningful on a real desk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantic.problem.constraints.base import check_position_length
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class Cardinality:
    """At most ``k`` assets may have ``x[i,t] > 0`` in any single bucket ``t``."""

    k: int
    name: str = field(default="cardinality")

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(
                f"k must be at least 1, got {self.k}: a cardinality of zero forbids all "
                "trading and makes full liquidation impossible"
            )

    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float:
        check_position_length(schedule, initial_lots)
        return float(
            sum(
                max(0, schedule.active_names(t) - self.k)
                for t in range(schedule.n_buckets)
            )
        )

    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool:
        return self.violation(schedule, initial_lots) == 0.0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "k": self.k}
