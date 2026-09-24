"""Dial D2, part 2: a venue minimum quantity.

``x[i,t]`` is either zero or at least ``m_i`` -- a semi-continuous variable,
which is where the big-M binaries come from. Spec section 6.2 keeps the
encoding quadratic by writing ``x = m*y + sum over b of 2^b z_b`` with the
penalty ``sum over b of z_b * (1 - y)``; the naive form is cubic.

None of that is this module's business. Here the constraint is only asked
whether a schedule satisfies it, and by how much.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantic.problem.constraints.base import check_position_length
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class MinParticipation:
    """For every cell, ``x[i,t] == 0`` or ``x[i,t] >= min_lots[i]``."""

    min_lots: tuple[int, ...]
    name: str = field(default="min_participation")

    def __post_init__(self) -> None:
        for i, m in enumerate(self.min_lots):
            if m < 1:
                raise ValueError(
                    f"min_lots[{i}] must be at least 1, got {m}: a minimum of zero is not "
                    "a constraint, and a negative one is meaningless"
                )

    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float:
        check_position_length(schedule, initial_lots)
        if len(self.min_lots) != schedule.n_assets:
            raise ValueError(
                f"schedule has {schedule.n_assets} assets but min_lots has "
                f"{len(self.min_lots)} entries"
            )
        total = 0.0
        for i, row in enumerate(schedule.lots):
            minimum = self.min_lots[i]
            for value in row:
                if 0 < value < minimum:
                    total += minimum - value
        return total

    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool:
        return self.violation(schedule, initial_lots) == 0.0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "min_lots": list(self.min_lots)}
