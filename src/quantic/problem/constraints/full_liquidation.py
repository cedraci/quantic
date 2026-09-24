"""Spec section 5.3: every position is fully liquidated by the horizon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantic.problem.constraints.base import check_position_length
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class FullLiquidation:
    """``sum over t of x[i,t] == X[i]`` for every asset ``i``.

    Present under every dial combination. It is what makes the problem a
    liquidation rather than an unconstrained trade-off.
    """

    name: str = field(default="full_liquidation")

    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float:
        check_position_length(schedule, initial_lots)
        return float(
            sum(abs(schedule.sold(i) - initial_lots[i]) for i in range(schedule.n_assets))
        )

    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool:
        return self.violation(schedule, initial_lots) == 0.0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}
