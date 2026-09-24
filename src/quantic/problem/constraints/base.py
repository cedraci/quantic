"""The constraint interface.

A constraint is a frozen descriptor that carries its parameters **and** the
problem-level logic for deciding whether a schedule satisfies it. It does not
emit QUBO terms: spec section 12 forbids ``problem/`` importing ``encoding/``,
and each constraint has a genuinely different quadratisation anyway (spec
section 6.2's min-participation trick is not derivable from a generic
interface).

Feasibility rate is a headline metric (spec section 6.5) and must be
computable with no encoder in sight -- including for M4's CP-SAT and MILP
solvers, which express constraints natively and never build a QUBO. That is
why the logic lives here rather than in the encoder.

``violation`` returns a **magnitude**, not a boolean, because M3's penalty
derivation (spec section 6.3) needs the size of a one-unit violation and M4's
repair strategies need to know how far off a sample is.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quantic.problem.schedule import Schedule


@runtime_checkable
class Constraint(Protocol):
    name: str

    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float:
        """Non-negative magnitude of the breach; exactly ``0.0`` when satisfied."""
        ...

    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool: ...

    def describe(self) -> dict[str, Any]:
        """JSON-safe parameters, for the results row and the instance hash."""
        ...


def check_position_length(schedule: Schedule, initial_lots: tuple[int, ...]) -> None:
    """Shared guard: the position vector must match the schedule's asset count."""
    if len(initial_lots) != schedule.n_assets:
        raise ValueError(
            f"schedule has {schedule.n_assets} assets but initial_lots has "
            f"{len(initial_lots)} entries"
        )
