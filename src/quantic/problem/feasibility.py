"""The single answer to "is this schedule feasible".

Spec section 6.5 makes feasibility rate a headline metric reported alongside
cost, and names scoring an infeasible quantum solution against a feasible
classical one as the error that invalidates most published comparisons in this
area.

That metric is only trustworthy if there is exactly one implementation of the
test. M3's decoder, M4's CP-SAT and MILP solvers -- which express constraints
natively and never build a QUBO -- and M6's metrics all call this function.
Three independent copies is how a headline number silently disagrees with
itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from quantic.problem.instance import Instance
from quantic.problem.schedule import Schedule


@dataclass(frozen=True)
class FeasibilityReport:
    """Per-constraint breach magnitudes and the verdict they imply."""

    feasible: bool
    violations: Mapping[str, float]
    total_violation: float


def classify(instance: Instance, schedule: Schedule) -> FeasibilityReport:
    """Evaluate every constraint on ``instance`` against ``schedule``.

    Every constraint appears in ``violations``, including the satisfied ones
    at ``0.0``: a report that lists only breaches cannot distinguish a clean
    schedule from one whose constraints were never checked.
    """
    if (schedule.n_assets, schedule.n_buckets) != (instance.n_assets, instance.n_buckets):
        raise ValueError(
            f"schedule shape ({schedule.n_assets}, {schedule.n_buckets}) does not match "
            f"the instance ({instance.n_assets}, {instance.n_buckets})"
        )

    names = [c.name for c in instance.constraints]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate constraint name(s) {duplicates} on instance "
            f"{instance.instance_id}: they would overwrite each other in the report"
        )

    violations = {
        c.name: float(c.violation(schedule, instance.initial_lots))
        for c in instance.constraints
    }
    total = float(sum(violations.values()))
    return FeasibilityReport(
        feasible=total == 0.0,
        violations=MappingProxyType(violations),
        total_violation=total,
    )
