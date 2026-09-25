"""Bits back to a schedule, and the verdict spec section 6.5 demands.

Section 6.5 names scoring an infeasible quantum solution against a feasible
classical one as the error that invalidates most published comparisons in this
area. Every sample therefore arrives carrying its classification, and the line
between the three classes is structural rather than a matter of degree:

* **rejected** -- the bits do not describe a schedule at all. Two one-hot bits
  in a cell, a hole in a domain wall, residual bits under a clear indicator.
  There is nothing to score and nothing to repair.
* **repairable** -- the bits decode cleanly but the schedule breaches a
  constraint. M4's named repair strategies act here.
* **feasible** -- ``problem.feasibility.classify`` reports no violation.

Repair itself is not here. Section 6.5 requires it to be a named strategy
recorded on the results row, and it only means anything against samples a
solver returned, which is M4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.encoding.encode import EncodedInstance
from quantic.problem.feasibility import FeasibilityReport, classify
from quantic.problem.schedule import Schedule

FEASIBLE = "feasible"
REPAIRABLE = "repairable"
REJECTED = "rejected"


@dataclass(frozen=True)
class DecodedSample:
    """One returned bitstring, read back and judged."""

    schedule: Schedule | None
    zeta: float | None
    classification: str
    report: FeasibilityReport | None
    reason: str | None = None


def decode(encoded: EncodedInstance, bits) -> DecodedSample:
    """Read ``bits`` back into a schedule and classify it."""
    vector = np.asarray(bits)
    if vector.shape != (encoded.n_vars,):
        raise ValueError(
            f"decode needs {encoded.n_vars} variables, got a vector of shape "
            f"{vector.shape}"
        )
    by_key = {key: int(vector[i]) for i, key in enumerate(encoded.registry.keys)}

    grid = [[0] * encoded.layout.n_buckets for _ in range(encoded.layout.n_assets)]
    for plan in encoded.layout.cells:
        try:
            grid[plan.asset][plan.bucket] = plan.decode(by_key)
        except ValueError as exc:
            return DecodedSample(
                schedule=None, zeta=None, classification=REJECTED,
                report=None, reason=str(exc),
            )

    schedule = Schedule(lots=tuple(tuple(row) for row in grid))
    zeta = None
    if encoded.gadget is not None:
        zeta = encoded.gadget.zeta.decode(by_key)

    report = classify(encoded.instance, schedule)
    return DecodedSample(
        schedule=schedule,
        zeta=zeta,
        classification=FEASIBLE if report.feasible else REPAIRABLE,
        report=report,
    )
