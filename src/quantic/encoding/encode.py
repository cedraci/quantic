"""One instance, one encoding, one QUBO -- and the reference it must equal.

Every stage below names its own bits: the layout names the lot cells, the
penalty module names cardinality slack, the CVaR gadget names its three grids.
They must all land in **one** registry, because column ``k`` has to mean the
same bit in the objective, the penalties and the gadget alike. This module
assembles that registry and then rebuilds every piece against it.

The strength derivation needs two passes. The gadget's *keys* depend only on
the instance and the bit budget, never on a strength, so the gadget is built
once at strength zero to fix the registry. The hard constraints' strengths are
derived from the objective expanded against that provisional gadget; the CVaR
strength is derived by the gadget from its own trade-off, which the objective
does not enter. Then everything is rebuilt at the derived strengths. Both
passes are pure, so the result is deterministic.

What this leaves open: the objective the hard strengths are derived from has
the CVaR penalty at zero, so no lot bit is coupled to the risk term in it.
With D3 on, the hard strengths therefore bound the gain in spread and impact
and not the gain in risk.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantic.encoding.cvar import CVaRGadget, build_cvar_gadget
from quantic.encoding.expand import QuboBreakdown, expand_objective
from quantic.encoding.integer import IntegerEncoding
from quantic.encoding.layout import Layout, encode_schedule, plan_layout
from quantic.encoding.penalty import (
    PenaltyStrengths,
    derive_penalties,
    expand_penalties,
    penalty_aux_keys,
    slack_bits,
)
from quantic.encoding.qubo import Qubo
from quantic.encoding.registry import VarRegistry
from quantic.problem.instance import Instance
from quantic.problem.objective import ObjectiveBreakdown, evaluate
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, eq=False)
class EncodedInstance:
    """Everything the gate, the diagnostics and M3b's solvers need."""

    instance: Instance
    layout: Layout
    registry: VarRegistry
    breakdown: QuboBreakdown
    penalties: Qubo
    strengths: PenaltyStrengths
    gadget: CVaRGadget | None

    @property
    def qubo(self) -> Qubo:
        return self.breakdown.total + self.penalties

    @property
    def n_vars(self) -> int:
        return len(self.registry)

    def describe(self) -> dict[str, Any]:
        """JSON-safe, for the results row."""
        approximand = self.breakdown.approximand
        return {
            "encoding": self.layout.encoding_name,
            "n_vars": self.n_vars,
            "penalties": self.strengths.describe(),
            "approximand": None if approximand is None else approximand.describe(),
            "cvar_resolution": None if self.gadget is None else self.gadget.resolution,
            "cvar_zeta_bits": None if self.gadget is None else self.gadget.zeta.n_bits,
        }


def _assemble(instance: Instance, layout: Layout, gadget: CVaRGadget | None) -> VarRegistry:
    keys = layout.registry.keys + penalty_aux_keys(instance, layout)
    if gadget is not None:
        keys = keys + gadget.keys
    return VarRegistry(keys=keys)


def encode(
    instance: Instance,
    encoding: IntegerEncoding,
    *,
    n_zeta_bits: int = 8,
    safety_factor: float = 2.0,
) -> EncodedInstance:
    """Turn ``instance`` into a QUBO under ``encoding``."""
    layout = plan_layout(instance, encoding)
    needs_gadget = isinstance(instance.risk, CVaRRisk)

    provisional = (
        build_cvar_gadget(instance, layout, n_zeta_bits=n_zeta_bits, penalty=0.0)
        if needs_gadget
        else None
    )
    registry = _assemble(instance, layout, provisional)
    probe = expand_objective(instance, layout, registry, gadget=provisional)
    strengths = derive_penalties(
        probe.total, layout, safety_factor=safety_factor,
        cvar=(
            provisional.derived_penalty(safety_factor=safety_factor)
            if needs_gadget
            else 0.0
        ),
    )

    gadget = (
        dataclasses.replace(provisional, penalty=strengths.cvar)
        if needs_gadget
        else None
    )
    breakdown = expand_objective(instance, layout, registry, gadget=gadget)
    penalties = expand_penalties(instance, layout, registry, strengths)
    return EncodedInstance(
        instance=instance, layout=layout, registry=registry, breakdown=breakdown,
        penalties=penalties, strengths=strengths, gadget=gadget,
    )


def encode_solution(
    encoded: EncodedInstance, schedule: Schedule, *, zeta: float | None = None
) -> np.ndarray:
    """The canonical bit vector for ``schedule``, auxiliaries included."""
    if encoded.gadget is None and zeta is not None:
        raise ValueError(
            "zeta was supplied but this instance carries no CVaR gadget, so it has "
            "no zeta to encode"
        )
    if encoded.gadget is not None and zeta is None:
        raise ValueError(
            "this instance needs an explicit zeta. Rockafellar-Uryasev treats it as a "
            "decision variable, and the encoding pins it to a grid, so the call site "
            "must say which zeta it is encoding. Snap "
            "problem.objective.optimal_zeta onto encoded.gadget.zeta"
        )

    vector = encode_schedule(encoded.layout, schedule, encoded.registry)
    for key, value in slack_bits(encoded.instance, encoded.layout, schedule).items():
        vector[encoded.registry.index(key)] = value

    if encoded.gadget is not None:
        snapped = encoded.gadget.zeta.snap(float(zeta))
        for key, value in encoded.gadget.zeta.encode(snapped).items():
            vector[encoded.registry.index(key)] = value
        losses = encoded.gadget.losses_from(
            encoded.instance, encoded.layout, schedule
        )
        for key, value in encoded.gadget.optimal_aux(losses, snapped).items():
            vector[encoded.registry.index(key)] = value
    return vector


def reference_breakdown(
    encoded: EncodedInstance, schedule: Schedule, *, zeta: float | None = None
) -> ObjectiveBreakdown:
    """What the QUBO's objective block is expected to equal.

    ``problem.objective.evaluate`` stays the only evaluator. When an
    approximand was emitted, one field of its output is replaced -- never
    recomputed by a second implementation, which is how the two would drift.
    """
    base = evaluate(encoded.instance, schedule, zeta=zeta)
    approximand = encoded.breakdown.approximand
    if approximand is None:
        return base
    return dataclasses.replace(
        base, temporary_impact_cost=approximand.temporary_cost(schedule)
    )
