"""Hard constraints as penalties, and the magnitudes that make them bite.

Spec section 6.3 is blunt about why this is a first-class module rather than a
helper. Too small a penalty and a constraint-violating sample scores as cheap
and gets reported as a win. Too large and the penalty swamps the objective's
dynamic range: on real hardware the objective differences drop below the
analogue precision floor, and the annealer solves a feasibility problem while
being reported as an optimiser.

The derivation here is structural -- section 6.3's first strategy. The
adaptive strategy needs a reference sampler to measure a feasibility rate
against, and none exists until M4.

Two constraints carry no penalty at all. A block cell is one bit whose only
two values are the two the constraint allows, and under a table encoding the
values a venue minimum forbids are simply absent. Both are enforced by the
layout, so there is nothing to tune -- which is the section 6.1 encoding
result showing up as an absent term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.encoding.integer import BinaryEncoding, bounded_coefficients
from quantic.encoding.layout import Layout
from quantic.encoding.qubo import LinearForm, Qubo, QuboBuilder
from quantic.encoding.registry import VarKey, VarRegistry
from quantic.problem.constraints.block_trades import BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.instance import Instance
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class PenaltyStrengths:
    """One magnitude per penalty family, and how they were arrived at."""

    validity: float
    full_liquidation: float
    min_participation: float
    cardinality: float
    cvar: float
    safety_factor: float
    strategy: str = "derived_bounds"

    def describe(self) -> dict:
        """JSON-safe, for the results row -- section 6.3 requires it recorded."""
        return {
            "strategy": self.strategy,
            "safety_factor": self.safety_factor,
            "validity": self.validity,
            "full_liquidation": self.full_liquidation,
            "min_participation": self.min_participation,
            "cardinality": self.cardinality,
            "cvar": self.cvar,
        }


def _cardinality_of(instance: Instance) -> Cardinality | None:
    for c in instance.constraints:
        if isinstance(c, Cardinality):
            return c
    return None


def penalty_aux_keys(instance: Instance, layout: Layout) -> tuple[VarKey, ...]:
    """The cardinality slack bits, in registry order.

    ``sum_i y[i,t] + s_t == k`` with ``s_t`` in ``[0, k]``: the slack absorbs
    the shortfall below the cap, never a surplus above it.
    """
    cardinality = _cardinality_of(instance)
    if cardinality is None:
        return ()
    coefficients = bounded_coefficients(cardinality.k)
    return tuple(
        ("card", t, j)
        for t in range(layout.n_buckets)
        for j in range(len(coefficients))
    )


def slack_bits(
    instance: Instance, layout: Layout, schedule: Schedule
) -> dict[VarKey, int]:
    """Cardinality slack for ``schedule``, so a feasible point pays exactly zero."""
    cardinality = _cardinality_of(instance)
    if cardinality is None:
        return {}

    coefficients = bounded_coefficients(cardinality.k)
    encoder = BinaryEncoding()
    values = tuple(range(cardinality.k + 1))
    bits: dict[VarKey, int] = {}
    for t in range(layout.n_buckets):
        active = schedule.active_names(t)
        shortfall = cardinality.k - active
        keys = tuple(("card", t, j) for j in range(len(coefficients)))
        if shortfall < 0:
            bits.update(dict.fromkeys(keys, 0))
            continue
        bits.update(encoder.encode_value(keys, values, shortfall))
    return bits


def expand_penalties(
    instance: Instance,
    layout: Layout,
    registry: VarRegistry,
    strengths: PenaltyStrengths,
) -> Qubo:
    """Every hard constraint as a quadratic penalty over ``registry``."""
    builder = QuboBuilder(registry)

    for plan in layout.cells:
        for form in plan.validity_forms:
            builder.add_square(form, strengths.validity)
        for lhs, rhs in plan.validity_products:
            builder.add_product(lhs, rhs, strengths.validity)

    for c in instance.constraints:
        if isinstance(c, FullLiquidation):
            for i in range(layout.n_assets):
                form = LinearForm.constant_form(-float(instance.initial_lots[i]))
                for t in range(layout.n_buckets):
                    form = form + layout.cell(i, t).value_form
                builder.add_square(form, strengths.full_liquidation)

        elif isinstance(c, MinParticipation):
            for plan in layout.cells:
                if plan.indicator_key is None:
                    continue  # enforced by the layout under a table encoding
                complement = LinearForm.constant_form(1.0) + LinearForm.of(
                    plan.indicator_key, -1.0
                )
                for key in plan.residual_keys:
                    builder.add_product(
                        LinearForm.of(key, 1.0), complement, strengths.min_participation
                    )

        elif isinstance(c, Cardinality):
            coefficients = bounded_coefficients(c.k)
            for t in range(layout.n_buckets):
                form = LinearForm.constant_form(-float(c.k))
                for i in range(layout.n_assets):
                    indicator = layout.cell(i, t).indicator_form
                    if indicator is None:
                        raise ValueError(
                            f"cell (asset {i}, bucket {t}) has no indicator, so "
                            "cardinality cannot be expressed. plan_layout should have "
                            "refused this instance"
                        )
                    form = form + indicator
                for j, coef in enumerate(coefficients):
                    form = form + LinearForm.of(("card", t, j), float(coef))
                builder.add_square(form, strengths.cardinality)

        elif isinstance(c, BlockTrades):
            continue  # enforced by the layout: the cell is one bit

        else:
            raise NotImplementedError(
                f"expand_penalties does not know constraint {c.name!r} "
                f"({type(c).__name__}). Dispatch is on type, never on the name"
            )

    return builder.build()


def max_single_cell_gain(objective: Qubo, layout: Layout) -> float:
    """An upper bound on the objective change any one cell's bits can produce.

    Flipping a subset of one cell's bits moves the objective by at most the
    sum, over those bits, of the absolute coefficients in their rows and
    columns. Rigorous, cheap, and tighter than the whole objective's range --
    which is the difference between a penalty that bites and one that swamps
    the dynamic range section 6.4 measures.
    """
    q = np.abs(np.asarray(objective.Q, dtype=float))
    worst = 0.0
    for plan in layout.cells:
        total = 0.0
        for key in plan.bit_keys:
            k = objective.registry.index(key)
            total += float(q[k, k] + q[k, :].sum() + q[:, k].sum())
        worst = max(worst, total)
    return worst


def derive_penalties(
    objective: Qubo, layout: Layout, *, safety_factor: float = 2.0
) -> PenaltyStrengths:
    """Section 6.3's first strategy: a penalty larger than any gain it could buy.

    Every penalty residual in this milestone is an integer-valued square, so a
    one-unit violation costs at least ``1`` and the strength need only exceed
    the gain. The floor of ``1.0`` keeps the constraints binding when the
    objective is identically zero -- which happens whenever ``lam`` is zero and
    the dials are off.
    """
    if safety_factor <= 1.0:
        raise ValueError(
            f"safety_factor must exceed 1, got {safety_factor}: at or below 1 the "
            "penalty only ties the gain it is meant to beat"
        )
    strength = max(safety_factor * max_single_cell_gain(objective, layout), 1.0)
    needs_min_participation = any(c.indicator_key is not None for c in layout.cells)
    return PenaltyStrengths(
        validity=strength,
        full_liquidation=strength,
        min_participation=strength if needs_min_participation else 0.0,
        cardinality=strength,
        cvar=strength,
        safety_factor=safety_factor,
    )
