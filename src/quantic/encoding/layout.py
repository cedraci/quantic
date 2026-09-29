"""How each ``x[i,t]`` is represented, given the constraints on it.

The representation is not a property of the encoding alone. Spec section 6.2's
min-participation formulation, ``x = m*y + sum 2**b z_b``, is a *different
expansion of x*, and a block cell collapses to a single all-or-nothing bit. So
the constraint set has to be read before any bit is named.

The asymmetry that falls out is a result for the section 6.1 encoding study
rather than an implementation detail: under one-hot and domain-wall the values
a venue minimum forbids are simply absent from the table, so **D2 is free**;
under the log encoding it costs an indicator, a residual expansion and a
penalty.

A cell's bound is ``min(max_lots_per_bucket, initial_lots[i])``. Full
liquidation caps the row total at the position, so the tighter bound excludes
no feasible schedule and removes bits from every tier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from quantic.encoding.integer import IntegerEncoding
from quantic.encoding.qubo import LinearForm
from quantic.encoding.registry import RegistryBuilder, VarKey, VarRegistry
from quantic.problem.constraints.block_trades import BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.instance import Instance
from quantic.problem.schedule import Schedule

PLAIN = "plain"
SEMI_CONTINUOUS = "semi_continuous"
BLOCK = "block"


@dataclass(frozen=True, slots=True)
class CellPlan:
    """How one ``(asset, bucket)`` cell becomes bits."""

    asset: int
    bucket: int
    kind: str
    values: tuple[int, ...]
    bit_keys: tuple[VarKey, ...]
    value_form: LinearForm
    indicator_form: LinearForm | None
    validity_forms: tuple[LinearForm, ...]
    validity_products: tuple[tuple[LinearForm, LinearForm], ...]
    encoding: IntegerEncoding
    table_keys: tuple[VarKey, ...]
    indicator_key: VarKey | None
    residual_keys: tuple[VarKey, ...]
    min_lots: int

    @property
    def residual_values(self) -> tuple[int, ...]:
        return tuple(range(max(self.values) - self.min_lots + 1))

    def exact_cost_form(self, costs: Mapping[int, float]) -> LinearForm | None:
        """``f(x)`` as an exact linear form, or ``None`` if this cell has no table."""
        if self.kind == BLOCK:
            biggest = max(self.values)
            return LinearForm.constant_form(float(costs[0])) + LinearForm.of(
                self.bit_keys[0], float(costs[biggest]) - float(costs[0])
            )
        if not self.table_keys:
            return None
        return self.encoding.cost_form(self.table_keys, self.values, costs)

    def encode(self, x: int) -> dict[VarKey, int]:
        if x not in self.values:
            raise ValueError(
                f"{x} is not one of this cell's values {self.values} at "
                f"(asset {self.asset}, bucket {self.bucket})"
            )
        if self.kind == BLOCK:
            return {self.bit_keys[0]: int(x != 0)}
        if self.table_keys:
            return self.encoding.encode_value(self.table_keys, self.values, x)
        if self.indicator_key is None:
            return self.encoding.encode_value(self.bit_keys, self.values, x)
        bits: dict[VarKey, int] = dict.fromkeys(self.bit_keys, 0)
        if x == 0:
            return bits
        bits[self.indicator_key] = 1
        bits.update(
            self.encoding.encode_value(
                self.residual_keys, self.residual_values, x - self.min_lots
            )
        )
        return bits

    def decode(self, bits: Mapping[VarKey, int]) -> int:
        if self.kind == BLOCK:
            return max(self.values) if bits[self.bit_keys[0]] else 0
        if self.table_keys:
            return self.encoding.decode_value(self.table_keys, self.values, bits)
        if self.indicator_key is None:
            return self.encoding.decode_value(self.bit_keys, self.values, bits)
        if not bits[self.indicator_key]:
            if any(bits[k] for k in self.residual_keys):
                raise ValueError(
                    f"cell (asset {self.asset}, bucket {self.bucket}) has residual bits "
                    "set while its indicator is clear; spec section 6.2's penalty "
                    "forbids this state"
                )
            return 0
        return self.min_lots + self.encoding.decode_value(
            self.residual_keys, self.residual_values, bits
        )


@dataclass(frozen=True, slots=True)
class Layout:
    """Every cell's plan, plus the registry every bit was named into."""

    cells: tuple[CellPlan, ...]
    registry: VarRegistry
    encoding_name: str
    n_assets: int
    n_buckets: int

    def cell(self, asset: int, bucket: int) -> CellPlan:
        return self.cells[asset * self.n_buckets + bucket]


def _cell_values(instance: Instance, i: int, minimum: int) -> tuple[int, ...]:
    cap = min(instance.max_lots_per_bucket, instance.initial_lots[i])
    if minimum <= 1:
        return tuple(range(cap + 1))
    return (0, *range(minimum, cap + 1))


def _constraint_view(instance: Instance):
    """Read the constraint set once, dispatching on type rather than on a name."""
    minimums: dict[int, int] = {}
    blocks: dict[tuple[int, int], int] = {}
    has_cardinality = False
    has_min_participation = False
    for c in instance.constraints:
        if isinstance(c, FullLiquidation):
            continue
        if isinstance(c, Cardinality):
            has_cardinality = True
        elif isinstance(c, MinParticipation):
            minimums = dict(enumerate(c.min_lots))
            has_min_participation = True
        elif isinstance(c, BlockTrades):
            blocks = {(b.asset, b.bucket): b.lots for b in c.blocks}
        else:
            raise NotImplementedError(
                f"the layout planner does not know constraint {c.name!r} "
                f"({type(c).__name__}). Dispatch is on type, never on the name, so a "
                "new constraint must be taught here rather than inherited by accident"
            )
    return minimums, blocks, has_cardinality, has_min_participation


def plan_layout(instance: Instance, encoding: IntegerEncoding) -> Layout:
    """Name every bit and decide how each cell's value is built from them."""
    minimums, blocks, has_cardinality, has_min_participation = _constraint_view(instance)
    builder = RegistryBuilder()
    cells: list[CellPlan] = []

    for i in range(instance.n_assets):
        for t in range(instance.n_buckets):
            minimum = int(minimums.get(i, 1))
            block_lots = blocks.get((i, t))

            if block_lots is not None:
                key: VarKey = ("block", i, t)
                builder.add(key)
                cells.append(
                    CellPlan(
                        asset=i, bucket=t, kind=BLOCK, values=(0, int(block_lots)),
                        bit_keys=(key,),
                        value_form=LinearForm.of(key, float(block_lots)),
                        indicator_form=LinearForm.of(key, 1.0),
                        validity_forms=(), validity_products=(),
                        encoding=encoding, table_keys=(), indicator_key=None,
                        residual_keys=(), min_lots=minimum,
                    )
                )
                continue

            values = _cell_values(instance, i, minimum)
            kind = SEMI_CONTINUOUS if minimum > 1 else PLAIN

            if encoding.supports_exact_table or len(values) <= 1:
                keys = encoding.bit_keys(("x", i, t), values)
                for k in keys:
                    builder.add(k)
                if encoding.supports_exact_table:
                    indicator_form = encoding.indicator_form(keys, values)
                else:
                    # Reached only because len(values) <= 1 -- the log
                    # encoding's own indicator_form is None here, since it has
                    # no table to compute 1[x > 0] from. But a cell that can
                    # only take the value 0 never trades, so its contribution
                    # to sum_i y[i,t] is identically zero, not unknown.
                    indicator_form = LinearForm.constant_form(0.0)
                cells.append(
                    CellPlan(
                        asset=i, bucket=t, kind=kind, values=values, bit_keys=keys,
                        value_form=encoding.value_form(keys, values),
                        indicator_form=indicator_form,
                        validity_forms=encoding.validity_forms(keys, values),
                        validity_products=encoding.validity_products(keys, values),
                        encoding=encoding, table_keys=keys, indicator_key=None,
                        residual_keys=(), min_lots=minimum,
                    )
                )
                continue

            if kind == PLAIN and not has_cardinality:
                keys = encoding.bit_keys(("x", i, t), values)
                for k in keys:
                    builder.add(k)
                cells.append(
                    CellPlan(
                        asset=i, bucket=t, kind=kind, values=values, bit_keys=keys,
                        value_form=encoding.value_form(keys, values),
                        indicator_form=None, validity_forms=(), validity_products=(),
                        encoding=encoding, table_keys=(), indicator_key=None,
                        residual_keys=(), min_lots=minimum,
                    )
                )
                continue

            # Spec section 6.2: x = m*y + residual, with the residual forced to
            # zero when y is clear. The penalty itself is Task 7's; the layout
            # only has to name the bits and build the value.
            #
            # A PLAIN cell (no real venue minimum) falls through here too when
            # cardinality is present: the log encoding has no other way to
            # express 1[x > 0], so effective_minimum forces y to double as that
            # indicator even with no minimum to enforce. With
            # effective_minimum == 1 this is x = 1*y + residual over
            # [0, cap - 1], whose value set (0..cap) is exactly the PLAIN set --
            # x == 0 is y == 0; x == v > 0 is y == 1, residual == v - 1.
            effective_minimum = max(minimum, 1)
            y_key: VarKey = ("y", i, t)
            builder.add(y_key)
            residual_values = tuple(range(max(values) - effective_minimum + 1))
            residual_keys = encoding.bit_keys(("x", i, t), residual_values)
            for k in residual_keys:
                builder.add(k)
            value_form = LinearForm.of(
                y_key, float(effective_minimum)
            ) + encoding.value_form(residual_keys, residual_values)
            cells.append(
                CellPlan(
                    asset=i, bucket=t, kind=SEMI_CONTINUOUS, values=values,
                    bit_keys=(y_key, *residual_keys), value_form=value_form,
                    indicator_form=LinearForm.of(y_key, 1.0),
                    validity_forms=(), validity_products=(),
                    encoding=encoding, table_keys=(), indicator_key=y_key,
                    residual_keys=residual_keys, min_lots=effective_minimum,
                )
            )

    if has_cardinality and not has_min_participation:
        raise ValueError(
            "cardinality needs a 1[x > 0] indicator on every cell, and only "
            "MinParticipation supplies one. The generator only ever emits Cardinality "
            "alongside MinParticipation; this pairing is unsupported"
        )

    return Layout(
        cells=tuple(cells),
        registry=builder.freeze(),
        encoding_name=encoding.name,
        n_assets=instance.n_assets,
        n_buckets=instance.n_buckets,
    )


def encode_schedule(
    layout: Layout, schedule: Schedule, registry: VarRegistry
) -> np.ndarray:
    """The canonical bit vector for ``schedule``, in ``registry`` order.

    Bits the layout does not name -- cardinality slack, the CVaR grids -- are
    left at zero here; their values depend on the penalty and gadget
    configuration and are filled in by the callers that own them.
    """
    if (schedule.n_assets, schedule.n_buckets) != (layout.n_assets, layout.n_buckets):
        raise ValueError(
            f"schedule shape ({schedule.n_assets}, {schedule.n_buckets}) does not match "
            f"the layout ({layout.n_assets}, {layout.n_buckets})"
        )
    bits: dict[VarKey, int] = {}
    for plan in layout.cells:
        bits.update(plan.encode(int(schedule.lots[plan.asset][plan.bucket])))
    return np.array([bits.get(k, 0) for k in registry.keys], dtype=np.int64)
