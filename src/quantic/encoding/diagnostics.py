"""The two numbers no encoded instance may be reported without.

**Dynamic range** (spec section 6.4). D-Wave couplers carry roughly four to
five bits of effective precision. The finest distinction the hardware must
resolve is the smallest non-zero coefficient in the matrix, so the ratio of
the largest to that is what says whether the objective's structure survives
the analogue floor. Above it the annealer is solving something coarser than
the problem posed -- while being reported as an optimiser. Section 6.4: "No
result is interpretable without it."

**The variable budget** (spec section 3.2). Dense embedding runs out near 230
variables. With dial D3 on, the auxiliaries outnumber the decision variables
by an order of magnitude, so the budget is what says which cells of the study
can reach real hardware at all -- a question otherwise discovered at M5, on
the QPU, with the month's quota already spent.

The budget counts **keys, never matrices**. At T3 under one-hot the layout
names roughly 5,760 bits, and one dense term of that size is 265 MB. Counting
is free and is all the budget needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quantic.encoding.cvar import build_cvar_gadget
from quantic.encoding.integer import ENCODINGS, IntegerEncoding
from quantic.encoding.layout import plan_layout
from quantic.encoding.penalty import penalty_aux_keys
from quantic.encoding.qubo import Qubo
from quantic.problem.instance import Instance
from quantic.problem.risk import CVaRRisk

# Spec section 3.2: dense minor-embedding on Advantage2 fails above roughly
# this many logical variables.
DENSE_EMBEDDING_CEILING = 230


def dynamic_range_ratio(qubo: Qubo) -> float:
    """``max|coefficient| / min non-zero |coefficient|``, or ``0.0`` if empty.

    The constant is excluded: it shifts every energy equally, so no coupler
    has to resolve it.
    """
    magnitudes = np.abs(np.asarray(qubo.Q, dtype=float))
    non_zero = magnitudes[magnitudes != 0.0]
    if non_zero.size == 0:
        return 0.0
    return float(non_zero.max() / non_zero.min())


def variable_count(
    instance: Instance, encoding: IntegerEncoding, *, n_zeta_bits: int = 8
) -> int:
    """How many binaries ``encode`` would name, without building anything.

    The penalty strength does not change which keys exist, so the gadget is
    counted at a strength of zero.
    """
    layout = plan_layout(instance, encoding)
    total = len(layout.registry) + len(penalty_aux_keys(instance, layout))
    if isinstance(instance.risk, CVaRRisk):
        gadget = build_cvar_gadget(
            instance, layout, n_zeta_bits=n_zeta_bits, penalty=0.0
        )
        total += len(gadget.keys)
    return total


@dataclass(frozen=True, slots=True)
class BudgetRow:
    """One cell of the tier x dials x encoding table."""

    tier: str
    dials: str
    encoding: str
    n_vars: int
    fits_dense_embedding: bool

    def describe(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "dials": self.dials,
            "encoding": self.encoding,
            "n_vars": self.n_vars,
            "fits_dense_embedding": self.fits_dense_embedding,
        }


def variable_budget(
    instances,
    *,
    encodings: tuple[IntegerEncoding, ...] = ENCODINGS,
    n_zeta_bits: int = 8,
) -> tuple[BudgetRow, ...]:
    """The table spec section 13 of the M3a design promises."""
    rows: list[BudgetRow] = []
    for instance in instances:
        for encoding in encodings:
            n_vars = variable_count(instance, encoding, n_zeta_bits=n_zeta_bits)
            rows.append(
                BudgetRow(
                    tier=instance.tier,
                    dials=instance.dials.label,
                    encoding=encoding.name,
                    n_vars=n_vars,
                    fits_dense_embedding=n_vars <= DENSE_EMBEDDING_CEILING,
                )
            )
    return tuple(rows)
