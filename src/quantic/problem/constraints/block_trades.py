"""Dial D4: all-or-nothing block trades.

Block and dark crossing is discrete by construction (spec section 4), and the
resulting structure is purely binary: spec section 6.2 introduces one binary
``b_k`` per block, contributing its fixed quantity to the full-liquidation
constraint.

At the schedule level a block makes its cell indivisible: ``x[asset][bucket]``
is either ``0`` or exactly ``lots``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantic.problem.constraints.base import check_position_length
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class Block:
    """An indivisible quantity offered in one asset-bucket cell."""

    asset: int
    bucket: int
    lots: int

    def __post_init__(self) -> None:
        if self.asset < 0:
            raise ValueError(f"asset index must be non-negative, got {self.asset}")
        if self.bucket < 0:
            raise ValueError(f"bucket index must be non-negative, got {self.bucket}")
        if self.lots < 1:
            raise ValueError(
                f"block lots must be at least 1, got {self.lots}: a zero-size block is "
                "not a trade"
            )


@dataclass(frozen=True, slots=True)
class BlockTrades:
    """Each block's cell trades whole or not at all."""

    blocks: tuple[Block, ...]
    name: str = field(default="block_trades")

    def __post_init__(self) -> None:
        cells = [(b.asset, b.bucket) for b in self.blocks]
        duplicates = sorted({c for c in cells if cells.count(c) > 1})
        if duplicates:
            raise ValueError(
                f"more than one block on cell(s) {duplicates}: two blocks on one cell "
                "impose contradictory all-or-nothing sizes on the same variable"
            )

    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float:
        check_position_length(schedule, initial_lots)
        total = 0.0
        for b in self.blocks:
            if b.asset >= schedule.n_assets:
                raise ValueError(
                    f"block references asset {b.asset} but the schedule has "
                    f"{schedule.n_assets} assets"
                )
            if b.bucket >= schedule.n_buckets:
                raise ValueError(
                    f"block references bucket {b.bucket} but the schedule has "
                    f"{schedule.n_buckets} buckets"
                )
            value = schedule.lots[b.asset][b.bucket]
            total += float(min(value, abs(value - b.lots)))
        return total

    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool:
        return self.violation(schedule, initial_lots) == 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "blocks": [
                {"asset": b.asset, "bucket": b.bucket, "lots": b.lots} for b in self.blocks
            ],
        }
