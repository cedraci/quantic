"""Non-parametric execution cost: walk the observed book.

This is the reference the fitted parametric models are checked against. It
takes a book rather than ``ImpactParams``, so it deliberately does not
implement the ``ImpactModel`` protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from quantic.core.types import BookSnapshot, EmptyBookError, Side


class InsufficientDepthError(ValueError):
    """Raised when the visible book cannot absorb the requested size."""


@dataclass(frozen=True, slots=True)
class WalkResult:
    shares: int
    avg_price: float
    notional: float
    levels_consumed: int
    fractional_cost: float


def walk(book: BookSnapshot, side: Side, shares: int) -> WalkResult:
    """Cost of executing ``shares`` immediately against the resting book."""
    if shares <= 0:
        raise ValueError(f"shares must be positive, got {shares}")

    mid = book.mid
    if mid is None:
        raise EmptyBookError(f"one-sided book for {book.symbol} at ts_ns={book.ts_ns}")

    levels = book.asks if side is Side.BUY else book.bids
    available = sum(lvl.size for lvl in levels)
    if available < shares:
        raise InsufficientDepthError(
            f"{book.symbol}: requested {shares} shares but only {available} visible "
            f"on the {side.value} side at ts_ns={book.ts_ns}"
        )

    remaining = shares
    notional = 0.0
    consumed = 0
    for lvl in levels:
        if remaining <= 0:
            break
        taken = min(lvl.size, remaining)
        notional += taken * lvl.price
        remaining -= taken
        consumed += 1

    avg_price = notional / shares
    fractional_cost = (avg_price - mid) / mid if side is Side.BUY else (mid - avg_price) / mid
    return WalkResult(
        shares=shares,
        avg_price=avg_price,
        notional=notional,
        levels_consumed=consumed,
        fractional_cost=fractional_cost,
    )


def empirical_impact_curve(
    book: BookSnapshot, side: Side, quantities: Sequence[int]
) -> pl.DataFrame:
    """Cost curve over ``quantities``; sizes exceeding visible depth are skipped."""
    levels = book.asks if side is Side.BUY else book.bids
    available = sum(lvl.size for lvl in levels)

    rows = []
    for q in quantities:
        if q <= 0 or q > available:
            continue
        result = walk(book, side, q)
        rows.append(
            {
                "shares": q,
                "participation_of_depth": q / available,
                "fractional_cost": result.fractional_cost,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "shares": pl.Int64,
            "participation_of_depth": pl.Float64,
            "fractional_cost": pl.Float64,
        },
    )
