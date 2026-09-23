"""Cross-validate L3 replay against independently sourced L2 snapshots.

Spec section 10.3. L2 and L3 are distinct feed products, so agreement between a
replayed book and a published snapshot is genuine evidence of correctness
rather than a tautology.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from quantic.core.types import BookSnapshot
from quantic.micro.book_reconstruct import snapshots_at


@dataclass(frozen=True)
class ReconstructionMismatch:
    symbol: str
    ts_ns: int
    side: str
    level: int
    field: str
    expected: float
    actual: float


def _observed_levels(snap: BookSnapshot, side: str) -> list[tuple[float, int]]:
    levels = snap.bids if side == "buy" else snap.asks
    return [(lvl.price, lvl.size) for lvl in levels]


def compare_to_l2(
    l3: pl.DataFrame,
    l2: pl.DataFrame,
    *,
    levels: int,
    symbols: Sequence[str] | None = None,
    price_tol: float = 1e-9,
) -> list[ReconstructionMismatch]:
    """Replay ``l3`` and report every disagreement with ``l2``.

    An empty result means the replay reproduces every published snapshot
    exactly, to ``price_tol`` on prices and bit-for-bit on sizes.
    """
    present = sorted(l2["symbol"].unique().to_list())
    if symbols is not None:
        missing = sorted(set(symbols) - set(present))
        if missing:
            raise ValueError(f"requested symbols absent from l2: {missing}")
        wanted = sorted(symbols)
    else:
        wanted = present

    if not wanted:
        raise ValueError("no symbols to compare: symbols=[] or l2 has no rows")

    mismatches: list[ReconstructionMismatch] = []

    for symbol in wanted:
        sym_l2 = l2.filter(pl.col("symbol") == symbol)
        sym_l3 = l3.filter(pl.col("symbol") == symbol)

        ts_list = sorted(sym_l2["ts_ns"].unique().to_list())
        snaps = snapshots_at(sym_l3, ts_list, levels=levels)

        for ts_ns, snap in zip(ts_list, snaps, strict=True):
            at_ts = sym_l2.filter(pl.col("ts_ns") == ts_ns)
            for side in ("buy", "sell"):
                expected = (
                    at_ts.filter(pl.col("side") == side)
                    .sort("level")
                    .select("px", "size")
                    .rows()
                )
                actual = _observed_levels(snap, side)

                if len(expected) != len(actual):
                    mismatches.append(
                        ReconstructionMismatch(
                            symbol, ts_ns, side, -1, "depth", len(expected), len(actual)
                        )
                    )
                    continue

                for level, ((exp_px, exp_sz), (act_px, act_sz)) in enumerate(
                    zip(expected, actual, strict=True)
                ):
                    if abs(exp_px - act_px) > price_tol:
                        mismatches.append(
                            ReconstructionMismatch(
                                symbol, ts_ns, side, level, "px", exp_px, act_px
                            )
                        )
                    if exp_sz != act_sz:
                        mismatches.append(
                            ReconstructionMismatch(
                                symbol, ts_ns, side, level, "size", exp_sz, act_sz
                            )
                        )

    return mismatches
