"""Compare an L3 replay against published L2 snapshots.

Spec section 10.3 frames this as cross-validation against an independent
source: L2 and L3 are distinct feed products, so agreement between a replayed
book and a published snapshot is evidence rather than a tautology.

**That framing holds only for real data, and real data has not arrived.** On
the synthetic bundles this is currently run against, both sides originate in
the same `data.synth.build_l3_and_l2` call. What the gate demonstrates there
is that three independently-written order book implementations agree -- the
generator's `_GeneratorBook`, this comparison, and `micro.book_reconstruct` --
which is a genuinely strong property and has caught four injected
reconstructor bugs. But three implementations agreeing about a wrong `replace`
semantic is three agreements, not three checks, and a shared misconception
originating in the generator is invisible to all of them.

`tests/micro/test_reconstruction_gate.py::test_l3_l2_holdout_against_real_data`
is the named holdout that converts this into true cross-validation; it is
skipped until a bundle with real paired L2 and L3 exists.
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


def _expected_levels(
    sym_l2: pl.DataFrame,
) -> dict[tuple[int, str], list[tuple[float, int]]]:
    """Published levels keyed by ``(ts_ns, side)``, built in a single pass."""
    grouped: dict[tuple[int, str], list[tuple[float, int]]] = {}
    ordered = sym_l2.sort(["ts_ns", "side", "level"]).select(
        "ts_ns", "side", "px", "size"
    ).to_dict(as_series=False)
    for ts_ns, side, px, size in zip(
        ordered["ts_ns"], ordered["side"], ordered["px"], ordered["size"], strict=True
    ):
        grouped.setdefault((ts_ns, side), []).append((px, size))
    return grouped


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

        # Group once, up front. Filtering the symbol's whole L2 frame inside
        # the loop makes the comparison O(snapshots x l2_rows); at real L3
        # densities that term dominates.
        expected_by_ts = _expected_levels(sym_l2)

        for ts_ns, snap in zip(ts_list, snaps, strict=True):
            for side in ("buy", "sell"):
                expected = expected_by_ts.get((ts_ns, side), [])
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
