"""Reconstruct order book state from an L3 message stream.

Deliberately independent of ``quantic.data.synth``'s internal book: comparing
the two is the correctness gate in Task 10 and spec section 10.3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from quantic.core.types import BookSnapshot, PriceLevel

_VALID_ACTIONS = frozenset({"add", "cancel", "execute", "replace"})
_VALID_SIDES = frozenset({"buy", "sell"})


class BookReconstructionError(RuntimeError):
    """Raised when a message cannot be applied to the current book state."""


@dataclass(slots=True)
class _Order:
    side: str
    px: float
    size: int


class BookBuilder:
    def __init__(self, symbol: str, *, strict: bool = True) -> None:
        self.symbol = symbol
        self.strict = strict
        self._orders: dict[int, _Order] = {}
        self._levels: dict[str, dict[float, int]] = {"buy": {}, "sell": {}}

    @property
    def open_orders(self) -> int:
        return len(self._orders)

    def _add_size(self, side: str, px: float, size: int) -> None:
        level = self._levels[side]
        level[px] = level.get(px, 0) + size
        if level[px] <= 0:
            del level[px]

    def _insert(self, order_id: int, side: str, px: float, size: int) -> None:
        if side not in _VALID_SIDES:
            raise BookReconstructionError(f"unknown side {side!r}")
        if order_id in self._orders:
            raise BookReconstructionError(f"duplicate order_id {order_id}")
        self._orders[order_id] = _Order(side, px, size)
        self._add_size(side, px, size)

    def _remove(self, order_id: int) -> _Order:
        order = self._orders.pop(order_id)
        self._add_size(order.side, order.px, -order.size)
        return order

    def apply(self, action: str, order_id: int, side: str, px: float, size: int) -> None:
        if action not in _VALID_ACTIONS:
            raise BookReconstructionError(f"unknown action {action!r}")

        if action == "add":
            self._insert(order_id, side, px, size)
            return

        order = self._orders.get(order_id)
        if order is None:
            if self.strict:
                raise BookReconstructionError(
                    f"unknown order {order_id} for action {action!r} on {self.symbol}"
                )
            return

        if action == "replace":
            self._remove(order_id)
            self._insert(order_id, side, px, size)
            return

        # cancel and execute share bookkeeping; only execute is traded volume.
        if size > order.size:
            raise BookReconstructionError(
                f"{action} size {size} exceeds resting size {order.size} for order {order_id}"
            )
        if size == order.size:
            self._remove(order_id)
        else:
            order.size -= size
            self._add_size(order.side, order.px, -size)

    def snapshot(self, ts_ns: int, levels: int) -> BookSnapshot:
        bids = sorted(self._levels["buy"].items(), key=lambda kv: -kv[0])[:levels]
        asks = sorted(self._levels["sell"].items())[:levels]
        return BookSnapshot(
            ts_ns=ts_ns,
            symbol=self.symbol,
            bids=tuple(PriceLevel(px, size) for px, size in bids),
            asks=tuple(PriceLevel(px, size) for px, size in asks),
        )


def _single_symbol(messages: pl.DataFrame) -> str:
    symbols = messages["symbol"].unique().to_list()
    if len(symbols) != 1:
        raise ValueError(f"expected messages for a single symbol, got {sorted(symbols)}")
    return symbols[0]


def snapshots_at(
    messages: pl.DataFrame,
    ts_list: Sequence[int],
    *,
    levels: int,
    strict: bool = True,
) -> list[BookSnapshot]:
    """Return one snapshot per requested timestamp, in the order requested."""
    symbol = _single_symbol(messages)
    builder = BookBuilder(symbol, strict=strict)
    ordered = sorted(range(len(ts_list)), key=lambda i: ts_list[i])
    out: dict[int, BookSnapshot] = {}
    cursor = 0

    for row in messages.sort(["ts_ns", "seq"]).iter_rows(named=True):
        while cursor < len(ordered) and row["ts_ns"] > ts_list[ordered[cursor]]:
            idx = ordered[cursor]
            out[idx] = builder.snapshot(ts_list[idx], levels)
            cursor += 1
        builder.apply(row["action"], row["order_id"], row["side"], row["px"], row["size"])

    while cursor < len(ordered):
        idx = ordered[cursor]
        out[idx] = builder.snapshot(ts_list[idx], levels)
        cursor += 1

    return [out[i] for i in range(len(ts_list))]


def final_book(messages: pl.DataFrame, *, levels: int, strict: bool = True) -> BookSnapshot:
    symbol = _single_symbol(messages)
    builder = BookBuilder(symbol, strict=strict)
    last_ts = 0
    for row in messages.sort(["ts_ns", "seq"]).iter_rows(named=True):
        builder.apply(row["action"], row["order_id"], row["side"], row["px"], row["size"])
        last_ts = row["ts_ns"]
    return builder.snapshot(last_ts, levels)
