"""Core domain types. This module must not import from any other quantic package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class EmptyBookError(ValueError):
    """Raised when a derived price is requested from a book that cannot supply it."""


@dataclass(frozen=True, slots=True)
class Asset:
    symbol: str
    lot_size: int
    tick_size: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {self.lot_size}")
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")


@dataclass(frozen=True, slots=True)
class PriceLevel:
    price: float
    size: int


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """A point-in-time view of one symbol's order book.

    ``bids`` are ordered by descending price, ``asks`` by ascending price.
    Both orderings are validated on construction, because every downstream
    consumer indexes level 0 as the touch.
    """

    ts_ns: int
    symbol: str
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]

    def __post_init__(self) -> None:
        bid_prices = [lvl.price for lvl in self.bids]
        if bid_prices != sorted(bid_prices, reverse=True):
            raise ValueError(f"bids must be in descending price order, got {bid_prices}")
        ask_prices = [lvl.price for lvl in self.asks]
        if ask_prices != sorted(ask_prices):
            raise ValueError(f"asks must be in ascending price order, got {ask_prices}")

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        if bid >= ask:
            raise EmptyBookError(
                f"crossed book for {self.symbol} at ts_ns={self.ts_ns}: bid {bid} >= ask {ask}"
            )
        return (bid + ask) / 2.0

    @property
    def spread(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid
