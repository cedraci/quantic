"""Core domain types. This module must not import from any other quantic package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum


class Side(StrEnum):
    """The side a **resting** order sits on.

    This is the meaning carried by the ``side`` column of ``l2_depth`` and
    ``l3_messages``, and by ``micro.liquidity.signed_order_flow``: an
    execution with ``side == SELL`` was a resting sell order that someone
    lifted. It is *not* the side of the participant who crossed the spread --
    that is :class:`Aggressor`.
    """

    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class Aggressor(Enum):
    """The side of the participant **crossing** the spread.

    Deliberately a separate type from :class:`Side` even though both spell
    their members ``"buy"``/``"sell"``, because for one and the same trade
    they take opposite values: a buying aggressor consumes sell-side resting
    liquidity. The two meanings previously shared :class:`Side` -- resting in
    the data layer and in ``signed_order_flow``, aggressor in
    ``impact.depth_walk.walk`` -- so a caller passing one where the other was
    expected produced a plausible, silently inverted answer.

    M2's liquidation problem is written entirely in aggressor terms (we are
    the seller crossing the spread), so this is the type that crosses the
    M1/M2 boundary.

    A plain :class:`enum.Enum`, not a :class:`enum.StrEnum`, unlike
    :class:`Side`. ``StrEnum`` members compare equal to any string of the same
    value, so ``Aggressor.BUY == Side.BUY`` would be ``True`` and separating
    the two types would buy nothing. ``Side`` stays a ``StrEnum`` because it
    is a stored value -- it is literally the ``side`` column of
    ``l2_depth`` and ``l3_messages``. ``Aggressor`` never appears in storage.
    """

    BUY = "buy"
    SELL = "sell"

    @property
    def consumes(self) -> Side:
        """The resting side this aggressor trades against."""
        return Side.SELL if self is Aggressor.BUY else Side.BUY

    @classmethod
    def from_resting_side(cls, side: Side) -> Aggressor:
        """Infer the aggressor from the resting side that was executed."""
        return cls.BUY if side is Side.SELL else cls.SELL


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
