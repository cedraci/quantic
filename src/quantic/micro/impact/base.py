"""Impact model protocol and unit conventions.

Units, fixed once and relied on everywhere:

* ``price_impact`` returns a **fractional** mid displacement, always
  non-negative; the caller applies the sign.
* ``temporary_cost`` returns **fractional cost**: currency cost divided by
  ``|q| * price``.
* Integrating a power-law impact over the executed quantity gives
  ``temporary_cost = price_impact / (delta + 1)`` — the familiar 2/3 rule at
  ``delta = 0.5`` and 1/2 for linear impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ImpactParams:
    symbol: str
    sigma: float
    bucket_volume: float
    price: float

    def __post_init__(self) -> None:
        if self.bucket_volume <= 0:
            raise ValueError(f"bucket_volume must be positive, got {self.bucket_volume}")
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")


@runtime_checkable
class ImpactModel(Protocol):
    name: str

    def price_impact(self, q: float, p: ImpactParams) -> float: ...

    def temporary_cost(self, q: float, p: ImpactParams) -> float: ...

    def permanent_impact(self, q: float, p: ImpactParams) -> float: ...


def currency_cost(model: ImpactModel, q: float, p: ImpactParams) -> float:
    return model.temporary_cost(q, p) * abs(q) * p.price


def cost_in_bps(model: ImpactModel, q: float, p: ImpactParams) -> float:
    return model.temporary_cost(q, p) * 10_000.0
