"""Linear temporary impact: the delta = 1 member of the power-law family."""

from __future__ import annotations

from dataclasses import dataclass, field

from quantic.micro.impact.base import ImpactParams


@dataclass(frozen=True)
class AlmgrenChriss:
    eta: float
    gamma: float = 0.0
    name: str = field(default="almgren_chriss")

    def price_impact(self, q: float, p: ImpactParams) -> float:
        return self.eta * p.sigma * (abs(q) / p.bucket_volume)

    def temporary_cost(self, q: float, p: ImpactParams) -> float:
        return self.price_impact(q, p) / 2.0

    def permanent_impact(self, q: float, p: ImpactParams) -> float:
        return self.gamma * p.sigma * (abs(q) / p.bucket_volume)
