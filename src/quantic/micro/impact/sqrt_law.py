"""Concave power-law impact — hardness dial D1 (spec section 4).

``delta = 0.5`` is the empirically supported square-root law. Any ``delta < 1``
makes the liquidation objective concave, so minimising it is NP-hard and MIQP
solvers lose their bound. ``delta = 1`` degenerates to Almgren-Chriss, which is
how dial D1 is switched off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantic.micro.impact.base import ImpactParams


@dataclass(frozen=True)
class PowerLawImpact:
    delta: float
    y_coef: float
    gamma: float = 0.0
    name: str = field(default="power_law")

    def __post_init__(self) -> None:
        if not 0.0 < self.delta <= 1.0:
            raise ValueError(f"delta must lie in (0, 1], got {self.delta}")
        if self.y_coef < 0:
            raise ValueError(f"y_coef must be non-negative, got {self.y_coef}")

    def price_impact(self, q: float, p: ImpactParams) -> float:
        return self.y_coef * p.sigma_bucket * p.participation(q) ** self.delta

    def temporary_cost(self, q: float, p: ImpactParams) -> float:
        return self.price_impact(q, p) / (self.delta + 1.0)

    def permanent_impact(self, q: float, p: ImpactParams) -> float:
        return self.gamma * p.sigma_bucket * p.participation(q)


def sqrt_law(y_coef: float, gamma: float = 0.0) -> PowerLawImpact:
    return PowerLawImpact(delta=0.5, y_coef=y_coef, gamma=gamma)
