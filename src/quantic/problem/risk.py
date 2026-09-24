"""How risk enters the objective -- and the seam M2b extends.

Spec section 5.2's risk term is either a variance or a CVaR. M2a ships
:class:`VarianceRisk`; M2b adds ``CVaRRisk(alpha, scenarios, weights,
reduction_error)`` alongside it and a matching branch in
``problem.objective.evaluate``.

``Instance.risk`` is typed to the :class:`RiskSpec` protocol precisely so that
addition reshapes nothing else. CVaR costs a binary-expanded ``zeta`` plus one
auxiliary per scenario (spec section 5.4), so the scenario set has to live
somewhere that does not perturb the rest of the instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RiskSpec(Protocol):
    name: str

    def describe(self) -> dict[str, Any]:
        """JSON-safe parameters, for the results row and the instance hash."""
        ...


@dataclass(frozen=True, slots=True)
class VarianceRisk:
    """Mean-variance risk: ``lam * sum over t of h_t' Sigma_price h_t``.

    The bucket horizon is already inside ``MarketParams.covariance``, so spec
    section 5.2's ``tau`` does not appear again here.
    """

    lam: float
    name: str = field(default="variance")

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError(
                f"lam must be non-negative, got {self.lam}: a negative risk aversion "
                "rewards variance and makes the objective unbounded below"
            )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "lam": self.lam}
