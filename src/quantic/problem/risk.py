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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from quantic.problem.scenarios import ScenarioSet


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


@dataclass(frozen=True)
class CVaRRisk:
    """Conditional Value-at-Risk over a weighted scenario set -- dial D3.

    Evaluated in Rockafellar-Uryasev form against an explicit ``zeta``:

        CVaR_alpha(x, zeta) = zeta
            + 1/(1 - alpha) * sum over s of w_s * max(L_s(x) - zeta, 0)

    ``zeta`` is a decision variable, not a field here: it depends on the
    schedule being scored, and M3 binary-expands it. It is passed to
    ``objective.evaluate`` as a keyword.

    A plain frozen dataclass despite holding array-backed scenarios:
    :class:`~quantic.problem.scenarios.ScenarioSet` defines its own
    ``__eq__``/``__hash__``, so the generated ones here never touch a raw
    ndarray.
    """

    lam: float
    alpha: float
    scenarios: ScenarioSet
    name: str = field(default="cvar")

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError(
                f"lam must be non-negative, got {self.lam}: a negative risk aversion "
                "rewards tail loss and makes the objective unbounded below"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(
                f"alpha must lie strictly in (0, 1), got {self.alpha}: at 0 CVaR is "
                "the mean and at 1 the tail is empty"
            )

    def describe(self) -> dict[str, Any]:
        """JSON-safe, for the instance content hash.

        Deliberately omits ``returns``: the seed and scenario count identify
        the set, and embedding the array would put megabytes of floats into
        every hash payload.
        """
        reduction = self.scenarios.reduction
        return {
            "name": self.name,
            "lam": self.lam,
            "alpha": self.alpha,
            "n_scenarios": self.scenarios.n_scenarios,
            "scenario_seed": self.scenarios.seed,
            "reduction": None if reduction is None else reduction.describe(),
        }
