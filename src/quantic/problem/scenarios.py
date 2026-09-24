"""Price scenarios and the mark-to-market loss they induce.

Dial D3 replaces the variance risk term with CVaR. Scenarios perturb
**prices only**, and the loss is the mark-to-market hit on the holdings still
outstanding:

    L_s(x) = - sum over t, i of  h[i,t] * lot_size[i] * price[i] * returns[s,t,i]

The variance term is already the variance of exactly this quantity, so D3
swaps how one loss distribution is summarised rather than introducing a second
one -- which is what lets the screening design attribute a crossover shift to
D3 rather than to an interaction with D1.

The leading minus makes ``L_s`` a loss: we are long and liquidating, so a
negative return on an outstanding position hurts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

from quantic.problem.instance import Instance
from quantic.problem.params import MarketParams
from quantic.problem.schedule import Schedule

# Weights come from summing float probabilities, so exact equality with 1.0 is
# not available. Loose enough to absorb accumulation over thousands of
# scenarios, tight enough that a genuinely unnormalised vector is rejected.
_WEIGHT_TOL = 1e-9


@dataclass(frozen=True)
class ReductionReport:
    """What a scenario reduction cost, in both abstract and currency terms.

    The three CVaR fields are ``None`` until a witness schedule exists to
    evaluate them at, which is only after the instance is built. The generator
    fills them in via :func:`dataclasses.replace`.
    """

    n_raw: int
    n_reduced: int
    wasserstein: float
    cvar_raw: float | None = None
    cvar_reduced: float | None = None
    cvar_discrepancy: float | None = None

    def __post_init__(self) -> None:
        if self.n_raw < 1:
            raise ValueError(f"n_raw must be positive, got {self.n_raw}")
        if not 1 <= self.n_reduced < self.n_raw:
            raise ValueError(
                f"n_reduced must lie in [1, n_raw), got {self.n_reduced} against "
                f"n_raw={self.n_raw}: a 'reduction' that keeps every scenario has "
                "reduced nothing"
            )
        if self.wasserstein < 0 or not np.isfinite(self.wasserstein):
            raise ValueError(
                f"wasserstein must be finite and non-negative, got {self.wasserstein}"
            )

    def describe(self) -> dict[str, Any]:
        """JSON-safe, for the instance content hash."""
        return {
            "n_raw": self.n_raw,
            "n_reduced": self.n_reduced,
            "wasserstein": self.wasserstein,
            "cvar_raw": self.cvar_raw,
            "cvar_reduced": self.cvar_reduced,
            "cvar_discrepancy": self.cvar_discrepancy,
        }


@dataclass(frozen=True, eq=False)
class ScenarioSet:
    """Weighted price-shock scenarios.

    ``eq=False`` with explicit ``__eq__``/``__hash__``: a frozen dataclass
    holding an ndarray gets a generated ``__eq__`` that raises ``ValueError``
    for arrays of two or more elements, and a ``__hash__`` that raises
    ``TypeError``. ``MarketParams`` and ``Instance`` carry the same fix.
    """

    returns: np.ndarray          # (S, T, N) fractional price shocks
    weights: np.ndarray          # (S,) probabilities summing to 1
    seed: int
    reduction: ReductionReport | None = None

    def __post_init__(self) -> None:
        returns = np.asarray(self.returns, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if returns.ndim != 3:
            raise ValueError(
                f"returns must be 3-D (scenarios, buckets, assets), got shape "
                f"{returns.shape}"
            )
        if returns.shape[0] < 1:
            raise ValueError("a ScenarioSet needs at least one scenario")
        if weights.shape != (returns.shape[0],):
            raise ValueError(
                f"weights has shape {weights.shape} but there are "
                f"{returns.shape[0]} scenarios"
            )
        if not np.isfinite(returns).all():
            raise ValueError("returns must be finite")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be finite and non-negative")
        total = float(weights.sum())
        if abs(total - 1.0) > _WEIGHT_TOL:
            raise ValueError(f"weights must sum to 1, got {total!r}")

    @property
    def n_scenarios(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_buckets(self) -> int:
        return int(self.returns.shape[1])

    @property
    def n_assets(self) -> int:
        return int(self.returns.shape[2])

    def losses(self, instance: Instance, schedule: Schedule) -> np.ndarray:
        """``(S,)`` mark-to-market loss per scenario, in currency."""
        if (self.n_buckets, self.n_assets) != (instance.n_buckets, instance.n_assets):
            raise ValueError(
                f"scenario shape (buckets={self.n_buckets}, assets={self.n_assets}) "
                f"does not match the instance ({instance.n_buckets}, "
                f"{instance.n_assets})"
            )
        holdings = instance.remaining_lots(schedule) * instance.lot_sizes()[:, None]
        exposure = holdings * instance.params.prices()[:, None]     # (N, T) currency
        return -np.einsum("it,sti->s", exposure, np.asarray(self.returns, dtype=float))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScenarioSet):
            return NotImplemented
        return (
            self.seed == other.seed
            and self.reduction == other.reduction
            and np.array_equal(self.returns, other.returns)
            and np.array_equal(self.weights, other.weights)
        )

    def __hash__(self) -> int:
        returns = np.asarray(self.returns, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        return hash(
            (
                self.seed,
                self.reduction,
                returns.shape,
                returns.tobytes(),
                weights.tobytes(),
            )
        )


def generate_scenarios(
    params: MarketParams,
    n_buckets: int,
    *,
    seed: int,
    n_raw: int = 2000,
) -> ScenarioSet:
    """Draw ``n_raw`` price paths from ``params.covariance``.

    Shocks are independent across buckets, which is what makes ``Var(L_s)``
    equal the variance term's quadratic form.

    ``params.covariance`` is already at the bucket horizon, so nothing is
    rescaled here.

    Sampling goes through an explicit eigendecomposition rather than
    ``Generator.multivariate_normal``: the covariance passes a *relative* PSD
    check, so a tiny negative eigenvalue from rounding is legitimate, and
    ``multivariate_normal`` would emit a RuntimeWarning for it. Clipping those
    to zero here is deterministic, warning-free, and says exactly what is
    being tolerated.
    """
    if n_raw < 1:
        raise ValueError(f"n_raw must be positive, got {n_raw}")
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be positive, got {n_buckets}")

    covariance = np.asarray(params.covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Rounding-scale negatives only; MarketParams already rejected a
    # genuinely non-PSD covariance at construction.
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))

    rng = np.random.default_rng([seed, n_raw, n_buckets, params.n_assets])
    standard = rng.standard_normal((n_raw, n_buckets, params.n_assets))
    returns = standard @ factor.T

    return ScenarioSet(
        returns=returns, weights=np.full(n_raw, 1.0 / n_raw), seed=seed
    )


def reduce_scenarios(raw: ScenarioSet, *, n_target: int = 25) -> ScenarioSet:
    """Reduce ``raw`` to ``n_target`` weighted representatives.

    Dupacova fast-forward selection: greedily pick the scenario that most
    reduces

        D(selected) = sum over s of p_s * min over j in selected of ||xi_s - xi_j||

    then assign every raw scenario to its nearest representative and give each
    representative the total probability assigned to it.

    ``D`` at termination is the Kantorovich (Wasserstein-1) distance between
    the raw and reduced distributions under that assignment, so the error the
    spec requires comes out of the algorithm rather than needing a separate
    optimal-transport solve.

    Representatives are **actual scenarios**, never centroids: CVaR is a tail
    statistic and an averaged path is smoother than any real one, so centroids
    would systematically thin the tail being measured.
    """
    if raw.reduction is not None:
        raise ValueError(
            "this ScenarioSet is already reduced; reducing again would report a "
            "Wasserstein distance against the reduced set rather than the original"
        )
    if n_target < 1:
        raise ValueError(f"n_target must be positive, got {n_target}")
    if n_target >= raw.n_scenarios:
        raise ValueError(
            f"n_target={n_target} must be below the {raw.n_scenarios} raw scenarios; "
            "keeping them all reduces nothing"
        )

    flat = np.asarray(raw.returns, dtype=float).reshape(raw.n_scenarios, -1)
    distances = squareform(pdist(flat))
    probabilities = np.asarray(raw.weights, dtype=float)

    selected: list[int] = []
    nearest = np.full(raw.n_scenarios, np.inf)
    for _ in range(n_target):
        # Cost of adding each candidate, vectorised over all candidates at once.
        costs = probabilities @ np.minimum(nearest[:, None], distances)
        costs[selected] = np.inf
        chosen = int(np.argmin(costs))
        selected.append(chosen)
        nearest = np.minimum(nearest, distances[:, chosen])

    order = np.array(sorted(selected))
    assignment = np.argmin(distances[:, order], axis=1)
    weights = np.zeros(n_target, dtype=float)
    np.add.at(weights, assignment, probabilities)
    # Renormalise against accumulated float error so __post_init__'s tolerance
    # is a check on the algorithm rather than on floating-point addition.
    weights = weights / weights.sum()

    report = ReductionReport(
        n_raw=raw.n_scenarios,
        n_reduced=n_target,
        wasserstein=float(probabilities @ nearest),
    )
    return ScenarioSet(
        returns=np.asarray(raw.returns, dtype=float)[order],
        weights=weights,
        seed=raw.seed,
        reduction=report,
    )
