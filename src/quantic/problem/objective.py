"""The reference objective evaluator -- spec section 5.2's four terms.

Terms are returned **separately**, not merely summed. M3's load-bearing
property test (spec section 10.1) asserts that a QUBO's energy equals the
directly evaluated objective; when that fails, a single scalar says nothing
about which term is wrong.

This is also the only place in the project where lots become shares. Spec
section 5.1 writes the decision variable in lots; everything in ``micro/`` is
in shares. The conversion is ``Asset.lot_size``, applied once, here.

``evaluate`` does **not** check feasibility. Scoring an infeasible schedule is
a legitimate thing to want -- it is how M3 calibrates penalty magnitudes --
and conflating the two is how an infeasible quantum solution gets compared
against a feasible classical one, which spec section 6.5 names as the error
that invalidates most published work in this area. Feasibility is
``problem.feasibility.classify``'s job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.micro.impact.almgren_chriss import AlmgrenChriss
from quantic.micro.impact.base import ImpactModel
from quantic.micro.impact.sqrt_law import PowerLawImpact
from quantic.problem.instance import Instance
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule

_BPS = 10_000.0


@dataclass(frozen=True, slots=True)
class ObjectiveBreakdown:
    """Spec section 5.2's four terms, each in currency."""

    spread_cost: float
    temporary_impact_cost: float
    permanent_impact_cost: float
    risk_cost: float

    @property
    def total(self) -> float:
        return (
            self.spread_cost
            + self.temporary_impact_cost
            + self.permanent_impact_cost
            + self.risk_cost
        )

    def in_bps(self, notional: float) -> ObjectiveBreakdown:
        """The same breakdown in basis points of notional.

        Spec section 9.1: the translation that makes results legible to
        practitioners rather than only to physicists.
        """
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        scale = _BPS / notional
        return ObjectiveBreakdown(
            spread_cost=self.spread_cost * scale,
            temporary_impact_cost=self.temporary_impact_cost * scale,
            permanent_impact_cost=self.permanent_impact_cost * scale,
            risk_cost=self.risk_cost * scale,
        )


def impact_model_for(instance: Instance, i: int) -> ImpactModel:
    """The temporary impact model dial D1 selects for asset ``i``.

    D1 on gives the calibrated concave power law, which is what makes the
    objective non-convex and the minimisation NP-hard. D1 off gives linear
    Almgren-Chriss, recovered by setting ``eta`` to the same coefficient so
    that only the exponent differs between the two arms.
    """
    a = instance.params.assets[i]
    if instance.dials.concave_impact:
        return PowerLawImpact(delta=a.delta, y_coef=a.y_coef, gamma=a.gamma)
    return AlmgrenChriss(eta=a.y_coef, gamma=a.gamma)


def optimal_zeta(instance: Instance, schedule: Schedule) -> float:
    """The weighted empirical VaR at the instance's tail level.

    Rockafellar-Uryasev attains its minimum here, so evaluating at this
    ``zeta`` gives the true CVaR. M3 will instead pass a binary-expanded
    ``zeta``, and the gap between the two is the discretisation cost of the
    encoding -- which is why the evaluator takes ``zeta`` rather than
    minimising internally.
    """
    if not isinstance(instance.risk, CVaRRisk):
        raise TypeError(
            f"optimal_zeta needs a CVaRRisk, got {type(instance.risk).__name__}"
        )
    scenarios = instance.risk.scenarios
    losses = scenarios.losses(instance, schedule)
    order = np.argsort(losses)
    cumulative = np.cumsum(np.asarray(scenarios.weights, dtype=float)[order])
    index = min(int(np.searchsorted(cumulative, instance.risk.alpha)), losses.size - 1)
    return float(losses[order][index])


def cvar_of(
    scenarios: ScenarioSet,
    instance: Instance,
    schedule: Schedule,
    *,
    alpha: float,
    zeta: float | None = None,
) -> float:
    """CVaR of ``schedule`` under an arbitrary ``scenarios``, unweighted by ``lam``.

    A diagnostic rather than part of the objective: the generator uses it to
    report how much scenario reduction moved the CVaR. ``zeta=None`` uses the
    optimum for *this* set, which is the right choice for a diagnostic --
    unlike the objective, nothing downstream has to reproduce it.
    """
    losses = scenarios.losses(instance, schedule)
    weights = np.asarray(scenarios.weights, dtype=float)
    if zeta is None:
        order = np.argsort(losses)
        cumulative = np.cumsum(weights[order])
        index = min(int(np.searchsorted(cumulative, alpha)), losses.size - 1)
        zeta = float(losses[order][index])
    excess = float(np.sum(weights * np.maximum(losses - zeta, 0.0)))
    return float(zeta + excess / (1.0 - alpha))


def evaluate(
    instance: Instance, schedule: Schedule, *, zeta: float | None = None
) -> ObjectiveBreakdown:
    """Evaluate spec section 5.2's objective on ``schedule``. Currency units."""
    if (schedule.n_assets, schedule.n_buckets) != (instance.n_assets, instance.n_buckets):
        raise ValueError(
            f"schedule shape ({schedule.n_assets}, {schedule.n_buckets}) does not match "
            f"the instance ({instance.n_assets}, {instance.n_buckets})"
        )

    lot_sizes = instance.lot_sizes()[:, None]
    q = schedule.as_array() * lot_sizes                  # shares traded per bucket
    cum_before = schedule.cumulative_before() * lot_sizes  # shares traded strictly before
    h = instance.remaining_lots(schedule) * lot_sizes      # shares held after each bucket

    spread = 0.0
    temporary = 0.0
    permanent = 0.0

    for i in range(instance.n_assets):
        a = instance.params.assets[i]
        p = instance.params.impact_params(i)
        model = impact_model_for(instance, i)
        for t in range(instance.n_buckets):
            traded = float(q[i, t])
            if traded == 0.0:
                continue
            spread += a.half_spread * traded
            temporary += model.temporary_cost(traded, p) * traded * a.price
            # Midpoint: the displacement already caused, plus half of our own,
            # because we trade through it. Using cum_before alone makes the
            # term path-dependent and understates it by sum(q**2) / 2.
            displaced = float(cum_before[i, t]) + traded / 2.0
            permanent += (
                a.gamma
                * a.sigma_bucket
                * (displaced / a.bucket_volume_shares)
                * traded
                * a.price
            )

    risk = _risk_cost(instance, schedule, h, zeta)

    return ObjectiveBreakdown(
        spread_cost=float(spread),
        temporary_impact_cost=float(temporary),
        permanent_impact_cost=float(permanent),
        risk_cost=float(risk),
    )


def _risk_cost(
    instance: Instance,
    schedule: Schedule,
    holdings_shares: np.ndarray,
    zeta: float | None,
) -> float:
    """Dispatch the risk term on the spec's type.

    Never on the presence of an attribute: ``CVaRRisk`` also carries a
    ``lam``, so an attribute probe would score it as variance -- silently, and
    with an entirely plausible number.
    """
    risk = instance.risk

    if isinstance(risk, VarianceRisk):
        if zeta is not None:
            raise ValueError(
                "zeta was supplied but the risk spec is VarianceRisk, which has no "
                "zeta. Passing one means the caller is unsure which objective they "
                "are scoring"
            )
        if risk.lam == 0.0:
            return 0.0
        sigma_price = instance.params.price_covariance()
        total = 0.0
        for t in range(holdings_shares.shape[1]):
            h_t = holdings_shares[:, t].astype(float)
            total += float(h_t @ sigma_price @ h_t)
        return risk.lam * total

    if isinstance(risk, CVaRRisk):
        if zeta is None:
            raise ValueError(
                "evaluate needs an explicit zeta under CVaRRisk. Rockafellar-Uryasev "
                "treats zeta as a decision variable, and M3 binary-expands it, so the "
                "call site must say which zeta it is scoring. "
                "problem.objective.optimal_zeta gives the minimising one"
            )
        if risk.lam == 0.0:
            return 0.0
        return risk.lam * cvar_of(
            risk.scenarios, instance, schedule, alpha=risk.alpha, zeta=zeta
        )

    raise NotImplementedError(
        f"objective.evaluate does not know how to score risk spec "
        f"{risk.name!r} ({type(risk).__name__})"
    )
