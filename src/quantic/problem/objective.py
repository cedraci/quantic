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


def evaluate(instance: Instance, schedule: Schedule) -> ObjectiveBreakdown:
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

    risk = _risk_cost(instance, h)

    return ObjectiveBreakdown(
        spread_cost=float(spread),
        temporary_impact_cost=float(temporary),
        permanent_impact_cost=float(permanent),
        risk_cost=float(risk),
    )


def _risk_cost(instance: Instance, holdings_shares: np.ndarray) -> float:
    """``lam * sum over t of h_t' Sigma_price h_t``, with ``h`` in shares.

    ``MarketParams.covariance`` is a **log-return** covariance and is
    dimensionless; ``price_covariance`` converts it so the quadratic form
    comes out in currency squared. Using the return covariance directly would
    be a silent error of ``price**2``.

    The bucket horizon is already inside the covariance, so spec section 5.2's
    ``tau`` does not appear.
    """
    lam = getattr(instance.risk, "lam", None)
    if lam is None:
        raise NotImplementedError(
            f"objective.evaluate does not know how to score risk spec "
            f"{instance.risk.name!r}. M2a implements VarianceRisk; CVaRRisk arrives "
            "with M2b"
        )
    if lam == 0.0:
        return 0.0
    sigma_price = instance.params.price_covariance()
    total = 0.0
    for t in range(holdings_shares.shape[1]):
        h_t = holdings_shares[:, t].astype(float)
        total += float(h_t @ sigma_price @ h_t)
    return lam * total
