"""Spec section 5.2's objective, as one QUBO per term.

``problem/objective.py`` returns its four terms separately, for the stated
reason that "a single scalar says nothing about which term is wrong". The
encoder mirrors that: the gate compares field by field, so a mismatch names
the term that broke rather than only the instance.

Three of the four terms are exactly representable under every encoding:
spread is linear in the lots, permanent impact is a cross-bucket quadratic,
and variance is a quadratic in the remaining holdings. The fourth -- temporary
impact -- is exact whenever the encoding offers a per-value cost table, and
under the log encoding it is exact only while dial D1 is off. D1 on under log
is Task 6's explicit approximand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.encoding.layout import Layout
from quantic.encoding.qubo import LinearForm, Qubo, QuboBuilder
from quantic.encoding.registry import VarRegistry
from quantic.problem.instance import Instance
from quantic.problem.objective import impact_model_for
from quantic.problem.risk import CVaRRisk, VarianceRisk


@dataclass(frozen=True, eq=False)
class QuboBreakdown:
    """One QUBO per objective term, plus whatever was approximated to get them."""

    spread: Qubo
    temporary: Qubo
    permanent: Qubo
    risk: Qubo
    approximand: object | None = None

    @property
    def total(self) -> Qubo:
        return self.spread + self.temporary + self.permanent + self.risk


def cell_temporary_cost(instance: Instance, i: int, t: int, lots: int) -> float:
    """Exactly what ``objective.evaluate`` charges cell ``(i, t)`` for ``lots``.

    The single source of truth for the cost table under a table encoding and
    for Task 6's quadratic fit under the log encoding. A second implementation
    is how the encoder and the evaluator would silently drift apart.
    """
    if lots == 0:
        return 0.0
    a = instance.params.assets[i]
    p = instance.params.impact_params(i)
    model = impact_model_for(instance, i)
    traded = float(lots) * float(instance.assets[i].lot_size)
    return float(model.temporary_cost(traded, p) * traded * a.price)


def _spread(instance: Instance, layout: Layout, registry: VarRegistry) -> Qubo:
    builder = QuboBuilder(registry)
    for plan in layout.cells:
        a = instance.params.assets[plan.asset]
        per_lot = a.half_spread * float(instance.assets[plan.asset].lot_size)
        builder.add_form(plan.value_form, per_lot)
    return builder.build()


def _temporary(instance: Instance, layout: Layout, registry: VarRegistry) -> Qubo:
    builder = QuboBuilder(registry)
    for plan in layout.cells:
        costs = {v: cell_temporary_cost(instance, plan.asset, plan.bucket, v)
                 for v in plan.values}
        exact = plan.exact_cost_form(costs)
        if exact is not None:
            builder.add_form(exact)
            continue
        if instance.dials.concave_impact:
            raise NotImplementedError(
                f"concave impact (D1) has no exact quadratic form under the "
                f"{layout.encoding_name!r} encoding, so it needs an explicit "
                "approximand and a reported error. See M3a plan Task 6"
            )
        # D1 off is Almgren-Chriss, whose cost is exactly quadratic in the lots,
        # so one coefficient carries the whole cell: cost(x) == cost(1) * x**2.
        builder.add_square(
            plan.value_form,
            cell_temporary_cost(instance, plan.asset, plan.bucket, 1),
        )
    return builder.build()


def _permanent(instance: Instance, layout: Layout, registry: VarRegistry) -> Qubo:
    builder = QuboBuilder(registry)
    for i in range(instance.n_assets):
        a = instance.params.assets[i]
        lot = float(instance.assets[i].lot_size)
        coef = a.gamma * a.sigma_bucket * a.price * lot * lot / a.bucket_volume_shares
        for t in range(instance.n_buckets):
            here = layout.cell(i, t).value_form
            # The evaluator's midpoint: everything displaced before this bucket,
            # plus half of our own, because we trade through it.
            displaced = here.scaled(0.5)
            for s in range(t):
                displaced = displaced + layout.cell(i, s).value_form
            builder.add_product(displaced, here, coef)
    return builder.build()


def holdings_form(instance: Instance, layout: Layout, i: int, t: int) -> LinearForm:
    """Shares of asset ``i`` still held after bucket ``t``, as a linear form."""
    lot = float(instance.assets[i].lot_size)
    form = LinearForm.constant_form(lot * float(instance.initial_lots[i]))
    for s in range(t + 1):
        form = form + layout.cell(i, s).value_form.scaled(-lot)
    return form


def _variance(
    instance: Instance, layout: Layout, registry: VarRegistry, lam: float
) -> Qubo:
    builder = QuboBuilder(registry)
    if lam == 0.0:
        return builder.build()
    sigma = np.asarray(instance.params.price_covariance(), dtype=float)
    for t in range(instance.n_buckets):
        forms = [holdings_form(instance, layout, i, t) for i in range(instance.n_assets)]
        for i in range(instance.n_assets):
            for j in range(instance.n_assets):
                builder.add_product(forms[i], forms[j], lam * float(sigma[i, j]))
    return builder.build()


def expand_objective(
    instance: Instance, layout: Layout, registry: VarRegistry
) -> QuboBreakdown:
    """Spec section 5.2's four terms as four QUBOs over ``registry``."""
    risk = instance.risk
    if isinstance(risk, VarianceRisk):
        risk_qubo = _variance(instance, layout, registry, risk.lam)
    elif isinstance(risk, CVaRRisk):
        raise NotImplementedError(
            "CVaRRisk needs the Rockafellar-Uryasev gadget: a binary-expanded zeta "
            "plus two auxiliaries per scenario. See M3a plan Task 8"
        )
    else:
        raise NotImplementedError(
            f"expand_objective does not know how to encode risk spec {risk.name!r} "
            f"({type(risk).__name__}). Dispatch is on type, never on a lam attribute"
        )
    return QuboBreakdown(
        spread=_spread(instance, layout, registry),
        temporary=_temporary(instance, layout, registry),
        permanent=_permanent(instance, layout, registry),
        risk=risk_qubo,
    )
