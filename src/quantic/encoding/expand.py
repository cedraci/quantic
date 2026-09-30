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

from dataclasses import dataclass, field

import numpy as np

from quantic.encoding.cvar import CVaRGadget, expand_cvar
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
    approximand: ApproximandReport | None = None

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


@dataclass(frozen=True)
class ApproximandReport:
    """The quadratic substituted for concave impact, and what it cost.

    Emitted only when the encoding offers no exact per-value table -- that is,
    only for the log encoding with dial D1 on. Its absence is the encoder's
    claim that the term was represented exactly.

    Not every cell is fitted even then. A block cell is one bit and a cell
    capped at zero lots has nothing to fit, so both keep their exact cost, and
    ``exact`` carries those tables: the substituted term is a sum over every
    cell, not only over the ones the fit touched.
    """

    coefficients: dict[tuple[int, int], tuple[float, float]]
    max_abs_error: float
    max_rel_error: float
    worst_cell: tuple[int, int]
    exact: dict[tuple[int, int], dict[int, float]] = field(default_factory=dict)

    def temporary_cost(self, schedule) -> float:
        """The substituted temporary-impact term, summed over cells."""
        total = 0.0
        for (i, t), (b, c) in self.coefficients.items():
            x = float(schedule.lots[i][t])
            total += b * x + c * x * x
        for (i, t), table in self.exact.items():
            lots = int(schedule.lots[i][t])
            if lots not in table:
                raise ValueError(
                    f"{lots} is not one of the values {tuple(table)} the cell at "
                    f"(asset {i}, bucket {t}) can take, so the encoder has no cost "
                    "for it"
                )
            total += table[lots]
        return total

    def cell_cost(self, i: int, t: int, lots: int) -> float:
        """The substituted temporary impact of one cell trading ``lots``.

        What an exact solver minimises when it is asked for the optimum of
        the function the encoder declares, rather than of the power law.
        """
        if (i, t) in self.coefficients:
            b, c = self.coefficients[(i, t)]
            x = float(lots)
            return b * x + c * x * x
        if (i, t) not in self.exact:
            raise KeyError(
                f"the cell at (asset {i}, bucket {t}) is in neither the fitted "
                "coefficients nor the exact tables of this report"
            )
        table = self.exact[(i, t)]
        if lots not in table:
            raise ValueError(
                f"{lots} is not one of the values {tuple(table)} the cell at "
                f"(asset {i}, bucket {t}) can take, so the encoder has no cost "
                "for it"
            )
        return table[lots]

    def describe(self) -> dict:
        """JSON-safe, for the results row."""
        return {
            "kind": "quadratic_least_squares",
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "worst_cell": list(self.worst_cell),
        }


def _fit_quadratic(values: tuple[int, ...], costs: dict[int, float]) -> tuple[float, float]:
    """Least-squares ``b*x + c*x**2`` over the cell's admissible lot values.

    No constant term: ``f(0) == 0`` exactly, and a fitted offset would shift
    every energy including the empty schedule's.
    """
    grid = [v for v in values if v != 0]
    if not grid:
        return 0.0, 0.0
    design = np.array([[float(v), float(v) * float(v)] for v in grid], dtype=float)
    target = np.array([costs[v] for v in grid], dtype=float)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(solution[0]), float(solution[1])


def _spread(instance: Instance, layout: Layout, registry: VarRegistry) -> Qubo:
    builder = QuboBuilder(registry)
    for plan in layout.cells:
        a = instance.params.assets[plan.asset]
        per_lot = a.half_spread * float(instance.assets[plan.asset].lot_size)
        builder.add_form(plan.value_form, per_lot)
    return builder.build()


def _temporary(
    instance: Instance, layout: Layout, registry: VarRegistry
) -> tuple[Qubo, ApproximandReport | None]:
    builder = QuboBuilder(registry)
    coefficients: dict[tuple[int, int], tuple[float, float]] = {}
    exact_tables: dict[tuple[int, int], dict[int, float]] = {}
    max_abs = 0.0
    max_rel = 0.0
    worst: tuple[int, int] = (0, 0)

    for plan in layout.cells:
        costs = {v: cell_temporary_cost(instance, plan.asset, plan.bucket, v)
                 for v in plan.values}
        exact = plan.exact_cost_form(costs)
        if exact is not None:
            builder.add_form(exact)
            exact_tables[(plan.asset, plan.bucket)] = costs
            continue
        if not instance.dials.concave_impact:
            # Almgren-Chriss is exactly quadratic in the lots, so one
            # coefficient carries the whole cell: cost(x) == cost(1) * x**2.
            builder.add_square(
                plan.value_form,
                cell_temporary_cost(instance, plan.asset, plan.bucket, 1),
            )
            continue
        b, c = _fit_quadratic(plan.values, costs)
        coefficients[(plan.asset, plan.bucket)] = (b, c)
        builder.add_form(plan.value_form, b)
        builder.add_square(plan.value_form, c)
        for v in plan.values:
            error = abs(costs[v] - (b * v + c * v * v))
            if error > max_abs:
                max_abs, worst = error, (plan.asset, plan.bucket)
            if costs[v] != 0.0:
                max_rel = max(max_rel, error / abs(costs[v]))

    if not coefficients:
        return builder.build(), None
    return builder.build(), ApproximandReport(
        coefficients=coefficients,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        worst_cell=worst,
        exact=exact_tables,
    )


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
    instance: Instance,
    layout: Layout,
    registry: VarRegistry,
    *,
    gadget: CVaRGadget | None = None,
) -> QuboBreakdown:
    """Spec section 5.2's four terms as four QUBOs over ``registry``."""
    risk = instance.risk
    if isinstance(risk, VarianceRisk):
        risk_qubo = _variance(instance, layout, registry, risk.lam)
    elif isinstance(risk, CVaRRisk):
        if gadget is None:
            raise ValueError(
                "CVaRRisk needs a gadget: build_cvar_gadget derives the zeta, u and g "
                "grids from the instance, and the penalty strength comes from "
                "derive_penalties. expand_objective will not invent either"
            )
        cost, cvar_penalty = expand_cvar(instance, layout, registry, gadget)
        risk_qubo = cost + cvar_penalty
    else:
        raise NotImplementedError(
            f"expand_objective does not know how to encode risk spec {risk.name!r} "
            f"({type(risk).__name__}). Dispatch is on type, never on a lam attribute"
        )
    temporary, approximand = _temporary(instance, layout, registry)
    return QuboBreakdown(
        spread=_spread(instance, layout, registry),
        temporary=temporary,
        permanent=_permanent(instance, layout, registry),
        risk=risk_qubo,
        approximand=approximand,
    )
