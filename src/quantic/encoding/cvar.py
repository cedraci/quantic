"""Rockafellar-Uryasev CVaR as a QUBO, and the discretisation it declares.

The risk term is

    lam * ( zeta + 1/(1 - alpha) * sum_s w_s * max(L_s(x) - zeta, 0) )

and ``max(u, 0)`` is what a QUBO cannot say directly. The standard device is
``u_s >= L_s - zeta`` with ``u_s`` minimised, which stays quadratic only with a
slack:

    cost      lam * ( zeta + 1/(1 - alpha) * sum_s w_s * u_s )
    penalty   P * sum_s ( u_s - L_s + zeta - g_s )**2

``L_s`` is linear in the holdings -- scenarios perturb prices only, which is
M2b's design decision -- but with real coefficients, while ``zeta``, ``u`` and
``g`` live on finite binary grids. So that equality cannot hold exactly, for
the same reason M2b accepted that a binary-expanded ``zeta`` cannot land on the
empirical quantile.

This module therefore states exactly what it encodes. The gate proves the QUBO
equals :meth:`CVaRGadget.encoded_cvar` exactly, and separately proves that
``encoded_cvar`` at the penalty-optimal auxiliaries sits within
:attr:`CVaRGadget.resolution` of continuous Rockafellar-Uryasev. Two exact
assertions; no tolerance on the correctness claim.

``u`` and ``g`` share ``zeta``'s step, so ``u - g`` ranges over integer
multiples of one step and the best auxiliaries are closed-form rather than a
search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.encoding.layout import Layout
from quantic.encoding.qubo import LinearForm, Qubo, QuboBuilder
from quantic.encoding.registry import VarKey, VarRegistry
from quantic.problem.instance import Instance
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class Grid:
    """A uniform binary grid: ``value = lo + step * sum 2**b z_b``.

    Plain powers of two rather than the bounded coefficients the lot cells use.
    Every one of the ``2**n_bits`` states is a meaningful grid point here, so
    there is no forbidden region to suppress and the map is injective, which is
    what lets :meth:`encode` and :meth:`decode` round-trip exactly.
    """

    lo: float
    step: float
    n_bits: int
    keys: tuple[VarKey, ...]

    def __post_init__(self) -> None:
        if self.n_bits < 1:
            raise ValueError(f"a grid needs at least one bit, got {self.n_bits}")
        if self.step <= 0 or not np.isfinite(self.step):
            raise ValueError(f"step must be finite and positive, got {self.step}")
        if len(self.keys) != self.n_bits:
            raise ValueError(
                f"{len(self.keys)} keys for {self.n_bits} bits"
            )

    @property
    def hi(self) -> float:
        return self.lo + self.step * float(2**self.n_bits - 1)

    @property
    def resolution(self) -> float:
        """Worst-case distance from an arbitrary real to the nearest grid point."""
        return self.step / 2.0

    def form(self) -> LinearForm:
        out = LinearForm.constant_form(self.lo)
        for b, key in enumerate(self.keys):
            out = out + LinearForm.of(key, self.step * float(1 << b))
        return out

    def _level(self, value: float) -> int:
        raw = round((float(value) - self.lo) / self.step)
        return int(min(max(raw, 0), 2**self.n_bits - 1))

    def snap(self, value: float) -> float:
        return self.lo + self.step * float(self._level(value))

    def encode(self, value: float) -> dict[VarKey, int]:
        level = self._level(value)
        return {key: (level >> b) & 1 for b, key in enumerate(self.keys)}

    def decode(self, bits) -> float:
        level = sum((1 << b) for b, key in enumerate(self.keys) if bits[key])
        return self.lo + self.step * float(level)


@dataclass(frozen=True, eq=False)
class CVaRGadget:
    """The Rockafellar-Uryasev expansion, and the function it actually implements."""

    zeta: Grid
    u: tuple[Grid, ...]
    g: tuple[Grid, ...]
    alpha: float
    lam: float
    weights: np.ndarray
    penalty: float
    loss_coefficients: np.ndarray  # (S, N, T) coefficient on holdings in lots

    @property
    def n_scenarios(self) -> int:
        return len(self.u)

    @property
    def keys(self) -> tuple[VarKey, ...]:
        out: list[VarKey] = list(self.zeta.keys)
        for grid in self.u:
            out.extend(grid.keys)
        for grid in self.g:
            out.extend(grid.keys)
        return tuple(out)

    @property
    def resolution(self) -> float:
        """How far ``encoded_cvar`` can sit from continuous Rockafellar-Uryasev.

        Two sources, both at one grid step. Each ``u_s`` misses
        ``max(L_s - zeta, 0)`` by at most half a step, which the cost weight
        carries; and the residual the penalty charges is at most half a step,
        squared. The second term is why a coarse grid is expensive: it shrinks
        quadratically, so buying accuracy costs only bits.
        """
        half = self.zeta.step / 2.0
        cost_gap = self.lam * half / (1.0 - self.alpha)
        floor = self.penalty * float(self.n_scenarios) * half * half
        return float(cost_gap + floor)

    def losses_from(
        self, instance: Instance, layout: Layout, schedule: Schedule
    ) -> np.ndarray:
        """``L_s`` for every scenario, from the same coefficients the QUBO uses."""
        holdings = instance.remaining_lots(schedule).astype(float)  # (N, T)
        return np.einsum("snt,nt->s", self.loss_coefficients, holdings)

    def optimal_aux(self, losses: np.ndarray, zeta: float) -> dict[VarKey, int]:
        """The penalty-minimising auxiliaries, in closed form.

        ``u`` and ``g`` share a step, so the residual is minimised by putting
        the whole target into whichever of the two can hold it and leaving the
        other at zero. Ties never arise: the target's sign decides.
        """
        bits: dict[VarKey, int] = {}
        for s, target in enumerate(np.asarray(losses, dtype=float) - float(zeta)):
            if target >= 0.0:
                bits.update(self.u[s].encode(target))
                bits.update(self.g[s].encode(0.0))
            else:
                bits.update(self.u[s].encode(0.0))
                bits.update(self.g[s].encode(-target))
        return bits

    def encoded_cvar(self, losses: np.ndarray, zeta: float, aux) -> float:
        """The exact algebraic value the QUBO implements, penalty included."""
        losses = np.asarray(losses, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        excess = 0.0
        residual = 0.0
        for s in range(self.n_scenarios):
            u = self.u[s].decode(aux)
            g = self.g[s].decode(aux)
            excess += weights[s] * u
            gap = u - float(losses[s]) + float(zeta) - g
            residual += gap * gap
        cost = self.lam * (float(zeta) + excess / (1.0 - self.alpha))
        return float(cost + self.penalty * residual)


def _aux_grid(step: float, n_bits: int, prefix: VarKey) -> Grid:
    return Grid(
        lo=0.0, step=step, n_bits=n_bits,
        keys=tuple((*prefix, b) for b in range(n_bits)),
    )


def build_cvar_gadget(
    instance: Instance, layout: Layout, *, n_zeta_bits: int = 8, penalty: float
) -> CVaRGadget:
    """Derive every grid from the instance alone -- never from a schedule.

    ``h[i,t]`` lies in ``[0, X_i]``, so bounding ``L_s`` cell by cell bounds it
    over every schedule at once, and the true optimal ``zeta`` -- a quantile of
    the loss distribution -- always lands inside.
    """
    risk = instance.risk
    if not isinstance(risk, CVaRRisk):
        raise TypeError(
            f"build_cvar_gadget needs a CVaRRisk, got {type(risk).__name__}"
        )
    if n_zeta_bits < 1:
        raise ValueError(f"n_zeta_bits must be at least 1, got {n_zeta_bits}")

    returns = np.asarray(risk.scenarios.returns, dtype=float)      # (S, T, N)
    per_lot = instance.lot_sizes().astype(float) * instance.params.prices()
    # L_s = -sum_{t,i} h[i,t] * lot_i * price_i * returns[s,t,i]
    coefficients = np.einsum("stn,n->snt", -returns, per_lot)      # (S, N, T)
    position = np.asarray(instance.initial_lots, dtype=float)[None, :, None]
    upper = float((np.maximum(coefficients, 0.0) * position).sum(axis=(1, 2)).max())
    lower = float((np.minimum(coefficients, 0.0) * position).sum(axis=(1, 2)).min())

    span = upper - lower
    if span <= 0.0 or not np.isfinite(span):
        # Degenerate: every scenario is flat, so every loss is zero.
        span = 1.0
        lower, upper = 0.0, 1.0
    step = span / float(2**n_zeta_bits - 1) if n_zeta_bits > 1 else span

    zeta = Grid(
        lo=lower, step=step, n_bits=n_zeta_bits,
        keys=tuple(("zeta", b) for b in range(n_zeta_bits)),
    )
    n_scenarios = int(returns.shape[0])
    # u and g share zeta's step and must reach zeta's span, so they take
    # zeta's width. Recovering the width from span / step instead lets that
    # quotient round to a hair above 2**n - 1 and cost every grid a bit.
    u = tuple(_aux_grid(step, n_zeta_bits, ("u", s)) for s in range(n_scenarios))
    g = tuple(_aux_grid(step, n_zeta_bits, ("g", s)) for s in range(n_scenarios))
    return CVaRGadget(
        zeta=zeta, u=u, g=g, alpha=risk.alpha, lam=risk.lam,
        weights=np.asarray(risk.scenarios.weights, dtype=float),
        penalty=float(penalty), loss_coefficients=coefficients,
    )


def _holdings_lots_form(instance: Instance, layout: Layout, i: int, t: int) -> LinearForm:
    form = LinearForm.constant_form(float(instance.initial_lots[i]))
    for s in range(t + 1):
        form = form + layout.cell(i, s).value_form.scaled(-1.0)
    return form


def expand_cvar(
    instance: Instance, layout: Layout, registry: VarRegistry, gadget: CVaRGadget
) -> tuple[Qubo, Qubo]:
    """``(cost, penalty)`` -- kept apart so the gate can see which one broke."""
    cost = QuboBuilder(registry)
    pen = QuboBuilder(registry)
    zeta_form = gadget.zeta.form()

    cost.add_form(zeta_form, gadget.lam)
    weights = np.asarray(gadget.weights, dtype=float)
    scale = gadget.lam / (1.0 - gadget.alpha)

    holdings = [
        [_holdings_lots_form(instance, layout, i, t) for t in range(layout.n_buckets)]
        for i in range(layout.n_assets)
    ]

    for s in range(gadget.n_scenarios):
        u_form = gadget.u[s].form()
        g_form = gadget.g[s].form()
        cost.add_form(u_form, scale * float(weights[s]))

        residual = u_form + zeta_form + g_form.scaled(-1.0)
        for i in range(layout.n_assets):
            for t in range(layout.n_buckets):
                coef = float(gadget.loss_coefficients[s, i, t])
                if coef != 0.0:
                    residual = residual + holdings[i][t].scaled(-coef)
        pen.add_square(residual, gadget.penalty)

    return cost.build(), pen.build()
