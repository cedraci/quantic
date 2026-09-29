"""The whole energy landscape of a small CVaR QUBO, for tests.

Every state of the bits outside the gadget is enumerated. For each, the gadget
is minimised exactly: the scenarios separate once the schedule and ``zeta`` are
fixed, so the minimum over the auxiliaries is a closed form per scenario and
the minimum over ``zeta`` is a scan of its grid. What comes back is the lowest
energy the QUBO can reach with those bits set -- the landscape a sampler that
found the global minimum would have been searching.

The closed form here is vectorised and so is a second implementation of
``CVaRGadget.cheapest_aux``. ``Landscape.state`` rebuilds a full bit vector
through the production method, and callers check it against ``Qubo.energy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.encoding.encode import EncodedInstance


@dataclass(frozen=True)
class Landscape:
    encoded: EncodedInstance
    outside: np.ndarray      # registry indices of the bits outside the gadget
    states: np.ndarray       # (B, len(outside)) every 0/1 assignment of them
    lots: np.ndarray         # (B, N, T) what each state's value forms add up to
    hard: np.ndarray         # (B,) what the hard constraints charge
    energy: np.ndarray       # (B,) lowest total energy reachable from the state
    zeta_level: np.ndarray   # (B,) the zeta grid level that reaches it

    def state(self, k: int) -> np.ndarray:
        """The full bit vector behind ``energy[k]``, auxiliaries included."""
        enc, gadget = self.encoded, self.encoded.gadget
        registry = enc.registry
        bits = np.zeros(len(registry), dtype=np.int64)
        bits[self.outside] = self.states[k]
        zeta = gadget.zeta.lo + gadget.zeta.step * float(self.zeta_level[k])
        holdings = (
            np.asarray(enc.instance.initial_lots, dtype=float)[:, None]
            - np.cumsum(self.lots[k], axis=1)
        )
        losses = np.einsum("snt,nt->s", gadget.loss_coefficients, holdings)
        for key, value in gadget.zeta.encode(zeta).items():
            bits[registry.index(key)] = value
        for key, value in gadget.cheapest_aux(losses, zeta).items():
            bits[registry.index(key)] = value
        return bits


def _quadratic(q: np.ndarray, states: np.ndarray) -> np.ndarray:
    return ((states @ q) * states).sum(axis=1)


def landscape(enc: EncodedInstance) -> Landscape:
    gadget, registry = enc.gadget, enc.registry
    inside = {registry.index(k) for k in gadget.keys}
    outside = np.array([i for i in range(len(registry)) if i not in inside])
    n = len(outside)
    states = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)

    costs = enc.breakdown.spread + enc.breakdown.temporary + enc.breakdown.permanent
    pick = np.ix_(outside, outside)
    hard = _quadratic(enc.penalties.Q[pick], states) + enc.penalties.constant
    flat = _quadratic(costs.Q[pick], states) + costs.constant + hard

    column = {int(i): j for j, i in enumerate(outside)}
    lots = np.zeros((len(states), enc.layout.n_assets, enc.layout.n_buckets))
    for plan in enc.layout.cells:
        lots[:, plan.asset, plan.bucket] += plan.value_form.constant
        for key, coef in plan.value_form.terms:
            bit = states[:, column[registry.index(key)]]
            lots[:, plan.asset, plan.bucket] += coef * bit

    position = np.asarray(enc.instance.initial_lots, dtype=float)[None, :, None]
    holdings = position - np.cumsum(lots, axis=2)
    losses = np.einsum("snt,bnt->bs", gadget.loss_coefficients, holdings)

    step, top = gadget.zeta.step, 2**gadget.zeta.n_bits - 1
    grid = gadget.zeta.lo + step * np.arange(top + 1)
    weights = np.asarray(gadget.weights, dtype=float)
    per_step = gadget.lam * weights / (1.0 - gadget.alpha) * step
    pull = per_step / (2.0 * gadget.penalty * step * step)

    unique, where = np.unique(losses.round(9), axis=0, return_inverse=True)
    where = where.ravel()
    target = unique[:, None, :] - grid[None, :, None]                 # (U, Z, S)
    above = np.clip(np.floor(target / step - pull + 0.5), 0, top)
    below = np.clip(np.floor(target / step + 0.5), -top, 0)
    charge = np.minimum(
        per_step * above + gadget.penalty * (above * step - target) ** 2,
        gadget.penalty * (below * step - target) ** 2,
    ).sum(axis=2) + gadget.lam * grid[None, :]

    return Landscape(
        encoded=enc, outside=outside, states=states.astype(np.int64), lots=lots,
        hard=hard, energy=flat + charge.min(axis=1)[where],
        zeta_level=charge.argmin(axis=1)[where],
    )
