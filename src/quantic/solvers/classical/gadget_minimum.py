"""The CVaR gadget's charge for a state, minimised in closed form.

Once the lot bits are fixed the losses are fixed. Once ``zeta`` is fixed too
the scenarios separate, and each scenario's auxiliaries have a closed form --
``CVaRGadget.cheapest_aux``. What is left is a minimum over ``zeta``'s grid,
and at twenty bits that grid has a million levels.

This module does that minimum for many states at once. The closed form is
vectorised, and so is a second implementation of ``cheapest_aux``; the tests
compare the two at every schedule of an instance, and ``qubo_minimum``
rebuilds its winning state through the production method.
"""

from __future__ import annotations

import numpy as np

from quantic.encoding.encode import EncodedInstance

_SCAN_CHUNK = 256
_FIRST_WINDOW = 64


class GadgetMinimiser:
    """The gadget's charge for a state, minimised over ``zeta`` and the auxiliaries."""

    def __init__(self, enc: EncodedInstance) -> None:
        gadget = enc.gadget
        self.gadget = gadget
        self.lam = float(gadget.lam)
        self.step = float(gadget.zeta.step)
        self.lo = float(gadget.zeta.lo)
        self.top = 2**gadget.zeta.n_bits - 1
        self.penalty = float(gadget.penalty)
        self.weights = np.asarray(gadget.weights, dtype=float)
        self.slope = self.lam * self.weights / (1.0 - float(gadget.alpha))
        self.per_step = self.slope * self.step
        self.alpha = float(gadget.alpha)
        if self.penalty > 0.0:
            self.pull = self.per_step / (2.0 * self.penalty * self.step * self.step)
            # The charge is never below lam * Rockafellar-Uryasev less this:
            # relax each scenario's step count to a real and minimise.
            self.slack = float(
                np.sum(self.per_step**2) / (4.0 * self.penalty * self.step**2)
            )
        else:
            self.pull = np.zeros_like(self.per_step)
            self.slack = 0.0

        # L_s = base_s + sum over bits of loss_matrix[s, bit] * bit.
        registry = enc.registry
        coefficients = np.asarray(gadget.loss_coefficients, dtype=float)   # (S, N, T)
        tail = np.cumsum(coefficients[:, :, ::-1], axis=2)[:, :, ::-1]
        position = np.asarray(enc.instance.initial_lots, dtype=float)
        self.base = np.einsum("snt,n->s", coefficients, position)
        self.loss_matrix = np.zeros((coefficients.shape[0], len(registry)))
        for plan in enc.layout.cells:
            reach = tail[:, plan.asset, plan.bucket]
            self.base = self.base - plan.value_form.constant * reach
            for key, coef in plan.value_form.folded().terms:
                self.loss_matrix[:, registry.index(key)] -= coef * reach

    def continuous(self, losses: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """``lam`` times Rockafellar-Uryasev at each level. ``(B, W)``."""
        zeta = self.lo + self.step * levels
        excess = np.maximum(losses[:, None, :] - zeta[:, :, None], 0.0)
        return self.lam * zeta + (excess * self.slope[None, None, :]).sum(axis=2)

    def charge(self, losses: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """The gadget's exact energy at each level, auxiliaries at their best."""
        zeta = self.lo + self.step * levels
        if self.penalty == 0.0:
            return self.lam * zeta
        target = losses[:, None, :] - zeta[:, :, None]                 # (B, W, S)
        above = np.clip(np.floor(target / self.step - self.pull + 0.5), 0, self.top)
        below = np.clip(np.floor(target / self.step + 0.5), -self.top, 0)
        best = np.minimum(
            self.per_step * above + self.penalty * (above * self.step - target) ** 2,
            self.penalty * (below * self.step - target) ** 2,
        )
        return best.sum(axis=2) + self.lam * zeta

    def centre(self, losses: np.ndarray) -> np.ndarray:
        """The grid level nearest the weighted quantile, where the scan starts."""
        order = np.argsort(losses, axis=1)
        ranked = np.take_along_axis(losses, order, axis=1)
        cumulative = np.cumsum(self.weights[order], axis=1)
        at = np.minimum((cumulative < self.alpha).sum(axis=1), losses.shape[1] - 1)
        quantile = ranked[np.arange(losses.shape[0]), at]
        return np.clip(np.round((quantile - self.lo) / self.step), 0, self.top)

    def minimise(self, losses: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """The least charge for each state, the level that gives it, the widest scan.

        A window of levels is scanned exactly. It is wide enough once the
        bound at both of its edges is above the best charge inside it:
        Rockafellar-Uryasev is convex in the level, so beyond an edge where it
        is rising the bound only rises further. A state whose window is not
        wide enough is scanned again at twice the width.
        """
        n = losses.shape[0]
        energy = np.full(n, np.inf)
        level = np.zeros(n)
        if self.penalty == 0.0:
            # No excess is tracked, so zeta only costs lam * zeta -- and lam
            # is zero whenever the penalty is.
            return self.charge(losses, np.zeros((n, 1)))[:, 0], level, 1

        centre = self.centre(losses)
        pending = np.arange(n)
        window = _FIRST_WINDOW
        widest = 0
        while pending.size:
            widest = max(widest, 2 * window + 1)
            offsets = np.arange(-window, window + 1, dtype=float)
            for start in range(0, pending.size, _SCAN_CHUNK):
                rows = pending[start:start + _SCAN_CHUNK]
                levels = np.clip(centre[rows, None] + offsets[None, :], 0, self.top)
                charges = self.charge(losses[rows], levels)
                at = charges.argmin(axis=1)
                energy[rows] = charges[np.arange(rows.size), at]
                level[rows] = levels[np.arange(rows.size), at]
            edges = np.stack(
                [
                    np.clip(centre[pending] - window, 0, self.top),
                    centre[pending],
                    np.clip(centre[pending] + window, 0, self.top),
                ],
                axis=1,
            )
            bound = self.continuous(losses[pending], edges)
            need = energy[pending] + self.slack
            left_ok = (edges[:, 0] == 0) | (
                (bound[:, 0] > need) & (bound[:, 0] > bound[:, 1])
            )
            right_ok = (edges[:, 2] == self.top) | (
                (bound[:, 2] > need) & (bound[:, 2] > bound[:, 1])
            )
            pending = pending[~(left_ok & right_ok)]
            window *= 2
        return energy, level, widest
