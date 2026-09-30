import numpy as np
import pytest

from quantic.encoding.encode import encode, encode_solution
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.solvers.classical import gadget_minimum as module
from quantic.solvers.classical.gadget_minimum import GadgetMinimiser
from quantic.solvers.classical.lot_grid import lot_grid
from tests.solvers.instances import cvar, cvar_discrete, cvar_without_risk_aversion

REL = 1e-9


def _by_the_production_method(enc, schedules):
    """Each schedule's least charge: every zeta level, ``cheapest_aux`` at each."""
    gadget = enc.gadget
    out = []
    for schedule in schedules:
        losses = gadget.losses_from(enc.instance, enc.layout, schedule)
        charges = []
        for level in range(2**gadget.zeta.n_bits):
            zeta = gadget.zeta.lo + gadget.zeta.step * level
            aux = gadget.cheapest_aux(losses, zeta)
            charges.append(gadget.encoded_cvar(losses, zeta, aux))
        out.append(min(charges))
    return np.array(out)


def _losses(enc, schedules):
    return np.array(
        [enc.gadget.losses_from(enc.instance, enc.layout, s) for s in schedules]
    )


@pytest.mark.parametrize("n_zeta_bits", [3, 5, 7])
def test_the_closed_form_is_the_production_methods_minimum_state_by_state(n_zeta_bits):
    """The winner alone says little: the closed form ranks every state, and a
    closed form that overcharged the losers would still crown the winner."""
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=n_zeta_bits)
    schedules = list(lot_grid(inst).schedules())
    charge, level, _ = GadgetMinimiser(enc).minimise(_losses(enc, schedules))
    assert charge == pytest.approx(_by_the_production_method(enc, schedules), rel=REL)
    assert np.all((level >= 0) & (level <= 2**n_zeta_bits - 1))


def test_the_loss_matrix_prices_the_losses_the_gadget_prices():
    inst = cvar_discrete()
    for encoding in ENCODINGS:
        enc = encode(inst, encoding)
        mine = GadgetMinimiser(enc)
        for schedule in lot_grid(inst).schedules():
            bits = encode_solution(enc, schedule, zeta=enc.gadget.zeta.lo).astype(float)
            priced = mine.base + bits @ mine.loss_matrix.T
            assert priced == pytest.approx(
                enc.gadget.losses_from(inst, enc.layout, schedule), rel=REL, abs=1e-9
            )


def test_the_auxiliaries_are_the_cheapest_and_not_the_canonical(monkeypatch):
    """M3a gate result, second review, I2. With ``optimal_aux``'s rule the
    closed form charges some states more than the QUBO's own minimum does."""
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=5)
    schedules = list(lot_grid(inst).schedules())
    losses = _losses(enc, schedules)
    production = _by_the_production_method(enc, schedules)

    real = GadgetMinimiser.__init__

    def canonical(self, encoded):
        real(self, encoded)
        self.pull = np.zeros_like(self.pull)

    monkeypatch.setattr(GadgetMinimiser, "__init__", canonical)
    charge, _, _ = GadgetMinimiser(enc).minimise(losses)
    assert np.all(charge >= production - 1e-9)
    assert np.any(charge > production + 1e-6)


@pytest.mark.parametrize("shift", [-3.0, -1.0, 1.0, 3.0])
def test_losses_outside_the_grid_widen_the_scan_and_do_not_break_it(shift):
    """An oversold state's losses need not lie where ``zeta`` can reach.

    The grid is built to hold every loss a schedule inside the position can
    produce. The enumeration also visits states that are not such schedules.
    """
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=5)
    gadget = enc.gadget
    span = gadget.zeta.step * (2**gadget.zeta.n_bits - 1)
    schedules = list(lot_grid(inst).schedules())[:12]
    losses = _losses(enc, schedules) + shift * span

    expected = []
    for row in losses:
        charges = []
        for level in range(2**gadget.zeta.n_bits):
            zeta = gadget.zeta.lo + gadget.zeta.step * level
            charges.append(gadget.encoded_cvar(row, zeta, gadget.cheapest_aux(row, zeta)))
        expected.append(min(charges))

    charge, _, _ = GadgetMinimiser(enc).minimise(losses)
    assert charge == pytest.approx(np.array(expected), rel=REL)


def test_a_window_that_starts_at_one_level_reaches_the_same_minimum(monkeypatch):
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=8)
    losses = _losses(enc, list(lot_grid(inst).schedules()))
    wide, _, first = GadgetMinimiser(enc).minimise(losses)
    monkeypatch.setattr(module, "_FIRST_WINDOW", 1)
    narrow, _, grown = GadgetMinimiser(enc).minimise(losses)
    assert narrow == pytest.approx(wide, rel=1e-12)
    assert grown > 3
    assert first == 2 * 64 + 1


def test_with_no_risk_aversion_the_charge_is_nothing():
    inst = cvar_without_risk_aversion()
    enc = encode(inst, BinaryEncoding())
    losses = _losses(enc, list(lot_grid(inst).schedules()))
    charge, level, widest = GadgetMinimiser(enc).minimise(losses)
    assert np.all(charge == 0.0)
    assert np.all(level == 0.0)
    assert widest == 1


def test_the_bound_is_never_above_the_charge():
    """``lam`` times Rockafellar-Uryasev, less the slack, at every level."""
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=6)
    mine = GadgetMinimiser(enc)
    losses = _losses(enc, list(lot_grid(inst).schedules()))
    levels = np.tile(np.arange(2**6, dtype=float), (losses.shape[0], 1))
    assert np.all(
        mine.charge(losses, levels) >= mine.continuous(losses, levels) - mine.slack - 1e-9
    )
