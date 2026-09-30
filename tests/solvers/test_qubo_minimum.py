import dataclasses
import itertools

import numpy as np
import pytest

import quantic.encoding.encode as encode_module
from quantic.encoding.decode import FEASIBLE, REJECTED, decode
from quantic.encoding.encode import encode
from quantic.encoding.integer import ENCODINGS, BinaryEncoding, OneHotEncoding
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.classical import gadget_minimum as gadget_module
from quantic.solvers.classical import qubo_minimum as module
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.qubo_minimum import (
    DECODABLE,
    EVERY_STATE,
    EXHAUSTIVE_BIT_LIMIT,
    qubo_minimum,
)
from quantic.solvers.classical.scoring import score
from tests.encoding.landscape import landscape
from tests.solvers.instances import (
    blocks,
    concave,
    cvar,
    cvar_without_risk_aversion,
    discrete,
    instance,
    no_risk_aversion,
    nothing_to_sell_in_one_name,
    plain,
    scenarios,
)

REL = 1e-9


def _tiny(**kw):
    """Two names, two buckets, two lots a bucket: few enough bits to list."""
    base = dict(
        n_buckets=2, initial_lots=(2, 3), max_lots_per_bucket=2,
        witness=Schedule(lots=((1, 1), (1, 2))),
    )
    base.update(kw)
    return instance(**base)


def _tiny_cvar():
    return _tiny(
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=0.02, alpha=0.7, scenarios=scenarios(n_buckets=2)),
    )


def _tiny_cvar_discrete():
    return _tiny(
        initial_lots=(2, 2),
        dials=Dials(cvar_risk=True, discrete_participation=True),
        constraints=(
            FullLiquidation(), Cardinality(k=1), MinParticipation(min_lots=(2, 1)),
        ),
        risk=CVaRRisk(lam=0.02, alpha=0.7, scenarios=scenarios(n_buckets=2)),
        witness=Schedule(lots=((2, 0), (0, 2))),
    )


def _every_energy(enc):
    """Every state's energy, one ``Qubo.energy`` call at a time."""
    qubo = enc.qubo
    return {
        bits: qubo.energy(np.array(bits))
        for bits in itertools.product((0, 1), repeat=enc.n_vars)
    }


# -- without a gadget -------------------------------------------------------


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize("dials", [Dials(), Dials(concave_impact=True)], ids=["ac", "d1"])
def test_it_finds_the_lowest_energy_there_is(encoding, dials):
    enc = encode(_tiny(dials=dials), encoding)
    assert enc.n_vars <= 16
    energies = _every_energy(enc)
    found = qubo_minimum(enc)
    assert found.scope == EVERY_STATE
    assert found.n_states == 2**enc.n_vars
    assert found.energy == pytest.approx(min(energies.values()), rel=REL)
    assert enc.qubo.energy(found.bits) == pytest.approx(found.energy, rel=REL)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_it_counts_the_states_that_reach_the_minimum(encoding):
    enc = encode(_tiny(), encoding)
    energies = np.array(list(_every_energy(enc).values()))
    lowest = energies.min()
    expected = int(np.sum(energies <= lowest + 1e-9 * max(1.0, abs(lowest))))
    assert qubo_minimum(enc).n_ties == expected


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize(
    "build",
    [plain, concave, discrete, blocks, no_risk_aversion, nothing_to_sell_in_one_name],
    ids=lambda b: b.__name__,
)
def test_the_minimum_decodes_to_the_schedule_enumeration_finds(build, encoding):
    """The gate's claim, at a size that runs with the unit tests."""
    inst = build()
    enc = encode(inst, encoding)
    found = qubo_minimum(enc)
    sample = decode(enc, found.bits)
    assert sample.classification == FEASIBLE
    assert score(inst, sample.schedule)[0].total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )
    assert enc.qubo.energy(found.bits) == pytest.approx(found.energy, rel=REL)


def test_above_the_limit_only_the_states_that_decode_are_examined(monkeypatch):
    enc = encode(_tiny(), OneHotEncoding())
    monkeypatch.setattr(module, "EXHAUSTIVE_BIT_LIMIT", enc.n_vars - 1)
    found = qubo_minimum(enc)

    decodable = {
        bits: energy for bits, energy in _every_energy(enc).items()
        if decode(enc, np.array(bits)).classification != REJECTED
    }
    assert found.scope == DECODABLE
    assert found.n_states == len(decodable)
    assert found.n_states < 2**enc.n_vars
    assert found.energy == pytest.approx(min(decodable.values()), rel=REL)


def test_the_states_that_decode_include_the_infeasible_ones(monkeypatch):
    """A schedule that breaks a constraint is a state; a broken one-hot is not."""
    enc = encode(_tiny(), OneHotEncoding())
    monkeypatch.setattr(module, "EXHAUSTIVE_BIT_LIMIT", 0)
    found = qubo_minimum(enc)
    per_cell = [len(plan.values) for plan in enc.layout.cells]
    assert found.n_states == int(np.prod(per_cell))
    assert found.n_states > brute_force(_tiny()).detail["n_feasible"]


def test_the_slack_bits_are_enumerated_with_the_states_that_decode(monkeypatch):
    inst = discrete()
    enc = encode(inst, OneHotEncoding())
    monkeypatch.setattr(module, "EXHAUSTIVE_BIT_LIMIT", 0)
    found = qubo_minimum(enc)
    n_slack = sum(1 for key in enc.registry.keys if key[0] == "card")
    per_cell = [len(plan.values) for plan in enc.layout.cells]
    assert n_slack > 0
    assert found.n_states == int(np.prod(per_cell)) * 2**n_slack
    assert decode(enc, found.bits).classification == FEASIBLE
    assert score(inst, decode(enc, found.bits).schedule)[0].total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )


def test_the_limit_is_the_one_the_design_names():
    assert EXHAUSTIVE_BIT_LIMIT == 28


def test_penalties_too_weak_to_bind_show_up_as_an_infeasible_minimum(monkeypatch):
    """The mutation the gate is asked to catch. The minimiser reports the
    QUBO's minimum, not the best feasible state, so a QUBO whose penalties do
    not bind is seen to have an infeasible one."""
    real = encode_module.derive_penalties

    def weak(*args, **kwargs):
        strengths = real(*args, **kwargs)
        return dataclasses.replace(
            strengths,
            validity=strengths.validity * 1e-4,
            full_liquidation=strengths.full_liquidation * 1e-4,
        )

    monkeypatch.setattr(encode_module, "derive_penalties", weak)
    enc = encode(plain(), BinaryEncoding())
    assert decode(enc, qubo_minimum(enc).bits).classification != FEASIBLE


# -- with a gadget ----------------------------------------------------------


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize(
    "build", [_tiny_cvar, _tiny_cvar_discrete], ids=lambda b: b.__name__
)
def test_with_a_gadget_it_finds_what_the_whole_landscape_holds(build, encoding):
    """Against tests/encoding/landscape.py, which scans every zeta level of
    every state and prunes nothing."""
    enc = encode(build(), encoding)
    scape = landscape(enc)
    found = qubo_minimum(enc)
    assert found.energy == pytest.approx(float(scape.energy.min()), rel=REL)
    assert enc.qubo.energy(found.bits) == pytest.approx(found.energy, rel=REL)
    assert found.n_states == len(scape.energy)
    assert 0 < found.n_scanned <= found.n_states


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_window_that_starts_too_narrow_grows_until_it_is_wide_enough(
    encoding, monkeypatch
):
    enc = encode(cvar(), encoding)
    wide = qubo_minimum(enc)
    monkeypatch.setattr(gadget_module, "_FIRST_WINDOW", 1)
    narrow = qubo_minimum(enc)
    assert narrow.energy == pytest.approx(wide.energy, rel=1e-12)
    assert narrow.widest_window > 3


@pytest.mark.parametrize("n_zeta_bits", [4, 8, 12, 20])
def test_the_winning_state_has_the_energy_claimed_at_any_width(n_zeta_bits):
    """The closed form against the matrix, at the width the gate uses too."""
    enc = encode(cvar(), BinaryEncoding(), n_zeta_bits=n_zeta_bits)
    found = qubo_minimum(enc)
    assert enc.qubo.energy(found.bits) == pytest.approx(found.energy, rel=1e-6)
    assert decode(enc, found.bits).classification == FEASIBLE


def test_at_a_wide_grid_the_minimum_is_the_schedule_enumeration_finds():
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=20)
    sample = decode(enc, qubo_minimum(enc).bits)
    assert score(inst, sample.schedule)[0].total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )


def test_the_regret_at_a_coarse_grid_is_inside_what_the_gadget_declares():
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=5)
    sample = decode(enc, qubo_minimum(enc).bits)
    regret = score(inst, sample.schedule)[0].total - brute_force(inst).objective.total
    assert 0.0 <= regret <= 2.0 * enc.gadget.resolution + enc.gadget.cost_gap


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_gadget_with_no_risk_aversion_charges_nothing(encoding):
    inst = cvar_without_risk_aversion()
    enc = encode(inst, encoding)
    assert enc.gadget.penalty == 0.0
    found = qubo_minimum(enc)
    sample = decode(enc, found.bits)
    assert sample.classification == FEASIBLE
    assert enc.qubo.energy(found.bits) == pytest.approx(found.energy, rel=REL)
    assert score(inst, sample.schedule)[0].total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )
