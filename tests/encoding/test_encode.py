import itertools
import json

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.encode import encode, encode_solution, reference_breakdown
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import cvar_of, evaluate, optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule
from tests.encoding.landscape import landscape

BUCKET_NS = 1800 * 1_000_000_000
REL = 1e-9


def _params():
    return MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
            for j, s in enumerate(("A", "B"))
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )


def _instance(**kw) -> Instance:
    base = dict(
        instance_id="x", tier="T0",
        assets=tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B")),
        n_buckets=2, initial_lots=(4, 4), max_lots_per_bucket=4, params=_params(),
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def _cvar_instance():
    rng = np.random.default_rng(7)
    scenarios = ScenarioSet(
        returns=rng.normal(scale=0.01, size=(4, 2, 2)),
        weights=np.full(4, 0.25), seed=7,
    )
    return _instance(
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=1e-6, alpha=0.95, scenarios=scenarios),
    )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_energy_at_the_witness_equals_the_reference_objective(encoding):
    inst = _instance()
    enc = encode(inst, encoding)
    bits = encode_solution(enc, inst.witness)
    assert enc.qubo.energy(bits) == pytest.approx(
        reference_breakdown(enc, inst.witness).total, rel=REL
    )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_penalty_block_is_exactly_zero_at_the_witness(encoding):
    """Exactly zero, up to the float noise a dense Q-matrix accumulation leaves
    when a non-integer derived strength scales an algebraically-zero square --
    see tests/encoding/test_penalty.py for the full justification of the bound."""
    inst = _instance()
    enc = encode(inst, encoding)
    floor = 1e-9 * max(enc.strengths.full_liquidation, 1.0)
    assert abs(enc.penalties.energy(encode_solution(enc, inst.witness))) < floor


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_cvar_instance_encodes_and_scores_at_a_snapped_zeta(encoding):
    inst = _cvar_instance()
    enc = encode(inst, encoding)
    zeta = enc.gadget.zeta.snap(optimal_zeta(inst, inst.witness))
    bits = encode_solution(enc, inst.witness, zeta=zeta)
    losses = enc.gadget.losses_from(inst, enc.layout, inst.witness)
    aux = enc.gadget.optimal_aux(losses, zeta)
    continuous = inst.risk.lam * cvar_of(
        inst.risk.scenarios, inst, inst.witness, alpha=inst.risk.alpha, zeta=zeta
    )
    expected = (
        reference_breakdown(enc, inst.witness, zeta=zeta).total
        - continuous
        + enc.gadget.encoded_cvar(losses, zeta, aux)
    )
    assert enc.qubo.energy(bits) == pytest.approx(expected, rel=REL)


def test_encoding_a_cvar_instance_without_a_zeta_is_refused():
    enc = encode(_cvar_instance(), BinaryEncoding())
    with pytest.raises(ValueError, match="zeta"):
        encode_solution(enc, enc.instance.witness)


def test_passing_a_zeta_to_a_variance_instance_is_refused():
    enc = encode(_instance(), BinaryEncoding())
    with pytest.raises(ValueError, match="no zeta"):
        encode_solution(enc, enc.instance.witness, zeta=0.0)


def test_the_derived_strength_does_not_depend_on_its_own_penalty_block():
    """The two-pass derivation must not feed the CVaR penalty back into the bound."""
    inst = _cvar_instance()
    first = encode(inst, BinaryEncoding())
    second = encode(inst, BinaryEncoding())
    assert first.strengths == second.strengths
    assert first.gadget.penalty == first.strengths.cvar


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_encoding_is_deterministic(encoding):
    inst = _instance()
    a, b = encode(inst, encoding), encode(inst, encoding)
    assert np.array_equal(a.qubo.Q, b.qubo.Q)
    assert a.qubo.constant == b.qubo.constant
    assert a.registry.keys == b.registry.keys


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_registry_covers_every_bit_any_stage_named(encoding):
    enc = encode(_cvar_instance(), encoding)
    named = {k for plan in enc.layout.cells for k in plan.bit_keys}
    named |= set(enc.gadget.keys)
    assert named <= set(enc.registry.keys)
    assert enc.qubo.n_vars == len(enc.registry)


def test_the_description_is_json_safe_and_names_the_encoding():
    enc = encode(_instance(), BinaryEncoding())
    payload = json.loads(json.dumps(enc.describe()))
    assert payload["encoding"] == "binary"
    assert payload["n_vars"] == enc.n_vars
    assert payload["penalties"]["strategy"] == "derived_bounds"


def test_the_reference_breakdown_swaps_the_impact_term_when_approximated():
    import dataclasses

    from quantic.problem.objective import evaluate

    inst = _instance(dials=Dials(concave_impact=True))
    enc = encode(inst, BinaryEncoding())
    assert enc.breakdown.approximand is not None
    expected = dataclasses.replace(
        evaluate(inst, inst.witness),
        temporary_impact_cost=enc.breakdown.approximand.temporary_cost(inst.witness),
    )
    assert reference_breakdown(enc, inst.witness) == expected


def test_the_reference_breakdown_is_untouched_when_nothing_was_approximated():
    from quantic.problem.objective import evaluate

    inst = _instance()
    enc = encode(inst, BinaryEncoding())
    assert enc.breakdown.approximand is None
    assert reference_breakdown(enc, inst.witness) == evaluate(inst, inst.witness)


def _risk_that_matters(lam=0.01):
    """Three buckets, and twenty unevenly weighted scenarios with fat tails.

    ``alpha`` is 0.7, so the tail holds several scenarios and the excess
    ``u_s`` is positive in several of them at any schedule. With one scenario
    in the tail CVaR is the largest loss, every excess is zero at the optimal
    ``zeta``, and half of the gadget is never exercised.
    """
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.5, 1.5, size=20)
    scenarios = ScenarioSet(
        returns=rng.standard_t(df=3, size=(20, 3, 2)) * 0.01,
        weights=weights / weights.sum(), seed=7,
    )
    return _instance(
        n_buckets=3, dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=lam, alpha=0.7, scenarios=scenarios),
        witness=Schedule(lots=((2, 2, 0), (2, 2, 0))),
    )


def _every_feasible_schedule():
    rows = [r for r in itertools.product(range(5), repeat=3) if sum(r) == 4]
    return [Schedule(lots=(a, b)) for a in rows for b in rows]


def _canonical(enc, schedule):
    zeta = enc.gadget.zeta.snap(optimal_zeta(enc.instance, schedule))
    return encode_solution(enc, schedule, zeta=zeta)


def test_the_instance_these_tests_lean_on_has_a_tail_with_scenarios_in_it():
    inst = _risk_that_matters()
    enc = encode(inst, BinaryEncoding())
    in_the_tail = []
    for schedule in _every_feasible_schedule():
        bits = dict(zip(enc.registry.keys, _canonical(enc, schedule), strict=True))
        in_the_tail.append(sum(grid.decode(bits) > 0.0 for grid in enc.gadget.u))
    # Selling everything in the first bucket holds nothing and loses nothing,
    # so not every schedule can have a tail.
    assert np.mean(np.array(in_the_tail) >= 4) > 0.95


def test_with_cvar_on_the_encoding_resolves_the_objective_it_encodes():
    """The M3a gate's second finding, at a size that can be enumerated.

    The gadget's error is declared, and the declaration was honest -- but it
    was hundreds of times the whole range of the objective, so the energy
    ranked the 225 feasible schedules by where their losses fell on the grid.
    A declared error has to be smaller than what it is an error in.

    The energy is the lowest the QUBO can reach for each schedule, over
    ``zeta`` and the auxiliaries, not the energy of one chosen encoding of it.
    """
    inst = _risk_that_matters()
    enc = encode(inst, BinaryEncoding())
    scape = landscape(enc)
    lowest = {}
    for k in np.flatnonzero(scape.hard < 0.5):
        key = tuple(map(tuple, scape.lots[k].round().astype(int).tolist()))
        lowest[key] = min(lowest.get(key, np.inf), scape.energy[k])
    schedules = _every_feasible_schedule()
    assert {s.lots for s in schedules} == set(lowest)

    true = np.array(
        [evaluate(inst, s, zeta=optimal_zeta(inst, s)).total for s in schedules]
    )
    energy = np.array([lowest[s.lots] for s in schedules])
    slack = enc.gadget.resolution + enc.gadget.cost_gap

    assert np.abs(energy - true).max() <= slack
    assert slack < true.max() - true.min()
    assert true[int(np.argmin(energy))] - true.min() <= 2.0 * slack


def test_the_landscape_agrees_with_the_matrix():
    """The enumeration above is only as good as its closed form."""
    enc = encode(_risk_that_matters(), BinaryEncoding())
    scape = landscape(enc)
    rng = np.random.default_rng(5)
    for k in rng.choice(len(scape.energy), size=50, replace=False):
        assert enc.qubo.energy(scape.state(k)) == pytest.approx(
            scape.energy[k], rel=1e-9
        )


def test_the_cvar_strength_prices_one_grid_step_of_excess():
    """Design section 8: a strength is derived per constraint.

    Dropping ``u_s`` one step below a target on the grid saves ``lam * w_s *
    step / (1 - alpha)`` and costs ``P * step**2``, so that is the trade the
    strength has to win for the heaviest scenario -- by the safety factor, and
    by no more.
    """
    inst = _risk_that_matters()
    enc = encode(inst, BinaryEncoding(), safety_factor=3.0)
    step = enc.gadget.zeta.step
    heaviest = float(np.max(inst.risk.scenarios.weights))
    saved = inst.risk.lam * heaviest * step / (1.0 - inst.risk.alpha)
    assert enc.strengths.cvar * step * step == pytest.approx(3.0 * saved, rel=1e-12)
    assert enc.gadget.penalty == enc.strengths.cvar


def test_a_cvar_instance_with_no_risk_aversion_carries_no_cvar_penalty():
    """``lam == 0`` leaves the gadget nothing to track, so it charges nothing."""
    enc = encode(_risk_that_matters(lam=0.0), BinaryEncoding())
    assert enc.strengths.cvar == 0.0
    assert enc.gadget.resolution == 0.0


def test_no_single_cell_buys_more_risk_than_the_gadget_says_it_can():
    """The bound the hard strengths are built on, checked against every pair.

    Schedules on the lot grid, feasible or not, against every schedule that
    differs from each in one cell.
    """
    inst = _risk_that_matters(lam=1.0)
    enc = encode(inst, BinaryEncoding())
    bound = enc.gadget.risk_gain_bound(enc.layout)
    grids = list(itertools.product(range(5), repeat=6))
    rng = np.random.default_rng(0)

    def risk(grid):
        return enc.breakdown.risk.energy(_canonical(enc, Schedule.from_array(grid)))

    worst = 0.0
    for pick in rng.choice(len(grids), size=300, replace=False):
        base = np.array(grids[pick]).reshape(2, 3)
        here = risk(base)
        for i, t in itertools.product(range(2), range(3)):
            for value in range(5):
                if value == base[i, t]:
                    continue
                other = base.copy()
                other[i, t] = value
                worst = max(worst, abs(risk(other) - here))
    assert 0.0 < worst <= bound


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_with_cvar_on_the_hard_strengths_move_with_risk_aversion(encoding):
    """The M3a gate's last finding. Derived from an objective with the CVaR
    penalty at zero, the hard strengths bounded the gain in spread and impact
    and said nothing about risk -- so they did not move with ``lam`` at all."""
    mild = encode(_risk_that_matters(lam=0.01), encoding)
    averse = encode(_risk_that_matters(lam=1.0), encoding)
    assert averse.strengths.full_liquidation > 10.0 * mild.strengths.full_liquidation
    for enc in (mild, averse):
        assert enc.strengths.validity == enc.strengths.full_liquidation
        assert enc.strengths.cardinality == enc.strengths.full_liquidation


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_with_cvar_on_no_unsold_lot_is_cheaper_than_selling_it(encoding):
    """The property the gate checks at one cell of one schedule, checked here at
    every cell of every feasible schedule."""
    inst = _risk_that_matters(lam=1.0)
    enc = encode(inst, encoding)

    def energy(schedule):
        return enc.qubo.energy(_canonical(enc, schedule))

    cheaper = []
    for schedule in _every_feasible_schedule():
        full = energy(schedule)
        grid = schedule.as_array()
        for i, t in np.argwhere(grid > 0):
            short = grid.copy()
            short[i, t] -= 1
            if energy(Schedule.from_array(short)) <= full:
                cheaper.append((schedule.lots, (int(i), int(t))))
    assert not cheaper, cheaper[:5]


def _one_name_a_bucket(lam=1.0):
    """Two names, two buckets, one name a bucket -- and a tail scenario in which
    both fall six percent in the first bucket and recover three in the second.

    Selling both names at once breaches cardinality by one unit and escapes
    the fall in both. Against the feasible schedule that holds one of them
    through it, that is a move of lots between two buckets, and what it buys
    in risk is the first bucket's coefficient alone: twice the largest tail
    sum, because the recovery that the tail sum nets off is not given up.
    """
    returns = np.random.default_rng(9).normal(scale=0.0005, size=(6, 2, 2))
    returns[0, :, 0] = (-0.06, 0.03)
    returns[0, :, 1] = (-0.06, 0.03)
    scenarios = ScenarioSet(returns=returns, weights=np.full(6, 1.0 / 6), seed=0)
    return _instance(
        n_buckets=2, initial_lots=(3, 3), max_lots_per_bucket=3,
        dials=Dials(cvar_risk=True, discrete_participation=True),
        constraints=(
            FullLiquidation(), Cardinality(k=1), MinParticipation(min_lots=(1, 1)),
        ),
        risk=CVaRRisk(lam=lam, alpha=0.9, scenarios=scenarios),
        witness=Schedule(lots=((3, 0), (0, 3))),
    )


@pytest.mark.parametrize("safety_factor", [2.0, 1.5, 1.2, 1.05])
@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_lowest_energy_in_the_whole_landscape_is_a_feasible_schedule(
    encoding, safety_factor
):
    """Found by the second review. A bound on what one cell can buy covered a
    breach that moves lots between two only while the safety factor was two."""
    enc = encode(_one_name_a_bucket(), encoding, safety_factor=safety_factor)
    scape = landscape(enc)
    feasible = scape.hard < 0.5
    best = int(np.flatnonzero(feasible)[scape.energy[feasible].argmin()])
    rival = int(np.flatnonzero(~feasible)[scape.energy[~feasible].argmin()])
    for k in (best, rival):
        assert enc.qubo.energy(scape.state(k)) == pytest.approx(
            scape.energy[k], rel=1e-9
        )
    assert scape.energy[best] < scape.energy[rival], (
        scape.lots[best].tolist(), scape.lots[rival].tolist()
    )
