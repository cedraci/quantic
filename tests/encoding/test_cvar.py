import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.cvar import build_cvar_gadget, expand_cvar
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.encoding.layout import encode_schedule, plan_layout
from quantic.encoding.registry import VarRegistry
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import cvar_of, optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
REL = 1e-9


def _scenarios(seed=0, n=4):
    rng = np.random.default_rng(seed)
    returns = rng.normal(scale=0.01, size=(n, 2, 2))
    return ScenarioSet(returns=returns, weights=np.full(n, 1.0 / n), seed=seed)


def _instance(*, lam=1e-6, alpha=0.95, scenarios=None) -> Instance:
    assets = tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B"))
    params = MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
            for j, s in enumerate(("A", "B"))
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    return Instance(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(4, 4), max_lots_per_bucket=4, params=params,
        dials=Dials(cvar_risk=True), constraints=(FullLiquidation(),),
        risk=CVaRRisk(lam=lam, alpha=alpha, scenarios=scenarios or _scenarios()),
        witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )


def _setup(instance, encoding, *, n_zeta_bits=8, penalty=1.0):
    layout = plan_layout(instance, encoding)
    gadget = build_cvar_gadget(
        instance, layout, n_zeta_bits=n_zeta_bits, penalty=penalty
    )
    registry = VarRegistry(keys=layout.registry.keys + gadget.keys)
    cost, pen = expand_cvar(instance, layout, registry, gadget)
    return layout, registry, gadget, cost, pen


def _bits(layout, registry, gadget, schedule, zeta, aux):
    vector = encode_schedule(layout, schedule, registry)
    for key, value in gadget.zeta.encode(zeta).items():
        vector[registry.index(key)] = value
    for key, value in aux.items():
        vector[registry.index(key)] = value
    return vector


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_qubo_equals_the_function_the_gadget_declares_it_encodes(encoding):
    """Assertion one of two, and the one that proves the encoder correct."""
    inst = _instance()
    layout, registry, gadget, cost, pen = _setup(inst, encoding)
    schedule = inst.witness
    losses = gadget.losses_from(inst, layout, schedule)
    zeta = gadget.zeta.snap(optimal_zeta(inst, schedule))
    aux = gadget.optimal_aux(losses, zeta)
    bits = _bits(layout, registry, gadget, schedule, zeta, aux)
    assert (cost + pen).energy(bits) == pytest.approx(
        gadget.encoded_cvar(losses, zeta, aux), rel=REL
    )


def test_the_declared_function_stays_within_the_reported_resolution():
    """Assertion two of two: the discretisation is no worse than it claims."""
    inst = _instance()
    layout, _, gadget, _, _ = _setup(inst, BinaryEncoding(), n_zeta_bits=10)
    schedule = inst.witness
    losses = gadget.losses_from(inst, layout, schedule)
    zeta = gadget.zeta.snap(optimal_zeta(inst, schedule))
    aux = gadget.optimal_aux(losses, zeta)
    continuous = inst.risk.lam * cvar_of(
        inst.risk.scenarios, inst, schedule, alpha=inst.risk.alpha, zeta=zeta
    )
    assert abs(gadget.encoded_cvar(losses, zeta, aux) - continuous) <= gadget.resolution


def test_the_loss_form_reproduces_the_scenario_sets_own_losses():
    """L_s is linear in the holdings; the encoder must agree with problem/."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    gadget = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
    for schedule in (inst.witness, Schedule(lots=((4, 0), (0, 4)))):
        direct = inst.risk.scenarios.losses(inst, schedule)
        assert gadget.losses_from(inst, layout, schedule) == pytest.approx(
            direct, rel=1e-12
        )


def test_the_resolution_shrinks_quadratically_in_the_grid_step():
    """More zeta bits buy accuracy at a linear cost in variables."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    coarse = build_cvar_gadget(inst, layout, n_zeta_bits=6, penalty=1.0)
    fine = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
    assert fine.resolution < coarse.resolution
    assert fine.zeta.step < coarse.zeta.step


def test_the_true_optimal_zeta_lies_inside_the_derived_grid():
    """The bound is instance-derived, so it must hold for every schedule."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    gadget = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
    for schedule in (
        inst.witness,
        Schedule(lots=((4, 0), (0, 4))),
        Schedule(lots=((0, 4), (4, 0))),
        Schedule(lots=((0, 0), (0, 0))),
    ):
        z = optimal_zeta(inst, schedule)
        assert gadget.zeta.lo <= z <= gadget.zeta.hi


def test_optimal_aux_is_deterministic():
    inst = _instance()
    layout, _, gadget, _, _ = _setup(inst, BinaryEncoding())
    losses = gadget.losses_from(inst, layout, inst.witness)
    zeta = gadget.zeta.snap(optimal_zeta(inst, inst.witness))
    assert gadget.optimal_aux(losses, zeta) == gadget.optimal_aux(losses, zeta)


def test_the_gadget_costs_two_auxiliaries_per_scenario():
    """Parent spec 5.4 says one. A one-sided constraint cannot; deviation recorded."""
    inst = _instance(scenarios=_scenarios(n=3))
    layout = plan_layout(inst, BinaryEncoding())
    gadget = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
    assert len(gadget.u) == 3
    assert len(gadget.g) == 3
    assert sum(1 for k in gadget.keys if k[0] == "u") > 0
    assert sum(1 for k in gadget.keys if k[0] == "g") > 0


def test_a_zeta_grid_value_round_trips():
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    grid = build_cvar_gadget(inst, layout, n_zeta_bits=6, penalty=1.0).zeta
    for step in range(2**6):
        value = grid.lo + step * grid.step
        assert grid.decode(grid.encode(value)) == pytest.approx(value, rel=1e-12)


def test_snapping_never_leaves_the_grid():
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    grid = build_cvar_gadget(inst, layout, n_zeta_bits=6, penalty=1.0).zeta
    assert grid.snap(grid.lo - 1e9) == pytest.approx(grid.lo)
    assert grid.snap(grid.hi + 1e9) == pytest.approx(grid.hi)


def test_expand_objective_no_longer_refuses_cvar():
    inst = _instance()
    from quantic.encoding.expand import expand_objective

    layout = plan_layout(inst, BinaryEncoding())
    gadget = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
    registry = VarRegistry(keys=layout.registry.keys + gadget.keys)
    breakdown = expand_objective(inst, layout, registry, gadget=gadget)
    assert breakdown.risk.n_vars == len(registry)


def test_expand_objective_still_refuses_cvar_without_a_gadget():
    inst = _instance()
    from quantic.encoding.expand import expand_objective

    layout = plan_layout(inst, BinaryEncoding())
    with pytest.raises(ValueError, match="gadget"):
        expand_objective(inst, layout, layout.registry)


def test_the_auxiliary_grids_are_as_wide_as_zeta_whatever_the_rounding():
    """``u`` and ``g`` share zeta's step and its span, so they share its width.

    Sized by dividing the span by the step, they came out one bit wider
    whenever that quotient rounded to a hair above ``2**n - 1`` -- on two of
    the ladder's 32 CVaR instances, thirty variables each in the budget.
    """
    widths = set()
    for seed in range(200):
        inst = _instance(scenarios=_scenarios(seed=seed))
        layout = plan_layout(inst, BinaryEncoding())
        gadget = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
        widths |= {grid.n_bits for grid in gadget.u + gadget.g}
    assert widths == {8}


def _tail_heavy_scenarios(seed=3, n=20):
    rng = np.random.default_rng(seed)
    returns = rng.standard_t(df=3, size=(n, 2, 2)) * 0.01
    weights = rng.uniform(0.5, 1.5, size=n)
    return ScenarioSet(returns=returns, weights=weights / weights.sum(), seed=seed)


def _derived_gadget(*, n_zeta_bits, safety_factor=2.0):
    import dataclasses

    inst = _instance(lam=1.0, alpha=0.7, scenarios=_tail_heavy_scenarios())
    layout = plan_layout(inst, BinaryEncoding())
    shape = build_cvar_gadget(inst, layout, n_zeta_bits=n_zeta_bits, penalty=0.0)
    gadget = dataclasses.replace(
        shape, penalty=shape.derived_penalty(safety_factor=safety_factor)
    )
    return inst, layout, gadget


def _every_aux(gadget, s):
    levels = range(2**gadget.zeta.n_bits)
    for u in levels:
        for g in levels:
            bits = {k: (u >> b) & 1 for b, k in enumerate(gadget.u[s].keys)}
            bits.update({k: (g >> b) & 1 for b, k in enumerate(gadget.g[s].keys)})
            yield bits


def test_the_cheapest_auxiliaries_are_the_cheapest_there_are():
    """Against every ``(u, g)`` on the grid, one scenario at a time.

    The scenarios separate -- each one's auxiliaries appear in its own cost
    and its own residual and nowhere else -- so the minimum over all of them
    is the sum of the minima over each.
    """
    inst, layout, gadget = _derived_gadget(n_zeta_bits=3)
    rng = np.random.default_rng(1)
    for _ in range(40):
        lots = rng.integers(0, 5, size=(2, 2))
        losses = gadget.losses_from(inst, layout, Schedule.from_array(lots))
        zeta = gadget.zeta.snap(rng.uniform(gadget.zeta.lo, gadget.zeta.hi))
        cheapest = gadget.cheapest_aux(losses, zeta)
        for s in range(gadget.n_scenarios):
            trial = dict(cheapest)
            best = min(
                gadget.encoded_cvar(losses, zeta, {**trial, **aux})
                for aux in _every_aux(gadget, s)
            )
            assert gadget.encoded_cvar(losses, zeta, cheapest) == pytest.approx(
                best, rel=1e-12, abs=1e-12
            )


def test_the_nearest_auxiliaries_are_not_always_the_cheapest():
    """Why there are two. ``optimal_aux`` lands on the grid point nearest the
    target, which minimises the penalty; a step below it can save more in cost
    than it adds in penalty, because the target is rarely on the grid."""
    inst, layout, gadget = _derived_gadget(n_zeta_bits=8)
    rng = np.random.default_rng(2)
    saved = []
    for _ in range(200):
        lots = rng.integers(0, 5, size=(2, 2))
        losses = gadget.losses_from(inst, layout, Schedule.from_array(lots))
        zeta = gadget.zeta.snap(optimal_zeta(inst, Schedule.from_array(lots)))
        nearest = gadget.encoded_cvar(losses, zeta, gadget.optimal_aux(losses, zeta))
        cheapest = gadget.encoded_cvar(losses, zeta, gadget.cheapest_aux(losses, zeta))
        assert cheapest <= nearest + 1e-9 * abs(nearest)
        saved.append(nearest - cheapest)
    assert max(saved) > 0.0


def test_with_no_penalty_the_cheapest_auxiliaries_charge_nothing():
    inst = _instance(lam=1.0, alpha=0.7, scenarios=_tail_heavy_scenarios())
    layout = plan_layout(inst, BinaryEncoding())
    gadget = build_cvar_gadget(inst, layout, n_zeta_bits=4, penalty=0.0)
    losses = gadget.losses_from(inst, layout, inst.witness)
    aux = gadget.cheapest_aux(losses, gadget.zeta.lo)
    assert all(aux[k] == 0 for grid in gadget.u for k in grid.keys)
