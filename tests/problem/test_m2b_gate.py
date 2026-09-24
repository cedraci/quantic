"""THE M2b GATE -- the parent spec's section 14 criterion, in full.

16 dial combinations x 4 tiers = 64 instances, every one carrying a provably
feasible witness and evaluating to a finite objective.
"""

import numpy as np
import pytest

from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.generator import generate_ladder
from quantic.problem.liquidation import LADDER
from quantic.problem.objective import evaluate, optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk, VarianceRisk

BUCKET_NS = 1800 * 1_000_000_000
SMALL = {"n_raw": 300, "n_target": 15}


def _params_for(tier):
    rng = np.random.default_rng(tier.n_assets)
    a = rng.normal(size=(tier.n_assets, tier.n_assets))
    return MarketParams(
        assets=tuple(
            AssetParams(f"S{i:02d}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + i, 0.01)
            for i in range(tier.n_assets)
        ),
        covariance=(a @ a.T) / tier.n_assets * 1e-5,
        bucket_ns=BUCKET_NS,
    )


def _zeta(inst):
    return optimal_zeta(inst, inst.witness) if isinstance(inst.risk, CVaRRisk) else None


@pytest.fixture(scope="module")
def full_ladder():
    return generate_ladder(_params_for, seed=2026, include_cvar=True, **SMALL)


def test_the_full_ladder_is_sixteen_combinations_on_four_tiers(full_ladder):
    assert len(full_ladder) == 16 * 4 == 64


def test_half_the_ladder_uses_cvar_risk(full_ladder):
    assert sum(isinstance(i.risk, CVaRRisk) for i in full_ladder) == 32
    assert sum(isinstance(i.risk, VarianceRisk) for i in full_ladder) == 32


def test_every_witness_is_feasible(full_ladder):
    """The gate. An empty feasible set anywhere makes the study unmeasurable."""
    broken = {
        i.instance_id: dict(classify(i, i.witness).violations)
        for i in full_ladder
        if not classify(i, i.witness).feasible
    }
    assert not broken, broken


def test_every_instance_evaluates_to_a_finite_positive_objective(full_ladder):
    for inst in full_ladder:
        total = evaluate(inst, inst.witness, zeta=_zeta(inst)).total
        assert np.isfinite(total), inst.instance_id
        assert total > 0.0, inst.instance_id


def test_every_cvar_instance_carries_a_reduction_report(full_ladder):
    """Spec 5.4: reduction error is reported alongside every CVaR result."""
    for inst in full_ladder:
        if isinstance(inst.risk, CVaRRisk):
            report = inst.risk.scenarios.reduction
            assert report is not None, inst.instance_id
            assert report.wasserstein > 0.0
            assert report.cvar_discrepancy is not None


def test_content_hashes_are_unique_across_the_full_ladder(full_ladder):
    hashes = [i.content_hash for i in full_ladder]
    assert len(set(hashes)) == len(hashes)


def test_the_non_cvar_half_is_byte_identical_to_the_m2a_ladder(full_ladder):
    """D3 must ADD a dial, not perturb the eight combinations that existed.

    If these hashes moved, every M2a result would silently stop being
    comparable with every M2b result.
    """
    m2a = generate_ladder(_params_for, seed=2026, **SMALL)
    assert len(m2a) == 32
    expected = {i.content_hash for i in m2a}
    actual = {i.content_hash for i in full_ladder if not i.dials.cvar_risk}
    assert actual == expected


def test_the_full_ladder_is_reproducible_from_its_seed(full_ladder):
    again = generate_ladder(_params_for, seed=2026, include_cvar=True, **SMALL)
    assert [i.content_hash for i in again] == [i.content_hash for i in full_ladder]


def test_every_tier_and_dial_pair_appears_exactly_once(full_ladder):
    seen = {(i.tier, i.dials.label) for i in full_ladder}
    expected = {
        (t.name, d.label) for t in LADDER for d in Dials.combinations(include_cvar=True)
    }
    assert seen == expected


def test_cvar_risk_is_positive_exactly_when_the_witness_holds_inventory(full_ladder):
    """The honest invariant, rather than blanket positivity.

    A position liquidated entirely in the first bucket leaves nothing
    outstanding, so every scenario's mark-to-market loss is exactly zero and
    CVaR is correctly zero. That happens at T0, whose positions are small: one
    instance draws `initial_lots=(1, 1, 1)`, and a single lot goes wholly into
    bucket 0. Asserting blanket positivity would assert something the model
    does not claim -- risk exists only where inventory is at risk.
    """
    for inst in full_ladder:
        if not isinstance(inst.risk, CVaRRisk):
            continue
        holds_inventory = int(inst.remaining_lots(inst.witness).max()) > 0
        risk_cost = evaluate(inst, inst.witness, zeta=_zeta(inst)).risk_cost
        assert (risk_cost > 0.0) == holds_inventory, inst.instance_id


def test_degenerate_zero_risk_instances_stay_rare(full_ladder):
    """Guards the ladder against silently degenerating into trivial instances.

    Measured at seed 2026: exactly one of the 32 CVaR instances has no
    inventory outstanding. If a change made most of them trivial, D3 would
    stop being a dial that moves anything and this gate would still pass
    without this check.
    """
    cvar = [i for i in full_ladder if isinstance(i.risk, CVaRRisk)]
    with_inventory = sum(
        1 for i in cvar if int(i.remaining_lots(i.witness).max()) > 0
    )
    assert with_inventory >= len(cvar) - 2, (
        f"only {with_inventory} of {len(cvar)} CVaR instances carry inventory at risk"
    )
