"""THE M2a GATE -- spec section 14's done-criterion, asserted.

8 dial combinations x 4 tiers = 32 instances, every one carrying a feasible
witness and evaluating to a finite objective.
"""

import numpy as np
import pytest

from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.generator import generate_ladder
from quantic.problem.liquidation import LADDER
from quantic.problem.objective import evaluate
from quantic.problem.params import AssetParams, MarketParams

BUCKET_NS = 1800 * 1_000_000_000


def _params_for(tier):
    rng = np.random.default_rng(tier.n_assets)
    assets = tuple(
        AssetParams(
            symbol=f"S{i:02d}", delta=0.5, y_coef=0.8, gamma=0.1,
            sigma_bucket=0.005, bucket_volume_shares=400_000.0,
            price=100.0 + i, half_spread=0.01,
        )
        for i in range(tier.n_assets)
    )
    a = rng.normal(size=(tier.n_assets, tier.n_assets))
    cov = (a @ a.T) / tier.n_assets * 1e-5
    return MarketParams(assets=assets, covariance=cov, bucket_ns=BUCKET_NS)


@pytest.fixture(scope="module")
def ladder():
    return generate_ladder(_params_for, seed=2026)


def test_the_ladder_has_eight_combinations_on_each_of_four_tiers(ladder):
    assert len(ladder) == 8 * 4 == 32


def test_every_tier_and_dial_pair_appears_exactly_once(ladder):
    seen = {(i.tier, i.dials.label) for i in ladder}
    expected = {(t.name, d.label) for t in LADDER for d in Dials.combinations()}
    assert seen == expected
    assert len(ladder) == len(seen)


def test_every_witness_is_feasible(ladder):
    """The gate. An empty feasible set anywhere makes the study unmeasurable."""
    broken = {
        i.instance_id: dict(classify(i, i.witness).violations)
        for i in ladder
        if not classify(i, i.witness).feasible
    }
    assert not broken, broken


def test_every_instance_evaluates_to_a_finite_positive_objective(ladder):
    for inst in ladder:
        total = evaluate(inst, inst.witness).total
        assert np.isfinite(total), inst.instance_id
        assert total > 0.0, inst.instance_id


def test_every_instance_reports_a_positive_notional(ladder):
    for inst in ladder:
        assert inst.notional() > 0.0, inst.instance_id


def test_content_hashes_are_unique_across_the_ladder(ladder):
    """A collision would merge two cells of the experiment design."""
    hashes = [i.content_hash for i in ladder]
    assert len(set(hashes)) == len(hashes)


def test_instance_ids_are_unique_across_the_ladder(ladder):
    ids = [i.instance_id for i in ladder]
    assert len(set(ids)) == len(ids)


def test_no_cvar_instance_is_generated_by_default(ladder):
    """D3 is M2b. Generating it now would claim coverage M2a does not have."""
    assert all(not i.dials.cvar_risk for i in ladder)


def test_including_cvar_doubles_the_ladder():
    """The M2b switch. One flag, no other change."""
    full = generate_ladder(_params_for, seed=2026, include_cvar=True)
    assert len(full) == 64
    assert sum(i.dials.cvar_risk for i in full) == 32


def test_the_ladder_is_reproducible_from_its_seed(ladder):
    again = generate_ladder(_params_for, seed=2026)
    assert [i.content_hash for i in again] == [i.content_hash for i in ladder]


def test_a_different_seed_changes_every_instance(ladder):
    other = generate_ladder(_params_for, seed=7)
    assert not (
        {i.content_hash for i in other} & {i.content_hash for i in ladder}
    )


def test_variable_counts_climb_across_the_tiers(ladder):
    """Spec 5.5's ladder is a ramp; if it flattened, the study has no x-axis."""
    by_tier = {}
    for inst in ladder:
        by_tier[inst.tier] = inst.n_assets * inst.n_buckets
    ordered = [by_tier[t.name] for t in LADDER]
    assert ordered == sorted(ordered)
    assert ordered[0] < ordered[-1]
