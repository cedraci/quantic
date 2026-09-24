import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(**kw) -> Instance:
    assets = (Asset("A", lot_size=100, tick_size=0.01), Asset("B", lot_size=10, tick_size=0.01))
    params = MarketParams(
        assets=(
            AssetParams("A", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01),
            AssetParams("B", 0.5, 0.8, 0.1, 0.005, 400_000.0, 50.0, 0.01),
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    base = dict(
        instance_id="T0-none-0",
        tier="T0",
        assets=assets,
        n_buckets=2,
        initial_lots=(3, 4),
        max_lots_per_bucket=3,
        params=params,
        dials=Dials(),
        constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6),
        witness=Schedule(lots=((2, 1), (3, 1))),
        seed=0,
    )
    base.update(kw)
    return Instance(**base)


def test_shape_is_derived_from_the_assets():
    assert _instance().n_assets == 2
    assert _instance().n_buckets == 2


def test_lot_sizes_are_read_from_the_core_asset_type():
    assert _instance().lot_sizes().tolist() == [100, 10]


def test_remaining_lots_is_the_position_minus_inclusive_cumulative():
    """Spec 5.1: h[i,t] = X[i] - sum over s <= t of x[i,s]."""
    h = _instance().remaining_lots(Schedule(lots=((2, 1), (3, 1))))
    assert h.tolist() == [[1, 0], [1, 0]]


def test_remaining_lots_reaches_zero_under_full_liquidation():
    h = _instance().remaining_lots(_instance().witness)
    assert h[:, -1].tolist() == [0, 0]


def test_remaining_lots_rejects_a_schedule_of_the_wrong_shape():
    with pytest.raises(ValueError, match="shape|assets|buckets"):
        _instance().remaining_lots(Schedule(lots=((1, 1, 1),)))


def test_notional_converts_lots_to_shares_before_pricing():
    """3 lots x 100 shares x $100 + 4 lots x 10 shares x $50 = 32000."""
    assert _instance().notional() == pytest.approx(3 * 100 * 100.0 + 4 * 10 * 50.0)


# --- content hash ----------------------------------------------------------


def test_content_hash_is_deterministic():
    assert _instance().content_hash == _instance().content_hash


def test_content_hash_is_a_sha256_hex_digest():
    h = _instance().content_hash
    assert len(h) == 64
    assert set(h) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    "field,value",
    [
        ("tier", "T1"),
        ("seed", 1),
        ("initial_lots", (4, 4)),
        ("max_lots_per_bucket", 7),
        ("dials", Dials(concave_impact=True)),
        ("risk", VarianceRisk(lam=2e-6)),
        ("witness", Schedule(lots=((3, 0), (3, 1)))),
        ("constraints", (FullLiquidation(), Cardinality(k=1))),
    ],
)
def test_changing_any_field_changes_the_hash(field, value):
    assert _instance().content_hash != _instance(**{field: value}).content_hash


def test_changing_the_covariance_changes_the_hash():
    """The hash must cover the parameters, not only the problem shape."""
    p = _instance().params
    other = dataclasses.replace(p, covariance=np.eye(2) * 2e-5)
    assert _instance().content_hash != _instance(params=other).content_hash


def test_instance_id_is_not_hashed():
    """The hash identifies the content; the id is a human-facing label.

    Two instances with identical content must collide in the hash, so a
    duplicate is detectable.
    """
    assert _instance().content_hash == _instance(instance_id="something-else").content_hash


# --- rejections ------------------------------------------------------------


def test_a_position_vector_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="initial_lots"):
        _instance(initial_lots=(3,))


def test_params_describing_a_different_asset_count_are_rejected():
    p = _instance().params.subset(1)
    with pytest.raises(ValueError, match="params"):
        _instance(params=p)


def test_symbols_that_disagree_between_assets_and_params_are_rejected():
    """Silently mismatched ordering would misprice every asset."""
    swapped = (Asset("B", lot_size=100, tick_size=0.01), Asset("A", lot_size=10, tick_size=0.01))
    with pytest.raises(ValueError, match="symbol"):
        _instance(assets=swapped)


def test_a_witness_of_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="witness"):
        _instance(witness=Schedule(lots=((1, 1, 1), (1, 1, 1))))


def test_a_witness_exceeding_max_lots_per_bucket_is_rejected():
    with pytest.raises(ValueError, match="max_lots_per_bucket"):
        _instance(witness=Schedule(lots=((9, 0), (4, 0))), initial_lots=(9, 4))


def test_a_negative_position_is_rejected():
    with pytest.raises(ValueError, match="initial_lots"):
        _instance(initial_lots=(-1, 4))


def test_a_non_positive_bucket_count_is_rejected():
    with pytest.raises(ValueError, match="n_buckets"):
        _instance(n_buckets=0)


def test_no_constraints_at_all_is_rejected():
    """Without full liquidation the trivial empty schedule is optimal."""
    with pytest.raises(ValueError, match="constraint"):
        _instance(constraints=())


# --- equality and hashing ---------------------------------------------------
#
# @dataclass(frozen=True) generates __eq__/__hash__ from field tuples, and
# Instance carries a MarketParams holding an ndarray, so the generated
# versions inherit MarketParams's defect. Both are defined explicitly,
# keyed on content_hash, instead.


def test_instances_differing_only_in_instance_id_compare_and_hash_equal():
    a = _instance(instance_id="one")
    b = _instance(instance_id="two")
    assert a == b
    assert hash(a) == hash(b)


def test_instances_differing_in_a_hashed_field_compare_unequal():
    a = _instance()
    b = _instance(seed=1)
    assert a != b


def test_a_set_of_instances_deduplicates_by_content():
    a = _instance(instance_id="one")
    b = _instance(instance_id="two")
    c = _instance(instance_id="three", seed=1)
    assert {a, b, c} == {a, c}
    assert len({a, b, c}) == 2


def test_an_instance_works_as_a_dict_key():
    a = _instance(instance_id="one")
    b = _instance(instance_id="two")
    d = {a: "value"}
    assert d[b] == "value"


def test_instance_compared_against_an_unrelated_object_returns_false():
    assert _instance() != "not an Instance"
    assert _instance() != 42


def test_dataclasses_replace_still_works_on_instance():
    a = _instance()
    b = dataclasses.replace(a, risk=VarianceRisk(lam=5e-6))
    assert b.risk.lam == 5e-6
    assert b.content_hash != a.content_hash
