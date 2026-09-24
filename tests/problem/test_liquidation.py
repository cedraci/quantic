import pytest

from quantic.problem.liquidation import LADDER, T0, T1, T2, T3, TierSpec, by_name


def test_the_ladder_reproduces_spec_5_5():
    assert [(t.name, t.n_assets, t.n_buckets, t.bits) for t in LADDER] == [
        ("T0", 3, 4, 2),
        ("T1", 5, 8, 3),
        ("T2", 8, 8, 3),
        ("T3", 30, 12, 4),
    ]


def test_max_lots_per_bucket_is_the_integer_range_the_bit_width_implies():
    assert T0.max_lots_per_bucket == 3       # 2**2 - 1
    assert T3.max_lots_per_bucket == 15      # 2**4 - 1


def test_approx_variables_tracks_spec_5_5s_sizing_column():
    """~24, ~120, ~192, ~1440 against the spec's 24 / 120-180 / 190-250 / 1500+."""
    assert T0.approx_variables == 24
    assert T1.approx_variables == 120
    assert T2.approx_variables == 192
    assert T3.approx_variables == 1440


def test_t2_sits_near_the_dense_embedding_ceiling():
    """Spec 3.2: Advantage2 fits ~230 dense logical variables. T2 is the edge tier."""
    assert 180 < T2.approx_variables < 260


def test_by_name_finds_a_tier():
    assert by_name("T2") is T2


def test_by_name_lists_the_ladder_when_asked_for_something_else():
    with pytest.raises(KeyError, match="T0"):
        by_name("T9")


# --- the capacity arithmetic Task 12 relies on ------------------------------


@pytest.mark.parametrize("tier", LADDER)
def test_the_concentrated_bound_fits_within_bucket_capacity(tier: TierSpec):
    """THE BOUND THE SPEC SELF-REVIEW CAUGHT.

    With D2 on, each asset may use `buckets_per_asset_cap` buckets and each
    bucket hosts at most `active_per_bucket` names. The demanded asset-bucket
    slots must not exceed capacity, or no feasible witness exists.

    The naive bound -- letting every asset use all T buckets -- fails on every
    tier: T0 demands 12 slots against a capacity of 8, T3 demands 360 against
    180.
    """
    demanded = tier.n_assets * tier.buckets_per_asset_cap
    capacity = tier.n_buckets * tier.active_per_bucket
    assert demanded <= capacity, (
        f"{tier.name}: {demanded} asset-bucket slots demanded but only {capacity} available"
    )


@pytest.mark.parametrize("tier", LADDER)
def test_the_naive_bound_would_not_have_fitted(tier: TierSpec):
    """Pins why the tightened bound exists, so nobody loosens it back."""
    naive = tier.n_assets * tier.n_buckets
    capacity = tier.n_buckets * tier.active_per_bucket
    assert naive > capacity


@pytest.mark.parametrize("tier", LADDER)
def test_active_per_bucket_actually_binds_cardinality(tier: TierSpec):
    """If it equalled n_assets, dial D2 would constrain nothing."""
    assert 1 <= tier.active_per_bucket < tier.n_assets


@pytest.mark.parametrize("tier", LADDER)
def test_every_asset_gets_at_least_one_bucket(tier: TierSpec):
    assert tier.buckets_per_asset_cap >= 1


@pytest.mark.parametrize("tier", LADDER)
def test_max_position_is_looser_when_d2_is_off(tier: TierSpec):
    assert tier.max_position_lots(discrete_participation=False) >= tier.max_position_lots(
        discrete_participation=True
    )


def test_a_tier_with_a_non_positive_dimension_is_rejected():
    with pytest.raises(ValueError, match="n_assets"):
        TierSpec(name="bad", n_assets=0, n_buckets=4, bits=2)
    with pytest.raises(ValueError, match="bits"):
        TierSpec(name="bad", n_assets=3, n_buckets=4, bits=0)


def test_tier_names_are_unique():
    assert len({t.name for t in LADDER}) == len(LADDER)
