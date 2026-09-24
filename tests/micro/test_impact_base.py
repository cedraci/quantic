import pytest

from quantic.micro.impact.almgren_chriss import AlmgrenChriss
from quantic.micro.impact.base import ImpactParams, cost_in_bps, currency_cost

P = ImpactParams(
    symbol="SYNA", sigma_bucket=0.005, bucket_ns=1800 * 1_000_000_000,
    bucket_volume_shares=400_000.0, price=100.0,
)


def test_zero_quantity_has_zero_impact_and_cost():
    m = AlmgrenChriss(eta=0.8)
    assert m.price_impact(0.0, P) == 0.0
    assert m.temporary_cost(0.0, P) == 0.0


def test_impact_is_linear_in_quantity():
    m = AlmgrenChriss(eta=0.8)
    assert m.price_impact(40_000.0, P) == pytest.approx(2 * m.price_impact(20_000.0, P))


def test_impact_scales_with_sigma_and_participation():
    m = AlmgrenChriss(eta=0.8)
    assert m.price_impact(40_000.0, P) == pytest.approx(0.8 * 0.005 * 0.1)


def test_linear_model_obeys_the_one_half_cost_rule():
    """Integrating linear impact over the executed quantity halves it."""
    m = AlmgrenChriss(eta=0.8)
    assert m.temporary_cost(40_000.0, P) == pytest.approx(m.price_impact(40_000.0, P) / 2.0)


def test_impact_is_sign_insensitive():
    m = AlmgrenChriss(eta=0.8)
    assert m.price_impact(-40_000.0, P) == pytest.approx(m.price_impact(40_000.0, P))


def test_permanent_impact_uses_gamma():
    m = AlmgrenChriss(eta=0.8, gamma=0.3)
    assert m.permanent_impact(40_000.0, P) == pytest.approx(0.3 * 0.005 * 0.1)
    assert AlmgrenChriss(eta=0.8).permanent_impact(40_000.0, P) == 0.0


def test_currency_cost_and_bps_are_consistent():
    m = AlmgrenChriss(eta=0.8)
    q = 40_000.0
    assert currency_cost(m, q, P) == pytest.approx(m.temporary_cost(q, P) * q * P.price)
    assert cost_in_bps(m, q, P) == pytest.approx(m.temporary_cost(q, P) * 10_000)


def test_model_satisfies_the_protocol():
    from quantic.micro.impact.base import ImpactModel

    assert isinstance(AlmgrenChriss(eta=0.8), ImpactModel)


def test_negative_bucket_volume_is_rejected():
    with pytest.raises(ValueError, match="bucket_volume_shares"):
        ImpactParams(
            symbol="SYNA", sigma_bucket=0.005, bucket_ns=1800 * 1_000_000_000,
            bucket_volume_shares=0.0, price=100.0,
        )
