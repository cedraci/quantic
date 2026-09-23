import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantic.micro.impact.almgren_chriss import AlmgrenChriss
from quantic.micro.impact.base import ImpactParams
from quantic.micro.impact.sqrt_law import PowerLawImpact, sqrt_law

P = ImpactParams(symbol="SYNA", sigma=0.005, bucket_volume=400_000.0, price=100.0)


def test_sqrt_law_matches_the_closed_form():
    m = sqrt_law(y_coef=0.8)
    assert m.delta == 0.5
    assert m.price_impact(40_000.0, P) == pytest.approx(0.8 * 0.005 * 0.1**0.5)


def test_sqrt_law_obeys_the_two_thirds_cost_rule():
    m = sqrt_law(y_coef=0.8)
    assert m.temporary_cost(40_000.0, P) == pytest.approx(
        m.price_impact(40_000.0, P) * 2.0 / 3.0
    )


def test_delta_one_reproduces_almgren_chriss_exactly():
    power = PowerLawImpact(delta=1.0, y_coef=0.8, gamma=0.3)
    linear = AlmgrenChriss(eta=0.8, gamma=0.3)
    for q in (1_000.0, 40_000.0, 200_000.0):
        assert power.price_impact(q, P) == pytest.approx(linear.price_impact(q, P))
        assert power.temporary_cost(q, P) == pytest.approx(linear.temporary_cost(q, P))
        assert power.permanent_impact(q, P) == pytest.approx(linear.permanent_impact(q, P))


def test_zero_quantity_is_zero():
    assert sqrt_law(y_coef=0.8).price_impact(0.0, P) == 0.0


def test_invalid_delta_is_rejected():
    with pytest.raises(ValueError, match="delta"):
        PowerLawImpact(delta=0.0, y_coef=0.8)
    with pytest.raises(ValueError, match="delta"):
        PowerLawImpact(delta=1.5, y_coef=0.8)


@settings(max_examples=200)
@given(
    delta=st.floats(min_value=0.2, max_value=0.95),
    q=st.floats(min_value=1.0, max_value=1e5),
)
def test_impact_is_strictly_concave_below_delta_one(delta: float, q: float):
    """The source of NP-hardness: doubling size less than doubles impact."""
    m = PowerLawImpact(delta=delta, y_coef=0.8)
    assert m.price_impact(2 * q, P) < 2 * m.price_impact(q, P)


@settings(max_examples=200)
@given(
    delta=st.floats(min_value=0.2, max_value=1.0),
    q=st.floats(min_value=1.0, max_value=1e5),
)
def test_cost_is_impact_divided_by_delta_plus_one(delta: float, q: float):
    m = PowerLawImpact(delta=delta, y_coef=0.8)
    assert m.temporary_cost(q, P) == pytest.approx(m.price_impact(q, P) / (delta + 1.0))


@settings(max_examples=200)
@given(
    delta=st.floats(min_value=0.2, max_value=1.0),
    a=st.floats(min_value=1.0, max_value=1e5),
    b=st.floats(min_value=1.0, max_value=1e5),
)
def test_impact_is_monotone_in_quantity(delta: float, a: float, b: float):
    m = PowerLawImpact(delta=delta, y_coef=0.8)
    lo, hi = sorted((a, b))
    assert m.price_impact(lo, P) <= m.price_impact(hi, P) + 1e-15
