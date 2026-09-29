import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantic.encoding.integer import (
    ENCODINGS,
    BinaryEncoding,
    DomainWallEncoding,
    OneHotEncoding,
    bounded_coefficients,
)
from quantic.encoding.qubo import QuboBuilder
from quantic.encoding.registry import RegistryBuilder


@given(st.integers(min_value=0, max_value=512))
def test_bounded_coefficients_reach_every_value_and_no_more(m):
    coeffs = bounded_coefficients(m)
    reachable = {0}
    for c in coeffs:
        reachable |= {r + c for r in reachable}
    assert max(reachable) == m
    assert set(range(m + 1)) <= reachable


@given(st.integers(min_value=0, max_value=512))
def test_bounded_coefficients_use_the_minimal_bit_count(m):
    assert len(bounded_coefficients(m)) == (m.bit_length() if m else 0)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize("values", [(0, 1), (0, 1, 2, 3, 4, 5), tuple(range(9))])
def test_every_value_round_trips(encoding, values):
    keys = encoding.bit_keys(("x", 0, 0), values)
    for v in values:
        bits = encoding.encode_value(keys, values, v)
        assert encoding.decode_value(keys, values, bits) == v


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_value_form_agrees_with_the_encoded_bits(encoding):
    values = tuple(range(6))
    keys = encoding.bit_keys(("x", 0, 0), values)
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    reg = b.freeze()
    form = encoding.value_form(keys, values)
    for v in values:
        bits = encoding.encode_value(keys, values, v)
        builder = QuboBuilder(reg)
        builder.add_form(form)
        vector = [bits[k] for k in reg.keys]
        assert builder.build().energy(vector) == pytest.approx(float(v))


@pytest.mark.parametrize(
    "encoding", [OneHotEncoding(), DomainWallEncoding()], ids=lambda e: e.name
)
def test_a_table_encoding_represents_an_arbitrary_cost_exactly(encoding):
    """This is what makes D1 free: any f over the value set is linear in the bits."""
    values = (0, 2, 3, 7)
    costs = {0: 0.0, 2: 1.9, 3: -4.25, 7: 11.0}
    keys = encoding.bit_keys(("x", 0, 0), values)
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    reg = b.freeze()
    form = encoding.cost_form(keys, values, costs)
    assert form is not None
    for v in values:
        bits = encoding.encode_value(keys, values, v)
        builder = QuboBuilder(reg)
        builder.add_form(form)
        vector = [bits[k] for k in reg.keys]
        assert builder.build().energy(vector) == pytest.approx(costs[v])


def test_the_log_encoding_offers_no_exact_cost_table():
    """x**delta has no exact quadratic form in log bits. Saying so is the point."""
    enc = BinaryEncoding()
    assert enc.supports_exact_table is False
    values = tuple(range(4))
    keys = enc.bit_keys(("x", 0, 0), values)
    assert enc.cost_form(keys, values, {v: float(v) for v in values}) is None


def test_the_log_encoding_refuses_a_non_contiguous_value_set():
    enc = BinaryEncoding()
    with pytest.raises(ValueError, match="contiguous"):
        enc.bit_keys(("x", 0, 0), (0, 3, 4))


def test_the_log_encoding_refuses_a_value_set_not_starting_at_zero():
    enc = BinaryEncoding()
    with pytest.raises(ValueError, match="contiguous"):
        enc.bit_keys(("x", 0, 0), (2, 3, 4))


def test_one_hot_validity_is_zero_exactly_on_the_valid_states():
    enc = OneHotEncoding()
    values = (0, 1, 2)
    keys = enc.bit_keys(("x", 0, 0), values)
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    reg = b.freeze()
    forms = enc.validity_forms(keys, values)
    for vector in np.ndindex(*(2,) * len(keys)):
        builder = QuboBuilder(reg)
        for f in forms:
            builder.add_square(f)
        energy = builder.build().energy(list(vector))
        valid = sum(vector) == 1
        assert (energy == 0.0) is valid


def test_domain_wall_validity_is_zero_exactly_on_the_walls():
    enc = DomainWallEncoding()
    values = (0, 1, 2, 3)
    keys = enc.bit_keys(("x", 0, 0), values)
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    reg = b.freeze()
    for vector in np.ndindex(*(2,) * len(keys)):
        builder = QuboBuilder(reg)
        for f in enc.validity_forms(keys, values):
            builder.add_square(f)
        for lhs, rhs in enc.validity_products(keys, values):
            builder.add_product(lhs, rhs)
        energy = builder.build().energy(list(vector))
        monotone = all(vector[j] >= vector[j + 1] for j in range(len(vector) - 1))
        assert (energy == 0.0) is monotone


def test_domain_wall_uses_one_fewer_bit_than_one_hot():
    values = tuple(range(6))
    assert len(DomainWallEncoding().bit_keys(("x", 0, 0), values)) == 5
    assert len(OneHotEncoding().bit_keys(("x", 0, 0), values)) == 6


@pytest.mark.parametrize(
    "encoding", [OneHotEncoding(), DomainWallEncoding()], ids=lambda e: e.name
)
def test_a_table_encoding_gives_the_positivity_indicator_for_free(encoding):
    """Cardinality needs 1[x > 0]. Under a table encoding it is already linear."""
    values = (0, 3, 4)
    keys = encoding.bit_keys(("x", 0, 0), values)
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    reg = b.freeze()
    form = encoding.indicator_form(keys, values)
    assert form is not None
    for v in values:
        bits = encoding.encode_value(keys, values, v)
        builder = QuboBuilder(reg)
        builder.add_form(form)
        vector = [bits[k] for k in reg.keys]
        assert builder.build().energy(vector) == pytest.approx(1.0 if v > 0 else 0.0)


def test_the_log_encoding_has_no_free_indicator():
    enc = BinaryEncoding()
    values = tuple(range(4))
    keys = enc.bit_keys(("x", 0, 0), values)
    assert enc.indicator_form(keys, values) is None


def test_decoding_an_invalid_one_hot_state_is_refused():
    enc = OneHotEncoding()
    values = (0, 1, 2)
    keys = enc.bit_keys(("x", 0, 0), values)
    with pytest.raises(ValueError, match="exactly one"):
        enc.decode_value(keys, values, dict.fromkeys(keys, 1))


def test_decoding_a_broken_domain_wall_is_refused():
    enc = DomainWallEncoding()
    values = (0, 1, 2, 3)
    keys = enc.bit_keys(("x", 0, 0), values)
    broken = {keys[0]: 0, keys[1]: 1, keys[2]: 0}
    with pytest.raises(ValueError, match="monotone"):
        enc.decode_value(keys, values, broken)
