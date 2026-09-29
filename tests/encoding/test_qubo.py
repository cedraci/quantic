import numpy as np
import pytest

from quantic.encoding.qubo import LinearForm, Qubo, QuboBuilder
from quantic.encoding.registry import RegistryBuilder


def _registry(*keys):
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    return b.freeze()


def test_a_linear_form_scales_and_adds():
    f = LinearForm.of(("a",), 2.0) + LinearForm.constant_form(3.0)
    g = f.scaled(0.5)
    assert g.constant == 1.5
    assert g.terms == ((("a",), 1.0),)


def test_energy_is_the_quadratic_form_plus_the_constant():
    reg = _registry(("a",), ("b",))
    b = QuboBuilder(reg)
    b.add_constant(5.0)
    b.add_linear(("a",), 2.0)
    b.add_quadratic(("a",), ("b",), 3.0)
    q = b.build()
    assert q.energy([0, 0]) == 5.0
    assert q.energy([1, 0]) == 7.0
    assert q.energy([1, 1]) == 10.0


def test_a_product_folds_z_squared_to_z():
    """z**2 == z for a binary, so a self-product is linear, not quadratic."""
    reg = _registry(("a",),)
    b = QuboBuilder(reg)
    b.add_square(LinearForm.of(("a",), 3.0))
    q = b.build()
    assert q.energy([1]) == pytest.approx(9.0)
    assert q.energy([0]) == 0.0


def test_a_product_of_two_forms_expands_every_cross_term():
    reg = _registry(("a",), ("b",))
    f = LinearForm.of(("a",), 2.0) + LinearForm.constant_form(1.0)
    g = LinearForm.of(("b",), 5.0) + LinearForm.constant_form(-1.0)
    b = QuboBuilder(reg)
    b.add_product(f, g)
    q = b.build()
    for za, zb in ((0, 0), (0, 1), (1, 0), (1, 1)):
        expected = (2.0 * za + 1.0) * (5.0 * zb - 1.0)
        assert q.energy([za, zb]) == pytest.approx(expected)


def test_a_square_of_a_three_term_form_matches_direct_arithmetic():
    """The case a naive implementation gets wrong: cross terms counted once."""
    reg = _registry(("a",), ("b",), ("c",))
    f = (
        LinearForm.of(("a",), 1.0)
        + LinearForm.of(("b",), 2.0)
        + LinearForm.of(("c",), 3.0)
        + LinearForm.constant_form(-4.0)
    )
    builder = QuboBuilder(reg)
    builder.add_square(f)
    q = builder.build()
    for za in (0, 1):
        for zb in (0, 1):
            for zc in (0, 1):
                expected = (za + 2 * zb + 3 * zc - 4) ** 2
                assert q.energy([za, zb, zc]) == pytest.approx(expected)


def test_the_matrix_is_upper_triangular():
    reg = _registry(("a",), ("b",))
    b = QuboBuilder(reg)
    b.add_quadratic(("b",), ("a",), 7.0)
    q = b.build()
    assert q.Q[1, 0] == 0.0
    assert q.Q[0, 1] == 7.0


def test_energy_refuses_a_non_binary_vector():
    reg = _registry(("a",))
    q = QuboBuilder(reg).build()
    with pytest.raises(ValueError, match="0 or 1"):
        q.energy([2])


def test_energy_refuses_a_wrong_length_vector():
    reg = _registry(("a",), ("b",))
    q = QuboBuilder(reg).build()
    with pytest.raises(ValueError, match="2 variables"):
        q.energy([1])


def test_two_qubos_over_one_registry_add():
    reg = _registry(("a",))
    x = QuboBuilder(reg)
    x.add_linear(("a",), 1.0)
    y = QuboBuilder(reg)
    y.add_constant(4.0)
    total = x.build() + y.build()
    assert total.energy([1]) == 5.0


def test_adding_qubos_over_different_registries_is_refused():
    a = QuboBuilder(_registry(("a",))).build()
    b = QuboBuilder(_registry(("b",))).build()
    with pytest.raises(ValueError, match="different registries"):
        _ = a + b


def test_a_non_finite_coefficient_is_refused_at_build():
    reg = _registry(("a",))
    b = QuboBuilder(reg)
    b.add_linear(("a",), float("inf"))
    with pytest.raises(ValueError, match="finite"):
        b.build()


def test_equality_is_defined_despite_the_array_field():
    reg = _registry(("a",))
    b = QuboBuilder(reg)
    b.add_linear(("a",), 1.0)
    assert b.build() == QuboBuilder(reg).build() + b.build()
    assert np.array_equal(b.build().Q, b.build().Q)


def test_qubo_converts_and_stores_python_list_as_float64_array():
    """Qubo converts Q to ndarray float64 at construction and stores it."""
    reg = _registry(("a",), ("b",))
    # Construct from Python nested list, not numpy array
    q = Qubo(Q=[[1.0, 2.0], [0.0, 3.0]], constant=0.0, registry=reg)
    # Q field must be ndarray
    assert isinstance(q.Q, np.ndarray)
    assert q.Q.dtype == np.float64
    # hash must work (requires tobytes method on Q)
    assert isinstance(hash(q), int)
    # energy should work
    assert q.energy([1, 1]) == pytest.approx(1.0 + 2.0 * 1.0 + 3.0)


def test_folding_sums_repeated_keys_in_first_appearance_order():
    f = (
        LinearForm.of(("b",), 1.0)
        + LinearForm.of(("a",), 2.0)
        + LinearForm.of(("b",), 4.0)
        + LinearForm.constant_form(3.0)
    )
    g = f.folded()
    assert g.constant == 3.0
    assert g.terms == ((("b",), 5.0), (("a",), 2.0))


def test_folding_a_form_without_repeats_changes_nothing():
    f = LinearForm.of(("a",), 2.0) + LinearForm.of(("b",), 0.1)
    assert f.folded() == f


class _CountingBuilder(QuboBuilder):
    def __init__(self, registry):
        super().__init__(registry)
        self.quadratic_calls = 0

    def add_quadratic(self, key_a, key_b, coef):
        self.quadratic_calls += 1
        super().add_quadratic(key_a, key_b, coef)


def test_a_product_costs_the_distinct_keys_not_the_repeats():
    """A holdings sum names each bit once per bucket it is still held through.

    Expanding the repeats pairwise is what put a single T3 CVaR instance past
    ten minutes: the work must scale with the bits, not with how the form was
    assembled.
    """
    reg = _registry(("a",), ("b",))
    f = LinearForm.constant_form(1.0)
    for _ in range(12):
        f = f + LinearForm.of(("a",), 1.0) + LinearForm.of(("b",), 2.0)
    b = _CountingBuilder(reg)
    b.add_square(f)
    assert b.quadratic_calls == 2 * 2


def test_a_product_of_forms_with_repeated_keys_matches_direct_arithmetic():
    reg = _registry(("a",), ("b",))
    f = LinearForm.constant_form(-4.0)
    for _ in range(3):
        f = f + LinearForm.of(("a",), 1.0) + LinearForm.of(("b",), 2.0)
    g = LinearForm.of(("b",), 5.0) + LinearForm.of(("b",), -1.0)
    builder = QuboBuilder(reg)
    builder.add_product(f, g)
    q = builder.build()
    for za in (0, 1):
        for zb in (0, 1):
            expected = (3 * za + 6 * zb - 4) * (4 * zb)
            assert q.energy([za, zb]) == pytest.approx(expected)
