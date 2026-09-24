import pytest

from quantic.problem.constraints.base import Constraint
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.schedule import Schedule

X = (5, 5, 5)


def test_it_satisfies_the_constraint_protocol():
    assert isinstance(Cardinality(k=2), Constraint)


def test_a_schedule_within_the_limit_is_satisfied():
    s = Schedule(lots=((3, 0), (2, 0), (0, 5)))
    assert Cardinality(k=2).violation(s, X) == 0.0
    assert Cardinality(k=2).is_satisfied(s, X)


def test_exceeding_the_limit_counts_the_excess_names():
    s = Schedule(lots=((3, 0), (2, 0), (1, 0)))
    assert Cardinality(k=2).violation(s, X) == 1.0


def test_excess_accumulates_across_buckets():
    s = Schedule(lots=((3, 1), (2, 1), (1, 1)))
    assert Cardinality(k=2).violation(s, X) == 2.0


def test_a_larger_breach_gives_a_larger_violation():
    c = Cardinality(k=1)
    assert c.violation(Schedule(lots=((1, 0), (1, 0), (1, 0))), X) == 2.0
    assert c.violation(Schedule(lots=((1, 0), (1, 0), (0, 0))), X) == 1.0


def test_a_zero_entry_does_not_count_as_a_traded_name():
    s = Schedule(lots=((0, 5), (0, 5), (0, 5)))
    assert Cardinality(k=1).violation(s, X) == 2.0


def test_violation_is_zero_exactly_when_satisfied():
    c = Cardinality(k=2)
    for lots in (((3, 0), (2, 0), (0, 5)), ((1, 1), (1, 1), (1, 1))):
        s = Schedule(lots=lots)
        assert (c.violation(s, X) == 0.0) == c.is_satisfied(s, X)


def test_k_at_least_the_asset_count_constrains_nothing():
    """A generator that picks such a k has made D2 a no-op dial."""
    s = Schedule(lots=((1, 1), (1, 1), (1, 1)))
    assert Cardinality(k=3).violation(s, X) == 0.0


def test_a_non_positive_k_is_rejected():
    with pytest.raises(ValueError, match="k"):
        Cardinality(k=0)


def test_describe_carries_k_for_the_results_row():
    assert Cardinality(k=3).describe() == {"name": "cardinality", "k": 3}
