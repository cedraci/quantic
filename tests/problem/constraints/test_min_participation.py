import pytest

from quantic.problem.constraints.base import Constraint
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.schedule import Schedule

X = (6, 6)
MIN = (3, 2)


def test_it_satisfies_the_constraint_protocol():
    assert isinstance(MinParticipation(min_lots=MIN), Constraint)


def test_zero_or_at_least_the_minimum_is_satisfied():
    """The constraint is semi-continuous: x == 0 OR x >= m."""
    s = Schedule(lots=((3, 3), (0, 6)))
    assert MinParticipation(min_lots=MIN).violation(s, X) == 0.0
    assert MinParticipation(min_lots=MIN).is_satisfied(s, X)


def test_a_non_zero_entry_below_the_minimum_is_a_violation_of_the_shortfall():
    s = Schedule(lots=((1, 5), (0, 6)))
    assert MinParticipation(min_lots=MIN).violation(s, X) == 2.0


def test_violations_accumulate_across_cells():
    s = Schedule(lots=((1, 1), (1, 5)))
    assert MinParticipation(min_lots=MIN).violation(s, X) == 5.0


def test_a_larger_shortfall_gives_a_larger_violation():
    c = MinParticipation(min_lots=MIN)
    assert c.violation(Schedule(lots=((1, 5), (0, 6))), X) > c.violation(
        Schedule(lots=((2, 4), (0, 6))), X
    )


def test_the_minimum_is_per_asset():
    s = Schedule(lots=((2, 4), (2, 4)))
    assert MinParticipation(min_lots=MIN).violation(s, X) == 1.0


def test_violation_is_zero_exactly_when_satisfied():
    c = MinParticipation(min_lots=MIN)
    for lots in (((3, 3), (0, 6)), ((1, 5), (0, 6)), ((0, 0), (0, 0))):
        s = Schedule(lots=lots)
        assert (c.violation(s, X) == 0.0) == c.is_satisfied(s, X)


def test_a_minimum_of_one_constrains_nothing():
    """Any non-zero integer entry is already at least 1."""
    s = Schedule(lots=((1, 5), (1, 5)))
    assert MinParticipation(min_lots=(1, 1)).violation(s, X) == 0.0


def test_a_mismatched_minimum_length_is_rejected():
    s = Schedule(lots=((3, 3), (0, 6)))
    with pytest.raises(ValueError, match="min_lots"):
        MinParticipation(min_lots=(3, 2, 1)).violation(s, X)


def test_a_non_positive_minimum_is_rejected():
    with pytest.raises(ValueError, match="min_lots"):
        MinParticipation(min_lots=(3, 0))


def test_describe_carries_the_minimums_as_a_list():
    assert MinParticipation(min_lots=MIN).describe() == {
        "name": "min_participation", "min_lots": [3, 2]
    }
