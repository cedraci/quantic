import pytest

from quantic.problem.constraints.base import Constraint
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.schedule import Schedule

X = (3, 4)


def test_it_satisfies_the_constraint_protocol():
    assert isinstance(FullLiquidation(), Constraint)


def test_a_schedule_selling_exactly_the_position_is_satisfied():
    s = Schedule(lots=((2, 1), (0, 4)))
    assert FullLiquidation().violation(s, X) == 0.0
    assert FullLiquidation().is_satisfied(s, X)


def test_under_selling_is_a_violation_of_the_shortfall():
    s = Schedule(lots=((2, 0), (0, 4)))
    assert FullLiquidation().violation(s, X) == 1.0
    assert not FullLiquidation().is_satisfied(s, X)


def test_over_selling_is_a_violation_of_the_excess():
    s = Schedule(lots=((2, 3), (0, 4)))
    assert FullLiquidation().violation(s, X) == 2.0


def test_violations_accumulate_across_assets():
    s = Schedule(lots=((1, 0), (0, 1)))
    assert FullLiquidation().violation(s, X) == 5.0


def test_a_larger_shortfall_gives_a_larger_violation():
    """M3's penalty derivation needs the magnitude, not just the boolean."""
    near = Schedule(lots=((2, 0), (0, 4)))
    far = Schedule(lots=((0, 0), (0, 4)))
    assert FullLiquidation().violation(far, X) > FullLiquidation().violation(near, X)


def test_violation_is_zero_exactly_when_satisfied():
    for lots in (((2, 1), (0, 4)), ((3, 0), (4, 0)), ((0, 0), (0, 0))):
        s = Schedule(lots=lots)
        c = FullLiquidation()
        assert (c.violation(s, X) == 0.0) == c.is_satisfied(s, X)


def test_a_mismatched_position_length_is_rejected():
    s = Schedule(lots=((2, 1), (0, 4)))
    with pytest.raises(ValueError, match="2 assets"):
        FullLiquidation().violation(s, (3, 4, 5))


def test_describe_is_json_safe_for_the_results_row():
    assert FullLiquidation().describe() == {"name": "full_liquidation"}
