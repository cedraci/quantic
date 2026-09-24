import numpy as np
import pytest

from quantic.problem.schedule import Schedule


def _schedule() -> Schedule:
    # 2 assets x 3 buckets
    return Schedule(lots=((2, 0, 1), (0, 0, 4)))


def test_shape_is_read_from_the_grid():
    s = _schedule()
    assert s.n_assets == 2
    assert s.n_buckets == 3


def test_sold_totals_a_row():
    s = _schedule()
    assert s.sold(0) == 3
    assert s.sold(1) == 4


def test_active_names_counts_nonzero_entries_in_a_bucket():
    s = _schedule()
    assert s.active_names(0) == 1   # only asset 0 trades
    assert s.active_names(1) == 0   # nobody trades
    assert s.active_names(2) == 2   # both trade


def test_cumulative_is_inclusive_of_the_current_bucket():
    s = _schedule()
    assert s.cumulative().tolist() == [[2, 2, 3], [0, 0, 4]]


def test_cumulative_before_is_exclusive_and_starts_at_zero():
    s = _schedule()
    assert s.cumulative_before().tolist() == [[0, 2, 2], [0, 0, 0]]


def test_the_two_cumulative_conventions_differ_by_exactly_the_schedule():
    """The permanent impact term uses the exclusive form.

    Reading the inclusive one would charge us for impact we had not yet
    caused, which is a plausible number and an invisible error.
    """
    s = _schedule()
    assert (s.cumulative() - s.cumulative_before()).tolist() == s.as_array().tolist()


def test_round_trips_through_an_array():
    s = _schedule()
    assert Schedule.from_array(s.as_array()) == s


def test_a_negative_entry_is_rejected():
    """Non-negativity is spec 5.3's constraint; the type enforces it."""
    with pytest.raises(ValueError, match="negative"):
        Schedule(lots=((1, -1),))


def test_a_ragged_grid_is_rejected():
    with pytest.raises(ValueError, match="rectangular|ragged"):
        Schedule(lots=((1, 2), (3,)))


def test_an_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        Schedule(lots=())


def test_schedules_are_hashable_and_compare_by_value():
    assert _schedule() == Schedule(lots=((2, 0, 1), (0, 0, 4)))
    assert len({_schedule(), _schedule()}) == 1


def test_as_array_is_int64_so_cumulative_sums_do_not_overflow():
    assert _schedule().as_array().dtype == np.int64
