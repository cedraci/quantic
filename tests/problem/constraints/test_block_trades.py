import pytest

from quantic.problem.constraints.base import Constraint
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.schedule import Schedule

X = (6, 6)
BLOCKS = (Block(asset=0, bucket=0, lots=4),)


def test_it_satisfies_the_constraint_protocol():
    assert isinstance(BlockTrades(blocks=BLOCKS), Constraint)


def test_trading_the_block_whole_is_satisfied():
    s = Schedule(lots=((4, 2), (0, 6)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 0.0
    assert BlockTrades(blocks=BLOCKS).is_satisfied(s, X)


def test_not_trading_the_block_at_all_is_satisfied():
    """All-or-nothing means nothing is a legitimate choice."""
    s = Schedule(lots=((0, 6), (0, 6)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 0.0


def test_a_partial_fill_is_a_violation_of_the_distance_to_the_nearer_end():
    s = Schedule(lots=((3, 3), (0, 6)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 1.0


def test_a_small_partial_fill_is_measured_against_zero():
    s = Schedule(lots=((1, 5), (0, 6)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 1.0


def test_overfilling_a_block_is_a_violation():
    s = Schedule(lots=((6, 0), (0, 6)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 2.0


def test_violations_accumulate_across_blocks():
    blocks = (Block(asset=0, bucket=0, lots=4), Block(asset=1, bucket=1, lots=4))
    s = Schedule(lots=((3, 3), (0, 3)))
    assert BlockTrades(blocks=blocks).violation(s, X) == 2.0


def test_cells_without_a_block_are_unconstrained():
    s = Schedule(lots=((4, 2), (1, 5)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 0.0


def test_violation_is_zero_exactly_when_satisfied():
    c = BlockTrades(blocks=BLOCKS)
    for lots in (((4, 2), (0, 6)), ((3, 3), (0, 6)), ((0, 6), (0, 6))):
        s = Schedule(lots=lots)
        assert (c.violation(s, X) == 0.0) == c.is_satisfied(s, X)


def test_a_block_out_of_range_of_the_schedule_is_rejected():
    s = Schedule(lots=((4, 2), (0, 6)))
    bad = BlockTrades(blocks=(Block(asset=5, bucket=0, lots=4),))
    with pytest.raises(ValueError, match="asset 5"):
        bad.violation(s, X)


def test_two_blocks_on_the_same_cell_are_rejected():
    """They would impose two contradictory all-or-nothing sizes on one variable."""
    with pytest.raises(ValueError, match="more than one block"):
        BlockTrades(blocks=(Block(0, 0, 4), Block(0, 0, 2)))


def test_a_non_positive_block_size_is_rejected():
    with pytest.raises(ValueError, match="lots"):
        Block(asset=0, bucket=0, lots=0)


def test_a_negative_index_is_rejected():
    with pytest.raises(ValueError, match="asset"):
        Block(asset=-1, bucket=0, lots=4)


def test_describe_carries_every_block_as_json_safe_data():
    assert BlockTrades(blocks=BLOCKS).describe() == {
        "name": "block_trades",
        "blocks": [{"asset": 0, "bucket": 0, "lots": 4}],
    }
