import itertools
from dataclasses import dataclass, field

import pytest

from quantic.encoding.integer import ENCODINGS
from quantic.encoding.layout import plan_layout
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.feasibility import classify
from quantic.problem.schedule import Schedule
from quantic.solvers.classical.lot_grid import lot_grid
from tests.ladder import gate_ladder
from tests.solvers.instances import (
    blocks,
    discrete,
    instance,
    nothing_to_sell_in_one_name,
    plain,
)


def test_a_plain_cell_ranges_up_to_the_cap_or_the_position():
    grid = lot_grid(instance(initial_lots=(4, 2), witness=Schedule(lots=((2, 1, 1), (1, 1, 0)))))
    assert grid.values[0] == ((0, 1, 2, 3),) * 3      # the cap of three binds
    assert grid.values[1] == ((0, 1, 2),) * 3         # the position of two binds


def test_a_venue_minimum_removes_the_values_below_it_and_keeps_zero():
    grid = lot_grid(discrete())
    assert grid.values[0] == ((0, 2, 3),) * 3
    assert grid.values[1] == ((0, 1, 2, 3),) * 3
    assert grid.cardinality == 1


def test_a_block_cell_trades_the_block_or_nothing():
    grid = lot_grid(blocks())
    assert grid.values[0][0] == (0, 3)
    assert grid.values[0][1] == (0, 1, 2, 3)
    assert grid.cardinality is None


def test_a_block_below_the_venue_minimum_can_only_stay_untraded():
    inst = instance(
        constraints=(
            FullLiquidation(),
            MinParticipation(min_lots=(3, 1)),
            BlockTrades(blocks=(Block(asset=0, bucket=1, lots=2),)),
        ),
        witness=Schedule(lots=((3, 0, 1), (1, 1, 1))),
    )
    assert lot_grid(inst).values[0][1] == (0,)


def test_a_name_with_nothing_to_sell_has_one_row_and_it_is_empty():
    grid = lot_grid(nothing_to_sell_in_one_name())
    assert grid.values[1] == ((0,),) * 3
    assert grid.rows(1) == ((0, 0, 0),)


def test_the_rows_are_every_way_to_sell_the_position():
    grid = lot_grid(plain())
    expected = {
        r for r in itertools.product(range(4), repeat=3) if sum(r) == 4
    }
    assert set(grid.rows(0)) == expected
    assert len(grid.rows(0)) == len(expected) == 12


def test_no_row_is_listed_twice():
    grid = lot_grid(discrete())
    for i in range(grid.n_assets):
        assert len(set(grid.rows(i))) == len(grid.rows(i))


def test_a_position_the_cells_cannot_reach_has_no_rows():
    inst = instance(
        n_buckets=2, initial_lots=(7, 3), max_lots_per_bucket=3,
        witness=Schedule(lots=((3, 3), (2, 1))),
    )
    grid = lot_grid(inst)
    assert grid.rows(0) == ()
    assert grid.n_candidates() == 0
    assert list(grid.schedules()) == []


@pytest.mark.parametrize("build", [plain, discrete, blocks, nothing_to_sell_in_one_name])
def test_the_count_is_the_number_of_schedules_yielded(build):
    grid = lot_grid(build())
    assert grid.n_candidates() == sum(1 for _ in grid.schedules())


@pytest.mark.parametrize("build", [plain, discrete, blocks])
def test_every_schedule_yielded_breaks_nothing_but_cardinality(build):
    """Cardinality couples the rows and is left to classify, by design."""
    inst = build()
    for schedule in lot_grid(inst).schedules():
        breaches = {
            name for name, size in classify(inst, schedule).violations.items() if size
        }
        assert breaches <= {"cardinality"}


@pytest.mark.parametrize("build", [plain, discrete, blocks])
def test_no_feasible_schedule_is_missing_from_the_grid(build):
    """The grid is the search space, so what it leaves out is never found."""
    inst = build()
    cap = inst.max_lots_per_bucket
    cells = inst.n_assets * inst.n_buckets
    feasible = set()
    for flat in itertools.product(range(cap + 1), repeat=cells):
        rows = tuple(
            tuple(flat[i * inst.n_buckets:(i + 1) * inst.n_buckets])
            for i in range(inst.n_assets)
        )
        schedule = Schedule(lots=rows)
        if classify(inst, schedule).feasible:
            feasible.add(schedule)
    assert feasible <= set(lot_grid(inst).schedules())
    assert feasible


def test_a_constraint_the_grid_does_not_know_is_refused():
    @dataclass(frozen=True)
    class SellOnTuesdays:
        name: str = field(default="sell_on_tuesdays")

        def violation(self, schedule, initial_lots):
            return 0.0

        def is_satisfied(self, schedule, initial_lots):
            return True

        def describe(self):
            return {"name": self.name}

    inst = instance(constraints=(FullLiquidation(), SellOnTuesdays()))
    with pytest.raises(NotImplementedError, match="sell_on_tuesdays"):
        lot_grid(inst)


def test_an_instance_that_need_not_liquidate_is_refused():
    inst = instance(constraints=(MinParticipation(min_lots=(1, 1)),))
    with pytest.raises(ValueError, match="FullLiquidation"):
        lot_grid(inst)


def test_a_venue_minimum_for_the_wrong_number_of_names_is_refused():
    inst = instance(
        constraints=(FullLiquidation(), MinParticipation(min_lots=(1, 1, 1)))
    )
    with pytest.raises(ValueError, match="min_lots"):
        lot_grid(inst)


def test_the_grid_and_the_encoder_agree_on_every_cell_of_the_ladder():
    """Two derivations of one search space, neither importing the other.

    If they differed the gate would be comparing a QUBO's minimum over one
    set of schedules with a baseline's over another.
    """
    differing = {}
    for inst in gate_ladder():
        grid = lot_grid(inst)
        for encoding in ENCODINGS:
            layout = plan_layout(inst, encoding)
            for plan in layout.cells:
                mine = grid.values[plan.asset][plan.bucket]
                if tuple(plan.values) != mine:
                    differing[(inst.instance_id, encoding.name, plan.asset, plan.bucket)] = (
                        mine, tuple(plan.values)
                    )
    assert not differing, differing
