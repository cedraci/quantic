"""Which lots each cell may trade -- the search space both exact methods share.

The constraints of spec sections 5.3 and 6.2 act in two ways. Three of them
restrict a single cell: the per-bucket cap, a venue minimum, an
all-or-nothing block. Those become the cell's set of admissible values. The
other two couple cells -- full liquidation along a row, cardinality down a
column -- and are left to the method that searches.

``encoding/layout.py`` derives the same sets for the encoder, and this module
does not import it. The gate compares a QUBO's minimum with the baseline's,
and a baseline that took its search space from the encoder would inherit the
encoder's mistakes and agree with them.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

from quantic.problem.constraints.block_trades import BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.instance import Instance
from quantic.problem.schedule import Schedule


@dataclass(frozen=True)
class LotGrid:
    """Every cell's admissible values, and what couples the cells."""

    values: tuple[tuple[tuple[int, ...], ...], ...]   # [i][t] -> lots, ascending
    initial_lots: tuple[int, ...]
    cardinality: int | None

    @property
    def n_assets(self) -> int:
        return len(self.values)

    @property
    def n_buckets(self) -> int:
        return len(self.values[0])

    def rows(self, i: int) -> tuple[tuple[int, ...], ...]:
        """Every way asset ``i`` can sell its whole position on admissible values."""
        cells = self.values[i]
        # The most the cells from t onward can still sell, so a branch that
        # cannot reach the position is cut where it starts.
        reach = [0] * (len(cells) + 1)
        for t in range(len(cells) - 1, -1, -1):
            reach[t] = reach[t + 1] + max(cells[t])

        out: list[tuple[int, ...]] = []

        def extend(t: int, left: int, prefix: tuple[int, ...]) -> None:
            if t == len(cells):
                if left == 0:
                    out.append(prefix)
                return
            for v in cells[t]:
                if v > left:
                    break
                if left - v > reach[t + 1]:
                    continue
                extend(t + 1, left - v, (*prefix, v))

        extend(0, self.initial_lots[i], ())
        return tuple(out)

    def n_candidates(self) -> int:
        """How many schedules :meth:`schedules` yields."""
        return math.prod(len(self.rows(i)) for i in range(self.n_assets))

    def schedules(self) -> Iterator[Schedule]:
        """Every schedule that liquidates in full on admissible values.

        Cardinality is not applied: it couples the rows, and
        ``problem.feasibility.classify`` is the only verdict on it.
        """
        per_asset = [self.rows(i) for i in range(self.n_assets)]
        for combo in itertools.product(*per_asset):
            yield Schedule(lots=tuple(combo))


def lot_grid(instance: Instance) -> LotGrid:
    """Read the constraint set into a grid. Dispatch is on type, never on a name."""
    minimums = [1] * instance.n_assets
    blocks: dict[tuple[int, int], int] = {}
    cardinality: int | None = None
    liquidates = False

    kinds = [type(c) for c in instance.constraints]
    repeated = sorted({k.__name__ for k in kinds if kinds.count(k) > 1})
    if repeated:
        raise ValueError(
            f"{instance.instance_id} carries more than one {repeated} constraint. "
            "The grid reads each kind once, so a second would silently replace "
            "the first"
        )

    for c in instance.constraints:
        if isinstance(c, FullLiquidation):
            liquidates = True
        elif isinstance(c, Cardinality):
            cardinality = c.k
        elif isinstance(c, MinParticipation):
            if len(c.min_lots) != instance.n_assets:
                raise ValueError(
                    f"min_lots has {len(c.min_lots)} entries but the instance has "
                    f"{instance.n_assets} assets"
                )
            minimums = [int(m) for m in c.min_lots]
        elif isinstance(c, BlockTrades):
            blocks = {(b.asset, b.bucket): int(b.lots) for b in c.blocks}
        else:
            raise NotImplementedError(
                f"lot_grid does not know constraint {c.name!r} ({type(c).__name__}). "
                "A grid that ignored it would make both exact methods agree on the "
                "wrong answer"
            )

    if not liquidates:
        raise ValueError(
            "the instance carries no FullLiquidation constraint. The grid "
            "enumerates rows that sell the whole position, so without it the "
            "search space is not the problem's"
        )

    values = []
    for i in range(instance.n_assets):
        cap = min(instance.max_lots_per_bucket, instance.initial_lots[i])
        row = []
        for t in range(instance.n_buckets):
            block = blocks.get((i, t))
            if block is not None:
                # The block is taken as the venue fixes it, even above the
                # per-bucket cap: the cap bounds what a schedule chooses, a block
                # is a size it does not choose. The generator carves blocks from
                # a witness inside the cap, so the ladder never meets the case.
                # A block below the venue minimum can only stay untraded.
                row.append((0, block) if block >= minimums[i] else (0,))
            elif minimums[i] <= 1:
                row.append(tuple(range(cap + 1)))
            else:
                row.append((0, *range(minimums[i], cap + 1)))
        values.append(tuple(row))

    return LotGrid(
        values=tuple(values),
        initial_lots=tuple(int(x) for x in instance.initial_lots),
        cardinality=cardinality,
    )
