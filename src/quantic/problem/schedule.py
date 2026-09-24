"""The decision variable: how many lots of each asset trade in each bucket.

Spec section 5.1 writes ``x[i,t]`` in **lots**, and this type is the project's
only representation of it. Everything in ``micro/`` is in shares; the
conversion happens exactly once, in ``problem.objective``, via
``Asset.lot_size``. No other module converts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Schedule:
    """An ``N x T`` grid of lots, immutable and hashable.

    A tuple of tuples rather than an array because :class:`Instance` is frozen
    and content-hashed, and the largest tier is 30 x 12 = 360 integers. Use
    :meth:`as_array` for numeric work.

    Non-negativity (spec section 5.3) is enforced here rather than checked
    later, so no consumer has to wonder whether a negative entry is possible.
    """

    lots: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.lots or not self.lots[0]:
            raise ValueError("schedule is empty; it needs at least one asset and one bucket")
        width = len(self.lots[0])
        for i, row in enumerate(self.lots):
            if len(row) != width:
                raise ValueError(
                    f"schedule must be rectangular: row 0 has {width} buckets but row {i} "
                    f"has {len(row)}"
                )
            for t, value in enumerate(row):
                if value < 0:
                    raise ValueError(
                        f"schedule has a negative entry at asset {i}, bucket {t}: {value}. "
                        "Liquidation quantities are non-negative (spec section 5.3)"
                    )

    @classmethod
    def from_array(cls, array: np.ndarray) -> Schedule:
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"schedule array must be 2-D, got shape {arr.shape}")
        return cls(lots=tuple(tuple(int(v) for v in row) for row in arr))

    @property
    def n_assets(self) -> int:
        return len(self.lots)

    @property
    def n_buckets(self) -> int:
        return len(self.lots[0])

    def sold(self, i: int) -> int:
        """Total lots of asset ``i`` traded across every bucket."""
        return sum(self.lots[i])

    def active_names(self, t: int) -> int:
        """How many assets trade a non-zero quantity in bucket ``t``."""
        return sum(1 for row in self.lots if row[t] > 0)

    def as_array(self) -> np.ndarray:
        return np.array(self.lots, dtype=np.int64)

    def cumulative(self) -> np.ndarray:
        """``(N, T)`` cumulative lots **inclusive** of each bucket."""
        return np.cumsum(self.as_array(), axis=1)

    def cumulative_before(self) -> np.ndarray:
        """``(N, T)`` cumulative lots **strictly before** each bucket; column 0 is zero.

        This is the form the permanent impact term needs: at the moment we
        trade in bucket ``t`` we have only caused the displacement from
        everything before ``t``.
        """
        return self.cumulative() - self.as_array()
