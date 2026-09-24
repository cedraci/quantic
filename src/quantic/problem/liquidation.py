"""The instance ladder -- spec section 5.5.

Four tiers, sized against the hardware ceiling of spec section 3.2: the
liquidation QUBO is dense, and dense minor-embedding fits roughly 230 logical
variables on Advantage2. T2 is deliberately the edge tier and T3 is
hybrid-only by construction.

``bits`` is recorded here, where the ladder is defined in the spec's own
``N x T x B`` terms, and converted to ``max_lots_per_bucket`` for
:class:`~quantic.problem.instance.Instance`, which is encoding-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TierSpec:
    """One rung of the ladder, plus the capacity arithmetic the generator needs."""

    name: str
    n_assets: int
    n_buckets: int
    bits: int

    def __post_init__(self) -> None:
        if self.n_assets < 1:
            raise ValueError(f"{self.name}: n_assets must be at least 1, got {self.n_assets}")
        if self.n_buckets < 1:
            raise ValueError(f"{self.name}: n_buckets must be at least 1, got {self.n_buckets}")
        if self.bits < 1:
            raise ValueError(f"{self.name}: bits must be at least 1, got {self.bits}")

    @property
    def max_lots_per_bucket(self) -> int:
        """The integer range a ``bits``-wide binary expansion covers."""
        return 2**self.bits - 1

    @property
    def approx_variables(self) -> int:
        """``N * T * B`` -- spec section 5.5's sizing column, before auxiliaries."""
        return self.n_assets * self.n_buckets * self.bits

    @property
    def active_per_bucket(self) -> int:
        """How many names may trade in one bucket once dial D2 concentrates.

        Strictly below ``n_assets`` so that the cardinality derived from it
        actually binds. A cardinality equal to the asset count constrains
        nothing, and a dial that does not bind contributes nothing to the
        screening design in spec section 9.2 while still appearing in every
        results row as though it did.
        """
        return max(1, math.ceil(self.n_assets / 2))

    @property
    def buckets_per_asset_cap(self) -> int:
        """How many buckets one asset may spread across under D2.

        ``n_buckets * active_per_bucket`` asset-bucket slots exist in total,
        shared between ``n_assets`` assets. The naive alternative -- letting
        every asset use all ``n_buckets`` -- overruns capacity on **every**
        tier of this ladder, so no feasible witness would exist.
        """
        return max(1, (self.n_buckets * self.active_per_bucket) // self.n_assets)

    def max_position_lots(self, *, discrete_participation: bool) -> int:
        """The largest ``X[i]`` for which a feasible schedule provably exists."""
        buckets = self.buckets_per_asset_cap if discrete_participation else self.n_buckets
        return buckets * self.max_lots_per_bucket


T0 = TierSpec(name="T0", n_assets=3, n_buckets=4, bits=2)     # ~24 variables
T1 = TierSpec(name="T1", n_assets=5, n_buckets=8, bits=3)     # ~120
T2 = TierSpec(name="T2", n_assets=8, n_buckets=8, bits=3)     # ~192, embedding edge
T3 = TierSpec(name="T3", n_assets=30, n_buckets=12, bits=4)   # ~1440, hybrid-only

LADDER: tuple[TierSpec, ...] = (T0, T1, T2, T3)


def by_name(name: str) -> TierSpec:
    for tier in LADDER:
        if tier.name == name:
            return tier
    raise KeyError(f"unknown tier {name!r}; the ladder is {[t.name for t in LADDER]}")
