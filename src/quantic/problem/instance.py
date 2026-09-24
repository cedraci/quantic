"""The frozen problem artifact that crosses into encoding and solvers.

Spec section 7.1 gives every solver the signature ``solve(instance, budget)``,
and section 12 forbids ``solvers/`` reaching into ``problem/`` internals. This
type is therefore the whole interface, and it is deliberately inert: shape,
parameters, dials, constraints, a risk specification and a witness. No
behaviour that a solver could depend on beyond the accessors here.

It carries ``max_lots_per_bucket`` rather than a bit width. Bit width belongs
to an encoding, and spec section 6.1 makes the integer encoding a study
dimension with three competing choices -- binary, one-hot, domain-wall --
which need different variable counts for the same problem.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantic.core.types import Asset
from quantic.problem.constraints.base import Constraint
from quantic.problem.dials import Dials
from quantic.problem.params import MarketParams
from quantic.problem.risk import RiskSpec
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, eq=False)
class Instance:
    """One liquidation problem, fully specified.

    ``eq=False``: ``Instance`` carries a ``MarketParams`` holding an ndarray,
    so the dataclass-generated ``__eq__``/``__hash__`` would inherit
    ``MarketParams``'s defect (see its docstring). ``__eq__`` and ``__hash__``
    are defined explicitly below, keyed on ``content_hash`` -- the property
    that already defines this instance's identity and deliberately excludes
    ``instance_id``.
    """

    instance_id: str
    tier: str
    assets: tuple[Asset, ...]
    n_buckets: int
    initial_lots: tuple[int, ...]
    max_lots_per_bucket: int
    params: MarketParams
    dials: Dials
    constraints: tuple[Constraint, ...]
    risk: RiskSpec
    witness: Schedule
    seed: int

    def __post_init__(self) -> None:
        n = len(self.assets)
        if n == 0:
            raise ValueError("an instance needs at least one asset")
        if self.n_buckets < 1:
            raise ValueError(f"n_buckets must be at least 1, got {self.n_buckets}")
        if self.max_lots_per_bucket < 1:
            raise ValueError(
                f"max_lots_per_bucket must be at least 1, got {self.max_lots_per_bucket}"
            )
        if len(self.initial_lots) != n:
            raise ValueError(
                f"initial_lots has {len(self.initial_lots)} entries but there are {n} assets"
            )
        if any(x < 0 for x in self.initial_lots):
            raise ValueError(f"initial_lots must be non-negative, got {self.initial_lots}")
        if self.params.n_assets != n:
            raise ValueError(
                f"params describes {self.params.n_assets} assets but the instance has {n}"
            )
        if self.params.symbols != tuple(a.symbol for a in self.assets):
            raise ValueError(
                f"symbol mismatch: assets are {tuple(a.symbol for a in self.assets)} but "
                f"params are {self.params.symbols}. Mismatched ordering would misprice "
                "every asset silently"
            )
        if not self.constraints:
            raise ValueError(
                "an instance needs at least one constraint; without full liquidation the "
                "trivial empty schedule is optimal"
            )
        if (self.witness.n_assets, self.witness.n_buckets) != (n, self.n_buckets):
            raise ValueError(
                f"witness has shape ({self.witness.n_assets}, {self.witness.n_buckets}) "
                f"but the instance is ({n}, {self.n_buckets})"
            )
        worst = int(self.witness.as_array().max())
        if worst > self.max_lots_per_bucket:
            raise ValueError(
                f"witness trades {worst} lots in a bucket but max_lots_per_bucket is "
                f"{self.max_lots_per_bucket}"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instance):
            return NotImplemented
        return self.content_hash == other.content_hash

    def __hash__(self) -> int:
        return hash(self.content_hash)

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def lot_sizes(self) -> np.ndarray:
        return np.array([a.lot_size for a in self.assets], dtype=np.int64)

    def remaining_lots(self, schedule: Schedule) -> np.ndarray:
        """``(N, T)`` lots still held after each bucket -- spec section 5.1's ``h``."""
        if (schedule.n_assets, schedule.n_buckets) != (self.n_assets, self.n_buckets):
            raise ValueError(
                f"schedule shape ({schedule.n_assets}, {schedule.n_buckets}) does not match "
                f"the instance ({self.n_assets}, {self.n_buckets})"
            )
        return np.array(self.initial_lots, dtype=np.int64)[:, None] - schedule.cumulative()

    def notional(self) -> float:
        """Total currency value of the position, for reporting cost in bps."""
        return float(
            sum(
                self.initial_lots[i] * self.assets[i].lot_size * self.params.assets[i].price
                for i in range(self.n_assets)
            )
        )

    def _canonical_payload(self) -> dict[str, Any]:
        """Everything that defines the instance's content, JSON-safe and ordered.

        ``instance_id`` is deliberately excluded: it is a human-facing label,
        and two instances with identical content must collide so a duplicate
        is detectable.
        """
        return {
            "tier": self.tier,
            "n_buckets": self.n_buckets,
            "initial_lots": list(self.initial_lots),
            "max_lots_per_bucket": self.max_lots_per_bucket,
            "seed": self.seed,
            "assets": [
                {
                    "symbol": a.symbol,
                    "lot_size": a.lot_size,
                    "tick_size": a.tick_size,
                    "currency": a.currency,
                }
                for a in self.assets
            ],
            "params": {
                "bucket_ns": self.params.bucket_ns,
                "assets": [
                    {
                        "symbol": a.symbol,
                        "delta": a.delta,
                        "y_coef": a.y_coef,
                        "gamma": a.gamma,
                        "sigma_bucket": a.sigma_bucket,
                        "bucket_volume_shares": a.bucket_volume_shares,
                        "price": a.price,
                        "half_spread": a.half_spread,
                    }
                    for a in self.params.assets
                ],
                "covariance": np.asarray(self.params.covariance, dtype=float).tolist(),
            },
            "dials": {
                "concave_impact": self.dials.concave_impact,
                "discrete_participation": self.dials.discrete_participation,
                "cvar_risk": self.dials.cvar_risk,
                "block_trades": self.dials.block_trades,
            },
            "constraints": [c.describe() for c in self.constraints],
            "risk": self.risk.describe(),
            "witness": [list(row) for row in self.witness.lots],
        }

    @property
    def content_hash(self) -> str:
        """SHA-256 over the instance's content.

        A computed property rather than a stored field: a stored hash can go
        stale against the fields it summarises, and a results row pins it, so
        a stale hash would attribute a published figure to the wrong instance.

        Python's ``repr`` of a float round-trips exactly, and ``json.dumps``
        uses it, so no rounding is applied and no precision is lost.
        """
        payload = json.dumps(
            self._canonical_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
