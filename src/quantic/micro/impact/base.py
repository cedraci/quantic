"""Impact model protocol and unit conventions.

Every quantity crossing the M1 -> M2 boundary names its units *in the type*.
This is not decoration. Three volatility time bases coexisted on this branch
-- a per-bucket sigma from calibration, a daily ``SynthConfig.daily_vol``, and
an annualised ``CovarianceEstimate.matrix`` -- and none of them was named
anywhere a caller could see it. An objective built on the wrong one is wrong
by a factor of ~sqrt(13) or ~sqrt(3276) while remaining entirely plausible.

Units, fixed once and relied on everywhere:

* ``sigma_bucket`` is the **fractional** standard deviation of the mid over
  **one bucket** of length ``bucket_ns``. Not daily, not annualised. Use
  :meth:`ImpactParams.rescale_to` to move between bucket widths.
* ``bucket_volume_shares`` is total traded volume in that bucket, in
  **shares**. Spec section 5.2 writes participation in notional; the whole
  code path is in shares, so the conversion is made explicit through
  :meth:`ImpactParams.notional` and :attr:`bucket_volume_notional` rather
  than left to whichever value a caller happened to have to hand.
* ``q`` throughout is a signed quantity in **shares**. Models take its
  absolute value; the caller owns the sign.
* :meth:`ImpactParams.participation` is the only sanctioned way to form
  ``|q| / V``, so a notional ``q`` can never meet a share ``V``.
* ``price_impact`` returns a **fractional** mid displacement, always
  non-negative.
* ``temporary_cost`` returns **fractional cost**: currency cost divided by
  ``|q| * price``.
* Integrating a power-law impact over the executed quantity gives
  ``temporary_cost = price_impact / (delta + 1)`` -- the familiar 2/3 rule at
  ``delta = 0.5`` and 1/2 for linear impact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ImpactParams:
    """Per-symbol, per-bucket inputs to an impact model.

    ``sigma_bucket`` and ``bucket_volume_shares`` both describe the *same*
    bucket, whose width ``bucket_ns`` is carried alongside them so the horizon
    is never implicit.
    """

    symbol: str
    sigma_bucket: float
    bucket_ns: int
    bucket_volume_shares: float
    price: float

    def __post_init__(self) -> None:
        if self.bucket_volume_shares <= 0:
            raise ValueError(
                f"bucket_volume_shares must be positive, got {self.bucket_volume_shares}"
            )
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if self.sigma_bucket < 0:
            raise ValueError(f"sigma_bucket must be non-negative, got {self.sigma_bucket}")
        if self.bucket_ns <= 0:
            raise ValueError(f"bucket_ns must be positive, got {self.bucket_ns}")

    def participation(self, q: float) -> float:
        """``|q| / V``, both in shares.

        The single sanctioned way to form participation. Doing the division at
        the call site is how a notional ``q`` ends up over a share ``V``.
        """
        return abs(q) / self.bucket_volume_shares

    def notional(self, q: float) -> float:
        """Currency value of ``q`` shares at this bucket's price."""
        return abs(q) * self.price

    @property
    def bucket_volume_notional(self) -> float:
        return self.bucket_volume_shares * self.price

    def rescale_to(self, *, bucket_ns: int) -> ImpactParams:
        """The same symbol and bucket, restated at a different bucket width.

        Volatility scales as the square root of time; volume is a flow and
        scales linearly. Both conversions are wrong in different directions,
        which is exactly why neither should be done by hand at a call site.
        """
        if bucket_ns <= 0:
            raise ValueError(f"bucket_ns must be positive, got {bucket_ns}")
        ratio = bucket_ns / self.bucket_ns
        return replace(
            self,
            bucket_ns=bucket_ns,
            sigma_bucket=self.sigma_bucket * math.sqrt(ratio),
            bucket_volume_shares=self.bucket_volume_shares * ratio,
        )


@runtime_checkable
class ImpactModel(Protocol):
    name: str

    def price_impact(self, q: float, p: ImpactParams) -> float: ...

    def temporary_cost(self, q: float, p: ImpactParams) -> float: ...

    def permanent_impact(self, q: float, p: ImpactParams) -> float: ...


def currency_cost(model: ImpactModel, q: float, p: ImpactParams) -> float:
    return model.temporary_cost(q, p) * p.notional(q)


def cost_in_bps(model: ImpactModel, q: float, p: ImpactParams) -> float:
    return model.temporary_cost(q, p) * 10_000.0
