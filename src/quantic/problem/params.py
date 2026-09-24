"""Per-asset market inputs to the liquidation objective.

Units are named in the field names, deliberately. Three volatility time bases
coexist in this project -- a per-bucket sigma from calibration, a daily
``SynthConfig.daily_vol``, and an annualised ``CovarianceEstimate.matrix`` --
and an objective built on the wrong one is wrong by a factor of ~sqrt(13) or
~sqrt(3276) while remaining entirely plausible.

``covariance`` is a **log-return** covariance at the **bucket** horizon.
:meth:`MarketParams.price_covariance` converts it to the price-change
covariance the Almgren-Chriss risk term actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.micro.impact.base import ImpactParams

# Ledoit-Wolf output is PSD up to floating-point noise, so an exactly-zero
# floor would reject valid estimates. The floor is scaled by the matrix's own
# largest eigenvalue, with NO absolute fallback: a bucket-horizon log-return
# covariance has eigenvalues around 1e-5, so clamping the scale to a minimum
# of 1.0 would turn this into a fixed -1e-10 floor and admit a genuinely
# negative eigenvalue five orders of magnitude above the noise. `eigvalsh`
# rounding noise is about eps*|A| ~ 2.2e-16 relative, so a 1e-10 relative
# floor still leaves ~450,000x headroom against false rejection.
_PSD_RTOL = 1e-10


@dataclass(frozen=True, slots=True)
class AssetParams:
    """Everything the objective needs about one asset, with units in the names.

    ``half_spread`` is spec section 5.2's ``s_i / 2``, in currency per share.
    ``gamma`` is the permanent impact coefficient. ``delta`` and ``y_coef``
    are the power-law parameters; ``delta`` must lie in ``(0, 1]``, matching
    ``micro.impact.sqrt_law.PowerLawImpact``.
    """

    symbol: str
    delta: float
    y_coef: float
    gamma: float
    sigma_bucket: float
    bucket_volume_shares: float
    price: float
    half_spread: float

    def __post_init__(self) -> None:
        if not 0.0 < self.delta <= 1.0:
            raise ValueError(
                f"{self.symbol}: delta must lie in (0, 1], got {self.delta}. Above 1 is a "
                "convex impact law, which hardness dial D1 does not model"
            )
        if self.y_coef < 0:
            raise ValueError(f"{self.symbol}: y_coef must be non-negative, got {self.y_coef}")
        if self.gamma < 0:
            raise ValueError(f"{self.symbol}: gamma must be non-negative, got {self.gamma}")
        if self.sigma_bucket < 0:
            raise ValueError(
                f"{self.symbol}: sigma_bucket must be non-negative, got {self.sigma_bucket}"
            )
        if self.bucket_volume_shares <= 0:
            raise ValueError(
                f"{self.symbol}: bucket_volume_shares must be positive, got "
                f"{self.bucket_volume_shares}"
            )
        if self.price <= 0:
            raise ValueError(f"{self.symbol}: price must be positive, got {self.price}")
        if self.half_spread < 0:
            raise ValueError(
                f"{self.symbol}: half_spread must be non-negative, got {self.half_spread}"
            )


@dataclass(frozen=True)
class MarketParams:
    """Per-asset parameters plus the cross-asset covariance.

    Not ``slots=True``: it holds a numpy array, and the frozen-dataclass
    machinery is enough here.
    """

    assets: tuple[AssetParams, ...]
    covariance: np.ndarray
    bucket_ns: int

    def __post_init__(self) -> None:
        if not self.assets:
            raise ValueError("MarketParams needs at least one asset")
        if self.bucket_ns <= 0:
            raise ValueError(f"bucket_ns must be positive, got {self.bucket_ns}")

        symbols = [a.symbol for a in self.assets]
        duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate symbols in MarketParams: {duplicates}. Two rows for one symbol "
                "means the covariance rows cannot be matched to assets unambiguously"
            )

        cov = np.asarray(self.covariance, dtype=float)
        n = len(self.assets)
        if cov.shape != (n, n):
            raise ValueError(
                f"covariance shape {cov.shape} does not match the {n}-asset dimension"
            )
        if not np.allclose(cov, cov.T, rtol=1e-9, atol=1e-18):
            raise ValueError("covariance must be symmetric")

        eigenvalues = np.linalg.eigvalsh(cov)
        floor = -_PSD_RTOL * float(np.max(np.abs(eigenvalues)))
        if float(eigenvalues.min()) < floor:
            raise ValueError(
                f"covariance is not positive semi-definite (smallest eigenvalue "
                f"{eigenvalues.min():.3e}). A non-PSD covariance makes the risk term "
                "unbounded below, so every optimum derived from it is meaningless"
            )

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(a.symbol for a in self.assets)

    def prices(self) -> np.ndarray:
        return np.array([a.price for a in self.assets], dtype=float)

    def impact_params(self, i: int) -> ImpactParams:
        """Build ``micro``'s per-bucket impact input for asset ``i``.

        The single place ``ImpactParams`` is constructed in this layer, which
        keeps the unit contract in one function.
        """
        a = self.assets[i]
        return ImpactParams(
            symbol=a.symbol,
            sigma_bucket=a.sigma_bucket,
            bucket_ns=self.bucket_ns,
            bucket_volume_shares=a.bucket_volume_shares,
            price=a.price,
        )

    def price_covariance(self) -> np.ndarray:
        """``diag(price) @ covariance @ diag(price)``.

        The stored covariance is of log returns and is dimensionless; the risk
        term needs a covariance of price changes so that ``h' Sigma h`` with
        ``h`` in shares comes out in currency squared.
        """
        p = self.prices()
        return np.asarray(self.covariance, dtype=float) * np.outer(p, p)

    def subset(self, n: int) -> MarketParams:
        """The first ``n`` assets and the matching covariance block."""
        if n > self.n_assets:
            raise ValueError(
                f"cannot take {n} assets from MarketParams holding {self.n_assets} assets"
            )
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        cov = np.asarray(self.covariance, dtype=float)[:n, :n]
        return MarketParams(assets=self.assets[:n], covariance=cov, bucket_ns=self.bucket_ns)
