"""Synthetic market data with known ground-truth impact parameters.

This module is the keystone of the test strategy: because the true impact
exponent and coefficient are known by construction, calibration can be tested
by parameter recovery rather than by inspection. See spec section 10.4.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

_SESSION_OPEN_SEC = 9 * 3600 + 30 * 60  # 09:30
_SESSION_SECONDS = 23_400  # 6.5 hours
_NS = 1_000_000_000

_DEFAULT_DELTAS = (0.45, 0.50, 0.60)
_DEFAULT_YS = (0.70, 0.80, 0.90)


@dataclass(frozen=True)
class SynthConfig:
    symbols: tuple[str, ...] = ("SYNA", "SYNB", "SYNC")
    n_days: int = 20
    buckets_per_day: int = 13
    seed: int = 0
    start_date: dt.date = dt.date(2026, 1, 5)
    base_price: float = 100.0
    daily_vol: float = 0.02
    adv_shares: int = 5_000_000
    tick_size: float = 0.01
    lot_size: int = 100
    depth_levels: int = 10
    level_size: int = 500
    flow_sd: float = 0.08
    # Diffusion noise as a multiple of sigma_bucket. The default of 1.0 is the
    # realistic regime: mid returns are dominated by diffusion and impact is a
    # small component, so realised bucket volatility tracks daily_vol. Tests
    # that need a clean signal lower it explicitly.
    noise_frac: float = 1.00
    impact_delta: Mapping[str, float] | None = None
    impact_Y: Mapping[str, float] | None = None

    @property
    def sigma_bucket(self) -> float:
        return self.daily_vol / np.sqrt(self.buckets_per_day)


def ground_truth(cfg: SynthConfig) -> dict[str, Any]:
    deltas = dict(cfg.impact_delta) if cfg.impact_delta else {
        s: _DEFAULT_DELTAS[i % len(_DEFAULT_DELTAS)] for i, s in enumerate(cfg.symbols)
    }
    ys = dict(cfg.impact_Y) if cfg.impact_Y else {
        s: _DEFAULT_YS[i % len(_DEFAULT_YS)] for i, s in enumerate(cfg.symbols)
    }
    missing = set(cfg.symbols) - set(deltas) | set(cfg.symbols) - set(ys)
    if missing:
        raise ValueError(f"ground truth missing for symbols {sorted(missing)}")
    return {
        "impact_delta": deltas,
        "impact_Y": ys,
        "daily_vol": cfg.daily_vol,
        "buckets_per_day": cfg.buckets_per_day,
        "sigma_bucket": float(cfg.sigma_bucket),
    }


def trading_dates(cfg: SynthConfig) -> list[dt.date]:
    dates: list[dt.date] = []
    d = cfg.start_date
    while len(dates) < cfg.n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)
    return dates


def bucket_ts_ns(d: dt.date, bucket: int, buckets_per_day: int) -> tuple[int, int]:
    midnight = int(
        dt.datetime(d.year, d.month, d.day, tzinfo=dt.UTC).timestamp()
    ) * _NS
    length = _SESSION_SECONDS // buckets_per_day
    start = midnight + (_SESSION_OPEN_SEC + bucket * length) * _NS
    end = midnight + (_SESSION_OPEN_SEC + (bucket + 1) * length) * _NS
    if bucket == buckets_per_day - 1:
        end = midnight + (_SESSION_OPEN_SEC + _SESSION_SECONDS) * _NS
    return start, end


def round_to_tick(px: float, tick: float) -> float:
    return round(round(px / tick) * tick, 10)


def generate_buckets(cfg: SynthConfig) -> pl.DataFrame:
    gt = ground_truth(cfg)
    dates = trading_dates(cfg)
    sigma_b = cfg.sigma_bucket
    rows: list[dict[str, Any]] = []

    for sym_idx, symbol in enumerate(cfg.symbols):
        # One independent stream per symbol keeps a symbol's path stable when
        # other symbols are added or removed from the config.
        rng = np.random.default_rng([cfg.seed, sym_idx])
        delta = gt["impact_delta"][symbol]
        y_coef = gt["impact_Y"][symbol]
        mid = cfg.base_price

        for day_index, date in enumerate(dates):
            for bucket_index in range(cfg.buckets_per_day):
                f = 0.0
                while abs(f) < 1e-4:
                    f = float(np.clip(rng.normal(0.0, cfg.flow_sd), -0.3, 0.3))

                volume = int(
                    round(cfg.adv_shares / cfg.buckets_per_day * rng.lognormal(0.0, 0.15))
                )
                impact = y_coef * sigma_b * np.sign(f) * abs(f) ** delta
                noise = rng.normal(0.0, cfg.noise_frac * sigma_b)
                mid_open = mid
                mid_close = mid_open * (1.0 + impact + noise)
                mid = mid_close

                ts_start, ts_end = bucket_ts_ns(date, bucket_index, cfg.buckets_per_day)
                rows.append(
                    {
                        "symbol": symbol,
                        "day_index": day_index,
                        "bucket_index": bucket_index,
                        "date": date,
                        "ts_start_ns": ts_start,
                        "ts_end_ns": ts_end,
                        "mid_open": mid_open,
                        "mid_close": mid_close,
                        "bucket_volume": volume,
                        "net_flow": f * volume,
                        "participation": f,
                    }
                )

    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "day_index": pl.Int64,
            "bucket_index": pl.Int64,
            "date": pl.Date,
            "ts_start_ns": pl.Int64,
            "ts_end_ns": pl.Int64,
            "mid_open": pl.Float64,
            "mid_close": pl.Float64,
            "bucket_volume": pl.Int64,
            "net_flow": pl.Float64,
            "participation": pl.Float64,
        },
    ).sort(["symbol", "day_index", "bucket_index"])


def build_daily_bars(buckets: pl.DataFrame, cfg: SynthConfig) -> pl.DataFrame:
    daily = (
        buckets.sort(["symbol", "day_index", "bucket_index"])
        .group_by(["symbol", "date"], maintain_order=True)
        .agg(
            pl.col("mid_open").first().alias("open"),
            pl.max_horizontal(pl.col("mid_open"), pl.col("mid_close")).max().alias("high"),
            pl.min_horizontal(pl.col("mid_open"), pl.col("mid_close")).min().alias("low"),
            pl.col("mid_close").last().alias("close"),
            pl.col("bucket_volume").sum().alias("volume"),
        )
        .sort(["symbol", "date"])
    )
    return (
        daily.with_columns(
            pl.col("volume")
            .cast(pl.Float64)
            .rolling_mean(window_size=20, min_samples=1)
            .over("symbol")
            .alias("adv")
        )
        .select("date", "symbol", "open", "high", "low", "close", "volume", "adv")
        .with_columns(pl.col("volume").cast(pl.Int64))
    )


def build_l1(buckets: pl.DataFrame, cfg: SynthConfig) -> pl.DataFrame:
    tick = cfg.tick_size
    mid_tick = (buckets["mid_close"] / tick).round(0)
    bid = ((mid_tick - 1) * tick).round(10)
    ask = ((mid_tick + 1) * tick).round(10)
    return pl.DataFrame(
        {
            "ts_ns": buckets["ts_end_ns"],
            "symbol": buckets["symbol"],
            "bid": bid,
            "ask": ask,
            "bid_size": pl.Series([cfg.level_size] * buckets.height, dtype=pl.Int64),
            "ask_size": pl.Series([cfg.level_size] * buckets.height, dtype=pl.Int64),
            "last_px": (mid_tick * tick).round(10),
            "last_size": buckets["net_flow"].abs().round(0).cast(pl.Int64),
        }
    ).select("ts_ns", "symbol", "bid", "ask", "bid_size", "ask_size", "last_px", "last_size")
