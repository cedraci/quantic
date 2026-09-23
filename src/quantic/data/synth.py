"""Synthetic market data with known ground-truth impact parameters.

This module is the keystone of the test strategy: because the true impact
exponent and coefficient are known by construction, calibration can be tested
by parameter recovery rather than by inspection. See spec section 10.4.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quantic.data.bundle import DatasetBundle
from quantic.data.schemas import L3Action

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


class _GeneratorBook:
    """A deliberately naive book used only while emitting messages.

    ``micro.book_reconstruct`` is an independent implementation; comparing the
    two is the Task 10 correctness gate, so this class must stay simple and
    must never import from ``quantic.micro``.
    """

    def __init__(self) -> None:
        # side -> price -> order_id -> size
        self.levels: dict[str, dict[float, dict[int, int]]] = {"buy": {}, "sell": {}}

    def prices(self, side: str) -> list[float]:
        return sorted(
            (p for p, orders in self.levels[side].items() if sum(orders.values()) > 0),
            reverse=(side == "buy"),
        )

    def size_at(self, side: str, px: float) -> int:
        return sum(self.levels[side].get(px, {}).values())

    def add(self, side: str, px: float, order_id: int, size: int) -> None:
        self.levels[side].setdefault(px, {})[order_id] = size

    def remove(self, side: str, px: float, order_id: int) -> None:
        self.levels[side].get(px, {}).pop(order_id, None)

    def reduce(self, side: str, px: float, order_id: int, size: int) -> None:
        orders = self.levels[side][px]
        orders[order_id] -= size
        if orders[order_id] <= 0:
            del orders[order_id]


class _MessageLog:
    """Accumulates L3 messages for one symbol with monotone sequence numbers.

    A class rather than a closure defined inside the bucket loop: closures over
    loop-local state are what ruff's B023 warns about, and this is also easier
    to reason about.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.rows: list[dict[str, Any]] = []
        self.seq = 0
        self._ts_base = 0
        self._offset = 0

    def start_bucket(self, ts_base: int) -> None:
        self._ts_base = ts_base
        self._offset = 0

    def emit(self, action: str, side: str, px: float, size: int, order_id: int) -> None:
        self.rows.append(
            {
                "ts_ns": self._ts_base + self._offset,
                "seq": self.seq,
                "symbol": self.symbol,
                "order_id": order_id,
                "action": action,
                "side": side,
                "px": px,
                "size": size,
            }
        )
        self.seq += 1
        self._offset += 1


def build_l3_and_l2(buckets: pl.DataFrame, cfg: SynthConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    tick = cfg.tick_size
    replace_every = 3
    msgs: list[dict[str, Any]] = []
    snaps: list[dict[str, Any]] = []

    for symbol in cfg.symbols:
        book = _GeneratorBook()
        log = _MessageLog(symbol)
        next_order_id = 1
        sub = buckets.filter(pl.col("symbol") == symbol).sort(["day_index", "bucket_index"])

        for n, row in enumerate(sub.iter_rows(named=True)):
            log.start_bucket(row["ts_start_ns"])

            mid_tick = round(row["mid_close"] / tick)
            desired = {
                "buy": [round((mid_tick - 1 - k) * tick, 10) for k in range(cfg.depth_levels)],
                "sell": [round((mid_tick + 1 + k) * tick, 10) for k in range(cfg.depth_levels)],
            }

            # 1. cancel orders that left the desired price set
            for side in ("buy", "sell"):
                keep = set(desired[side])
                for px in sorted(book.levels[side]):
                    if px in keep:
                        continue
                    for order_id, size in sorted(book.levels[side][px].items()):
                        log.emit(L3Action.CANCEL.value, side, px, size, order_id)
                    book.levels[side][px] = {}

            # 2. add missing levels
            for side in ("buy", "sell"):
                for px in desired[side]:
                    if book.size_at(side, px) == 0:
                        log.emit(L3Action.ADD.value, side, px, cfg.level_size, next_order_id)
                        book.add(side, px, next_order_id, cfg.level_size)
                        next_order_id += 1

            # 3. execute against the resting side
            hit_side = "sell" if row["net_flow"] > 0 else "buy"
            remaining = int(abs(round(row["net_flow"])))
            for px in book.prices(hit_side):
                if remaining <= 0:
                    break
                for order_id, size in sorted(book.levels[hit_side][px].items()):
                    if remaining <= 0:
                        break
                    taken = min(size, remaining)
                    log.emit(L3Action.EXECUTE.value, hit_side, px, taken, order_id)
                    book.reduce(hit_side, px, order_id, taken)
                    remaining -= taken

            # 4. replenish depleted levels
            for side in ("buy", "sell"):
                for px in desired[side]:
                    short = cfg.level_size - book.size_at(side, px)
                    if short > 0:
                        log.emit(L3Action.ADD.value, side, px, short, next_order_id)
                        book.add(side, px, next_order_id, short)
                        next_order_id += 1

            # 5. exercise the replace path on the deepest bid
            if n % replace_every == 0:
                px = desired["buy"][-1]
                order_id = sorted(book.levels["buy"][px])[0]
                new_size = cfg.level_size + 100
                log.emit(L3Action.REPLACE.value, "buy", px, new_size, order_id)
                book.levels["buy"][px] = {order_id: new_size}

            # 6. snapshot
            snap_ts = row["ts_end_ns"] - 1
            for side in ("buy", "sell"):
                for level, px in enumerate(book.prices(side)[: cfg.depth_levels]):
                    snaps.append(
                        {
                            "ts_ns": snap_ts,
                            "symbol": symbol,
                            "side": side,
                            "level": level,
                            "px": px,
                            "size": book.size_at(side, px),
                        }
                    )

        msgs.extend(log.rows)

    l3 = pl.DataFrame(
        msgs,
        schema={
            "ts_ns": pl.Int64,
            "seq": pl.Int64,
            "symbol": pl.Utf8,
            "order_id": pl.Int64,
            "action": pl.Utf8,
            "side": pl.Utf8,
            "px": pl.Float64,
            "size": pl.Int64,
        },
    ).select("ts_ns", "seq", "symbol", "order_id", "action", "side", "px", "size")

    l2 = pl.DataFrame(
        snaps,
        schema={
            "ts_ns": pl.Int64,
            "symbol": pl.Utf8,
            "side": pl.Utf8,
            "level": pl.Int32,
            "px": pl.Float64,
            "size": pl.Int64,
        },
    ).select("ts_ns", "symbol", "side", "level", "px", "size")

    return l3, l2


def _config_payload(cfg: SynthConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["start_date"] = cfg.start_date.isoformat()
    payload["symbols"] = list(cfg.symbols)
    payload["impact_delta"] = dict(cfg.impact_delta) if cfg.impact_delta else None
    payload["impact_Y"] = dict(cfg.impact_Y) if cfg.impact_Y else None
    return payload


def generate_bundle(
    root: Path, cfg: SynthConfig, *, bundle_id: str | None = None
) -> DatasetBundle:
    buckets = generate_buckets(cfg)
    l3, l2 = build_l3_and_l2(buckets, cfg)
    tables = {
        "daily_bars": build_daily_bars(buckets, cfg),
        "l1_taq": build_l1(buckets, cfg),
        "l2_depth": l2,
        "l3_messages": l3,
    }
    return DatasetBundle.write(
        root,
        tables,
        bundle_id=bundle_id or f"synth-seed{cfg.seed}-{cfg.n_days}d",
        provenance="synthetic",
        extra={"ground_truth": ground_truth(cfg), "synth_config": _config_payload(cfg)},
    )
