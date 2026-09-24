import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantic.data.synth import (
    SynthConfig,
    bucket_ts_ns,
    build_daily_bars,
    build_l1,
    generate_buckets,
    ground_truth,
    round_to_tick,
    trading_dates,
)

CFG = SynthConfig(symbols=("SYNA", "SYNB"), n_days=4, buckets_per_day=5, seed=7)


def test_trading_dates_are_consecutive_weekdays():
    dates = trading_dates(SynthConfig(n_days=7, start_date=dt.date(2026, 1, 2)))
    assert dates[0] == dt.date(2026, 1, 2)  # Friday
    assert dates[1] == dt.date(2026, 1, 5)  # skips the weekend
    assert all(d.weekday() < 5 for d in dates)
    assert len(dates) == 7


def test_bucket_ts_ns_spans_the_session():
    start, end = bucket_ts_ns(dt.date(2026, 1, 5), 0, 13)
    assert end > start
    last_start, last_end = bucket_ts_ns(dt.date(2026, 1, 5), 12, 13)
    assert last_end - start == 23_400 * 1_000_000_000  # 6.5 hours


def test_round_to_tick():
    assert round_to_tick(100.004, 0.01) == pytest.approx(100.00)
    assert round_to_tick(100.006, 0.01) == pytest.approx(100.01)


def test_ground_truth_covers_every_symbol():
    gt = ground_truth(CFG)
    assert set(gt["impact_delta"]) == set(CFG.symbols)
    assert set(gt["impact_Y"]) == set(CFG.symbols)
    assert all(0.3 < d < 1.0 for d in gt["impact_delta"].values())


def test_generate_buckets_shape_and_columns():
    b = generate_buckets(CFG)
    assert b.height == len(CFG.symbols) * CFG.n_days * CFG.buckets_per_day
    assert set(b.columns) == {
        "symbol", "day_index", "bucket_index", "date", "ts_start_ns", "ts_end_ns",
        "mid_open", "mid_close", "bucket_volume", "net_flow", "participation",
    }


def test_generation_is_deterministic_under_seed():
    a = generate_buckets(CFG)
    b = generate_buckets(CFG)
    assert a.equals(b)


def test_different_seeds_give_different_paths():
    a = generate_buckets(CFG)
    b = generate_buckets(SynthConfig(symbols=CFG.symbols, n_days=4, buckets_per_day=5, seed=8))
    assert not a.equals(b)


def test_mid_is_continuous_across_buckets():
    b = generate_buckets(CFG).filter(pl.col("symbol") == "SYNA").sort(
        ["day_index", "bucket_index"]
    )
    closes = b["mid_close"].to_list()[:-1]
    opens = b["mid_open"].to_list()[1:]
    assert closes == pytest.approx(opens)


def test_impact_sign_follows_flow_sign_on_average():
    """With low noise, buying buckets push the mid up.

    noise_frac is small but non-zero: the generator now holds total bucket
    variance at sigma_bucket**2 by trading diffusion off against impact, and a
    mid with literally no diffusion leaves nothing to trade off (finding 4.1).
    """
    cfg = SynthConfig(symbols=("SYNA",), n_days=20, buckets_per_day=13, seed=1, noise_frac=0.01)
    b = generate_buckets(cfg)
    ret = (b["mid_close"] / b["mid_open"] - 1.0).to_numpy()
    flow = b["participation"].to_numpy()
    assert np.corrcoef(np.sign(flow), np.sign(ret))[0, 1] > 0.99


def test_daily_bars_conform_and_aggregate_buckets():
    from quantic.data.schemas import validate

    b = generate_buckets(CFG)
    daily = build_daily_bars(b, CFG)
    validate("daily_bars", daily.to_arrow())
    assert daily.height == len(CFG.symbols) * CFG.n_days
    row = daily.filter(pl.col("symbol") == "SYNA").sort("date").row(0, named=True)
    assert row["low"] <= row["open"] <= row["high"]
    assert row["low"] <= row["close"] <= row["high"]
    assert row["volume"] > 0
    assert row["adv"] > 0


def test_l1_conforms_and_is_never_crossed():
    from quantic.data.schemas import validate

    b = generate_buckets(CFG)
    l1 = build_l1(b, CFG)
    validate("l1_taq", l1.to_arrow())
    assert l1.height == b.height
    assert (l1["ask"] > l1["bid"]).all()
    spread = (l1["ask"] - l1["bid"]).to_numpy()
    assert spread == pytest.approx(2 * CFG.tick_size)
