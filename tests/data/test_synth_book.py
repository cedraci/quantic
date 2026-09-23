import numpy as np
import polars as pl

from quantic.data.schemas import L3Action, validate
from quantic.data.synth import SynthConfig, build_l3_and_l2, generate_buckets, generate_bundle
from quantic.micro.liquidity import signed_order_flow

CFG = SynthConfig(symbols=("SYNA", "SYNB"), n_days=3, buckets_per_day=4, seed=3, depth_levels=5)


def test_l3_and_l2_conform_to_schema():
    l3, l2 = build_l3_and_l2(generate_buckets(CFG), CFG)
    validate("l3_messages", l3.to_arrow())
    validate("l2_depth", l2.to_arrow())


def test_all_four_actions_are_exercised():
    l3, _ = build_l3_and_l2(generate_buckets(CFG), CFG)
    assert set(l3["action"].unique().to_list()) == {a.value for a in L3Action}


def test_sequence_numbers_are_unique_and_monotone_per_symbol():
    l3, _ = build_l3_and_l2(generate_buckets(CFG), CFG)
    for symbol in CFG.symbols:
        seqs = l3.filter(pl.col("symbol") == symbol).sort("seq")["seq"].to_list()
        assert seqs == sorted(seqs)
        assert len(seqs) == len(set(seqs))


def test_message_timestamps_never_reach_the_snapshot_boundary():
    buckets = generate_buckets(CFG)
    l3, l2 = build_l3_and_l2(buckets, CFG)
    snapshot_ts = set(l2["ts_ns"].to_list())
    assert not snapshot_ts & set(l3["ts_ns"].to_list())


def test_l2_has_both_sides_at_full_depth():
    _, l2 = build_l3_and_l2(generate_buckets(CFG), CFG)
    per_snapshot = l2.group_by(["symbol", "ts_ns", "side"]).len()
    assert per_snapshot["len"].unique().to_list() == [CFG.depth_levels]
    assert set(l2["side"].unique().to_list()) == {"buy", "sell"}


def test_l2_levels_are_zero_indexed_and_ordered():
    _, l2 = build_l3_and_l2(generate_buckets(CFG), CFG)
    one = l2.filter(
        (pl.col("symbol") == "SYNA") & (pl.col("ts_ns") == l2["ts_ns"].min())
    )
    bids = one.filter(pl.col("side") == "buy").sort("level")
    asks = one.filter(pl.col("side") == "sell").sort("level")
    assert bids["level"].to_list() == list(range(CFG.depth_levels))
    assert bids["px"].to_list() == sorted(bids["px"].to_list(), reverse=True)
    assert asks["px"].to_list() == sorted(asks["px"].to_list())
    assert asks["px"][0] > bids["px"][0]


def test_generate_bundle_is_valid_and_carries_ground_truth(tmp_path):
    bundle = generate_bundle(tmp_path / "synth", CFG)
    bundle.validate()
    assert set(bundle.manifest.granularities) == {
        "l1_taq", "l2_depth", "l3_messages", "daily_bars"
    }
    gt = bundle.manifest.extra["ground_truth"]
    assert set(gt["impact_delta"]) == set(CFG.symbols)
    assert bundle.l3().height > 0
    assert bundle.l2().height > 0


def test_generate_bundle_is_reproducible(tmp_path):
    a = generate_bundle(tmp_path / "a", CFG, bundle_id="fixed")
    b = generate_bundle(tmp_path / "b", CFG, bundle_id="fixed")
    assert a.content_hash == b.content_hash


def _replay_l3_and_count_l2_mismatches(l3: pl.DataFrame, l2: pl.DataFrame, cfg: SynthConfig) -> int:
    """Independently replay ``l3`` messages and count l2 snapshot-sides that disagree.

    Deliberately does not import ``_GeneratorBook`` or anything from
    ``quantic.micro`` — the point is to verify that ``l2_depth`` is fully
    reproducible from ``l3_messages`` alone, using only the message
    semantics (add / cancel / execute / replace).
    """
    orders: dict[int, tuple[str, float, int]] = {}
    messages = l3.sort(["ts_ns", "seq"]).to_dicts()
    idx = 0
    mismatches = 0

    for snap_ts in sorted(l2["ts_ns"].unique().to_list()):
        while idx < len(messages) and messages[idx]["ts_ns"] <= snap_ts:
            m = messages[idx]
            oid = m["order_id"]
            if m["action"] == L3Action.ADD.value:
                orders[oid] = (m["side"], m["px"], m["size"])
            elif m["action"] in (L3Action.CANCEL.value, L3Action.EXECUTE.value):
                if oid in orders:
                    side, px, size = orders[oid]
                    size -= m["size"]
                    if size <= 0:
                        del orders[oid]
                    else:
                        orders[oid] = (side, px, size)
            elif m["action"] == L3Action.REPLACE.value:
                orders[oid] = (m["side"], m["px"], m["size"])
            idx += 1

        for side in ("buy", "sell"):
            book: dict[float, int] = {}
            for s, px, size in orders.values():
                if s == side:
                    book[px] = book.get(px, 0) + size
            prices = sorted(book, reverse=(side == "buy"))[: cfg.depth_levels]
            replayed = [(px, book[px]) for px in prices]

            snap_rows = l2.filter((pl.col("ts_ns") == snap_ts) & (pl.col("side") == side)).sort(
                "level"
            )
            expected = list(
                zip(snap_rows["px"].to_list(), snap_rows["size"].to_list(), strict=True)
            )
            if replayed != expected:
                mismatches += 1

    return mismatches


def test_l3_replay_reproduces_l2_at_default_scale():
    cfg = SynthConfig(symbols=("SYNA",), n_days=20, buckets_per_day=13, seed=0)
    l3, l2 = build_l3_and_l2(generate_buckets(cfg), cfg)
    assert _replay_l3_and_count_l2_mismatches(l3, l2, cfg) == 0


def test_signed_flow_recovers_true_participation(tmp_path):
    """R19 guard: two-sided execution must recover the true participation `f`.

    Single-sided execution (pre-R19) pins observed participation at +-1 every
    bucket, which is what made Task 16's calibration on log|participation|
    unachievable (zero-variance regressor, SVD did not converge).
    """
    cfg = SynthConfig(symbols=("SYNA",), n_days=20, buckets_per_day=13, seed=0)
    bucket_ns = (23_400 // cfg.buckets_per_day) * 1_000_000_000

    bundle = generate_bundle(tmp_path / "synth", cfg)
    flow = signed_order_flow(bundle.l3(), bucket_ns=bucket_ns).select(
        "symbol", "bucket_id", pl.col("participation").alias("observed_participation")
    )

    buckets = generate_buckets(cfg).with_columns(
        (pl.col("ts_start_ns") // bucket_ns).alias("bucket_id")
    ).select("symbol", "bucket_id", pl.col("participation").alias("true_participation"))

    joined = flow.join(buckets, on=["symbol", "bucket_id"], how="inner")
    assert joined.height == cfg.n_days * cfg.buckets_per_day

    obs = np.asarray(joined["observed_participation"].to_list())
    tru = np.asarray(joined["true_participation"].to_list())
    corr = float(np.corrcoef(obs, tru)[0, 1])
    max_abs_diff = float(np.max(np.abs(obs - tru)))

    assert corr > 0.99
    assert max_abs_diff < 0.02
