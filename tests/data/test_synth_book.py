import polars as pl

from quantic.data.schemas import L3Action, validate
from quantic.data.synth import SynthConfig, build_l3_and_l2, generate_buckets, generate_bundle

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
