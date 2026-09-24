import os

import polars as pl
import pytest

from quantic.data.synth import SynthConfig, generate_bundle
from quantic.micro.validation import compare_to_l2

CFG = SynthConfig(
    symbols=("SYNA", "SYNB"), n_days=3, buckets_per_day=6, seed=11, depth_levels=6
)

DEFAULT_SCALE = SynthConfig(
    symbols=("SYNA", "SYNB"), n_days=20, buckets_per_day=13, seed=11, depth_levels=6
)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return generate_bundle(tmp_path_factory.mktemp("gate") / "synth", CFG)


@pytest.fixture(scope="module")
def default_scale_bundle(tmp_path_factory):
    return generate_bundle(tmp_path_factory.mktemp("gate_scale") / "synth", DEFAULT_SCALE)


def test_reconstruction_matches_l2_exactly(bundle):
    """THE M1 GATE: an independent replay of L3 reproduces every L2 snapshot."""
    mismatches = compare_to_l2(bundle.l3(), bundle.l2(), levels=CFG.depth_levels)
    assert mismatches == [], mismatches[:10]


def test_gate_covers_every_snapshot(bundle):
    l2 = bundle.l2()
    expected = l2.select("symbol", "ts_ns").unique().height
    assert expected == len(CFG.symbols) * CFG.n_days * CFG.buckets_per_day


def test_comparator_detects_an_injected_size_error(bundle):
    corrupted = bundle.l2().with_columns(
        pl.when((pl.col("level") == 0) & (pl.col("side") == "buy"))
        .then(pl.col("size") + 7)
        .otherwise(pl.col("size"))
        .alias("size")
    )
    mismatches = compare_to_l2(bundle.l3(), corrupted, levels=CFG.depth_levels)
    assert mismatches
    assert all(m.field == "size" for m in mismatches)


def test_comparator_detects_an_injected_price_error(bundle):
    corrupted = bundle.l2().with_columns(
        pl.when((pl.col("level") == 0) & (pl.col("side") == "sell"))
        .then(pl.col("px") + 0.05)
        .otherwise(pl.col("px"))
        .alias("px")
    )
    mismatches = compare_to_l2(bundle.l3(), corrupted, levels=CFG.depth_levels)
    assert mismatches
    assert any(m.field == "px" for m in mismatches)


def test_comparator_can_be_restricted_to_symbols(bundle):
    mismatches = compare_to_l2(
        bundle.l3(), bundle.l2(), levels=CFG.depth_levels, symbols=["SYNA"]
    )
    assert mismatches == []


def test_reconstruction_matches_l2_at_default_scale(default_scale_bundle):
    """The gate must hold at realistic scale, not only on a toy config."""
    mismatches = compare_to_l2(
        default_scale_bundle.l3(), default_scale_bundle.l2(), levels=DEFAULT_SCALE.depth_levels
    )
    assert mismatches == [], mismatches[:10]


def test_unknown_requested_symbol_raises(bundle):
    with pytest.raises(ValueError, match="absent"):
        compare_to_l2(bundle.l3(), bundle.l2(), levels=CFG.depth_levels, symbols=["NONEXISTENT"])


def test_empty_l2_raises_rather_than_reporting_clean(bundle):
    empty_l2 = pl.DataFrame(
        schema={
            "ts_ns": pl.Int64,
            "symbol": pl.String,
            "side": pl.String,
            "level": pl.Int32,
            "px": pl.Float64,
            "size": pl.Int64,
        }
    )
    with pytest.raises(ValueError):
        compare_to_l2(bundle.l3(), empty_l2, levels=CFG.depth_levels)


def test_empty_symbols_list_raises(bundle):
    with pytest.raises(ValueError):
        compare_to_l2(bundle.l3(), bundle.l2(), levels=CFG.depth_levels, symbols=[])


@pytest.mark.skipif(
    not os.environ.get("QUANTIC_REAL_BUNDLE"),
    reason="needs a bundle with real paired L2 and L3; set QUANTIC_REAL_BUNDLE to its path",
)
def test_l3_l2_holdout_against_real_data():
    """The holdout that makes spec 10.3's cross-validation claim true.

    Every other test in this file runs against synthetic data where both the
    L3 stream and the L2 snapshots come from the same `build_l3_and_l2` call.
    Three independently-written book implementations agreeing is strong, but
    it cannot catch a shared misconception about message semantics that
    originates in the generator -- real feeds vary on in-place size modify,
    price-unchanged replace, and whether priority is retained on a size
    decrease.

    This test is the one that reads two genuinely separate feed products. It
    stays skipped, and named, until such a bundle exists.
    """
    from pathlib import Path

    from quantic.data.bundle import DatasetBundle

    bundle = DatasetBundle.load(Path(os.environ["QUANTIC_REAL_BUNDLE"]))
    bundle.validate()
    levels = int(os.environ.get("QUANTIC_REAL_BUNDLE_LEVELS", "10"))

    mismatches = compare_to_l2(bundle.l3(), bundle.l2(), levels=levels)
    assert mismatches == [], (
        f"{len(mismatches)} disagreement(s) between replayed L3 and published L2 "
        f"(first: {mismatches[0]})"
    )
