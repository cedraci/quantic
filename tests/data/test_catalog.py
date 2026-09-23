import pytest

from quantic.data.catalog import Catalog, DuplicateBundleError
from quantic.data.synth import SynthConfig, generate_bundle

CFG = SynthConfig(symbols=("SYNA",), n_days=2, buckets_per_day=3, seed=1, depth_levels=3)


def test_register_then_get(tmp_path):
    bundle = generate_bundle(tmp_path / "b1", CFG, bundle_id="b1")
    cat = Catalog(tmp_path / "catalog.json")
    entry = cat.register(bundle)
    assert entry.bundle_id == "b1"
    assert cat.get("b1").content_hash == bundle.content_hash
    assert cat.get("b1").symbols == ("SYNA",)


def test_catalog_persists_across_instances(tmp_path):
    bundle = generate_bundle(tmp_path / "b1", CFG, bundle_id="b1")
    Catalog(tmp_path / "catalog.json").register(bundle)
    assert [e.bundle_id for e in Catalog(tmp_path / "catalog.json").entries()] == ["b1"]


def test_reregistering_identical_bundle_is_idempotent(tmp_path):
    bundle = generate_bundle(tmp_path / "b1", CFG, bundle_id="b1")
    cat = Catalog(tmp_path / "catalog.json")
    cat.register(bundle)
    cat.register(bundle)
    assert len(cat.entries()) == 1


def test_conflicting_content_hash_is_rejected(tmp_path):
    a = generate_bundle(tmp_path / "a", CFG, bundle_id="same-id")
    other = SynthConfig(symbols=("SYNA",), n_days=2, buckets_per_day=3, seed=2, depth_levels=3)
    b = generate_bundle(tmp_path / "b", other, bundle_id="same-id")
    cat = Catalog(tmp_path / "catalog.json")
    cat.register(a)
    with pytest.raises(DuplicateBundleError, match="same-id"):
        cat.register(b)


def test_get_unknown_bundle_raises(tmp_path):
    with pytest.raises(KeyError, match="nope"):
        Catalog(tmp_path / "catalog.json").get("nope")
