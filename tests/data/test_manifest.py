import json

import pytest

from quantic.data.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    compute_content_hash,
    sha256_file,
)


def _manifest(**overrides) -> Manifest:
    base = dict(
        schema_version="1",
        bundle_id="synth-test",
        created_utc="2026-09-22T00:00:00Z",
        provenance="synthetic",
        symbols=("SYNA", "SYNB"),
        start_date="2026-01-02",
        end_date="2026-01-09",
        granularities=("l1_taq", "daily_bars"),
        files={"daily_bars/part.parquet": "aa" * 32},
        content_hash="",
        extra={},
    )
    base.update(overrides)
    base["content_hash"] = compute_content_hash(base["files"])
    return Manifest(**base)


def test_sha256_file_matches_known_value(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    # sha256("hello")
    assert sha256_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_content_hash_is_independent_of_insertion_order():
    a = compute_content_hash({"b.parquet": "11" * 32, "a.parquet": "22" * 32})
    b = compute_content_hash({"a.parquet": "22" * 32, "b.parquet": "11" * 32})
    assert a == b


def test_content_hash_changes_when_any_file_hash_changes():
    base = compute_content_hash({"a.parquet": "22" * 32})
    changed = compute_content_hash({"a.parquet": "23" * 32})
    assert base != changed


def test_content_hash_is_not_confusable_across_path_boundaries():
    """Concatenating path and hash without a separator would collide these."""
    a = compute_content_hash({"ab": "cd"})
    b = compute_content_hash({"a": "bcd"})
    assert a != b


def test_manifest_roundtrips_through_disk(tmp_path):
    m = _manifest()
    m.write(tmp_path / MANIFEST_FILENAME)
    loaded = Manifest.read(tmp_path / MANIFEST_FILENAME)
    assert loaded == m


def test_manifest_json_is_stable_and_sorted():
    m = _manifest()
    payload = json.loads(m.to_json())
    assert list(payload) == sorted(payload)
    assert payload["symbols"] == ["SYNA", "SYNB"]


def test_manifest_rejects_inconsistent_content_hash():
    with pytest.raises(ValueError, match="content_hash"):
        Manifest(
            schema_version="1",
            bundle_id="bad",
            created_utc="2026-09-22T00:00:00Z",
            provenance="synthetic",
            symbols=("SYNA",),
            start_date="2026-01-02",
            end_date="2026-01-02",
            granularities=("daily_bars",),
            files={"daily_bars/part.parquet": "aa" * 32},
            content_hash="deadbeef",
            extra={},
        )
