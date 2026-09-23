import dataclasses
import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from quantic.cli import app
from quantic.data.manifest import MANIFEST_FILENAME, Manifest, compute_content_hash, sha256_file

runner = CliRunner()


def _corrupt_l1_bid_size_in_place(out: Path) -> None:
    """Overwrite one l1_taq partition with a schema-conforming but
    value-invalid table (negative bid_size), then patch the manifest's file
    hash and content_hash so ``bundle.validate()``'s hash check still passes
    — it is ``validate_values`` (C4) that must be what catches this."""
    victim = next((out / "l1_taq").rglob("*.parquet"))
    df = pl.read_parquet(victim)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(-5)
        .otherwise(pl.col("bid_size"))
        .alias("bid_size")
    )
    df.write_parquet(victim, compression="zstd")

    manifest_path = out / MANIFEST_FILENAME
    manifest = Manifest.read(manifest_path)
    rel = victim.relative_to(out).as_posix()
    files = dict(manifest.files)
    files[rel] = sha256_file(victim)
    manifest = dataclasses.replace(manifest, files=files, content_hash=compute_content_hash(files))
    manifest.write(manifest_path)


def test_synth_creates_a_valid_bundle_and_registers_it(tmp_path):
    out = tmp_path / "synth"
    catalog = tmp_path / "catalog.json"
    result = runner.invoke(
        app,
        [
            "data", "synth",
            "--out", str(out),
            "--symbols", "SYNA,SYNB",
            "--days", "2",
            "--buckets", "3",
            "--seed", "5",
            "--catalog", str(catalog),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "manifest.json").exists()
    assert json.loads(catalog.read_text())[0]["bundle_id"]


def test_ingest_validates_and_registers(tmp_path):
    out = tmp_path / "synth"
    runner.invoke(
        app,
        ["data", "synth", "--out", str(out), "--days", "2", "--buckets", "3",
         "--catalog", str(tmp_path / "c.json")],
    )
    result = runner.invoke(
        app, ["data", "ingest", str(out), "--catalog", str(tmp_path / "c.json")]
    )
    assert result.exit_code == 0, result.output
    assert "content_hash" in result.output


def test_ingest_fails_loudly_on_tampered_bundle(tmp_path):
    out = tmp_path / "synth"
    runner.invoke(
        app,
        ["data", "synth", "--out", str(out), "--days", "2", "--buckets", "3",
         "--catalog", str(tmp_path / "c.json")],
    )
    victim = next((out / "l1_taq").rglob("*.parquet"))
    victim.write_bytes(victim.read_bytes() + b"x")
    result = runner.invoke(
        app, ["data", "ingest", str(out), "--catalog", str(tmp_path / "c.json")]
    )
    assert result.exit_code != 0


def test_ingest_fails_loudly_on_value_invalid_bundle(tmp_path):
    out = tmp_path / "synth"
    runner.invoke(
        app,
        ["data", "synth", "--out", str(out), "--days", "2", "--buckets", "3",
         "--catalog", str(tmp_path / "c.json")],
    )
    _corrupt_l1_bid_size_in_place(out)
    result = runner.invoke(
        app, ["data", "ingest", str(out), "--catalog", str(tmp_path / "c2.json")]
    )
    assert result.exit_code != 0
    assert "bid_size" in result.output


def test_list_shows_registered_bundles(tmp_path):
    out = tmp_path / "synth"
    catalog = tmp_path / "catalog.json"
    runner.invoke(
        app,
        ["data", "synth", "--out", str(out), "--days", "2", "--buckets", "3",
         "--catalog", str(catalog)],
    )
    result = runner.invoke(app, ["data", "list", "--catalog", str(catalog)])
    assert result.exit_code == 0, result.output
    assert "synth-seed" in result.output


def test_request_writes_a_spec_file(tmp_path):
    out = tmp_path / "request.json"
    result = runner.invoke(
        app,
        ["data", "request", "--out", str(out), "--symbols", "AAA,BBB",
         "--l3-symbols", "AAA", "--market", "XNAS"],
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(out.read_text())["items"]) == 4
