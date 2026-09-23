import json

from typer.testing import CliRunner

from quantic.cli import app

runner = CliRunner()


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
