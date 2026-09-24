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


def test_synth_rerun_with_different_symbols_fails_loudly_not_with_a_traceback(tmp_path):
    catalog = tmp_path / "catalog.json"
    first = runner.invoke(
        app,
        ["data", "synth", "--out", str(tmp_path / "a"), "--symbols", "SYNA,SYNB",
         "--days", "2", "--buckets", "3", "--seed", "5", "--catalog", str(catalog)],
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        app,
        ["data", "synth", "--out", str(tmp_path / "b"), "--symbols", "SYNA,SYNB,SYNC",
         "--days", "2", "--buckets", "3", "--seed", "5", "--catalog", str(catalog)],
    )
    assert second.exit_code != 0
    assert second.exception is None or isinstance(second.exception, SystemExit)
    assert "Traceback" not in second.output
    assert "registration failed" in second.output


def test_synth_rerun_with_explicit_distinct_bundle_id_succeeds(tmp_path):
    catalog = tmp_path / "catalog.json"
    first = runner.invoke(
        app,
        ["data", "synth", "--out", str(tmp_path / "a"), "--symbols", "SYNA,SYNB",
         "--days", "2", "--buckets", "3", "--seed", "5", "--catalog", str(catalog)],
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        app,
        ["data", "synth", "--out", str(tmp_path / "b"), "--symbols", "SYNA,SYNB,SYNC",
         "--days", "2", "--buckets", "3", "--seed", "5",
         "--bundle-id", "synth-seed5-2d-three-symbols",
         "--catalog", str(catalog)],
    )
    assert second.exit_code == 0, second.output
    entries = json.loads(catalog.read_text())
    assert {e["bundle_id"] for e in entries} == {
        "synth-seed5-2d",
        "synth-seed5-2d-three-symbols",
    }


def test_synth_identical_rerun_is_idempotent(tmp_path):
    catalog = tmp_path / "catalog.json"
    args = [
        "data", "synth", "--out", str(tmp_path / "a"), "--symbols", "SYNA,SYNB",
        "--days", "2", "--buckets", "3", "--seed", "5", "--catalog", str(catalog),
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    entries = json.loads(catalog.read_text())
    assert len(entries) == 1


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


# --- P1: ingest from a raw export ------------------------------------------


def _write_export(root: Path) -> Path:
    """A minimal export as a market-data machine would deliver it."""
    import datetime as dt

    root.mkdir(parents=True, exist_ok=True)
    rows = []
    px = 100.0
    for i in range(5):
        rows.append(
            {
                "date": dt.date(2026, 1, 5) + dt.timedelta(days=i),
                "symbol": "AAA",
                "open": px, "high": px + 1, "low": px - 1, "close": px + 0.5,
                "volume": 1_000 + i, "adv": 1_000.0,
                "venue": "XNYS",  # a vendor column the schema does not define
            }
        )
        px += 0.5
    pl.DataFrame(rows).write_parquet(root / "daily_bars.parquet")
    return root


def test_ingest_builds_a_bundle_from_a_raw_export(tmp_path):
    """Finding 2.1: this used to raise FileNotFoundError."""
    export = _write_export(tmp_path / "export")
    out = tmp_path / "bundle"
    catalog = tmp_path / "catalog.json"

    result = runner.invoke(
        app,
        ["data", "ingest", str(export), "--out", str(out),
         "--session", "nyse", "--bundle-id", "real-1", "--catalog", str(catalog)],
    )

    assert result.exit_code == 0, result.output
    assert (out / MANIFEST_FILENAME).exists()
    assert "venue" in result.output, "dropped vendor columns must be reported"
    assert json.loads(catalog.read_text())


def test_ingest_of_a_raw_export_requires_a_session(tmp_path):
    export = _write_export(tmp_path / "export")
    result = runner.invoke(
        app, ["data", "ingest", str(export), "--out", str(tmp_path / "b")]
    )
    assert result.exit_code != 0
    assert "session" in result.output.lower()


def test_ingest_of_a_raw_export_requires_an_out_directory(tmp_path):
    export = _write_export(tmp_path / "export")
    result = runner.invoke(app, ["data", "ingest", str(export), "--session", "nyse"])
    assert result.exit_code != 0
    assert "--out" in result.output


def test_ingest_still_validates_an_existing_bundle_without_out(tmp_path):
    """The existing verb keeps working; --out is what selects export mode."""
    out = tmp_path / "synth"
    catalog = tmp_path / "catalog.json"
    runner.invoke(
        app, ["data", "synth", "--out", str(out), "--days", "2", "--buckets", "3",
              "--symbols", "SYNA", "--depth-levels", "3", "--catalog", str(catalog)]
    )
    catalog.unlink()

    result = runner.invoke(app, ["data", "ingest", str(out), "--catalog", str(catalog)])
    assert result.exit_code == 0, result.output
    assert "registered" in result.output


def test_ingest_reports_a_bad_export_without_a_traceback(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    result = runner.invoke(
        app,
        ["data", "ingest", str(export), "--out", str(tmp_path / "b"), "--session", "nyse"],
    )
    assert result.exit_code == 1
    assert "daily_bars" in result.output
    assert "Traceback" not in result.output


# --- 4.5: the synth CLI must reach every knob that matters -----------------


def _synth_config_from(out: Path) -> dict:
    return json.loads((out / MANIFEST_FILENAME).read_text())["extra"]["synth_config"]


def test_synth_exposes_every_synthconfig_field():
    """Finding 4.5: 5 of 16 fields were reachable, so the calibration-grade
    fixture had to be built with a Python script. `noise_frac` in particular
    is the single knob that decides whether impact calibration works at all."""
    import dataclasses

    from quantic.data.synth import SynthConfig

    result = runner.invoke(app, ["data", "synth", "--help"])
    help_text = result.output
    expected = {
        "symbols": "--symbols", "n_days": "--days", "buckets_per_day": "--buckets",
        "seed": "--seed", "start_date": "--start-date", "base_price": "--base-price",
        "daily_vol": "--daily-vol", "adv_shares": "--adv-shares",
        "tick_size": "--tick-size", "lot_size": "--lot-size",
        "depth_levels": "--depth-levels", "level_size": "--level-size",
        "flow_sd": "--flow-sd", "noise_frac": "--noise-frac",
        "impact_delta": "--impact-delta", "impact_Y": "--impact-y",
    }
    assert set(expected) == {f.name for f in dataclasses.fields(SynthConfig)}

    # Typer wraps long help; strip newlines so a wrapped flag still matches.
    flat = help_text.replace("\n", " ")
    missing = [flag for flag in expected.values() if flag not in flat]
    assert not missing, f"unreachable from the CLI: {missing}"


def test_synth_knobs_reach_the_generator(tmp_path):
    out = tmp_path / "synth"
    result = runner.invoke(
        app,
        ["data", "synth", "--out", str(out), "--symbols", "AAA",
         "--days", "2", "--buckets", "3", "--depth-levels", "3",
         "--noise-frac", "0.05", "--base-price", "1000", "--daily-vol", "0.03",
         "--adv-shares", "250000", "--flow-sd", "0.12", "--level-size", "500",
         "--tick-size", "0.05", "--lot-size", "50", "--start-date", "2026-02-02",
         "--catalog", str(tmp_path / "c.json")],
    )
    assert result.exit_code == 0, result.output

    cfg = _synth_config_from(out)
    assert cfg["noise_frac"] == 0.05
    assert cfg["base_price"] == 1000.0
    assert cfg["daily_vol"] == 0.03
    assert cfg["adv_shares"] == 250_000
    assert cfg["flow_sd"] == 0.12
    assert cfg["level_size"] == 500
    assert cfg["tick_size"] == 0.05
    assert cfg["lot_size"] == 50
    assert cfg["start_date"] == "2026-02-02"


def test_synth_accepts_per_symbol_impact_parameters(tmp_path):
    out = tmp_path / "synth"
    result = runner.invoke(
        app,
        ["data", "synth", "--out", str(out), "--symbols", "AAA,BBB",
         "--days", "2", "--buckets", "3", "--depth-levels", "3",
         "--impact-delta", "AAA=0.4,BBB=0.7", "--impact-y", "AAA=0.5,BBB=1.2",
         "--catalog", str(tmp_path / "c.json")],
    )
    assert result.exit_code == 0, result.output

    gt = json.loads((out / MANIFEST_FILENAME).read_text())["extra"]["ground_truth"]
    assert gt["impact_delta"] == {"AAA": 0.4, "BBB": 0.7}
    assert gt["impact_Y_configured"] == {"AAA": 0.5, "BBB": 1.2}


def test_a_malformed_impact_override_is_reported_clearly(tmp_path):
    result = runner.invoke(
        app,
        ["data", "synth", "--out", str(tmp_path / "s"), "--symbols", "AAA",
         "--days", "2", "--buckets", "3", "--impact-delta", "AAA:0.4",
         "--catalog", str(tmp_path / "c.json")],
    )
    assert result.exit_code == 1
    assert "impact-delta" in result.output or "SYMBOL=VALUE" in result.output
    assert "Traceback" not in result.output


def test_an_override_for_an_unknown_symbol_is_reported(tmp_path):
    result = runner.invoke(
        app,
        ["data", "synth", "--out", str(tmp_path / "s"), "--symbols", "AAA",
         "--days", "2", "--buckets", "3", "--impact-delta", "ZZZ=0.4",
         "--catalog", str(tmp_path / "c.json")],
    )
    assert result.exit_code == 1
    assert "ZZZ" in result.output
    assert "Traceback" not in result.output
