"""Typer CLI. Excluded from the layer contract: it wires layers together."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from quantic.data.bundle import DatasetBundle
from quantic.data.catalog import DEFAULT_CATALOG_PATH, Catalog
from quantic.data.request import default_request
from quantic.data.synth import SynthConfig, generate_bundle

app = typer.Typer(help="Quantic: quantum vs classical liquidation benchmarking.")
data_app = typer.Typer(help="Dataset bundles.")
app.add_typer(data_app, name="data")
# Rich defaults to width 80 when stdout is not a TTY (e.g. under Typer's
# CliRunner in tests), which truncates the `data list` table. Pin a wide
# enough width so CLI output is deterministic regardless of terminal.
console = Console(width=120)


def _split(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


@data_app.command("synth")
def synth(
    out: Path = typer.Option(..., "--out", help="Directory to write the bundle into."),
    symbols: str = typer.Option("SYNA,SYNB,SYNC", "--symbols"),
    days: int = typer.Option(20, "--days"),
    buckets: int = typer.Option(13, "--buckets"),
    seed: int = typer.Option(0, "--seed"),
    depth_levels: int = typer.Option(10, "--depth-levels"),
    catalog: Path = typer.Option(DEFAULT_CATALOG_PATH, "--catalog"),
) -> None:
    """Generate a synthetic bundle with known ground-truth impact parameters."""
    cfg = SynthConfig(
        symbols=_split(symbols),
        n_days=days,
        buckets_per_day=buckets,
        seed=seed,
        depth_levels=depth_levels,
    )
    bundle = generate_bundle(out, cfg)
    entry = Catalog(catalog).register(bundle)
    console.print(f"[green]wrote[/green] {entry.bundle_id} -> {out}")
    console.print(f"content_hash {entry.content_hash}")


@data_app.command("ingest")
def ingest(
    path: Path = typer.Argument(..., help="Bundle directory to validate and register."),
    catalog: Path = typer.Option(DEFAULT_CATALOG_PATH, "--catalog"),
) -> None:
    """Validate a bundle's integrity and register it in the local catalog."""
    bundle = DatasetBundle.load(path)
    try:
        bundle.validate()
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        console.print(f"[red]integrity check failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    entry = Catalog(catalog).register(bundle)
    console.print(f"[green]registered[/green] {entry.bundle_id}")
    console.print(f"content_hash {entry.content_hash}")


@data_app.command("list")
def list_bundles(catalog: Path = typer.Option(DEFAULT_CATALOG_PATH, "--catalog")) -> None:
    """List registered bundles."""
    table = Table("bundle_id", "provenance", "symbols", "range", "content_hash")
    for entry in Catalog(catalog).entries():
        table.add_row(
            entry.bundle_id,
            entry.provenance,
            str(len(entry.symbols)),
            f"{entry.start_date}..{entry.end_date}",
            entry.content_hash[:12],
        )
    console.print(table)


@data_app.command("request")
def request(
    out: Path = typer.Option(..., "--out"),
    symbols: str = typer.Option(..., "--symbols"),
    l3_symbols: str = typer.Option(..., "--l3-symbols"),
    market: str = typer.Option(..., "--market"),
    l2_days: int = typer.Option(20, "--l2-days"),
    l3_days: int = typer.Option(5, "--l3-days"),
    daily_days: int = typer.Option(365, "--daily-days"),
    end_date: str = typer.Option("", "--end-date", help="ISO date; defaults to today."),
) -> None:
    """Emit a data-request spec to hand to the market data machine."""
    end = dt.date.fromisoformat(end_date) if end_date else dt.date.today()
    req = default_request(
        symbols=_split(symbols),
        l3_symbols=_split(l3_symbols),
        market=market,
        l2_start=(end - dt.timedelta(days=l2_days)).isoformat(),
        l2_end=end.isoformat(),
        l3_start=(end - dt.timedelta(days=l3_days)).isoformat(),
        l3_end=end.isoformat(),
        daily_start=(end - dt.timedelta(days=daily_days)).isoformat(),
        daily_end=end.isoformat(),
    )
    req.write(out)
    console.print(f"[green]wrote request[/green] {req.request_id} -> {out}")
