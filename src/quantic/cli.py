"""Typer CLI. Excluded from the layer contract: it wires layers together."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from quantic.core.session import NAMED_SESSIONS, SessionError, TradingSession
from quantic.data.bundle import DatasetBundle
from quantic.data.catalog import DEFAULT_CATALOG_PATH, Catalog, DuplicateBundleError
from quantic.data.ingest import build_bundle_from_export
from quantic.data.manifest import MANIFEST_FILENAME
from quantic.data.request import default_request
from quantic.data.schemas import validate_values
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
    out: Annotated[Path, typer.Option("--out", help="Directory to write the bundle into.")],
    symbols: Annotated[str, typer.Option("--symbols")] = "SYNA,SYNB,SYNC",
    days: Annotated[int, typer.Option("--days")] = 20,
    buckets: Annotated[int, typer.Option("--buckets")] = 13,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    depth_levels: Annotated[int, typer.Option("--depth-levels")] = 10,
    bundle_id: Annotated[
        str | None,
        typer.Option(
            "--bundle-id",
            help="Override the derived bundle id (default: synth-seed{seed}-{days}d).",
        ),
    ] = None,
    catalog: Annotated[Path, typer.Option("--catalog")] = DEFAULT_CATALOG_PATH,
) -> None:
    """Generate a synthetic bundle with known ground-truth impact parameters."""
    cfg = SynthConfig(
        symbols=_split(symbols),
        n_days=days,
        buckets_per_day=buckets,
        seed=seed,
        depth_levels=depth_levels,
    )
    bundle = generate_bundle(out, cfg, bundle_id=bundle_id)
    try:
        entry = Catalog(catalog).register(bundle)
    except DuplicateBundleError as exc:
        console.print(f"[red]registration failed:[/red] {exc}")
        console.print(
            "[red]hint:[/red] pass a different --bundle-id, or write to a different --out"
        )
        raise typer.Exit(code=1) from exc
    console.print(f"[green]wrote[/green] {entry.bundle_id} -> {out}")
    console.print(f"content_hash {entry.content_hash}")


@data_app.command("ingest")
def ingest(
    path: Annotated[
        Path,
        typer.Argument(
            help="A bundle directory to validate, or a raw export directory to build "
            "into one (pass --out and --session for that)."
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Build a bundle here from a raw export. Omit to validate an "
            "existing bundle in place.",
        ),
    ] = None,
    session: Annotated[
        str,
        typer.Option(
            "--session",
            help="Trading calendar the export's timestamps belong to: a name "
            f"({', '.join(sorted(NAMED_SESSIONS))}) or 'HH:MM-HH:MM@Area/City'.",
        ),
    ] = "",
    bundle_id: Annotated[str | None, typer.Option("--bundle-id")] = None,
    provenance: Annotated[
        str, typer.Option("--provenance", help="Where the export came from.")
    ] = "export",
    catalog: Annotated[Path, typer.Option("--catalog")] = DEFAULT_CATALOG_PATH,
) -> None:
    """Register a bundle, building it from a raw export first if needed.

    Two modes, selected by ``--out``. Without it, ``path`` must already be a
    bundle: its hashes are checked and it is registered. With it, ``path`` is
    a raw export from the market-data machine and is normalised, validated and
    written to ``--out`` as a new bundle (spec section 8.3).
    """
    if out is not None:
        bundle = _build_from_export(
            path, out, session=session, bundle_id=bundle_id, provenance=provenance
        )
    else:
        if not (path / MANIFEST_FILENAME).exists():
            console.print(
                f"[red]no {MANIFEST_FILENAME} in[/red] {path}"
            )
            console.print(
                "[red]hint:[/red] this looks like a raw export rather than a bundle. "
                "Pass --out <dir> and --session <calendar> to build a bundle from it"
            )
            raise typer.Exit(code=1)
        bundle = DatasetBundle.load(path)
        try:
            bundle.validate()
            for name in bundle.manifest.granularities:
                validate_values(name, bundle.table(name).to_arrow())
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
            console.print(f"[red]integrity check failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    try:
        entry = Catalog(catalog).register(bundle)
    except DuplicateBundleError as exc:
        console.print(f"[red]registration failed:[/red] {exc}")
        console.print(
            "[red]hint:[/red] this bundle's manifest.json bundle_id collides with a "
            "differently-content bundle already in the catalog; edit bundle_id in "
            "manifest.json, or register into a different --catalog"
        )
        raise typer.Exit(code=1) from exc
    console.print(f"[green]registered[/green] {entry.bundle_id}")
    console.print(f"content_hash {entry.content_hash}")


def _build_from_export(
    path: Path,
    out: Path,
    *,
    session: str,
    bundle_id: str | None,
    provenance: str,
) -> DatasetBundle:
    if not session:
        console.print("[red]--session is required when building from a raw export[/red]")
        console.print(
            "[red]hint:[/red] every intraday metric is bucketed session-relative, so "
            "the calendar cannot be guessed. Try --session nyse, or "
            "--session '09:30-16:00@America/New_York'"
        )
        raise typer.Exit(code=1)
    try:
        parsed = TradingSession.parse(session)
    except SessionError as exc:
        console.print(f"[red]bad --session:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        report = build_bundle_from_export(
            path,
            out,
            bundle_id=bundle_id or path.name,
            session=parsed,
            provenance=provenance,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        console.print(f"[red]ingest failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]built[/green] {report.bundle_id} -> {out}")
    for name, rows in sorted(report.rows.items()):
        console.print(f"  {name}: {rows} rows from {len(report.sources[name])} file(s)")
    for name, columns in sorted(report.dropped_columns.items()):
        if columns:
            console.print(
                f"  [yellow]dropped from {name}:[/yellow] {', '.join(columns)} "
                "(not in the schema)"
            )
    return DatasetBundle.load(out)


@data_app.command("list")
def list_bundles(
    catalog: Annotated[Path, typer.Option("--catalog")] = DEFAULT_CATALOG_PATH,
) -> None:
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
    out: Annotated[Path, typer.Option("--out")],
    symbols: Annotated[str, typer.Option("--symbols")],
    l3_symbols: Annotated[str, typer.Option("--l3-symbols")],
    market: Annotated[str, typer.Option("--market")],
    l2_days: Annotated[int, typer.Option("--l2-days")] = 20,
    l3_days: Annotated[int, typer.Option("--l3-days")] = 5,
    daily_days: Annotated[int, typer.Option("--daily-days")] = 365,
    end_date: Annotated[
        str, typer.Option("--end-date", help="ISO date; defaults to today.")
    ] = "",
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
