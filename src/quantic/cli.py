"""Typer CLI. Excluded from the layer contract: it wires layers together."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from quantic.bench import config as bench_config
from quantic.bench import report as bench_report
from quantic.bench import runner as bench_runner
from quantic.bench.pool import WorkerStartError
from quantic.bench.store import SchemaMismatch, StoreBusy
from quantic.core.session import NAMED_SESSIONS, SessionError, TradingSession
from quantic.data.bundle import DatasetBundle
from quantic.data.catalog import DEFAULT_CATALOG_PATH, Catalog, DuplicateBundleError
from quantic.data.ingest import build_bundle_from_export
from quantic.data.manifest import MANIFEST_FILENAME
from quantic.data.request import default_request
from quantic.data.schemas import validate_values
from quantic.data.synth import SynthConfig, generate_bundle
from quantic.solvers.quantum.ibm import NoAccount
from quantic.solvers.quantum.quota import QuotaBusy, QuotaExhausted

app = typer.Typer(help="Quantic: quantum vs classical liquidation benchmarking.")
data_app = typer.Typer(help="Dataset bundles.")
app.add_typer(data_app, name="data")
bench_app = typer.Typer(help="The benchmark: plan, run and report an experiment.")
app.add_typer(bench_app, name="bench")
# Rich defaults to width 80 when stdout is not a TTY (e.g. under Typer's
# CliRunner in tests), which truncates the `data list` table. Pin a wide
# enough width so CLI output is deterministic regardless of terminal.
console = Console(width=120)


def _split(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _parse_overrides(flag: str, value: str, symbols: tuple[str, ...]) -> dict[str, float] | None:
    """Parse ``SYMBOL=VALUE,SYMBOL=VALUE`` into a mapping, or None if empty."""
    if not value.strip():
        return None
    out: dict[str, float] = {}
    for item in _split(value):
        symbol, sep, raw = item.partition("=")
        if not sep:
            raise typer.BadParameter(
                f"{flag} expects SYMBOL=VALUE pairs, got {item!r}", param_hint=flag
            )
        try:
            out[symbol.strip()] = float(raw)
        except ValueError as exc:
            raise typer.BadParameter(
                f"{flag} value for {symbol.strip()!r} is not a number: {raw!r}",
                param_hint=flag,
            ) from exc
    unknown = sorted(set(out) - set(symbols))
    if unknown:
        raise typer.BadParameter(
            f"{flag} names symbol(s) {unknown} that are not in --symbols "
            f"{sorted(symbols)}",
            param_hint=flag,
        )
    missing = sorted(set(symbols) - set(out))
    if missing:
        raise typer.BadParameter(
            f"{flag} must cover every symbol; missing {missing}", param_hint=flag
        )
    return out


@data_app.command("synth")
def synth(
    out: Annotated[Path, typer.Option("--out", help="Directory to write the bundle into.")],
    symbols: Annotated[str, typer.Option("--symbols")] = "SYNA,SYNB,SYNC",
    days: Annotated[int, typer.Option("--days")] = 20,
    buckets: Annotated[
        int,
        typer.Option("--buckets", help="Buckets per session; must divide 23400s exactly."),
    ] = 13,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    start_date: Annotated[
        str, typer.Option("--start-date", help="ISO date of the first session.")
    ] = "2026-01-05",
    base_price: Annotated[float, typer.Option("--base-price")] = 100.0,
    daily_vol: Annotated[
        float, typer.Option("--daily-vol", help="Fractional daily volatility.")
    ] = 0.02,
    adv_shares: Annotated[int, typer.Option("--adv-shares")] = 100_000,
    tick_size: Annotated[float, typer.Option("--tick-size")] = 0.01,
    lot_size: Annotated[int, typer.Option("--lot-size")] = 100,
    depth_levels: Annotated[int, typer.Option("--depth-levels")] = 10,
    level_size: Annotated[int, typer.Option("--level-size")] = 1_000,
    flow_sd: Annotated[
        float,
        typer.Option("--flow-sd", help="Std dev of per-bucket net order-flow imbalance."),
    ] = 0.08,
    noise_frac: Annotated[
        float,
        typer.Option(
            "--noise-frac",
            help="Diffusion weight against impact. Total bucket variance is held at "
            "sigma_bucket^2 either way, so this sets the split. 1.0 is the realistic "
            "regime; lower it to make impact recoverable.",
        ),
    ] = 1.0,
    impact_delta: Annotated[
        str,
        typer.Option(
            "--impact-delta",
            help="Per-symbol true exponent as SYMBOL=VALUE,SYMBOL=VALUE.",
        ),
    ] = "",
    impact_y: Annotated[
        str,
        typer.Option(
            "--impact-y", help="Per-symbol true coefficient as SYMBOL=VALUE,SYMBOL=VALUE."
        ),
    ] = "",
    bundle_id: Annotated[
        str | None,
        typer.Option(
            "--bundle-id",
            help="Override the derived bundle id (default: synth-seed{seed}-{days}d).",
        ),
    ] = None,
    catalog: Annotated[Path, typer.Option("--catalog")] = DEFAULT_CATALOG_PATH,
) -> None:
    """Generate a synthetic bundle with known ground-truth impact parameters.

    Every ``SynthConfig`` field is reachable here. It used to expose 5 of 16,
    so a calibration-grade fixture had to be built with a Python script --
    and ``--noise-frac``, the one knob that decides whether impact
    calibration works at all, was among the unreachable ones.
    """
    parsed_symbols = _split(symbols)
    try:
        cfg = SynthConfig(
            symbols=parsed_symbols,
            n_days=days,
            buckets_per_day=buckets,
            seed=seed,
            start_date=dt.date.fromisoformat(start_date),
            base_price=base_price,
            daily_vol=daily_vol,
            adv_shares=adv_shares,
            tick_size=tick_size,
            lot_size=lot_size,
            depth_levels=depth_levels,
            level_size=level_size,
            flow_sd=flow_sd,
            noise_frac=noise_frac,
            impact_delta=_parse_overrides("--impact-delta", impact_delta, parsed_symbols),
            impact_Y=_parse_overrides("--impact-y", impact_y, parsed_symbols),
        )
    except typer.BadParameter as exc:
        console.print(f"[red]{exc.message}[/red]")
        raise typer.Exit(code=1) from exc
    except (ValueError, SessionError) as exc:
        console.print(f"[red]invalid configuration:[/red] {exc}")
        raise typer.Exit(code=1) from exc

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


# The repository whose commit every bench row names: the one this package runs from.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _say_why(exc: Exception) -> None:
    # Unwrapped, and escaped: a path or a list in the message is not markup.
    console.print(f"[red]{escape(str(exc))}[/red]", soft_wrap=True)


def _experiment(path: Path) -> bench_config.Experiment:
    try:
        return bench_config.load(path)
    except bench_config.ConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="CONFIG") from exc


@bench_app.command("plan")
def bench_plan(
    config: Annotated[Path, typer.Argument(help="Experiment TOML.")],
    store: Annotated[Path | None, typer.Option("--store")] = None,
    workers: Annotated[int | None, typer.Option("--workers")] = None,
) -> None:
    """Count an experiment's tasks and estimate the time left. Runs nothing."""
    p = bench_runner.plan(
        _experiment(config), root=_REPO_ROOT, store=store, workers=workers
    )
    console.print(
        f"{p.n_tasks} tasks, {p.n_done} already in the store, {p.n_todo} to run; "
        f"about {p.estimate_s / 3600:.1f} h on {p.n_workers} workers "
        "(every budget spent in full: an upper estimate)"
    )
    if p.n_over_max or p.n_over_noisy:
        console.print(
            f"left out of the plan: {p.n_over_max} quantum tasks above max_qubits, "
            f"{p.n_over_noisy} full noisy runs above noisy_max_qubits"
        )
    if p.n_hardware:
        console.print(f"{p.n_hardware} hardware tasks wait for the IBM lane")


@bench_app.command("run")
def bench_run(
    config: Annotated[Path, typer.Argument(help="Experiment TOML.")],
    store: Annotated[Path, typer.Option("--store")],
    workers: Annotated[int | None, typer.Option("--workers")] = None,
    allow_dirty: Annotated[
        bool, typer.Option("--allow-dirty", help="Run on uncommitted changes; recorded.")
    ] = False,
    no_hardware: Annotated[
        bool, typer.Option("--no-hardware", help="Run everything except IBM hardware jobs.")
    ] = False,
    repeat_hardware: Annotated[
        bool, typer.Option("--repeat-hardware",
                           help="Submit hardware tasks the store already holds, again.")
    ] = False,
) -> None:
    """Run every task the store does not hold yet. Safe to interrupt and rerun."""
    try:
        s = bench_runner.run(
            _experiment(config), root=_REPO_ROOT, store=store, workers=workers,
            allow_dirty=allow_dirty, no_hardware=no_hardware,
            repeat_hardware=repeat_hardware,
        )
    except (bench_runner.DirtyTree, StoreBusy, SchemaMismatch, WorkerStartError,
            QuotaExhausted, QuotaBusy, NoAccount) as exc:
        _say_why(exc)
        raise typer.Exit(code=1) from exc
    console.print(
        f"wrote {s.n_written} rows: {s.n_killed} killed, {s.n_errors} errors"
    )
    if s.hardware_stop:
        console.print(f"hardware lane stopped: {s.hardware_stop}")


@bench_app.command("report")
def bench_report_command(
    store: Annotated[Path, typer.Argument(help="The DuckDB store.")],
    out: Annotated[Path, typer.Option("--out")],
    commit: Annotated[str | None, typer.Option("--commit")] = None,
) -> None:
    """Write the crossover tables, dial effects, bracket check and failed references."""
    try:
        written = bench_report.report(store, out, commit=commit)
    except (FileNotFoundError, StoreBusy, SchemaMismatch, bench_report.CommitChoice) as exc:
        _say_why(exc)
        raise typer.Exit(code=1) from exc
    for path in written:
        console.print(str(path))
