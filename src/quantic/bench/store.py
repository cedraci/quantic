"""The results store -- parent spec section 9.3.

One DuckDB file. Only the bench's main process opens it for writing, one
transaction per finished task, so a crash loses at most the tasks in flight.
A file written under another schema version is refused rather than read.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl

from quantic.bench.keys import CodeVersion, Task

SCHEMA_VERSION = 1

# store_status: what happened to the task, as opposed to the solver's status.
OK = "ok"
ERROR = "error"
KILLED = "killed"

_TABLES = """
CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR);
CREATE TABLE IF NOT EXISTS runs (
    key VARCHAR PRIMARY KEY,
    kind VARCHAR, tier VARCHAR, dials VARCHAR, instance_seed BIGINT,
    instance_hash VARCHAR, notional DOUBLE,
    solver VARCHAR, encoding VARCHAR, zeta_bits INTEGER, solver_seed BIGINT,
    budget_s DOUBLE, max_samples BIGINT,
    commit VARCHAR, lock_hash VARCHAR, dirty BOOLEAN, machine VARCHAR,
    packages VARCHAR,
    store_status VARCHAR, status VARCHAR, error VARCHAR,
    schedule VARCHAR, zeta DOUBLE,
    spread DOUBLE, temporary DOUBLE, permanent DOUBLE, risk DOUBLE, total DOUBLE,
    total_bps DOUBLE,
    lower DOUBLE, upper DOUBLE,
    feasible BOOLEAN, violations VARCHAR,
    n_samples BIGINT, n_feasible_samples BIGINT, best_energy DOUBLE,
    energy_mismatch DOUBLE,
    qubo_energy DOUBLE, qubo_total DOUBLE,
    dynamic_range DOUBLE, n_vars INTEGER,
    total_s DOUBLE, timings VARCHAR, detail VARCHAR
);
CREATE TABLE IF NOT EXISTS samples (
    key VARCHAR, digest VARCHAR, count BIGINT, energy DOUBLE,
    classification VARCHAR, total DOUBLE
);
CREATE TABLE IF NOT EXISTS traces (key VARCHAR, seconds DOUBLE, best_energy DOUBLE);
"""

RUN_COLUMNS = (
    "key", "kind", "tier", "dials", "instance_seed", "instance_hash", "notional",
    "solver", "encoding", "zeta_bits", "solver_seed", "budget_s", "max_samples",
    "commit", "lock_hash", "dirty", "machine", "packages",
    "store_status", "status", "error", "schedule", "zeta",
    "spread", "temporary", "permanent", "risk", "total", "total_bps",
    "lower", "upper", "feasible", "violations",
    "n_samples", "n_feasible_samples", "best_energy", "energy_mismatch",
    "qubo_energy", "qubo_total", "dynamic_range", "n_vars",
    "total_s", "timings", "detail",
)


# Column types of the tables written from frames, after ``key``, in table order.
_TYPES = {
    "samples": (pl.String, pl.Int64, pl.Float64, pl.String, pl.Float64),
    "traces": (pl.Float64, pl.Float64),
}


def _frame(key: str, rows: list[tuple], types: tuple) -> pl.DataFrame:
    """``key`` and the rows' columns, by position, as the table declares them.

    Ragged rows are refused. Rows of one uniform wrong width are left for
    DuckDB to refuse, as it would refuse them row by row.
    """
    columns = list(zip(*rows, strict=True))
    data = {"key": pl.Series("key", [key] * len(rows), dtype=pl.String)}
    for i, values in enumerate(columns):
        dtype = types[i] if i < len(types) else None
        data[f"c{i}"] = pl.Series(f"c{i}", values, dtype=dtype)
    return pl.DataFrame(data)


def blank_row(
    key: str, task: Task, code: CodeVersion, where: str, versions: str
) -> dict:
    """Every column, with the task's coordinates filled and nothing measured."""
    row = dict.fromkeys(RUN_COLUMNS)
    row.update(
        key=key, kind=task.kind, tier=task.tier, dials=task.dials,
        instance_seed=task.instance_seed, solver=task.solver, encoding=task.encoding,
        zeta_bits=task.zeta_bits, solver_seed=task.solver_seed, budget_s=task.budget_s,
        max_samples=task.max_samples, commit=code.commit, lock_hash=code.lock_hash,
        dirty=code.dirty, machine=where, packages=versions,
    )
    return row


class SchemaMismatch(RuntimeError):
    """The file was written under another schema version."""


class StoreBusy(RuntimeError):
    """Another process has the store open for writing."""


class Store:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        if read_only and not self.path.exists():
            raise FileNotFoundError(f"no store at {self.path}")
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.con = duckdb.connect(str(self.path), read_only=read_only)
        except duckdb.IOException as exc:
            # The wording is the operating system's, and localised on Windows,
            # so the test is that the file is there and still cannot be opened.
            if not self.path.exists():
                raise
            raise StoreBusy(
                f"{self.path} exists but cannot be opened, most likely because a "
                "running `quantic bench run` holds it: DuckDB lets one process at "
                "a time open a file. Wait for the run, or copy the file and read "
                f"the copy. DuckDB said: {exc}"
            ) from exc
        if not read_only:
            self.con.execute(_TABLES)
        found = self.con.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if found is None and not read_only:
            self.con.execute(
                "INSERT INTO meta VALUES ('schema_version', ?)", [str(SCHEMA_VERSION)]
            )
        elif found is not None and int(found[0]) != SCHEMA_VERSION:
            raise SchemaMismatch(
                f"{self.path} has schema version {found[0]}; this code reads "
                f"{SCHEMA_VERSION}"
            )

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    def done_keys(self) -> set[str]:
        return {row[0] for row in self.con.execute("SELECT key FROM runs").fetchall()}

    def write(self, row: dict, samples: list[tuple] = (), trace: list[tuple] = ()) -> None:
        """One finished task, all its rows, in one transaction.

        Samples and trace points go in as one frame each: a sampler run can keep
        a hundred thousand distinct samples, and row-at-a-time inserts would
        leave the study waiting on its writer.
        """
        missing = set(RUN_COLUMNS) - set(row)
        extra = set(row) - set(RUN_COLUMNS)
        if missing or extra:
            raise ValueError(f"run row: missing {sorted(missing)}, unknown {sorted(extra)}")
        values = [
            json.dumps(row[c]) if isinstance(row[c], (dict, list, tuple)) else row[c]
            for c in RUN_COLUMNS
        ]
        marks = ", ".join("?" for _ in RUN_COLUMNS)
        self.con.execute("BEGIN TRANSACTION")
        try:
            self.con.execute(f"INSERT INTO runs VALUES ({marks})", values)
            for table, rows in (("samples", samples), ("traces", trace)):
                if rows:
                    self._insert(table, _frame(row["key"], rows, _TYPES[table]))
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def _insert(self, table: str, frame: pl.DataFrame) -> None:
        self.con.register("_rows", frame)
        try:
            self.con.execute(f"INSERT INTO {table} SELECT * FROM _rows")
        finally:
            self.con.unregister("_rows")

    def runs(self) -> pl.DataFrame:
        return self.con.execute("SELECT * FROM runs ORDER BY key").pl()

    def samples(self) -> pl.DataFrame:
        return self.con.execute("SELECT * FROM samples ORDER BY key, digest").pl()
