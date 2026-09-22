# Quantic M0–M1: Data Layer and Microstructure Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a content-hashed market data bundle format with a synthetic generator carrying known ground-truth impact parameters, plus a microstructure core (book reconstruction, liquidity metrics, covariance, impact models) whose correctness is verified by L3-vs-L2 cross-validation and ground-truth parameter recovery.

**Architecture:** Strictly layered — `core` (types, no deps) → `data` (bundle format, schemas, synthetic generator) → `micro` (reconstruction, liquidity, impact). Layer boundaries are enforced mechanically by `import-linter` in CI from the very first task, because the entire benchmark design depends on hardness features and solvers varying independently of the microstructure code. The synthetic generator is the keystone: it emits data whose true impact exponents are known, so calibration can be tested by parameter recovery rather than by eyeball.

**Tech Stack:** Python 3.11+, `uv`, Polars, PyArrow, NumPy, SciPy, scikit-learn, Typer, pytest, Hypothesis, ruff, import-linter.

**Spec:** `docs/superpowers/specs/2026-09-22-quantum-liquidation-lvar-design.md`

## Global Constraints

- **Python `>=3.11`.** Environment managed with `uv`. All commands run as `uv run ...`.
- **Free-tier only.** No task may introduce a dependency requiring a paid licence or account. Gurobi is optional and must never be imported at module scope.
- **Layer boundaries (spec §12), enforced by `import-linter` in CI:** `micro/` must not import `problem/`, `encoding/`, or `solvers/`; `problem/` must not import `encoding/` or `solvers/`; `solvers/` must not import `problem/` internals.
- **Determinism.** Every stochastic function takes an explicit `seed`. Identical seed plus identical inputs must produce byte-identical output. There are no implicit global RNGs.
- **Content hashing.** Every dataset bundle carries a manifest with per-file SHA-256 and a deterministic aggregate `content_hash`. No benchmark result may reference unhashed data.
- **Strictness over silent repair.** Malformed messages, unknown order IDs, and insufficient book depth raise typed exceptions. Nothing is silently dropped or patched.
- **Line length 100**, ruff rules `E,F,I,UP,B,SIM`.

## Deliberate Deviations From The Spec

Two simplifications, recorded so a reviewer does not flag them as drift:

1. **No `data/loaders/` package.** The spec listed `loaders/{l1,l2,l3,daily}.py`. Since bundles are Parquet, a "loader" is `pl.read_parquet` plus schema validation — four near-empty files. Loading is folded into `DatasetBundle` accessor methods (`bundle.l1()`, `bundle.l2()`, …), which is the single access point anyway.
2. **Partition by symbol only, not symbol *and* date.** Symbol partitioning keeps per-file sizes manageable at the planned 25-symbol / 20-day scale. Date partitioning is added only if file sizes demand it.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, ruff/pytest/import-linter config |
| `src/quantic/core/types.py` | `Side`, `Asset`, `PriceLevel`, `BookSnapshot`. No dependencies. |
| `src/quantic/data/schemas.py` | PyArrow schemas for the four granularities plus `validate()` |
| `src/quantic/data/manifest.py` | `Manifest` dataclass, SHA-256 file hashing, deterministic `content_hash` |
| `src/quantic/data/bundle.py` | `DatasetBundle`: write, load, validate, typed table accessors |
| `src/quantic/data/synth.py` | `SynthConfig` and the generator producing all four granularities with known ground truth |
| `src/quantic/data/catalog.py` | Local registry mapping bundle IDs to paths |
| `src/quantic/data/request.py` | Emits a data-request spec for the external market data machine |
| `src/quantic/cli.py` | Typer app: `quantic data synth\|ingest\|request\|list` |
| `src/quantic/micro/book_reconstruct.py` | `BookBuilder` and `replay()` — L3 message stream to `BookSnapshot` |
| `src/quantic/micro/liquidity.py` | Spread, depth, bucket volume, ADV, resiliency half-life |
| `src/quantic/micro/covariance.py` | Ledoit–Wolf shrinkage covariance from daily bars |
| `src/quantic/micro/impact/base.py` | `ImpactParams`, `ImpactModel` protocol, unit conventions |
| `src/quantic/micro/impact/almgren_chriss.py` | Linear temporary impact, quadratic cost |
| `src/quantic/micro/impact/sqrt_law.py` | Concave power-law impact, the hardness source (spec dial D1) |
| `src/quantic/micro/impact/depth_walk.py` | Empirical cost of walking a real `BookSnapshot` |
| `src/quantic/micro/impact/calibrate.py` | Bucket-level impact regression recovering `delta` and `Y` |

**Unit convention, fixed here and used by every impact model** (a later task that breaks it is a bug):

- `price_impact(q, ...)` returns the **fractional** mid displacement (dimensionless), always non-negative; the caller applies the sign.
- `temporary_cost(q, ...)` returns **fractional cost**, i.e. currency cost divided by `q * price`. Total currency cost is `temporary_cost * q * price`.
- Integrating a power-law impact over the executed quantity gives `temporary_cost = price_impact / (delta + 1)`. For `delta = 0.5` that is the standard 2/3 rule; for linear impact (`delta = 1`) it is the standard 1/2. Both are asserted in tests.

---

### Task 1: Project scaffold with enforced layer boundaries

**Files:**
- Create: `pyproject.toml`, `.python-version`
- Create: `src/quantic/__init__.py` and empty `__init__.py` for every package in the final layout
- Test: `tests/test_architecture.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable `quantic` package with `quantic.__version__: str`; the full package skeleton (`core`, `data`, `micro`, `micro.impact`, `problem`, `problem.constraints`, `encoding`, `solvers`, `solvers.classical`, `solvers.quantum`, `bench`, `viz`, `dashboard`) so layer contracts are enforceable from day one.

- [ ] **Step 1: Write the failing test**

Create `tests/test_architecture.py`:

```python
import subprocess
import sys


def test_package_imports_and_exposes_version():
    import quantic

    assert isinstance(quantic.__version__, str)
    assert quantic.__version__


def test_layer_contract_holds():
    """The layered architecture from spec section 12 is enforced mechanically."""
    result = subprocess.run(
        [sys.executable, "-m", "importlinter.cli", "lint-imports"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_architecture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic'`

- [ ] **Step 3: Create the project and package skeleton**

Create `.python-version` containing exactly:

```
3.11
```

Create `pyproject.toml`:

```toml
[project]
name = "quantic"
version = "0.1.0"
description = "Quantum vs classical benchmark platform for combinatorial liquidation and L-VaR"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "polars>=1.0",
    "pyarrow>=16.0",
    "scikit-learn>=1.4",
    "typer>=0.12",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "hypothesis>=6.100",
    "ruff>=0.5",
    "import-linter>=2.0",
]

[project.scripts]
quantic = "quantic.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/quantic"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.importlinter]
root_package = "quantic"

[[tool.importlinter.contracts]]
name = "Layered architecture (spec section 12)"
type = "layers"
layers = [
    "quantic.dashboard",
    "quantic.viz",
    "quantic.bench",
    "quantic.solvers",
    "quantic.encoding",
    "quantic.problem",
    "quantic.micro",
    "quantic.data",
    "quantic.core",
]
exhaustive = false
```

Create the package skeleton. Every directory listed gets an empty `__init__.py`:

```bash
mkdir -p src/quantic/{core,data,micro/impact,problem/constraints,encoding,solvers/classical,solvers/quantum,bench,viz,dashboard}
mkdir -p tests/{core,data,micro}
for d in core data micro micro/impact problem problem/constraints encoding solvers solvers/classical solvers/quantum bench viz dashboard; do
  touch "src/quantic/$d/__init__.py"
done
```

Create `src/quantic/__init__.py`:

```python
"""Quantic: quantum vs classical benchmarking for combinatorial liquidation and L-VaR."""

__version__ = "0.1.0"
```

Then install: `uv sync --extra dev`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_architecture.py -v`
Expected: PASS, 2 tests

Also confirm the contract is genuinely active rather than vacuously passing:

Run: `uv run lint-imports`
Expected: output names the contract "Layered architecture (spec section 12)" and reports it KEPT.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .python-version uv.lock src/quantic tests/test_architecture.py
git commit -m "feat: project scaffold with mechanically enforced layer boundaries"
```

---

### Task 2: Core domain types

**Files:**
- Create: `src/quantic/core/types.py`
- Test: `tests/core/test_types.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Side` — `str` enum with members `BUY = "buy"`, `SELL = "sell"`
  - `Asset(symbol: str, lot_size: int, tick_size: float, currency: str = "USD")` — frozen
  - `PriceLevel(price: float, size: int)` — frozen
  - `BookSnapshot(ts_ns: int, symbol: str, bids: tuple[PriceLevel, ...], asks: tuple[PriceLevel, ...])` — frozen, `bids` descending by price, `asks` ascending. Properties: `best_bid`, `best_ask`, `mid`, `spread`, each `float | None` when the relevant side is empty.
  - `EmptyBookError(ValueError)`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_types.py`:

```python
import pytest

from quantic.core.types import Asset, BookSnapshot, EmptyBookError, PriceLevel, Side


def _book() -> BookSnapshot:
    return BookSnapshot(
        ts_ns=1_000,
        symbol="SYNA",
        bids=(PriceLevel(99.99, 500), PriceLevel(99.98, 800)),
        asks=(PriceLevel(100.01, 400), PriceLevel(100.02, 900)),
    )


def test_side_values():
    assert Side.BUY.value == "buy"
    assert Side.SELL.value == "sell"


def test_asset_is_frozen():
    a = Asset(symbol="SYNA", lot_size=100, tick_size=0.01)
    assert a.currency == "USD"
    with pytest.raises(AttributeError):
        a.symbol = "OTHER"  # type: ignore[misc]


def test_book_touch_and_derived_prices():
    b = _book()
    assert b.best_bid == pytest.approx(99.99)
    assert b.best_ask == pytest.approx(100.01)
    assert b.mid == pytest.approx(100.00)
    assert b.spread == pytest.approx(0.02)


def test_empty_side_yields_none_not_crash():
    b = BookSnapshot(ts_ns=1, symbol="SYNA", bids=(), asks=(PriceLevel(100.01, 10),))
    assert b.best_bid is None
    assert b.mid is None
    assert b.spread is None


def test_bid_ordering_is_validated():
    with pytest.raises(ValueError, match="descending"):
        BookSnapshot(
            ts_ns=1,
            symbol="SYNA",
            bids=(PriceLevel(99.98, 10), PriceLevel(99.99, 10)),
            asks=(),
        )


def test_ask_ordering_is_validated():
    with pytest.raises(ValueError, match="ascending"):
        BookSnapshot(
            ts_ns=1,
            symbol="SYNA",
            bids=(),
            asks=(PriceLevel(100.02, 10), PriceLevel(100.01, 10)),
        )


def test_crossed_book_is_rejected():
    with pytest.raises(EmptyBookError) as exc:
        BookSnapshot(
            ts_ns=1,
            symbol="SYNA",
            bids=(PriceLevel(100.05, 10),),
            asks=(PriceLevel(100.01, 10),),
        ).mid
    assert "crossed" in str(exc.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.core.types'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/core/types.py`:

```python
"""Core domain types. This module must not import from any other quantic package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class EmptyBookError(ValueError):
    """Raised when a derived price is requested from a book that cannot supply it."""


@dataclass(frozen=True, slots=True)
class Asset:
    symbol: str
    lot_size: int
    tick_size: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {self.lot_size}")
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")


@dataclass(frozen=True, slots=True)
class PriceLevel:
    price: float
    size: int


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """A point-in-time view of one symbol's order book.

    ``bids`` are ordered by descending price, ``asks`` by ascending price.
    Both orderings are validated on construction, because every downstream
    consumer indexes level 0 as the touch.
    """

    ts_ns: int
    symbol: str
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]

    def __post_init__(self) -> None:
        bid_prices = [lvl.price for lvl in self.bids]
        if bid_prices != sorted(bid_prices, reverse=True):
            raise ValueError(f"bids must be in descending price order, got {bid_prices}")
        ask_prices = [lvl.price for lvl in self.asks]
        if ask_prices != sorted(ask_prices):
            raise ValueError(f"asks must be in ascending price order, got {ask_prices}")

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        if bid >= ask:
            raise EmptyBookError(
                f"crossed book for {self.symbol} at ts_ns={self.ts_ns}: bid {bid} >= ask {ask}"
            )
        return (bid + ask) / 2.0

    @property
    def spread(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_types.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/core/types.py tests/core/test_types.py
git commit -m "feat(core): add domain types with book ordering invariants"
```

---

### Task 3: Parquet schemas for the four granularities

**Files:**
- Create: `src/quantic/data/schemas.py`
- Test: `tests/data/test_schemas.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `SCHEMA_VERSION: str` (value `"1"`)
  - `L1_TAQ`, `L2_DEPTH`, `L3_MESSAGES`, `DAILY_BARS` — `pyarrow.Schema` objects
  - `SCHEMAS: dict[str, pa.Schema]` keyed by `"l1_taq"`, `"l2_depth"`, `"l3_messages"`, `"daily_bars"`
  - `SchemaError(ValueError)`
  - `validate(name: str, table: pa.Table) -> None` — raises `SchemaError` on unknown name, wrong column set, wrong column order, or wrong dtype
  - `L3Action` — `str` enum: `ADD = "add"`, `CANCEL = "cancel"`, `EXECUTE = "execute"`, `REPLACE = "replace"`

Note the `seq` column on `L3_MESSAGES`: message streams carry ties in `ts_ns`, and replay must be deterministic, so an explicit monotone sequence number is part of the schema rather than an afterthought.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_schemas.py`:

```python
import polars as pl
import pyarrow as pa
import pytest

from quantic.data.schemas import SCHEMAS, L3Action, SchemaError, validate


def test_all_four_granularities_present():
    assert set(SCHEMAS) == {"l1_taq", "l2_depth", "l3_messages", "daily_bars"}


def test_l3_carries_sequence_number_for_deterministic_replay():
    assert "seq" in SCHEMAS["l3_messages"].names


def test_l3_action_values():
    assert {a.value for a in L3Action} == {"add", "cancel", "execute", "replace"}


def test_validate_accepts_conforming_table():
    df = pl.DataFrame(
        {
            "ts_ns": pl.Series([1], dtype=pl.Int64),
            "symbol": pl.Series(["SYNA"], dtype=pl.Utf8),
            "bid": pl.Series([99.99], dtype=pl.Float64),
            "ask": pl.Series([100.01], dtype=pl.Float64),
            "bid_size": pl.Series([500], dtype=pl.Int64),
            "ask_size": pl.Series([400], dtype=pl.Int64),
            "last_px": pl.Series([100.0], dtype=pl.Float64),
            "last_size": pl.Series([100], dtype=pl.Int64),
        }
    )
    validate("l1_taq", df.to_arrow())


def test_validate_rejects_unknown_granularity():
    with pytest.raises(SchemaError, match="unknown granularity"):
        validate("l4_telepathy", pa.table({"x": [1]}))


def test_validate_rejects_wrong_column_set():
    with pytest.raises(SchemaError, match="column mismatch"):
        validate("l1_taq", pa.table({"ts_ns": pa.array([1], type=pa.int64())}))


def test_validate_rejects_wrong_dtype():
    df = pl.DataFrame(
        {
            "ts_ns": pl.Series([1], dtype=pl.Int32),  # wrong: must be Int64
            "symbol": pl.Series(["SYNA"], dtype=pl.Utf8),
            "bid": pl.Series([99.99], dtype=pl.Float64),
            "ask": pl.Series([100.01], dtype=pl.Float64),
            "bid_size": pl.Series([500], dtype=pl.Int64),
            "ask_size": pl.Series([400], dtype=pl.Int64),
            "last_px": pl.Series([100.0], dtype=pl.Float64),
            "last_size": pl.Series([100], dtype=pl.Int64),
        }
    )
    with pytest.raises(SchemaError, match="dtype mismatch"):
        validate("l1_taq", df.to_arrow())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.data.schemas'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/data/schemas.py`:

```python
"""PyArrow schemas for the four market data granularities, plus validation."""

from __future__ import annotations

from enum import Enum

import pyarrow as pa

SCHEMA_VERSION = "1"


class L3Action(str, Enum):
    ADD = "add"
    CANCEL = "cancel"
    EXECUTE = "execute"
    REPLACE = "replace"


class SchemaError(ValueError):
    """Raised when a table does not conform to its declared granularity schema."""


L1_TAQ = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("symbol", pa.string()),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("bid_size", pa.int64()),
        ("ask_size", pa.int64()),
        ("last_px", pa.float64()),
        ("last_size", pa.int64()),
    ]
)

L2_DEPTH = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("symbol", pa.string()),
        ("side", pa.string()),
        ("level", pa.int32()),
        ("px", pa.float64()),
        ("size", pa.int64()),
    ]
)

# ``seq`` disambiguates messages sharing a timestamp so replay is deterministic.
L3_MESSAGES = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("seq", pa.int64()),
        ("symbol", pa.string()),
        ("order_id", pa.int64()),
        ("action", pa.string()),
        ("side", pa.string()),
        ("px", pa.float64()),
        ("size", pa.int64()),
    ]
)

DAILY_BARS = pa.schema(
    [
        ("date", pa.date32()),
        ("symbol", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
        ("adv", pa.float64()),
    ]
)

SCHEMAS: dict[str, pa.Schema] = {
    "l1_taq": L1_TAQ,
    "l2_depth": L2_DEPTH,
    "l3_messages": L3_MESSAGES,
    "daily_bars": DAILY_BARS,
}


def validate(name: str, table: pa.Table) -> None:
    """Raise :class:`SchemaError` unless ``table`` conforms exactly to ``name``'s schema.

    Column order is part of the contract: bundles are content-hashed, and a
    reordered table would hash differently while being semantically identical,
    which would silently break reproducibility claims.
    """
    expected = SCHEMAS.get(name)
    if expected is None:
        raise SchemaError(f"unknown granularity {name!r}; expected one of {sorted(SCHEMAS)}")

    if list(table.schema.names) != list(expected.names):
        raise SchemaError(
            f"column mismatch for {name!r}: "
            f"expected {list(expected.names)}, got {list(table.schema.names)}"
        )

    for field in expected:
        actual = table.schema.field(field.name).type
        if actual != field.type:
            raise SchemaError(
                f"dtype mismatch for {name!r}.{field.name}: expected {field.type}, got {actual}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_schemas.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/data/schemas.py tests/data/test_schemas.py
git commit -m "feat(data): add parquet schemas and strict validation for four granularities"
```

---

### Task 4: Manifest and deterministic content hashing

**Files:**
- Create: `src/quantic/data/manifest.py`
- Test: `tests/data/test_manifest.py`

**Interfaces:**
- Consumes: `quantic.data.schemas.SCHEMA_VERSION`
- Produces:
  - `sha256_file(path: Path) -> str` — streaming hash, 1 MiB chunks
  - `compute_content_hash(file_hashes: Mapping[str, str]) -> str` — order-independent aggregate
  - `Manifest` — frozen dataclass with fields `schema_version: str`, `bundle_id: str`, `created_utc: str`, `provenance: str`, `symbols: tuple[str, ...]`, `start_date: str`, `end_date: str`, `granularities: tuple[str, ...]`, `files: dict[str, str]`, `content_hash: str`, `extra: dict[str, Any]`. Methods `to_json() -> str`, `write(path: Path) -> None`, classmethod `read(path: Path) -> Manifest`.
  - `MANIFEST_FILENAME: str` (value `"manifest.json"`)

`compute_content_hash` sorts paths and uses NUL separators so that no combination of path and hash can be reinterpreted as a different combination.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_manifest.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.data.manifest'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/data/manifest.py`:

```python
"""Bundle manifests and deterministic content hashing.

Every benchmark result pins a bundle's ``content_hash``, so this module is
load-bearing for the reproducibility claims in spec section 9.3.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "manifest.json"
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def compute_content_hash(file_hashes: Mapping[str, str]) -> str:
    """Aggregate per-file hashes into one order-independent digest.

    Paths are sorted, and each path and hash is NUL-terminated so that no two
    distinct mappings can serialise to the same byte stream.
    """
    h = hashlib.sha256()
    for path in sorted(file_hashes):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(file_hashes[path].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass(frozen=True)
class Manifest:
    schema_version: str
    bundle_id: str
    created_utc: str
    provenance: str
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    granularities: tuple[str, ...]
    files: dict[str, str]
    content_hash: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = compute_content_hash(self.files)
        if self.content_hash != expected:
            raise ValueError(
                f"content_hash does not match files: expected {expected}, got {self.content_hash}"
            )

    def to_json(self) -> str:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["granularities"] = list(self.granularities)
        return json.dumps(payload, indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Manifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["symbols"] = tuple(payload["symbols"])
        payload["granularities"] = tuple(payload["granularities"])
        return cls(**payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_manifest.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/data/manifest.py tests/data/test_manifest.py
git commit -m "feat(data): add manifest with deterministic content hashing"
```

---
### Task 5: DatasetBundle write, load, and integrity validation

**Files:**
- Create: `src/quantic/data/bundle.py`
- Test: `tests/data/test_bundle.py`

**Interfaces:**
- Consumes: `quantic.data.schemas.{SCHEMAS, SCHEMA_VERSION, validate}`, `quantic.data.manifest.{Manifest, MANIFEST_FILENAME, compute_content_hash, sha256_file}`
- Produces:
  - `BundleIntegrityError(RuntimeError)`
  - `SORT_KEYS: dict[str, list[str]]` — canonical row ordering per granularity
  - `DatasetBundle(root: Path, manifest: Manifest)` with:
    - classmethod `write(root: Path, tables: Mapping[str, pl.DataFrame], *, bundle_id: str, provenance: str, extra: dict | None = None) -> DatasetBundle`
    - classmethod `load(root: Path) -> DatasetBundle`
    - `validate() -> None` — re-hashes every file, raises `BundleIntegrityError` on mismatch
    - `table(name: str) -> pl.DataFrame`
    - `l1() / l2() / l3() / daily()` accessors
    - properties `content_hash: str`, `symbols: tuple[str, ...]`

Rows are sorted to a canonical order before writing. Without this, two semantically identical bundles would produce different content hashes, which would quietly invalidate the reproducibility guarantee.

`write` requires a `daily_bars` table, because the manifest's date range is derived from it and every later milestone needs it for covariance estimation.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_bundle.py`:

```python
import datetime as dt

import polars as pl
import pytest

from quantic.data.bundle import BundleIntegrityError, DatasetBundle


def _daily() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": pl.Series(
                [dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 5)],
                dtype=pl.Date,
            ),
            "symbol": pl.Series(["SYNB", "SYNB", "SYNA"], dtype=pl.Utf8),
            "open": pl.Series([100.0, 101.0, 50.0], dtype=pl.Float64),
            "high": pl.Series([102.0, 103.0, 51.0], dtype=pl.Float64),
            "low": pl.Series([99.0, 100.0, 49.0], dtype=pl.Float64),
            "close": pl.Series([101.0, 102.0, 50.5], dtype=pl.Float64),
            "volume": pl.Series([1000, 1100, 900], dtype=pl.Int64),
            "adv": pl.Series([1000.0, 1050.0, 900.0], dtype=pl.Float64),
        }
    )


def _l1() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_ns": pl.Series([200, 100], dtype=pl.Int64),
            "symbol": pl.Series(["SYNA", "SYNA"], dtype=pl.Utf8),
            "bid": pl.Series([49.99, 49.98], dtype=pl.Float64),
            "ask": pl.Series([50.01, 50.00], dtype=pl.Float64),
            "bid_size": pl.Series([500, 500], dtype=pl.Int64),
            "ask_size": pl.Series([500, 500], dtype=pl.Int64),
            "last_px": pl.Series([50.0, 49.99], dtype=pl.Float64),
            "last_size": pl.Series([100, 100], dtype=pl.Int64),
        }
    )


def _write(tmp_path) -> DatasetBundle:
    return DatasetBundle.write(
        tmp_path / "bundle",
        {"daily_bars": _daily(), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
        extra={"ground_truth": {"impact_delta": {"SYNA": 0.5}}},
    )


def test_write_then_load_roundtrips(tmp_path):
    written = _write(tmp_path)
    loaded = DatasetBundle.load(tmp_path / "bundle")
    assert loaded.content_hash == written.content_hash
    assert loaded.symbols == ("SYNA", "SYNB")
    assert loaded.manifest.extra["ground_truth"]["impact_delta"]["SYNA"] == 0.5


def test_tables_are_returned_in_canonical_sorted_order(tmp_path):
    _write(tmp_path)
    l1 = DatasetBundle.load(tmp_path / "bundle").l1()
    assert l1["ts_ns"].to_list() == [100, 200]


def test_date_range_is_derived_from_daily_bars(tmp_path):
    b = _write(tmp_path)
    assert b.manifest.start_date == "2026-01-05"
    assert b.manifest.end_date == "2026-01-06"


def test_content_hash_is_stable_across_identical_writes(tmp_path):
    a = _write(tmp_path)
    b = DatasetBundle.write(
        tmp_path / "bundle2",
        {"daily_bars": _daily(), "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
        extra={"ground_truth": {"impact_delta": {"SYNA": 0.5}}},
    )
    assert a.content_hash == b.content_hash


def test_content_hash_changes_when_data_changes(tmp_path):
    a = _write(tmp_path)
    changed = _daily().with_columns(pl.col("close") + 1.0)
    b = DatasetBundle.write(
        tmp_path / "bundle3",
        {"daily_bars": changed, "l1_taq": _l1()},
        bundle_id="test-bundle",
        provenance="unit-test",
    )
    assert a.content_hash != b.content_hash


def test_validate_detects_tampering(tmp_path):
    _write(tmp_path)
    bundle = DatasetBundle.load(tmp_path / "bundle")
    bundle.validate()  # clean

    victim = next((tmp_path / "bundle" / "l1_taq").rglob("*.parquet"))
    victim.write_bytes(victim.read_bytes() + b"tampered")

    with pytest.raises(BundleIntegrityError, match="hash mismatch"):
        DatasetBundle.load(tmp_path / "bundle").validate()


def test_validate_detects_missing_file(tmp_path):
    _write(tmp_path)
    next((tmp_path / "bundle" / "l1_taq").rglob("*.parquet")).unlink()
    with pytest.raises(BundleIntegrityError, match="missing"):
        DatasetBundle.load(tmp_path / "bundle").validate()


def test_write_rejects_nonconforming_table(tmp_path):
    from quantic.data.schemas import SchemaError

    bad = _l1().drop("last_size")
    with pytest.raises(SchemaError):
        DatasetBundle.write(
            tmp_path / "bad",
            {"daily_bars": _daily(), "l1_taq": bad},
            bundle_id="bad",
            provenance="unit-test",
        )


def test_write_requires_daily_bars(tmp_path):
    with pytest.raises(ValueError, match="daily_bars"):
        DatasetBundle.write(
            tmp_path / "nodaily",
            {"l1_taq": _l1()},
            bundle_id="nodaily",
            provenance="unit-test",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_bundle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.data.bundle'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/data/bundle.py`:

```python
"""The ingestion boundary: a content-hashed, on-disk market data bundle.

Layout::

    root/manifest.json
    root/l1_taq/symbol=SYNA/part.parquet
    root/l2_depth/symbol=SYNA/part.parquet
    root/l3_messages/symbol=SYNA/part.parquet
    root/daily_bars/part.parquet
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from quantic.data.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    compute_content_hash,
    sha256_file,
)
from quantic.data.schemas import SCHEMA_VERSION, SCHEMAS, validate

# Canonical row ordering. Bundles are content-hashed, so ordering is part of
# the data's identity and must not depend on how a caller happened to build it.
SORT_KEYS: dict[str, list[str]] = {
    "l1_taq": ["symbol", "ts_ns"],
    "l2_depth": ["symbol", "ts_ns", "side", "level"],
    "l3_messages": ["symbol", "ts_ns", "seq"],
    "daily_bars": ["symbol", "date"],
}

_PARTITIONED = ("l1_taq", "l2_depth", "l3_messages")


class BundleIntegrityError(RuntimeError):
    """Raised when on-disk bytes do not match the manifest."""


class DatasetBundle:
    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = Path(root)
        self.manifest = manifest

    @property
    def content_hash(self) -> str:
        return self.manifest.content_hash

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.manifest.symbols

    @classmethod
    def write(
        cls,
        root: Path,
        tables: Mapping[str, pl.DataFrame],
        *,
        bundle_id: str,
        provenance: str,
        extra: dict[str, Any] | None = None,
    ) -> DatasetBundle:
        root = Path(root)
        if "daily_bars" not in tables:
            raise ValueError(
                "a bundle must include daily_bars: the manifest date range and all "
                "covariance estimation derive from it"
            )

        root.mkdir(parents=True, exist_ok=True)
        file_hashes: dict[str, str] = {}
        symbols: set[str] = set()

        for name in sorted(tables):
            df = tables[name].sort(SORT_KEYS[name])
            validate(name, df.to_arrow())
            symbols.update(df["symbol"].unique().to_list())

            if name in _PARTITIONED:
                for symbol in sorted(df["symbol"].unique().to_list()):
                    rel = f"{name}/symbol={symbol}/part.parquet"
                    dest = root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    df.filter(pl.col("symbol") == symbol).write_parquet(
                        dest, compression="zstd"
                    )
                    file_hashes[rel] = sha256_file(dest)
            else:
                rel = f"{name}/part.parquet"
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(dest, compression="zstd")
                file_hashes[rel] = sha256_file(dest)

        daily = tables["daily_bars"]
        start: dt.date = daily["date"].min()  # type: ignore[assignment]
        end: dt.date = daily["date"].max()  # type: ignore[assignment]

        manifest = Manifest(
            schema_version=SCHEMA_VERSION,
            bundle_id=bundle_id,
            created_utc=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
            provenance=provenance,
            symbols=tuple(sorted(symbols)),
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            granularities=tuple(sorted(tables)),
            files=file_hashes,
            content_hash=compute_content_hash(file_hashes),
            extra=dict(extra or {}),
        )
        manifest.write(root / MANIFEST_FILENAME)
        return cls(root, manifest)

    @classmethod
    def load(cls, root: Path) -> DatasetBundle:
        root = Path(root)
        return cls(root, Manifest.read(root / MANIFEST_FILENAME))

    def validate(self) -> None:
        for rel, expected in sorted(self.manifest.files.items()):
            path = self.root / rel
            if not path.exists():
                raise BundleIntegrityError(f"missing file {rel} in bundle {self.root}")
            actual = sha256_file(path)
            if actual != expected:
                raise BundleIntegrityError(
                    f"hash mismatch for {rel}: manifest {expected}, on disk {actual}"
                )
        recomputed = compute_content_hash(self.manifest.files)
        if recomputed != self.manifest.content_hash:
            raise BundleIntegrityError(
                f"content_hash mismatch: manifest {self.manifest.content_hash}, "
                f"recomputed {recomputed}"
            )

    def table(self, name: str) -> pl.DataFrame:
        if name not in SCHEMAS:
            raise KeyError(f"unknown granularity {name!r}")
        if name not in self.manifest.granularities:
            raise KeyError(f"bundle {self.manifest.bundle_id} has no {name!r} table")
        files = sorted((self.root / name).rglob("*.parquet"))
        return pl.read_parquet(files).sort(SORT_KEYS[name])

    def l1(self) -> pl.DataFrame:
        return self.table("l1_taq")

    def l2(self) -> pl.DataFrame:
        return self.table("l2_depth")

    def l3(self) -> pl.DataFrame:
        return self.table("l3_messages")

    def daily(self) -> pl.DataFrame:
        return self.table("daily_bars")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_bundle.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/data/bundle.py tests/data/test_bundle.py
git commit -m "feat(data): add content-hashed DatasetBundle with integrity validation"
```

---

### Task 6: Synthetic generator — bucket paths, daily bars, L1

**Files:**
- Create: `src/quantic/data/synth.py`
- Test: `tests/data/test_synth_paths.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure generation)
- Produces:
  - `SynthConfig` — frozen dataclass, fields and defaults listed below
  - `ground_truth(cfg: SynthConfig) -> dict[str, Any]` — the parameters calibration must recover
  - `trading_dates(cfg: SynthConfig) -> list[datetime.date]` — consecutive weekdays
  - `bucket_ts_ns(d: datetime.date, bucket: int, buckets_per_day: int) -> tuple[int, int]` — session 09:30–16:00 UTC
  - `round_to_tick(px: float, tick: float) -> float`
  - `generate_buckets(cfg: SynthConfig) -> pl.DataFrame` — columns `symbol, day_index, bucket_index, date, ts_start_ns, ts_end_ns, mid_open, mid_close, bucket_volume, net_flow, participation`
  - `build_daily_bars(buckets: pl.DataFrame, cfg: SynthConfig) -> pl.DataFrame`
  - `build_l1(buckets: pl.DataFrame, cfg: SynthConfig) -> pl.DataFrame`

**The generative model, stated precisely** — calibration in Task 16 must be the exact inverse of this, or the recovery test proves nothing:

For symbol *i*, bucket *t*, with `sigma_bucket = daily_vol / sqrt(buckets_per_day)`:

```
participation f  ~ Normal(0, flow_sd), clipped to [-0.3, 0.3], resampled if |f| < 1e-4
net_flow q       = f * bucket_volume
impact_return    = Y_i * sigma_bucket * sign(f) * |f| ** delta_i
noise            ~ Normal(0, noise_frac * sigma_bucket)
mid_close        = mid_open * (1 + impact_return + noise)
```

`delta_i` and `Y_i` are the ground truth. The default `noise_frac = 1.0` puts realised bucket volatility at roughly `sigma_bucket`, matching the convention that sigma is the return volatility — which is what lets Task 16 estimate sigma independently of the returns it regresses. Prices live on the tick grid: `mid_tick = round(mid / tick)`, `bid = (mid_tick - 1) * tick`, `ask = (mid_tick + 1) * tick`, so the book is never crossed and the spread is exactly two ticks.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_synth_paths.py`:

```python
import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantic.data.synth import (
    SynthConfig,
    bucket_ts_ns,
    build_daily_bars,
    build_l1,
    generate_buckets,
    ground_truth,
    round_to_tick,
    trading_dates,
)

CFG = SynthConfig(symbols=("SYNA", "SYNB"), n_days=4, buckets_per_day=5, seed=7)


def test_trading_dates_are_consecutive_weekdays():
    dates = trading_dates(SynthConfig(n_days=7, start_date=dt.date(2026, 1, 2)))
    assert dates[0] == dt.date(2026, 1, 2)  # Friday
    assert dates[1] == dt.date(2026, 1, 5)  # skips the weekend
    assert all(d.weekday() < 5 for d in dates)
    assert len(dates) == 7


def test_bucket_ts_ns_spans_the_session():
    start, end = bucket_ts_ns(dt.date(2026, 1, 5), 0, 13)
    assert end > start
    last_start, last_end = bucket_ts_ns(dt.date(2026, 1, 5), 12, 13)
    assert last_end - start == 23_400 * 1_000_000_000  # 6.5 hours


def test_round_to_tick():
    assert round_to_tick(100.004, 0.01) == pytest.approx(100.00)
    assert round_to_tick(100.006, 0.01) == pytest.approx(100.01)


def test_ground_truth_covers_every_symbol():
    gt = ground_truth(CFG)
    assert set(gt["impact_delta"]) == set(CFG.symbols)
    assert set(gt["impact_Y"]) == set(CFG.symbols)
    assert all(0.3 < d < 1.0 for d in gt["impact_delta"].values())


def test_generate_buckets_shape_and_columns():
    b = generate_buckets(CFG)
    assert b.height == len(CFG.symbols) * CFG.n_days * CFG.buckets_per_day
    assert set(b.columns) == {
        "symbol", "day_index", "bucket_index", "date", "ts_start_ns", "ts_end_ns",
        "mid_open", "mid_close", "bucket_volume", "net_flow", "participation",
    }


def test_generation_is_deterministic_under_seed():
    a = generate_buckets(CFG)
    b = generate_buckets(CFG)
    assert a.equals(b)


def test_different_seeds_give_different_paths():
    a = generate_buckets(CFG)
    b = generate_buckets(SynthConfig(symbols=CFG.symbols, n_days=4, buckets_per_day=5, seed=8))
    assert not a.equals(b)


def test_mid_is_continuous_across_buckets():
    b = generate_buckets(CFG).filter(pl.col("symbol") == "SYNA").sort(
        ["day_index", "bucket_index"]
    )
    closes = b["mid_close"].to_list()[:-1]
    opens = b["mid_open"].to_list()[1:]
    assert closes == pytest.approx(opens)


def test_impact_sign_follows_flow_sign_on_average():
    """With low noise, buying buckets push the mid up."""
    cfg = SynthConfig(symbols=("SYNA",), n_days=20, buckets_per_day=13, seed=1, noise_frac=0.0)
    b = generate_buckets(cfg)
    ret = (b["mid_close"] / b["mid_open"] - 1.0).to_numpy()
    flow = b["participation"].to_numpy()
    assert np.corrcoef(np.sign(flow), np.sign(ret))[0, 1] > 0.99


def test_daily_bars_conform_and_aggregate_buckets():
    from quantic.data.schemas import validate

    b = generate_buckets(CFG)
    daily = build_daily_bars(b, CFG)
    validate("daily_bars", daily.to_arrow())
    assert daily.height == len(CFG.symbols) * CFG.n_days
    row = daily.filter((pl.col("symbol") == "SYNA")).sort("date").row(0, named=True)
    assert row["low"] <= row["open"] <= row["high"]
    assert row["low"] <= row["close"] <= row["high"]
    assert row["volume"] > 0
    assert row["adv"] > 0


def test_l1_conforms_and_is_never_crossed():
    from quantic.data.schemas import validate

    b = generate_buckets(CFG)
    l1 = build_l1(b, CFG)
    validate("l1_taq", l1.to_arrow())
    assert l1.height == b.height
    assert (l1["ask"] > l1["bid"]).all()
    spread = (l1["ask"] - l1["bid"]).to_numpy()
    assert spread == pytest.approx(2 * CFG.tick_size)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_synth_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.data.synth'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/data/synth.py`:

```python
"""Synthetic market data with known ground-truth impact parameters.

This module is the keystone of the test strategy: because the true impact
exponent and coefficient are known by construction, calibration can be tested
by parameter recovery rather than by inspection. See spec section 10.4.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

_SESSION_OPEN_SEC = 9 * 3600 + 30 * 60  # 09:30
_SESSION_SECONDS = 23_400  # 6.5 hours
_NS = 1_000_000_000

_DEFAULT_DELTAS = (0.45, 0.50, 0.60)
_DEFAULT_YS = (0.70, 0.80, 0.90)


@dataclass(frozen=True)
class SynthConfig:
    symbols: tuple[str, ...] = ("SYNA", "SYNB", "SYNC")
    n_days: int = 20
    buckets_per_day: int = 13
    seed: int = 0
    start_date: dt.date = dt.date(2026, 1, 5)
    base_price: float = 100.0
    daily_vol: float = 0.02
    adv_shares: int = 5_000_000
    tick_size: float = 0.01
    lot_size: int = 100
    depth_levels: int = 10
    level_size: int = 500
    flow_sd: float = 0.08
    # Diffusion noise as a multiple of sigma_bucket. The default of 1.0 is the
    # realistic regime: mid returns are dominated by diffusion and impact is a
    # small component, so realised bucket volatility tracks daily_vol. Tests
    # that need a clean signal lower it explicitly.
    noise_frac: float = 1.00
    impact_delta: Mapping[str, float] | None = None
    impact_Y: Mapping[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def sigma_bucket(self) -> float:
        return self.daily_vol / np.sqrt(self.buckets_per_day)


def ground_truth(cfg: SynthConfig) -> dict[str, Any]:
    deltas = dict(cfg.impact_delta) if cfg.impact_delta else {
        s: _DEFAULT_DELTAS[i % len(_DEFAULT_DELTAS)] for i, s in enumerate(cfg.symbols)
    }
    ys = dict(cfg.impact_Y) if cfg.impact_Y else {
        s: _DEFAULT_YS[i % len(_DEFAULT_YS)] for i, s in enumerate(cfg.symbols)
    }
    missing = set(cfg.symbols) - set(deltas) | set(cfg.symbols) - set(ys)
    if missing:
        raise ValueError(f"ground truth missing for symbols {sorted(missing)}")
    return {
        "impact_delta": deltas,
        "impact_Y": ys,
        "daily_vol": cfg.daily_vol,
        "buckets_per_day": cfg.buckets_per_day,
        "sigma_bucket": float(cfg.sigma_bucket),
    }


def trading_dates(cfg: SynthConfig) -> list[dt.date]:
    dates: list[dt.date] = []
    d = cfg.start_date
    while len(dates) < cfg.n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)
    return dates


def bucket_ts_ns(d: dt.date, bucket: int, buckets_per_day: int) -> tuple[int, int]:
    midnight = int(
        dt.datetime(d.year, d.month, d.day, tzinfo=dt.UTC).timestamp()
    ) * _NS
    length = _SESSION_SECONDS // buckets_per_day
    start = midnight + (_SESSION_OPEN_SEC + bucket * length) * _NS
    end = midnight + (_SESSION_OPEN_SEC + (bucket + 1) * length) * _NS
    if bucket == buckets_per_day - 1:
        end = midnight + (_SESSION_OPEN_SEC + _SESSION_SECONDS) * _NS
    return start, end


def round_to_tick(px: float, tick: float) -> float:
    return round(round(px / tick) * tick, 10)


def generate_buckets(cfg: SynthConfig) -> pl.DataFrame:
    gt = ground_truth(cfg)
    dates = trading_dates(cfg)
    sigma_b = cfg.sigma_bucket
    rows: list[dict[str, Any]] = []

    for sym_idx, symbol in enumerate(cfg.symbols):
        # One independent stream per symbol keeps a symbol's path stable when
        # other symbols are added or removed from the config.
        rng = np.random.default_rng([cfg.seed, sym_idx])
        delta = gt["impact_delta"][symbol]
        y_coef = gt["impact_Y"][symbol]
        mid = cfg.base_price

        for day_index, date in enumerate(dates):
            for bucket_index in range(cfg.buckets_per_day):
                f = 0.0
                while abs(f) < 1e-4:
                    f = float(np.clip(rng.normal(0.0, cfg.flow_sd), -0.3, 0.3))

                volume = int(
                    round(cfg.adv_shares / cfg.buckets_per_day * rng.lognormal(0.0, 0.15))
                )
                impact = y_coef * sigma_b * np.sign(f) * abs(f) ** delta
                noise = rng.normal(0.0, cfg.noise_frac * sigma_b)
                mid_open = mid
                mid_close = mid_open * (1.0 + impact + noise)
                mid = mid_close

                ts_start, ts_end = bucket_ts_ns(date, bucket_index, cfg.buckets_per_day)
                rows.append(
                    {
                        "symbol": symbol,
                        "day_index": day_index,
                        "bucket_index": bucket_index,
                        "date": date,
                        "ts_start_ns": ts_start,
                        "ts_end_ns": ts_end,
                        "mid_open": mid_open,
                        "mid_close": mid_close,
                        "bucket_volume": volume,
                        "net_flow": f * volume,
                        "participation": f,
                    }
                )

    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "day_index": pl.Int64,
            "bucket_index": pl.Int64,
            "date": pl.Date,
            "ts_start_ns": pl.Int64,
            "ts_end_ns": pl.Int64,
            "mid_open": pl.Float64,
            "mid_close": pl.Float64,
            "bucket_volume": pl.Int64,
            "net_flow": pl.Float64,
            "participation": pl.Float64,
        },
    ).sort(["symbol", "day_index", "bucket_index"])


def build_daily_bars(buckets: pl.DataFrame, cfg: SynthConfig) -> pl.DataFrame:
    daily = (
        buckets.sort(["symbol", "day_index", "bucket_index"])
        .group_by(["symbol", "date"], maintain_order=True)
        .agg(
            pl.col("mid_open").first().alias("open"),
            pl.max_horizontal(pl.col("mid_open"), pl.col("mid_close")).max().alias("high"),
            pl.min_horizontal(pl.col("mid_open"), pl.col("mid_close")).min().alias("low"),
            pl.col("mid_close").last().alias("close"),
            pl.col("bucket_volume").sum().alias("volume"),
        )
        .sort(["symbol", "date"])
    )
    return (
        daily.with_columns(
            pl.col("volume")
            .cast(pl.Float64)
            .rolling_mean(window_size=20, min_samples=1)
            .over("symbol")
            .alias("adv")
        )
        .select("date", "symbol", "open", "high", "low", "close", "volume", "adv")
        .with_columns(pl.col("volume").cast(pl.Int64))
    )


def build_l1(buckets: pl.DataFrame, cfg: SynthConfig) -> pl.DataFrame:
    tick = cfg.tick_size
    mid_tick = (buckets["mid_close"] / tick).round(0)
    bid = ((mid_tick - 1) * tick).round(10)
    ask = ((mid_tick + 1) * tick).round(10)
    return pl.DataFrame(
        {
            "ts_ns": buckets["ts_end_ns"],
            "symbol": buckets["symbol"],
            "bid": bid,
            "ask": ask,
            "bid_size": pl.Series([cfg.level_size] * buckets.height, dtype=pl.Int64),
            "ask_size": pl.Series([cfg.level_size] * buckets.height, dtype=pl.Int64),
            "last_px": (mid_tick * tick).round(10),
            "last_size": buckets["net_flow"].abs().round(0).cast(pl.Int64),
        }
    ).select("ts_ns", "symbol", "bid", "ask", "bid_size", "ask_size", "last_px", "last_size")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_synth_paths.py -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/data/synth.py tests/data/test_synth_paths.py
git commit -m "feat(data): add synthetic bucket paths, daily bars and L1 with known ground truth"
```

---
### Task 7: Synthetic generator — L3 message stream, L2 snapshots, full bundle

**Files:**
- Modify: `src/quantic/data/synth.py` (append; do not alter Task 6's functions)
- Test: `tests/data/test_synth_book.py`

**Interfaces:**
- Consumes: `SynthConfig`, `generate_buckets`, `build_daily_bars`, `build_l1`, `ground_truth` from Task 6; `DatasetBundle.write` from Task 5
- Produces:
  - `build_l3_and_l2(buckets: pl.DataFrame, cfg: SynthConfig) -> tuple[pl.DataFrame, pl.DataFrame]` returning `(l3_messages, l2_depth)`
  - `generate_bundle(root: Path, cfg: SynthConfig, *, bundle_id: str | None = None) -> DatasetBundle`

**Why this is not circular.** The generator maintains its own plain-dictionary book while emitting messages, and snapshots L2 from that. `micro/book_reconstruct.py` (Task 9) is a *separate* implementation that consumes only the message stream. The Task 10 gate compares the two. On real data the independence is total — L2 and L3 are different feed products — so the same test carries over unchanged.

**Timestamp convention.** L2 snapshots are stamped `ts_end_ns - 1`, so that replaying every message with `ts_ns <= snapshot_ts` reproduces exactly the snapshotted state. The next bucket's messages begin at `ts_end_ns`, strictly after. L1 rows (Task 6) are stamped `ts_end_ns` — a separate feed product with its own convention, as in real data.

**Per-bucket message sequence** (deterministic; all prices iterated in sorted order):

1. Cancel every resting order whose price left the desired set.
2. Add an order at each desired price that has none.
3. Execute `|net_flow|` shares against the ask side if flow is positive, the bid side otherwise, walking outward from the touch.
4. Replenish depleted levels with fresh adds.
5. Every `replace_every` buckets, emit one `replace` on the deepest bid so that code path is exercised.
6. Snapshot the top `depth_levels` of each side.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_synth_book.py`:

```python
import polars as pl
import pytest

from quantic.data.schemas import L3Action, validate
from quantic.data.synth import SynthConfig, build_l3_and_l2, generate_buckets, generate_bundle

CFG = SynthConfig(symbols=("SYNA", "SYNB"), n_days=3, buckets_per_day=4, seed=3, depth_levels=5)


def test_l3_and_l2_conform_to_schema():
    l3, l2 = build_l3_and_l2(generate_buckets(CFG), CFG)
    validate("l3_messages", l3.to_arrow())
    validate("l2_depth", l2.to_arrow())


def test_all_four_actions_are_exercised():
    l3, _ = build_l3_and_l2(generate_buckets(CFG), CFG)
    assert set(l3["action"].unique().to_list()) == {a.value for a in L3Action}


def test_sequence_numbers_are_unique_and_monotone_per_symbol():
    l3, _ = build_l3_and_l2(generate_buckets(CFG), CFG)
    for symbol in CFG.symbols:
        seqs = l3.filter(pl.col("symbol") == symbol).sort("seq")["seq"].to_list()
        assert seqs == sorted(seqs)
        assert len(seqs) == len(set(seqs))


def test_message_timestamps_never_reach_the_snapshot_boundary():
    buckets = generate_buckets(CFG)
    l3, l2 = build_l3_and_l2(buckets, CFG)
    snapshot_ts = set(l2["ts_ns"].to_list())
    assert not snapshot_ts & set(l3["ts_ns"].to_list())


def test_l2_has_both_sides_at_full_depth():
    _, l2 = build_l3_and_l2(generate_buckets(CFG), CFG)
    per_snapshot = l2.group_by(["symbol", "ts_ns", "side"]).len()
    assert per_snapshot["len"].unique().to_list() == [CFG.depth_levels]
    assert set(l2["side"].unique().to_list()) == {"buy", "sell"}


def test_l2_levels_are_zero_indexed_and_ordered():
    _, l2 = build_l3_and_l2(generate_buckets(CFG), CFG)
    one = l2.filter(
        (pl.col("symbol") == "SYNA") & (pl.col("ts_ns") == l2["ts_ns"].min())
    )
    bids = one.filter(pl.col("side") == "buy").sort("level")
    asks = one.filter(pl.col("side") == "sell").sort("level")
    assert bids["level"].to_list() == list(range(CFG.depth_levels))
    assert bids["px"].to_list() == sorted(bids["px"].to_list(), reverse=True)
    assert asks["px"].to_list() == sorted(asks["px"].to_list())
    assert asks["px"][0] > bids["px"][0]


def test_generate_bundle_is_valid_and_carries_ground_truth(tmp_path):
    bundle = generate_bundle(tmp_path / "synth", CFG)
    bundle.validate()
    assert set(bundle.manifest.granularities) == {
        "l1_taq", "l2_depth", "l3_messages", "daily_bars"
    }
    gt = bundle.manifest.extra["ground_truth"]
    assert set(gt["impact_delta"]) == set(CFG.symbols)
    assert bundle.l3().height > 0
    assert bundle.l2().height > 0


def test_generate_bundle_is_reproducible(tmp_path):
    a = generate_bundle(tmp_path / "a", CFG, bundle_id="fixed")
    b = generate_bundle(tmp_path / "b", CFG, bundle_id="fixed")
    assert a.content_hash == b.content_hash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_synth_book.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_l3_and_l2'`

- [ ] **Step 3: Append the implementation to `src/quantic/data/synth.py`**

Add these imports at the top of the existing file:

```python
from dataclasses import asdict
from pathlib import Path

from quantic.data.bundle import DatasetBundle
from quantic.data.schemas import L3Action
```

Append:

```python
class _GeneratorBook:
    """A deliberately naive book used only while emitting messages.

    ``micro.book_reconstruct`` is an independent implementation; comparing the
    two is the Task 10 correctness gate, so this class must stay simple and
    must never import from ``quantic.micro``.
    """

    def __init__(self) -> None:
        # side -> price -> order_id -> size
        self.levels: dict[str, dict[float, dict[int, int]]] = {"buy": {}, "sell": {}}

    def prices(self, side: str) -> list[float]:
        return sorted(
            (p for p, orders in self.levels[side].items() if sum(orders.values()) > 0),
            reverse=(side == "buy"),
        )

    def size_at(self, side: str, px: float) -> int:
        return sum(self.levels[side].get(px, {}).values())

    def add(self, side: str, px: float, order_id: int, size: int) -> None:
        self.levels[side].setdefault(px, {})[order_id] = size

    def remove(self, side: str, px: float, order_id: int) -> None:
        self.levels[side].get(px, {}).pop(order_id, None)

    def reduce(self, side: str, px: float, order_id: int, size: int) -> None:
        orders = self.levels[side][px]
        orders[order_id] -= size
        if orders[order_id] <= 0:
            del orders[order_id]


class _MessageLog:
    """Accumulates L3 messages for one symbol with monotone sequence numbers.

    A class rather than a closure defined inside the bucket loop: closures over
    loop-local state are what ruff's B023 warns about, and this is also easier
    to reason about.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.rows: list[dict[str, Any]] = []
        self.seq = 0
        self._ts_base = 0
        self._offset = 0

    def start_bucket(self, ts_base: int) -> None:
        self._ts_base = ts_base
        self._offset = 0

    def emit(self, action: str, side: str, px: float, size: int, order_id: int) -> None:
        self.rows.append(
            {
                "ts_ns": self._ts_base + self._offset,
                "seq": self.seq,
                "symbol": self.symbol,
                "order_id": order_id,
                "action": action,
                "side": side,
                "px": px,
                "size": size,
            }
        )
        self.seq += 1
        self._offset += 1


def build_l3_and_l2(buckets: pl.DataFrame, cfg: SynthConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    tick = cfg.tick_size
    replace_every = 3
    msgs: list[dict[str, Any]] = []
    snaps: list[dict[str, Any]] = []

    for symbol in cfg.symbols:
        book = _GeneratorBook()
        log = _MessageLog(symbol)
        next_order_id = 1
        sub = buckets.filter(pl.col("symbol") == symbol).sort(["day_index", "bucket_index"])

        for n, row in enumerate(sub.iter_rows(named=True)):
            log.start_bucket(row["ts_start_ns"])

            mid_tick = round(row["mid_close"] / tick)
            desired = {
                "buy": [round((mid_tick - 1 - k) * tick, 10) for k in range(cfg.depth_levels)],
                "sell": [round((mid_tick + 1 + k) * tick, 10) for k in range(cfg.depth_levels)],
            }

            # 1. cancel orders that left the desired price set
            for side in ("buy", "sell"):
                keep = set(desired[side])
                for px in sorted(book.levels[side]):
                    if px in keep:
                        continue
                    for order_id, size in sorted(book.levels[side][px].items()):
                        log.emit(L3Action.CANCEL.value, side, px, size, order_id)
                    book.levels[side][px] = {}

            # 2. add missing levels
            for side in ("buy", "sell"):
                for px in desired[side]:
                    if book.size_at(side, px) == 0:
                        log.emit(L3Action.ADD.value, side, px, cfg.level_size, next_order_id)
                        book.add(side, px, next_order_id, cfg.level_size)
                        next_order_id += 1

            # 3. execute against the resting side
            hit_side = "sell" if row["net_flow"] > 0 else "buy"
            remaining = int(abs(round(row["net_flow"])))
            for px in book.prices(hit_side):
                if remaining <= 0:
                    break
                for order_id, size in sorted(book.levels[hit_side][px].items()):
                    if remaining <= 0:
                        break
                    taken = min(size, remaining)
                    log.emit(L3Action.EXECUTE.value, hit_side, px, taken, order_id)
                    book.reduce(hit_side, px, order_id, taken)
                    remaining -= taken

            # 4. replenish depleted levels
            for side in ("buy", "sell"):
                for px in desired[side]:
                    short = cfg.level_size - book.size_at(side, px)
                    if short > 0:
                        log.emit(L3Action.ADD.value, side, px, short, next_order_id)
                        book.add(side, px, next_order_id, short)
                        next_order_id += 1

            # 5. exercise the replace path on the deepest bid
            if n % replace_every == 0:
                px = desired["buy"][-1]
                order_id = sorted(book.levels["buy"][px])[0]
                new_size = cfg.level_size + 100
                log.emit(L3Action.REPLACE.value, "buy", px, new_size, order_id)
                book.levels["buy"][px] = {order_id: new_size}

            # 6. snapshot
            snap_ts = row["ts_end_ns"] - 1
            for side in ("buy", "sell"):
                for level, px in enumerate(book.prices(side)[: cfg.depth_levels]):
                    snaps.append(
                        {
                            "ts_ns": snap_ts,
                            "symbol": symbol,
                            "side": side,
                            "level": level,
                            "px": px,
                            "size": book.size_at(side, px),
                        }
                    )

        msgs.extend(log.rows)

    l3 = pl.DataFrame(
        msgs,
        schema={
            "ts_ns": pl.Int64,
            "seq": pl.Int64,
            "symbol": pl.Utf8,
            "order_id": pl.Int64,
            "action": pl.Utf8,
            "side": pl.Utf8,
            "px": pl.Float64,
            "size": pl.Int64,
        },
    ).select("ts_ns", "seq", "symbol", "order_id", "action", "side", "px", "size")

    l2 = pl.DataFrame(
        snaps,
        schema={
            "ts_ns": pl.Int64,
            "symbol": pl.Utf8,
            "side": pl.Utf8,
            "level": pl.Int32,
            "px": pl.Float64,
            "size": pl.Int64,
        },
    ).select("ts_ns", "symbol", "side", "level", "px", "size")

    return l3, l2


def _config_payload(cfg: SynthConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["start_date"] = cfg.start_date.isoformat()
    payload["symbols"] = list(cfg.symbols)
    payload["impact_delta"] = dict(cfg.impact_delta) if cfg.impact_delta else None
    payload["impact_Y"] = dict(cfg.impact_Y) if cfg.impact_Y else None
    return payload


def generate_bundle(
    root: Path, cfg: SynthConfig, *, bundle_id: str | None = None
) -> DatasetBundle:
    buckets = generate_buckets(cfg)
    l3, l2 = build_l3_and_l2(buckets, cfg)
    tables = {
        "daily_bars": build_daily_bars(buckets, cfg),
        "l1_taq": build_l1(buckets, cfg),
        "l2_depth": l2,
        "l3_messages": l3,
    }
    return DatasetBundle.write(
        root,
        tables,
        bundle_id=bundle_id or f"synth-seed{cfg.seed}-{cfg.n_days}d",
        provenance="synthetic",
        extra={"ground_truth": ground_truth(cfg), "synth_config": _config_payload(cfg)},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_synth_book.py -v`
Expected: PASS, 8 tests

Confirm the layer contract still holds (`data` must not import `micro`):

Run: `uv run lint-imports`
Expected: contract KEPT

- [ ] **Step 5: Commit**

```bash
git add src/quantic/data/synth.py tests/data/test_synth_book.py
git commit -m "feat(data): generate L3 message stream, L2 snapshots and complete synthetic bundles"
```

---

### Task 8: Catalog, data-request spec, and the CLI

**Files:**
- Create: `src/quantic/data/catalog.py`, `src/quantic/data/request.py`, `src/quantic/cli.py`
- Test: `tests/data/test_catalog.py`, `tests/data/test_request.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `DatasetBundle` (Task 5), `SynthConfig`/`generate_bundle` (Tasks 6–7)
- Produces:
  - `catalog.CatalogEntry` — frozen dataclass: `bundle_id`, `path`, `content_hash`, `provenance`, `symbols: tuple[str, ...]`, `start_date`, `end_date`, `granularities: tuple[str, ...]`
  - `catalog.Catalog(path: Path)` with `register(bundle: DatasetBundle) -> CatalogEntry`, `entries() -> list[CatalogEntry]`, `get(bundle_id: str) -> CatalogEntry`
  - `catalog.DuplicateBundleError(ValueError)`
  - `request.GranularityRequest` — `granularity`, `symbols: tuple[str, ...]`, `start_date`, `end_date`, `options: dict[str, Any]`
  - `request.DataRequest` — `request_id`, `created_utc`, `market`, `notes`, `items: tuple[GranularityRequest, ...]`, with `to_json()` and `write(path)`
  - `request.default_request(symbols, l3_symbols, *, market, l2_start, l2_end, daily_start, daily_end, l3_start, l3_end, depth_levels=10) -> DataRequest` — builds the spec §8.4 shape
  - `cli.app` — Typer app with subcommands `data synth`, `data ingest`, `data request`, `data list`

Registering the same `bundle_id` twice is idempotent when the content hash matches and raises `DuplicateBundleError` when it does not — which is exactly the situation where a stale result would otherwise be attributed to the wrong data.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_catalog.py`:

```python
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
```

Create `tests/data/test_request.py`:

```python
import json

from quantic.data.request import default_request


def test_default_request_matches_spec_section_8_4(tmp_path):
    req = default_request(
        symbols=tuple(f"SYM{i:02d}" for i in range(25)),
        l3_symbols=("SYM00", "SYM01", "SYM02"),
        market="XNAS",
        l2_start="2026-02-02",
        l2_end="2026-02-27",
        daily_start="2025-03-01",
        daily_end="2026-02-27",
        l3_start="2026-02-23",
        l3_end="2026-02-27",
    )
    by_granularity = {item.granularity: item for item in req.items}
    assert set(by_granularity) == {"l1_taq", "l2_depth", "l3_messages", "daily_bars"}
    assert len(by_granularity["l1_taq"].symbols) == 25
    assert by_granularity["l2_depth"].options["depth_levels"] == 10
    assert by_granularity["l3_messages"].symbols == ("SYM00", "SYM01", "SYM02")
    assert by_granularity["daily_bars"].start_date == "2025-03-01"


def test_request_writes_readable_json(tmp_path):
    req = default_request(
        symbols=("AAA",),
        l3_symbols=("AAA",),
        market="XNAS",
        l2_start="2026-02-02",
        l2_end="2026-02-27",
        daily_start="2025-03-01",
        daily_end="2026-02-27",
        l3_start="2026-02-23",
        l3_end="2026-02-27",
    )
    out = tmp_path / "request.json"
    req.write(out)
    payload = json.loads(out.read_text())
    assert payload["market"] == "XNAS"
    assert len(payload["items"]) == 4
```

Create `tests/test_cli.py`:

```python
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
    runner.invoke(app, ["data", "synth", "--out", str(out), "--days", "2", "--buckets", "3"])
    result = runner.invoke(
        app, ["data", "ingest", str(out), "--catalog", str(tmp_path / "c.json")]
    )
    assert result.exit_code == 0, result.output
    assert "content_hash" in result.output


def test_ingest_fails_loudly_on_tampered_bundle(tmp_path):
    out = tmp_path / "synth"
    runner.invoke(app, ["data", "synth", "--out", str(out), "--days", "2", "--buckets", "3"])
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_catalog.py tests/data/test_request.py tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError` for `quantic.data.catalog`

- [ ] **Step 3: Write the implementations**

Create `src/quantic/data/catalog.py`:

```python
"""A local registry mapping bundle IDs to paths and content hashes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from quantic.data.bundle import DatasetBundle

DEFAULT_CATALOG_PATH = Path("data/catalog.json")


class DuplicateBundleError(ValueError):
    """Raised when a bundle_id is reused for different content."""


@dataclass(frozen=True)
class CatalogEntry:
    bundle_id: str
    path: str
    content_hash: str
    provenance: str
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    granularities: tuple[str, ...]


class Catalog:
    def __init__(self, path: Path = DEFAULT_CATALOG_PATH) -> None:
        self.path = Path(path)

    def entries(self) -> list[CatalogEntry]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            CatalogEntry(
                **{**r, "symbols": tuple(r["symbols"]), "granularities": tuple(r["granularities"])}
            )
            for r in raw
        ]

    def get(self, bundle_id: str) -> CatalogEntry:
        for entry in self.entries():
            if entry.bundle_id == bundle_id:
                return entry
        raise KeyError(f"no bundle {bundle_id!r} in catalog {self.path}")

    def register(self, bundle: DatasetBundle) -> CatalogEntry:
        m = bundle.manifest
        entry = CatalogEntry(
            bundle_id=m.bundle_id,
            path=str(Path(bundle.root).resolve()),
            content_hash=m.content_hash,
            provenance=m.provenance,
            symbols=m.symbols,
            start_date=m.start_date,
            end_date=m.end_date,
            granularities=m.granularities,
        )
        existing = self.entries()
        for prior in existing:
            if prior.bundle_id != entry.bundle_id:
                continue
            if prior.content_hash == entry.content_hash:
                return prior
            raise DuplicateBundleError(
                f"bundle_id {entry.bundle_id!r} already registered with content_hash "
                f"{prior.content_hash}, refusing to overwrite with {entry.content_hash}"
            )

        existing.append(entry)
        payload = [
            {**asdict(e), "symbols": list(e.symbols), "granularities": list(e.granularities)}
            for e in sorted(existing, key=lambda e: e.bundle_id)
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return entry
```

Create `src/quantic/data/request.py`:

```python
"""Emit a precise data-request spec for the external market data machine.

The database is not reachable from this project (spec section 3.3), so requests
are produced as files and fulfilled out of band.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GranularityRequest:
    granularity: str
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataRequest:
    request_id: str
    created_utc: str
    market: str
    notes: str
    items: tuple[GranularityRequest, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["items"] = [
            {**asdict(i), "symbols": list(i.symbols)} for i in self.items
        ]
        return json.dumps(payload, indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


def default_request(
    *,
    symbols: tuple[str, ...],
    l3_symbols: tuple[str, ...],
    market: str,
    l2_start: str,
    l2_end: str,
    daily_start: str,
    daily_end: str,
    l3_start: str,
    l3_end: str,
    depth_levels: int = 10,
    request_id: str | None = None,
) -> DataRequest:
    """Build the first-request shape described in spec section 8.4."""
    unknown = set(l3_symbols) - set(symbols)
    if unknown:
        raise ValueError(f"l3_symbols must be a subset of symbols; extra: {sorted(unknown)}")

    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    return DataRequest(
        request_id=request_id or f"req-{now.strftime('%Y%m%dT%H%M%SZ')}",
        created_utc=now.isoformat(),
        market=market,
        notes=(
            "Quantic vertical B. L1+L2 for impact calibration and liquidity metrics; "
            "daily bars for covariance and ADV; L3 on a subset for book reconstruction "
            "cross-validation."
        ),
        items=(
            GranularityRequest("l1_taq", symbols, l2_start, l2_end),
            GranularityRequest(
                "l2_depth", symbols, l2_start, l2_end, {"depth_levels": depth_levels}
            ),
            GranularityRequest("l3_messages", l3_symbols, l3_start, l3_end),
            GranularityRequest("daily_bars", symbols, daily_start, daily_end),
        ),
    )
```

Create `src/quantic/cli.py`:

```python
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
console = Console()


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_catalog.py tests/data/test_request.py tests/test_cli.py -v`
Expected: PASS, 12 tests

Then confirm the whole suite and the CLI entry point:

Run: `uv run pytest && uv run quantic data --help`
Expected: all tests pass; help lists `synth`, `ingest`, `list`, `request`

- [ ] **Step 5: Commit**

```bash
git add src/quantic/data/catalog.py src/quantic/data/request.py src/quantic/cli.py tests/
git commit -m "feat(data): add bundle catalog, data-request spec and CLI"
```

**M0 is complete at this point.** `quantic data synth` produces a valid, schema-conformant, content-hashed bundle carrying known ground truth — the spec's M0 exit criterion.

---
## Milestone M1 — Microstructure Core

### Task 9: Order book reconstruction from the L3 message stream

**Files:**
- Create: `src/quantic/micro/book_reconstruct.py`
- Test: `tests/micro/test_book_reconstruct.py`

**Interfaces:**
- Consumes: `quantic.core.types.{BookSnapshot, PriceLevel, Side}`
- Produces:
  - `BookReconstructionError(RuntimeError)`
  - `BookBuilder(symbol: str, *, strict: bool = True)` with `apply(action: str, order_id: int, side: str, px: float, size: int) -> None`, `snapshot(ts_ns: int, levels: int) -> BookSnapshot`, and property `open_orders: int`
  - `snapshots_at(messages: pl.DataFrame, ts_list: Sequence[int], *, levels: int, strict: bool = True) -> list[BookSnapshot]` — single pass over a **single symbol's** messages, returning one snapshot per requested timestamp in the order given
  - `final_book(messages: pl.DataFrame, *, levels: int, strict: bool = True) -> BookSnapshot`

**Message semantics** (strict by default; violations raise rather than being silently absorbed, per the global constraints):

| Action | Effect |
|---|---|
| `add` | Record a new order. A repeated `order_id` is an error. |
| `cancel` | Reduce the order by `size`; remove it when it reaches zero. Unknown `order_id` is an error. |
| `execute` | Identical bookkeeping to `cancel`; kept distinct because only executions count as traded volume. |
| `replace` | Remove the order entirely, then re-add it at the new `px`/`size` under the same `order_id` (priority is lost, as on a real venue). |

`snapshots_at` consumes messages ordered by `(ts_ns, seq)` and emits the snapshot for target *T* once it encounters the first message with `ts_ns > T` — so a snapshot reflects every message with `ts_ns <= T`.

- [ ] **Step 1: Write the failing test**

Create `tests/micro/test_book_reconstruct.py`:

```python
import polars as pl
import pytest

from quantic.micro.book_reconstruct import (
    BookBuilder,
    BookReconstructionError,
    final_book,
    snapshots_at,
)

COLUMNS = ["ts_ns", "seq", "symbol", "order_id", "action", "side", "px", "size"]


def _messages(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=COLUMNS, orient="row").with_columns(
        pl.col("ts_ns").cast(pl.Int64),
        pl.col("seq").cast(pl.Int64),
        pl.col("order_id").cast(pl.Int64),
        pl.col("px").cast(pl.Float64),
        pl.col("size").cast(pl.Int64),
    )


def test_add_builds_both_sides():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 500)
    b.apply("add", 2, "sell", 100.01, 400)
    snap = b.snapshot(10, levels=5)
    assert snap.best_bid == pytest.approx(99.99)
    assert snap.best_ask == pytest.approx(100.01)
    assert snap.bids[0].size == 500


def test_sizes_aggregate_across_orders_at_one_price():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 500)
    b.apply("add", 2, "buy", 99.99, 300)
    assert b.snapshot(10, levels=5).bids[0].size == 800


def test_partial_execute_reduces_then_full_execute_removes():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 500)
    b.apply("execute", 1, "buy", 99.99, 200)
    assert b.snapshot(10, levels=5).bids[0].size == 300
    b.apply("execute", 1, "buy", 99.99, 300)
    assert b.snapshot(11, levels=5).bids == ()
    assert b.open_orders == 0


def test_cancel_removes_the_order():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "sell", 100.01, 400)
    b.apply("cancel", 1, "sell", 100.01, 400)
    assert b.snapshot(10, levels=5).asks == ()


def test_replace_moves_price_and_size_under_same_id():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 500)
    b.apply("replace", 1, "buy", 99.97, 700)
    snap = b.snapshot(10, levels=5)
    assert [lvl.price for lvl in snap.bids] == pytest.approx([99.97])
    assert snap.bids[0].size == 700


def test_levels_are_returned_in_book_order_and_truncated():
    b = BookBuilder("SYNA")
    for i, px in enumerate([99.95, 99.99, 99.97]):
        b.apply("add", i + 1, "buy", px, 100)
    for i, px in enumerate([100.05, 100.01, 100.03]):
        b.apply("add", i + 10, "sell", px, 100)
    snap = b.snapshot(10, levels=2)
    assert [lvl.price for lvl in snap.bids] == pytest.approx([99.99, 99.97])
    assert [lvl.price for lvl in snap.asks] == pytest.approx([100.01, 100.03])


def test_duplicate_order_id_is_rejected():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 500)
    with pytest.raises(BookReconstructionError, match="duplicate"):
        b.apply("add", 1, "buy", 99.98, 100)


def test_unknown_order_id_is_rejected_when_strict():
    b = BookBuilder("SYNA")
    with pytest.raises(BookReconstructionError, match="unknown order"):
        b.apply("cancel", 42, "buy", 99.99, 100)


def test_unknown_order_id_is_tolerated_when_not_strict():
    b = BookBuilder("SYNA", strict=False)
    b.apply("cancel", 42, "buy", 99.99, 100)
    assert b.snapshot(10, levels=5).bids == ()


def test_overcancel_is_rejected():
    b = BookBuilder("SYNA")
    b.apply("add", 1, "buy", 99.99, 100)
    with pytest.raises(BookReconstructionError, match="exceeds"):
        b.apply("cancel", 1, "buy", 99.99, 500)


def test_unknown_action_is_rejected():
    b = BookBuilder("SYNA")
    with pytest.raises(BookReconstructionError, match="unknown action"):
        b.apply("teleport", 1, "buy", 99.99, 100)


def test_snapshots_at_reflects_state_at_each_timestamp():
    msgs = _messages(
        [
            (100, 0, "SYNA", 1, "add", "buy", 99.99, 500),
            (100, 1, "SYNA", 2, "add", "sell", 100.01, 400),
            (200, 2, "SYNA", 1, "execute", "buy", 99.99, 200),
            (300, 3, "SYNA", 3, "add", "buy", 99.98, 900),
        ]
    )
    snaps = snapshots_at(msgs, [150, 250, 350], levels=5)
    assert [s.ts_ns for s in snaps] == [150, 250, 350]
    assert snaps[0].bids[0].size == 500
    assert snaps[1].bids[0].size == 300
    assert len(snaps[2].bids) == 2


def test_snapshots_at_requires_a_single_symbol():
    msgs = _messages(
        [
            (100, 0, "SYNA", 1, "add", "buy", 99.99, 500),
            (100, 1, "SYNB", 2, "add", "buy", 99.99, 500),
        ]
    )
    with pytest.raises(ValueError, match="single symbol"):
        snapshots_at(msgs, [150], levels=5)


def test_final_book_applies_every_message():
    msgs = _messages(
        [
            (100, 0, "SYNA", 1, "add", "buy", 99.99, 500),
            (200, 1, "SYNA", 1, "cancel", "buy", 99.99, 500),
        ]
    )
    assert final_book(msgs, levels=5).bids == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/micro/test_book_reconstruct.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.micro.book_reconstruct'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/micro/book_reconstruct.py`:

```python
"""Reconstruct order book state from an L3 message stream.

Deliberately independent of ``quantic.data.synth``'s internal book: comparing
the two is the correctness gate in Task 10 and spec section 10.3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from quantic.core.types import BookSnapshot, PriceLevel

_VALID_ACTIONS = frozenset({"add", "cancel", "execute", "replace"})
_VALID_SIDES = frozenset({"buy", "sell"})


class BookReconstructionError(RuntimeError):
    """Raised when a message cannot be applied to the current book state."""


@dataclass(slots=True)
class _Order:
    side: str
    px: float
    size: int


class BookBuilder:
    def __init__(self, symbol: str, *, strict: bool = True) -> None:
        self.symbol = symbol
        self.strict = strict
        self._orders: dict[int, _Order] = {}
        self._levels: dict[str, dict[float, int]] = {"buy": {}, "sell": {}}

    @property
    def open_orders(self) -> int:
        return len(self._orders)

    def _add_size(self, side: str, px: float, size: int) -> None:
        level = self._levels[side]
        level[px] = level.get(px, 0) + size
        if level[px] <= 0:
            del level[px]

    def _insert(self, order_id: int, side: str, px: float, size: int) -> None:
        if side not in _VALID_SIDES:
            raise BookReconstructionError(f"unknown side {side!r}")
        if order_id in self._orders:
            raise BookReconstructionError(f"duplicate order_id {order_id}")
        self._orders[order_id] = _Order(side, px, size)
        self._add_size(side, px, size)

    def _remove(self, order_id: int) -> _Order:
        order = self._orders.pop(order_id)
        self._add_size(order.side, order.px, -order.size)
        return order

    def apply(self, action: str, order_id: int, side: str, px: float, size: int) -> None:
        if action not in _VALID_ACTIONS:
            raise BookReconstructionError(f"unknown action {action!r}")

        if action == "add":
            self._insert(order_id, side, px, size)
            return

        order = self._orders.get(order_id)
        if order is None:
            if self.strict:
                raise BookReconstructionError(
                    f"unknown order {order_id} for action {action!r} on {self.symbol}"
                )
            return

        if action == "replace":
            self._remove(order_id)
            self._insert(order_id, side, px, size)
            return

        # cancel and execute share bookkeeping; only execute is traded volume.
        if size > order.size:
            raise BookReconstructionError(
                f"{action} size {size} exceeds resting size {order.size} for order {order_id}"
            )
        if size == order.size:
            self._remove(order_id)
        else:
            order.size -= size
            self._add_size(order.side, order.px, -size)

    def snapshot(self, ts_ns: int, levels: int) -> BookSnapshot:
        bids = sorted(self._levels["buy"].items(), key=lambda kv: -kv[0])[:levels]
        asks = sorted(self._levels["sell"].items())[:levels]
        return BookSnapshot(
            ts_ns=ts_ns,
            symbol=self.symbol,
            bids=tuple(PriceLevel(px, size) for px, size in bids),
            asks=tuple(PriceLevel(px, size) for px, size in asks),
        )


def _single_symbol(messages: pl.DataFrame) -> str:
    symbols = messages["symbol"].unique().to_list()
    if len(symbols) != 1:
        raise ValueError(f"expected messages for a single symbol, got {sorted(symbols)}")
    return symbols[0]


def snapshots_at(
    messages: pl.DataFrame,
    ts_list: Sequence[int],
    *,
    levels: int,
    strict: bool = True,
) -> list[BookSnapshot]:
    """Return one snapshot per requested timestamp, in the order requested."""
    symbol = _single_symbol(messages)
    builder = BookBuilder(symbol, strict=strict)
    ordered = sorted(range(len(ts_list)), key=lambda i: ts_list[i])
    out: dict[int, BookSnapshot] = {}
    cursor = 0

    for row in messages.sort(["ts_ns", "seq"]).iter_rows(named=True):
        while cursor < len(ordered) and row["ts_ns"] > ts_list[ordered[cursor]]:
            idx = ordered[cursor]
            out[idx] = builder.snapshot(ts_list[idx], levels)
            cursor += 1
        builder.apply(row["action"], row["order_id"], row["side"], row["px"], row["size"])

    while cursor < len(ordered):
        idx = ordered[cursor]
        out[idx] = builder.snapshot(ts_list[idx], levels)
        cursor += 1

    return [out[i] for i in range(len(ts_list))]


def final_book(messages: pl.DataFrame, *, levels: int, strict: bool = True) -> BookSnapshot:
    symbol = _single_symbol(messages)
    builder = BookBuilder(symbol, strict=strict)
    last_ts = 0
    for row in messages.sort(["ts_ns", "seq"]).iter_rows(named=True):
        builder.apply(row["action"], row["order_id"], row["side"], row["px"], row["size"])
        last_ts = row["ts_ns"]
    return builder.snapshot(last_ts, levels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/micro/test_book_reconstruct.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/micro/book_reconstruct.py tests/micro/test_book_reconstruct.py
git commit -m "feat(micro): reconstruct order book state from L3 messages"
```

---

### Task 10: L3-vs-L2 cross-validation — the M1 correctness gate

**Files:**
- Create: `src/quantic/micro/validation.py`
- Test: `tests/micro/test_reconstruction_gate.py`

**Interfaces:**
- Consumes: `snapshots_at` (Task 9), `DatasetBundle` (Task 5), `generate_bundle`/`SynthConfig` (Tasks 6–7)
- Produces:
  - `ReconstructionMismatch` — frozen dataclass: `symbol: str`, `ts_ns: int`, `side: str`, `level: int`, `field: str`, `expected: float`, `actual: float`
  - `compare_to_l2(l3: pl.DataFrame, l2: pl.DataFrame, *, levels: int, symbols: Sequence[str] | None = None, price_tol: float = 1e-9) -> list[ReconstructionMismatch]` — empty list means perfect agreement

This is a reusable report, not only a test: run it against the real bundle when it arrives, and any mismatch localises the feed discrepancy to a symbol, timestamp, side, and level.

- [ ] **Step 1: Write the failing test**

Create `tests/micro/test_reconstruction_gate.py`:

```python
import polars as pl
import pytest

from quantic.data.synth import SynthConfig, generate_bundle
from quantic.micro.validation import compare_to_l2

CFG = SynthConfig(
    symbols=("SYNA", "SYNB"), n_days=3, buckets_per_day=6, seed=11, depth_levels=6
)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return generate_bundle(tmp_path_factory.mktemp("gate") / "synth", CFG)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/micro/test_reconstruction_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.micro.validation'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/micro/validation.py`:

```python
"""Cross-validate L3 replay against independently sourced L2 snapshots.

Spec section 10.3. L2 and L3 are distinct feed products, so agreement between a
replayed book and a published snapshot is genuine evidence of correctness
rather than a tautology.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from quantic.core.types import BookSnapshot
from quantic.micro.book_reconstruct import snapshots_at


@dataclass(frozen=True)
class ReconstructionMismatch:
    symbol: str
    ts_ns: int
    side: str
    level: int
    field: str
    expected: float
    actual: float


def _observed_levels(snap: BookSnapshot, side: str) -> list[tuple[float, int]]:
    levels = snap.bids if side == "buy" else snap.asks
    return [(lvl.price, lvl.size) for lvl in levels]


def compare_to_l2(
    l3: pl.DataFrame,
    l2: pl.DataFrame,
    *,
    levels: int,
    symbols: Sequence[str] | None = None,
    price_tol: float = 1e-9,
) -> list[ReconstructionMismatch]:
    """Replay ``l3`` and report every disagreement with ``l2``.

    An empty result means the replay reproduces every published snapshot
    exactly, to ``price_tol`` on prices and bit-for-bit on sizes.
    """
    wanted = sorted(symbols) if symbols is not None else sorted(l2["symbol"].unique().to_list())
    mismatches: list[ReconstructionMismatch] = []

    for symbol in wanted:
        sym_l2 = l2.filter(pl.col("symbol") == symbol)
        sym_l3 = l3.filter(pl.col("symbol") == symbol)
        if sym_l2.height == 0:
            continue

        ts_list = sorted(sym_l2["ts_ns"].unique().to_list())
        snaps = snapshots_at(sym_l3, ts_list, levels=levels)

        for ts_ns, snap in zip(ts_list, snaps, strict=True):
            at_ts = sym_l2.filter(pl.col("ts_ns") == ts_ns)
            for side in ("buy", "sell"):
                expected = (
                    at_ts.filter(pl.col("side") == side)
                    .sort("level")
                    .select("px", "size")
                    .rows()
                )
                actual = _observed_levels(snap, side)

                if len(expected) != len(actual):
                    mismatches.append(
                        ReconstructionMismatch(
                            symbol, ts_ns, side, -1, "depth", len(expected), len(actual)
                        )
                    )
                    continue

                for level, ((exp_px, exp_sz), (act_px, act_sz)) in enumerate(
                    zip(expected, actual, strict=True)
                ):
                    if abs(exp_px - act_px) > price_tol:
                        mismatches.append(
                            ReconstructionMismatch(
                                symbol, ts_ns, side, level, "px", exp_px, act_px
                            )
                        )
                    if exp_sz != act_sz:
                        mismatches.append(
                            ReconstructionMismatch(
                                symbol, ts_ns, side, level, "size", exp_sz, act_sz
                            )
                        )

    return mismatches
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/micro/test_reconstruction_gate.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/micro/validation.py tests/micro/test_reconstruction_gate.py
git commit -m "feat(micro): add L3-vs-L2 reconstruction cross-validation gate"
```

---

### Task 11: Liquidity metrics

**Files:**
- Create: `src/quantic/micro/liquidity.py`
- Test: `tests/micro/test_liquidity.py`

**Interfaces:**
- Consumes: `quantic.core.types.BookSnapshot`
- Produces:
  - `quoted_spread(book) -> float`, `relative_spread(book) -> float`
  - `depth_shares(book, levels) -> tuple[int, int]`, `depth_notional(book, levels) -> tuple[float, float]`
  - `bucket_volume(l3, *, bucket_ns) -> pl.DataFrame` with columns `symbol, bucket_id, volume`
  - `signed_order_flow(l3, *, bucket_ns) -> pl.DataFrame` with columns `symbol, bucket_id, volume, signed_flow, participation`
  - `adv(daily, *, window=20) -> pl.DataFrame` with columns `symbol, date, adv`
  - `resiliency_halflife(elapsed_ns, spreads) -> float`
  - `InsufficientDataError(ValueError)`

**Sign convention, fixed here and relied on by calibration in Task 16:** an `execute` resting on the **sell** side means a buyer lifted the offer, so it is **positive** flow; an execute on the **buy** side is negative. This is exactly the convention the synthetic generator uses (`hit_side = "sell"` when `net_flow > 0`), so calibration recovers the injected parameters rather than their mirror image.

- [ ] **Step 1: Write the failing test**

Create `tests/micro/test_liquidity.py`:

```python
import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantic.core.types import BookSnapshot, PriceLevel
from quantic.micro.liquidity import (
    InsufficientDataError,
    adv,
    bucket_volume,
    depth_notional,
    depth_shares,
    quoted_spread,
    relative_spread,
    resiliency_halflife,
    signed_order_flow,
)


def _book() -> BookSnapshot:
    return BookSnapshot(
        ts_ns=1,
        symbol="SYNA",
        bids=(PriceLevel(99.99, 500), PriceLevel(99.98, 300)),
        asks=(PriceLevel(100.01, 400), PriceLevel(100.02, 600)),
    )


def test_quoted_and_relative_spread():
    assert quoted_spread(_book()) == pytest.approx(0.02)
    assert relative_spread(_book()) == pytest.approx(0.02 / 100.00)


def test_depth_shares_and_notional():
    assert depth_shares(_book(), levels=2) == (800, 1000)
    bid_notional, ask_notional = depth_notional(_book(), levels=2)
    assert bid_notional == pytest.approx(99.99 * 500 + 99.98 * 300)
    assert ask_notional == pytest.approx(100.01 * 400 + 100.02 * 600)


def test_depth_truncates_to_requested_levels():
    assert depth_shares(_book(), levels=1) == (500, 400)


def _l3() -> pl.DataFrame:
    rows = [
        # bucket 0: buyer lifts 300 on the offer, seller hits 100 on the bid -> +200
        (0, 0, "SYNA", 1, "execute", "sell", 100.01, 300),
        (1, 1, "SYNA", 2, "execute", "buy", 99.99, 100),
        (2, 2, "SYNA", 3, "add", "buy", 99.98, 900),  # not traded volume
        # bucket 1: seller hits 400 on the bid -> -400
        (1_000, 3, "SYNA", 4, "execute", "buy", 99.97, 400),
    ]
    return pl.DataFrame(
        rows,
        schema=["ts_ns", "seq", "symbol", "order_id", "action", "side", "px", "size"],
        orient="row",
    ).with_columns(
        pl.col("ts_ns").cast(pl.Int64),
        pl.col("seq").cast(pl.Int64),
        pl.col("order_id").cast(pl.Int64),
        pl.col("px").cast(pl.Float64),
        pl.col("size").cast(pl.Int64),
    )


def test_bucket_volume_counts_only_executions():
    out = bucket_volume(_l3(), bucket_ns=1_000).sort("bucket_id")
    assert out["volume"].to_list() == [400, 400]


def test_signed_order_flow_uses_the_documented_convention():
    out = signed_order_flow(_l3(), bucket_ns=1_000).sort("bucket_id")
    assert out["signed_flow"].to_list() == [200, -400]
    assert out["participation"].to_list() == pytest.approx([0.5, -1.0])


def test_adv_is_a_trailing_mean():
    daily = pl.DataFrame(
        {
            "date": [dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 7)],
            "symbol": ["SYNA"] * 3,
            "volume": [100, 200, 300],
        }
    )
    out = adv(daily, window=2).sort("date")
    assert out["adv"].to_list() == pytest.approx([100.0, 150.0, 250.0])


def test_resiliency_halflife_recovers_a_known_decay():
    tau = 5.0e9  # 5 seconds in nanoseconds
    elapsed = np.linspace(0.0, 4 * tau, 60)
    spreads = 0.01 + 0.03 * np.exp(-elapsed / tau)
    assert resiliency_halflife(elapsed, spreads) == pytest.approx(tau * np.log(2), rel=0.02)


def test_resiliency_halflife_needs_enough_points():
    with pytest.raises(InsufficientDataError):
        resiliency_halflife(np.array([0.0, 1.0]), np.array([0.02, 0.01]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/micro/test_liquidity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.micro.liquidity'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/micro/liquidity.py`:

```python
"""Liquidity metrics derived from books, message streams and daily bars."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from quantic.core.types import BookSnapshot, EmptyBookError


class InsufficientDataError(ValueError):
    """Raised when an estimator has too few observations to be meaningful."""


def quoted_spread(book: BookSnapshot) -> float:
    spread = book.spread
    if spread is None:
        raise EmptyBookError(f"one-sided book for {book.symbol} at ts_ns={book.ts_ns}")
    return spread


def relative_spread(book: BookSnapshot) -> float:
    mid = book.mid
    if mid is None:
        raise EmptyBookError(f"one-sided book for {book.symbol} at ts_ns={book.ts_ns}")
    return quoted_spread(book) / mid


def depth_shares(book: BookSnapshot, levels: int) -> tuple[int, int]:
    return (
        sum(lvl.size for lvl in book.bids[:levels]),
        sum(lvl.size for lvl in book.asks[:levels]),
    )


def depth_notional(book: BookSnapshot, levels: int) -> tuple[float, float]:
    return (
        sum(lvl.price * lvl.size for lvl in book.bids[:levels]),
        sum(lvl.price * lvl.size for lvl in book.asks[:levels]),
    )


def _executions(l3: pl.DataFrame, bucket_ns: int) -> pl.DataFrame:
    return l3.filter(pl.col("action") == "execute").with_columns(
        (pl.col("ts_ns") // bucket_ns).alias("bucket_id")
    )


def bucket_volume(l3: pl.DataFrame, *, bucket_ns: int) -> pl.DataFrame:
    return (
        _executions(l3, bucket_ns)
        .group_by(["symbol", "bucket_id"])
        .agg(pl.col("size").sum().alias("volume"))
        .sort(["symbol", "bucket_id"])
    )


def signed_order_flow(l3: pl.DataFrame, *, bucket_ns: int) -> pl.DataFrame:
    """Aggregate executions into signed flow per bucket.

    An execution resting on the sell side means a buyer lifted the offer and is
    counted positive; one resting on the buy side is counted negative. The
    synthetic generator uses the same convention, so calibration recovers the
    injected parameters rather than their mirror image.
    """
    return (
        _executions(l3, bucket_ns)
        .with_columns(
            pl.when(pl.col("side") == "sell")
            .then(pl.col("size"))
            .otherwise(-pl.col("size"))
            .alias("signed")
        )
        .group_by(["symbol", "bucket_id"])
        .agg(
            pl.col("size").sum().alias("volume"),
            pl.col("signed").sum().alias("signed_flow"),
        )
        .with_columns((pl.col("signed_flow") / pl.col("volume")).alias("participation"))
        .sort(["symbol", "bucket_id"])
    )


def adv(daily: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    return (
        daily.sort(["symbol", "date"])
        .with_columns(
            pl.col("volume")
            .cast(pl.Float64)
            .rolling_mean(window_size=window, min_samples=1)
            .over("symbol")
            .alias("adv")
        )
        .select("symbol", "date", "adv")
    )


def _decay(t: np.ndarray, s_inf: float, amplitude: float, tau: float) -> np.ndarray:
    return s_inf + amplitude * np.exp(-t / tau)


def resiliency_halflife(elapsed_ns: np.ndarray, spreads: np.ndarray) -> float:
    """Half-life, in nanoseconds, of spread decay back towards its floor."""
    elapsed_ns = np.asarray(elapsed_ns, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    if elapsed_ns.size < 5:
        raise InsufficientDataError(
            f"need at least 5 observations to fit a decay, got {elapsed_ns.size}"
        )

    span = max(elapsed_ns.max() - elapsed_ns.min(), 1.0)
    guess = (float(spreads.min()), float(spreads.max() - spreads.min()), span / 4.0)
    try:
        params, _ = curve_fit(
            _decay, elapsed_ns, spreads, p0=guess, maxfev=10_000,
            bounds=([-np.inf, 0.0, 1e-9], [np.inf, np.inf, np.inf]),
        )
    except RuntimeError as exc:
        raise InsufficientDataError(f"spread decay fit did not converge: {exc}") from exc

    return float(params[2] * np.log(2.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/micro/test_liquidity.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/micro/liquidity.py tests/micro/test_liquidity.py
git commit -m "feat(micro): add spread, depth, bucket flow, ADV and resiliency metrics"
```

---

### Task 12: Cross-asset covariance with Ledoit–Wolf shrinkage

**Files:**
- Create: `src/quantic/micro/covariance.py`
- Test: `tests/micro/test_covariance.py`

**Interfaces:**
- Consumes: daily bars (`date`, `symbol`, `close`)
- Produces:
  - `CovarianceEstimate` — frozen dataclass: `symbols: tuple[str, ...]`, `matrix: np.ndarray` (annualised), `n_observations: int`, `shrinkage: float`; method `to_frame() -> pl.DataFrame`
  - `log_returns(daily: pl.DataFrame) -> pl.DataFrame` — wide, one column per symbol, sorted by date, first row dropped
  - `estimate_covariance(daily, *, method="ledoit_wolf", trading_days=252) -> CovarianceEstimate`
  - `InsufficientHistoryError(ValueError)`

Shrinkage matters here because the benchmark's T3 tier uses 30 assets against roughly 20 days of history, where the sample covariance is singular and would make the risk term meaningless.

- [ ] **Step 1: Write the failing test**

Create `tests/micro/test_covariance.py`:

```python
import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantic.micro.covariance import (
    InsufficientHistoryError,
    estimate_covariance,
    log_returns,
)


def _daily(n_days: int = 60, n_symbols: int = 4, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = [f"S{i}" for i in range(n_symbols)]
    dates = [dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(n_days)]
    rows = []
    for s_idx, symbol in enumerate(symbols):
        price = 100.0
        for d in dates:
            price *= float(np.exp(rng.normal(0.0, 0.01 * (1 + s_idx * 0.2))))
            rows.append({"date": d, "symbol": symbol, "close": price})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_log_returns_shape_and_columns():
    r = log_returns(_daily(n_days=10, n_symbols=3))
    assert r.height == 9  # first row dropped
    assert r.columns == ["date", "S0", "S1", "S2"]


def test_estimate_is_symmetric_and_positive_semidefinite():
    est = estimate_covariance(_daily())
    assert est.symbols == ("S0", "S1", "S2", "S3")
    assert np.allclose(est.matrix, est.matrix.T)
    assert np.linalg.eigvalsh(est.matrix).min() > -1e-12


def test_shrinkage_is_a_valid_intensity():
    est = estimate_covariance(_daily())
    assert 0.0 <= est.shrinkage <= 1.0


def test_annualisation_scales_the_matrix():
    daily = _daily()
    a = estimate_covariance(daily, trading_days=252)
    b = estimate_covariance(daily, trading_days=1)
    assert np.allclose(a.matrix, b.matrix * 252)


def test_shrinkage_conditions_the_matrix_in_the_short_sample_regime():
    """30 assets on 20 days: the sample estimator is singular, shrinkage is not."""
    daily = _daily(n_days=21, n_symbols=30, seed=3)
    lw = estimate_covariance(daily, method="ledoit_wolf")
    sample = estimate_covariance(daily, method="sample")
    assert np.linalg.cond(lw.matrix) < np.linalg.cond(sample.matrix)
    assert np.linalg.eigvalsh(lw.matrix).min() > 0


def test_to_frame_is_labelled():
    frame = estimate_covariance(_daily(n_symbols=2)).to_frame()
    assert frame.columns == ["symbol", "S0", "S1"]
    assert frame["symbol"].to_list() == ["S0", "S1"]


def test_short_history_is_rejected():
    with pytest.raises(InsufficientHistoryError, match="at least"):
        estimate_covariance(_daily(n_days=2, n_symbols=3))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        estimate_covariance(_daily(), method="wishful")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/micro/test_covariance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.micro.covariance'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/micro/covariance.py`:

```python
"""Cross-asset covariance estimation from daily bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf

MIN_OBSERVATIONS = 5


class InsufficientHistoryError(ValueError):
    """Raised when there are too few return observations to estimate covariance."""


@dataclass(frozen=True)
class CovarianceEstimate:
    symbols: tuple[str, ...]
    matrix: np.ndarray
    n_observations: int
    shrinkage: float

    def to_frame(self) -> pl.DataFrame:
        data = {"symbol": list(self.symbols)}
        for j, symbol in enumerate(self.symbols):
            data[symbol] = self.matrix[:, j].tolist()
        return pl.DataFrame(data)


def log_returns(daily: pl.DataFrame) -> pl.DataFrame:
    wide = (
        daily.select("date", "symbol", "close")
        .sort(["date", "symbol"])
        .pivot(on="symbol", index="date", values="close")
        .sort("date")
    )
    symbols = [c for c in wide.columns if c != "date"]
    return wide.with_columns(
        [(pl.col(s) / pl.col(s).shift(1)).log().alias(s) for s in symbols]
    ).drop_nulls()


def estimate_covariance(
    daily: pl.DataFrame,
    *,
    method: str = "ledoit_wolf",
    trading_days: int = 252,
) -> CovarianceEstimate:
    if method not in {"ledoit_wolf", "sample"}:
        raise ValueError(f"unknown method {method!r}; expected 'ledoit_wolf' or 'sample'")

    returns = log_returns(daily)
    symbols = tuple(c for c in returns.columns if c != "date")
    values = returns.select(symbols).to_numpy()

    if values.shape[0] < MIN_OBSERVATIONS:
        raise InsufficientHistoryError(
            f"need at least {MIN_OBSERVATIONS} return observations, got {values.shape[0]}"
        )

    if method == "ledoit_wolf":
        estimator = LedoitWolf().fit(values)
        cov = np.asarray(estimator.covariance_)
        shrinkage = float(estimator.shrinkage_)
    else:
        cov = np.cov(values, rowvar=False, ddof=1)
        shrinkage = 0.0

    return CovarianceEstimate(
        symbols=symbols,
        matrix=cov * trading_days,
        n_observations=values.shape[0],
        shrinkage=shrinkage,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/micro/test_covariance.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/micro/covariance.py tests/micro/test_covariance.py
git commit -m "feat(micro): add Ledoit-Wolf shrinkage covariance estimation"
```

---
### Task 13: Impact model protocol and the Almgren–Chriss baseline

**Files:**
- Create: `src/quantic/micro/impact/base.py`, `src/quantic/micro/impact/almgren_chriss.py`
- Test: `tests/micro/test_impact_base.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `base.ImpactParams` — frozen dataclass: `symbol: str`, `sigma: float` (volatility over the impact horizon), `bucket_volume: float` (shares), `price: float` (reference mid)
  - `base.ImpactModel` — `Protocol` with `name: str` and methods `price_impact(q: float, p: ImpactParams) -> float`, `temporary_cost(q: float, p: ImpactParams) -> float`, `permanent_impact(q: float, p: ImpactParams) -> float`
  - `base.currency_cost(model: ImpactModel, q: float, p: ImpactParams) -> float` — `temporary_cost * |q| * price`
  - `base.cost_in_bps(model: ImpactModel, q: float, p: ImpactParams) -> float` — `temporary_cost * 10_000`
  - `almgren_chriss.AlmgrenChriss(eta: float, gamma: float = 0.0)` — the linear (`delta = 1`) special case

`AlmgrenChriss.price_impact` is `eta * sigma * (|q| / bucket_volume)`: scaling by `sigma` makes it the exact `delta = 1` member of the power-law family in Task 14, which is what lets dial D1 be a clean one-parameter ablation rather than a swap between incomparable models.

- [ ] **Step 1: Write the failing test**

Create `tests/micro/test_impact_base.py`:

```python
import pytest

from quantic.micro.impact.almgren_chriss import AlmgrenChriss
from quantic.micro.impact.base import ImpactParams, cost_in_bps, currency_cost

P = ImpactParams(symbol="SYNA", sigma=0.005, bucket_volume=400_000.0, price=100.0)


def test_zero_quantity_has_zero_impact_and_cost():
    m = AlmgrenChriss(eta=0.8)
    assert m.price_impact(0.0, P) == 0.0
    assert m.temporary_cost(0.0, P) == 0.0


def test_impact_is_linear_in_quantity():
    m = AlmgrenChriss(eta=0.8)
    assert m.price_impact(40_000.0, P) == pytest.approx(2 * m.price_impact(20_000.0, P))


def test_impact_scales_with_sigma_and_participation():
    m = AlmgrenChriss(eta=0.8)
    assert m.price_impact(40_000.0, P) == pytest.approx(0.8 * 0.005 * 0.1)


def test_linear_model_obeys_the_one_half_cost_rule():
    """Integrating linear impact over the executed quantity halves it."""
    m = AlmgrenChriss(eta=0.8)
    assert m.temporary_cost(40_000.0, P) == pytest.approx(m.price_impact(40_000.0, P) / 2.0)


def test_impact_is_sign_insensitive():
    m = AlmgrenChriss(eta=0.8)
    assert m.price_impact(-40_000.0, P) == pytest.approx(m.price_impact(40_000.0, P))


def test_permanent_impact_uses_gamma():
    m = AlmgrenChriss(eta=0.8, gamma=0.3)
    assert m.permanent_impact(40_000.0, P) == pytest.approx(0.3 * 0.005 * 0.1)
    assert AlmgrenChriss(eta=0.8).permanent_impact(40_000.0, P) == 0.0


def test_currency_cost_and_bps_are_consistent():
    m = AlmgrenChriss(eta=0.8)
    q = 40_000.0
    assert currency_cost(m, q, P) == pytest.approx(m.temporary_cost(q, P) * q * P.price)
    assert cost_in_bps(m, q, P) == pytest.approx(m.temporary_cost(q, P) * 10_000)


def test_model_satisfies_the_protocol():
    from quantic.micro.impact.base import ImpactModel

    assert isinstance(AlmgrenChriss(eta=0.8), ImpactModel)


def test_negative_bucket_volume_is_rejected():
    with pytest.raises(ValueError, match="bucket_volume"):
        ImpactParams(symbol="SYNA", sigma=0.005, bucket_volume=0.0, price=100.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/micro/test_impact_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.micro.impact.base'`

- [ ] **Step 3: Write the implementations**

Create `src/quantic/micro/impact/base.py`:

```python
"""Impact model protocol and unit conventions.

Units, fixed once and relied on everywhere:

* ``price_impact`` returns a **fractional** mid displacement, always
  non-negative; the caller applies the sign.
* ``temporary_cost`` returns **fractional cost**: currency cost divided by
  ``|q| * price``.
* Integrating a power-law impact over the executed quantity gives
  ``temporary_cost = price_impact / (delta + 1)`` — the familiar 2/3 rule at
  ``delta = 0.5`` and 1/2 for linear impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ImpactParams:
    symbol: str
    sigma: float
    bucket_volume: float
    price: float

    def __post_init__(self) -> None:
        if self.bucket_volume <= 0:
            raise ValueError(f"bucket_volume must be positive, got {self.bucket_volume}")
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")


@runtime_checkable
class ImpactModel(Protocol):
    name: str

    def price_impact(self, q: float, p: ImpactParams) -> float: ...

    def temporary_cost(self, q: float, p: ImpactParams) -> float: ...

    def permanent_impact(self, q: float, p: ImpactParams) -> float: ...


def currency_cost(model: ImpactModel, q: float, p: ImpactParams) -> float:
    return model.temporary_cost(q, p) * abs(q) * p.price


def cost_in_bps(model: ImpactModel, q: float, p: ImpactParams) -> float:
    return model.temporary_cost(q, p) * 10_000.0
```

Create `src/quantic/micro/impact/almgren_chriss.py`:

```python
"""Linear temporary impact: the delta = 1 member of the power-law family."""

from __future__ import annotations

from dataclasses import dataclass, field

from quantic.micro.impact.base import ImpactParams


@dataclass(frozen=True)
class AlmgrenChriss:
    eta: float
    gamma: float = 0.0
    name: str = field(default="almgren_chriss")

    def price_impact(self, q: float, p: ImpactParams) -> float:
        return self.eta * p.sigma * (abs(q) / p.bucket_volume)

    def temporary_cost(self, q: float, p: ImpactParams) -> float:
        return self.price_impact(q, p) / 2.0

    def permanent_impact(self, q: float, p: ImpactParams) -> float:
        return self.gamma * p.sigma * (abs(q) / p.bucket_volume)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/micro/test_impact_base.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/micro/impact/base.py src/quantic/micro/impact/almgren_chriss.py tests/micro/test_impact_base.py
git commit -m "feat(micro): add impact model protocol and Almgren-Chriss baseline"
```

---

### Task 14: Power-law (square-root) impact — hardness dial D1

**Files:**
- Create: `src/quantic/micro/impact/sqrt_law.py`
- Test: `tests/micro/test_sqrt_law.py`

**Interfaces:**
- Consumes: `base.ImpactParams`, `almgren_chriss.AlmgrenChriss`
- Produces:
  - `PowerLawImpact(delta: float, y_coef: float, gamma: float = 0.0)` — validates `0 < delta <= 1`
  - `sqrt_law(y_coef: float, gamma: float = 0.0) -> PowerLawImpact` — convenience for `delta = 0.5`

This is the concavity that makes the liquidation problem NP-hard (spec §4). The property test below is the one that matters: for `delta < 1` the model must be strictly concave, because that is precisely the feature the whole benchmark rests on.

- [ ] **Step 1: Write the failing test**

Create `tests/micro/test_sqrt_law.py`:

```python
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantic.micro.impact.almgren_chriss import AlmgrenChriss
from quantic.micro.impact.base import ImpactParams
from quantic.micro.impact.sqrt_law import PowerLawImpact, sqrt_law

P = ImpactParams(symbol="SYNA", sigma=0.005, bucket_volume=400_000.0, price=100.0)


def test_sqrt_law_matches_the_closed_form():
    m = sqrt_law(y_coef=0.8)
    assert m.delta == 0.5
    assert m.price_impact(40_000.0, P) == pytest.approx(0.8 * 0.005 * 0.1**0.5)


def test_sqrt_law_obeys_the_two_thirds_cost_rule():
    m = sqrt_law(y_coef=0.8)
    assert m.temporary_cost(40_000.0, P) == pytest.approx(
        m.price_impact(40_000.0, P) * 2.0 / 3.0
    )


def test_delta_one_reproduces_almgren_chriss_exactly():
    power = PowerLawImpact(delta=1.0, y_coef=0.8, gamma=0.3)
    linear = AlmgrenChriss(eta=0.8, gamma=0.3)
    for q in (1_000.0, 40_000.0, 200_000.0):
        assert power.price_impact(q, P) == pytest.approx(linear.price_impact(q, P))
        assert power.temporary_cost(q, P) == pytest.approx(linear.temporary_cost(q, P))
        assert power.permanent_impact(q, P) == pytest.approx(linear.permanent_impact(q, P))


def test_zero_quantity_is_zero():
    assert sqrt_law(y_coef=0.8).price_impact(0.0, P) == 0.0


def test_invalid_delta_is_rejected():
    with pytest.raises(ValueError, match="delta"):
        PowerLawImpact(delta=0.0, y_coef=0.8)
    with pytest.raises(ValueError, match="delta"):
        PowerLawImpact(delta=1.5, y_coef=0.8)


@settings(max_examples=200)
@given(
    delta=st.floats(min_value=0.2, max_value=0.95),
    q=st.floats(min_value=1.0, max_value=1e5),
)
def test_impact_is_strictly_concave_below_delta_one(delta: float, q: float):
    """The source of NP-hardness: doubling size less than doubles impact."""
    m = PowerLawImpact(delta=delta, y_coef=0.8)
    assert m.price_impact(2 * q, P) < 2 * m.price_impact(q, P)


@settings(max_examples=200)
@given(
    delta=st.floats(min_value=0.2, max_value=1.0),
    q=st.floats(min_value=1.0, max_value=1e5),
)
def test_cost_is_impact_divided_by_delta_plus_one(delta: float, q: float):
    m = PowerLawImpact(delta=delta, y_coef=0.8)
    assert m.temporary_cost(q, P) == pytest.approx(m.price_impact(q, P) / (delta + 1.0))


@settings(max_examples=200)
@given(
    delta=st.floats(min_value=0.2, max_value=1.0),
    a=st.floats(min_value=1.0, max_value=1e5),
    b=st.floats(min_value=1.0, max_value=1e5),
)
def test_impact_is_monotone_in_quantity(delta: float, a: float, b: float):
    m = PowerLawImpact(delta=delta, y_coef=0.8)
    lo, hi = sorted((a, b))
    assert m.price_impact(lo, P) <= m.price_impact(hi, P) + 1e-15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/micro/test_sqrt_law.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.micro.impact.sqrt_law'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/micro/impact/sqrt_law.py`:

```python
"""Concave power-law impact — hardness dial D1 (spec section 4).

``delta = 0.5`` is the empirically supported square-root law. Any ``delta < 1``
makes the liquidation objective concave, so minimising it is NP-hard and MIQP
solvers lose their bound. ``delta = 1`` degenerates to Almgren-Chriss, which is
how dial D1 is switched off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantic.micro.impact.base import ImpactParams


@dataclass(frozen=True)
class PowerLawImpact:
    delta: float
    y_coef: float
    gamma: float = 0.0
    name: str = field(default="power_law")

    def __post_init__(self) -> None:
        if not 0.0 < self.delta <= 1.0:
            raise ValueError(f"delta must lie in (0, 1], got {self.delta}")
        if self.y_coef < 0:
            raise ValueError(f"y_coef must be non-negative, got {self.y_coef}")

    def price_impact(self, q: float, p: ImpactParams) -> float:
        return self.y_coef * p.sigma * (abs(q) / p.bucket_volume) ** self.delta

    def temporary_cost(self, q: float, p: ImpactParams) -> float:
        return self.price_impact(q, p) / (self.delta + 1.0)

    def permanent_impact(self, q: float, p: ImpactParams) -> float:
        return self.gamma * p.sigma * (abs(q) / p.bucket_volume)


def sqrt_law(y_coef: float, gamma: float = 0.0) -> PowerLawImpact:
    return PowerLawImpact(delta=0.5, y_coef=y_coef, gamma=gamma)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/micro/test_sqrt_law.py -v`
Expected: PASS, 8 tests (three of them Hypothesis property tests)

- [ ] **Step 5: Commit**

```bash
git add src/quantic/micro/impact/sqrt_law.py tests/micro/test_sqrt_law.py
git commit -m "feat(micro): add concave power-law impact with concavity property tests"
```

---

### Task 15: Empirical depth-walk cost

**Files:**
- Create: `src/quantic/micro/impact/depth_walk.py`
- Test: `tests/micro/test_depth_walk.py`

**Interfaces:**
- Consumes: `quantic.core.types.{BookSnapshot, Side}`
- Produces:
  - `InsufficientDepthError(ValueError)`
  - `WalkResult` — frozen dataclass: `shares: int`, `avg_price: float`, `notional: float`, `levels_consumed: int`, `fractional_cost: float`
  - `walk(book: BookSnapshot, side: Side, shares: int) -> WalkResult`
  - `empirical_impact_curve(book: BookSnapshot, side: Side, quantities: Sequence[int]) -> pl.DataFrame` — columns `shares, participation_of_depth, fractional_cost`

**Why this is a function and not an `ImpactModel`.** It needs a book, not `ImpactParams`, so it deliberately does not implement the protocol. It serves as the *non-parametric reference* against which the fitted parametric models are judged: if `sqrt_law` with calibrated parameters disagrees badly with the depth walk on the same book, the calibration is wrong.

- [ ] **Step 1: Write the failing test**

Create `tests/micro/test_depth_walk.py`:

```python
import pytest

from quantic.core.types import BookSnapshot, PriceLevel, Side
from quantic.micro.impact.depth_walk import (
    InsufficientDepthError,
    empirical_impact_curve,
    walk,
)


def _book() -> BookSnapshot:
    return BookSnapshot(
        ts_ns=1,
        symbol="SYNA",
        bids=(PriceLevel(99.99, 100), PriceLevel(99.98, 200), PriceLevel(99.97, 300)),
        asks=(PriceLevel(100.01, 100), PriceLevel(100.02, 200), PriceLevel(100.03, 300)),
    )


def test_buy_within_the_touch_pays_half_the_spread():
    r = walk(_book(), Side.BUY, 100)
    assert r.avg_price == pytest.approx(100.01)
    assert r.levels_consumed == 1
    assert r.fractional_cost == pytest.approx(0.01 / 100.0)


def test_buy_across_levels_averages_correctly():
    r = walk(_book(), Side.BUY, 250)
    expected = (100.01 * 100 + 100.02 * 150) / 250
    assert r.avg_price == pytest.approx(expected)
    assert r.levels_consumed == 2
    assert r.notional == pytest.approx(expected * 250)
    assert r.fractional_cost == pytest.approx((expected - 100.0) / 100.0)


def test_sell_walks_the_bid_side():
    r = walk(_book(), Side.SELL, 250)
    expected = (99.99 * 100 + 99.98 * 150) / 250
    assert r.avg_price == pytest.approx(expected)
    assert r.fractional_cost == pytest.approx((100.0 - expected) / 100.0)


def test_cost_is_monotone_in_size():
    costs = [walk(_book(), Side.BUY, q).fractional_cost for q in (100, 250, 500)]
    assert costs == sorted(costs)


def test_exhausting_the_book_raises():
    with pytest.raises(InsufficientDepthError, match="600"):
        walk(_book(), Side.BUY, 700)


def test_non_positive_size_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        walk(_book(), Side.BUY, 0)


def test_empirical_curve_shape():
    curve = empirical_impact_curve(_book(), Side.BUY, [100, 250, 600])
    assert curve.columns == ["shares", "participation_of_depth", "fractional_cost"]
    assert curve.height == 3
    assert curve["participation_of_depth"].to_list() == pytest.approx([100 / 600, 250 / 600, 1.0])


def test_empirical_curve_skips_sizes_beyond_depth():
    curve = empirical_impact_curve(_book(), Side.BUY, [100, 700])
    assert curve.height == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/micro/test_depth_walk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.micro.impact.depth_walk'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/micro/impact/depth_walk.py`:

```python
"""Non-parametric execution cost: walk the observed book.

This is the reference the fitted parametric models are checked against. It
takes a book rather than ``ImpactParams``, so it deliberately does not
implement the ``ImpactModel`` protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from quantic.core.types import BookSnapshot, EmptyBookError, Side


class InsufficientDepthError(ValueError):
    """Raised when the visible book cannot absorb the requested size."""


@dataclass(frozen=True, slots=True)
class WalkResult:
    shares: int
    avg_price: float
    notional: float
    levels_consumed: int
    fractional_cost: float


def walk(book: BookSnapshot, side: Side, shares: int) -> WalkResult:
    """Cost of executing ``shares`` immediately against the resting book."""
    if shares <= 0:
        raise ValueError(f"shares must be positive, got {shares}")

    mid = book.mid
    if mid is None:
        raise EmptyBookError(f"one-sided book for {book.symbol} at ts_ns={book.ts_ns}")

    levels = book.asks if side is Side.BUY else book.bids
    available = sum(lvl.size for lvl in levels)
    if available < shares:
        raise InsufficientDepthError(
            f"{book.symbol}: requested {shares} shares but only {available} visible "
            f"on the {side.value} side at ts_ns={book.ts_ns}"
        )

    remaining = shares
    notional = 0.0
    consumed = 0
    for lvl in levels:
        if remaining <= 0:
            break
        taken = min(lvl.size, remaining)
        notional += taken * lvl.price
        remaining -= taken
        consumed += 1

    avg_price = notional / shares
    fractional_cost = (avg_price - mid) / mid if side is Side.BUY else (mid - avg_price) / mid
    return WalkResult(
        shares=shares,
        avg_price=avg_price,
        notional=notional,
        levels_consumed=consumed,
        fractional_cost=fractional_cost,
    )


def empirical_impact_curve(
    book: BookSnapshot, side: Side, quantities: Sequence[int]
) -> pl.DataFrame:
    """Cost curve over ``quantities``; sizes exceeding visible depth are skipped."""
    levels = book.asks if side is Side.BUY else book.bids
    available = sum(lvl.size for lvl in levels)

    rows = []
    for q in quantities:
        if q <= 0 or q > available:
            continue
        result = walk(book, side, q)
        rows.append(
            {
                "shares": q,
                "participation_of_depth": q / available,
                "fractional_cost": result.fractional_cost,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "shares": pl.Int64,
            "participation_of_depth": pl.Float64,
            "fractional_cost": pl.Float64,
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/micro/test_depth_walk.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add src/quantic/micro/impact/depth_walk.py tests/micro/test_depth_walk.py
git commit -m "feat(micro): add non-parametric depth-walk execution cost"
```

---

### Task 16: Impact calibration — the M1 parameter-recovery gate

**Files:**
- Create: `src/quantic/micro/impact/calibrate.py`
- Test: `tests/micro/test_calibrate.py`

**Interfaces:**
- Consumes: `liquidity.signed_order_flow` (Task 11), `covariance.log_returns` (Task 12), `sqrt_law.PowerLawImpact` (Task 14), `DatasetBundle` (Task 5)
- Produces:
  - `CalibrationError(ValueError)`
  - `CalibrationResult` — frozen dataclass: `symbol: str`, `delta: float`, `y_coef: float`, `r_squared: float`, `n_observations: int`, `n_bins: int`; method `to_model(gamma: float = 0.0) -> PowerLawImpact`
  - `observations_from_buckets(l1: pl.DataFrame, l3: pl.DataFrame, *, bucket_ns: int) -> pl.DataFrame` — columns `symbol, bucket_id, mid, volume, signed_flow, participation, impact`
  - `fit_power_law(observations: pl.DataFrame, *, sigma: float, n_bins: int = 20, min_participation: float = 1e-4) -> CalibrationResult`
  - `estimate_bucket_sigma(daily: pl.DataFrame, *, buckets_per_day: int) -> dict[str, float]`
  - `calibrate_bundle(bundle: DatasetBundle, *, bucket_ns: int, sigma: Mapping[str, float] | None = None) -> dict[str, CalibrationResult]`

**Methodology — binned log-log regression.** Per-bucket impact is dominated by diffusion, so a raw regression on individual buckets is hopeless. The standard approach, and the one used here: sort observations by `|participation|`, split into `n_bins` equal-count bins, take the mean of `|participation|` and the mean of *signed* impact (`impact * sign(participation)`) within each bin, and fit `log(mean_signed_impact / sigma) = log(Y) + delta * log(mean_participation)` by ordinary least squares. Averaging signed impact within a bin cancels the diffusion noise; taking absolute impact instead would bias `Y` upward.

**Bucket alignment.** L1 rows stamped exactly on a bucket boundary belong to the *closing* bucket, so L1 is assigned `(ts_ns - 1) // bucket_ns` while L3 executions use `ts_ns // bucket_ns`. Impact for bucket *k* is `mid[k] / mid[k-1] - 1`, and rows whose predecessor bucket is not `k-1` are dropped — that removes overnight returns, which are not intraday impact.

**Where sigma comes from.** `sigma` is supplied, not estimated from the same returns being regressed. `estimate_bucket_sigma` derives it from daily close-to-close volatility divided by the square root of buckets per day — an independent estimate, as a practitioner would use.

- [ ] **Step 1: Write the failing test**

Create `tests/micro/test_calibrate.py`:

```python
import polars as pl
import pytest

from quantic.data.synth import SynthConfig, generate_bundle
from quantic.micro.impact.calibrate import (
    CalibrationError,
    calibrate_bundle,
    estimate_bucket_sigma,
    fit_power_law,
    observations_from_buckets,
)

BUCKET_NS = (23_400 // 13) * 1_000_000_000  # 30-minute buckets

LOW_NOISE = SynthConfig(
    symbols=("SYNA", "SYNB", "SYNC"),
    n_days=40,
    buckets_per_day=13,
    seed=17,
    base_price=1000.0,   # keeps tick rounding far below the impact signal
    noise_frac=0.2,
    depth_levels=4,
)

REALISTIC = SynthConfig(
    symbols=("SYNA", "SYNB"),
    n_days=60,
    buckets_per_day=13,
    seed=23,
    base_price=1000.0,
    noise_frac=1.0,
    depth_levels=4,
)


@pytest.fixture(scope="module")
def low_noise_bundle(tmp_path_factory):
    return generate_bundle(tmp_path_factory.mktemp("cal_low") / "b", LOW_NOISE)


@pytest.fixture(scope="module")
def realistic_bundle(tmp_path_factory):
    return generate_bundle(tmp_path_factory.mktemp("cal_real") / "b", REALISTIC)


def test_observations_have_expected_columns(low_noise_bundle):
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), bucket_ns=BUCKET_NS
    )
    assert obs.columns == [
        "symbol", "bucket_id", "mid", "volume", "signed_flow", "participation", "impact"
    ]
    assert obs.height > 0


def test_overnight_returns_are_excluded(low_noise_bundle):
    """13 buckets per day, 40 days: each day loses its first bucket to the gap."""
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), bucket_ns=BUCKET_NS
    )
    per_symbol = obs.group_by("symbol").len()["len"].unique().to_list()
    assert per_symbol == [LOW_NOISE.n_days * (LOW_NOISE.buckets_per_day - 1)]


def test_recovers_ground_truth_delta_and_y_in_low_noise(low_noise_bundle):
    """THE M1 GATE: calibration recovers the injected impact parameters."""
    gt = low_noise_bundle.manifest.extra["ground_truth"]
    sigma = gt["sigma_bucket"]
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), bucket_ns=BUCKET_NS
    )

    for symbol in LOW_NOISE.symbols:
        result = fit_power_law(obs.filter(pl.col("symbol") == symbol), sigma=sigma)
        assert result.delta == pytest.approx(gt["impact_delta"][symbol], abs=0.05)
        assert result.y_coef == pytest.approx(gt["impact_Y"][symbol], rel=0.30)
        assert result.r_squared > 0.90


def test_recovers_delta_under_realistic_diffusion_noise(realistic_bundle):
    gt = realistic_bundle.manifest.extra["ground_truth"]
    results = calibrate_bundle(
        realistic_bundle, bucket_ns=BUCKET_NS, sigma=dict.fromkeys(
            REALISTIC.symbols, gt["sigma_bucket"]
        )
    )
    for symbol in REALISTIC.symbols:
        assert results[symbol].delta == pytest.approx(gt["impact_delta"][symbol], abs=0.15)


def test_estimate_bucket_sigma_is_in_the_right_ballpark(realistic_bundle):
    """With diffusion-dominated returns, realised vol should track daily_vol."""
    gt = realistic_bundle.manifest.extra["ground_truth"]
    sigma = estimate_bucket_sigma(
        realistic_bundle.daily(), buckets_per_day=REALISTIC.buckets_per_day
    )
    for symbol in REALISTIC.symbols:
        assert sigma[symbol] == pytest.approx(gt["sigma_bucket"], rel=0.5)


def test_calibrate_bundle_estimates_sigma_when_not_supplied(realistic_bundle):
    results = calibrate_bundle(realistic_bundle, bucket_ns=BUCKET_NS)
    assert set(results) == set(REALISTIC.symbols)
    assert all(0.0 < r.delta <= 1.0 for r in results.values())


def test_to_model_round_trips_into_a_usable_impact_model(low_noise_bundle):
    from quantic.micro.impact.base import ImpactParams

    gt = low_noise_bundle.manifest.extra["ground_truth"]
    obs = observations_from_buckets(
        low_noise_bundle.l1(), low_noise_bundle.l3(), bucket_ns=BUCKET_NS
    )
    model = fit_power_law(
        obs.filter(pl.col("symbol") == "SYNA"), sigma=gt["sigma_bucket"]
    ).to_model()
    p = ImpactParams(symbol="SYNA", sigma=gt["sigma_bucket"], bucket_volume=4e5, price=1000.0)
    assert model.price_impact(4e4, p) > 0


def test_too_few_bins_is_rejected():
    obs = pl.DataFrame(
        {
            "symbol": ["SYNA"] * 3,
            "bucket_id": [0, 1, 2],
            "mid": [100.0, 100.1, 100.2],
            "volume": [1000, 1000, 1000],
            "signed_flow": [100, 200, 300],
            "participation": [0.1, 0.2, 0.3],
            "impact": [0.001, 0.002, 0.003],
        }
    )
    with pytest.raises(CalibrationError, match="bins"):
        fit_power_law(obs, sigma=0.005, n_bins=20)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/micro/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantic.micro.impact.calibrate'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/micro/impact/calibrate.py`:

```python
"""Calibrate the power-law impact exponent from bucket-level order flow.

Binned log-log regression (Almgren et al.; Toth et al.): per-bucket impact is
dominated by diffusion, so observations are grouped into equal-count bins by
absolute participation and the *signed* impact is averaged within each bin,
which cancels the noise. Averaging absolute impact instead would bias Y upward.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl

from quantic.data.bundle import DatasetBundle
from quantic.micro.covariance import log_returns
from quantic.micro.impact.sqrt_law import PowerLawImpact
from quantic.micro.liquidity import signed_order_flow

MIN_BINS = 4


class CalibrationError(ValueError):
    """Raised when there is not enough signal to fit an impact law."""


@dataclass(frozen=True)
class CalibrationResult:
    symbol: str
    delta: float
    y_coef: float
    r_squared: float
    n_observations: int
    n_bins: int

    def to_model(self, gamma: float = 0.0) -> PowerLawImpact:
        return PowerLawImpact(delta=self.delta, y_coef=self.y_coef, gamma=gamma)


def observations_from_buckets(
    l1: pl.DataFrame, l3: pl.DataFrame, *, bucket_ns: int
) -> pl.DataFrame:
    """Join bucket-level signed flow to the mid return realised over that bucket."""
    # A quote stamped exactly on a boundary belongs to the closing bucket.
    mids = (
        l1.with_columns(
            ((pl.col("ts_ns") - 1) // bucket_ns).alias("bucket_id"),
            ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
        )
        .sort(["symbol", "bucket_id", "ts_ns"])
        .group_by(["symbol", "bucket_id"])
        .agg(pl.col("mid").last())
        .sort(["symbol", "bucket_id"])
        .with_columns(
            pl.col("mid").shift(1).over("symbol").alias("prev_mid"),
            pl.col("bucket_id").shift(1).over("symbol").alias("prev_bucket"),
        )
        # Drop gaps, which are overnight returns rather than intraday impact.
        .filter(pl.col("bucket_id") - pl.col("prev_bucket") == 1)
        .with_columns((pl.col("mid") / pl.col("prev_mid") - 1.0).alias("impact"))
    )

    flow = signed_order_flow(l3, bucket_ns=bucket_ns)

    return (
        mids.join(flow, on=["symbol", "bucket_id"], how="inner")
        .select(
            "symbol", "bucket_id", "mid", "volume", "signed_flow", "participation", "impact"
        )
        .sort(["symbol", "bucket_id"])
    )


def fit_power_law(
    observations: pl.DataFrame,
    *,
    sigma: float,
    n_bins: int = 20,
    min_participation: float = 1e-4,
) -> CalibrationResult:
    if sigma <= 0:
        raise CalibrationError(f"sigma must be positive, got {sigma}")

    symbols = observations["symbol"].unique().to_list()
    if len(symbols) != 1:
        raise CalibrationError(f"fit one symbol at a time, got {sorted(symbols)}")

    usable = observations.filter(pl.col("participation").abs() >= min_participation)
    if usable.height < n_bins * 2:
        raise CalibrationError(
            f"need at least {n_bins * 2} observations for {n_bins} bins, got {usable.height}"
        )

    abs_f = usable["participation"].abs().to_numpy()
    signed_impact = (
        usable["impact"].to_numpy() * np.sign(usable["participation"].to_numpy())
    )

    order = np.argsort(abs_f)
    groups = [g for g in np.array_split(order, n_bins) if g.size > 0]

    x_vals: list[float] = []
    y_vals: list[float] = []
    for g in groups:
        mean_f = float(abs_f[g].mean())
        mean_impact = float(signed_impact[g].mean())
        if mean_f <= 0 or mean_impact <= 0:
            continue  # a bin whose mean impact is negative carries no power-law signal
        x_vals.append(np.log(mean_f))
        y_vals.append(np.log(mean_impact / sigma))

    if len(x_vals) < MIN_BINS:
        raise CalibrationError(
            f"only {len(x_vals)} usable bins after filtering; need at least {MIN_BINS}"
        )

    x = np.asarray(x_vals)
    y = np.asarray(y_vals)
    slope, intercept = np.polyfit(x, y, 1)

    residuals = y - (slope * x + intercept)
    ss_res = float((residuals**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return CalibrationResult(
        symbol=symbols[0],
        delta=float(slope),
        y_coef=float(np.exp(intercept)),
        r_squared=r_squared,
        n_observations=usable.height,
        n_bins=len(x_vals),
    )


def estimate_bucket_sigma(daily: pl.DataFrame, *, buckets_per_day: int) -> dict[str, float]:
    """Per-bucket volatility from daily close-to-close returns.

    Independent of the returns being regressed, which is what keeps the fitted
    Y coefficient meaningful.
    """
    returns = log_returns(daily)
    out: dict[str, float] = {}
    for symbol in (c for c in returns.columns if c != "date"):
        daily_vol = float(returns[symbol].std(ddof=1))
        out[symbol] = daily_vol / np.sqrt(buckets_per_day)
    return out


def calibrate_bundle(
    bundle: DatasetBundle,
    *,
    bucket_ns: int,
    sigma: Mapping[str, float] | None = None,
    n_bins: int = 20,
) -> dict[str, CalibrationResult]:
    session_ns = 23_400 * 1_000_000_000
    buckets_per_day = max(int(round(session_ns / bucket_ns)), 1)
    sigmas = dict(sigma) if sigma is not None else estimate_bucket_sigma(
        bundle.daily(), buckets_per_day=buckets_per_day
    )

    observations = observations_from_buckets(bundle.l1(), bundle.l3(), bucket_ns=bucket_ns)
    results: dict[str, CalibrationResult] = {}
    for symbol in sorted(observations["symbol"].unique().to_list()):
        if symbol not in sigmas:
            raise CalibrationError(f"no sigma supplied for {symbol}")
        results[symbol] = fit_power_law(
            observations.filter(pl.col("symbol") == symbol),
            sigma=sigmas[symbol],
            n_bins=n_bins,
        )
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/micro/test_calibrate.py -v`
Expected: PASS, 8 tests

Then run the whole suite and the architecture contract one final time:

Run: `uv run pytest && uv run lint-imports && uv run ruff check .`
Expected: all tests pass; contract KEPT; no lint errors

- [ ] **Step 5: Commit**

```bash
git add src/quantic/micro/impact/calibrate.py tests/micro/test_calibrate.py
git commit -m "feat(micro): calibrate power-law impact with binned log-log regression"
```

**M1 is complete at this point.** Both spec exit criteria are met: L3 replay matches L2 snapshots (Task 10), and `delta` is recovered on synthetic data within tolerance (Task 16).

---

## What M2 Picks Up

The next plan starts from `problem/`: `LiquidationProblem`, the four hardness-dial constraint modules, and the instance ladder generator. It consumes exactly three things from this plan — `CalibrationResult.to_model()`, `CovarianceEstimate.matrix`, and `liquidity.adv()` — which is the whole point of the layer contract.
