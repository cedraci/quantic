# Quantic M2a: Problem Model, Hardness Dials D1/D2/D4, and the Instance Ladder

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/quantic/problem/` — the discrete multi-asset liquidation problem, dials D1/D2/D4, and a generator producing provably feasible instances across the four-tier ladder.

**Architecture:** A frozen `Instance` is the boundary type M3 and M4 consume. Constraints are frozen descriptors carrying problem-level feasibility logic; encoding pattern-matches on them later. The generator derives constraint parameters from the tier shape so a feasible witness schedule provably exists, and attaches it. The risk term is a variant point (`RiskSpec`) so M2b's CVaR slots in without reshaping anything.

**Tech Stack:** Python 3.11+, NumPy, Polars, pytest, Hypothesis, ruff, import-linter. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-24-m2a-problem-model-design.md`

## Global Constraints

- **Python `>=3.11`.** Environment managed with `uv`. Tests run as `.venv/Scripts/python.exe -m pytest` (Windows) or `uv run pytest`.
- **Layer boundaries (spec §12), enforced by import-linter:** `problem/` may import `core/` and `micro/`. It must **never** import `encoding/`, `solvers/`, `bench/`, `viz/` or `dashboard/`.
- **Determinism.** Every stochastic function takes an explicit `seed`. Identical seed plus identical inputs produce byte-identical output. No implicit global RNGs.
- **Strictness over silent repair.** Malformed input raises a typed exception. Nothing is silently dropped, clamped or patched.
- **Units are named in the type.** `Schedule` is in **lots**; everything in `micro/` is in **shares**; conversion happens once, at the objective boundary, via `Asset.lot_size`. `MarketParams.covariance` is a log-return covariance at the **bucket** horizon.
- **Line length 100**, ruff rules `E,F,I,UP,B,SIM`. Run `.venv/Scripts/python.exe -m ruff check src tests` before every commit.
- **Test-first.** Write the failing test, watch it fail for the right reason, then implement. A test that passes on first run is testing nothing.

## File Structure

| File | Responsibility |
|---|---|
| `problem/schedule.py` | `Schedule` — the N×T lot grid and its derived quantities |
| `problem/dials.py` | `Dials` — the four hardness switches and their combinations |
| `problem/risk.py` | `RiskSpec` protocol, `VarianceRisk`. M2b's extension point |
| `problem/params.py` | `AssetParams`, `MarketParams` — per-asset market inputs |
| `problem/constraints/base.py` | `Constraint` protocol |
| `problem/constraints/full_liquidation.py` | `FullLiquidation` |
| `problem/constraints/cardinality.py` | `Cardinality` (D2) |
| `problem/constraints/min_participation.py` | `MinParticipation` (D2) |
| `problem/constraints/block_trades.py` | `Block`, `BlockTrades` (D4) |
| `problem/instance.py` | `Instance` and its content hash |
| `problem/feasibility.py` | `FeasibilityReport`, `classify` — the single feasibility answer |
| `problem/objective.py` | `ObjectiveBreakdown`, `evaluate` |
| `problem/liquidation.py` | `TierSpec`, `T0`–`T3`, `LADDER` |
| `problem/generator.py` | `generate`, `generate_ladder`, `InstanceGenerationError` |
| `problem/from_bundle.py` | `market_params_from_bundle` — the only `micro/` consumer |

---

## Task 1: `Schedule`

**Files:**
- Create: `src/quantic/problem/schedule.py`
- Test: `tests/problem/test_schedule.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Schedule(lots: tuple[tuple[int, ...], ...])` with `.n_assets`, `.n_buckets`, `.sold(i) -> int`, `.active_names(t) -> int`, `.as_array() -> np.ndarray`, `.cumulative() -> np.ndarray`, `.cumulative_before() -> np.ndarray`, and classmethod `.from_array(arr) -> Schedule`.

- [ ] **Step 1: Create the package directories**

```bash
mkdir -p src/quantic/problem/constraints tests/problem
touch src/quantic/problem/__init__.py src/quantic/problem/constraints/__init__.py
```

(`src/quantic/problem/__init__.py` and `constraints/__init__.py` already exist and are empty. Leave them empty — the M0/M1 convention is that packages export nothing.)

- [ ] **Step 2: Write the failing test**

Create `tests/problem/test_schedule.py`:

```python
import numpy as np
import pytest

from quantic.problem.schedule import Schedule


def _schedule() -> Schedule:
    # 2 assets x 3 buckets
    return Schedule(lots=((2, 0, 1), (0, 0, 4)))


def test_shape_is_read_from_the_grid():
    s = _schedule()
    assert s.n_assets == 2
    assert s.n_buckets == 3


def test_sold_totals_a_row():
    s = _schedule()
    assert s.sold(0) == 3
    assert s.sold(1) == 4


def test_active_names_counts_nonzero_entries_in_a_bucket():
    s = _schedule()
    assert s.active_names(0) == 1   # only asset 0 trades
    assert s.active_names(1) == 0   # nobody trades
    assert s.active_names(2) == 2   # both trade


def test_cumulative_is_inclusive_of_the_current_bucket():
    s = _schedule()
    assert s.cumulative().tolist() == [[2, 2, 3], [0, 0, 4]]


def test_cumulative_before_is_exclusive_and_starts_at_zero():
    s = _schedule()
    assert s.cumulative_before().tolist() == [[0, 2, 2], [0, 0, 0]]


def test_the_two_cumulative_conventions_differ_by_exactly_the_schedule():
    """The permanent impact term uses the exclusive form.

    Reading the inclusive one would charge us for impact we had not yet
    caused, which is a plausible number and an invisible error.
    """
    s = _schedule()
    assert (s.cumulative() - s.cumulative_before()).tolist() == s.as_array().tolist()


def test_round_trips_through_an_array():
    s = _schedule()
    assert Schedule.from_array(s.as_array()) == s


def test_a_negative_entry_is_rejected():
    """Non-negativity is spec 5.3's constraint; the type enforces it."""
    with pytest.raises(ValueError, match="negative"):
        Schedule(lots=((1, -1),))


def test_a_ragged_grid_is_rejected():
    with pytest.raises(ValueError, match="rectangular|ragged"):
        Schedule(lots=((1, 2), (3,)))


def test_an_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        Schedule(lots=())


def test_schedules_are_hashable_and_compare_by_value():
    assert _schedule() == Schedule(lots=((2, 0, 1), (0, 0, 4)))
    assert len({_schedule(), _schedule()}) == 1


def test_as_array_is_int64_so_cumulative_sums_do_not_overflow():
    assert _schedule().as_array().dtype == np.int64
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_schedule.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.schedule'`

- [ ] **Step 4: Write the implementation**

Create `src/quantic/problem/schedule.py`:

```python
"""The decision variable: how many lots of each asset trade in each bucket.

Spec section 5.1 writes ``x[i,t]`` in **lots**, and this type is the project's
only representation of it. Everything in ``micro/`` is in shares; the
conversion happens exactly once, in ``problem.objective``, via
``Asset.lot_size``. No other module converts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Schedule:
    """An ``N x T`` grid of lots, immutable and hashable.

    A tuple of tuples rather than an array because :class:`Instance` is frozen
    and content-hashed, and the largest tier is 30 x 12 = 360 integers. Use
    :meth:`as_array` for numeric work.

    Non-negativity (spec section 5.3) is enforced here rather than checked
    later, so no consumer has to wonder whether a negative entry is possible.
    """

    lots: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.lots or not self.lots[0]:
            raise ValueError("schedule is empty; it needs at least one asset and one bucket")
        width = len(self.lots[0])
        for i, row in enumerate(self.lots):
            if len(row) != width:
                raise ValueError(
                    f"schedule must be rectangular: row 0 has {width} buckets but row {i} "
                    f"has {len(row)}"
                )
            for t, value in enumerate(row):
                if value < 0:
                    raise ValueError(
                        f"schedule has a negative entry at asset {i}, bucket {t}: {value}. "
                        "Liquidation quantities are non-negative (spec section 5.3)"
                    )

    @classmethod
    def from_array(cls, array: np.ndarray) -> Schedule:
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"schedule array must be 2-D, got shape {arr.shape}")
        return cls(lots=tuple(tuple(int(v) for v in row) for row in arr))

    @property
    def n_assets(self) -> int:
        return len(self.lots)

    @property
    def n_buckets(self) -> int:
        return len(self.lots[0])

    def sold(self, i: int) -> int:
        """Total lots of asset ``i`` traded across every bucket."""
        return sum(self.lots[i])

    def active_names(self, t: int) -> int:
        """How many assets trade a non-zero quantity in bucket ``t``."""
        return sum(1 for row in self.lots if row[t] > 0)

    def as_array(self) -> np.ndarray:
        return np.array(self.lots, dtype=np.int64)

    def cumulative(self) -> np.ndarray:
        """``(N, T)`` cumulative lots **inclusive** of each bucket."""
        return np.cumsum(self.as_array(), axis=1)

    def cumulative_before(self) -> np.ndarray:
        """``(N, T)`` cumulative lots **strictly before** each bucket; column 0 is zero.

        This is the form the permanent impact term needs: at the moment we
        trade in bucket ``t`` we have only caused the displacement from
        everything before ``t``.
        """
        return self.cumulative() - self.as_array()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_schedule.py -q`
Expected: 12 passed

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/schedule.py tests/problem/test_schedule.py
git commit -m "feat(problem): add Schedule, the lot-denominated decision variable"
```

---

## Task 2: `Dials` and `RiskSpec`

**Files:**
- Create: `src/quantic/problem/dials.py`, `src/quantic/problem/risk.py`
- Test: `tests/problem/test_dials.py`, `tests/problem/test_risk.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Dials(concave_impact, discrete_participation, cvar_risk, block_trades)` with `.combinations(include_cvar=False) -> tuple[Dials, ...]` and `.label -> str`; `RiskSpec` protocol with `name: str` and `describe() -> dict[str, Any]`; `VarianceRisk(lam: float)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/problem/test_dials.py`:

```python
import pytest

from quantic.problem.dials import Dials


def test_every_dial_defaults_off():
    d = Dials()
    assert not d.concave_impact
    assert not d.discrete_participation
    assert not d.cvar_risk
    assert not d.block_trades


def test_m2a_enumerates_eight_combinations():
    """D3 is M2b; M2a's gate is 8 combinations, not 16."""
    combos = Dials.combinations()
    assert len(combos) == 8
    assert all(not d.cvar_risk for d in combos)
    assert len(set(combos)) == 8


def test_including_cvar_enumerates_all_sixteen():
    combos = Dials.combinations(include_cvar=True)
    assert len(combos) == 16
    assert len(set(combos)) == 16
    assert sum(d.cvar_risk for d in combos) == 8


def test_combinations_are_deterministically_ordered():
    assert Dials.combinations() == Dials.combinations()


def test_the_all_off_combination_is_present_and_first():
    assert Dials.combinations()[0] == Dials()


def test_label_names_the_active_dials():
    assert Dials().label == "none"
    assert Dials(concave_impact=True).label == "D1"
    assert Dials(concave_impact=True, block_trades=True).label == "D1+D4"
    assert Dials(discrete_participation=True).label == "D2"
    assert Dials(cvar_risk=True).label == "D3"


def test_labels_are_unique_across_all_combinations():
    """Every results row is keyed by this, so a collision would merge two cells."""
    combos = Dials.combinations(include_cvar=True)
    assert len({d.label for d in combos}) == len(combos)


def test_dials_are_hashable():
    assert len({Dials(), Dials()}) == 1
```

Create `tests/problem/test_risk.py`:

```python
import pytest

from quantic.problem.risk import RiskSpec, VarianceRisk


def test_variance_risk_carries_a_risk_aversion():
    assert VarianceRisk(lam=1e-6).lam == 1e-6


def test_variance_risk_names_itself():
    assert VarianceRisk(lam=1.0).name == "variance"


def test_variance_risk_satisfies_the_protocol():
    """M2b adds CVaRRisk alongside; Instance.risk is typed to the protocol."""
    assert isinstance(VarianceRisk(lam=1.0), RiskSpec)


def test_a_negative_risk_aversion_is_rejected():
    """Negative lam rewards risk, which makes the objective unbounded below."""
    with pytest.raises(ValueError, match="lam"):
        VarianceRisk(lam=-1.0)


def test_zero_risk_aversion_is_allowed():
    """Switching risk off entirely is a legitimate ablation."""
    assert VarianceRisk(lam=0.0).lam == 0.0


def test_describe_is_json_safe_for_the_instance_hash():
    """Instance.content_hash serialises the risk spec through this.

    M2b's CVaRRisk will hold numpy scenario arrays, so the protocol requires a
    describe() rather than relying on dataclasses.asdict, which would choke on
    them at the point the hash is computed.
    """
    assert VarianceRisk(lam=1e-6).describe() == {"name": "variance", "lam": 1e-6}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_dials.py tests/problem/test_risk.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.dials'`

- [ ] **Step 3: Write `dials.py`**

Create `src/quantic/problem/dials.py`:

```python
"""The four hardness dials of spec section 4.

Each is an independent on/off switch, and the combinations form a factor in
the experiment design (spec section 9.2): the study measures *which* hardness
features move the crossover, not merely whether one exists.

D2 governs cardinality and minimum lot **together**, per spec section 4:
cardinality without a minimum lot size is not meaningful on a real desk.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

# Order matters: it fixes the label format and the enumeration order, both of
# which appear in every results row.
_DIAL_LABELS: tuple[tuple[str, str], ...] = (
    ("concave_impact", "D1"),
    ("discrete_participation", "D2"),
    ("cvar_risk", "D3"),
    ("block_trades", "D4"),
)


@dataclass(frozen=True, slots=True)
class Dials:
    """Which sources of combinatorial hardness are switched on."""

    concave_impact: bool = False           # D1 -- square-root / concave impact law
    discrete_participation: bool = False   # D2 -- cardinality AND minimum lot
    cvar_risk: bool = False                # D3 -- CVaR instead of variance (M2b)
    block_trades: bool = False             # D4 -- all-or-nothing blocks

    @classmethod
    def combinations(cls, *, include_cvar: bool = False) -> tuple[Dials, ...]:
        """Every dial combination, all-off first.

        ``include_cvar=False`` yields M2a's 8. Once M2b lands, flipping this to
        ``True`` yields all 16 with no other change.
        """
        switchable = [name for name, _ in _DIAL_LABELS if include_cvar or name != "cvar_risk"]
        out: list[Dials] = []
        for flags in itertools.product((False, True), repeat=len(switchable)):
            out.append(cls(**dict(zip(switchable, flags, strict=True))))
        return tuple(out)

    @property
    def label(self) -> str:
        """A stable, readable identifier such as ``"D1+D4"`` or ``"none"``."""
        active = [label for name, label in _DIAL_LABELS if getattr(self, name)]
        return "+".join(active) if active else "none"
```

- [ ] **Step 4: Write `risk.py`**

Create `src/quantic/problem/risk.py`:

```python
"""How risk enters the objective -- and the seam M2b extends.

Spec section 5.2's risk term is either a variance or a CVaR. M2a ships
:class:`VarianceRisk`; M2b adds ``CVaRRisk(alpha, scenarios, weights,
reduction_error)`` alongside it and a matching branch in
``problem.objective.evaluate``.

``Instance.risk`` is typed to the :class:`RiskSpec` protocol precisely so that
addition reshapes nothing else. CVaR costs a binary-expanded ``zeta`` plus one
auxiliary per scenario (spec section 5.4), so the scenario set has to live
somewhere that does not perturb the rest of the instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RiskSpec(Protocol):
    name: str

    def describe(self) -> dict[str, Any]:
        """JSON-safe parameters, for the results row and the instance hash."""
        ...


@dataclass(frozen=True, slots=True)
class VarianceRisk:
    """Mean-variance risk: ``lam * sum over t of h_t' Sigma_price h_t``.

    The bucket horizon is already inside ``MarketParams.covariance``, so spec
    section 5.2's ``tau`` does not appear again here.
    """

    lam: float
    name: str = field(default="variance")

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError(
                f"lam must be non-negative, got {self.lam}: a negative risk aversion "
                "rewards variance and makes the objective unbounded below"
            )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "lam": self.lam}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_dials.py tests/problem/test_risk.py -q`
Expected: 14 passed

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/dials.py src/quantic/problem/risk.py tests/problem/test_dials.py tests/problem/test_risk.py
git commit -m "feat(problem): add Dials and the RiskSpec extension point"
```

---

## Task 3: `MarketParams`

**Files:**
- Create: `src/quantic/problem/params.py`
- Test: `tests/problem/test_params.py`

**Interfaces:**
- Consumes: `quantic.micro.impact.base.ImpactParams`.
- Produces: `AssetParams(symbol, delta, y_coef, gamma, sigma_bucket, bucket_volume_shares, price, half_spread)`; `MarketParams(assets, covariance, bucket_ns)` with `.n_assets`, `.symbols`, `.prices() -> np.ndarray`, `.impact_params(i) -> ImpactParams`, `.price_covariance() -> np.ndarray`, `.subset(n) -> MarketParams`.

- [ ] **Step 1: Write the failing test**

Create `tests/problem/test_params.py`:

```python
import numpy as np
import pytest

from quantic.micro.impact.base import ImpactParams
from quantic.problem.params import AssetParams, MarketParams

BUCKET_NS = 1800 * 1_000_000_000


def _asset(symbol="AAA", **kw) -> AssetParams:
    base = dict(
        symbol=symbol, delta=0.5, y_coef=0.8, gamma=0.1,
        sigma_bucket=0.005, bucket_volume_shares=400_000.0,
        price=100.0, half_spread=0.01,
    )
    base.update(kw)
    return AssetParams(**base)


def _params(n=2, cov=None) -> MarketParams:
    assets = tuple(_asset(f"S{i}", price=100.0 + i) for i in range(n))
    if cov is None:
        cov = np.eye(n) * 1e-5
    return MarketParams(assets=assets, covariance=cov, bucket_ns=BUCKET_NS)


def test_shape_and_symbols():
    p = _params(3)
    assert p.n_assets == 3
    assert p.symbols == ("S0", "S1", "S2")


def test_prices_are_returned_in_asset_order():
    assert _params(3).prices().tolist() == [100.0, 101.0, 102.0]


def test_impact_params_builds_micros_type_with_the_bucket_horizon_attached():
    """This is the only place micro.ImpactParams is constructed."""
    ip = _params().impact_params(0)
    assert isinstance(ip, ImpactParams)
    assert ip.symbol == "S0"
    assert ip.sigma_bucket == 0.005
    assert ip.bucket_ns == BUCKET_NS
    assert ip.bucket_volume_shares == 400_000.0
    assert ip.price == 100.0


def test_price_covariance_converts_returns_to_price_changes():
    """Sigma_price = diag(p) @ Sigma_returns @ diag(p).

    Left implicit this is a silent error of price**2 -- four orders of
    magnitude at a $100 stock -- and entirely plausible in the output.
    """
    cov = np.array([[1e-4, 2e-5], [2e-5, 9e-5]])
    p = MarketParams(
        assets=(_asset("A", price=10.0), _asset("B", price=50.0)),
        covariance=cov, bucket_ns=BUCKET_NS,
    )
    expected = np.array([[1e-4 * 100, 2e-5 * 500], [2e-5 * 500, 9e-5 * 2500]])
    assert np.allclose(p.price_covariance(), expected)


def test_price_covariance_stays_symmetric():
    assert np.allclose(_params(3).price_covariance(), _params(3).price_covariance().T)


def test_subset_keeps_the_leading_assets_and_the_matching_covariance_block():
    full = _params(4)
    sub = full.subset(2)
    assert sub.n_assets == 2
    assert sub.symbols == ("S0", "S1")
    assert sub.covariance.shape == (2, 2)
    assert np.allclose(sub.covariance, full.covariance[:2, :2])


def test_subset_rejects_asking_for_more_assets_than_exist():
    with pytest.raises(ValueError, match="4 assets"):
        _params(4).subset(9)


# --- rejections: each is a silent-wrongness source, not a nicety -----------


def test_a_non_symmetric_covariance_is_rejected():
    cov = np.array([[1e-4, 2e-5], [7e-5, 9e-5]])
    with pytest.raises(ValueError, match="symmetric"):
        MarketParams(assets=(_asset("A"), _asset("B")), covariance=cov, bucket_ns=BUCKET_NS)


def test_a_non_psd_covariance_is_rejected():
    """A non-PSD covariance makes the risk term unbounded below.

    Every optimum downstream is then meaningless, so it is refused at
    construction rather than discovered as a suspiciously good solution.
    """
    cov = np.array([[1e-4, 5e-4], [5e-4, 1e-4]])   # eigenvalue -4e-4
    with pytest.raises(ValueError, match="positive semi-definite|PSD"):
        MarketParams(assets=(_asset("A"), _asset("B")), covariance=cov, bucket_ns=BUCKET_NS)


def test_a_covariance_of_the_wrong_size_is_rejected():
    with pytest.raises(ValueError, match="shape|dimension"):
        MarketParams(assets=(_asset("A"),), covariance=np.eye(3) * 1e-5, bucket_ns=BUCKET_NS)


def test_a_tiny_negative_eigenvalue_from_rounding_is_tolerated():
    """Ledoit-Wolf output is PSD up to floating-point noise; reject only real violations."""
    cov = np.eye(2) * 1e-5
    cov[0, 0] -= 1e-20
    MarketParams(assets=(_asset("A"), _asset("B")), covariance=cov, bucket_ns=BUCKET_NS)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("delta", 0.0, "delta"),
        ("delta", 1.5, "delta"),
        ("y_coef", -0.1, "y_coef"),
        ("gamma", -0.1, "gamma"),
        ("sigma_bucket", -0.1, "sigma_bucket"),
        ("bucket_volume_shares", 0.0, "bucket_volume_shares"),
        ("price", 0.0, "price"),
        ("half_spread", -0.01, "half_spread"),
    ],
)
def test_out_of_range_asset_params_are_rejected(field, value, match):
    with pytest.raises(ValueError, match=match):
        _asset(**{field: value})


def test_a_non_positive_bucket_ns_is_rejected():
    with pytest.raises(ValueError, match="bucket_ns"):
        MarketParams(assets=(_asset("A"),), covariance=np.eye(1) * 1e-5, bucket_ns=0)


def test_duplicate_symbols_are_rejected():
    """Two rows for one symbol means the covariance rows cannot be trusted."""
    with pytest.raises(ValueError, match="duplicate"):
        MarketParams(
            assets=(_asset("A"), _asset("A")), covariance=np.eye(2) * 1e-5,
            bucket_ns=BUCKET_NS,
        )


def test_no_assets_is_rejected():
    with pytest.raises(ValueError, match="at least one asset"):
        MarketParams(assets=(), covariance=np.zeros((0, 0)), bucket_ns=BUCKET_NS)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_params.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.params'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/problem/params.py`:

```python
"""Per-asset market inputs to the liquidation objective.

Units are named in the field names, deliberately. Three volatility time bases
coexist in this project -- a per-bucket sigma from calibration, a daily
``SynthConfig.daily_vol``, and an annualised ``CovarianceEstimate.matrix`` --
and an objective built on the wrong one is wrong by a factor of ~sqrt(13) or
~sqrt(3276) while remaining entirely plausible.

``covariance`` is a **log-return** covariance at the **bucket** horizon.
:meth:`MarketParams.price_covariance` converts it to the price-change
covariance the Almgren-Chriss risk term actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.micro.impact.base import ImpactParams

# Ledoit-Wolf output is PSD up to floating-point noise, so an exactly-zero
# floor would reject valid estimates. Scaled by the largest eigenvalue so the
# tolerance means the same thing at any magnitude.
_PSD_RTOL = 1e-10


@dataclass(frozen=True, slots=True)
class AssetParams:
    """Everything the objective needs about one asset, with units in the names.

    ``half_spread`` is spec section 5.2's ``s_i / 2``, in currency per share.
    ``gamma`` is the permanent impact coefficient. ``delta`` and ``y_coef``
    are the power-law parameters; ``delta`` must lie in ``(0, 1]``, matching
    ``micro.impact.sqrt_law.PowerLawImpact``.
    """

    symbol: str
    delta: float
    y_coef: float
    gamma: float
    sigma_bucket: float
    bucket_volume_shares: float
    price: float
    half_spread: float

    def __post_init__(self) -> None:
        if not 0.0 < self.delta <= 1.0:
            raise ValueError(
                f"{self.symbol}: delta must lie in (0, 1], got {self.delta}. Above 1 is a "
                "convex impact law, which hardness dial D1 does not model"
            )
        if self.y_coef < 0:
            raise ValueError(f"{self.symbol}: y_coef must be non-negative, got {self.y_coef}")
        if self.gamma < 0:
            raise ValueError(f"{self.symbol}: gamma must be non-negative, got {self.gamma}")
        if self.sigma_bucket < 0:
            raise ValueError(
                f"{self.symbol}: sigma_bucket must be non-negative, got {self.sigma_bucket}"
            )
        if self.bucket_volume_shares <= 0:
            raise ValueError(
                f"{self.symbol}: bucket_volume_shares must be positive, got "
                f"{self.bucket_volume_shares}"
            )
        if self.price <= 0:
            raise ValueError(f"{self.symbol}: price must be positive, got {self.price}")
        if self.half_spread < 0:
            raise ValueError(
                f"{self.symbol}: half_spread must be non-negative, got {self.half_spread}"
            )


@dataclass(frozen=True)
class MarketParams:
    """Per-asset parameters plus the cross-asset covariance.

    Not ``slots=True``: it holds a numpy array, and the frozen-dataclass
    machinery is enough here.
    """

    assets: tuple[AssetParams, ...]
    covariance: np.ndarray
    bucket_ns: int

    def __post_init__(self) -> None:
        if not self.assets:
            raise ValueError("MarketParams needs at least one asset")
        if self.bucket_ns <= 0:
            raise ValueError(f"bucket_ns must be positive, got {self.bucket_ns}")

        symbols = [a.symbol for a in self.assets]
        duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate symbols in MarketParams: {duplicates}. Two rows for one symbol "
                "means the covariance rows cannot be matched to assets unambiguously"
            )

        cov = np.asarray(self.covariance, dtype=float)
        n = len(self.assets)
        if cov.shape != (n, n):
            raise ValueError(
                f"covariance shape {cov.shape} does not match the {n}-asset dimension"
            )
        if not np.allclose(cov, cov.T, rtol=1e-9, atol=1e-18):
            raise ValueError("covariance must be symmetric")

        eigenvalues = np.linalg.eigvalsh(cov)
        floor = -_PSD_RTOL * max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(eigenvalues.min()) < floor:
            raise ValueError(
                f"covariance is not positive semi-definite (smallest eigenvalue "
                f"{eigenvalues.min():.3e}). A non-PSD covariance makes the risk term "
                "unbounded below, so every optimum derived from it is meaningless"
            )

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(a.symbol for a in self.assets)

    def prices(self) -> np.ndarray:
        return np.array([a.price for a in self.assets], dtype=float)

    def impact_params(self, i: int) -> ImpactParams:
        """Build ``micro``'s per-bucket impact input for asset ``i``.

        The single place ``ImpactParams`` is constructed in this layer, which
        keeps the unit contract in one function.
        """
        a = self.assets[i]
        return ImpactParams(
            symbol=a.symbol,
            sigma_bucket=a.sigma_bucket,
            bucket_ns=self.bucket_ns,
            bucket_volume_shares=a.bucket_volume_shares,
            price=a.price,
        )

    def price_covariance(self) -> np.ndarray:
        """``diag(price) @ covariance @ diag(price)``.

        The stored covariance is of log returns and is dimensionless; the risk
        term needs a covariance of price changes so that ``h' Sigma h`` with
        ``h`` in shares comes out in currency squared.
        """
        p = self.prices()
        return np.asarray(self.covariance, dtype=float) * np.outer(p, p)

    def subset(self, n: int) -> MarketParams:
        """The first ``n`` assets and the matching covariance block."""
        if n > self.n_assets:
            raise ValueError(
                f"cannot take {n} assets from MarketParams holding {self.n_assets} assets"
            )
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        cov = np.asarray(self.covariance, dtype=float)[:n, :n]
        return MarketParams(assets=self.assets[:n], covariance=cov, bucket_ns=self.bucket_ns)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_params.py -q`
Expected: 21 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/params.py tests/problem/test_params.py
git commit -m "feat(problem): add MarketParams with PSD and unit validation"
```

---

## Task 4: `Constraint` protocol and `FullLiquidation`

**Files:**
- Create: `src/quantic/problem/constraints/base.py`, `src/quantic/problem/constraints/full_liquidation.py`
- Test: `tests/problem/constraints/test_full_liquidation.py`

**Interfaces:**
- Consumes: `Schedule` (Task 1).
- Produces: `Constraint` protocol with `name: str`, `violation(schedule, initial_lots) -> float`, `is_satisfied(schedule, initial_lots) -> bool`, `describe() -> dict[str, Any]`; module function `check_position_length(schedule, initial_lots)`; `FullLiquidation()`.

The signature takes `initial_lots` rather than the whole `Instance`, which would be circular — `Instance` holds the constraints. Only `FullLiquidation` uses it; a uniform signature keeps aggregation trivial.

- [ ] **Step 1: Create the test package**

```bash
mkdir -p tests/problem/constraints
```

- [ ] **Step 2: Write the failing test**

Create `tests/problem/constraints/test_full_liquidation.py`:

```python
import pytest

from quantic.problem.constraints.base import Constraint
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.schedule import Schedule

X = (3, 4)


def test_it_satisfies_the_constraint_protocol():
    assert isinstance(FullLiquidation(), Constraint)


def test_a_schedule_selling_exactly_the_position_is_satisfied():
    s = Schedule(lots=((2, 1), (0, 4)))
    assert FullLiquidation().violation(s, X) == 0.0
    assert FullLiquidation().is_satisfied(s, X)


def test_under_selling_is_a_violation_of_the_shortfall():
    s = Schedule(lots=((2, 0), (0, 4)))
    assert FullLiquidation().violation(s, X) == 1.0
    assert not FullLiquidation().is_satisfied(s, X)


def test_over_selling_is_a_violation_of_the_excess():
    s = Schedule(lots=((2, 3), (0, 4)))
    assert FullLiquidation().violation(s, X) == 2.0


def test_violations_accumulate_across_assets():
    s = Schedule(lots=((1, 0), (0, 1)))
    assert FullLiquidation().violation(s, X) == 5.0


def test_a_larger_shortfall_gives_a_larger_violation():
    """M3's penalty derivation needs the magnitude, not just the boolean."""
    near = Schedule(lots=((2, 0), (0, 4)))
    far = Schedule(lots=((0, 0), (0, 4)))
    assert FullLiquidation().violation(far, X) > FullLiquidation().violation(near, X)


def test_violation_is_zero_exactly_when_satisfied():
    for lots in (((2, 1), (0, 4)), ((3, 0), (4, 0)), ((0, 0), (0, 0))):
        s = Schedule(lots=lots)
        c = FullLiquidation()
        assert (c.violation(s, X) == 0.0) == c.is_satisfied(s, X)


def test_a_mismatched_position_length_is_rejected():
    s = Schedule(lots=((2, 1), (0, 4)))
    with pytest.raises(ValueError, match="2 assets"):
        FullLiquidation().violation(s, (3, 4, 5))


def test_describe_is_json_safe_for_the_results_row():
    assert FullLiquidation().describe() == {"name": "full_liquidation"}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/constraints/test_full_liquidation.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.constraints.base'`

- [ ] **Step 4: Write `base.py`**

Create `src/quantic/problem/constraints/base.py`:

```python
"""The constraint interface.

A constraint is a frozen descriptor that carries its parameters **and** the
problem-level logic for deciding whether a schedule satisfies it. It does not
emit QUBO terms: spec section 12 forbids ``problem/`` importing ``encoding/``,
and each constraint has a genuinely different quadratisation anyway (spec
section 6.2's min-participation trick is not derivable from a generic
interface).

Feasibility rate is a headline metric (spec section 6.5) and must be
computable with no encoder in sight -- including for M4's CP-SAT and MILP
solvers, which express constraints natively and never build a QUBO. That is
why the logic lives here rather than in the encoder.

``violation`` returns a **magnitude**, not a boolean, because M3's penalty
derivation (spec section 6.3) needs the size of a one-unit violation and M4's
repair strategies need to know how far off a sample is.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quantic.problem.schedule import Schedule


@runtime_checkable
class Constraint(Protocol):
    name: str

    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float:
        """Non-negative magnitude of the breach; exactly ``0.0`` when satisfied."""
        ...

    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool: ...

    def describe(self) -> dict[str, Any]:
        """JSON-safe parameters, for the results row and the instance hash."""
        ...


def check_position_length(schedule: Schedule, initial_lots: tuple[int, ...]) -> None:
    """Shared guard: the position vector must match the schedule's asset count."""
    if len(initial_lots) != schedule.n_assets:
        raise ValueError(
            f"schedule has {schedule.n_assets} assets but initial_lots has "
            f"{len(initial_lots)} entries"
        )
```

- [ ] **Step 5: Write `full_liquidation.py`**

Create `src/quantic/problem/constraints/full_liquidation.py`:

```python
"""Spec section 5.3: every position is fully liquidated by the horizon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantic.problem.constraints.base import check_position_length
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class FullLiquidation:
    """``sum over t of x[i,t] == X[i]`` for every asset ``i``.

    Present under every dial combination. It is what makes the problem a
    liquidation rather than an unconstrained trade-off.
    """

    name: str = field(default="full_liquidation")

    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float:
        check_position_length(schedule, initial_lots)
        return float(
            sum(abs(schedule.sold(i) - initial_lots[i]) for i in range(schedule.n_assets))
        )

    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool:
        return self.violation(schedule, initial_lots) == 0.0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/constraints/test_full_liquidation.py -q`
Expected: 9 passed

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/constraints tests/problem/constraints
git commit -m "feat(problem): add the Constraint protocol and FullLiquidation"
```

---

## Task 5: `Cardinality` (dial D2, part 1)

**Files:**
- Create: `src/quantic/problem/constraints/cardinality.py`
- Test: `tests/problem/constraints/test_cardinality.py`

**Interfaces:**
- Consumes: `Schedule` (Task 1), `check_position_length` (Task 4).
- Produces: `Cardinality(k: int)`.

- [ ] **Step 1: Write the failing test**

Create `tests/problem/constraints/test_cardinality.py`:

```python
import pytest

from quantic.problem.constraints.base import Constraint
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.schedule import Schedule

X = (5, 5, 5)


def test_it_satisfies_the_constraint_protocol():
    assert isinstance(Cardinality(k=2), Constraint)


def test_a_schedule_within_the_limit_is_satisfied():
    s = Schedule(lots=((3, 0), (2, 0), (0, 5)))
    assert Cardinality(k=2).violation(s, X) == 0.0
    assert Cardinality(k=2).is_satisfied(s, X)


def test_exceeding_the_limit_counts_the_excess_names():
    s = Schedule(lots=((3, 0), (2, 0), (1, 0)))
    assert Cardinality(k=2).violation(s, X) == 1.0


def test_excess_accumulates_across_buckets():
    s = Schedule(lots=((3, 1), (2, 1), (1, 1)))
    assert Cardinality(k=2).violation(s, X) == 2.0


def test_a_larger_breach_gives_a_larger_violation():
    c = Cardinality(k=1)
    assert c.violation(Schedule(lots=((1, 0), (1, 0), (1, 0))), X) == 2.0
    assert c.violation(Schedule(lots=((1, 0), (1, 0), (0, 0))), X) == 1.0


def test_a_zero_entry_does_not_count_as_a_traded_name():
    s = Schedule(lots=((0, 5), (0, 5), (0, 5)))
    assert Cardinality(k=1).violation(s, X) == 2.0


def test_violation_is_zero_exactly_when_satisfied():
    c = Cardinality(k=2)
    for lots in (((3, 0), (2, 0), (0, 5)), ((1, 1), (1, 1), (1, 1))):
        s = Schedule(lots=lots)
        assert (c.violation(s, X) == 0.0) == c.is_satisfied(s, X)


def test_k_at_least_the_asset_count_constrains_nothing():
    """A generator that picks such a k has made D2 a no-op dial."""
    s = Schedule(lots=((1, 1), (1, 1), (1, 1)))
    assert Cardinality(k=3).violation(s, X) == 0.0


def test_a_non_positive_k_is_rejected():
    with pytest.raises(ValueError, match="k"):
        Cardinality(k=0)


def test_describe_carries_k_for_the_results_row():
    assert Cardinality(k=3).describe() == {"name": "cardinality", "k": 3}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/constraints/test_cardinality.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.constraints.cardinality'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/problem/constraints/cardinality.py`:

```python
"""Dial D2, part 1: touch at most ``k`` names per bucket.

Standard desk practice, and combinatorial with no convex relaxation -- the
semi-continuous structure forces big-M binaries in any MIQP formulation (spec
section 4). Spec section 6.2 encodes it with a binary slack:
``(sum over i of y[i,t] + sum over j of 2^j s_j - k)^2``.

Paired with :class:`MinParticipation` under one dial, because cardinality
without a minimum lot size is not meaningful on a real desk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantic.problem.constraints.base import check_position_length
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class Cardinality:
    """At most ``k`` assets may have ``x[i,t] > 0`` in any single bucket ``t``."""

    k: int
    name: str = field(default="cardinality")

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(
                f"k must be at least 1, got {self.k}: a cardinality of zero forbids all "
                "trading and makes full liquidation impossible"
            )

    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float:
        check_position_length(schedule, initial_lots)
        return float(
            sum(
                max(0, schedule.active_names(t) - self.k)
                for t in range(schedule.n_buckets)
            )
        )

    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool:
        return self.violation(schedule, initial_lots) == 0.0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "k": self.k}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/constraints/test_cardinality.py -q`
Expected: 10 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/constraints/cardinality.py tests/problem/constraints/test_cardinality.py
git commit -m "feat(problem): add the Cardinality constraint (dial D2)"
```

---

## Task 6: `MinParticipation` (dial D2, part 2)

**Files:**
- Create: `src/quantic/problem/constraints/min_participation.py`
- Test: `tests/problem/constraints/test_min_participation.py`

**Interfaces:**
- Consumes: `Schedule` (Task 1), `check_position_length` (Task 4).
- Produces: `MinParticipation(min_lots: tuple[int, ...])`.

- [ ] **Step 1: Write the failing test**

Create `tests/problem/constraints/test_min_participation.py`:

```python
import pytest

from quantic.problem.constraints.base import Constraint
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.schedule import Schedule

X = (6, 6)
MIN = (3, 2)


def test_it_satisfies_the_constraint_protocol():
    assert isinstance(MinParticipation(min_lots=MIN), Constraint)


def test_zero_or_at_least_the_minimum_is_satisfied():
    """The constraint is semi-continuous: x == 0 OR x >= m."""
    s = Schedule(lots=((3, 3), (0, 6)))
    assert MinParticipation(min_lots=MIN).violation(s, X) == 0.0
    assert MinParticipation(min_lots=MIN).is_satisfied(s, X)


def test_a_non_zero_entry_below_the_minimum_is_a_violation_of_the_shortfall():
    s = Schedule(lots=((1, 5), (0, 6)))
    assert MinParticipation(min_lots=MIN).violation(s, X) == 2.0


def test_violations_accumulate_across_cells():
    s = Schedule(lots=((1, 1), (1, 5)))
    assert MinParticipation(min_lots=MIN).violation(s, X) == 5.0


def test_a_larger_shortfall_gives_a_larger_violation():
    c = MinParticipation(min_lots=MIN)
    assert c.violation(Schedule(lots=((1, 5), (0, 6))), X) > c.violation(
        Schedule(lots=((2, 4), (0, 6))), X
    )


def test_the_minimum_is_per_asset():
    s = Schedule(lots=((2, 4), (2, 4)))
    assert MinParticipation(min_lots=MIN).violation(s, X) == 1.0


def test_violation_is_zero_exactly_when_satisfied():
    c = MinParticipation(min_lots=MIN)
    for lots in (((3, 3), (0, 6)), ((1, 5), (0, 6)), ((0, 0), (0, 0))):
        s = Schedule(lots=lots)
        assert (c.violation(s, X) == 0.0) == c.is_satisfied(s, X)


def test_a_minimum_of_one_constrains_nothing():
    """Any non-zero integer entry is already at least 1."""
    s = Schedule(lots=((1, 5), (1, 5)))
    assert MinParticipation(min_lots=(1, 1)).violation(s, X) == 0.0


def test_a_mismatched_minimum_length_is_rejected():
    s = Schedule(lots=((3, 3), (0, 6)))
    with pytest.raises(ValueError, match="min_lots"):
        MinParticipation(min_lots=(3, 2, 1)).violation(s, X)


def test_a_non_positive_minimum_is_rejected():
    with pytest.raises(ValueError, match="min_lots"):
        MinParticipation(min_lots=(3, 0))


def test_describe_carries_the_minimums_as_a_list():
    assert MinParticipation(min_lots=MIN).describe() == {
        "name": "min_participation", "min_lots": [3, 2]
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/constraints/test_min_participation.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.constraints.min_participation'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/problem/constraints/min_participation.py`:

```python
"""Dial D2, part 2: a venue minimum quantity.

``x[i,t]`` is either zero or at least ``m_i`` -- a semi-continuous variable,
which is where the big-M binaries come from. Spec section 6.2 keeps the
encoding quadratic by writing ``x = m*y + sum over b of 2^b z_b`` with the
penalty ``sum over b of z_b * (1 - y)``; the naive form is cubic.

None of that is this module's business. Here the constraint is only asked
whether a schedule satisfies it, and by how much.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantic.problem.constraints.base import check_position_length
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class MinParticipation:
    """For every cell, ``x[i,t] == 0`` or ``x[i,t] >= min_lots[i]``."""

    min_lots: tuple[int, ...]
    name: str = field(default="min_participation")

    def __post_init__(self) -> None:
        for i, m in enumerate(self.min_lots):
            if m < 1:
                raise ValueError(
                    f"min_lots[{i}] must be at least 1, got {m}: a minimum of zero is not "
                    "a constraint, and a negative one is meaningless"
                )

    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float:
        check_position_length(schedule, initial_lots)
        if len(self.min_lots) != schedule.n_assets:
            raise ValueError(
                f"schedule has {schedule.n_assets} assets but min_lots has "
                f"{len(self.min_lots)} entries"
            )
        total = 0.0
        for i, row in enumerate(schedule.lots):
            minimum = self.min_lots[i]
            for value in row:
                if 0 < value < minimum:
                    total += minimum - value
        return total

    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool:
        return self.violation(schedule, initial_lots) == 0.0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "min_lots": list(self.min_lots)}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/constraints/test_min_participation.py -q`
Expected: 11 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/constraints/min_participation.py tests/problem/constraints/test_min_participation.py
git commit -m "feat(problem): add the MinParticipation constraint (dial D2)"
```

---

## Task 7: `BlockTrades` (dial D4)

**Files:**
- Create: `src/quantic/problem/constraints/block_trades.py`
- Test: `tests/problem/constraints/test_block_trades.py`

**Interfaces:**
- Consumes: `Schedule` (Task 1), `check_position_length` (Task 4).
- Produces: `Block(asset: int, bucket: int, lots: int)`; `BlockTrades(blocks: tuple[Block, ...])`.

**Semantics:** a block declares cell `(asset, bucket)` indivisible — `x[asset][bucket]` must be either `0` or exactly `lots`. Violation is the distance to the nearer of the two, so a nearly-complete block scores as nearly feasible, which is what M4's repair strategies need in order to rank candidate repairs.

- [ ] **Step 1: Write the failing test**

Create `tests/problem/constraints/test_block_trades.py`:

```python
import pytest

from quantic.problem.constraints.base import Constraint
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.schedule import Schedule

X = (6, 6)
BLOCKS = (Block(asset=0, bucket=0, lots=4),)


def test_it_satisfies_the_constraint_protocol():
    assert isinstance(BlockTrades(blocks=BLOCKS), Constraint)


def test_trading_the_block_whole_is_satisfied():
    s = Schedule(lots=((4, 2), (0, 6)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 0.0
    assert BlockTrades(blocks=BLOCKS).is_satisfied(s, X)


def test_not_trading_the_block_at_all_is_satisfied():
    """All-or-nothing means nothing is a legitimate choice."""
    s = Schedule(lots=((0, 6), (0, 6)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 0.0


def test_a_partial_fill_is_a_violation_of_the_distance_to_the_nearer_end():
    s = Schedule(lots=((3, 3), (0, 6)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 1.0


def test_a_small_partial_fill_is_measured_against_zero():
    s = Schedule(lots=((1, 5), (0, 6)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 1.0


def test_overfilling_a_block_is_a_violation():
    s = Schedule(lots=((6, 0), (0, 6)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 2.0


def test_violations_accumulate_across_blocks():
    blocks = (Block(asset=0, bucket=0, lots=4), Block(asset=1, bucket=1, lots=4))
    s = Schedule(lots=((3, 3), (0, 3)))
    assert BlockTrades(blocks=blocks).violation(s, X) == 2.0


def test_cells_without_a_block_are_unconstrained():
    s = Schedule(lots=((4, 2), (1, 5)))
    assert BlockTrades(blocks=BLOCKS).violation(s, X) == 0.0


def test_violation_is_zero_exactly_when_satisfied():
    c = BlockTrades(blocks=BLOCKS)
    for lots in (((4, 2), (0, 6)), ((3, 3), (0, 6)), ((0, 6), (0, 6))):
        s = Schedule(lots=lots)
        assert (c.violation(s, X) == 0.0) == c.is_satisfied(s, X)


def test_a_block_out_of_range_of_the_schedule_is_rejected():
    s = Schedule(lots=((4, 2), (0, 6)))
    bad = BlockTrades(blocks=(Block(asset=5, bucket=0, lots=4),))
    with pytest.raises(ValueError, match="asset 5"):
        bad.violation(s, X)


def test_two_blocks_on_the_same_cell_are_rejected():
    """They would impose two contradictory all-or-nothing sizes on one variable."""
    with pytest.raises(ValueError, match="more than one block"):
        BlockTrades(blocks=(Block(0, 0, 4), Block(0, 0, 2)))


def test_a_non_positive_block_size_is_rejected():
    with pytest.raises(ValueError, match="lots"):
        Block(asset=0, bucket=0, lots=0)


def test_a_negative_index_is_rejected():
    with pytest.raises(ValueError, match="asset"):
        Block(asset=-1, bucket=0, lots=4)


def test_describe_carries_every_block_as_json_safe_data():
    assert BlockTrades(blocks=BLOCKS).describe() == {
        "name": "block_trades",
        "blocks": [{"asset": 0, "bucket": 0, "lots": 4}],
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/constraints/test_block_trades.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.constraints.block_trades'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/problem/constraints/block_trades.py`:

```python
"""Dial D4: all-or-nothing block trades.

Block and dark crossing is discrete by construction (spec section 4), and the
resulting structure is purely binary: spec section 6.2 introduces one binary
``b_k`` per block, contributing its fixed quantity to the full-liquidation
constraint.

At the schedule level a block makes its cell indivisible: ``x[asset][bucket]``
is either ``0`` or exactly ``lots``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantic.problem.constraints.base import check_position_length
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class Block:
    """An indivisible quantity offered in one asset-bucket cell."""

    asset: int
    bucket: int
    lots: int

    def __post_init__(self) -> None:
        if self.asset < 0:
            raise ValueError(f"asset index must be non-negative, got {self.asset}")
        if self.bucket < 0:
            raise ValueError(f"bucket index must be non-negative, got {self.bucket}")
        if self.lots < 1:
            raise ValueError(
                f"block lots must be at least 1, got {self.lots}: a zero-size block is "
                "not a trade"
            )


@dataclass(frozen=True, slots=True)
class BlockTrades:
    """Each block's cell trades whole or not at all."""

    blocks: tuple[Block, ...]
    name: str = field(default="block_trades")

    def __post_init__(self) -> None:
        cells = [(b.asset, b.bucket) for b in self.blocks]
        duplicates = sorted({c for c in cells if cells.count(c) > 1})
        if duplicates:
            raise ValueError(
                f"more than one block on cell(s) {duplicates}: two blocks on one cell "
                "impose contradictory all-or-nothing sizes on the same variable"
            )

    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float:
        check_position_length(schedule, initial_lots)
        total = 0.0
        for b in self.blocks:
            if b.asset >= schedule.n_assets:
                raise ValueError(
                    f"block references asset {b.asset} but the schedule has "
                    f"{schedule.n_assets} assets"
                )
            if b.bucket >= schedule.n_buckets:
                raise ValueError(
                    f"block references bucket {b.bucket} but the schedule has "
                    f"{schedule.n_buckets} buckets"
                )
            value = schedule.lots[b.asset][b.bucket]
            total += float(min(value, abs(value - b.lots)))
        return total

    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool:
        return self.violation(schedule, initial_lots) == 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "blocks": [
                {"asset": b.asset, "bucket": b.bucket, "lots": b.lots} for b in self.blocks
            ],
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/constraints/test_block_trades.py -q`
Expected: 14 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/constraints/block_trades.py tests/problem/constraints/test_block_trades.py
git commit -m "feat(problem): add the BlockTrades constraint (dial D4)"
```

---

## Task 8: `Instance`

**Files:**
- Create: `src/quantic/problem/instance.py`
- Test: `tests/problem/test_instance.py`

**Interfaces:**
- Consumes: `Asset` (`quantic.core.types`), `Schedule` (1), `Dials` (2), `RiskSpec` (2), `MarketParams` (3), `Constraint` (4).
- Produces: `Instance(instance_id, tier, assets, n_buckets, initial_lots, max_lots_per_bucket, params, dials, constraints, risk, witness, seed)` with `.n_assets`, `.lot_sizes() -> np.ndarray`, `.remaining_lots(schedule) -> np.ndarray`, `.notional() -> float`, `.content_hash -> str`.

**Refinement of the spec sketch.** Spec §5.8 lists `content_hash` as a *field*. Here it is a computed `@property`. A stored hash can go stale against the fields it summarises, and a results row pins it — a stale hash silently attributes a figure to the wrong instance. Computing it on demand makes that impossible. It is cheap (one JSON dump of a few hundred numbers).

- [ ] **Step 1: Write the failing test**

Create `tests/problem/test_instance.py`:

```python
import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(**kw) -> Instance:
    assets = (Asset("A", lot_size=100, tick_size=0.01), Asset("B", lot_size=10, tick_size=0.01))
    params = MarketParams(
        assets=(
            AssetParams("A", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01),
            AssetParams("B", 0.5, 0.8, 0.1, 0.005, 400_000.0, 50.0, 0.01),
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    base = dict(
        instance_id="T0-none-0",
        tier="T0",
        assets=assets,
        n_buckets=2,
        initial_lots=(3, 4),
        max_lots_per_bucket=3,
        params=params,
        dials=Dials(),
        constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6),
        witness=Schedule(lots=((2, 1), (3, 1))),
        seed=0,
    )
    base.update(kw)
    return Instance(**base)


def test_shape_is_derived_from_the_assets():
    assert _instance().n_assets == 2
    assert _instance().n_buckets == 2


def test_lot_sizes_are_read_from_the_core_asset_type():
    assert _instance().lot_sizes().tolist() == [100, 10]


def test_remaining_lots_is_the_position_minus_inclusive_cumulative():
    """Spec 5.1: h[i,t] = X[i] - sum over s <= t of x[i,s]."""
    h = _instance().remaining_lots(Schedule(lots=((2, 1), (3, 1))))
    assert h.tolist() == [[1, 0], [1, 0]]


def test_remaining_lots_reaches_zero_under_full_liquidation():
    h = _instance().remaining_lots(_instance().witness)
    assert h[:, -1].tolist() == [0, 0]


def test_remaining_lots_rejects_a_schedule_of_the_wrong_shape():
    with pytest.raises(ValueError, match="shape|assets|buckets"):
        _instance().remaining_lots(Schedule(lots=((1, 1, 1),)))


def test_notional_converts_lots_to_shares_before_pricing():
    """3 lots x 100 shares x $100 + 4 lots x 10 shares x $50 = 32000."""
    assert _instance().notional() == pytest.approx(3 * 100 * 100.0 + 4 * 10 * 50.0)


# --- content hash ----------------------------------------------------------


def test_content_hash_is_deterministic():
    assert _instance().content_hash == _instance().content_hash


def test_content_hash_is_a_sha256_hex_digest():
    h = _instance().content_hash
    assert len(h) == 64
    assert set(h) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    "field,value",
    [
        ("tier", "T1"),
        ("seed", 1),
        ("initial_lots", (4, 4)),
        ("max_lots_per_bucket", 7),
        ("dials", Dials(concave_impact=True)),
        ("risk", VarianceRisk(lam=2e-6)),
        ("witness", Schedule(lots=((3, 0), (4, 0)))),
        ("constraints", (FullLiquidation(), Cardinality(k=1))),
    ],
)
def test_changing_any_field_changes_the_hash(field, value):
    assert _instance().content_hash != _instance(**{field: value}).content_hash


def test_changing_the_covariance_changes_the_hash():
    """The hash must cover the parameters, not only the problem shape."""
    p = _instance().params
    other = dataclasses.replace(p, covariance=np.eye(2) * 2e-5)
    assert _instance().content_hash != _instance(params=other).content_hash


def test_instance_id_is_not_hashed():
    """The hash identifies the content; the id is a human-facing label.

    Two instances with identical content must collide in the hash, so a
    duplicate is detectable.
    """
    assert _instance().content_hash == _instance(instance_id="something-else").content_hash


# --- rejections ------------------------------------------------------------


def test_a_position_vector_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="initial_lots"):
        _instance(initial_lots=(3,))


def test_params_describing_a_different_asset_count_are_rejected():
    p = _instance().params.subset(1)
    with pytest.raises(ValueError, match="params"):
        _instance(params=p)


def test_symbols_that_disagree_between_assets_and_params_are_rejected():
    """Silently mismatched ordering would misprice every asset."""
    swapped = (Asset("B", lot_size=100, tick_size=0.01), Asset("A", lot_size=10, tick_size=0.01))
    with pytest.raises(ValueError, match="symbol"):
        _instance(assets=swapped)


def test_a_witness_of_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="witness"):
        _instance(witness=Schedule(lots=((1, 1, 1), (1, 1, 1))))


def test_a_witness_exceeding_max_lots_per_bucket_is_rejected():
    with pytest.raises(ValueError, match="max_lots_per_bucket"):
        _instance(witness=Schedule(lots=((9, 0), (4, 0))), initial_lots=(9, 4))


def test_a_negative_position_is_rejected():
    with pytest.raises(ValueError, match="initial_lots"):
        _instance(initial_lots=(-1, 4))


def test_a_non_positive_bucket_count_is_rejected():
    with pytest.raises(ValueError, match="n_buckets"):
        _instance(n_buckets=0)


def test_no_constraints_at_all_is_rejected():
    """Without full liquidation the trivial empty schedule is optimal."""
    with pytest.raises(ValueError, match="constraint"):
        _instance(constraints=())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_instance.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.instance'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/problem/instance.py`:

```python
"""The frozen problem artifact that crosses into encoding and solvers.

Spec section 7.1 gives every solver the signature ``solve(instance, budget)``,
and section 12 forbids ``solvers/`` reaching into ``problem/`` internals. This
type is therefore the whole interface, and it is deliberately inert: shape,
parameters, dials, constraints, a risk specification and a witness. No
behaviour that a solver could depend on beyond the accessors here.

It carries ``max_lots_per_bucket`` rather than a bit width. Bit width belongs
to an encoding, and spec section 6.1 makes the integer encoding a study
dimension with three competing choices -- binary, one-hot, domain-wall --
which need different variable counts for the same problem.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantic.core.types import Asset
from quantic.problem.constraints.base import Constraint
from quantic.problem.dials import Dials
from quantic.problem.params import MarketParams
from quantic.problem.risk import RiskSpec
from quantic.problem.schedule import Schedule


@dataclass(frozen=True)
class Instance:
    """One liquidation problem, fully specified."""

    instance_id: str
    tier: str
    assets: tuple[Asset, ...]
    n_buckets: int
    initial_lots: tuple[int, ...]
    max_lots_per_bucket: int
    params: MarketParams
    dials: Dials
    constraints: tuple[Constraint, ...]
    risk: RiskSpec
    witness: Schedule
    seed: int

    def __post_init__(self) -> None:
        n = len(self.assets)
        if n == 0:
            raise ValueError("an instance needs at least one asset")
        if self.n_buckets < 1:
            raise ValueError(f"n_buckets must be at least 1, got {self.n_buckets}")
        if self.max_lots_per_bucket < 1:
            raise ValueError(
                f"max_lots_per_bucket must be at least 1, got {self.max_lots_per_bucket}"
            )
        if len(self.initial_lots) != n:
            raise ValueError(
                f"initial_lots has {len(self.initial_lots)} entries but there are {n} assets"
            )
        if any(x < 0 for x in self.initial_lots):
            raise ValueError(f"initial_lots must be non-negative, got {self.initial_lots}")
        if self.params.n_assets != n:
            raise ValueError(
                f"params describes {self.params.n_assets} assets but the instance has {n}"
            )
        if self.params.symbols != tuple(a.symbol for a in self.assets):
            raise ValueError(
                f"symbol mismatch: assets are {tuple(a.symbol for a in self.assets)} but "
                f"params are {self.params.symbols}. Mismatched ordering would misprice "
                "every asset silently"
            )
        if not self.constraints:
            raise ValueError(
                "an instance needs at least one constraint; without full liquidation the "
                "trivial empty schedule is optimal"
            )
        if (self.witness.n_assets, self.witness.n_buckets) != (n, self.n_buckets):
            raise ValueError(
                f"witness has shape ({self.witness.n_assets}, {self.witness.n_buckets}) "
                f"but the instance is ({n}, {self.n_buckets})"
            )
        worst = int(self.witness.as_array().max())
        if worst > self.max_lots_per_bucket:
            raise ValueError(
                f"witness trades {worst} lots in a bucket but max_lots_per_bucket is "
                f"{self.max_lots_per_bucket}"
            )

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def lot_sizes(self) -> np.ndarray:
        return np.array([a.lot_size for a in self.assets], dtype=np.int64)

    def remaining_lots(self, schedule: Schedule) -> np.ndarray:
        """``(N, T)`` lots still held after each bucket -- spec section 5.1's ``h``."""
        if (schedule.n_assets, schedule.n_buckets) != (self.n_assets, self.n_buckets):
            raise ValueError(
                f"schedule shape ({schedule.n_assets}, {schedule.n_buckets}) does not match "
                f"the instance ({self.n_assets}, {self.n_buckets})"
            )
        return np.array(self.initial_lots, dtype=np.int64)[:, None] - schedule.cumulative()

    def notional(self) -> float:
        """Total currency value of the position, for reporting cost in bps."""
        return float(
            sum(
                self.initial_lots[i] * self.assets[i].lot_size * self.params.assets[i].price
                for i in range(self.n_assets)
            )
        )

    def _canonical_payload(self) -> dict[str, Any]:
        """Everything that defines the instance's content, JSON-safe and ordered.

        ``instance_id`` is deliberately excluded: it is a human-facing label,
        and two instances with identical content must collide so a duplicate
        is detectable.
        """
        return {
            "tier": self.tier,
            "n_buckets": self.n_buckets,
            "initial_lots": list(self.initial_lots),
            "max_lots_per_bucket": self.max_lots_per_bucket,
            "seed": self.seed,
            "assets": [
                {
                    "symbol": a.symbol,
                    "lot_size": a.lot_size,
                    "tick_size": a.tick_size,
                    "currency": a.currency,
                }
                for a in self.assets
            ],
            "params": {
                "bucket_ns": self.params.bucket_ns,
                "assets": [
                    {
                        "symbol": a.symbol,
                        "delta": a.delta,
                        "y_coef": a.y_coef,
                        "gamma": a.gamma,
                        "sigma_bucket": a.sigma_bucket,
                        "bucket_volume_shares": a.bucket_volume_shares,
                        "price": a.price,
                        "half_spread": a.half_spread,
                    }
                    for a in self.params.assets
                ],
                "covariance": np.asarray(self.params.covariance, dtype=float).tolist(),
            },
            "dials": {
                "concave_impact": self.dials.concave_impact,
                "discrete_participation": self.dials.discrete_participation,
                "cvar_risk": self.dials.cvar_risk,
                "block_trades": self.dials.block_trades,
            },
            "constraints": [c.describe() for c in self.constraints],
            "risk": self.risk.describe(),
            "witness": [list(row) for row in self.witness.lots],
        }

    @property
    def content_hash(self) -> str:
        """SHA-256 over the instance's content.

        A computed property rather than a stored field: a stored hash can go
        stale against the fields it summarises, and a results row pins it, so
        a stale hash would attribute a published figure to the wrong instance.

        Python's ``repr`` of a float round-trips exactly, and ``json.dumps``
        uses it, so no rounding is applied and no precision is lost.
        """
        payload = json.dumps(
            self._canonical_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_instance.py -q`
Expected: 22 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/instance.py tests/problem/test_instance.py
git commit -m "feat(problem): add Instance with a computed content hash"
```

---

## Task 9: `classify` — one feasibility answer

**Files:**
- Create: `src/quantic/problem/feasibility.py`
- Test: `tests/problem/test_feasibility.py`

**Interfaces:**
- Consumes: `Instance` (8), `Schedule` (1), constraints (4–7).
- Produces: `FeasibilityReport(feasible: bool, violations: dict[str, float], total_violation: float)`; `classify(instance, schedule) -> FeasibilityReport`.

- [ ] **Step 1: Write the failing test**

Create `tests/problem/test_feasibility.py`:

```python
import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.feasibility import FeasibilityReport, classify
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(constraints=None, witness=None) -> Instance:
    assets = (Asset("A", lot_size=100, tick_size=0.01), Asset("B", lot_size=100, tick_size=0.01))
    params = MarketParams(
        assets=(
            AssetParams("A", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01),
            AssetParams("B", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01),
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    return Instance(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(3, 4), max_lots_per_bucket=3, params=params, dials=Dials(),
        constraints=constraints or (FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6),
        witness=witness or Schedule(lots=((2, 1), (3, 1))),
        seed=0,
    )


def test_the_witness_is_feasible():
    inst = _instance()
    report = classify(inst, inst.witness)
    assert isinstance(report, FeasibilityReport)
    assert report.feasible
    assert report.total_violation == 0.0


def test_every_constraint_appears_in_the_report_even_when_satisfied():
    """A metric that only lists breaches cannot distinguish 'clean' from 'unchecked'."""
    inst = _instance(constraints=(FullLiquidation(), Cardinality(k=2)))
    report = classify(inst, inst.witness)
    assert set(report.violations) == {"full_liquidation", "cardinality"}
    assert all(v == 0.0 for v in report.violations.values())


def test_a_breach_is_reported_against_the_constraint_that_caused_it():
    inst = _instance()
    bad = Schedule(lots=((1, 1), (3, 1)))       # asset 0 sells 2 of 3
    report = classify(inst, bad)
    assert not report.feasible
    assert report.violations["full_liquidation"] == 1.0
    assert report.total_violation == 1.0


def test_violations_from_several_constraints_are_summed():
    inst = _instance(constraints=(FullLiquidation(), Cardinality(k=1)))
    bad = Schedule(lots=((1, 1), (3, 1)))       # 1 short, and 2 names in both buckets
    report = classify(inst, bad)
    assert report.violations["full_liquidation"] == 1.0
    assert report.violations["cardinality"] == 2.0
    assert report.total_violation == 3.0


def test_feasible_is_exactly_total_violation_being_zero():
    inst = _instance(constraints=(FullLiquidation(), Cardinality(k=1)))
    for lots in (((2, 1), (3, 1)), ((1, 1), (3, 1)), ((3, 0), (0, 4))):
        report = classify(inst, Schedule(lots=lots))
        assert report.feasible == (report.total_violation == 0.0)


def test_a_schedule_of_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        classify(_instance(), Schedule(lots=((1, 1, 1), (1, 1, 1))))


def test_duplicate_constraint_names_are_rejected():
    """Two constraints sharing a name would overwrite each other in the report."""
    inst = _instance(constraints=(Cardinality(k=1), Cardinality(k=2)))
    with pytest.raises(ValueError, match="duplicate|cardinality"):
        classify(inst, inst.witness)


def test_the_report_is_hashable_and_frozen():
    inst = _instance()
    report = classify(inst, inst.witness)
    with pytest.raises(Exception):
        report.feasible = False       # type: ignore[misc]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_feasibility.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.feasibility'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/problem/feasibility.py`:

```python
"""The single answer to "is this schedule feasible".

Spec section 6.5 makes feasibility rate a headline metric reported alongside
cost, and names scoring an infeasible quantum solution against a feasible
classical one as the error that invalidates most published comparisons in this
area.

That metric is only trustworthy if there is exactly one implementation of the
test. M3's decoder, M4's CP-SAT and MILP solvers -- which express constraints
natively and never build a QUBO -- and M6's metrics all call this function.
Three independent copies is how a headline number silently disagrees with
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from quantic.problem.instance import Instance
from quantic.problem.schedule import Schedule


@dataclass(frozen=True)
class FeasibilityReport:
    """Per-constraint breach magnitudes and the verdict they imply."""

    feasible: bool
    violations: Mapping[str, float]
    total_violation: float


def classify(instance: Instance, schedule: Schedule) -> FeasibilityReport:
    """Evaluate every constraint on ``instance`` against ``schedule``.

    Every constraint appears in ``violations``, including the satisfied ones
    at ``0.0``: a report that lists only breaches cannot distinguish a clean
    schedule from one whose constraints were never checked.
    """
    if (schedule.n_assets, schedule.n_buckets) != (instance.n_assets, instance.n_buckets):
        raise ValueError(
            f"schedule shape ({schedule.n_assets}, {schedule.n_buckets}) does not match "
            f"the instance ({instance.n_assets}, {instance.n_buckets})"
        )

    names = [c.name for c in instance.constraints]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate constraint name(s) {duplicates} on instance "
            f"{instance.instance_id}: they would overwrite each other in the report"
        )

    violations = {
        c.name: float(c.violation(schedule, instance.initial_lots))
        for c in instance.constraints
    }
    total = float(sum(violations.values()))
    return FeasibilityReport(
        feasible=total == 0.0,
        violations=MappingProxyType(violations),
        total_violation=total,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_feasibility.py -q`
Expected: 8 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/feasibility.py tests/problem/test_feasibility.py
git commit -m "feat(problem): add classify, the single feasibility implementation"
```

---

## Task 10: the objective

**Files:**
- Create: `src/quantic/problem/objective.py`
- Test: `tests/problem/test_objective.py`

**Interfaces:**
- Consumes: `Instance` (8), `Schedule` (1), `PowerLawImpact` and `AlmgrenChriss` from `quantic.micro.impact`.
- Produces: `ObjectiveBreakdown(spread_cost, temporary_impact_cost, permanent_impact_cost, risk_cost)` with `.total` and `.in_bps(notional)`; `evaluate(instance, schedule) -> ObjectiveBreakdown`; `impact_model_for(instance, i) -> ImpactModel`.

**The four terms** (all in currency; `q` shares traded, `cum_before` shares traded strictly before `t`, `h` shares remaining after `t`):

1. `spread = sum half_spread[i] * q[i,t]`
2. `temporary = sum model_i.temporary_cost(q, p_i) * q[i,t] * price[i]` — `PowerLawImpact` when D1 is on, `AlmgrenChriss` when off. **This is the only term D1 changes.**
3. `permanent = sum gamma[i] * sigma[i] * (cum_before[i,t] / V[i]) * q[i,t] * price[i]`
4. `risk = lam * sum over t of h_t @ Sigma_price @ h_t`

- [ ] **Step 1: Write the failing test**

Create `tests/problem/test_objective.py`:

```python
import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.micro.impact.almgren_chriss import AlmgrenChriss
from quantic.micro.impact.sqrt_law import PowerLawImpact
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import ObjectiveBreakdown, evaluate, impact_model_for
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(*, dials=None, lam=1e-6, lot_size=100, prices=(100.0, 100.0), **kw) -> Instance:
    assets = tuple(
        Asset(s, lot_size=lot_size, tick_size=0.01) for s in ("A", "B")
    )
    params = MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, p, 0.01)
            for s, p in zip(("A", "B"), prices, strict=True)
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    base = dict(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(4, 4), max_lots_per_bucket=4, params=params,
        dials=dials or Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=lam), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def test_the_breakdown_sums_to_the_total():
    b = evaluate(_instance(), _instance().witness)
    assert b.total == pytest.approx(
        b.spread_cost + b.temporary_impact_cost + b.permanent_impact_cost + b.risk_cost
    )


def test_every_term_is_finite_and_non_negative():
    b = evaluate(_instance(), _instance().witness)
    for f in dataclasses.fields(b):
        value = getattr(b, f.name)
        assert np.isfinite(value)
        assert value >= 0.0


def test_spread_cost_is_half_spread_times_shares():
    """4 lots x 100 shares x $0.01 per share, per asset, over the whole schedule."""
    inst = _instance()
    b = evaluate(inst, inst.witness)
    assert b.spread_cost == pytest.approx(2 * 4 * 100 * 0.01)


def test_an_empty_schedule_costs_nothing_but_risk():
    inst = _instance()
    empty = Schedule(lots=((0, 0), (0, 0)))
    b = evaluate(inst, empty)
    assert b.spread_cost == 0.0
    assert b.temporary_impact_cost == 0.0
    assert b.permanent_impact_cost == 0.0
    assert b.risk_cost > 0.0       # the position is still held


# --- dial D1 ---------------------------------------------------------------


def test_d1_off_selects_the_linear_almgren_chriss_model():
    assert isinstance(impact_model_for(_instance(), 0), AlmgrenChriss)


def test_d1_on_selects_the_calibrated_power_law():
    inst = _instance(dials=Dials(concave_impact=True))
    model = impact_model_for(inst, 0)
    assert isinstance(model, PowerLawImpact)
    assert model.delta == 0.5
    assert model.y_coef == 0.8


def test_d1_changes_the_temporary_impact_term_and_nothing_else():
    """If D1 moved any other term, the screening design could not attribute the effect."""
    off = _instance()
    on = _instance(dials=Dials(concave_impact=True))
    b_off, b_on = evaluate(off, off.witness), evaluate(on, on.witness)

    assert b_on.temporary_impact_cost != pytest.approx(b_off.temporary_impact_cost)
    assert b_on.spread_cost == pytest.approx(b_off.spread_cost)
    assert b_on.permanent_impact_cost == pytest.approx(b_off.permanent_impact_cost)
    assert b_on.risk_cost == pytest.approx(b_off.risk_cost)


# --- the unit-discipline tests, where this layer is most likely to be wrong -


def test_doubling_lot_size_and_halving_lots_leaves_every_term_unchanged():
    """The lots-to-shares conversion happens exactly once. If it happened twice,
    or not at all, this identity breaks."""
    small = _instance(lot_size=100)
    big = _instance(
        lot_size=200, initial_lots=(2, 2), max_lots_per_bucket=2,
        witness=Schedule(lots=((1, 1), (1, 1))),
    )
    a = evaluate(small, small.witness)
    b = evaluate(big, big.witness)
    assert a.spread_cost == pytest.approx(b.spread_cost)
    assert a.temporary_impact_cost == pytest.approx(b.temporary_impact_cost)
    assert a.permanent_impact_cost == pytest.approx(b.permanent_impact_cost)
    assert a.risk_cost == pytest.approx(b.risk_cost)


def test_permanent_impact_is_path_independent():
    """With linear permanent impact the term depends only on the total traded.

    Two different full-liquidation schedules must agree, which is a sharp test
    of the whole term including the cumulative-before convention.
    """
    inst = _instance()
    even = Schedule(lots=((2, 2), (2, 2)))
    front = Schedule(lots=((4, 0), (4, 0)))
    assert evaluate(inst, even).permanent_impact_cost == pytest.approx(
        evaluate(inst, front).permanent_impact_cost
    )


def test_permanent_impact_matches_the_closed_form():
    """0.5 * gamma * sigma * X_shares**2 / V * price, summed over assets."""
    inst = _instance()
    shares = 4 * 100
    expected = 2 * 0.5 * 0.1 * 0.005 * shares**2 / 400_000.0 * 100.0
    assert evaluate(inst, inst.witness).permanent_impact_cost == pytest.approx(expected)


def test_scaling_every_price_scales_the_risk_term_quadratically():
    """Sigma_price = diag(p) Sigma_returns diag(p) is quadratic in price.

    Getting this wrong -- using the return covariance directly -- is a silent
    error of price**2, four orders of magnitude at a $100 stock.
    """
    base = _instance(prices=(100.0, 100.0))
    scaled = _instance(prices=(200.0, 200.0))
    assert evaluate(scaled, scaled.witness).risk_cost == pytest.approx(
        4.0 * evaluate(base, base.witness).risk_cost
    )


def test_risk_is_linear_in_the_risk_aversion():
    a = _instance(lam=1e-6)
    b = _instance(lam=2e-6)
    assert evaluate(b, b.witness).risk_cost == pytest.approx(
        2.0 * evaluate(a, a.witness).risk_cost
    )


def test_zero_risk_aversion_removes_the_risk_term_entirely():
    inst = _instance(lam=0.0)
    assert evaluate(inst, inst.witness).risk_cost == 0.0


def test_the_final_bucket_carries_no_risk_under_full_liquidation():
    """h is the holding AFTER each bucket, so the last one is zero."""
    inst = _instance()
    h = inst.remaining_lots(inst.witness)
    assert h[:, -1].tolist() == [0, 0]


def test_front_loading_costs_more_impact_than_spreading():
    """Concave or linear, trading faster costs more. A sanity check on the sign."""
    inst = _instance(dials=Dials(concave_impact=True))
    even = evaluate(inst, Schedule(lots=((2, 2), (2, 2)))).temporary_impact_cost
    front = evaluate(inst, Schedule(lots=((4, 0), (4, 0)))).temporary_impact_cost
    assert front > even


# --- reporting -------------------------------------------------------------


def test_in_bps_rescales_every_term_against_notional():
    b = ObjectiveBreakdown(
        spread_cost=10.0, temporary_impact_cost=20.0,
        permanent_impact_cost=5.0, risk_cost=5.0,
    )
    bps = b.in_bps(notional=100_000.0)
    assert bps.spread_cost == pytest.approx(1.0)      # 10 / 1e5 * 1e4
    assert bps.total == pytest.approx(4.0)


def test_in_bps_rejects_a_non_positive_notional():
    b = ObjectiveBreakdown(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="notional"):
        b.in_bps(notional=0.0)


def test_evaluate_does_not_check_feasibility():
    """Scoring an infeasible schedule is how M3 calibrates penalty magnitudes.

    Conflating cost with feasibility is spec 6.5's named failure mode, so the
    two are deliberately separate calls.
    """
    inst = _instance()
    infeasible = Schedule(lots=((1, 1), (1, 1)))     # sells 2 of 4 on each asset
    assert np.isfinite(evaluate(inst, infeasible).total)


def test_evaluate_rejects_a_schedule_of_the_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        evaluate(_instance(), Schedule(lots=((1, 1, 1), (1, 1, 1))))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_objective.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.objective'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/problem/objective.py`:

```python
"""The reference objective evaluator -- spec section 5.2's four terms.

Terms are returned **separately**, not merely summed. M3's load-bearing
property test (spec section 10.1) asserts that a QUBO's energy equals the
directly evaluated objective; when that fails, a single scalar says nothing
about which term is wrong.

This is also the only place in the project where lots become shares. Spec
section 5.1 writes the decision variable in lots; everything in ``micro/`` is
in shares. The conversion is ``Asset.lot_size``, applied once, here.

``evaluate`` does **not** check feasibility. Scoring an infeasible schedule is
a legitimate thing to want -- it is how M3 calibrates penalty magnitudes --
and conflating the two is how an infeasible quantum solution gets compared
against a feasible classical one, which spec section 6.5 names as the error
that invalidates most published work in this area. Feasibility is
``problem.feasibility.classify``'s job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.micro.impact.almgren_chriss import AlmgrenChriss
from quantic.micro.impact.base import ImpactModel
from quantic.micro.impact.sqrt_law import PowerLawImpact
from quantic.problem.instance import Instance
from quantic.problem.schedule import Schedule

_BPS = 10_000.0


@dataclass(frozen=True, slots=True)
class ObjectiveBreakdown:
    """Spec section 5.2's four terms, each in currency."""

    spread_cost: float
    temporary_impact_cost: float
    permanent_impact_cost: float
    risk_cost: float

    @property
    def total(self) -> float:
        return (
            self.spread_cost
            + self.temporary_impact_cost
            + self.permanent_impact_cost
            + self.risk_cost
        )

    def in_bps(self, notional: float) -> ObjectiveBreakdown:
        """The same breakdown in basis points of notional.

        Spec section 9.1: the translation that makes results legible to
        practitioners rather than only to physicists.
        """
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        scale = _BPS / notional
        return ObjectiveBreakdown(
            spread_cost=self.spread_cost * scale,
            temporary_impact_cost=self.temporary_impact_cost * scale,
            permanent_impact_cost=self.permanent_impact_cost * scale,
            risk_cost=self.risk_cost * scale,
        )


def impact_model_for(instance: Instance, i: int) -> ImpactModel:
    """The temporary impact model dial D1 selects for asset ``i``.

    D1 on gives the calibrated concave power law, which is what makes the
    objective non-convex and the minimisation NP-hard. D1 off gives linear
    Almgren-Chriss, recovered by setting ``eta`` to the same coefficient so
    that only the exponent differs between the two arms.
    """
    a = instance.params.assets[i]
    if instance.dials.concave_impact:
        return PowerLawImpact(delta=a.delta, y_coef=a.y_coef, gamma=a.gamma)
    return AlmgrenChriss(eta=a.y_coef, gamma=a.gamma)


def evaluate(instance: Instance, schedule: Schedule) -> ObjectiveBreakdown:
    """Evaluate spec section 5.2's objective on ``schedule``. Currency units."""
    if (schedule.n_assets, schedule.n_buckets) != (instance.n_assets, instance.n_buckets):
        raise ValueError(
            f"schedule shape ({schedule.n_assets}, {schedule.n_buckets}) does not match "
            f"the instance ({instance.n_assets}, {instance.n_buckets})"
        )

    lot_sizes = instance.lot_sizes()[:, None]
    q = schedule.as_array() * lot_sizes                  # shares traded per bucket
    cum_before = schedule.cumulative_before() * lot_sizes  # shares traded strictly before
    h = instance.remaining_lots(schedule) * lot_sizes      # shares held after each bucket

    spread = 0.0
    temporary = 0.0
    permanent = 0.0

    for i in range(instance.n_assets):
        a = instance.params.assets[i]
        p = instance.params.impact_params(i)
        model = impact_model_for(instance, i)
        for t in range(instance.n_buckets):
            traded = float(q[i, t])
            if traded == 0.0:
                continue
            spread += a.half_spread * traded
            temporary += model.temporary_cost(traded, p) * traded * a.price
            permanent += (
                a.gamma
                * a.sigma_bucket
                * (float(cum_before[i, t]) / a.bucket_volume_shares)
                * traded
                * a.price
            )

    risk = _risk_cost(instance, h)

    return ObjectiveBreakdown(
        spread_cost=float(spread),
        temporary_impact_cost=float(temporary),
        permanent_impact_cost=float(permanent),
        risk_cost=float(risk),
    )


def _risk_cost(instance: Instance, holdings_shares: np.ndarray) -> float:
    """``lam * sum over t of h_t' Sigma_price h_t``, with ``h`` in shares.

    ``MarketParams.covariance`` is a **log-return** covariance and is
    dimensionless; ``price_covariance`` converts it so the quadratic form
    comes out in currency squared. Using the return covariance directly would
    be a silent error of ``price**2``.

    The bucket horizon is already inside the covariance, so spec section 5.2's
    ``tau`` does not appear.
    """
    lam = getattr(instance.risk, "lam", None)
    if lam is None:
        raise NotImplementedError(
            f"objective.evaluate does not know how to score risk spec "
            f"{instance.risk.name!r}. M2a implements VarianceRisk; CVaRRisk arrives "
            "with M2b"
        )
    if lam == 0.0:
        return 0.0
    sigma_price = instance.params.price_covariance()
    total = 0.0
    for t in range(holdings_shares.shape[1]):
        h_t = holdings_shares[:, t].astype(float)
        total += float(h_t @ sigma_price @ h_t)
    return lam * total
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_objective.py -q`
Expected: 20 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/objective.py tests/problem/test_objective.py
git commit -m "feat(problem): add the reference objective evaluator"
```

---

## Task 11: the tier ladder

**Files:**
- Create: `src/quantic/problem/liquidation.py`
- Test: `tests/problem/test_liquidation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TierSpec(name, n_assets, n_buckets, bits)` with `.max_lots_per_bucket`, `.approx_variables`, `.active_per_bucket`, `.buckets_per_asset_cap`, `.max_position_lots(discrete_participation)`; constants `T0`, `T1`, `T2`, `T3`, `LADDER`, and `by_name(name)`.

**The capacity arithmetic**, which Task 12 depends on and which the spec's self-review caught as wrong on every tier:

- With D2 on, each bucket hosts at most `active_per_bucket = ceil(n_assets / 2)` names, so total capacity is `n_buckets * active_per_bucket` asset-bucket slots.
- Shared across `n_assets` assets, each may use `buckets_per_asset_cap = max(1, (n_buckets * active_per_bucket) // n_assets)`.
- Hence `max_position_lots = buckets_per_asset_cap * max_lots_per_bucket`.

- [ ] **Step 1: Write the failing test**

Create `tests/problem/test_liquidation.py`:

```python
import math

import pytest

from quantic.problem.liquidation import LADDER, T0, T1, T2, T3, TierSpec, by_name


def test_the_ladder_reproduces_spec_5_5():
    assert [(t.name, t.n_assets, t.n_buckets, t.bits) for t in LADDER] == [
        ("T0", 3, 4, 2),
        ("T1", 5, 8, 3),
        ("T2", 8, 8, 3),
        ("T3", 30, 12, 4),
    ]


def test_max_lots_per_bucket_is_the_integer_range_the_bit_width_implies():
    assert T0.max_lots_per_bucket == 3       # 2**2 - 1
    assert T3.max_lots_per_bucket == 15      # 2**4 - 1


def test_approx_variables_tracks_spec_5_5s_sizing_column():
    """~24, ~120, ~192, ~1440 against the spec's 24 / 120-180 / 190-250 / 1500+."""
    assert T0.approx_variables == 24
    assert T1.approx_variables == 120
    assert T2.approx_variables == 192
    assert T3.approx_variables == 1440


def test_t2_sits_near_the_dense_embedding_ceiling():
    """Spec 3.2: Advantage2 fits ~230 dense logical variables. T2 is the edge tier."""
    assert 180 < T2.approx_variables < 260


def test_by_name_finds_a_tier():
    assert by_name("T2") is T2


def test_by_name_lists_the_ladder_when_asked_for_something_else():
    with pytest.raises(KeyError, match="T0"):
        by_name("T9")


# --- the capacity arithmetic Task 12 relies on ------------------------------


@pytest.mark.parametrize("tier", LADDER)
def test_the_concentrated_bound_fits_within_bucket_capacity(tier: TierSpec):
    """THE BOUND THE SPEC SELF-REVIEW CAUGHT.

    With D2 on, each asset may use `buckets_per_asset_cap` buckets and each
    bucket hosts at most `active_per_bucket` names. The demanded asset-bucket
    slots must not exceed capacity, or no feasible witness exists.

    The naive bound -- letting every asset use all T buckets -- fails on every
    tier: T0 demands 12 slots against a capacity of 8, T3 demands 360 against
    180.
    """
    demanded = tier.n_assets * tier.buckets_per_asset_cap
    capacity = tier.n_buckets * tier.active_per_bucket
    assert demanded <= capacity, (
        f"{tier.name}: {demanded} asset-bucket slots demanded but only {capacity} available"
    )


@pytest.mark.parametrize("tier", LADDER)
def test_the_naive_bound_would_not_have_fitted(tier: TierSpec):
    """Pins why the tightened bound exists, so nobody loosens it back."""
    naive = tier.n_assets * tier.n_buckets
    capacity = tier.n_buckets * tier.active_per_bucket
    assert naive > capacity


@pytest.mark.parametrize("tier", LADDER)
def test_active_per_bucket_actually_binds_cardinality(tier: TierSpec):
    """If it equalled n_assets, dial D2 would constrain nothing."""
    assert 1 <= tier.active_per_bucket < tier.n_assets


@pytest.mark.parametrize("tier", LADDER)
def test_every_asset_gets_at_least_one_bucket(tier: TierSpec):
    assert tier.buckets_per_asset_cap >= 1


@pytest.mark.parametrize("tier", LADDER)
def test_max_position_is_looser_when_d2_is_off(tier: TierSpec):
    assert tier.max_position_lots(discrete_participation=False) >= tier.max_position_lots(
        discrete_participation=True
    )


def test_a_tier_with_a_non_positive_dimension_is_rejected():
    with pytest.raises(ValueError, match="n_assets"):
        TierSpec(name="bad", n_assets=0, n_buckets=4, bits=2)
    with pytest.raises(ValueError, match="bits"):
        TierSpec(name="bad", n_assets=3, n_buckets=4, bits=0)


def test_tier_names_are_unique():
    assert len({t.name for t in LADDER}) == len(LADDER)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_liquidation.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.liquidation'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/problem/liquidation.py`:

```python
"""The instance ladder -- spec section 5.5.

Four tiers, sized against the hardware ceiling of spec section 3.2: the
liquidation QUBO is dense, and dense minor-embedding fits roughly 230 logical
variables on Advantage2. T2 is deliberately the edge tier and T3 is
hybrid-only by construction.

``bits`` is recorded here, where the ladder is defined in the spec's own
``N x T x B`` terms, and converted to ``max_lots_per_bucket`` for
:class:`~quantic.problem.instance.Instance`, which is encoding-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TierSpec:
    """One rung of the ladder, plus the capacity arithmetic the generator needs."""

    name: str
    n_assets: int
    n_buckets: int
    bits: int

    def __post_init__(self) -> None:
        if self.n_assets < 1:
            raise ValueError(f"{self.name}: n_assets must be at least 1, got {self.n_assets}")
        if self.n_buckets < 1:
            raise ValueError(f"{self.name}: n_buckets must be at least 1, got {self.n_buckets}")
        if self.bits < 1:
            raise ValueError(f"{self.name}: bits must be at least 1, got {self.bits}")

    @property
    def max_lots_per_bucket(self) -> int:
        """The integer range a ``bits``-wide binary expansion covers."""
        return 2**self.bits - 1

    @property
    def approx_variables(self) -> int:
        """``N * T * B`` -- spec section 5.5's sizing column, before auxiliaries."""
        return self.n_assets * self.n_buckets * self.bits

    @property
    def active_per_bucket(self) -> int:
        """How many names may trade in one bucket once dial D2 concentrates.

        Strictly below ``n_assets`` so that the cardinality derived from it
        actually binds. A cardinality equal to the asset count constrains
        nothing, and a dial that does not bind contributes nothing to the
        screening design in spec section 9.2 while still appearing in every
        results row as though it did.
        """
        return max(1, math.ceil(self.n_assets / 2))

    @property
    def buckets_per_asset_cap(self) -> int:
        """How many buckets one asset may spread across under D2.

        ``n_buckets * active_per_bucket`` asset-bucket slots exist in total,
        shared between ``n_assets`` assets. The naive alternative -- letting
        every asset use all ``n_buckets`` -- overruns capacity on **every**
        tier of this ladder, so no feasible witness would exist.
        """
        return max(1, (self.n_buckets * self.active_per_bucket) // self.n_assets)

    def max_position_lots(self, *, discrete_participation: bool) -> int:
        """The largest ``X[i]`` for which a feasible schedule provably exists."""
        buckets = self.buckets_per_asset_cap if discrete_participation else self.n_buckets
        return buckets * self.max_lots_per_bucket


T0 = TierSpec(name="T0", n_assets=3, n_buckets=4, bits=2)     # ~24 variables
T1 = TierSpec(name="T1", n_assets=5, n_buckets=8, bits=3)     # ~120
T2 = TierSpec(name="T2", n_assets=8, n_buckets=8, bits=3)     # ~192, embedding edge
T3 = TierSpec(name="T3", n_assets=30, n_buckets=12, bits=4)   # ~1440, hybrid-only

LADDER: tuple[TierSpec, ...] = (T0, T1, T2, T3)


def by_name(name: str) -> TierSpec:
    for tier in LADDER:
        if tier.name == name:
            return tier
    raise KeyError(f"unknown tier {name!r}; the ladder is {[t.name for t in LADDER]}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_liquidation.py -q`
Expected: 26 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/liquidation.py tests/problem/test_liquidation.py
git commit -m "feat(problem): add the four-tier instance ladder"
```

---

## Task 12: `generate` — feasible by construction

**Files:**
- Create: `src/quantic/problem/generator.py`
- Test: `tests/problem/test_generator.py`

**Interfaces:**
- Consumes: everything from Tasks 1–11.
- Produces: `InstanceGenerationError`; `generate(tier, dials, params, *, seed, lam=None) -> Instance`.

**The algorithm.** This is the hard part of M2a, and steps 4 and 5 are what make D2 real hardness rather than decoration.

1. Narrow `params` to the tier's asset count.
2. Draw `X[i]` in `[1, tier.max_position_lots(discrete_participation=...)]`.
3. Build a base schedule — spread across all buckets, or concentrated into `buckets_per_asset_cap` of them when D2 is on.
4. **Bind cardinality**: `k = max over t of active_names(t)` in the base schedule. The tightest `k` the witness satisfies.
5. **Bind minimum lot**: `min_lots[i] =` the smallest non-zero entry of row `i`. Again the tightest value the witness satisfies.
6. Carve blocks from non-zero cells if D4.
7. Auto-scale `lam` so the risk term equals the cost terms at the witness, unless given.
8. Assert `classify(instance, witness).feasible`.

- [ ] **Step 1: Write the failing test**

Create `tests/problem/test_generator.py`:

```python
import numpy as np
import pytest

from quantic.problem.constraints.block_trades import BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.generator import InstanceGenerationError, generate
from quantic.problem.liquidation import LADDER, T0, T1, T3
from quantic.problem.objective import evaluate
from quantic.problem.params import AssetParams, MarketParams

BUCKET_NS = 1800 * 1_000_000_000


def _params(n: int) -> MarketParams:
    rng = np.random.default_rng(0)
    assets = tuple(
        AssetParams(
            symbol=f"S{i:02d}", delta=0.5, y_coef=0.8, gamma=0.1,
            sigma_bucket=0.005, bucket_volume_shares=400_000.0,
            price=100.0 + i, half_spread=0.01,
        )
        for i in range(n)
    )
    a = rng.normal(size=(n, n))
    cov = (a @ a.T) / n * 1e-5           # PSD by construction
    return MarketParams(assets=assets, covariance=cov, bucket_ns=BUCKET_NS)


def _gen(tier, dials, seed=0, **kw):
    return generate(tier, dials, _params(tier.n_assets), seed=seed, **kw)


# --- the core promise ------------------------------------------------------


@pytest.mark.parametrize("tier", LADDER, ids=lambda t: t.name)
@pytest.mark.parametrize("dials", Dials.combinations(), ids=lambda d: d.label)
def test_the_witness_is_feasible_for_every_tier_and_dial_combination(tier, dials):
    """THE M2a PROMISE. If this fails, the done-criterion is unreachable."""
    inst = _gen(tier, dials)
    report = classify(inst, inst.witness)
    assert report.feasible, f"{tier.name}/{dials.label}: {dict(report.violations)}"


@pytest.mark.parametrize("tier", LADDER, ids=lambda t: t.name)
@pytest.mark.parametrize("dials", Dials.combinations(), ids=lambda d: d.label)
def test_every_generated_instance_evaluates_finitely(tier, dials):
    inst = _gen(tier, dials)
    assert np.isfinite(evaluate(inst, inst.witness).total)


def test_the_witness_fully_liquidates():
    inst = _gen(T1, Dials())
    assert [inst.witness.sold(i) for i in range(inst.n_assets)] == list(inst.initial_lots)


def test_no_witness_cell_exceeds_the_tier_range():
    inst = _gen(T3, Dials(discrete_participation=True))
    assert inst.witness.as_array().max() <= T3.max_lots_per_bucket


# --- dial wiring -----------------------------------------------------------


def test_full_liquidation_is_present_under_every_dial_combination():
    for dials in Dials.combinations():
        inst = _gen(T0, dials)
        assert any(isinstance(c, FullLiquidation) for c in inst.constraints)


def test_d2_off_adds_neither_cardinality_nor_minimum_lot():
    inst = _gen(T1, Dials())
    assert not any(isinstance(c, Cardinality | MinParticipation) for c in inst.constraints)


def test_d2_on_adds_both_constraints_together():
    """Spec 4: cardinality without a minimum lot is not meaningful on a desk."""
    inst = _gen(T1, Dials(discrete_participation=True))
    assert any(isinstance(c, Cardinality) for c in inst.constraints)
    assert any(isinstance(c, MinParticipation) for c in inst.constraints)


def test_d4_on_adds_block_trades():
    inst = _gen(T1, Dials(block_trades=True))
    assert any(isinstance(c, BlockTrades) for c in inst.constraints)


def test_d4_carves_at_least_one_block():
    inst = _gen(T1, Dials(block_trades=True))
    blocks = next(c for c in inst.constraints if isinstance(c, BlockTrades))
    assert len(blocks.blocks) >= 1


def test_d1_changes_no_constraint_only_the_objective():
    """D1 lives in the objective; a constraint difference would confound the design."""
    off = _gen(T1, Dials())
    on = _gen(T1, Dials(concave_impact=True))
    assert [c.describe() for c in off.constraints] == [c.describe() for c in on.constraints]


def test_the_dials_are_recorded_on_the_instance():
    d = Dials(concave_impact=True, block_trades=True)
    assert _gen(T1, d).dials == d


# --- the binding tests: a dial that does not bind is decoration -------------


@pytest.mark.parametrize("tier", LADDER, ids=lambda t: t.name)
def test_cardinality_binds_strictly_below_the_asset_count(tier):
    """k == n_assets constrains nothing, so D2 would contribute no hardness
    while still appearing in every results row as though it did."""
    inst = _gen(tier, Dials(discrete_participation=True))
    card = next(c for c in inst.constraints if isinstance(c, Cardinality))
    assert 1 <= card.k < tier.n_assets


@pytest.mark.parametrize("tier", LADDER, ids=lambda t: t.name)
def test_cardinality_is_tight_against_the_witness(tier):
    """Tightest k the witness satisfies: one lower and the witness would break."""
    inst = _gen(tier, Dials(discrete_participation=True))
    card = next(c for c in inst.constraints if isinstance(c, Cardinality))
    busiest = max(inst.witness.active_names(t) for t in range(inst.n_buckets))
    assert card.k == busiest
    assert Cardinality(k=card.k - 1).violation(inst.witness, inst.initial_lots) > 0


@pytest.mark.parametrize("tier", LADDER, ids=lambda t: t.name)
def test_the_minimum_lot_is_positive_and_tight(tier):
    inst = _gen(tier, Dials(discrete_participation=True))
    mp = next(c for c in inst.constraints if isinstance(c, MinParticipation))
    grid = inst.witness.as_array()
    for i, m in enumerate(mp.min_lots):
        assert m >= 1
        nonzero = grid[i][grid[i] > 0]
        if nonzero.size:
            assert m == int(nonzero.min())


# --- risk scaling ----------------------------------------------------------


def test_lam_is_auto_scaled_so_risk_balances_cost_at_the_witness():
    """An unscaled lam makes one side of the trade-off vanish, and a problem
    where risk or cost is irrelevant is not the problem being studied."""
    inst = _gen(T1, Dials())
    b = evaluate(inst, inst.witness)
    costs = b.spread_cost + b.temporary_impact_cost + b.permanent_impact_cost
    assert b.risk_cost == pytest.approx(costs, rel=1e-6)


def test_an_explicit_lam_is_respected():
    inst = _gen(T1, Dials(), lam=3e-7)
    assert inst.risk.lam == 3e-7


# --- determinism -----------------------------------------------------------


def test_the_same_seed_gives_an_identical_content_hash():
    assert _gen(T1, Dials(), seed=7).content_hash == _gen(T1, Dials(), seed=7).content_hash


def test_a_different_seed_gives_a_different_instance():
    assert _gen(T1, Dials(), seed=7).content_hash != _gen(T1, Dials(), seed=8).content_hash


def test_the_seed_is_recorded_on_the_instance():
    assert _gen(T1, Dials(), seed=11).seed == 11


def test_different_dials_give_different_instances_at_the_same_seed():
    a = _gen(T1, Dials(), seed=3)
    b = _gen(T1, Dials(discrete_participation=True), seed=3)
    assert a.content_hash != b.content_hash


# --- rejections ------------------------------------------------------------


def test_too_few_assets_in_params_is_rejected():
    with pytest.raises(InstanceGenerationError, match="30 assets|asset"):
        generate(T3, Dials(), _params(5), seed=0)


def test_surplus_assets_in_params_are_narrowed_to_the_tier():
    inst = generate(T0, Dials(), _params(10), seed=0)
    assert inst.n_assets == T0.n_assets
    assert inst.params.n_assets == T0.n_assets
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_generator.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.generator'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/problem/generator.py`:

```python
"""Generate instances that are feasible by construction.

Spec section 14 makes M2 done when every dial combination generates a *valid*
instance across every tier. Validity here means something a test can assert: a
feasible schedule provably exists, one is constructed, and it is attached to
the instance as its witness.

That witness is not decoration. It gives M3's decode/repair a known-good
target, M4 a guaranteed-feasible baseline to measure the optimality gap
against, and it means an infeasible result from any solver is unambiguously
the solver's fault rather than an empty feasible set.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from quantic.core.types import Asset
from quantic.problem.constraints.base import Constraint
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.liquidation import TierSpec
from quantic.problem.objective import evaluate
from quantic.problem.params import MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

# Fraction of the witness's non-zero cells carved into indivisible blocks
# under D4. A quarter leaves enough free cells that the block structure
# constrains without dictating the whole schedule.
_BLOCK_FRACTION = 0.25

_DEFAULT_LOT_SIZE = 100
_DEFAULT_TICK_SIZE = 0.01


class InstanceGenerationError(RuntimeError):
    """Raised when no feasible instance could be built for a tier and dial set."""


def _distribute(total: int, buckets: list[int], max_lots: int, width: int) -> list[int]:
    """Spread ``total`` lots as evenly as possible over ``buckets`` of a row of ``width``."""
    row = [0] * width
    n = len(buckets)
    base, remainder = divmod(total, n)
    for j, t in enumerate(buckets):
        value = base + (1 if j < remainder else 0)
        if value > max_lots:
            raise InstanceGenerationError(
                f"cannot place {total} lots in {n} bucket(s) without exceeding the "
                f"{max_lots}-lot range"
            )
        row[t] = value
    return row


def _spread_schedule(positions: np.ndarray, tier: TierSpec) -> Schedule:
    """Every asset trades in every bucket. Used when D2 is off."""
    all_buckets = list(range(tier.n_buckets))
    return Schedule(
        lots=tuple(
            tuple(
                _distribute(
                    int(x), all_buckets, tier.max_lots_per_bucket, tier.n_buckets
                )
            )
            for x in positions
        )
    )


def _concentrated_schedule(
    positions: np.ndarray, tier: TierSpec, rng: np.random.Generator
) -> Schedule:
    """Each asset trades in a subset of buckets, so cardinality can bind.

    Assets are placed largest-need-first into the least-loaded buckets. The
    assignment exists because ``TierSpec`` caps each asset's bucket count so
    that total demand never exceeds ``n_buckets * active_per_bucket``.
    """
    max_lots = tier.max_lots_per_bucket
    need = [max(1, math.ceil(int(x) / max_lots)) for x in positions]
    load = [0] * tier.n_buckets
    rows: list[tuple[int, ...]] = [()] * tier.n_assets

    # Largest need first: the tightest assets get the pick of the capacity.
    # Ties broken by index so the result depends only on the seed via
    # `positions`, never on dict or set ordering.
    order = sorted(range(tier.n_assets), key=lambda i: (-need[i], i))
    for i in order:
        available = [t for t in range(tier.n_buckets) if load[t] < tier.active_per_bucket]
        if len(available) < need[i]:
            raise InstanceGenerationError(
                f"{tier.name}: asset {i} needs {need[i]} bucket(s) but only "
                f"{len(available)} have capacity below {tier.active_per_bucket} names. "
                "TierSpec.buckets_per_asset_cap should have prevented this"
            )
        chosen = sorted(sorted(available, key=lambda t: (load[t], t))[: need[i]])
        for t in chosen:
            load[t] += 1
        rows[i] = tuple(
            _distribute(int(positions[i]), chosen, max_lots, tier.n_buckets)
        )

    del rng  # placement is deterministic given the positions; kept for symmetry
    return Schedule(lots=tuple(rows))


def _carve_blocks(witness: Schedule, rng: np.random.Generator) -> tuple[Block, ...]:
    """Declare a subset of the witness's non-zero cells indivisible."""
    grid = witness.as_array()
    cells = [
        (i, t)
        for i in range(witness.n_assets)
        for t in range(witness.n_buckets)
        if grid[i, t] > 0
    ]
    if not cells:
        raise InstanceGenerationError("witness has no non-zero cell to carve a block from")
    count = max(1, int(len(cells) * _BLOCK_FRACTION))
    picked = rng.choice(len(cells), size=count, replace=False)
    return tuple(
        Block(asset=cells[int(j)][0], bucket=cells[int(j)][1], lots=int(grid[cells[int(j)]]))
        for j in sorted(int(j) for j in picked)
    )


def _balanced_lam(instance: Instance) -> float:
    """``lam`` making the risk term equal the cost terms at the witness.

    An arbitrary ``lam`` makes one side of the trade-off vanish -- either risk
    swamps execution cost or it is invisible -- and a problem where one side
    is irrelevant is not the problem the crossover study is about. M4 sweeps
    ``lam`` around this point.
    """
    zero = dataclasses.replace(instance, risk=VarianceRisk(lam=0.0))
    costs = evaluate(zero, instance.witness).total
    unit = dataclasses.replace(instance, risk=VarianceRisk(lam=1.0))
    raw_risk = evaluate(unit, instance.witness).risk_cost
    if raw_risk <= 0.0:
        return 0.0
    return costs / raw_risk


def generate(
    tier: TierSpec,
    dials: Dials,
    params: MarketParams,
    *,
    seed: int,
    lam: float | None = None,
) -> Instance:
    """Build one feasible instance for ``tier`` under ``dials``.

    ``lam`` defaults to the value that balances risk against execution cost at
    the witness. ``params`` may describe more assets than the tier needs, in
    which case the leading ones are used.
    """
    if params.n_assets < tier.n_assets:
        raise InstanceGenerationError(
            f"{tier.name} needs {tier.n_assets} assets but MarketParams describes "
            f"{params.n_assets}"
        )
    params = params.subset(tier.n_assets)

    # Seeded from the full configuration so that two tiers, or two dial
    # combinations, never share a position draw at the same seed.
    rng = np.random.default_rng(
        [seed, tier.n_assets, tier.n_buckets, tier.bits, _dial_code(dials)]
    )

    ceiling = tier.max_position_lots(discrete_participation=dials.discrete_participation)
    positions = rng.integers(1, ceiling + 1, size=tier.n_assets)

    if dials.discrete_participation:
        witness = _concentrated_schedule(positions, tier, rng)
    else:
        witness = _spread_schedule(positions, tier)

    constraints: list[Constraint] = [FullLiquidation()]

    if dials.discrete_participation:
        # Tightest k the witness satisfies. A looser one would make D2 a
        # no-op dial that still shows up in every results row.
        busiest = max(witness.active_names(t) for t in range(tier.n_buckets))
        constraints.append(Cardinality(k=busiest))

        grid = witness.as_array()
        min_lots = []
        for i in range(tier.n_assets):
            nonzero = grid[i][grid[i] > 0]
            min_lots.append(int(nonzero.min()) if nonzero.size else 1)
        constraints.append(MinParticipation(min_lots=tuple(min_lots)))

    if dials.block_trades:
        constraints.append(BlockTrades(blocks=_carve_blocks(witness, rng)))

    assets = tuple(
        Asset(symbol=a.symbol, lot_size=_DEFAULT_LOT_SIZE, tick_size=_DEFAULT_TICK_SIZE)
        for a in params.assets
    )

    instance = Instance(
        instance_id=f"{tier.name}-{dials.label}-{seed}",
        tier=tier.name,
        assets=assets,
        n_buckets=tier.n_buckets,
        initial_lots=tuple(int(x) for x in positions),
        max_lots_per_bucket=tier.max_lots_per_bucket,
        params=params,
        dials=dials,
        constraints=tuple(constraints),
        risk=VarianceRisk(lam=0.0),
        witness=witness,
        seed=seed,
    )

    chosen_lam = _balanced_lam(instance) if lam is None else lam
    instance = dataclasses.replace(instance, risk=VarianceRisk(lam=chosen_lam))

    report = classify(instance, witness)
    if not report.feasible:
        raise InstanceGenerationError(
            f"{tier.name}/{dials.label}: the constructed witness is infeasible, which "
            f"means the construction is wrong rather than the instance hard. "
            f"Violations: {dict(report.violations)}"
        )
    return instance


def _dial_code(dials: Dials) -> int:
    """A small integer identifying the dial combination, for seeding."""
    return (
        int(dials.concave_impact)
        | int(dials.discrete_participation) << 1
        | int(dials.cvar_risk) << 2
        | int(dials.block_trades) << 3
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_generator.py -q`
Expected: 87 passed (the two parametrised gates contribute 32 each)

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/generator.py tests/problem/test_generator.py
git commit -m "feat(problem): generate instances that are feasible by construction"
```

---

## Task 13: `generate_ladder` and the M2a gate

**Files:**
- Modify: `src/quantic/problem/generator.py` (append `generate_ladder`)
- Test: `tests/problem/test_m2a_gate.py`

**Interfaces:**
- Consumes: `generate` (12), `LADDER` (11), `Dials.combinations` (2).
- Produces: `generate_ladder(params_for, *, seed, include_cvar=False) -> tuple[Instance, ...]`.

- [ ] **Step 1: Write the failing test**

Create `tests/problem/test_m2a_gate.py`:

```python
"""THE M2a GATE -- spec section 14's done-criterion, asserted.

8 dial combinations x 4 tiers = 32 instances, every one carrying a feasible
witness and evaluating to a finite objective.
"""

import numpy as np
import pytest

from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.generator import generate_ladder
from quantic.problem.liquidation import LADDER
from quantic.problem.objective import evaluate
from quantic.problem.params import AssetParams, MarketParams

BUCKET_NS = 1800 * 1_000_000_000


def _params_for(tier):
    rng = np.random.default_rng(tier.n_assets)
    assets = tuple(
        AssetParams(
            symbol=f"S{i:02d}", delta=0.5, y_coef=0.8, gamma=0.1,
            sigma_bucket=0.005, bucket_volume_shares=400_000.0,
            price=100.0 + i, half_spread=0.01,
        )
        for i in range(tier.n_assets)
    )
    a = rng.normal(size=(tier.n_assets, tier.n_assets))
    cov = (a @ a.T) / tier.n_assets * 1e-5
    return MarketParams(assets=assets, covariance=cov, bucket_ns=BUCKET_NS)


@pytest.fixture(scope="module")
def ladder():
    return generate_ladder(_params_for, seed=2026)


def test_the_ladder_has_eight_combinations_on_each_of_four_tiers(ladder):
    assert len(ladder) == 8 * 4 == 32


def test_every_tier_and_dial_pair_appears_exactly_once(ladder):
    seen = {(i.tier, i.dials.label) for i in ladder}
    expected = {(t.name, d.label) for t in LADDER for d in Dials.combinations()}
    assert seen == expected
    assert len(ladder) == len(seen)


def test_every_witness_is_feasible(ladder):
    """The gate. An empty feasible set anywhere makes the study unmeasurable."""
    broken = {
        i.instance_id: dict(classify(i, i.witness).violations)
        for i in ladder
        if not classify(i, i.witness).feasible
    }
    assert not broken, broken


def test_every_instance_evaluates_to_a_finite_positive_objective(ladder):
    for inst in ladder:
        total = evaluate(inst, inst.witness).total
        assert np.isfinite(total), inst.instance_id
        assert total > 0.0, inst.instance_id


def test_every_instance_reports_a_positive_notional(ladder):
    for inst in ladder:
        assert inst.notional() > 0.0, inst.instance_id


def test_content_hashes_are_unique_across_the_ladder(ladder):
    """A collision would merge two cells of the experiment design."""
    hashes = [i.content_hash for i in ladder]
    assert len(set(hashes)) == len(hashes)


def test_instance_ids_are_unique_across_the_ladder(ladder):
    ids = [i.instance_id for i in ladder]
    assert len(set(ids)) == len(ids)


def test_no_cvar_instance_is_generated_by_default(ladder):
    """D3 is M2b. Generating it now would claim coverage M2a does not have."""
    assert all(not i.dials.cvar_risk for i in ladder)


def test_including_cvar_doubles_the_ladder():
    """The M2b switch. One flag, no other change."""
    full = generate_ladder(_params_for, seed=2026, include_cvar=True)
    assert len(full) == 64
    assert sum(i.dials.cvar_risk for i in full) == 32


def test_the_ladder_is_reproducible_from_its_seed(ladder):
    again = generate_ladder(_params_for, seed=2026)
    assert [i.content_hash for i in again] == [i.content_hash for i in ladder]


def test_a_different_seed_changes_every_instance(ladder):
    other = generate_ladder(_params_for, seed=7)
    assert not (
        {i.content_hash for i in other} & {i.content_hash for i in ladder}
    )


def test_variable_counts_climb_across_the_tiers(ladder):
    """Spec 5.5's ladder is a ramp; if it flattened, the study has no x-axis."""
    by_tier = {}
    for inst in ladder:
        by_tier[inst.tier] = inst.n_assets * inst.n_buckets
    ordered = [by_tier[t.name] for t in LADDER]
    assert ordered == sorted(ordered)
    assert ordered[0] < ordered[-1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_m2a_gate.py -q`
Expected: `ImportError: cannot import name 'generate_ladder'`

- [ ] **Step 3: Append `generate_ladder` to `generator.py`**

Add to the imports at the top of `src/quantic/problem/generator.py`:

```python
from collections.abc import Callable

from quantic.problem.liquidation import LADDER, TierSpec
```

(replacing the existing `from quantic.problem.liquidation import TierSpec`)

Append at the end of the file:

```python
def generate_ladder(
    params_for: Callable[[TierSpec], MarketParams],
    *,
    seed: int,
    include_cvar: bool = False,
) -> tuple[Instance, ...]:
    """Every tier crossed with every dial combination.

    ``include_cvar=False`` gives M2a's 32 instances. Once M2b lands, flipping
    it to ``True`` gives all 64 with no other change -- which is the whole
    point of ``RiskSpec`` being a variant point.

    ``params_for`` is a callable rather than a single ``MarketParams`` because
    each tier needs a different asset count, from 3 at T0 to 30 at T3.
    """
    out: list[Instance] = []
    for tier in LADDER:
        params = params_for(tier)
        for dials in Dials.combinations(include_cvar=include_cvar):
            out.append(generate(tier, dials, params, seed=seed))
    return tuple(out)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_m2a_gate.py -q`
Expected: 12 passed

- [ ] **Step 5: Run the whole suite and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/quantic/problem/generator.py tests/problem/test_m2a_gate.py
git commit -m "feat(problem): add generate_ladder and the M2a gate"
```

---

## Task 14: `market_params_from_bundle`

**Files:**
- Create: `src/quantic/problem/from_bundle.py`
- Test: `tests/problem/test_from_bundle.py`

**Interfaces:**
- Consumes: `DatasetBundle`, `TradingSession`, `calibrate_bundle`, `estimate_covariance`, `bucket_spreads`, `MarketParams` (3).
- Produces: `market_params_from_bundle(bundle, *, session, bucket_ns, symbols=None, gamma=None, spread_boundary=Boundary.OPENING, assume_own_participation=False) -> MarketParams`.

This is the **only** module in `problem/` that imports `micro/`. Keeping it alone means the generator stays deterministic and testable while calibration failures surface here, where a human can see them.

- [ ] **Step 1: Write the failing test**

Create `tests/problem/test_from_bundle.py`:

```python
import numpy as np
import pytest

from quantic.core.session import SYNTH_SESSION, Boundary
from quantic.data.synth import SynthConfig, generate_bundle
from quantic.micro.impact.calibrate import CalibrationError
from quantic.problem.from_bundle import market_params_from_bundle
from quantic.problem.params import MarketParams

BUCKET_NS = SYNTH_SESSION.length_ns // 13

CFG = SynthConfig(
    symbols=("SYNA", "SYNB", "SYNC"), n_days=40, buckets_per_day=13,
    seed=17, base_price=1000.0, noise_frac=0.05, depth_levels=4,
)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return generate_bundle(tmp_path_factory.mktemp("fb") / "b", CFG)


def _build(bundle, **kw):
    base = dict(
        session=SYNTH_SESSION,
        bucket_ns=BUCKET_NS,
        spread_boundary=Boundary.CLOSING,   # the generator stamps L1 at bucket end
        assume_own_participation=True,
    )
    base.update(kw)
    return market_params_from_bundle(bundle, **base)


def test_it_builds_market_params_for_every_symbol(bundle):
    params = _build(bundle)
    assert isinstance(params, MarketParams)
    assert params.symbols == ("SYNA", "SYNB", "SYNC")
    assert params.bucket_ns == BUCKET_NS


def test_the_calibrated_exponent_reaches_the_params(bundle):
    """The M1-to-M2 contract, exercised end to end."""
    gt = bundle.manifest.extra["ground_truth"]
    params = _build(bundle)
    for a in params.assets:
        assert a.delta == pytest.approx(gt["impact_delta"][a.symbol], abs=0.05)


def test_the_covariance_is_rescaled_to_the_bucket_horizon(bundle):
    """estimate_covariance is annualised; the risk term needs per-bucket."""
    from quantic.micro.covariance import estimate_covariance

    annual = estimate_covariance(bundle.daily())
    params = _build(bundle)
    ratio = BUCKET_NS / SYNTH_SESSION.length_ns / 252.0
    assert np.allclose(params.covariance, annual.matrix * ratio, rtol=1e-9)


def test_the_covariance_rows_are_ordered_to_match_the_assets(bundle):
    """A mismatched ordering would misprice every cross-asset risk term."""
    params = _build(bundle, symbols=("SYNC", "SYNA"))
    assert params.symbols == ("SYNC", "SYNA")
    assert params.covariance.shape == (2, 2)
    # Diagonal entries must follow the requested order, not the bundle's.
    full = _build(bundle)
    idx = {s: i for i, s in enumerate(full.symbols)}
    assert params.covariance[0, 0] == pytest.approx(full.covariance[idx["SYNC"], idx["SYNC"]])
    assert params.covariance[1, 1] == pytest.approx(full.covariance[idx["SYNA"], idx["SYNA"]])


def test_half_spread_and_price_come_from_l1(bundle):
    params = _build(bundle)
    for a in params.assets:
        assert a.half_spread > 0.0
        assert a.price == pytest.approx(CFG.base_price, rel=0.5)


def test_bucket_volume_is_adv_divided_by_the_buckets_in_a_session(bundle):
    params = _build(bundle)
    for a in params.assets:
        assert a.bucket_volume_shares == pytest.approx(
            CFG.adv_shares / CFG.buckets_per_day, rel=0.4
        )


def test_gamma_defaults_to_zero_and_can_be_supplied(bundle):
    assert all(a.gamma == 0.0 for a in _build(bundle).assets)
    supplied = _build(bundle, gamma={"SYNA": 0.2, "SYNB": 0.3, "SYNC": 0.4})
    assert [a.gamma for a in supplied.assets] == [0.2, 0.3, 0.4]


def test_a_subset_of_symbols_can_be_requested(bundle):
    assert _build(bundle, symbols=("SYNB",)).symbols == ("SYNB",)


def test_an_unknown_symbol_is_named(bundle):
    with pytest.raises(KeyError, match="ZZZ"):
        _build(bundle, symbols=("ZZZ",))


def test_the_identification_assumption_is_not_assumed_by_default(bundle):
    """M2's objective evaluates own participation; the fit is on net imbalance.

    Defaulting this to True would bury the assumption at exactly the point a
    human should be making it.
    """
    with pytest.raises(CalibrationError, match="own participation|identification"):
        market_params_from_bundle(
            bundle, session=SYNTH_SESSION, bucket_ns=BUCKET_NS,
            spread_boundary=Boundary.CLOSING,
        )


def test_the_result_feeds_the_generator(bundle):
    """The point of the adapter: real parameters into the ladder."""
    from quantic.problem.dials import Dials
    from quantic.problem.feasibility import classify
    from quantic.problem.generator import generate
    from quantic.problem.liquidation import T0

    params = _build(bundle)
    inst = generate(T0, Dials(concave_impact=True), params, seed=0)
    assert classify(inst, inst.witness).feasible
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_from_bundle.py -q`
Expected: `ModuleNotFoundError: No module named 'quantic.problem.from_bundle'`

- [ ] **Step 3: Write the implementation**

Create `src/quantic/problem/from_bundle.py`:

```python
"""Build :class:`MarketParams` from a :class:`DatasetBundle`.

The only module in ``problem/`` that imports ``micro/``, and deliberately
separate from the generator. Measured on 2026-09-24, calibration **raises**
for 8 of 25 symbols at spec section 8.4's 20-day data shape and returns a
median delta error of 0.34 for the rest. Calling it inside instance
generation would make a third of the ladder fail to generate for reasons that
have nothing to do with the problem model, and would stop instance identity
being reproducible from a seed alone.

Here, a calibration failure surfaces where a human asked for real data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from quantic.core.session import Boundary, TradingSession
from quantic.data.bundle import DatasetBundle
from quantic.micro.books import bucket_spreads
from quantic.micro.covariance import estimate_covariance
from quantic.micro.impact.calibrate import calibrate_bundle
from quantic.micro.liquidity import adv
from quantic.problem.params import AssetParams, MarketParams


def market_params_from_bundle(
    bundle: DatasetBundle,
    *,
    session: TradingSession,
    bucket_ns: int,
    symbols: Sequence[str] | None = None,
    gamma: Mapping[str, float] | None = None,
    spread_boundary: Boundary = Boundary.OPENING,
    assume_own_participation: bool = False,
) -> MarketParams:
    """Compose M1's estimators into the M2 parameter set.

    ``assume_own_participation`` is passed straight through to
    ``CalibrationResult.to_model`` and is **not** defaulted to ``True``: the
    M2 objective evaluates our own participation while the fit was made on
    aggregate net order-flow imbalance, and that identification assumption
    belongs at the call site where a human chose it.

    ``spread_boundary`` defaults to ``OPENING``, which is right for real L1.
    The synthetic generator stamps one quote per bucket at that bucket's end
    and needs ``CLOSING``.
    """
    buckets_per_day = session.buckets_per_day(bucket_ns)

    calibrations = calibrate_bundle(bundle, session=session, bucket_ns=bucket_ns)
    wanted = tuple(symbols) if symbols is not None else tuple(sorted(calibrations))
    missing = [s for s in wanted if s not in calibrations]
    if missing:
        raise KeyError(
            f"no calibration for symbol(s) {missing}; the bundle calibrated "
            f"{sorted(calibrations)}"
        )

    # to_model() enforces the participation-basis contract. Its return value is
    # discarded -- delta and y_coef are read from the result -- but the check
    # is the point.
    for s in wanted:
        calibrations[s].to_model(assume_own_participation=assume_own_participation)

    spreads = (
        bucket_spreads(
            bundle.l1(), session=session, bucket_ns=bucket_ns, boundary=spread_boundary
        )
        .group_by("symbol")
        .agg(pl.col("half_spread").mean(), pl.col("mid").mean())
    )
    spread_by_symbol = {
        row["symbol"]: (row["half_spread"], row["mid"])
        for row in spreads.iter_rows(named=True)
    }

    adv_by_symbol = {
        row["symbol"]: row["adv"]
        for row in adv(bundle.daily()).group_by("symbol").agg(pl.col("adv").last()).iter_rows(
            named=True
        )
    }

    assets: list[AssetParams] = []
    for symbol in wanted:
        if symbol not in spread_by_symbol:
            raise KeyError(f"no L1 spread observations for {symbol}")
        if symbol not in adv_by_symbol:
            raise KeyError(f"no daily bars for {symbol}")
        half_spread, mid = spread_by_symbol[symbol]
        result = calibrations[symbol]
        assets.append(
            AssetParams(
                symbol=symbol,
                delta=result.delta,
                y_coef=result.y_coef,
                gamma=float(gamma[symbol]) if gamma else 0.0,
                sigma_bucket=result.sigma,
                bucket_volume_shares=float(adv_by_symbol[symbol]) / buckets_per_day,
                price=float(mid),
                half_spread=float(half_spread),
            )
        )

    return MarketParams(
        assets=tuple(assets),
        covariance=_bucket_covariance(bundle, session, bucket_ns, wanted),
        bucket_ns=bucket_ns,
    )


def _bucket_covariance(
    bundle: DatasetBundle,
    session: TradingSession,
    bucket_ns: int,
    wanted: tuple[str, ...],
) -> np.ndarray:
    """Annualised covariance rescaled to one bucket, reordered to ``wanted``.

    ``estimate_covariance`` returns its own symbol ordering. Reindexing here
    rather than assuming the two agree is what stops a silently transposed
    risk term.
    """
    estimate = estimate_covariance(bundle.daily())
    # at_horizon takes a number of trading days; one bucket is a fraction of one.
    bucket_fraction_of_a_day = bucket_ns / session.length_ns
    rescaled = estimate.at_horizon(days=bucket_fraction_of_a_day)

    index = {s: i for i, s in enumerate(estimate.symbols)}
    missing = [s for s in wanted if s not in index]
    if missing:
        raise KeyError(f"no daily returns for symbol(s) {missing}")
    order = [index[s] for s in wanted]
    return np.asarray(rescaled.matrix, dtype=float)[np.ix_(order, order)]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_from_bundle.py -q`
Expected: 11 passed

- [ ] **Step 5: Final verification**

```bash
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
.venv/Scripts/python.exe -m pytest -q
```

Expected: ruff clean, `Contracts: 1 kept, 0 broken.`, whole suite green.

- [ ] **Step 6: Commit**

```bash
git add src/quantic/problem/from_bundle.py tests/problem/test_from_bundle.py
git commit -m "feat(problem): build MarketParams from a dataset bundle"
```

---

## Task 15: cross-cutting property tests

**Files:**
- Test: `tests/problem/test_properties.py`

**Interfaces:**
- Consumes: everything. Adds no production code.

Spec §9 calls for Hypothesis property tests. The parametrised tests in Tasks 12 and 13 cover the tier × dial cross product exhaustively, which is stronger than sampling for *that* axis — but they only ever evaluate the **witness**. These properties must hold for an arbitrary schedule, including deeply infeasible ones, because that is exactly what a solver hands back and what M3's penalty calibration scores.

- [ ] **Step 1: Write the failing test**

Create `tests/problem/test_properties.py`:

```python
"""Properties that must hold for ANY schedule, not just the witness.

The parametrised gates in test_generator and test_m2a_gate cover the tier x
dial cross product exhaustively, but only ever evaluate the witness. A solver
returns arbitrary schedules, and M3 deliberately scores infeasible ones while
calibrating penalty magnitudes, so the invariants below are exercised over
generated grids instead.
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.generator import generate
from quantic.problem.liquidation import T0
from quantic.problem.objective import evaluate
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
MAX_LOTS = 6


@st.composite
def grids(draw, n_assets=None, n_buckets=None):
    """An arbitrary non-negative lot grid."""
    rows = n_assets if n_assets is not None else draw(st.integers(1, 4))
    cols = n_buckets if n_buckets is not None else draw(st.integers(1, 4))
    flat = draw(
        st.lists(
            st.integers(0, MAX_LOTS), min_size=rows * cols, max_size=rows * cols
        )
    )
    return Schedule(
        lots=tuple(tuple(flat[r * cols : (r + 1) * cols]) for r in range(rows))
    )


def _params(n: int) -> MarketParams:
    assets = tuple(
        AssetParams(
            symbol=f"S{i:02d}", delta=0.5, y_coef=0.8, gamma=0.1,
            sigma_bucket=0.005, bucket_volume_shares=400_000.0,
            price=100.0 + i, half_spread=0.01,
        )
        for i in range(n)
    )
    return MarketParams(assets=assets, covariance=np.eye(n) * 1e-5, bucket_ns=BUCKET_NS)


# --- Schedule --------------------------------------------------------------


@given(grids())
def test_the_two_cumulative_conventions_always_differ_by_the_schedule(s):
    assert (s.cumulative() - s.cumulative_before()).tolist() == s.as_array().tolist()


@given(grids())
def test_cumulative_before_always_starts_at_zero(s):
    assert s.cumulative_before()[:, 0].tolist() == [0] * s.n_assets


@given(grids())
def test_a_schedule_round_trips_through_its_array(s):
    assert Schedule.from_array(s.as_array()) == s


@given(grids())
def test_active_names_never_exceeds_the_asset_count(s):
    assert all(0 <= s.active_names(t) <= s.n_assets for t in range(s.n_buckets))


# --- constraints -----------------------------------------------------------


@given(grids(n_assets=3, n_buckets=3), st.lists(st.integers(0, 12), min_size=3, max_size=3))
def test_violation_is_zero_exactly_when_satisfied_for_every_constraint(s, positions):
    x = tuple(positions)
    constraints = [
        FullLiquidation(),
        Cardinality(k=2),
        MinParticipation(min_lots=(2, 3, 2)),
        BlockTrades(blocks=(Block(asset=0, bucket=0, lots=4),)),
    ]
    for c in constraints:
        assert (c.violation(s, x) == 0.0) == c.is_satisfied(s, x), c.name


@given(grids(n_assets=3, n_buckets=3), st.lists(st.integers(0, 12), min_size=3, max_size=3))
def test_violations_are_always_non_negative_and_finite(s, positions):
    x = tuple(positions)
    for c in (
        FullLiquidation(),
        Cardinality(k=2),
        MinParticipation(min_lots=(2, 3, 2)),
        BlockTrades(blocks=(Block(asset=0, bucket=0, lots=4),)),
    ):
        v = c.violation(s, x)
        assert v >= 0.0 and np.isfinite(v), c.name


@given(grids(n_assets=3, n_buckets=3), st.integers(1, 3), st.integers(1, 3))
def test_a_tighter_cardinality_never_reports_a_smaller_violation(s, a, b):
    """Monotone in k: tightening a constraint cannot make a schedule more feasible."""
    tight, loose = min(a, b), max(a, b)
    x = (0, 0, 0)
    assert Cardinality(k=tight).violation(s, x) >= Cardinality(k=loose).violation(s, x)


@given(grids(n_assets=3, n_buckets=3), st.integers(1, 6), st.integers(1, 6))
def test_a_higher_minimum_lot_never_reports_a_smaller_violation(s, a, b):
    tight, loose = max(a, b), min(a, b)
    x = (0, 0, 0)
    assert MinParticipation(min_lots=(tight,) * 3).violation(s, x) >= MinParticipation(
        min_lots=(loose,) * 3
    ).violation(s, x)


# --- objective and feasibility, against a real instance --------------------


@settings(max_examples=50, deadline=None)
@given(grids(n_assets=T0.n_assets, n_buckets=T0.n_buckets))
def test_the_breakdown_always_sums_to_the_total(s):
    inst = generate(T0, Dials(concave_impact=True), _params(T0.n_assets), seed=0)
    assume(s.as_array().max() <= inst.max_lots_per_bucket)
    b = evaluate(inst, s)
    assert b.total == float(
        b.spread_cost + b.temporary_impact_cost + b.permanent_impact_cost + b.risk_cost
    )


@settings(max_examples=50, deadline=None)
@given(grids(n_assets=T0.n_assets, n_buckets=T0.n_buckets))
def test_every_objective_term_is_finite_and_non_negative_for_any_schedule(s):
    """M3 scores infeasible schedules while calibrating penalties; none may be NaN."""
    inst = generate(T0, Dials(concave_impact=True), _params(T0.n_assets), seed=0)
    assume(s.as_array().max() <= inst.max_lots_per_bucket)
    b = evaluate(inst, s)
    for value in (
        b.spread_cost, b.temporary_impact_cost, b.permanent_impact_cost, b.risk_cost
    ):
        assert np.isfinite(value) and value >= 0.0


@settings(max_examples=50, deadline=None)
@given(grids(n_assets=T0.n_assets, n_buckets=T0.n_buckets))
def test_the_total_violation_always_equals_the_sum_of_its_parts(s):
    inst = generate(T0, Dials(discrete_participation=True), _params(T0.n_assets), seed=0)
    assume(s.as_array().max() <= inst.max_lots_per_bucket)
    report = classify(inst, s)
    assert report.total_violation == float(sum(report.violations.values()))
    assert report.feasible == (report.total_violation == 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/problem/test_properties.py -q`

Expected: failures only if an invariant is genuinely broken. If everything from Tasks 1–14 is correct these pass immediately — which is fine here, because this task adds no production code. Its value is the coverage of arbitrary schedules, which nothing else exercises. **If any of these fail, the bug is in the task that owns the invariant, not here** — fix it there and re-run.

- [ ] **Step 3: Run the whole suite**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/lint-imports.exe
.venv/Scripts/python.exe -m ruff check src tests
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 4: Commit**

```bash
git add tests/problem/test_properties.py
git commit -m "test(problem): add cross-cutting property tests over arbitrary schedules"
```

---

## M2a is complete at this point

Spec §14's done-criterion is met and asserted: 8 dial combinations × 4 tiers = 32 instances generate, each with a provably feasible witness and a finite objective.

## What M2b picks up

`CVaRRisk` as a second `RiskSpec` variant, a branch in `objective._risk_cost` (which already raises `NotImplementedError` naming the unhandled spec), and the scenario machinery behind it: ~2000 price and liquidity paths, clustering to 20–30 weighted representatives, and the Wasserstein reduction error reported alongside every CVaR-enabled result (spec §5.4). `generate_ladder(include_cvar=True)` then yields all 64 and the gate doubles.

## What M3 picks up

`Instance` is the whole interface. M3's encoder pattern-matches on `instance.constraints` to quadratise — `Cardinality` via binary slack, `MinParticipation` via spec §6.2's `x = m·y + Σ2^b·z` with penalty `Σ z·(1−y)`, `BlockTrades` via one binary per block — and derives bit widths from `max_lots_per_bucket`. The load-bearing property test of spec §10.1, that QUBO energy equals the directly evaluated objective, compares against `problem.objective.evaluate`, which is why that function returns its four terms separately.
