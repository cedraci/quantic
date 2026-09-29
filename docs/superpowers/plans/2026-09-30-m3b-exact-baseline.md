# Quantic M3b: The Exact Baseline and the Parent Gate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the encoder against exact optima — on all 16 T0 dial combinations, under all three encodings, the QUBO's minimum decodes to a feasible schedule whose true objective is the optimum two independent exact methods agree on.

**Architecture:** Three exact methods behind one `Result` type. `brute_force` enumerates the lot grid. `exact_pwl_milp` solves a mixed-integer linear program through HiGHS, with one binary per admissible lot value so that any per-cell cost is exactly linear, and outer approximation for the variance term. `qubo_minimum` enumerates the bits outside the CVaR gadget in two halves and minimises the gadget in closed form. The first two share the lot grid and the evaluator and nothing else; the third reads an `EncodedInstance`.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.optimize.milp`, which is HiGHS), pytest, ruff, import-linter. **No new dependencies.**

**Spec:** `docs/superpowers/specs/2026-09-29-m3b-exact-baseline-design.md`. Read it with `docs/superpowers/specs/2026-09-29-m3a-gate-result.md`, which records why `cheapest_aux` and not `optimal_aux`.

**This plan was dry-run.** Every code block below was copied from files that were run, in a throwaway worktree, against the repository at `a88c3f4`: 239 tests under `tests/solvers/`, the full suite at 1180 passed and 1 skipped, ruff clean, three import contracts kept. The dry run found two defects a plan written from the design alone would have carried — an outer-approximation loop that did not end, and a HiGHS presolve failure — and both fixes are in the code below. Where a step says what to expect, that is what was seen.

## Global Constraints

- **Python `>=3.11`.** Tests run as `.venv/Scripts/python.exe -m pytest` from the repo root. Test modules import each other as `tests.…`, so the repo root must be the working directory.
- **Branch.** Work on `feat/m3b-exact-baseline`. Nothing is pushed unless the owner says so.
- **Commits** are made under the repo-local git identity. End each commit message with the `Co-Authored-By` trailer the session specifies.
- **Layer boundaries, enforced by import-linter.** `solvers/` may import `encoding/`, `problem/`, `micro/`, `data/`, `core/`. It must never import `bench/`, `viz/` or `dashboard/`. Within `problem/` it must never import `generator`, `from_bundle` or `liquidation`. `base`, `lot_grid`, `scoring`, `brute_force`, `milp_model` and `exact_pwl_milp` must never import `encoding/`.
- **No new dependencies.** `highspy` is not installed and is not needed; `scipy.optimize.milp` is the interface.
- **One evaluator.** `problem.objective.evaluate` is the only place an objective is computed. A solver reports `evaluate` at its schedule, never its own account of the cost.
- **One verdict on feasibility.** `problem.feasibility.classify`.
- **`cheapest_aux`, never `optimal_aux`,** wherever the QUBO's minimum over the auxiliaries is meant.
- **The gate's width is `GATE_ZETA_BITS = 20`; the study's is `STUDY_ZETA_BITS = 8`.** Neither is tuned to make a test pass. Agreement between the QUBO's minimum and the optimum is not monotone in the width (spec §3.2).
- **Tolerances, and what each one is.** `1e-9` relative between two calls of the evaluator: round-off. `1e-6` relative on a bracket's width: HiGHS's feasibility tolerance of `1e-7`, with room. `1e-6` relative between the closed-form energy and the dense `z'Qz`: accumulation at a dynamic range of `2.4e12`. None is a modelling tolerance, and none may be loosened to make a test pass.
- **Determinism.** Identical inputs produce identical output. No implicit global RNGs, no module-level mutable caches in `src/`.
- **Strictness over silent repair.** Malformed input raises a typed exception. An instance nobody can solve is reported as `INFEASIBLE`, not raised.
- **Dispatch on type, never on an attribute name.**
- **Line length 100**, ruff rules `E,F,I,UP,B,SIM`, passing **without suppressions**. A `# noqa` added to make a rule pass is a defect — report the conflict instead.
- **Test-first.** Write the failing test, watch it fail for the stated reason, then implement.
- **Expected-result lines are advisory.** Judge success by whether tests pass, not by matching a count.
- **Never weaken the gate to make it pass.** If a gate check fails, stop and report which check, which instance and which encoding.

## Review Focus

The inputs the spec implies without spelling out, most likely to bite first. Each has its test in the task that owns the code.

1. **An instance with no feasible schedule.** `Instance` checks its witness's shape and not its feasibility, so one is constructible. Both exact methods return `INFEASIBLE` with no schedule and do not raise. — Task 3 (`test_an_instance_nobody_can_solve_is_reported_and_not_raised`), Task 5 (same name).
2. **HiGHS returning within its own tolerance, or not returning.** The same schedule comes back with a gap of `1e-7`; presolve fails with "Solve error"; the budget runs out mid-loop. The loop ends, the round is retried without presolve, the best schedule and an honest bracket come back. — Task 5 (`test_it_finds_what_enumeration_finds[discrete]`, `test_when_presolve_fails_the_program_is_solved_without_it`, `test_running_out_of_time_keeps_the_best_schedule_and_an_honest_bracket`).
3. **Risk aversion of zero, under either risk spec.** The generator produces one (`T0-D3+D4`). The program carries no risk variable, the gadget's strength is zero and its charge is nothing. — Task 5 (`test_with_no_risk_aversion_there_is_no_risk_in_the_program`), Task 6 (`test_with_no_risk_aversion_the_charge_is_nothing`), Task 7 (`test_a_gadget_with_no_risk_aversion_charges_nothing`).
4. **A state whose losses lie outside `ζ`'s grid.** The grid holds every loss a schedule inside the position can produce; the enumeration also visits oversold states. The scan widens and the answer stays right. — Task 6 (`test_losses_outside_the_grid_widen_the_scan_and_do_not_break_it`).
5. **A name with nothing to sell.** A position of zero gives a cell whose only value is zero: one row in the grid, one column in the program, no bits in the QUBO. — Task 2 (`test_a_name_with_nothing_to_sell_has_one_row_and_it_is_empty`), Tasks 3, 4, 5 and 7 through `nothing_to_sell_in_one_name` in their parametrised instances.

## File Structure

| File | Responsibility |
|---|---|
| `src/quantic/solvers/base.py` | `Budget`, `Bracket`, `Result`, `Solver` |
| `src/quantic/solvers/classical/lot_grid.py` | `LotGrid`, `lot_grid` — admissible values per cell, schedule enumeration |
| `src/quantic/solvers/classical/scoring.py` | `CellCost`, `score` — the one call into the evaluator |
| `src/quantic/solvers/classical/brute_force.py` | `brute_force`, `BruteForce` |
| `src/quantic/solvers/classical/milp_model.py` | `Columns`, `cell_costs`, `permanent_constant`, `loss_model`, `variance_gradient`, `coupling_rows` |
| `src/quantic/solvers/classical/exact_pwl_milp.py` | `exact_pwl_milp`, `ExactPwlMilp` — the solve loop and the bracket |
| `src/quantic/solvers/classical/gadget_minimum.py` | `GadgetMinimiser` — the gadget's charge, minimised over `ζ` and the auxiliaries |
| `src/quantic/solvers/classical/qubo_minimum.py` | `QuboMinimum`, `qubo_minimum` |
| `src/quantic/encoding/expand.py` | gains `ApproximandReport.cell_cost` |
| `pyproject.toml` | two `forbidden` import contracts |
| `tests/ladder.py` | the ladder both gates run on |
| `tests/solvers/instances.py` | small instances the solver tests share |
| `tests/solvers/gate.py` | the gate's checks, as functions |
| `tests/solvers/test_m3b_gate.py` | the gate, and the gate seen failing |
| `tests/solvers/test_t1_bracket.py` | the T1 measurement |
| `tests/solvers/report.py` | prints the result document's tables |

**Refinement of the spec's §11 table.** The spec lists `exact_pwl_milp.py` and `qubo_minimum.py` as one module each. This plan splits each in two: `milp_model.py` holds the program's pieces and `gadget_minimum.py` the gadget's closed form. Each piece is a claim about the objective that can be tested against the evaluator before any solver runs, and the split gives each its own test cycle.

## Task Order

Tasks 1 to 3 build the reference: the interface, the search space, enumeration. Tasks 4 and 5 build the program and check it against the reference. Tasks 6 and 7 build the QUBO's minimiser and check it against the matrix. Task 8 is the gate, and the first thing that can fail for an interesting reason. Task 9 measures and writes down.

---

## Task 1: The solver interface

**Files:**
- Create: `src/quantic/solvers/base.py`
- Create: `tests/solvers/__init__.py` (empty)
- Create: `tests/solvers/instances.py`
- Create: `tests/solvers/test_base.py`
- Modify: `pyproject.toml` (append one contract)

**Interfaces:**
- Consumes: `quantic.problem.instance.Instance`, `quantic.problem.schedule.Schedule`, `quantic.problem.objective.ObjectiveBreakdown`, `quantic.problem.feasibility.FeasibilityReport`.
- Produces:
  - `OPTIMAL = "optimal"`, `TIME_LIMIT = "time_limit"`, `INFEASIBLE = "infeasible"`
  - `Budget(wall_clock_s: float | None = None, target: float | None = None, max_samples: int | None = None)`
  - `Bracket(lower: float, upper: float)` with `.width -> float`; raises `ValueError` when `lower > upper` by more than `1e-6 * max(1, |upper|)`
  - `Result(solver, status, schedule, zeta, objective, bracket, feasibility, timings={}, detail={})`
  - `Solver` — a runtime-checkable protocol: `name: str`, `solve(instance, budget) -> Result`
  - `tests.solvers.instances`: `instance(**kw)`, `plain()`, `concave()`, `discrete()`, `blocks()`, `cvar(lam=0.02)`, `cvar_discrete(lam=0.02)`, `no_risk_aversion()`, `cvar_without_risk_aversion()`, `nothing_to_sell_in_one_name()`, `infeasible()`, `scenarios(n=8, n_buckets=3, seed=11)`, `EVERY_SOLVABLE`

- [ ] **Step 1: Write the shared instances**

Create `tests/solvers/__init__.py` as an empty file, then create `tests/solvers/instances.py`:

```python
"""Small instances the solver tests share.

Each is small enough to enumerate by hand-rolled loops inside a test, and each
turns on one thing: a dial, a risk spec, or a way of being infeasible. Risk
aversion is set high enough that the risk term moves the optimum, because a
solver that ignored the risk term would otherwise pass.
"""

from __future__ import annotations

import numpy as np

from quantic.core.types import Asset
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def params() -> MarketParams:
    return MarketParams(
        assets=(
            AssetParams("A", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01),
            AssetParams("B", 0.6, 0.7, 0.1, 0.004, 300_000.0, 101.0, 0.02),
        ),
        covariance=np.array([[2.0e-5, 0.8e-5], [0.8e-5, 1.5e-5]]),
        bucket_ns=BUCKET_NS,
    )


def instance(**kw) -> Instance:
    base = dict(
        instance_id="small", tier="T0",
        assets=tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B")),
        n_buckets=3, initial_lots=(4, 3), max_lots_per_bucket=3, params=params(),
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=2e-4),
        witness=Schedule(lots=((2, 1, 1), (1, 1, 1))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def plain() -> Instance:
    return instance(instance_id="plain")


def concave() -> Instance:
    return instance(instance_id="concave", dials=Dials(concave_impact=True))


def discrete() -> Instance:
    """One name a bucket, and a venue minimum of two lots on the first name."""
    return instance(
        instance_id="discrete",
        dials=Dials(discrete_participation=True),
        constraints=(
            FullLiquidation(), Cardinality(k=1), MinParticipation(min_lots=(2, 1)),
        ),
        witness=Schedule(lots=((2, 2, 0), (0, 0, 3))),
    )


def blocks() -> Instance:
    """The first name's first bucket trades three lots or none."""
    return instance(
        instance_id="blocks",
        dials=Dials(block_trades=True),
        constraints=(
            FullLiquidation(), BlockTrades(blocks=(Block(asset=0, bucket=0, lots=3),)),
        ),
        witness=Schedule(lots=((3, 1, 0), (1, 1, 1))),
    )


def scenarios(n: int = 8, n_buckets: int = 3, seed: int = 11) -> ScenarioSet:
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.5, 1.5, size=n)
    return ScenarioSet(
        returns=rng.standard_t(df=3, size=(n, n_buckets, 2)) * 0.01,
        weights=weights / weights.sum(), seed=seed,
    )


def cvar(lam: float = 0.02) -> Instance:
    """Eight unevenly weighted fat-tailed scenarios, several of them in the tail."""
    return instance(
        instance_id="cvar",
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=lam, alpha=0.7, scenarios=scenarios()),
    )


def cvar_discrete(lam: float = 0.02) -> Instance:
    return instance(
        instance_id="cvar_discrete",
        dials=Dials(cvar_risk=True, discrete_participation=True),
        constraints=(
            FullLiquidation(), Cardinality(k=1), MinParticipation(min_lots=(2, 1)),
        ),
        risk=CVaRRisk(lam=lam, alpha=0.7, scenarios=scenarios()),
        witness=Schedule(lots=((2, 2, 0), (0, 0, 3))),
    )


def no_risk_aversion() -> Instance:
    return instance(instance_id="no_risk_aversion", risk=VarianceRisk(lam=0.0))


def cvar_without_risk_aversion() -> Instance:
    return instance(
        instance_id="cvar_without_risk_aversion",
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=0.0, alpha=0.7, scenarios=scenarios()),
    )


def nothing_to_sell_in_one_name() -> Instance:
    """The second name's position is zero, so its cells can only hold zero."""
    return instance(
        instance_id="nothing_to_sell_in_one_name",
        initial_lots=(4, 0),
        witness=Schedule(lots=((2, 1, 1), (0, 0, 0))),
    )


def infeasible() -> Instance:
    """Two names that each need two buckets, two buckets, one name a bucket.

    ``Instance`` checks the witness's shape and not its feasibility, so an
    instance nobody can solve is constructible -- from a bundle, for one.
    """
    return instance(
        instance_id="infeasible",
        n_buckets=2, initial_lots=(4, 4), max_lots_per_bucket=3,
        dials=Dials(discrete_participation=True),
        constraints=(
            FullLiquidation(), Cardinality(k=1), MinParticipation(min_lots=(1, 1)),
        ),
        witness=Schedule(lots=((3, 1), (1, 3))),
    )


EVERY_SOLVABLE = (
    plain, concave, discrete, blocks, cvar, cvar_discrete,
    no_risk_aversion, cvar_without_risk_aversion, nothing_to_sell_in_one_name,
)
```

- [ ] **Step 2: Write the failing test**

Create `tests/solvers/test_base.py`:

```python
import dataclasses

import pytest

from quantic.problem.feasibility import classify
from quantic.problem.objective import evaluate
from quantic.solvers.base import (
    INFEASIBLE,
    OPTIMAL,
    TIME_LIMIT,
    Bracket,
    Budget,
    Result,
    Solver,
)
from tests.solvers.instances import plain


def _result(**kw) -> Result:
    inst = plain()
    base = dict(
        solver="s", status=OPTIMAL, schedule=inst.witness, zeta=None,
        objective=evaluate(inst, inst.witness),
        bracket=Bracket(lower=1.0, upper=1.0),
        feasibility=classify(inst, inst.witness),
    )
    base.update(kw)
    return Result(**base)


def test_a_bracket_reports_how_far_apart_its_ends_are():
    assert Bracket(lower=1.0, upper=1.5).width == 0.5


def test_a_bracket_refuses_a_lower_bound_above_the_value_attained():
    with pytest.raises(ValueError, match="one of the two is wrong"):
        Bracket(lower=2.0, upper=1.0)


def test_a_bracket_lets_a_solver_tolerance_through_and_reports_no_negative_width():
    """HiGHS proves its bound to about 1e-7, so the bound can sit a hair above."""
    bracket = Bracket(lower=30.0 + 1e-7, upper=30.0)
    assert bracket.width == 0.0


def test_a_bracket_does_not_let_more_than_a_tolerance_through():
    with pytest.raises(ValueError):
        Bracket(lower=30.0 + 1e-3, upper=30.0)


def test_a_bracket_with_nothing_proven_yet_is_infinitely_wide():
    assert Bracket(lower=float("-inf"), upper=3.0).width == float("inf")


@pytest.mark.parametrize("field", ["wall_clock_s", "max_samples"])
@pytest.mark.parametrize("value", [0, -1])
def test_a_budget_refuses_a_limit_that_allows_nothing(field, value):
    with pytest.raises(ValueError, match=field):
        Budget(**{field: value})


def test_a_budget_with_no_limits_is_the_default():
    assert Budget() == Budget(wall_clock_s=None, target=None, max_samples=None)


def test_a_result_refuses_a_status_nobody_defined():
    with pytest.raises(ValueError, match="status"):
        _result(status="probably_fine")


@pytest.mark.parametrize("missing", ["schedule", "bracket"])
def test_an_optimal_result_has_to_carry_what_makes_it_optimal(missing):
    with pytest.raises(ValueError, match="optimal"):
        _result(**{missing: None})


@pytest.mark.parametrize("status", [TIME_LIMIT, INFEASIBLE])
def test_a_result_that_is_not_optimal_may_carry_nothing(status):
    result = _result(
        status=status, schedule=None, objective=None, bracket=None, feasibility=None
    )
    assert result.schedule is None


def test_a_result_cannot_be_edited_after_the_fact():
    result = _result(timings={"total_s": 1.0}, detail={"rounds": 3})
    with pytest.raises(TypeError):
        result.timings["total_s"] = 0.0
    with pytest.raises(TypeError):
        result.detail["rounds"] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = TIME_LIMIT


def test_a_result_does_not_share_its_mappings_with_the_caller():
    timings = {"total_s": 1.0}
    result = _result(timings=timings)
    timings["total_s"] = 99.0
    assert result.timings["total_s"] == 1.0


def test_anything_with_a_name_and_a_solve_method_is_a_solver():
    class Stub:
        name = "stub"

        def solve(self, instance, budget):
            return _result()

    assert isinstance(Stub(), Solver)
    assert not isinstance(object(), Solver)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_base.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.solvers.base'`.

- [ ] **Step 4: Write the implementation**

Create `src/quantic/solvers/base.py`:

```python
"""The solver interface -- spec section 7.1, at the size the exact solvers need.

Every solver answers ``solve(instance, budget) -> Result``. What comes back is
a schedule and the evaluator's verdict on it, never the solver's own account
of what its schedule costs: ``Result.objective`` is
``problem.objective.evaluate`` at ``Result.schedule``, and a solver that
modelled the objective wrongly shows up as a bracket that does not close
rather than as a better number.

Section 7.1's energy trace is not here. Neither exact solver has one, and the
samplers of M4 add it with the first solver that does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from quantic.problem.feasibility import FeasibilityReport
from quantic.problem.instance import Instance
from quantic.problem.objective import ObjectiveBreakdown
from quantic.problem.schedule import Schedule

OPTIMAL = "optimal"
TIME_LIMIT = "time_limit"
INFEASIBLE = "infeasible"

_STATUSES = (OPTIMAL, TIME_LIMIT, INFEASIBLE)

# A proven lower bound comes out of a solver working to its own tolerance, so
# it can sit a hair above the value it bounds. Beyond this it is not round-off.
_BRACKET_RTOL = 1e-6


@dataclass(frozen=True, slots=True)
class Budget:
    """What a solver may spend. ``None`` means no limit."""

    wall_clock_s: float | None = None
    target: float | None = None
    max_samples: int | None = None

    def __post_init__(self) -> None:
        if self.wall_clock_s is not None and self.wall_clock_s <= 0:
            raise ValueError(f"wall_clock_s must be positive, got {self.wall_clock_s}")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError(f"max_samples must be at least 1, got {self.max_samples}")


@dataclass(frozen=True, slots=True)
class Bracket:
    """The optimum lies in ``[lower, upper]``.

    ``lower`` is proven: no feasible schedule costs less. ``upper`` is
    attained: it is the evaluator's total at a feasible schedule.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        slack = _BRACKET_RTOL * max(1.0, abs(self.upper))
        if self.lower > self.upper + slack:
            raise ValueError(
                f"lower bound {self.lower!r} is above the attained value "
                f"{self.upper!r}: a schedule cannot cost less than the proven "
                "minimum, so one of the two is wrong"
            )

    @property
    def width(self) -> float:
        return max(self.upper - self.lower, 0.0)


@dataclass(frozen=True)
class Result:
    """One solver's answer on one instance."""

    solver: str
    status: str
    schedule: Schedule | None
    zeta: float | None
    objective: ObjectiveBreakdown | None
    bracket: Bracket | None
    feasibility: FeasibilityReport | None
    timings: Mapping[str, float] = field(default_factory=dict)
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(
                f"status must be one of {_STATUSES}, got {self.status!r}"
            )
        if self.status == OPTIMAL and (self.schedule is None or self.bracket is None):
            raise ValueError(
                "an optimal result carries the schedule and the bracket that make "
                "it optimal"
            )
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))


@runtime_checkable
class Solver(Protocol):
    name: str

    def solve(self, instance: Instance, budget: Budget) -> Result: ...
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_base.py -q`
Expected: every test in the file passes (18).

- [ ] **Step 6: Add the import contract**

Append to `pyproject.toml`:

```toml

[[tool.importlinter.contracts]]
name = "Solvers read the problem and never build it (spec section 12)"
type = "forbidden"
source_modules = ["quantic.solvers"]
forbidden_modules = [
    "quantic.problem.generator",
    "quantic.problem.from_bundle",
    "quantic.problem.liquidation",
]
```

This closes a finding open since the M1 review: the `layers` contract permits `solvers` to import `problem`'s internals, and parent spec §12 forbids it.

- [ ] **Step 7: Lint**

Run: `.venv/Scripts/ruff.exe check src tests` and `.venv/Scripts/lint-imports.exe`
Expected: `All checks passed!` and `Contracts: 2 kept, 0 broken.`

- [ ] **Step 8: Commit**

```bash
git add src/quantic/solvers/base.py tests/solvers/ pyproject.toml
git commit -m "feat(solvers): the interface every solver answers to"
```

---

## Task 2: The lot grid

**Files:**
- Create: `tests/ladder.py`
- Modify: `tests/encoding/test_m3a_gate.py` (imports and the `ladder` fixture)
- Create: `src/quantic/solvers/classical/lot_grid.py`
- Create: `tests/solvers/test_lot_grid.py`

**Interfaces:**
- Consumes: the four constraint descriptors under `quantic.problem.constraints`; `Instance`; `Schedule`; `tests.solvers.instances` from Task 1.
- Produces:
  - `LotGrid(values, initial_lots, cardinality)` with `.n_assets`, `.n_buckets`, `.rows(i) -> tuple[tuple[int, ...], ...]`, `.n_candidates() -> int`, `.schedules() -> Iterator[Schedule]`
  - `lot_grid(instance: Instance) -> LotGrid`
  - `tests.ladder`: `params_for(tier)`, `gate_ladder() -> tuple[Instance, ...]` (cached), `tier(name: str) -> tuple[Instance, ...]`, `SEED`, `SMALL`, `BUCKET_NS`

`LotGrid.schedules()` applies full liquidation and the per-cell values. It does **not** apply cardinality, which couples the rows: `classify` is the only verdict on it.

- [ ] **Step 1: Move the ladder to where both gates can reach it**

Create `tests/ladder.py`:

```python
"""The ladder both gates are run on.

M3a's gate and M3b's build the same sixty-four instances, from the same
parameters and the same seed. They are built here so that "the same" is a
fact about one function and not about two copies of it.
"""

from __future__ import annotations

import functools

import numpy as np

from quantic.problem.generator import generate_ladder
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams

BUCKET_NS = 1800 * 1_000_000_000
SEED = 2026
SMALL = {"n_raw": 300, "n_target": 15}


def params_for(tier) -> MarketParams:
    rng = np.random.default_rng(tier.n_assets)
    a = rng.normal(size=(tier.n_assets, tier.n_assets))
    return MarketParams(
        assets=tuple(
            AssetParams(f"S{i:02d}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + i, 0.01)
            for i in range(tier.n_assets)
        ),
        covariance=(a @ a.T) / tier.n_assets * 1e-5,
        bucket_ns=BUCKET_NS,
    )


@functools.cache
def gate_ladder() -> tuple[Instance, ...]:
    """All sixty-four instances. Built once per process: generation is pure."""
    return generate_ladder(params_for, seed=SEED, include_cvar=True, **SMALL)


def tier(name: str) -> tuple[Instance, ...]:
    return tuple(i for i in gate_ladder() if i.tier == name)
```

In `tests/encoding/test_m3a_gate.py`, replace this block of imports and constants:

```python
from quantic.problem.generator import generate_ladder
from quantic.problem.objective import cvar_of, optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
SMALL = {"n_raw": 300, "n_target": 15}
REL = 1e-9
```

with:

```python
from quantic.problem.objective import cvar_of, optimal_zeta
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule
from tests.ladder import gate_ladder

REL = 1e-9
```

and replace `_params_for` and the `ladder` fixture:

```python
def _params_for(tier):
    rng = np.random.default_rng(tier.n_assets)
    a = rng.normal(size=(tier.n_assets, tier.n_assets))
    return MarketParams(
        assets=tuple(
            AssetParams(f"S{i:02d}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + i, 0.01)
            for i in range(tier.n_assets)
        ),
        covariance=(a @ a.T) / tier.n_assets * 1e-5,
        bucket_ns=BUCKET_NS,
    )


@pytest.fixture(scope="module")
def ladder():
    return generate_ladder(_params_for, seed=2026, include_cvar=True, **SMALL)
```

with:

```python
@pytest.fixture(scope="module")
def ladder():
    return gate_ladder()
```

Nothing else in the M3a gate changes. `np` is still used further down the file.

- [ ] **Step 2: Check the M3a gate still builds its ladder**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_m3a_gate.py -q -k "still_sixteen"`
Expected: 1 passed. The whole M3a gate takes six to ten minutes and is run in Task 9.

- [ ] **Step 3: Write the failing test**

Create `tests/solvers/test_lot_grid.py`:

```python
import itertools
from dataclasses import dataclass, field

import pytest

from quantic.encoding.integer import ENCODINGS
from quantic.encoding.layout import plan_layout
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.feasibility import classify
from quantic.problem.schedule import Schedule
from quantic.solvers.classical.lot_grid import lot_grid
from tests.ladder import gate_ladder
from tests.solvers.instances import (
    blocks,
    discrete,
    instance,
    nothing_to_sell_in_one_name,
    plain,
)


def test_a_plain_cell_ranges_up_to_the_cap_or_the_position():
    grid = lot_grid(instance(initial_lots=(4, 2), witness=Schedule(lots=((2, 1, 1), (1, 1, 0)))))
    assert grid.values[0] == ((0, 1, 2, 3),) * 3      # the cap of three binds
    assert grid.values[1] == ((0, 1, 2),) * 3         # the position of two binds


def test_a_venue_minimum_removes_the_values_below_it_and_keeps_zero():
    grid = lot_grid(discrete())
    assert grid.values[0] == ((0, 2, 3),) * 3
    assert grid.values[1] == ((0, 1, 2, 3),) * 3
    assert grid.cardinality == 1


def test_a_block_cell_trades_the_block_or_nothing():
    grid = lot_grid(blocks())
    assert grid.values[0][0] == (0, 3)
    assert grid.values[0][1] == (0, 1, 2, 3)
    assert grid.cardinality is None


def test_a_block_below_the_venue_minimum_can_only_stay_untraded():
    inst = instance(
        constraints=(
            FullLiquidation(),
            MinParticipation(min_lots=(3, 1)),
            BlockTrades(blocks=(Block(asset=0, bucket=1, lots=2),)),
        ),
        witness=Schedule(lots=((3, 0, 1), (1, 1, 1))),
    )
    assert lot_grid(inst).values[0][1] == (0,)


def test_a_name_with_nothing_to_sell_has_one_row_and_it_is_empty():
    grid = lot_grid(nothing_to_sell_in_one_name())
    assert grid.values[1] == ((0,),) * 3
    assert grid.rows(1) == ((0, 0, 0),)


def test_the_rows_are_every_way_to_sell_the_position():
    grid = lot_grid(plain())
    expected = {
        r for r in itertools.product(range(4), repeat=3) if sum(r) == 4
    }
    assert set(grid.rows(0)) == expected
    assert len(grid.rows(0)) == len(expected) == 12


def test_no_row_is_listed_twice():
    grid = lot_grid(discrete())
    for i in range(grid.n_assets):
        assert len(set(grid.rows(i))) == len(grid.rows(i))


def test_a_position_the_cells_cannot_reach_has_no_rows():
    inst = instance(
        n_buckets=2, initial_lots=(7, 3), max_lots_per_bucket=3,
        witness=Schedule(lots=((3, 3), (2, 1))),
    )
    grid = lot_grid(inst)
    assert grid.rows(0) == ()
    assert grid.n_candidates() == 0
    assert list(grid.schedules()) == []


@pytest.mark.parametrize("build", [plain, discrete, blocks, nothing_to_sell_in_one_name])
def test_the_count_is_the_number_of_schedules_yielded(build):
    grid = lot_grid(build())
    assert grid.n_candidates() == sum(1 for _ in grid.schedules())


@pytest.mark.parametrize("build", [plain, discrete, blocks])
def test_every_schedule_yielded_breaks_nothing_but_cardinality(build):
    """Cardinality couples the rows and is left to classify, by design."""
    inst = build()
    for schedule in lot_grid(inst).schedules():
        breaches = {
            name for name, size in classify(inst, schedule).violations.items() if size
        }
        assert breaches <= {"cardinality"}


@pytest.mark.parametrize("build", [plain, discrete, blocks])
def test_no_feasible_schedule_is_missing_from_the_grid(build):
    """The grid is the search space, so what it leaves out is never found."""
    inst = build()
    cap = inst.max_lots_per_bucket
    cells = inst.n_assets * inst.n_buckets
    feasible = set()
    for flat in itertools.product(range(cap + 1), repeat=cells):
        rows = tuple(
            tuple(flat[i * inst.n_buckets:(i + 1) * inst.n_buckets])
            for i in range(inst.n_assets)
        )
        schedule = Schedule(lots=rows)
        if classify(inst, schedule).feasible:
            feasible.add(schedule)
    assert feasible <= set(lot_grid(inst).schedules())
    assert feasible


def test_a_constraint_the_grid_does_not_know_is_refused():
    @dataclass(frozen=True)
    class SellOnTuesdays:
        name: str = field(default="sell_on_tuesdays")

        def violation(self, schedule, initial_lots):
            return 0.0

        def is_satisfied(self, schedule, initial_lots):
            return True

        def describe(self):
            return {"name": self.name}

    inst = instance(constraints=(FullLiquidation(), SellOnTuesdays()))
    with pytest.raises(NotImplementedError, match="sell_on_tuesdays"):
        lot_grid(inst)


def test_an_instance_that_need_not_liquidate_is_refused():
    inst = instance(constraints=(MinParticipation(min_lots=(1, 1)),))
    with pytest.raises(ValueError, match="FullLiquidation"):
        lot_grid(inst)


def test_a_venue_minimum_for_the_wrong_number_of_names_is_refused():
    inst = instance(
        constraints=(FullLiquidation(), MinParticipation(min_lots=(1, 1, 1)))
    )
    with pytest.raises(ValueError, match="min_lots"):
        lot_grid(inst)


def test_the_grid_and_the_encoder_agree_on_every_cell_of_the_ladder():
    """Two derivations of one search space, neither importing the other.

    If they differed the gate would be comparing a QUBO's minimum over one
    set of schedules with a baseline's over another.
    """
    differing = {}
    for inst in gate_ladder():
        grid = lot_grid(inst)
        for encoding in ENCODINGS:
            layout = plan_layout(inst, encoding)
            for plan in layout.cells:
                mine = grid.values[plan.asset][plan.bucket]
                if tuple(plan.values) != mine:
                    differing[(inst.instance_id, encoding.name, plan.asset, plan.bucket)] = (
                        mine, tuple(plan.values)
                    )
    assert not differing, differing
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_lot_grid.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.solvers.classical.lot_grid'`.

- [ ] **Step 5: Write the implementation**

Create `src/quantic/solvers/classical/lot_grid.py`:

```python
"""Which lots each cell may trade -- the search space both exact methods share.

The constraints of spec sections 5.3 and 6.2 act in two ways. Three of them
restrict a single cell: the per-bucket cap, a venue minimum, an
all-or-nothing block. Those become the cell's set of admissible values. The
other two couple cells -- full liquidation along a row, cardinality down a
column -- and are left to the method that searches.

``encoding/layout.py`` derives the same sets for the encoder, and this module
does not import it. The gate compares a QUBO's minimum with the baseline's,
and a baseline that took its search space from the encoder would inherit the
encoder's mistakes and agree with them.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

from quantic.problem.constraints.block_trades import BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.instance import Instance
from quantic.problem.schedule import Schedule


@dataclass(frozen=True)
class LotGrid:
    """Every cell's admissible values, and what couples the cells."""

    values: tuple[tuple[tuple[int, ...], ...], ...]   # [i][t] -> lots, ascending
    initial_lots: tuple[int, ...]
    cardinality: int | None

    @property
    def n_assets(self) -> int:
        return len(self.values)

    @property
    def n_buckets(self) -> int:
        return len(self.values[0])

    def rows(self, i: int) -> tuple[tuple[int, ...], ...]:
        """Every way asset ``i`` can sell its whole position on admissible values."""
        cells = self.values[i]
        # The most the cells from t onward can still sell, so a branch that
        # cannot reach the position is cut where it starts.
        reach = [0] * (len(cells) + 1)
        for t in range(len(cells) - 1, -1, -1):
            reach[t] = reach[t + 1] + max(cells[t])

        out: list[tuple[int, ...]] = []

        def extend(t: int, left: int, prefix: tuple[int, ...]) -> None:
            if t == len(cells):
                if left == 0:
                    out.append(prefix)
                return
            for v in cells[t]:
                if v > left:
                    break
                if left - v > reach[t + 1]:
                    continue
                extend(t + 1, left - v, (*prefix, v))

        extend(0, self.initial_lots[i], ())
        return tuple(out)

    def n_candidates(self) -> int:
        """How many schedules :meth:`schedules` yields."""
        return math.prod(len(self.rows(i)) for i in range(self.n_assets))

    def schedules(self) -> Iterator[Schedule]:
        """Every schedule that liquidates in full on admissible values.

        Cardinality is not applied: it couples the rows, and
        ``problem.feasibility.classify`` is the only verdict on it.
        """
        per_asset = [self.rows(i) for i in range(self.n_assets)]
        for combo in itertools.product(*per_asset):
            yield Schedule(lots=tuple(combo))


def lot_grid(instance: Instance) -> LotGrid:
    """Read the constraint set into a grid. Dispatch is on type, never on a name."""
    minimums = [1] * instance.n_assets
    blocks: dict[tuple[int, int], int] = {}
    cardinality: int | None = None
    liquidates = False

    for c in instance.constraints:
        if isinstance(c, FullLiquidation):
            liquidates = True
        elif isinstance(c, Cardinality):
            cardinality = c.k
        elif isinstance(c, MinParticipation):
            if len(c.min_lots) != instance.n_assets:
                raise ValueError(
                    f"min_lots has {len(c.min_lots)} entries but the instance has "
                    f"{instance.n_assets} assets"
                )
            minimums = [int(m) for m in c.min_lots]
        elif isinstance(c, BlockTrades):
            blocks = {(b.asset, b.bucket): int(b.lots) for b in c.blocks}
        else:
            raise NotImplementedError(
                f"lot_grid does not know constraint {c.name!r} ({type(c).__name__}). "
                "A grid that ignored it would make both exact methods agree on the "
                "wrong answer"
            )

    if not liquidates:
        raise ValueError(
            "the instance carries no FullLiquidation constraint. The grid "
            "enumerates rows that sell the whole position, so without it the "
            "search space is not the problem's"
        )

    values = []
    for i in range(instance.n_assets):
        cap = min(instance.max_lots_per_bucket, instance.initial_lots[i])
        row = []
        for t in range(instance.n_buckets):
            block = blocks.get((i, t))
            if block is not None:
                # A block below the venue minimum can only stay untraded.
                row.append((0, block) if block >= minimums[i] else (0,))
            elif minimums[i] <= 1:
                row.append(tuple(range(cap + 1)))
            else:
                row.append((0, *range(minimums[i], cap + 1)))
        values.append(tuple(row))

    return LotGrid(
        values=tuple(values),
        initial_lots=tuple(int(x) for x in instance.initial_lots),
        cardinality=cardinality,
    )
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_lot_grid.py -q`
Expected: every test in the file passes (22). `test_the_grid_and_the_encoder_agree_on_every_cell_of_the_ladder` builds all 64 instances and takes about ten seconds.

- [ ] **Step 7: Lint and commit**

Run: `.venv/Scripts/ruff.exe check src tests` and `.venv/Scripts/lint-imports.exe`
Expected: clean, `Contracts: 2 kept, 0 broken.`

```bash
git add tests/ladder.py tests/encoding/test_m3a_gate.py src/quantic/solvers/classical/lot_grid.py tests/solvers/test_lot_grid.py
git commit -m "feat(solvers): the lot grid both exact methods search"
```

---

## Task 3: Scoring and enumeration

**Files:**
- Create: `src/quantic/solvers/classical/scoring.py`
- Create: `src/quantic/solvers/classical/brute_force.py`
- Modify: `src/quantic/encoding/expand.py` (one method on `ApproximandReport`)
- Create: `tests/solvers/test_brute_force.py`

**Interfaces:**
- Consumes: `lot_grid` (Task 2); `Result`, `Bracket`, `Budget`, `OPTIMAL`, `INFEASIBLE` (Task 1); `evaluate`, `optimal_zeta`, `classify`.
- Produces:
  - `CellCost = Callable[[int, int, int], float]` — `(asset, bucket, lots) ->` the temporary impact cost of that one cell
  - `score(instance, schedule, *, cell_cost: CellCost | None = None) -> tuple[ObjectiveBreakdown, float | None]` — the breakdown and the `zeta` it was scored at
  - `brute_force(instance, *, cell_cost=None, max_schedules=10_000_000) -> Result`; `Result.detail` carries `n_candidates` and `n_feasible`
  - `BruteForce` — `name = "brute_force"`, `solve(instance, budget) -> Result`
  - `ApproximandReport.cell_cost(i: int, t: int, lots: int) -> float`

`ApproximandReport.temporary_cost` is left exactly as it is. Rewriting it as a sum of `cell_cost` would change its order of accumulation, and the M3a gate compares against it at `1e-9`.

- [ ] **Step 1: Write the failing test**

Create `tests/solvers/test_brute_force.py`:

```python
import itertools

import pytest

from quantic.encoding.encode import encode, reference_breakdown
from quantic.encoding.integer import BinaryEncoding
from quantic.problem.dials import Dials
from quantic.problem.feasibility import classify
from quantic.problem.objective import evaluate, optimal_zeta
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.base import INFEASIBLE, OPTIMAL, Budget, Solver
from quantic.solvers.classical.brute_force import BruteForce, brute_force
from quantic.solvers.classical.scoring import score
from tests.solvers.instances import (
    EVERY_SOLVABLE,
    blocks,
    concave,
    cvar,
    discrete,
    infeasible,
    instance,
    plain,
)


def _every_feasible(inst):
    """Every feasible schedule, found without the lot grid."""
    cap = inst.max_lots_per_bucket
    out = []
    for flat in itertools.product(range(cap + 1), repeat=inst.n_assets * inst.n_buckets):
        schedule = Schedule(
            lots=tuple(
                tuple(flat[i * inst.n_buckets:(i + 1) * inst.n_buckets])
                for i in range(inst.n_assets)
            )
        )
        if classify(inst, schedule).feasible:
            out.append(schedule)
    return out


def _true_total(inst, schedule):
    zeta = optimal_zeta(inst, schedule) if isinstance(inst.risk, CVaRRisk) else None
    return evaluate(inst, schedule, zeta=zeta).total


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_no_feasible_schedule_costs_less_than_the_one_returned(build):
    inst = build()
    result = brute_force(inst)
    cheapest = min(_true_total(inst, s) for s in _every_feasible(inst))
    assert result.status == OPTIMAL
    assert result.objective.total == cheapest
    assert result.feasibility.feasible


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_value_reported_is_the_evaluators_at_the_schedule_returned(build):
    inst = build()
    result = brute_force(inst)
    assert result.objective.total == _true_total(inst, result.schedule)
    assert result.bracket.lower == result.bracket.upper == result.objective.total


def test_the_risk_term_moves_the_optimum_on_the_instances_these_tests_lean_on():
    """Otherwise a solver that ignored risk would pass every test above."""
    for build in (plain, cvar):
        inst = build()
        with_risk = brute_force(inst).schedule
        costs_only = min(
            _every_feasible(inst),
            key=lambda s, inst=inst: _true_total(inst, s) - score(inst, s)[0].risk_cost,
        )
        assert with_risk != costs_only, build.__name__


def test_under_cvar_the_zeta_returned_is_the_minimising_one():
    inst = cvar()
    result = brute_force(inst)
    assert result.zeta == optimal_zeta(inst, result.schedule)


def test_without_cvar_there_is_no_zeta():
    assert brute_force(plain()).zeta is None


def test_it_says_how_many_schedules_it_looked_at():
    inst = discrete()
    result = brute_force(inst)
    assert result.detail["n_feasible"] == len(_every_feasible(inst))
    assert result.detail["n_candidates"] >= result.detail["n_feasible"]
    assert result.timings["total_s"] >= 0.0


def test_an_instance_nobody_can_solve_is_reported_and_not_raised():
    result = brute_force(infeasible())
    assert result.status == INFEASIBLE
    assert result.schedule is None
    assert result.bracket is None
    assert result.detail["n_feasible"] == 0


def test_a_grid_too_large_to_enumerate_is_refused_before_it_starts():
    with pytest.raises(ValueError, match="above the limit"):
        brute_force(plain(), max_schedules=10)


def test_a_cost_table_replaces_the_temporary_impact_and_nothing_else():
    inst = concave()
    true, _ = score(inst, inst.witness)
    doubled, _ = score(
        inst, inst.witness, cell_cost=lambda i, t, lots: 2.0 * _cell(inst, i, t, lots)
    )
    assert doubled.temporary_impact_cost == pytest.approx(
        2.0 * true.temporary_impact_cost, rel=1e-12
    )
    assert doubled.spread_cost == true.spread_cost
    assert doubled.permanent_impact_cost == true.permanent_impact_cost
    assert doubled.risk_cost == true.risk_cost


def _cell(inst, i, t, lots):
    grid = [[0] * inst.n_buckets for _ in range(inst.n_assets)]
    grid[i][t] = lots
    return evaluate(
        inst, Schedule(lots=tuple(tuple(r) for r in grid))
    ).temporary_impact_cost


def test_with_the_encoders_table_the_optimum_is_of_the_function_it_declares():
    """M3a's hand-off: the baseline for the log encoding is stated against the
    approximand, because that is what the QUBO minimises."""
    inst = concave()
    enc = encode(inst, BinaryEncoding())
    report = enc.breakdown.approximand
    assert report is not None

    result = brute_force(inst, cell_cost=report.cell_cost)
    declared = min(
        reference_breakdown(enc, s).total for s in _every_feasible(inst)
    )
    assert result.objective.total == pytest.approx(declared, rel=1e-12)
    assert result.objective.temporary_impact_cost == pytest.approx(
        report.temporary_cost(result.schedule), rel=1e-12
    )


def test_the_encoders_table_adds_up_to_its_own_total():
    inst = concave()
    report = encode(inst, BinaryEncoding()).breakdown.approximand
    for schedule in _every_feasible(inst):
        by_cell = sum(
            report.cell_cost(i, t, lots)
            for i, row in enumerate(schedule.lots)
            for t, lots in enumerate(row)
        )
        assert by_cell == pytest.approx(report.temporary_cost(schedule), rel=1e-12)


def test_the_encoders_table_refuses_a_value_the_cell_cannot_take():
    inst = instance(
        dials=Dials(concave_impact=True, block_trades=True),
        constraints=blocks().constraints, witness=blocks().witness,
    )
    report = encode(inst, BinaryEncoding()).breakdown.approximand
    with pytest.raises(ValueError, match="no cost"):
        report.cell_cost(0, 0, 2)          # the block cell trades 0 or 3
    with pytest.raises(KeyError, match="neither"):
        report.cell_cost(5, 5, 1)


def test_it_answers_to_the_solver_protocol():
    solver = BruteForce()
    assert isinstance(solver, Solver)
    inst = plain()
    assert solver.solve(inst, Budget()).schedule == brute_force(inst).schedule
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_brute_force.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.solvers.classical.brute_force'`.

- [ ] **Step 3: Write `scoring.py`**

Create `src/quantic/solvers/classical/scoring.py`:

```python
"""What a schedule costs, for every solver in ``classical/``.

``problem.objective.evaluate`` is the only evaluator in the project. This
module is the one place the solvers call it from, so that the question "which
``zeta``, and which temporary impact" has one answer.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from quantic.problem.instance import Instance
from quantic.problem.objective import ObjectiveBreakdown, evaluate, optimal_zeta
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule

# (asset, bucket, lots) -> the temporary impact cost of that one cell.
CellCost = Callable[[int, int, int], float]


def score(
    instance: Instance, schedule: Schedule, *, cell_cost: CellCost | None = None
) -> tuple[ObjectiveBreakdown, float | None]:
    """The objective at ``schedule``, and the ``zeta`` it was scored at.

    Under CVaR the ``zeta`` is the minimising one, so the risk term is the
    true CVaR. With a ``cell_cost`` one field of the evaluator's output is
    replaced, as ``encoding.encode.reference_breakdown`` does, and nothing is
    recomputed.
    """
    zeta = (
        optimal_zeta(instance, schedule)
        if isinstance(instance.risk, CVaRRisk)
        else None
    )
    breakdown = evaluate(instance, schedule, zeta=zeta)
    if cell_cost is not None:
        breakdown = dataclasses.replace(
            breakdown,
            temporary_impact_cost=float(
                sum(
                    cell_cost(i, t, lots)
                    for i, row in enumerate(schedule.lots)
                    for t, lots in enumerate(row)
                )
            ),
        )
    return breakdown, zeta
```

- [ ] **Step 4: Write `brute_force.py`**

Create `src/quantic/solvers/classical/brute_force.py`:

```python
"""The exact optimum by looking at every feasible schedule.

Nothing here is clever, which is the point. It shares the lot grid and the
evaluator with the MILP and nothing else: no bound, no relaxation, no model of
the objective. When the two agree, the MILP's model is right; when they do
not, this one is the reference.
"""

from __future__ import annotations

import time

from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.objective import ObjectiveBreakdown
from quantic.problem.schedule import Schedule
from quantic.solvers.base import INFEASIBLE, OPTIMAL, Bracket, Budget, Result
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.scoring import CellCost, score

NAME = "brute_force"
MAX_SCHEDULES = 10_000_000


def brute_force(
    instance: Instance,
    *,
    cell_cost: CellCost | None = None,
    max_schedules: int = MAX_SCHEDULES,
) -> Result:
    """Minimise the objective over every feasible schedule on the lot grid."""
    started = time.perf_counter()
    grid = lot_grid(instance)
    n_candidates = grid.n_candidates()
    if n_candidates > max_schedules:
        raise ValueError(
            f"{instance.instance_id} has {n_candidates} schedules on its lot grid, "
            f"above the limit of {max_schedules}. Enumeration is the reference at "
            "T0; above that the MILP carries the proof"
        )

    best: tuple[float, Schedule, ObjectiveBreakdown, float | None] | None = None
    n_feasible = 0
    for schedule in grid.schedules():
        if not classify(instance, schedule).feasible:
            continue
        n_feasible += 1
        breakdown, zeta = score(instance, schedule, cell_cost=cell_cost)
        if best is None or breakdown.total < best[0]:
            best = (breakdown.total, schedule, breakdown, zeta)

    timings = {"total_s": time.perf_counter() - started}
    detail = {"n_candidates": n_candidates, "n_feasible": n_feasible}
    if best is None:
        return Result(
            solver=NAME, status=INFEASIBLE, schedule=None, zeta=None,
            objective=None, bracket=None, feasibility=None,
            timings=timings, detail=detail,
        )
    total, schedule, breakdown, zeta = best
    return Result(
        solver=NAME, status=OPTIMAL, schedule=schedule, zeta=zeta,
        objective=breakdown, bracket=Bracket(lower=total, upper=total),
        feasibility=classify(instance, schedule),
        timings=timings, detail=detail,
    )


class BruteForce:
    """``brute_force`` behind the ``Solver`` protocol. The budget is not used."""

    name = NAME

    def solve(self, instance: Instance, budget: Budget) -> Result:
        return brute_force(instance)
```

- [ ] **Step 5: Give the approximand a per-cell table**

In `src/quantic/encoding/expand.py`, inside `ApproximandReport`, insert this method between `temporary_cost` and `describe`:

```python
    def cell_cost(self, i: int, t: int, lots: int) -> float:
        """The substituted temporary impact of one cell trading ``lots``.

        What an exact solver minimises when it is asked for the optimum of
        the function the encoder declares, rather than of the power law.
        """
        if (i, t) in self.coefficients:
            b, c = self.coefficients[(i, t)]
            x = float(lots)
            return b * x + c * x * x
        if (i, t) not in self.exact:
            raise KeyError(
                f"the cell at (asset {i}, bucket {t}) is in neither the fitted "
                "coefficients nor the exact tables of this report"
            )
        table = self.exact[(i, t)]
        if lots not in table:
            raise ValueError(
                f"{lots} is not one of the values {tuple(table)} the cell at "
                f"(asset {i}, bucket {t}) can take, so the encoder has no cost "
                "for it"
            )
        return table[lots]
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_brute_force.py tests/encoding/test_approximand.py tests/encoding/test_expand.py -q`
Expected: every test passes; `test_brute_force.py` has 29.

- [ ] **Step 7: Lint and commit**

Run: `.venv/Scripts/ruff.exe check src tests` and `.venv/Scripts/lint-imports.exe`
Expected: clean, `Contracts: 2 kept, 0 broken.`

```bash
git add src/quantic/solvers/classical/scoring.py src/quantic/solvers/classical/brute_force.py src/quantic/encoding/expand.py tests/solvers/test_brute_force.py
git commit -m "feat(solvers): the optimum by enumeration, and the evaluator it answers to"
```

---

## Task 4: The program's pieces

**Files:**
- Create: `src/quantic/solvers/classical/milp_model.py`
- Create: `tests/solvers/test_milp_model.py`

**Interfaces:**
- Consumes: `LotGrid`, `lot_grid` (Task 2); `CellCost` (Task 3); `evaluate`; `ScenarioSet.losses` (in tests).
- Produces:
  - `Columns(asset, bucket, lots, first, count)` — arrays; `.n -> int`; `.schedule(y: np.ndarray) -> Schedule`
  - `columns_of(grid: LotGrid) -> Columns`
  - `cell_costs(instance, columns, *, cell_cost: CellCost | None = None) -> np.ndarray` — shape `(columns.n,)`, spread plus temporary impact
  - `permanent_constant(instance) -> float`
  - `loss_model(instance) -> tuple[np.ndarray, np.ndarray]` — `base` of shape `(S,)` and `tail` of shape `(S, N, T)`, with `L_s = base_s − Σ x[i,t] · tail[s,i,t]`
  - `variance_gradient(instance, schedule) -> np.ndarray` — shape `(N, T)`
  - `coupling_rows(columns, grid, n_vars) -> tuple[sparse.csr_array, np.ndarray, np.ndarray]` — the matrix, its lower bounds, its upper bounds

**Why the costs are read off the evaluator.** Spread and temporary impact are sums over cells, so what one cell costs is what `evaluate` charges a schedule that trades in that cell alone. That takes one call per column and leaves no second implementation of the impact law. The loss model and the gradient cannot be read off that way, so each has a test that ties it to `ScenarioSet.losses` and to `evaluate` — at the schedule that sells nothing as well as at feasible ones, because a loss is a coefficient times a holding and a feasible schedule is blind to the cells it no longer holds. That was M3a's first review, finding I1.

- [ ] **Step 1: Write the failing test**

Create `tests/solvers/test_milp_model.py`:

```python
import numpy as np
import pytest

from quantic.problem.objective import evaluate
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.milp_model import (
    cell_costs,
    columns_of,
    coupling_rows,
    loss_model,
    permanent_constant,
    variance_gradient,
)
from tests.solvers.instances import (
    EVERY_SOLVABLE,
    concave,
    cvar,
    discrete,
    plain,
)

REL = 1e-9


def _zeta(inst):
    return 0.0 if isinstance(inst.risk, CVaRRisk) else None


def _selected(columns, schedule):
    """The 0/1 vector that picks ``schedule``'s value in every cell."""
    y = np.zeros(columns.n)
    for c in range(columns.n):
        i, t = int(columns.asset[c]), int(columns.bucket[c])
        y[c] = float(schedule.lots[i][t] == int(columns.lots[c]))
    return y


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_every_cell_has_one_column_per_value_it_may_take(build):
    grid = lot_grid(build())
    columns = columns_of(grid)
    for i in range(grid.n_assets):
        for t in range(grid.n_buckets):
            mine = (columns.asset == i) & (columns.bucket == t)
            assert tuple(columns.lots[mine]) == grid.values[i][t]
            assert columns.count[i, t] == len(grid.values[i][t])
            assert tuple(
                columns.lots[columns.first[i, t]:columns.first[i, t] + columns.count[i, t]]
            ) == grid.values[i][t]


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_columns_read_back_the_schedule_that_selected_them(build):
    inst = build()
    columns = columns_of(lot_grid(inst))
    for schedule in lot_grid(inst).schedules():
        assert columns.schedule(_selected(columns, schedule)) == schedule


def test_a_solution_a_hair_off_integer_still_reads_back():
    """HiGHS returns 0.9999999 and 1e-9, not 1 and 0."""
    inst = plain()
    columns = columns_of(lot_grid(inst))
    y = _selected(columns, inst.witness)
    noisy = np.where(y > 0.5, y - 1e-7, y + 1e-9)
    assert columns.schedule(noisy) == inst.witness


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_costs_add_up_to_what_the_evaluator_charges(build):
    inst = build()
    columns = columns_of(lot_grid(inst))
    costs = cell_costs(inst, columns)
    for schedule in lot_grid(inst).schedules():
        scored = evaluate(inst, schedule, zeta=_zeta(inst))
        assert costs @ _selected(columns, schedule) == pytest.approx(
            scored.spread_cost + scored.temporary_impact_cost, rel=REL
        )


def test_a_cost_table_replaces_the_temporary_impact_in_the_columns():
    inst = concave()
    columns = columns_of(lot_grid(inst))
    plain_costs = cell_costs(inst, columns)
    free = cell_costs(inst, columns, cell_cost=lambda i, t, lots: 0.0)
    a = inst.params.assets
    spread = np.array(
        [
            a[i].half_spread * lots * inst.assets[i].lot_size
            for i, lots in zip(columns.asset, columns.lots, strict=True)
        ]
    )
    assert free == pytest.approx(spread, rel=REL)
    assert np.all(plain_costs >= free)


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_permanent_impact_is_the_same_at_every_schedule_that_liquidates(build):
    inst = build()
    constant = permanent_constant(inst)
    for schedule in lot_grid(inst).schedules():
        assert evaluate(
            inst, schedule, zeta=_zeta(inst)
        ).permanent_impact_cost == pytest.approx(constant, rel=REL)


def test_permanent_impact_is_not_constant_once_liquidation_is_partial():
    """The identity is about full liquidation; the constant must not be used without it."""
    inst = plain()
    short = Schedule(lots=((2, 1, 0), (1, 1, 1)))
    assert evaluate(inst, short).permanent_impact_cost != pytest.approx(
        permanent_constant(inst), rel=1e-3
    )


def test_the_loss_model_is_the_scenario_sets():
    """At the witness, at the schedule that sells nothing, and everywhere between.

    A loss is a coefficient times a holding, so a feasible schedule alone is
    blind to every cell it no longer holds.
    """
    inst = cvar()
    base, tail = loss_model(inst)
    unsold = Schedule.from_array(np.zeros((inst.n_assets, inst.n_buckets), dtype=int))
    for schedule in (inst.witness, unsold, *lot_grid(inst).schedules()):
        x = schedule.as_array().astype(float)
        mine = base - np.einsum("snt,nt->s", tail, x)
        theirs = inst.risk.scenarios.losses(inst, schedule)
        assert mine == pytest.approx(theirs, rel=REL, abs=REL * np.abs(theirs).max())


def test_the_gradient_is_the_slope_of_the_evaluators_variance():
    """A quadratic's central difference is exact, so this is an equality."""
    inst = plain()
    at = Schedule(lots=((2, 1, 1), (1, 1, 1)))
    gradient = variance_gradient(inst, at)
    for i in range(inst.n_assets):
        for t in range(inst.n_buckets):
            up = at.as_array().copy()
            down = at.as_array().copy()
            up[i, t] += 1
            down[i, t] -= 1
            slope = (
                evaluate(inst, Schedule.from_array(up)).risk_cost
                - evaluate(inst, Schedule.from_array(down)).risk_cost
            ) / 2.0
            assert gradient[i, t] == pytest.approx(slope, rel=1e-7)


def test_selling_sooner_never_adds_variance():
    """A lot sold in a bucket is a holding given up from that bucket on.

    The last bucket's entry is zero at a schedule that liquidates: nothing is
    held after it, so there is nothing left to give up.
    """
    gradient = variance_gradient(plain(), plain().witness)
    assert np.all(gradient <= 0.0)
    assert np.all(gradient[:, 0] < gradient[:, 1])
    assert np.all(gradient[:, -1] == 0.0)


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_rows_hold_at_every_schedule_on_the_grid_and_at_no_other(build):
    """One value a cell, the whole position sold, and cardinality where it binds."""
    inst = build()
    grid = lot_grid(inst)
    columns = columns_of(grid)
    matrix, lower, upper = coupling_rows(columns, grid, columns.n)
    for schedule in grid.schedules():
        held = matrix @ _selected(columns, schedule)
        inside = bool(np.all((held >= lower - 1e-9) & (held <= upper + 1e-9)))
        within = grid.cardinality is None or all(
            schedule.active_names(t) <= grid.cardinality
            for t in range(grid.n_buckets)
        )
        assert inside == within


def test_the_rows_refuse_a_schedule_that_leaves_a_lot_unsold():
    inst = plain()
    grid = lot_grid(inst)
    columns = columns_of(grid)
    matrix, lower, upper = coupling_rows(columns, grid, columns.n)
    short = Schedule(lots=((2, 1, 0), (1, 1, 1)))
    held = matrix @ _selected(columns, short)
    assert not np.all((held >= lower - 1e-9) & (held <= upper + 1e-9))


def _n_rows(grid):
    columns = columns_of(grid)
    return coupling_rows(columns, grid, columns.n)[0].shape[0]


def test_cardinality_adds_one_row_a_bucket():
    with_it = lot_grid(discrete())
    without = lot_grid(plain())
    cells = with_it.n_assets * with_it.n_buckets
    assert _n_rows(without) == cells + without.n_assets
    assert _n_rows(with_it) == cells + with_it.n_assets + with_it.n_buckets
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_milp_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.solvers.classical.milp_model'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/solvers/classical/milp_model.py`:

```python
"""The pieces of the exact program -- spec section 7.3.

One binary ``y[i,t,k]`` per admissible lot value ``k`` of each cell. Any cost
that depends on one cell alone is then exactly linear in the binaries, however
concave it is in the lots: the piecewise-linear under- and over-estimators of
section 7.3 coincide on the lot grid.

Each piece here is a claim about the objective, and each is tested against
``problem.objective.evaluate`` or ``ScenarioSet.losses`` on its own, before any
solver sees it. Where a piece can be read off the evaluator it is, so that no
second implementation exists to drift; where it cannot -- the loss model, the
gradient -- the test is what ties it back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from quantic.problem.instance import Instance
from quantic.problem.objective import evaluate
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.classical.lot_grid import LotGrid
from quantic.solvers.classical.scoring import CellCost


@dataclass(frozen=True)
class Columns:
    """Which cell and which lot value each binary stands for."""

    asset: np.ndarray     # (n_y,)
    bucket: np.ndarray    # (n_y,)
    lots: np.ndarray      # (n_y,)
    first: np.ndarray     # (N, T) index of the cell's first column
    count: np.ndarray     # (N, T) how many columns the cell has

    @property
    def n(self) -> int:
        return int(self.lots.size)

    def schedule(self, y: np.ndarray) -> Schedule:
        """The schedule a solution's binaries select."""
        n_assets, n_buckets = self.first.shape
        grid = [[0] * n_buckets for _ in range(n_assets)]
        for i in range(n_assets):
            for t in range(n_buckets):
                lo = int(self.first[i, t])
                hi = lo + int(self.count[i, t])
                grid[i][t] = int(self.lots[lo + int(np.argmax(y[lo:hi]))])
        return Schedule(lots=tuple(tuple(row) for row in grid))


def columns_of(grid: LotGrid) -> Columns:
    asset, bucket, lots = [], [], []
    first = np.zeros((grid.n_assets, grid.n_buckets), dtype=np.int64)
    count = np.zeros((grid.n_assets, grid.n_buckets), dtype=np.int64)
    for i in range(grid.n_assets):
        for t in range(grid.n_buckets):
            first[i, t] = len(lots)
            count[i, t] = len(grid.values[i][t])
            for k in grid.values[i][t]:
                asset.append(i)
                bucket.append(t)
                lots.append(k)
    return Columns(
        asset=np.array(asset, dtype=np.int64),
        bucket=np.array(bucket, dtype=np.int64),
        lots=np.array(lots, dtype=np.int64),
        first=first,
        count=count,
    )


def cell_costs(
    instance: Instance, columns: Columns, *, cell_cost: CellCost | None = None
) -> np.ndarray:
    """Spread plus temporary impact of each column, read off the evaluator.

    Both terms are sums over cells, so what one cell costs is what the
    evaluator charges a schedule that trades in that cell alone. No second
    implementation of either formula exists to drift away from the first.
    """
    zeta = 0.0 if isinstance(instance.risk, CVaRRisk) else None
    out = np.zeros(columns.n)
    for c in range(columns.n):
        i, t, k = int(columns.asset[c]), int(columns.bucket[c]), int(columns.lots[c])
        if k == 0:
            continue
        lots = [[0] * instance.n_buckets for _ in range(instance.n_assets)]
        lots[i][t] = k
        alone = evaluate(
            instance, Schedule(lots=tuple(tuple(r) for r in lots)), zeta=zeta
        )
        temporary = (
            alone.temporary_impact_cost if cell_cost is None else cell_cost(i, t, k)
        )
        out[c] = alone.spread_cost + temporary
    return out


def permanent_constant(instance: Instance) -> float:
    """Permanent impact under full liquidation, which no schedule can change.

    Per asset the term is ``sum_t (sum_{s<t} q_s + q_t / 2) * q_t``, and that
    sum is ``(sum_t q_t)**2 / 2``. Selling everything in the first bucket is
    one schedule that liquidates in full, so the evaluator's figure there is
    the figure everywhere.
    """
    lots = tuple(
        (int(x), *([0] * (instance.n_buckets - 1))) for x in instance.initial_lots
    )
    zeta = 0.0 if isinstance(instance.risk, CVaRRisk) else None
    return float(
        evaluate(instance, Schedule(lots=lots), zeta=zeta).permanent_impact_cost
    )


def loss_model(instance: Instance) -> tuple[np.ndarray, np.ndarray]:
    """``L_s = base_s - sum over cells of x[i,t] * tail[s,i,t]``.

    A loss is linear in the holdings, and a lot sold in bucket ``t`` leaves
    every holding from ``t`` on, so its coefficient is the sum of the
    per-holding coefficients from ``t`` to the horizon.
    """
    risk = instance.risk
    returns = np.asarray(risk.scenarios.returns, dtype=float)          # (S, T, N)
    per_lot = instance.lot_sizes().astype(float) * instance.params.prices()
    per_holding = np.einsum("stn,n->snt", -returns, per_lot)           # (S, N, T)
    position = np.asarray(instance.initial_lots, dtype=float)
    base = np.einsum("snt,n->s", per_holding, position)
    tail = np.cumsum(per_holding[:, :, ::-1], axis=2)[:, :, ::-1]
    return base, tail


def variance_gradient(instance: Instance, schedule: Schedule) -> np.ndarray:
    """``d(lam * sum_t h_t' Sigma h_t) / d x[i,t]``, in currency per lot."""
    lam = float(instance.risk.lam)
    lot = instance.lot_sizes().astype(float)
    sigma = np.asarray(instance.params.price_covariance(), dtype=float)
    holdings = instance.remaining_lots(schedule).astype(float) * lot[:, None]
    pull = sigma @ holdings                                            # (N, T)
    tail = np.cumsum(pull[:, ::-1], axis=1)[:, ::-1]
    return -2.0 * lam * lot[:, None] * tail


def coupling_rows(columns: Columns, grid: LotGrid, n_vars: int):
    """The constraints that hold whatever the risk term is."""
    n_assets, n_buckets = columns.first.shape
    a_rows, a_cols, a_vals, lower, upper = [], [], [], [], []
    row = 0

    def add(cols, vals, lo, hi):
        nonlocal row
        a_rows.extend([row] * len(cols))
        a_cols.extend(int(c) for c in cols)
        a_vals.extend(float(v) for v in vals)
        lower.append(lo)
        upper.append(hi)
        row += 1

    every = np.arange(columns.n)
    for i in range(n_assets):
        for t in range(n_buckets):
            mine = every[(columns.asset == i) & (columns.bucket == t)]
            add(mine, np.ones(mine.size), 1.0, 1.0)

    for i in range(n_assets):
        mine = every[columns.asset == i]
        position = float(grid.initial_lots[i])
        add(mine, columns.lots[mine], position, position)

    if grid.cardinality is not None:
        for t in range(n_buckets):
            mine = every[(columns.bucket == t) & (columns.lots > 0)]
            add(mine, np.ones(mine.size), -np.inf, float(grid.cardinality))

    matrix = sparse.csr_array(
        (a_vals, (a_rows, a_cols)), shape=(row, n_vars), dtype=float
    )
    return matrix, np.array(lower), np.array(upper)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_milp_model.py -q`
Expected: every test in the file passes (53).

- [ ] **Step 5: Lint and commit**

Run: `.venv/Scripts/ruff.exe check src tests` and `.venv/Scripts/lint-imports.exe`
Expected: clean, `Contracts: 2 kept, 0 broken.`

```bash
git add src/quantic/solvers/classical/milp_model.py tests/solvers/test_milp_model.py
git commit -m "feat(solvers): the exact program's pieces, each against the evaluator"
```

---

## Task 5: The exact program

**Files:**
- Create: `src/quantic/solvers/classical/exact_pwl_milp.py`
- Create: `tests/solvers/test_exact_pwl_milp.py`
- Modify: `pyproject.toml` (append one contract)

**Interfaces:**
- Consumes: everything `milp_model` produces (Task 4); `lot_grid` (Task 2); `score`, `CellCost` (Task 3); `brute_force` (Task 3, in tests); `Result`, `Bracket`, `Budget` and the three statuses (Task 1).
- Produces:
  - `exact_pwl_milp(instance, *, budget: Budget | None = None, cell_cost: CellCost | None = None) -> Result`; `Result.detail` carries `rounds`, `n_cuts`, `n_binaries`, `presolve_retries`
  - `ExactPwlMilp` — `name = "exact_pwl_milp"`, `solve(instance, budget) -> Result`

**Three things the dry run established, all of them in the code below.**

1. *The loop must stop when a schedule comes back a second time.* HiGHS satisfies a cut to `1e-7`; the convergence test is at `1e-9`. Without the `cut_at` set the same schedule returns with the same gap and the same cut is added for ever. `test_it_finds_what_enumeration_finds[discrete]` hangs without it.
2. *Presolve fails on some small programs.* `T0-D1+D2` at its fourth round returns status 4, "Solve error", with presolve on and solves with it off. A failed round is retried once without presolve.
3. *`mip_rel_gap` must be zero.* The default of `1e-4` leaves a bracket wider than the gap between the two best schedules of `T0-D2+D3`, which is `0.0069` on `30.8`.

**The tests monkeypatch names in this module's own namespace** — `module.milp`, `module.variance_gradient`, `module.cell_costs`, `module.lot_grid`. Keep the `from … import` lines as written: importing the modules instead and calling `milp_model.cell_costs(...)` would leave the patches with nothing to replace.

- [ ] **Step 1: Write the failing test**

Create `tests/solvers/test_exact_pwl_milp.py`:

```python
import types

import pytest

from quantic.encoding.encode import encode
from quantic.encoding.integer import BinaryEncoding
from quantic.problem.objective import evaluate, optimal_zeta
from quantic.problem.risk import CVaRRisk
from quantic.solvers.base import INFEASIBLE, OPTIMAL, TIME_LIMIT, Budget, Solver
from quantic.solvers.classical import exact_pwl_milp as module
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.exact_pwl_milp import ExactPwlMilp, exact_pwl_milp
from tests.solvers.instances import (
    EVERY_SOLVABLE,
    concave,
    cvar,
    cvar_without_risk_aversion,
    discrete,
    infeasible,
    no_risk_aversion,
    plain,
)

REL = 1e-9


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_it_finds_what_enumeration_finds(build):
    inst = build()
    by_milp = exact_pwl_milp(inst)
    by_hand = brute_force(inst)
    assert by_milp.status == OPTIMAL
    assert by_milp.feasibility.feasible
    assert by_milp.objective.total == pytest.approx(by_hand.objective.total, rel=REL)


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_bracket_closes_on_the_evaluators_value(build):
    inst = build()
    result = exact_pwl_milp(inst)
    zeta = optimal_zeta(inst, result.schedule) if isinstance(inst.risk, CVaRRisk) else None
    assert result.bracket.upper == evaluate(inst, result.schedule, zeta=zeta).total
    assert result.bracket.width <= 1e-6 * max(1.0, abs(result.bracket.upper))
    assert result.zeta == zeta


def test_variance_takes_more_than_one_round_and_cvar_takes_one():
    assert exact_pwl_milp(plain()).detail["rounds"] > 1
    assert exact_pwl_milp(plain()).detail["n_cuts"] == (
        exact_pwl_milp(plain()).detail["rounds"] - 1
    )
    assert exact_pwl_milp(cvar()).detail["rounds"] == 1


@pytest.mark.parametrize("build", [no_risk_aversion, cvar_without_risk_aversion])
def test_with_no_risk_aversion_there_is_no_risk_in_the_program(build):
    result = exact_pwl_milp(build())
    assert result.detail["rounds"] == 1
    assert result.detail["n_cuts"] == 0
    assert result.objective.risk_cost == 0.0


def test_an_instance_nobody_can_solve_is_reported_and_not_raised():
    result = exact_pwl_milp(infeasible())
    assert result.status == INFEASIBLE
    assert result.schedule is None
    assert result.bracket is None


def test_with_the_encoders_table_it_finds_the_optimum_enumeration_finds():
    inst = concave()
    report = encode(inst, BinaryEncoding()).breakdown.approximand
    by_milp = exact_pwl_milp(inst, cell_cost=report.cell_cost)
    by_hand = brute_force(inst, cell_cost=report.cell_cost)
    assert by_milp.objective.total == pytest.approx(by_hand.objective.total, rel=REL)
    assert by_milp.objective.temporary_impact_cost == pytest.approx(
        report.temporary_cost(by_milp.schedule), rel=REL
    )


def test_a_cost_table_that_punishes_a_cell_moves_the_optimum_off_it():
    inst = plain()
    usual = exact_pwl_milp(inst).schedule
    i, t = next(
        (i, t) for i in range(inst.n_assets) for t in range(inst.n_buckets)
        if usual.lots[i][t] > 0
    )
    punished = exact_pwl_milp(
        inst,
        cell_cost=lambda a, b, lots: 1e6 if (a, b) == (i, t) and lots else 0.0,
    ).schedule
    assert punished.lots[i][t] == 0


def test_a_budget_already_spent_returns_without_a_schedule():
    result = exact_pwl_milp(plain(), budget=Budget(wall_clock_s=1e-12))
    assert result.status == TIME_LIMIT
    assert result.schedule is None
    assert result.bracket is None


def test_running_out_of_time_keeps_the_best_schedule_and_an_honest_bracket(monkeypatch):
    """HiGHS stops at its limit with an incumbent and a bound that has not met it."""
    real = module.milp

    def out_of_time(**kwargs):
        solved = real(**kwargs)
        return types.SimpleNamespace(
            status=1, x=solved.x, fun=solved.fun, message="time limit",
            mip_dual_bound=solved.mip_dual_bound - 5.0,
        )

    monkeypatch.setattr(module, "milp", out_of_time)
    inst = plain()
    result = exact_pwl_milp(inst, budget=Budget(wall_clock_s=60.0))
    assert result.status == TIME_LIMIT
    assert result.feasibility.feasible
    assert result.bracket.upper == evaluate(inst, result.schedule).total
    assert result.bracket.lower < brute_force(inst).objective.total
    assert result.bracket.width > 0.0
    assert result.detail["rounds"] == 1


def test_when_presolve_fails_the_program_is_solved_without_it(monkeypatch):
    """HiGHS's presolve gives up on some small programs once they carry cuts."""
    real = module.milp
    seen = []

    def presolve_fails(**kwargs):
        seen.append(dict(kwargs["options"]))
        if kwargs["options"].get("presolve", True):
            return types.SimpleNamespace(
                status=4, x=None, fun=None, message="Solve error", mip_dual_bound=None
            )
        return real(**kwargs)

    monkeypatch.setattr(module, "milp", presolve_fails)
    inst = plain()
    result = exact_pwl_milp(inst)
    assert result.status == OPTIMAL
    assert result.objective.total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )
    assert result.detail["presolve_retries"] == result.detail["rounds"]
    assert {"presolve" in options for options in seen} == {True, False}


def test_a_program_highs_cannot_solve_either_way_is_raised(monkeypatch):
    monkeypatch.setattr(
        module, "milp",
        lambda **kwargs: types.SimpleNamespace(
            status=4, x=None, fun=None, message="Solve error", mip_dual_bound=None
        ),
    )
    with pytest.raises(RuntimeError, match="Solve error"):
        exact_pwl_milp(plain())


def test_a_cut_with_the_wrong_sign_cannot_pass_for_an_answer(monkeypatch):
    """The mutation the gate is asked to catch, at a size that runs in a test.

    An invalid cut either removes the optimum, and the value no longer
    matches enumeration, or leaves a bound above the value attained, and the
    bracket refuses to be built.
    """
    real = module.variance_gradient
    monkeypatch.setattr(module, "variance_gradient", lambda i, s: -real(i, s))
    inst = plain()
    try:
        result = exact_pwl_milp(inst)
    except (ValueError, RuntimeError):
        return
    assert result.status != OPTIMAL or result.objective.total != pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )


def test_cardinality_is_in_the_program():
    """Dropping it returns a schedule that trades two names in a bucket."""
    inst = discrete()
    result = exact_pwl_milp(inst)
    assert all(result.schedule.active_names(t) <= 1 for t in range(inst.n_buckets))


def test_it_answers_to_the_solver_protocol():
    solver = ExactPwlMilp()
    assert isinstance(solver, Solver)
    inst = plain()
    assert solver.solve(inst, Budget()).schedule == exact_pwl_milp(inst).schedule
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_exact_pwl_milp.py -q`
Expected: FAIL — `ImportError: cannot import name 'exact_pwl_milp' from 'quantic.solvers.classical'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/solvers/classical/exact_pwl_milp.py`:

```python
"""The exact optimum through a mixed-integer linear program -- spec section 7.3.

``milp_model`` builds the program's pieces; this module solves it and says
what was proven.

Three things in the objective are not per-cell. Permanent impact sums to half
the square of the position, so under full liquidation it is a constant. CVaR
is Rockafellar-Uryasev, which is a linear program as written. Variance is
convex quadratic, which HiGHS does not take, and goes through outer
approximation: a tangent cut at each schedule the program returns, until the
program's value for the risk meets the evaluator's. A cut is tight at the
schedule it was taken at, so no schedule is cut twice and the loop ends.

The program's own account of what its schedule costs is never reported.
``Bracket.upper`` is the evaluator's total at the schedule, and
``Bracket.lower`` is the program's dual bound.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.base import INFEASIBLE, OPTIMAL, TIME_LIMIT, Bracket, Budget, Result
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.milp_model import (
    cell_costs,
    columns_of,
    coupling_rows,
    loss_model,
    permanent_constant,
    variance_gradient,
)
from quantic.solvers.classical.scoring import CellCost, score

NAME = "exact_pwl_milp"

# The risk the program reports and the risk the evaluator reports, relative to
# the objective. Below this the outer approximation has converged.
_CUT_RTOL = 1e-9
_MAX_ROUNDS = 10_000

# scipy.optimize.milp's status codes.
_SCIPY_OPTIMAL = 0
_SCIPY_LIMIT = 1
_SCIPY_INFEASIBLE = 2
_SCIPY_ERROR = 4


def exact_pwl_milp(
    instance: Instance,
    *,
    budget: Budget | None = None,
    cell_cost: CellCost | None = None,
) -> Result:
    """Minimise the objective over the lot grid, with a proven lower bound."""
    started = time.perf_counter()
    limit = None if budget is None else budget.wall_clock_s
    risk = instance.risk
    if not isinstance(risk, (VarianceRisk, CVaRRisk)):
        raise NotImplementedError(
            f"exact_pwl_milp does not know risk spec {risk.name!r} "
            f"({type(risk).__name__}). Dispatch is on type, never on a lam attribute"
        )

    grid = lot_grid(instance)
    columns = columns_of(grid)
    n_y = columns.n
    lam = float(risk.lam)
    cvar = isinstance(risk, CVaRRisk) and lam > 0.0
    variance = isinstance(risk, VarianceRisk) and lam > 0.0

    n_scenarios = risk.scenarios.n_scenarios if cvar else 0
    # y, then zeta and one u per scenario, or one theta.
    n_vars = n_y + (1 + n_scenarios if cvar else 0) + (1 if variance else 0)

    cost = np.zeros(n_vars)
    cost[:n_y] = cell_costs(instance, columns, cell_cost=cell_cost)
    constant = permanent_constant(instance)

    lower_bounds = np.zeros(n_vars)
    upper_bounds = np.ones(n_vars)
    upper_bounds[n_y:] = np.inf
    integrality = np.zeros(n_vars)
    integrality[:n_y] = 1

    matrix, row_lower, row_upper = coupling_rows(columns, grid, n_vars)
    blocks = [matrix]
    row_lowers = [row_lower]
    row_uppers = [row_upper]

    if cvar:
        zeta_at = n_y
        lower_bounds[zeta_at] = -np.inf
        weights = np.asarray(risk.scenarios.weights, dtype=float)
        cost[zeta_at] = lam
        cost[zeta_at + 1:] = lam * weights / (1.0 - risk.alpha)
        base, tail = loss_model(instance)
        # u_s + zeta - L_s >= 0, with L_s = base_s - sum x * tail.
        per_column = tail[:, columns.asset, columns.bucket] * columns.lots[None, :]
        excess = np.zeros((n_scenarios, n_vars))
        excess[:, :n_y] = per_column
        excess[:, zeta_at] = 1.0
        excess[np.arange(n_scenarios), zeta_at + 1 + np.arange(n_scenarios)] = 1.0
        blocks.append(sparse.csr_array(excess))
        row_lowers.append(base)
        row_uppers.append(np.full(n_scenarios, np.inf))

    if variance:
        cost[n_y] = 1.0

    best: tuple[float, Schedule] | None = None
    proven = -np.inf
    rounds = 0
    n_cuts = 0
    presolve_retries = 0
    cut_at: set[Schedule] = set()
    status = OPTIMAL

    while True:
        rounds += 1
        options = {"mip_rel_gap": 0.0}
        if limit is not None:
            left = limit - (time.perf_counter() - started)
            if left <= 0.0:
                status = TIME_LIMIT
                break
            options["time_limit"] = left

        program = {
            "c": cost,
            "integrality": integrality,
            "bounds": Bounds(lower_bounds, upper_bounds),
            "constraints": LinearConstraint(
                sparse.vstack(blocks, format="csr"),
                np.concatenate(row_lowers),
                np.concatenate(row_uppers),
            ),
        }
        solved = milp(**program, options=options)
        if solved.status == _SCIPY_ERROR:
            # HiGHS's presolve fails outright on some small programs once they
            # carry cuts -- T0-D1+D2 at the fourth round is one. The program
            # itself is sound: without presolve the same solver solves it.
            presolve_retries += 1
            solved = milp(**program, options={**options, "presolve": False})

        if solved.status == _SCIPY_INFEASIBLE:
            return Result(
                solver=NAME, status=INFEASIBLE, schedule=None, zeta=None,
                objective=None, bracket=None, feasibility=None,
                timings={"total_s": time.perf_counter() - started},
                detail={
        "rounds": rounds, "n_cuts": n_cuts, "n_binaries": n_y,
        "presolve_retries": presolve_retries,
    },
            )
        if solved.status not in (_SCIPY_OPTIMAL, _SCIPY_LIMIT):
            raise RuntimeError(
                f"HiGHS stopped on {instance.instance_id} with status "
                f"{solved.status}: {solved.message}"
            )

        bound = getattr(solved, "mip_dual_bound", None)
        if bound is not None and np.isfinite(bound):
            proven = max(proven, float(bound) + constant)

        if solved.x is not None:
            schedule = columns.schedule(solved.x[:n_y])
            breakdown, _ = score(instance, schedule, cell_cost=cell_cost)
            if best is None or breakdown.total < best[0]:
                best = (breakdown.total, schedule)

        if solved.status == _SCIPY_LIMIT:
            status = TIME_LIMIT
            break
        if not variance:
            break

        reported = float(solved.x[n_y])
        gap = breakdown.risk_cost - reported
        if gap <= _CUT_RTOL * max(1.0, abs(breakdown.total)):
            break
        if schedule in cut_at:
            # The cut at this schedule is already in the program and is tight
            # there, so what is left of the gap is HiGHS's own feasibility
            # tolerance. Adding the same cut again would change nothing, and
            # the loop would not end.
            break
        if rounds >= _MAX_ROUNDS:
            raise RuntimeError(
                f"outer approximation on {instance.instance_id} has not converged "
                f"in {_MAX_ROUNDS} rounds; the last gap was {gap!r}"
            )

        # theta >= v(x^) + g . (x - x^), with x linear in the binaries.
        gradient = variance_gradient(instance, schedule)
        at = schedule.as_array().astype(float)
        cut = np.zeros((1, n_vars))
        cut[0, :n_y] = -gradient[columns.asset, columns.bucket] * columns.lots
        cut[0, n_y] = 1.0
        blocks.append(sparse.csr_array(cut))
        row_lowers.append(np.array([breakdown.risk_cost - float((gradient * at).sum())]))
        row_uppers.append(np.array([np.inf]))
        cut_at.add(schedule)
        n_cuts += 1

    timings = {"total_s": time.perf_counter() - started}
    detail = {
        "rounds": rounds, "n_cuts": n_cuts, "n_binaries": n_y,
        "presolve_retries": presolve_retries,
    }
    if best is None:
        return Result(
            solver=NAME, status=TIME_LIMIT, schedule=None, zeta=None,
            objective=None, bracket=None, feasibility=None,
            timings=timings, detail=detail,
        )

    total, schedule = best
    breakdown, zeta = score(instance, schedule, cell_cost=cell_cost)
    return Result(
        solver=NAME, status=status, schedule=schedule, zeta=zeta,
        objective=breakdown,
        bracket=Bracket(lower=min(proven, total) if status == TIME_LIMIT else proven,
                        upper=total),
        feasibility=classify(instance, schedule),
        timings=timings, detail=detail,
    )


class ExactPwlMilp:
    """``exact_pwl_milp`` behind the ``Solver`` protocol."""

    name = NAME

    def solve(self, instance: Instance, budget: Budget) -> Result:
        return exact_pwl_milp(instance, budget=budget)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_exact_pwl_milp.py -q`
Expected: every test in the file passes (31), in a few seconds. If the run does not finish, the `cut_at` check is missing or misplaced.

- [ ] **Step 5: Add the second import contract**

Append to `pyproject.toml`:

```toml

[[tool.importlinter.contracts]]
name = "The exact baseline does not read the encoder (M3b design section 6)"
type = "forbidden"
source_modules = [
    "quantic.solvers.base",
    "quantic.solvers.classical.lot_grid",
    "quantic.solvers.classical.scoring",
    "quantic.solvers.classical.brute_force",
    "quantic.solvers.classical.milp_model",
    "quantic.solvers.classical.exact_pwl_milp",
]
forbidden_modules = ["quantic.encoding"]
```

The gate compares a QUBO's minimum with the baseline's. A baseline that read anything from the encoder would inherit its mistakes and agree with them.

- [ ] **Step 6: Lint and commit**

Run: `.venv/Scripts/ruff.exe check src tests` and `.venv/Scripts/lint-imports.exe`
Expected: clean, `Contracts: 3 kept, 0 broken.`

```bash
git add src/quantic/solvers/classical/exact_pwl_milp.py tests/solvers/test_exact_pwl_milp.py pyproject.toml
git commit -m "feat(solvers): the exact optimum through HiGHS, with a proven bracket"
```

---

## Task 6: The gadget, minimised

**Files:**
- Create: `src/quantic/solvers/classical/gadget_minimum.py`
- Create: `tests/solvers/test_gadget_minimum.py`

**Interfaces:**
- Consumes: `EncodedInstance` and its `.gadget` (`CVaRGadget`: `.zeta`, `.lam`, `.alpha`, `.weights`, `.penalty`, `.loss_coefficients`, `.cheapest_aux`, `.encoded_cvar`, `.losses_from`), `.layout`, `.registry`, `.instance`; `lot_grid` (Task 2, in tests).
- Produces `GadgetMinimiser(enc: EncodedInstance)` with:
  - `.gadget`, `.lam`, `.step`, `.lo`, `.top`, `.penalty`, `.weights`, `.alpha`, `.slack`
  - `.base: np.ndarray` of shape `(S,)` and `.loss_matrix: np.ndarray` of shape `(S, n_vars)`, with `L = base + bits @ loss_matrix.T`
  - `.continuous(losses, levels) -> np.ndarray` — `λ` times Rockafellar-Uryasev; `losses` is `(B, S)`, `levels` is `(B, W)`, the result `(B, W)`
  - `.charge(losses, levels) -> np.ndarray` — the gadget's exact energy at each level, auxiliaries at their best; `(B, W)`
  - `.minimise(losses) -> tuple[np.ndarray, np.ndarray, int]` — per state the least charge and the level that gives it, and the widest window any state needed

**What makes the scan a proof.** `charge(j) ≥ continuous(j) − slack` at every level of every state: relax each scenario's step count to a real and minimise. `continuous` is convex in the level. So a window is wide enough once, at each edge, either the edge is the end of the grid or `continuous` there exceeds both its value at the centre and the best charge inside plus `slack`. Beyond such an edge `continuous` only rises, and nothing out there can hold the minimum. The test uses the charge actually found and assumes nothing about the state.

**The tests check every state, not the winner.** With `optimal_aux`'s rule in place of `cheapest_aux`'s the closed form overcharges the states where the two differ and, on the instance tried, still crowned the same winner at the same energy. A test on the winner alone passes with the wrong rule. `test_the_closed_form_is_the_production_methods_minimum_state_by_state` compares against `CVaRGadget.cheapest_aux` and `encoded_cvar` at every schedule and every level.

- [ ] **Step 1: Write the failing test**

Create `tests/solvers/test_gadget_minimum.py`:

```python
import numpy as np
import pytest

from quantic.encoding.encode import encode, encode_solution
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.solvers.classical import gadget_minimum as module
from quantic.solvers.classical.gadget_minimum import GadgetMinimiser
from quantic.solvers.classical.lot_grid import lot_grid
from tests.solvers.instances import cvar, cvar_discrete, cvar_without_risk_aversion

REL = 1e-9


def _by_the_production_method(enc, schedules):
    """Each schedule's least charge: every zeta level, ``cheapest_aux`` at each."""
    gadget = enc.gadget
    out = []
    for schedule in schedules:
        losses = gadget.losses_from(enc.instance, enc.layout, schedule)
        charges = []
        for level in range(2**gadget.zeta.n_bits):
            zeta = gadget.zeta.lo + gadget.zeta.step * level
            aux = gadget.cheapest_aux(losses, zeta)
            charges.append(gadget.encoded_cvar(losses, zeta, aux))
        out.append(min(charges))
    return np.array(out)


def _losses(enc, schedules):
    return np.array(
        [enc.gadget.losses_from(enc.instance, enc.layout, s) for s in schedules]
    )


@pytest.mark.parametrize("n_zeta_bits", [3, 5, 7])
def test_the_closed_form_is_the_production_methods_minimum_state_by_state(n_zeta_bits):
    """The winner alone says little: the closed form ranks every state, and a
    closed form that overcharged the losers would still crown the winner."""
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=n_zeta_bits)
    schedules = list(lot_grid(inst).schedules())
    charge, level, _ = GadgetMinimiser(enc).minimise(_losses(enc, schedules))
    assert charge == pytest.approx(_by_the_production_method(enc, schedules), rel=REL)
    assert np.all((level >= 0) & (level <= 2**n_zeta_bits - 1))


def test_the_loss_matrix_prices_the_losses_the_gadget_prices():
    inst = cvar_discrete()
    for encoding in ENCODINGS:
        enc = encode(inst, encoding)
        mine = GadgetMinimiser(enc)
        for schedule in lot_grid(inst).schedules():
            bits = encode_solution(enc, schedule, zeta=enc.gadget.zeta.lo).astype(float)
            priced = mine.base + bits @ mine.loss_matrix.T
            assert priced == pytest.approx(
                enc.gadget.losses_from(inst, enc.layout, schedule), rel=REL, abs=1e-9
            )


def test_the_auxiliaries_are_the_cheapest_and_not_the_canonical(monkeypatch):
    """M3a gate result, second review, I2. With ``optimal_aux``'s rule the
    closed form charges some states more than the QUBO's own minimum does."""
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=5)
    schedules = list(lot_grid(inst).schedules())
    losses = _losses(enc, schedules)
    production = _by_the_production_method(enc, schedules)

    real = GadgetMinimiser.__init__

    def canonical(self, encoded):
        real(self, encoded)
        self.pull = np.zeros_like(self.pull)

    monkeypatch.setattr(GadgetMinimiser, "__init__", canonical)
    charge, _, _ = GadgetMinimiser(enc).minimise(losses)
    assert np.all(charge >= production - 1e-9)
    assert np.any(charge > production + 1e-6)


@pytest.mark.parametrize("shift", [-3.0, -1.0, 1.0, 3.0])
def test_losses_outside_the_grid_widen_the_scan_and_do_not_break_it(shift):
    """An oversold state's losses need not lie where ``zeta`` can reach.

    The grid is built to hold every loss a schedule inside the position can
    produce. The enumeration also visits states that are not such schedules.
    """
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=5)
    gadget = enc.gadget
    span = gadget.zeta.step * (2**gadget.zeta.n_bits - 1)
    schedules = list(lot_grid(inst).schedules())[:12]
    losses = _losses(enc, schedules) + shift * span

    expected = []
    for row in losses:
        charges = []
        for level in range(2**gadget.zeta.n_bits):
            zeta = gadget.zeta.lo + gadget.zeta.step * level
            charges.append(gadget.encoded_cvar(row, zeta, gadget.cheapest_aux(row, zeta)))
        expected.append(min(charges))

    charge, _, _ = GadgetMinimiser(enc).minimise(losses)
    assert charge == pytest.approx(np.array(expected), rel=REL)


def test_a_window_that_starts_at_one_level_reaches_the_same_minimum(monkeypatch):
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=8)
    losses = _losses(enc, list(lot_grid(inst).schedules()))
    wide, _, first = GadgetMinimiser(enc).minimise(losses)
    monkeypatch.setattr(module, "_FIRST_WINDOW", 1)
    narrow, _, grown = GadgetMinimiser(enc).minimise(losses)
    assert narrow == pytest.approx(wide, rel=1e-12)
    assert grown > 3
    assert first == 2 * 64 + 1


def test_with_no_risk_aversion_the_charge_is_nothing():
    inst = cvar_without_risk_aversion()
    enc = encode(inst, BinaryEncoding())
    losses = _losses(enc, list(lot_grid(inst).schedules()))
    charge, level, widest = GadgetMinimiser(enc).minimise(losses)
    assert np.all(charge == 0.0)
    assert np.all(level == 0.0)
    assert widest == 1


def test_the_bound_is_never_above_the_charge():
    """``lam`` times Rockafellar-Uryasev, less the slack, at every level."""
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=6)
    mine = GadgetMinimiser(enc)
    losses = _losses(enc, list(lot_grid(inst).schedules()))
    levels = np.tile(np.arange(2**6, dtype=float), (losses.shape[0], 1))
    assert np.all(
        mine.charge(losses, levels) >= mine.continuous(losses, levels) - mine.slack - 1e-9
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_gadget_minimum.py -q`
Expected: FAIL — `ImportError: cannot import name 'gadget_minimum' from 'quantic.solvers.classical'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/solvers/classical/gadget_minimum.py`:

```python
"""The CVaR gadget's charge for a state, minimised in closed form.

Once the lot bits are fixed the losses are fixed. Once ``zeta`` is fixed too
the scenarios separate, and each scenario's auxiliaries have a closed form --
``CVaRGadget.cheapest_aux``. What is left is a minimum over ``zeta``'s grid,
and at twenty bits that grid has a million levels.

This module does that minimum for many states at once. The closed form is
vectorised, and so is a second implementation of ``cheapest_aux``; the tests
compare the two at every schedule of an instance, and ``qubo_minimum``
rebuilds its winning state through the production method.
"""

from __future__ import annotations

import numpy as np

from quantic.encoding.encode import EncodedInstance

_SCAN_CHUNK = 256
_FIRST_WINDOW = 64


class GadgetMinimiser:
    """The gadget's charge for a state, minimised over ``zeta`` and the auxiliaries."""

    def __init__(self, enc: EncodedInstance) -> None:
        gadget = enc.gadget
        self.gadget = gadget
        self.lam = float(gadget.lam)
        self.step = float(gadget.zeta.step)
        self.lo = float(gadget.zeta.lo)
        self.top = 2**gadget.zeta.n_bits - 1
        self.penalty = float(gadget.penalty)
        self.weights = np.asarray(gadget.weights, dtype=float)
        self.slope = self.lam * self.weights / (1.0 - float(gadget.alpha))
        self.per_step = self.slope * self.step
        self.alpha = float(gadget.alpha)
        if self.penalty > 0.0:
            self.pull = self.per_step / (2.0 * self.penalty * self.step * self.step)
            # The charge is never below lam * Rockafellar-Uryasev less this:
            # relax each scenario's step count to a real and minimise.
            self.slack = float(
                np.sum(self.per_step**2) / (4.0 * self.penalty * self.step**2)
            )
        else:
            self.pull = np.zeros_like(self.per_step)
            self.slack = 0.0

        # L_s = base_s + sum over bits of loss_matrix[s, bit] * bit.
        registry = enc.registry
        coefficients = np.asarray(gadget.loss_coefficients, dtype=float)   # (S, N, T)
        tail = np.cumsum(coefficients[:, :, ::-1], axis=2)[:, :, ::-1]
        position = np.asarray(enc.instance.initial_lots, dtype=float)
        self.base = np.einsum("snt,n->s", coefficients, position)
        self.loss_matrix = np.zeros((coefficients.shape[0], len(registry)))
        for plan in enc.layout.cells:
            reach = tail[:, plan.asset, plan.bucket]
            self.base = self.base - plan.value_form.constant * reach
            for key, coef in plan.value_form.folded().terms:
                self.loss_matrix[:, registry.index(key)] -= coef * reach

    def continuous(self, losses: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """``lam`` times Rockafellar-Uryasev at each level. ``(B, W)``."""
        zeta = self.lo + self.step * levels
        excess = np.maximum(losses[:, None, :] - zeta[:, :, None], 0.0)
        return self.lam * zeta + (excess * self.slope[None, None, :]).sum(axis=2)

    def charge(self, losses: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """The gadget's exact energy at each level, auxiliaries at their best."""
        zeta = self.lo + self.step * levels
        if self.penalty == 0.0:
            return self.lam * zeta
        target = losses[:, None, :] - zeta[:, :, None]                 # (B, W, S)
        above = np.clip(np.floor(target / self.step - self.pull + 0.5), 0, self.top)
        below = np.clip(np.floor(target / self.step + 0.5), -self.top, 0)
        best = np.minimum(
            self.per_step * above + self.penalty * (above * self.step - target) ** 2,
            self.penalty * (below * self.step - target) ** 2,
        )
        return best.sum(axis=2) + self.lam * zeta

    def centre(self, losses: np.ndarray) -> np.ndarray:
        """The grid level nearest the weighted quantile, where the scan starts."""
        order = np.argsort(losses, axis=1)
        ranked = np.take_along_axis(losses, order, axis=1)
        cumulative = np.cumsum(self.weights[order], axis=1)
        at = np.minimum((cumulative < self.alpha).sum(axis=1), losses.shape[1] - 1)
        quantile = ranked[np.arange(losses.shape[0]), at]
        return np.clip(np.round((quantile - self.lo) / self.step), 0, self.top)

    def minimise(self, losses: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """The least charge for each state, the level that gives it, the widest scan.

        A window of levels is scanned exactly. It is wide enough once the
        bound at both of its edges is above the best charge inside it:
        Rockafellar-Uryasev is convex in the level, so beyond an edge where it
        is rising the bound only rises further. A state whose window is not
        wide enough is scanned again at twice the width.
        """
        n = losses.shape[0]
        energy = np.full(n, np.inf)
        level = np.zeros(n)
        if self.penalty == 0.0:
            # No excess is tracked, so zeta only costs lam * zeta -- and lam
            # is zero whenever the penalty is.
            return self.charge(losses, np.zeros((n, 1)))[:, 0], level, 1

        centre = self.centre(losses)
        pending = np.arange(n)
        window = _FIRST_WINDOW
        widest = 0
        while pending.size:
            widest = max(widest, 2 * window + 1)
            offsets = np.arange(-window, window + 1, dtype=float)
            for start in range(0, pending.size, _SCAN_CHUNK):
                rows = pending[start:start + _SCAN_CHUNK]
                levels = np.clip(centre[rows, None] + offsets[None, :], 0, self.top)
                charges = self.charge(losses[rows], levels)
                at = charges.argmin(axis=1)
                energy[rows] = charges[np.arange(rows.size), at]
                level[rows] = levels[np.arange(rows.size), at]
            edges = np.stack(
                [
                    np.clip(centre[pending] - window, 0, self.top),
                    centre[pending],
                    np.clip(centre[pending] + window, 0, self.top),
                ],
                axis=1,
            )
            bound = self.continuous(losses[pending], edges)
            need = energy[pending] + self.slack
            left_ok = (edges[:, 0] == 0) | (
                (bound[:, 0] > need) & (bound[:, 0] > bound[:, 1])
            )
            right_ok = (edges[:, 2] == self.top) | (
                (bound[:, 2] > need) & (bound[:, 2] > bound[:, 1])
            )
            pending = pending[~(left_ok & right_ok)]
            window *= 2
        return energy, level, widest
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_gadget_minimum.py -q`
Expected: every test in the file passes (12).

- [ ] **Step 5: Lint and commit**

Run: `.venv/Scripts/ruff.exe check src tests` and `.venv/Scripts/lint-imports.exe`
Expected: clean, `Contracts: 3 kept, 0 broken.`

```bash
git add src/quantic/solvers/classical/gadget_minimum.py tests/solvers/test_gadget_minimum.py
git commit -m "feat(solvers): the gadget's charge, minimised over zeta in closed form"
```

---

## Task 7: The QUBO's minimum

**Files:**
- Create: `src/quantic/solvers/classical/qubo_minimum.py`
- Create: `tests/solvers/test_qubo_minimum.py`

**Interfaces:**
- Consumes: `GadgetMinimiser` (Task 6); `EncodedInstance`, `encode_solution`, `Qubo`; in tests `brute_force` and `score` (Task 3), `decode`, and `tests.encoding.landscape.landscape`.
- Produces:
  - `EVERY_STATE = "every_state"`, `DECODABLE = "decodable"`, `EXHAUSTIVE_BIT_LIMIT = 28`
  - `QuboMinimum(bits, energy, scope, n_states, n_ties, n_scanned, widest_window)`
  - `qubo_minimum(enc: EncodedInstance) -> QuboMinimum`

**How `2^28` states take three seconds.** The bits outside the gadget are split in two halves. Each half's own energy is computed once for each of its patterns; per state only the cross term is paid, and that is a matrix product over a block of the second half's patterns. With a gadget the same product gives a lower bound on every state's energy — costs, hard penalties, and `λ` times the mean loss less the slack — and only states whose bound is at or below the witness's energy go on to the scan. On `T0-D3` that is 38,440 states of `2^24`: the feasible schedules and nothing else.

**Scope.** With 28 bits or fewer outside the gadget, every state. Above that, every state that decodes: each cell on the canonical encoding of each value the layout admits, feasible or not, and every setting of the slack bits. The states left out are the ones `decode` rejects. Spec §8 sets out what that leaves unproven.

- [ ] **Step 1: Write the failing test**

Create `tests/solvers/test_qubo_minimum.py`:

```python
import dataclasses
import itertools

import numpy as np
import pytest

import quantic.encoding.encode as encode_module
from quantic.encoding.decode import FEASIBLE, REJECTED, decode
from quantic.encoding.encode import encode
from quantic.encoding.integer import ENCODINGS, BinaryEncoding, OneHotEncoding
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.classical import gadget_minimum as gadget_module
from quantic.solvers.classical import qubo_minimum as module
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.qubo_minimum import (
    DECODABLE,
    EVERY_STATE,
    EXHAUSTIVE_BIT_LIMIT,
    qubo_minimum,
)
from quantic.solvers.classical.scoring import score
from tests.encoding.landscape import landscape
from tests.solvers.instances import (
    blocks,
    concave,
    cvar,
    cvar_without_risk_aversion,
    discrete,
    instance,
    no_risk_aversion,
    nothing_to_sell_in_one_name,
    plain,
    scenarios,
)

REL = 1e-9


def _tiny(**kw):
    """Two names, two buckets, two lots a bucket: few enough bits to list."""
    base = dict(
        n_buckets=2, initial_lots=(2, 3), max_lots_per_bucket=2,
        witness=Schedule(lots=((1, 1), (1, 2))),
    )
    base.update(kw)
    return instance(**base)


def _tiny_cvar():
    return _tiny(
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=0.02, alpha=0.7, scenarios=scenarios(n_buckets=2)),
    )


def _tiny_cvar_discrete():
    return _tiny(
        initial_lots=(2, 2),
        dials=Dials(cvar_risk=True, discrete_participation=True),
        constraints=(
            FullLiquidation(), Cardinality(k=1), MinParticipation(min_lots=(2, 1)),
        ),
        risk=CVaRRisk(lam=0.02, alpha=0.7, scenarios=scenarios(n_buckets=2)),
        witness=Schedule(lots=((2, 0), (0, 2))),
    )


def _every_energy(enc):
    """Every state's energy, one ``Qubo.energy`` call at a time."""
    qubo = enc.qubo
    return {
        bits: qubo.energy(np.array(bits))
        for bits in itertools.product((0, 1), repeat=enc.n_vars)
    }


# -- without a gadget -------------------------------------------------------


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize("dials", [Dials(), Dials(concave_impact=True)], ids=["ac", "d1"])
def test_it_finds_the_lowest_energy_there_is(encoding, dials):
    enc = encode(_tiny(dials=dials), encoding)
    assert enc.n_vars <= 16
    energies = _every_energy(enc)
    found = qubo_minimum(enc)
    assert found.scope == EVERY_STATE
    assert found.n_states == 2**enc.n_vars
    assert found.energy == pytest.approx(min(energies.values()), rel=REL)
    assert enc.qubo.energy(found.bits) == pytest.approx(found.energy, rel=REL)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_it_counts_the_states_that_reach_the_minimum(encoding):
    enc = encode(_tiny(), encoding)
    energies = np.array(list(_every_energy(enc).values()))
    lowest = energies.min()
    expected = int(np.sum(energies <= lowest + 1e-9 * max(1.0, abs(lowest))))
    assert qubo_minimum(enc).n_ties == expected


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize(
    "build",
    [plain, concave, discrete, blocks, no_risk_aversion, nothing_to_sell_in_one_name],
    ids=lambda b: b.__name__,
)
def test_the_minimum_decodes_to_the_schedule_enumeration_finds(build, encoding):
    """The gate's claim, at a size that runs with the unit tests."""
    inst = build()
    enc = encode(inst, encoding)
    found = qubo_minimum(enc)
    sample = decode(enc, found.bits)
    assert sample.classification == FEASIBLE
    assert score(inst, sample.schedule)[0].total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )
    assert enc.qubo.energy(found.bits) == pytest.approx(found.energy, rel=REL)


def test_above_the_limit_only_the_states_that_decode_are_examined(monkeypatch):
    enc = encode(_tiny(), OneHotEncoding())
    monkeypatch.setattr(module, "EXHAUSTIVE_BIT_LIMIT", enc.n_vars - 1)
    found = qubo_minimum(enc)

    decodable = {
        bits: energy for bits, energy in _every_energy(enc).items()
        if decode(enc, np.array(bits)).classification != REJECTED
    }
    assert found.scope == DECODABLE
    assert found.n_states == len(decodable)
    assert found.n_states < 2**enc.n_vars
    assert found.energy == pytest.approx(min(decodable.values()), rel=REL)


def test_the_states_that_decode_include_the_infeasible_ones(monkeypatch):
    """A schedule that breaks a constraint is a state; a broken one-hot is not."""
    enc = encode(_tiny(), OneHotEncoding())
    monkeypatch.setattr(module, "EXHAUSTIVE_BIT_LIMIT", 0)
    found = qubo_minimum(enc)
    per_cell = [len(plan.values) for plan in enc.layout.cells]
    assert found.n_states == int(np.prod(per_cell))
    assert found.n_states > brute_force(_tiny()).detail["n_feasible"]


def test_the_slack_bits_are_enumerated_with_the_states_that_decode(monkeypatch):
    inst = discrete()
    enc = encode(inst, OneHotEncoding())
    monkeypatch.setattr(module, "EXHAUSTIVE_BIT_LIMIT", 0)
    found = qubo_minimum(enc)
    n_slack = sum(1 for key in enc.registry.keys if key[0] == "card")
    per_cell = [len(plan.values) for plan in enc.layout.cells]
    assert n_slack > 0
    assert found.n_states == int(np.prod(per_cell)) * 2**n_slack
    assert decode(enc, found.bits).classification == FEASIBLE
    assert score(inst, decode(enc, found.bits).schedule)[0].total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )


def test_the_limit_is_the_one_the_design_names():
    assert EXHAUSTIVE_BIT_LIMIT == 28


def test_penalties_too_weak_to_bind_show_up_as_an_infeasible_minimum(monkeypatch):
    """The mutation the gate is asked to catch. The minimiser reports the
    QUBO's minimum, not the best feasible state, so a QUBO whose penalties do
    not bind is seen to have an infeasible one."""
    real = encode_module.derive_penalties

    def weak(*args, **kwargs):
        strengths = real(*args, **kwargs)
        return dataclasses.replace(
            strengths,
            validity=strengths.validity * 1e-4,
            full_liquidation=strengths.full_liquidation * 1e-4,
        )

    monkeypatch.setattr(encode_module, "derive_penalties", weak)
    enc = encode(plain(), BinaryEncoding())
    assert decode(enc, qubo_minimum(enc).bits).classification != FEASIBLE


# -- with a gadget ----------------------------------------------------------


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize(
    "build", [_tiny_cvar, _tiny_cvar_discrete], ids=lambda b: b.__name__
)
def test_with_a_gadget_it_finds_what_the_whole_landscape_holds(build, encoding):
    """Against tests/encoding/landscape.py, which scans every zeta level of
    every state and prunes nothing."""
    enc = encode(build(), encoding)
    scape = landscape(enc)
    found = qubo_minimum(enc)
    assert found.energy == pytest.approx(float(scape.energy.min()), rel=REL)
    assert enc.qubo.energy(found.bits) == pytest.approx(found.energy, rel=REL)
    assert found.n_states == len(scape.energy)
    assert 0 < found.n_scanned <= found.n_states


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_window_that_starts_too_narrow_grows_until_it_is_wide_enough(
    encoding, monkeypatch
):
    enc = encode(cvar(), encoding)
    wide = qubo_minimum(enc)
    monkeypatch.setattr(gadget_module, "_FIRST_WINDOW", 1)
    narrow = qubo_minimum(enc)
    assert narrow.energy == pytest.approx(wide.energy, rel=1e-12)
    assert narrow.widest_window > 3


@pytest.mark.parametrize("n_zeta_bits", [4, 8, 12, 20])
def test_the_winning_state_has_the_energy_claimed_at_any_width(n_zeta_bits):
    """The closed form against the matrix, at the width the gate uses too."""
    enc = encode(cvar(), BinaryEncoding(), n_zeta_bits=n_zeta_bits)
    found = qubo_minimum(enc)
    assert enc.qubo.energy(found.bits) == pytest.approx(found.energy, rel=1e-6)
    assert decode(enc, found.bits).classification == FEASIBLE


def test_at_a_wide_grid_the_minimum_is_the_schedule_enumeration_finds():
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=20)
    sample = decode(enc, qubo_minimum(enc).bits)
    assert score(inst, sample.schedule)[0].total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )


def test_the_regret_at_a_coarse_grid_is_inside_what_the_gadget_declares():
    inst = cvar()
    enc = encode(inst, BinaryEncoding(), n_zeta_bits=5)
    sample = decode(enc, qubo_minimum(enc).bits)
    regret = score(inst, sample.schedule)[0].total - brute_force(inst).objective.total
    assert 0.0 <= regret <= 2.0 * enc.gadget.resolution + enc.gadget.cost_gap


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_gadget_with_no_risk_aversion_charges_nothing(encoding):
    inst = cvar_without_risk_aversion()
    enc = encode(inst, encoding)
    assert enc.gadget.penalty == 0.0
    found = qubo_minimum(enc)
    sample = decode(enc, found.bits)
    assert sample.classification == FEASIBLE
    assert enc.qubo.energy(found.bits) == pytest.approx(found.energy, rel=REL)
    assert score(inst, sample.schedule)[0].total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_qubo_minimum.py -q`
Expected: FAIL — `ImportError: cannot import name 'qubo_minimum' from 'quantic.solvers.classical'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/solvers/classical/qubo_minimum.py`:

```python
"""The exact minimum of an encoded instance's energy.

The gate of spec section 14 asks where the QUBO's minimum is, and a sampler
cannot answer that: a sampler that returns a low energy has shown that the
energy is reachable, not that nothing lower exists. This module enumerates.

Two things make that affordable. The bits outside the CVaR gadget are split in
two, so that each half's own energy is computed once and only the cross term
is paid per state. And the gadget is never enumerated: once the lot bits and
``zeta`` are fixed the scenarios separate, the auxiliaries have a closed form,
and what is left is a scan over ``zeta``'s grid that a bound cuts short.

The closed form here is vectorised and so is a second implementation of
``CVaRGadget.cheapest_aux``. The winning state is rebuilt through the
production method, and callers check its energy against ``Qubo.energy``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from quantic.encoding.encode import EncodedInstance, encode_solution
from quantic.encoding.qubo import Qubo
from quantic.solvers.classical.gadget_minimum import GadgetMinimiser

EVERY_STATE = "every_state"
DECODABLE = "decodable"

# 2**28 states is a matter of minutes; each further bit doubles it.
EXHAUSTIVE_BIT_LIMIT = 28

# Energies closer than this, relative to the minimum, are the same energy.
_TIE_RTOL = 1e-9
_BLOCK = 64
_MAX_CANDIDATES = 20_000_000


@dataclass(frozen=True)
class QuboMinimum:
    """The lowest energy the QUBO reaches, and a state that reaches it."""

    bits: np.ndarray          # the full vector, auxiliaries included
    energy: float
    scope: str                # EVERY_STATE | DECODABLE
    n_states: int             # how many states of the bits outside the gadget
    n_ties: int               # of those, how many reach the minimum
    n_scanned: int            # how many went through the zeta scan
    widest_window: int        # the most zeta levels any one state needed


@dataclass(frozen=True)
class _Half:
    """Some of the bits outside the gadget, and the patterns they range over."""

    index: np.ndarray         # (k,) registry indices
    patterns: np.ndarray      # (m, k) 0/1


def _all_patterns(k: int) -> np.ndarray:
    return ((np.arange(2**k)[:, None] >> np.arange(k)[None, :]) & 1).astype(float)


def _halves_every_state(outside: np.ndarray) -> tuple[_Half, _Half]:
    cut = len(outside) // 2
    first, second = outside[:cut], outside[cut:]
    return (
        _Half(index=first, patterns=_all_patterns(len(first))),
        _Half(index=second, patterns=_all_patterns(len(second))),
    )


def _groups_decodable(enc: EncodedInstance, outside: np.ndarray) -> list[_Half]:
    """One group per cell, on the patterns its values encode to, then the slack."""
    registry = enc.registry
    groups: list[_Half] = []
    named: set[int] = set()
    for plan in enc.layout.cells:
        index = np.array([registry.index(k) for k in plan.bit_keys], dtype=np.int64)
        rows = []
        for value in plan.values:
            bits = plan.encode(value)
            rows.append([float(bits[k]) for k in plan.bit_keys])
        groups.append(_Half(index=index, patterns=np.array(rows, dtype=float)))
        named.update(int(i) for i in index)
    for i in outside:
        if int(i) not in named:
            groups.append(
                _Half(index=np.array([int(i)]), patterns=np.array([[0.0], [1.0]]))
            )
    return groups


def _merge(groups: list[_Half]) -> _Half:
    if not groups:
        return _Half(index=np.zeros(0, dtype=np.int64), patterns=np.zeros((1, 0)))
    index = np.concatenate([g.index for g in groups])
    rows = [
        np.concatenate(combo)
        for combo in itertools.product(*[list(g.patterns) for g in groups])
    ]
    return _Half(index=index, patterns=np.array(rows, dtype=float))


def _halves_decodable(enc: EncodedInstance, outside: np.ndarray) -> tuple[_Half, _Half]:
    groups = _groups_decodable(enc, outside)
    total = float(np.sum([np.log2(len(g.patterns)) for g in groups]))
    first: list[_Half] = []
    second: list[_Half] = []
    running = 0.0
    for g in groups:
        if running < total / 2.0:
            first.append(g)
            running += float(np.log2(len(g.patterns)))
        else:
            second.append(g)
    return _merge(first), _merge(second)


def _symmetric(qubo: Qubo) -> np.ndarray:
    q = np.asarray(qubo.Q, dtype=float)
    return q + q.T - np.diag(np.diag(q))


def _own_energy(sym: np.ndarray, half: _Half) -> np.ndarray:
    """``z' Q z`` over the half's own bits, for each of its patterns."""
    block = sym[np.ix_(half.index, half.index)]
    upper = np.triu(block)
    return ((half.patterns @ upper) * half.patterns).sum(axis=1)


def qubo_minimum(enc: EncodedInstance) -> QuboMinimum:
    """Minimise ``enc.qubo``'s energy, and say over which states."""
    registry = enc.registry
    inside = (
        set() if enc.gadget is None
        else {registry.index(k) for k in enc.gadget.keys}
    )
    outside = np.array(
        [i for i in range(len(registry)) if i not in inside], dtype=np.int64
    )

    if len(outside) <= EXHAUSTIVE_BIT_LIMIT:
        scope = EVERY_STATE
        first, second = _halves_every_state(outside)
    else:
        scope = DECODABLE
        first, second = _halves_decodable(enc, outside)
    n_states = len(first.patterns) * len(second.patterns)

    flat = enc.breakdown.spread + enc.breakdown.temporary + enc.breakdown.permanent
    flat = flat + enc.penalties
    if enc.gadget is None:
        flat = flat + enc.breakdown.risk
    sym = _symmetric(flat)

    own_first = _own_energy(sym, first)
    own_second = _own_energy(sym, second) + flat.constant
    cross = sym[np.ix_(first.index, second.index)]

    gadget = None if enc.gadget is None else GadgetMinimiser(enc)
    if gadget is not None:
        # The charge is at least lam * CVaR less the slack, and CVaR is at
        # least the mean loss, which is linear in the bits.
        mean = gadget.lam * (gadget.weights @ gadget.loss_matrix)
        bound_first = own_first + first.patterns @ mean[first.index]
        bound_second = (
            own_second + second.patterns @ mean[second.index]
            + gadget.lam * float(gadget.weights @ gadget.base) - gadget.slack
        )
    else:
        bound_first, bound_second = own_first, own_second

    # Somewhere to start: the witness is a state, so its energy is reachable.
    witness = enc.instance.witness
    start_bits = _rebuild(enc, gadget, _outside_of(enc, witness, outside), outside)
    incumbent = float(enc.qubo.energy(start_bits))
    threshold = incumbent + _TIE_RTOL * max(1.0, abs(incumbent))

    # Every state whose bound is at or below the incumbent.
    picked_first: list[np.ndarray] = []
    picked_second: list[np.ndarray] = []
    picked_bound: list[np.ndarray] = []
    n_picked = 0
    lead = first.patterns @ cross                                     # (m1, k2)
    for start in range(0, len(second.patterns), _BLOCK):
        block = second.patterns[start:start + _BLOCK]                 # (b, k2)
        values = (
            bound_first[:, None] + bound_second[None, start:start + _BLOCK]
            + lead @ block.T
        )
        a, b = np.nonzero(values <= threshold)
        if a.size:
            picked_first.append(a)
            picked_second.append(b + start)
            picked_bound.append(values[a, b])
            n_picked += a.size
            if n_picked > _MAX_CANDIDATES:
                raise MemoryError(
                    f"more than {_MAX_CANDIDATES} states sit at or below the "
                    f"witness's energy on {enc.instance.instance_id}; the witness "
                    "is too poor a start for this enumeration"
                )

    a = np.concatenate(picked_first)
    b = np.concatenate(picked_second)
    bound = np.concatenate(picked_bound)

    if gadget is None:
        energy = bound
        level = np.zeros(a.size)
        n_scanned, widest = 0, 0
    else:
        states = np.zeros((a.size, len(registry)))
        states[:, first.index] = first.patterns[a]
        states[:, second.index] = second.patterns[b]
        losses = gadget.base[None, :] + states @ gadget.loss_matrix.T
        flat_energy = bound - (
            states @ mean + gadget.lam * float(gadget.weights @ gadget.base)
            - gadget.slack
        )
        charge, level, widest = gadget.minimise(losses)
        energy = flat_energy + charge
        n_scanned = int(a.size)

    best = int(np.argmin(energy))
    lowest = float(energy[best])
    ties = int(np.sum(energy <= lowest + _TIE_RTOL * max(1.0, abs(lowest))))

    chosen = np.zeros(len(outside))
    position = {int(i): k for k, i in enumerate(outside)}
    for half, row in ((first, a[best]), (second, b[best])):
        for k, i in enumerate(half.index):
            chosen[position[int(i)]] = half.patterns[row, k]
    bits = _rebuild(enc, gadget, chosen, outside, level=float(level[best]))
    return QuboMinimum(
        bits=bits, energy=lowest, scope=scope, n_states=n_states, n_ties=ties,
        n_scanned=n_scanned, widest_window=int(widest),
    )


def _outside_and_inside(enc: EncodedInstance, schedule) -> np.ndarray:
    """``schedule``'s canonical state. Only its bits outside the gadget are used."""
    zeta = None if enc.gadget is None else enc.gadget.zeta.lo
    return encode_solution(enc, schedule, zeta=zeta).astype(float)


def _outside_of(enc: EncodedInstance, schedule, outside: np.ndarray) -> np.ndarray:
    """The bits outside the gadget for ``schedule``'s canonical state."""
    return _outside_and_inside(enc, schedule)[outside]


def _rebuild(
    enc: EncodedInstance,
    gadget: GadgetMinimiser | None,
    chosen: np.ndarray,
    outside: np.ndarray,
    *,
    level: float | None = None,
) -> np.ndarray:
    """The full bit vector: ``chosen`` outside the gadget, its best state inside.

    The auxiliaries come from ``CVaRGadget.cheapest_aux``, the production
    method, and not from the vectorised closed form above.
    """
    registry = enc.registry
    bits = np.zeros(len(registry), dtype=np.int64)
    bits[outside] = np.asarray(chosen).astype(np.int64)
    if gadget is None:
        return bits
    losses = gadget.base + bits.astype(float) @ gadget.loss_matrix.T
    if level is None:
        _, levels, _ = gadget.minimise(losses[None, :])
        level = float(levels[0])
    zeta = gadget.lo + gadget.step * level
    for key, value in gadget.gadget.zeta.encode(zeta).items():
        bits[registry.index(key)] = value
    for key, value in gadget.gadget.cheapest_aux(losses, zeta).items():
        bits[registry.index(key)] = value
    return bits
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_qubo_minimum.py -q`
Expected: every test in the file passes (50), in about ten seconds.

- [ ] **Step 5: Lint and commit**

Run: `.venv/Scripts/ruff.exe check src tests` and `.venv/Scripts/lint-imports.exe`
Expected: clean, `Contracts: 3 kept, 0 broken.`

```bash
git add src/quantic/solvers/classical/qubo_minimum.py tests/solvers/test_qubo_minimum.py
git commit -m "feat(solvers): the exact minimum of an encoded instance"
```

---

## Task 8: The M3b gate

**Files:**
- Create: `tests/solvers/gate.py`
- Create: `tests/solvers/test_m3b_gate.py`
- Create: `tests/solvers/test_t1_bracket.py`
- Create: `tests/solvers/report.py`

**Interfaces:**
- Consumes: everything. `tests.ladder.tier` (Task 2).
- Produces:
  - `tests.solvers.gate`: `GATE_ZETA_BITS = 20`, `STUDY_ZETA_BITS = 8`, `REL`, `BRACKET_RTOL`, `DENSE_RTOL`, `EVERY_STATE_CASES`, `classical(instance) -> (findings, record)`, `encoded(instance, encoding, optimum, *, n_zeta_bits) -> (findings, record)`
  - a verdict.

**What the gate asserts.** Spec §9, checks 1 to 9. For each of the 16 T0 instances: both exact methods solve it feasibly, agree on the optimum, the bracket closes, permanent impact is the constant the program carries. For each instance under each encoding: the QUBO's minimum is a feasible schedule, its true objective is the optimum, the energy claimed is the energy of the state returned, and the scope of the minimum is the one the design records.

**What it records without gating.** At the study's 8 zeta bits the minimum is not the optimum on five cells. The gate asserts *which* five, and that the regret is inside the bound the gadget declares. That a test asserts the claim is false at eight bits is deliberate: it is the finding, written where it cannot be missed, and it is the gate seen failing.

**The checks are functions so that a defect can be put in.** `tests/solvers/gate.py` holds them; the gate calls them over the ladder, and four further tests call the same functions over one instance with a defect monkeypatched in, asserting the check that reports it. A check nobody has seen fail is a check nobody knows works — M3a's second review found the gate passing against a gadget broken eight ways.

**One pass, many named assertions,** as in M3a: a module-scoped fixture runs every case once and the named tests assert on what it found. The pass takes a little over a minute.

- [ ] **Step 1: Write the checks**

Create `tests/solvers/gate.py`:

```python
"""The M3b gate's checks, as functions the gate and its mutation tests both call.

A check that has never been seen failing is a check nobody knows works. The
gate runs these over the whole ladder; the mutation tests run the same
functions over one instance with a defect put in, and assert on which check
reports it.

Every function returns ``(findings, record)``. ``findings`` maps a check's
name to what it found wrong and is empty when nothing is. ``record`` is what
was measured, for the result document.
"""

from __future__ import annotations

import pytest

from quantic.encoding.decode import FEASIBLE, decode
from quantic.encoding.encode import encode
from quantic.problem.risk import CVaRRisk
from quantic.solvers.base import OPTIMAL
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.exact_pwl_milp import exact_pwl_milp, permanent_constant
from quantic.solvers.classical.qubo_minimum import qubo_minimum
from quantic.solvers.classical.scoring import score

# The width at which the parent gate is stated on the cells that carry CVaR,
# and the width the study runs at. Design sections 3.2 and 4.
GATE_ZETA_BITS = 20
STUDY_ZETA_BITS = 8

# Two evaluations of one schedule, or of two schedules of equal cost.
REL = 1e-9
# HiGHS works to 1e-7 in feasibility; this is that, with room.
BRACKET_RTOL = 1e-6
# A dense z'Qz at a dynamic range of 2.4e12 carries about 7e-8.
DENSE_RTOL = 1e-6

# Design section 8: how many of the sixteen combinations are covered over
# every state, under each encoding.
EVERY_STATE_CASES = {"binary": 16, "domain_wall": 10, "one_hot": 5}


def classical(instance):
    """Checks 1 to 4: the two exact methods, against each other."""
    findings: dict[str, object] = {}
    try:
        by_hand = brute_force(instance)
        by_milp = exact_pwl_milp(instance)
    except Exception as exc:  # the gate reports; it does not filter
        return {"solves": repr(exc)}, None

    for name, result in (("brute_force", by_hand), ("exact_pwl_milp", by_milp)):
        if result.status != OPTIMAL:
            findings["solves"] = (name, result.status)
        elif not result.feasibility.feasible:
            findings["solves"] = (name, dict(result.feasibility.violations))
    if findings:
        return findings, None

    if by_milp.objective.total != pytest.approx(by_hand.objective.total, rel=REL):
        findings["methods_agree"] = (by_milp.objective.total, by_hand.objective.total)

    bracket = by_milp.bracket
    if bracket.width > BRACKET_RTOL * max(1.0, abs(bracket.upper)):
        findings["bracket_closes"] = (bracket.lower, bracket.upper)

    constant = permanent_constant(instance)
    if by_milp.objective.permanent_impact_cost != pytest.approx(constant, rel=REL):
        findings["permanent_is_constant"] = (
            by_milp.objective.permanent_impact_cost, constant
        )

    record = {
        "instance": instance.instance_id,
        "optimum": by_hand.objective.total,
        "n_feasible": by_hand.detail["n_feasible"],
        "brute_force_s": by_hand.timings["total_s"],
        "milp_s": by_milp.timings["total_s"],
        "milp_rounds": by_milp.detail["rounds"],
        "milp_binaries": by_milp.detail["n_binaries"],
        "presolve_retries": by_milp.detail["presolve_retries"],
        "bracket_width": bracket.width,
    }
    return findings, record


def encoded(instance, encoding, optimum: float, *, n_zeta_bits: int):
    """Checks 5 to 8: the QUBO's minimum, against the exact optimum."""
    width = n_zeta_bits if isinstance(instance.risk, CVaRRisk) else STUDY_ZETA_BITS
    try:
        enc = encode(instance, encoding, n_zeta_bits=width)
        found = qubo_minimum(enc)
        sample = decode(enc, found.bits)
    except Exception as exc:  # the gate reports; it does not filter
        return {"minimises": repr(exc)}, None

    findings: dict[str, object] = {}
    regret = None
    if sample.classification != FEASIBLE:
        findings["minimum_is_feasible"] = (
            sample.classification,
            None if sample.report is None else dict(sample.report.violations),
        )
    else:
        total = score(instance, sample.schedule)[0].total
        regret = total - optimum
        if total != pytest.approx(optimum, rel=REL):
            findings["minimum_is_the_optimum"] = (total, optimum)

    dense = enc.qubo.energy(found.bits)
    if found.energy != pytest.approx(dense, rel=DENSE_RTOL):
        findings["energy_is_the_matrix"] = (found.energy, dense)

    record = {
        "instance": instance.instance_id,
        "encoding": encoding.name,
        "zeta_bits": width if enc.gadget is not None else None,
        "n_vars": enc.n_vars,
        "scope": found.scope,
        "n_states": found.n_states,
        "n_scanned": found.n_scanned,
        "widest_window": found.widest_window,
        "n_ties": found.n_ties,
        "regret": regret,
        "resolution": None if enc.gadget is None else enc.gadget.resolution,
        "cost_gap": None if enc.gadget is None else enc.gadget.cost_gap,
        "dense_error": abs(found.energy - dense) / max(1.0, abs(dense)),
    }
    return findings, record
```

- [ ] **Step 2: Write the gate**

Create `tests/solvers/test_m3b_gate.py`:

```python
"""THE M3b GATE -- the QUBO's minimum is the exact optimum, on T0, sixteen ways.

Parent spec section 14. Two exact methods that share only the lot grid and
the evaluator agree on the optimum of every T0 instance; the minimum of every
QUBO the encoder builds for it decodes to a feasible schedule that costs
exactly that.

The cells that carry CVaR are encoded at twenty zeta bits for this, not the
study's eight. At eight the claim is false on five cells, and the last tests
here measure by how much.
"""

import collections
import dataclasses

import pytest

import quantic.encoding.encode as encode_module
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.problem.risk import CVaRRisk
from quantic.solvers.classical import exact_pwl_milp as milp_module
from quantic.solvers.classical.qubo_minimum import EVERY_STATE
from tests.ladder import tier
from tests.solvers.gate import (
    EVERY_STATE_CASES,
    GATE_ZETA_BITS,
    STUDY_ZETA_BITS,
    classical,
    encoded,
)

# Where the study's width does not find the optimum. Design section 3.2.
NOT_THE_OPTIMUM_AT_EIGHT_BITS = {
    "T0-D3-2026",
    "T0-D2+D3-2026",
    "T0-D2+D3+D4-2026",
    "T0-D1+D3-2026",
    "T0-D1+D2+D3+D4-2026",
}


@pytest.fixture(scope="module")
def ladder():
    return tier("T0")


@pytest.fixture(scope="module")
def gate(ladder):
    """One pass over every case, collecting each check's findings."""
    findings = collections.defaultdict(dict)
    records = {"classical": [], "encoded": [], "study": []}

    for instance in ladder:
        found, record = classical(instance)
        for check, what in found.items():
            findings[check][instance.instance_id] = what
        if record is None:
            continue
        records["classical"].append(record)

        for encoding in ENCODINGS:
            tag = (instance.instance_id, encoding.name)
            found, row = encoded(
                instance, encoding, record["optimum"], n_zeta_bits=GATE_ZETA_BITS
            )
            for check, what in found.items():
                findings[check][tag] = what
            if row is not None:
                records["encoded"].append(row)

            if not isinstance(instance.risk, CVaRRisk):
                continue
            found, row = encoded(
                instance, encoding, record["optimum"], n_zeta_bits=STUDY_ZETA_BITS
            )
            for check in ("minimises", "minimum_is_feasible", "energy_is_the_matrix"):
                if check in found:
                    findings[f"study_{check}"][tag] = found[check]
            if row is not None:
                records["study"].append(row)

    return {"findings": findings, "records": records}


def _none(gate, check):
    found = gate["findings"].get(check, {})
    assert not found, found


def test_the_ladder_is_still_sixteen_combinations(ladder):
    assert len(ladder) == 16
    assert sum(isinstance(i.risk, CVaRRisk) for i in ladder) == 8


def test_both_exact_methods_solve_every_instance_feasibly(gate):
    """THE GATE, check 1."""
    _none(gate, "solves")
    assert len(gate["records"]["classical"]) == 16


def test_the_two_exact_methods_agree_on_the_optimum(gate):
    """THE GATE, check 2. Enumeration and the program share no model."""
    _none(gate, "methods_agree")


def test_the_brackets_close(gate):
    """THE GATE, check 3. The lower bound is proven, the upper attained."""
    _none(gate, "bracket_closes")


def test_permanent_impact_is_the_constant_the_program_carries(gate):
    """THE GATE, check 4."""
    _none(gate, "permanent_is_constant")


def test_every_qubo_is_minimised(gate):
    _none(gate, "minimises")
    assert len(gate["records"]["encoded"]) == 16 * 3


def test_the_minimum_of_every_qubo_is_a_feasible_schedule(gate):
    """THE GATE, check 5. The penalties bind at the minimum, not only nearby."""
    _none(gate, "minimum_is_feasible")


def test_the_minimum_of_every_qubo_is_the_exact_optimum(gate):
    """THE GATE, check 6. Parent spec section 14, and the point of M3."""
    _none(gate, "minimum_is_the_optimum")


def test_the_energy_claimed_is_the_energy_of_the_state_returned(gate):
    """THE GATE, check 7. The closed form against the matrix."""
    _none(gate, "energy_is_the_matrix")


def test_the_scope_of_every_minimum_is_the_one_the_design_records(gate):
    """THE GATE, check 8. Design section 8."""
    every_state = collections.Counter(
        row["encoding"] for row in gate["records"]["encoded"]
        if row["scope"] == EVERY_STATE
    )
    assert dict(every_state) == EVERY_STATE_CASES
    covered = {
        row["instance"] for row in gate["records"]["encoded"]
        if row["scope"] == EVERY_STATE
    }
    assert len(covered) == 16, "a combination is covered over every state nowhere"


def test_the_gate_was_stated_at_the_width_the_design_names(gate):
    widths = {row["zeta_bits"] for row in gate["records"]["encoded"]}
    assert widths == {None, GATE_ZETA_BITS}


# -- recorded: the study's width --------------------------------------------


def test_at_the_studys_width_every_minimum_is_still_feasible(gate):
    for check in ("minimises", "minimum_is_feasible", "energy_is_the_matrix"):
        _none(gate, f"study_{check}")
    assert len(gate["records"]["study"]) == 8 * 3


def test_at_the_studys_width_the_regret_is_inside_what_the_gadget_declares(gate):
    """Check 9. The bound is loose and is not the finding; the regret is."""
    outside = {
        (row["instance"], row["encoding"]): (row["regret"], row["resolution"])
        for row in gate["records"]["study"]
        if not -1e-9 <= row["regret"] <= 2.0 * row["resolution"] + row["cost_gap"]
    }
    assert not outside, outside


def test_at_the_studys_width_the_minimum_is_not_the_optimum_on_five_cells(gate):
    """What the study's QUBOs do, written down where it cannot be missed.

    This is also the gate seen failing: at eight bits check 6 reports these
    five cells under every encoding, and no others.
    """
    off = {
        row["instance"] for row in gate["records"]["study"] if row["regret"] > 1e-9
    }
    assert off == NOT_THE_OPTIMUM_AT_EIGHT_BITS
    for name in off:
        regrets = {
            round(row["regret"], 9) for row in gate["records"]["study"]
            if row["instance"] == name
        }
        assert len(regrets) == 1, (name, regrets)


# -- the gate, seen failing -------------------------------------------------


def _instance(ladder, name):
    return next(i for i in ladder if i.instance_id == name)


def test_the_gate_sees_penalties_too_weak_to_bind(ladder, monkeypatch):
    real = encode_module.derive_penalties

    def weak(*args, **kwargs):
        strengths = real(*args, **kwargs)
        return dataclasses.replace(
            strengths,
            validity=strengths.validity * 1e-4,
            full_liquidation=strengths.full_liquidation * 1e-4,
        )

    instance = _instance(ladder, "T0-none-2026")
    _, record = classical(instance)
    monkeypatch.setattr(encode_module, "derive_penalties", weak)
    found, _ = encoded(
        instance, BinaryEncoding(), record["optimum"], n_zeta_bits=GATE_ZETA_BITS
    )
    assert "minimum_is_feasible" in found


def test_the_gate_sees_a_program_without_cardinality(ladder, monkeypatch):
    real = milp_module.lot_grid
    monkeypatch.setattr(
        milp_module, "lot_grid",
        lambda instance: dataclasses.replace(real(instance), cardinality=None),
    )
    found, _ = classical(_instance(ladder, "T0-D2-2026"))
    assert "solves" in found
    assert found["solves"][0] == "exact_pwl_milp"
    assert found["solves"][1]["cardinality"] > 0


def test_the_gate_sees_a_cut_with_the_wrong_sign(ladder, monkeypatch):
    real = milp_module.variance_gradient
    monkeypatch.setattr(
        milp_module, "variance_gradient", lambda i, s: -real(i, s)
    )
    found, _ = classical(_instance(ladder, "T0-none-2026"))
    assert found


def test_the_gate_sees_a_cost_the_program_got_wrong(ladder, monkeypatch):
    real = milp_module.cell_costs

    def shifted(instance, columns, **kwargs):
        costs = real(instance, columns, **kwargs)
        costs[columns.lots > 0] *= 3.0
        return costs

    monkeypatch.setattr(milp_module, "cell_costs", shifted)
    found, _ = classical(_instance(ladder, "T0-D1-2026"))
    assert found
```

- [ ] **Step 3: Run the gate**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_m3b_gate.py -q`
Expected: 18 passed, in a little over a minute.

**If a gate test fails, stop.** Do not change a tolerance, a width, an expected set or a count. Report the check, the instance and the encoding, with the values in the assertion message. The four `test_the_gate_sees_…` tests are the exception in kind: if one of those fails, the gate has stopped seeing a defect it used to see, and that is reported the same way.

- [ ] **Step 4: Write the T1 measurement**

Create `tests/solvers/test_t1_bracket.py`:

```python
"""The exact baseline at T1 -- recorded, not gated.

Parent spec section 5.5 gives T1 an exact MILP bracket as its ground truth,
and M4's optimality gaps will be stated against it. Nothing can check the
program's answer here the way enumeration does at T0: the smallest T1 grid
holds ten million schedules and the largest 2e27. What is asserted is what
can be: the schedule is feasible, the upper end is the evaluator's, and the
lower end is below it.
"""

import pytest

from quantic.problem.objective import evaluate, optimal_zeta
from quantic.problem.risk import CVaRRisk
from quantic.solvers.base import OPTIMAL, TIME_LIMIT, Budget
from quantic.solvers.classical.exact_pwl_milp import exact_pwl_milp
from tests.ladder import tier

BUDGET_S = 120.0


@pytest.fixture(scope="module")
def results():
    return {
        instance.instance_id: (
            instance, exact_pwl_milp(instance, budget=Budget(wall_clock_s=BUDGET_S))
        )
        for instance in tier("T1")
    }


def test_the_tier_is_sixteen_combinations(results):
    assert len(results) == 16


def test_every_instance_ends_with_a_verdict(results):
    verdicts = {name: result.status for name, (_, result) in results.items()}
    assert set(verdicts.values()) <= {OPTIMAL, TIME_LIMIT}, verdicts


def test_every_schedule_returned_is_feasible(results):
    for name, (_, result) in results.items():
        assert result.schedule is not None, name
        assert result.feasibility.feasible, (name, dict(result.feasibility.violations))


def test_every_upper_bound_is_the_evaluators(results):
    for name, (instance, result) in results.items():
        zeta = (
            optimal_zeta(instance, result.schedule)
            if isinstance(instance.risk, CVaRRisk) else None
        )
        assert result.bracket.upper == evaluate(
            instance, result.schedule, zeta=zeta
        ).total, name


def test_no_lower_bound_is_above_its_upper(results):
    for name, (_, result) in results.items():
        bracket = result.bracket
        assert bracket.lower <= bracket.upper + 1e-6 * max(1.0, abs(bracket.upper)), name


def test_no_schedule_beats_the_witness_by_less_than_nothing(results):
    """The witness is feasible, so the optimum cannot cost more than it does."""
    for name, (instance, result) in results.items():
        if result.status != OPTIMAL:
            continue
        zeta = (
            optimal_zeta(instance, instance.witness)
            if isinstance(instance.risk, CVaRRisk) else None
        )
        at_witness = evaluate(instance, instance.witness, zeta=zeta).total
        assert result.bracket.upper <= at_witness * (1.0 + 1e-9), name
```

- [ ] **Step 5: Run the T1 measurement**

Run: `.venv/Scripts/python.exe -m pytest tests/solvers/test_t1_bracket.py -q`
Expected: 6 passed, in under a minute. In the dry run every instance was solved to `OPTIMAL`; `TIME_LIMIT` is also a pass, and is recorded in Task 9.

- [ ] **Step 6: Write the report**

Create `tests/solvers/report.py`:

```python
"""Print the tables the M3b gate result document carries.

    python -m tests.solvers.report

Runs the gate's own checks over T0, and the exact program over T1, and prints
what was measured as Markdown. Nothing is asserted here: the gate does that.
"""

from __future__ import annotations

import time

from quantic.encoding.integer import ENCODINGS
from quantic.problem.risk import CVaRRisk
from quantic.solvers.base import Budget
from quantic.solvers.classical.exact_pwl_milp import exact_pwl_milp
from tests.ladder import tier
from tests.solvers.gate import GATE_ZETA_BITS, STUDY_ZETA_BITS, classical, encoded
from tests.solvers.test_t1_bracket import BUDGET_S


def _table(header: tuple[str, ...], rows: list[tuple]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    started = time.perf_counter()
    exact, at_gate, at_study, findings = [], [], [], []

    for instance in tier("T0"):
        found, record = classical(instance)
        findings += [(instance.instance_id, "-", k, v) for k, v in found.items()]
        if record is None:
            continue
        exact.append((
            f"`{record['instance']}`", record["n_feasible"],
            f"{record['optimum']:.6f}", f"{record['brute_force_s']:.2f}",
            f"{record['milp_s']:.2f}", record["milp_rounds"],
            record["milp_binaries"], record["presolve_retries"],
            f"{record['bracket_width']:.1e}",
        ))
        for encoding in ENCODINGS:
            found, row = encoded(
                instance, encoding, record["optimum"], n_zeta_bits=GATE_ZETA_BITS
            )
            findings += [
                (instance.instance_id, encoding.name, k, v) for k, v in found.items()
            ]
            if row is not None:
                at_gate.append((
                    f"`{row['instance']}`", row["encoding"], row["n_vars"],
                    row["scope"], row["n_states"], row["n_scanned"],
                    row["widest_window"], row["n_ties"],
                    "-" if row["regret"] is None else f"{row['regret']:.1e}",
                    f"{row['dense_error']:.1e}",
                ))
            if not isinstance(instance.risk, CVaRRisk):
                continue
            _, row = encoded(
                instance, encoding, record["optimum"], n_zeta_bits=STUDY_ZETA_BITS
            )
            if row is not None and encoding is ENCODINGS[0]:
                at_study.append((
                    f"`{row['instance']}`", f"{record['optimum']:.4f}",
                    f"{row['regret']:.4f}",
                    f"{100.0 * row['regret'] / record['optimum']:.2f}%",
                    f"{row['resolution']:.2f}",
                ))
    gate_s = time.perf_counter() - started

    started = time.perf_counter()
    t1 = []
    for instance in tier("T1"):
        result = exact_pwl_milp(instance, budget=Budget(wall_clock_s=BUDGET_S))
        t1.append((
            f"`{instance.instance_id}`", result.status,
            f"{result.bracket.upper:.6f}", f"{result.bracket.width:.1e}",
            result.detail["rounds"], result.detail["n_binaries"],
            result.detail["presolve_retries"], f"{result.timings['total_s']:.1f}",
        ))
    t1_s = time.perf_counter() - started

    print(f"## Findings\n\n{len(findings)} across every check.\n")
    for row in findings:
        print(f"- {row}")
    print("\n## The exact optimum, two ways (T0)\n")
    print(_table(
        ("instance", "feasible schedules", "optimum", "brute force, s", "MILP, s",
         "rounds", "binaries", "presolve retries", "bracket width"), exact,
    ))
    print(f"\n## The QUBO's minimum at {GATE_ZETA_BITS} zeta bits (T0)\n")
    print(_table(
        ("instance", "encoding", "variables", "scope", "states", "scanned",
         "widest window", "ties", "regret", "dense error"), at_gate,
    ))
    print(f"\n## Regret at the study's {STUDY_ZETA_BITS} zeta bits (T0, CVaR cells)\n")
    print(_table(
        ("instance", "optimum", "regret", "regret / optimum", "declared resolution"),
        at_study,
    ))
    print("\n## The exact program at T1\n")
    print(_table(
        ("instance", "status", "upper", "bracket width", "rounds", "binaries",
         "presolve retries", "seconds"), t1,
    ))
    print(f"\nGate checks: {gate_s:.0f} s. T1: {t1_s:.0f} s.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the report**

Run: `.venv/Scripts/python.exe -m tests.solvers.report`
Expected: five Markdown sections, the first reading `0 across every check.`, in about two and a half minutes. Keep the output; Task 9 pastes it.

- [ ] **Step 8: Lint and commit**

Run: `.venv/Scripts/ruff.exe check src tests` and `.venv/Scripts/lint-imports.exe`
Expected: clean, `Contracts: 3 kept, 0 broken.`

```bash
git add tests/solvers/gate.py tests/solvers/test_m3b_gate.py tests/solvers/test_t1_bracket.py tests/solvers/report.py
git commit -m "test(solvers): the M3b gate -- the QUBO's minimum is the exact optimum"
```

---

## Task 9: Measure, and write it down

**Files:**
- Create: `docs/superpowers/specs/2026-09-30-m3b-gate-result.md`

**Interfaces:**
- Consumes: the whole suite, and the report of Task 8.
- Produces: the document M4 reads first.

- [ ] **Step 1: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q --durations=8`
Expected: about 1180 passed and 1 skipped, in ten to twelve minutes. The M3a gate is most of that. The skip is `tests/micro/test_reconstruction_gate.py`, which needs a real paired L2/L3 bundle.

- [ ] **Step 2: Run both linters**

Run: `.venv/Scripts/ruff.exe check src tests` and `.venv/Scripts/lint-imports.exe`
Expected: `All checks passed!` and `Contracts: 3 kept, 0 broken.`

- [ ] **Step 3: Run the report again, uncontended**

Run: `.venv/Scripts/python.exe -m tests.solvers.report > m3b-report.md`
The timings in the document are these, taken with nothing else running. Delete `m3b-report.md` once its tables are pasted; it is not committed.

- [ ] **Step 4: Write the result document**

Create `docs/superpowers/specs/2026-09-30-m3b-gate-result.md` with exactly these sections, in this order. Every figure comes from Steps 1 to 3; none is copied from this plan or from the spec.

```markdown
# M3b gate — result

**Written:** <the date of the run>
**Branch:** `feat/m3b-exact-baseline`
**Spec:** `2026-09-29-m3b-exact-baseline-design.md`
**Plan:** `docs/superpowers/plans/2026-09-30-m3b-exact-baseline.md`

---

## Verdict

<One sentence: the gate passes, or it does not. Then the table.>

| what | result |
|---|---|
| gate | <n> of <n> pass |
| T1 measurement | <n> of <n> pass |
| full suite | <n> pass, <n> skipped, <n> fail |
| `ruff check src tests` | <clean, or the findings> |
| `lint-imports` | <n> kept, <n> broken |
| gate wall time | <from the report's last line> |

Machine: <OS, CPU, memory, Python version, pytest version>.

## What passing establishes, and what it does not

<Three paragraphs, in the spec's terms:
 1. What is proven: on sixteen combinations, under three encodings, at twenty
    zeta bits on the CVaR cells, the QUBO's minimum is a feasible schedule and
    its true objective is the optimum two exact methods agree on.
 2. What is not: the study's QUBOs run at eight bits, and on five cells their
    minimum is not the optimum. State the five and the regret of each.
 3. What is not: in seventeen of forty-eight cases the minimum is over the
    states that decode, and nothing shows that no rejected state lies below
    it. State which encodings and how many cases each.>

## The exact optimum, two ways

<The report's second table, pasted whole.>

## The QUBO's minimum at twenty zeta bits

<The report's third table, pasted whole.>

## Regret at the study's eight zeta bits

<The report's fourth table, pasted whole. Then one sentence on the largest
regret as a share of its optimum.>

## The exact program at T1

<The report's fifth table, pasted whole. Then: how many were solved to
optimality, the slowest and its rounds, and any presolve retries.>

## What the dry run and the build found

| finding | where | disposition |
|---|---|---|
| the outer-approximation loop did not end when HiGHS returned a schedule twice | `exact_pwl_milp`, on a three-bucket instance with cardinality | fixed before the plan was written: the loop stops at a schedule it has already cut |
| HiGHS's presolve fails on small programs that carry cuts | `T0-D1+D2`, fourth round | retried without presolve; counted in `Result.detail` |
| with `optimal_aux`'s rule the closed form still crowned the right winner | `gadget_minimum`, on the eight-scenario instance | the closed form is tested at every state, not at the winner |
| <anything the build itself found, one row each> | | |

## What is still open

<Carried from the spec's section 8 and from the M3a gate result, each in a
sentence or two:
 - the rejected states in the seventeen DECODABLE cases;
 - the study's zeta bit budget, which is M5's to choose;
 - `max_single_cell_gain` is per cell (M3a gate result, "What is still open");
 - nothing checks HiGHS's lower bound at T1.
 Then anything this build opened.>

## What M4 picks up

<The spec's section 13, restated against what was measured: the interface,
the regret to compare a sampler against, the approximand's cost table, and
that optimality gaps are stated against `Bracket.lower`.>
```

- [ ] **Step 5: Check the document against the run**

Read the document once against the report's output. Every number in it appears in the output of Steps 1 to 3. No angle bracket is left.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-09-30-m3b-gate-result.md
git commit -m "docs: record the M3b gate's result"
```

---

## M3b is complete at this point

The encoder is proven against exact optima at T0, and the project has a baseline whose lower bound is a proof and whose upper bound is the evaluator's.

## What M4 picks up

`Solver`, `Result` and `Bracket` are the interface. M4 adds the rest of the classical slate behind the same protocol — CP-SAT, simulated annealing, tabu, greedy, Almgren-Chriss analytic — then the bench runner and the DuckDB store.

Three hand-offs matter. First, a sampler that returns the QUBO's minimum at eight zeta bits has not failed when its schedule is not the baseline's: the regret recorded here is what to compare against. Second, under the binary encoding with D1 on the QUBO minimises the approximand, and `exact_pwl_milp(instance, cell_cost=report.cell_cost)` is the baseline for that comparison. Third, the optimality gap of parent spec §9.1 is stated against `Bracket.lower`, which is proven, and not against the best schedule anyone found.
