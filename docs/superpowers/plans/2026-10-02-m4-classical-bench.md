# Quantic M4: The Classical Slate, the Bench Runner and the Results Store

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A classical crossover table on T0–T1 — every free classical solver measured against proven optima, in a store from which every number regenerates — and the harness M5's quantum solvers plug into.

**Architecture:** Three new problem-level solvers behind the existing `Solver` protocol: `cpsat` (a third exact method whose integer program is a relaxation by directed rounding, so its bound stays proven), `almgren_chriss` (the desk reference, rounded to the lot grid), and three QUBO samplers sharing one sampling loop in `solvers/qubo/`. A bench layer expands a TOML experiment into keyed tasks, runs them on killable worker processes, writes each result to DuckDB from the main process, and resumes by skipping keys already stored. Metrics and a report turn rows into crossover tables, dial effects and a bracket cross-check.

**Tech Stack:** Python 3.11+, NumPy, SciPy (HiGHS through `scipy.optimize.milp`), OR-Tools CP-SAT, `dwave-samplers` and `dimod`, DuckDB, Polars, Typer, pytest, Hypothesis, ruff, import-linter.

**Spec:** `docs/superpowers/specs/2026-10-02-m4-classical-bench-design.md`. Read §15 first: it lists what the dry run changed. Read it with `docs/superpowers/specs/2026-09-30-m3b-gate-result.md`, which hands M4 the interface and the backlog.

**This plan was dry-run.** Every code block below was copied from files that were run, in a throwaway worktree, against the repository at `bfd5ca5`: the full suite (M3a, M3b and M4 gates included) at 1314 passed and 1 skipped, ruff clean, four import contracts kept. The M4 gate experiment ran in 13.5 minutes on 14 workers (1,248 tasks, no errors, no kills). The dry run found what a plan written from the design alone would have carried: a CP-SAT row scale that overflowed 64 bits, a tabu sampler that was not deterministic, an annealer that refuses seeds of `2**31` and above, a success rate that became NaN without a reference row, a busy-store error whose text is localised on Windows, an M3b gate self-test that met the stricter `OPTIMAL` in a new shape, and a gate check whose premise was false (spec §15). The fixes are in the code below. Where a step says what to expect, that is what was seen.

## Global Constraints

- **Python `>=3.11`.** Tests run as `.venv/Scripts/python.exe -m pytest` from the repo root. Test modules import each other as `tests.…`, so the repo root must be the working directory.
- **Windows shell.** Never call bare `python`: it is the Windows Store stub and hangs. Use `.venv/Scripts/python`. Do not pass multi-line text containing apostrophes or backticks through a heredoc: write it to a file with the Write tool and pass the path (`git commit -F <file>`). Check `git log` / `git status` after every git write.
- **Branch.** Work on `feat/m4-classical-bench` (it holds the spec at `bfd5ca5`). Nothing is pushed unless the owner says so.
- **Commits** are made under the repo-local git identity. End each commit message with the `Co-Authored-By` trailer the session specifies.
- **Layer boundaries, enforced by import-linter** (`tests/test_architecture.py`). `bench/` may import everything below it; `solvers/` never imports `bench/`. Within `problem/`, `solvers/` never imports `generator`, `from_bundle` or `liquidation`. `base`, `lot_grid`, `scoring`, `brute_force`, `milp_model`, `exact_pwl_milp`, `cpsat` and `almgren_chriss` never import `encoding/`. `brute_force`, `exact_pwl_milp` and `cpsat` never import one another.
- **Dependencies added:** `ortools` (Task 4), `dwave-samplers` and `dimod` (Task 6), `duckdb` (Task 8), each with `uv add <name>` and nothing else. The dry run resolved duckdb 1.5.6, ortools 9.15.6755, dwave-samplers 1.8.0, dimod 0.12.22, all from wheels on Windows.
- **One evaluator.** `problem.objective.evaluate`, reached through `solvers.classical.scoring.score`, is the only place an objective is computed. A solver reports `evaluate` at its schedule, never its own account of the cost.
- **One verdict on feasibility.** `problem.feasibility.classify`; samples go through `encoding.decode.decode`, which calls it. Nothing is repaired.
- **Statuses.** `OPTIMAL` carries a closed bracket and a feasible schedule (Task 1 enforces it). `TIME_LIMIT`: stopped by the clock, more time could have proven more. `INFEASIBLE`: proven to have no feasible schedule. `HEURISTIC`: a method that never proves anything. Store statuses `ok`, `error`, `killed` are separate: they say what happened to the task.
- **Tolerances, and what each one is.** `1e-9` relative between two calls of the evaluator: round-off. `1e-6` relative on a bracket's width (`base._BRACKET_RTOL`): a solver's own feasibility tolerance. `1e-6` relative between one energy accumulated in two orders: M3b's dense `z'Qz` tolerance, measured worst `6.9e-7` at a dynamic range of `1e10`. None is a modelling tolerance, and none may be loosened to make a test pass.
- **Zeta bits.** The samplers use `STUDY_ZETA_BITS = 8`. The study's choice of width is M5's.
- **Determinism.** Identical inputs produce identical output. No implicit global RNG. Caches live on objects (a worker's `Executor`), never at module level in `src/`. Sampler batch seeds come from `(seed, batch index)`.
- **Worker processes start with `spawn`** on every platform.
- **Strictness over silent repair.** Malformed input raises a typed exception: `ConfigError`, `SchemaMismatch`, `StoreBusy`, `DirtyTree`, `ScaleOverflow`, `EnumerationTooLarge`, `WorkerStartError`.
- **Dispatch on type, never on an attribute name.**
- **Line length 100**, ruff rules `E,F,I,UP,B,SIM`, passing **without suppressions**. A `# noqa` added to make a rule pass is a defect — report the conflict instead.
- **Test-first.** Write the failing test, watch it fail for the stated reason, then implement.
- **Expected-result lines are advisory.** Judge success by whether tests pass, not by matching a count.
- **Never weaken the gate to make it pass.** If a gate check fails, stop and report which check, which instance, which solver and which encoding.

## Review Focus

The inputs the spec implies without spelling out, most likely to bite a person using the bench first. Each has its test in the task that owns the code.

1. **Reading a store while a run writes it.** `quantic bench report` during a multi-day run is the obvious thing to try. DuckDB lets one process open a file, and on Windows the error is the operating system's, in the system's language. Expected: a `StoreBusy` that says a run holds the file and to copy it. — Task 8 (`test_a_store_another_process_is_writing_is_refused_with_a_reason`).
2. **A store path whose directory does not exist yet.** `--store studies/m4/crossover.duckdb` on a fresh checkout. Expected: the directory is created. — Task 8 (`test_a_store_in_a_directory_that_does_not_exist_yet_is_created`).
3. **A report on a store with no reference rows** — a run stopped before its references finished, or an experiment of samplers only. Expected: every cell present, gaps shown as `-`, TTS censored, never `nan`. — Task 10 (`test_a_report_on_a_store_with_no_references_says_so_cell_by_cell`).
4. **A task whose instance cannot be built** — a mistyped dial label in a hand-written task, a generator failure at a new seed. Expected: an `error` row naming the exception, and the run carries on. — Task 9 (`test_an_instance_that_cannot_be_built_is_an_error_row_not_a_crash`).
5. **A CP-SAT program that does not fit in 64-bit integers** — wider-ranged costs at a future tier. Expected: a typed `ScaleOverflow`, which the bench records as an `error` row, never a silently wrapped coefficient. — Task 4 (`test_a_program_too_wide_for_64_bit_integers_is_refused_by_type`).

## File Structure

| File | Responsibility |
|---|---|
| `src/quantic/solvers/base.py` | `Result` enforces `OPTIMAL`; new status `HEURISTIC` |
| `src/quantic/solvers/classical/exact_pwl_milp.py` | backlog: no clamp, retry time, round count, status 1 |
| `src/quantic/solvers/classical/lot_grid.py` | backlog: a constraint kind given twice raises |
| `src/quantic/solvers/classical/qubo_minimum.py` | backlog: `EnumerationTooLarge`, witness tolerance `1e-6` |
| `src/quantic/solvers/classical/cpsat.py` | `cpsat`, `CpSat`, `ScaleOverflow` |
| `src/quantic/solvers/classical/almgren_chriss.py` | `almgren_chriss`, `AlmgrenChrissSolver`, `holdings_fraction`, `kappa_for`, `nearest_row` |
| `src/quantic/solvers/qubo/sampler.py` | `run_sampler`, `QuboSampler`, `Sample`, `to_bqm`, `energies`, `encoding_named`, `STUDY_ZETA_BITS` |
| `src/quantic/solvers/qubo/{simulated_annealing,tabu,greedy}.py` | one `draw` each |
| `src/quantic/bench/instances.py` | `synthetic_params`, `tier_instances`, `InstanceCache`, `GATE_SEED` |
| `src/quantic/bench/config.py` | `load`, `Experiment`, `SolverPlan`, `ConfigError` |
| `src/quantic/bench/keys.py` | `Task`, `CodeVersion`, `code_version`, `expand`, reference solver names |
| `src/quantic/bench/store.py` | `Store`, `blank_row`, `RUN_COLUMNS`, store statuses, `StoreBusy`, `SchemaMismatch` |
| `src/quantic/bench/execute.py` | `Executor`: one task in a worker, into rows |
| `src/quantic/bench/pool.py` | `run_tasks`: killable worker processes |
| `src/quantic/bench/runner.py` | `plan`, `run`, `deadline_s`, `DirtyTree` |
| `src/quantic/bench/metrics.py` | `tts99`, `scored`, `cells` |
| `src/quantic/bench/report.py` | `report`, `crossover`, `best_encoding`, `dial_effects`, `bracket_check` |
| `src/quantic/cli.py` | `quantic bench plan | run | report` |
| `pyproject.toml` | four dependencies, two contract edits |
| `experiments/m4-gate.toml`, `experiments/m4-crossover.toml` | the gate and the study |
| `tests/ladder.py` | imports its parameters from `bench/instances.py` |
| `tests/bench/rows.py` | row builders the bench tests share |
| `tests/bench/stand_ins.py` | solvers that hang or raise on purpose |
| `tests/bench/test_m4_gate.py` | the gate |

---

### Task 1: `Result` — the `OPTIMAL` claim, and `HEURISTIC`

Spec §8 item 1 and D7. The bench trusts `Result`, so this lands first.

**Files:**
- Modify: `src/quantic/solvers/base.py`
- Test: `tests/solvers/test_base.py`
- Modify: `tests/solvers/test_m3b_gate.py` (one self-test, Step 5)

**Interfaces:**
- Consumes: nothing new.
- Produces: `quantic.solvers.base.HEURISTIC = "heuristic"`; `Result(status=OPTIMAL, ...)` raises `ValueError` unless `schedule`, `bracket` and `feasibility` are present, `feasibility.feasible` is true, and `bracket.width <= _BRACKET_RTOL * max(1, |bracket.upper|)`.

- [ ] **Step 1: Write the failing tests**

In `tests/solvers/test_base.py`, add `HEURISTIC` to the `quantic.solvers.base` import, add `from quantic.problem.schedule import Schedule` to the imports, widen two parametrisations, and append three tests:

```python
@pytest.mark.parametrize("missing", ["schedule", "bracket", "feasibility"])
def test_an_optimal_result_has_to_carry_what_makes_it_optimal(missing):
    with pytest.raises(ValueError, match="optimal"):
        _result(**{missing: None})


@pytest.mark.parametrize("status", [TIME_LIMIT, INFEASIBLE, HEURISTIC])
def test_a_result_that_is_not_optimal_may_carry_nothing(status):
    result = _result(
        status=status, schedule=None, objective=None, bracket=None, feasibility=None
    )
    assert result.schedule is None
```

```python
def test_an_optimal_result_with_an_open_bracket_is_refused():
    with pytest.raises(ValueError, match="closed"):
        _result(bracket=Bracket(lower=0.5, upper=1.0))


def test_an_optimal_result_inside_the_tolerance_is_accepted():
    assert _result(bracket=Bracket(lower=1.0 - 1e-7, upper=1.0)).status == OPTIMAL


def test_an_optimal_result_on_a_schedule_that_breaches_a_constraint_is_refused():
    inst = plain()
    oversold = Schedule(lots=((3, 3, 3), (1, 1, 1)))
    with pytest.raises(ValueError, match="feasible"):
        _result(schedule=oversold, feasibility=classify(inst, oversold))
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/solvers/test_base.py -q`
Expected: FAIL — `ImportError: cannot import name 'HEURISTIC'`.

- [ ] **Step 3: Implement**

In `src/quantic/solvers/base.py`, replace the last paragraph of the module docstring:

```python
A sampler's energy trace travels in ``Result.detail["trace"]`` as
``(seconds_since_start, best_energy)`` pairs; only samplers have one.
"""
```

Replace the status constants:

```python
OPTIMAL = "optimal"
TIME_LIMIT = "time_limit"
INFEASIBLE = "infeasible"
# A schedule from a method that never proves anything: a sampler, a rounding.
# Not TIME_LIMIT, which says that more time could have produced a proof.
HEURISTIC = "heuristic"

_STATUSES = (OPTIMAL, TIME_LIMIT, INFEASIBLE, HEURISTIC)
```

In `Result.__post_init__`, replace the `if self.status == OPTIMAL and (...)` block with a call, and add the method after `__post_init__`:

```python
        if self.status == OPTIMAL:
            self._check_optimal()
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))

    def _check_optimal(self) -> None:
        """``OPTIMAL`` is a claim, and the result carries its evidence."""
        if self.schedule is None or self.bracket is None or self.feasibility is None:
            raise ValueError(
                "an optimal result carries the schedule, the bracket and the "
                "feasibility report that make it optimal"
            )
        if not self.feasibility.feasible:
            raise ValueError(
                f"an optimal result's schedule must be feasible; {self.solver} "
                f"returned one that breaches {dict(self.feasibility.violations)}"
            )
        allowed = _BRACKET_RTOL * max(1.0, abs(self.bracket.upper))
        if self.bracket.width > allowed:
            raise ValueError(
                f"an optimal result's bracket must be closed; {self.solver} "
                f"returned [{self.bracket.lower!r}, {self.bracket.upper!r}], "
                f"{self.bracket.width!r} wide against {allowed!r}"
            )
```

- [ ] **Step 4: Run the solver tests, gates excepted**

Run: `.venv/Scripts/python -m pytest tests/solvers -q --deselect tests/solvers/test_m3b_gate.py`
Expected: PASS. Every existing `OPTIMAL` result (brute force, the MILP) already carries a closed bracket and a feasible schedule.

- [ ] **Step 5: Bring one M3b gate self-test up to date**

`test_the_gate_sees_a_program_without_cardinality` sabotages the MILP by dropping cardinality and asserts that the gate notices. It still does, one step earlier: the MILP's `Result(status=OPTIMAL)` on a schedule that breaches cardinality now raises, and the gate records the refusal as its `solves` finding — a string, where the test expected a `(solver, violations)` tuple.

Run: `.venv/Scripts/python -m pytest tests/solvers/test_m3b_gate.py -q -k without_cardinality`
Expected: FAIL — `AssertionError: assert 'V' == 'exact_pwl_milp'`.

In `tests/solvers/test_m3b_gate.py`, replace the three assertions at the end of that test with the same two facts — which solver, which constraint, breached — read from the refusal:

```python
    found, _ = classical(_instance(ladder, "T0-D2-2026"))
    # Since M4, Result refuses an OPTIMAL schedule that breaches a constraint, so
    # the sabotage is caught one step earlier: the gate records the refusal, and
    # the refusal names the solver and the constraint it breached.
    assert "solves" in found
    assert "exact_pwl_milp returned one that breaches" in found["solves"]
    assert "'cardinality': " in found["solves"]
    assert "'cardinality': 0.0" not in found["solves"]
```

Run the same command. Expected: PASS. The check is as strict as before; only the shape of the evidence changed.

- [ ] **Step 6: Lint and commit**

Run: `.venv/Scripts/ruff check src tests`
Write the message to a file in the scratchpad and commit:

```
fix(solvers): OPTIMAL carries its proof; HEURISTIC for methods that never prove
```

```bash
git add src/quantic/solvers/base.py tests/solvers/test_base.py tests/solvers/test_m3b_gate.py
git commit -F <message file>
```

---

### Task 2: The MILP's backlog

Spec §8 items 2, 5, 6 and 7.

**Files:**
- Modify: `src/quantic/solvers/classical/exact_pwl_milp.py`
- Test: `tests/solvers/test_exact_pwl_milp.py`

**Interfaces:**
- Consumes: `Bracket` raising on `lower > upper + slack` (exists).
- Produces: `exact_pwl_milp` raises `RuntimeError` on scipy status 1 when no wall clock was set; `Bracket.lower` is the proven bound unclamped; `detail["rounds"]` counts only rounds that called HiGHS; the presolve retry gets the time left.

- [ ] **Step 1: Write the failing tests**

Append to `tests/solvers/test_exact_pwl_milp.py`:

```python
def test_a_limit_nobody_set_is_raised_and_not_called_a_time_limit(monkeypatch):
    real = module.milp

    def iteration_limit(**kwargs):
        solved = real(**kwargs)
        return types.SimpleNamespace(
            status=1, x=solved.x, fun=solved.fun, message="iteration limit",
            mip_dual_bound=solved.mip_dual_bound,
        )

    monkeypatch.setattr(module, "milp", iteration_limit)
    with pytest.raises(RuntimeError, match="never set"):
        exact_pwl_milp(plain())


def test_a_bound_above_the_schedule_is_raised_not_clamped(monkeypatch):
    """The old min(proven, total) turned a wrong bound into a plausible one."""
    real = module.milp

    def wrong_bound(**kwargs):
        solved = real(**kwargs)
        return types.SimpleNamespace(
            status=1, x=solved.x, fun=solved.fun, message="time limit",
            mip_dual_bound=solved.mip_dual_bound + 1e3,
        )

    monkeypatch.setattr(module, "milp", wrong_bound)
    with pytest.raises(ValueError, match="one of the two is wrong"):
        exact_pwl_milp(plain(), budget=Budget(wall_clock_s=60.0))


def test_the_presolve_retry_gets_the_time_that_is_left(monkeypatch):
    real = module.milp
    limits = []
    clock = iter([0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock, 10.0))

    def presolve_fails(**kwargs):
        limits.append(kwargs["options"].get("time_limit"))
        if kwargs["options"].get("presolve", True):
            return types.SimpleNamespace(
                status=4, x=None, fun=None, message="Solve error", mip_dual_bound=None
            )
        return real(**kwargs)

    monkeypatch.setattr(module, "milp", presolve_fails)
    exact_pwl_milp(plain(), budget=Budget(wall_clock_s=60.0))
    assert limits[:2] == [60.0, 50.0]


def test_a_round_the_budget_never_let_run_is_not_counted(monkeypatch):
    """Variance needs a second round; the clock runs out before it starts."""
    calls = []
    real = module.milp
    clock = iter([0.0, 0.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock, 100.0))

    def counted(**kwargs):
        calls.append(1)
        return real(**kwargs)

    monkeypatch.setattr(module, "milp", counted)
    result = exact_pwl_milp(plain(), budget=Budget(wall_clock_s=60.0))
    assert result.status == TIME_LIMIT
    assert len(calls) == 1
    assert result.detail["rounds"] == 1
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/solvers/test_exact_pwl_milp.py -q -k "nobody_set or clamped or presolve_retry or never_let"`
Expected: 4 FAIL — no `RuntimeError`; no `ValueError` (the clamp hides the bound); the retry sees `60.0`, not `50.0`; `rounds == 2` against one call.

- [ ] **Step 3: Implement**

In `exact_pwl_milp`, move `rounds += 1` from the top of the loop to after the time check:

```python
    while True:
        options = {"mip_rel_gap": 0.0}
        if limit is not None:
            left = limit - (time.perf_counter() - started)
            if left <= 0.0:
                status = TIME_LIMIT
                break
            options["time_limit"] = left
        rounds += 1
```

Give the presolve retry the time that is left:

```python
            presolve_retries += 1
            retry = {**options, "presolve": False}
            if limit is not None:
                left = limit - (time.perf_counter() - started)
                if left <= 0.0:
                    status = TIME_LIMIT
                    break
                retry["time_limit"] = left
            solved = milp(**program, options=retry)
```

Refuse a status 1 nobody asked for:

```python
        if solved.status == _SCIPY_LIMIT:
            if limit is None:
                # No clock was set, so status 1 is an iteration or node limit
                # this solver never sets. Calling it TIME_LIMIT would hide it.
                raise RuntimeError(
                    f"HiGHS stopped on {instance.instance_id} at a limit this "
                    f"solver never set: {solved.message}"
                )
            status = TIME_LIMIT
            break
```

And report the proven bound as it is:

```python
        bracket=Bracket(lower=proven, upper=total),
```

- [ ] **Step 4: Run the MILP tests**

Run: `.venv/Scripts/python -m pytest tests/solvers/test_exact_pwl_milp.py tests/solvers/test_t1_bracket.py -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```
fix(solvers): the MILP reports its bound unclamped and counts only rounds it ran
```

```bash
git add src/quantic/solvers/classical/exact_pwl_milp.py tests/solvers/test_exact_pwl_milp.py
git commit -F <message file>
```

---

### Task 3: The rest of the backlog

Spec §8 items 3, 4, 9 and 10, and the recorded block-lot note.

**Files:**
- Modify: `src/quantic/solvers/classical/lot_grid.py`
- Modify: `src/quantic/solvers/classical/qubo_minimum.py`
- Modify: `docs/superpowers/specs/2026-09-30-m3b-gate-result.md`
- Test: `tests/solvers/test_lot_grid.py`, `tests/solvers/test_qubo_minimum.py`

**Interfaces:**
- Produces: `quantic.solvers.classical.qubo_minimum.EnumerationTooLarge(RuntimeError)`; `lot_grid` raises `ValueError` matching `"more than one"` when a constraint kind appears twice.

- [ ] **Step 1: Write the failing tests**

Append to `tests/solvers/test_lot_grid.py` (its imports already hold `Block`, `BlockTrades`, `FullLiquidation`, `MinParticipation`, `instance`):

```python
@pytest.mark.parametrize(
    "repeated",
    [
        (MinParticipation(min_lots=(1, 1)), MinParticipation(min_lots=(2, 1))),
        (
            BlockTrades(blocks=(Block(asset=0, bucket=0, lots=3),)),
            BlockTrades(blocks=(Block(asset=1, bucket=1, lots=2),)),
        ),
    ],
)
def test_a_constraint_given_twice_is_refused_not_overwritten(repeated):
    inst = instance(constraints=(FullLiquidation(), *repeated))
    with pytest.raises(ValueError, match="more than one"):
        lot_grid(inst)
```

Append to `tests/solvers/test_qubo_minimum.py`:

```python
def test_too_many_states_under_the_bound_raise_a_typed_error(monkeypatch):
    """Not the built-in MemoryError: the bench catches errors by type."""
    monkeypatch.setattr(module, "_MAX_CANDIDATES", 0)
    with pytest.raises(module.EnumerationTooLarge, match="witness"):
        qubo_minimum(encode(plain(), BinaryEncoding()))
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/solvers/test_lot_grid.py tests/solvers/test_qubo_minimum.py -q -k "twice or typed"`
Expected: 3 FAIL — no `ValueError` (the second constraint silently wins); `AttributeError: module ... has no attribute 'EnumerationTooLarge'`.

- [ ] **Step 3: Implement**

In `lot_grid`, before the `for c in instance.constraints:` loop:

```python
    kinds = [type(c) for c in instance.constraints]
    repeated = sorted({k.__name__ for k in kinds if kinds.count(k) > 1})
    if repeated:
        raise ValueError(
            f"{instance.instance_id} carries more than one {repeated} constraint. "
            "The grid reads each kind once, so a second would silently replace "
            "the first"
        )
```

and above the line `row.append((0, block) if block >= minimums[i] else (0,))`, record why a block is not clamped to the cap:

```python
                # The block is taken as the venue fixes it, even above the
                # per-bucket cap: the cap bounds what a schedule chooses, a block
                # is a size it does not choose. The generator carves blocks from
                # a witness inside the cap, so the ladder never meets the case.
```

In `qubo_minimum.py`, after `_MAX_CANDIDATES = 20_000_000`, add the tolerance and the error:

```python
# How far above the witness's energy a state may sit and still be kept. The
# split sum and the dense z'Qz accumulate in different orders, and at the
# dynamic ranges seen 1e-9 left round-off no room. M3b's final review measured
# the margin to the nearest state that matters at 5e-3 or more.
_WITNESS_RTOL = 1e-6


class EnumerationTooLarge(RuntimeError):
    """More states pass the bound than this enumeration will hold."""
```

use it for the threshold:

```python
    threshold = incumbent + _WITNESS_RTOL * max(1.0, abs(incumbent))
```

and replace `raise MemoryError(` with `raise EnumerationTooLarge(`.

In `docs/superpowers/specs/2026-09-30-m3b-gate-result.md`, line 178, replace the sentence `These, and the reviewers' longer list in \`progress.md\`, go to the final review.` with:

```
The final review passed these to M4, whose design (`2026-10-02-m4-classical-bench-design.md` §8) fixes the duplicate descriptors, the stale retry limit, the round count, the status-1 label and the `MemoryError`, and records the block-lot clamp as correct.
```

- [ ] **Step 4: Run the solver tests, gates excepted**

Run: `.venv/Scripts/python -m pytest tests/solvers -q --deselect tests/solvers/test_m3b_gate.py`
Expected: PASS.

- [ ] **Step 5: Run the M3b gate**

The witness tolerance is inside the enumeration the gate relies on.
Run: `.venv/Scripts/python -m pytest tests/solvers/test_m3b_gate.py -q`
Expected: 18 passed, about 80 s.

- [ ] **Step 6: Lint and commit**

```
fix(solvers): a constraint given twice raises; enumeration raises a typed error
```

```bash
git add src/quantic/solvers/classical/lot_grid.py src/quantic/solvers/classical/qubo_minimum.py docs/superpowers/specs/2026-09-30-m3b-gate-result.md tests/solvers/test_lot_grid.py tests/solvers/test_qubo_minimum.py
git commit -F <message file>
```

---

### Task 4: CP-SAT, the third exact method

Spec §4.1. The integer program is a relaxation by directed rounding; read the module docstring before the code.

**Files:**
- Create: `src/quantic/solvers/classical/cpsat.py`
- Modify: `pyproject.toml`
- Test: `tests/solvers/test_cpsat.py`

**Interfaces:**
- Consumes: `lot_grid`, `milp_model.{cell_costs, columns_of, loss_model, permanent_constant, variance_gradient}`, `scoring.score`, `HEURISTIC` (Task 1).
- Produces: `cpsat(instance, *, budget=None, seed=0) -> Result` with `solver="cpsat"`; `CpSat(seed=0)` with `.name == "cpsat"` and `.solve(instance, budget)`; `ScaleOverflow(ValueError)`. `detail` holds `rounds`, `n_cuts`, `n_binaries`, `ticks_per_unit`, `slack_ticks`.

- [ ] **Step 1: Add the dependency**

Run: `uv add ortools`
Expected: `pyproject.toml` gains `"ortools>=…"`; `uv.lock` changes. Check with `.venv/Scripts/python -c "import ortools; print(ortools.__version__)"`.

- [ ] **Step 2: Write the failing tests**

Create `tests/solvers/test_cpsat.py`:

```python
import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantic.problem.objective import evaluate, optimal_zeta
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.solvers.base import INFEASIBLE, OPTIMAL, TIME_LIMIT, Budget, Solver
from quantic.solvers.classical import cpsat as module
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.cpsat import CpSat, cpsat
from tests.solvers.instances import (
    EVERY_SOLVABLE,
    cvar,
    discrete,
    infeasible,
    plain,
)

REL = 1e-9


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_it_finds_what_enumeration_finds(build):
    inst = build()
    found = cpsat(inst)
    assert found.status == OPTIMAL
    assert found.feasibility.feasible
    assert found.objective.total == pytest.approx(
        brute_force(inst).objective.total, rel=REL
    )


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_bound_is_below_the_optimum_and_the_top_is_the_evaluators(build):
    inst = build()
    found = cpsat(inst)
    optimum = brute_force(inst).objective.total
    zeta = optimal_zeta(inst, found.schedule) if isinstance(inst.risk, CVaRRisk) else None
    assert found.bracket.upper == evaluate(inst, found.schedule, zeta=zeta).total
    assert found.bracket.lower <= optimum + REL * abs(optimum)


def test_an_instance_nobody_can_solve_is_reported_and_not_raised():
    found = cpsat(infeasible())
    assert found.status == INFEASIBLE
    assert found.schedule is None


@settings(max_examples=40, deadline=None)
@given(
    lam=st.floats(min_value=0.0, max_value=5e-3),
    use_cvar=st.booleans(),
    use_discrete=st.booleans(),
)
def test_the_scaled_bound_never_exceeds_the_true_optimum(lam, use_cvar, use_discrete):
    """The rounding to integers keeps a relaxation, so the bound stays a bound."""
    inst = discrete() if use_discrete else plain()
    if use_cvar:
        inst = dataclasses.replace(inst, risk=dataclasses.replace(cvar().risk, lam=lam))
    else:
        inst = dataclasses.replace(inst, risk=VarianceRisk(lam=lam))
    found = cpsat(inst)
    optimum = brute_force(inst).objective.total
    assert found.bracket.lower <= optimum + REL * max(1.0, abs(optimum))


def test_running_out_of_time_keeps_the_best_schedule_and_an_honest_bracket(monkeypatch):
    clock = iter([0.0, 0.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock, 100.0))
    inst = plain()
    found = cpsat(inst, budget=Budget(wall_clock_s=60.0))
    assert found.status == TIME_LIMIT
    assert found.feasibility.feasible
    assert found.bracket.lower <= brute_force(inst).objective.total
    assert found.detail["rounds"] == 1


def test_the_same_seed_gives_the_same_answer():
    first = cpsat(discrete(), seed=3)
    again = cpsat(discrete(), seed=3)
    assert first.schedule == again.schedule
    assert first.bracket == again.bracket


def test_it_is_a_solver():
    assert isinstance(CpSat(seed=1), Solver)
    assert CpSat(seed=1).solve(plain(), Budget()).status == OPTIMAL


def test_a_program_too_wide_for_64_bit_integers_is_refused_by_type(monkeypatch):
    monkeypatch.setattr(module, "_ROW_LIMIT", 1.0)
    with pytest.raises(module.ScaleOverflow, match=r"2\*\*60"):
        cpsat(cvar())
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/solvers/test_cpsat.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.solvers.classical.cpsat'`.

- [ ] **Step 4: Implement**

Create `src/quantic/solvers/classical/cpsat.py`:

```python
"""The exact optimum through OR-Tools CP-SAT -- parent spec section 7.2's opponent.

CP-SAT is the third exact method. It shares the lot grid and the program's
pieces with ``exact_pwl_milp`` -- one binary per admissible lot value of each
cell, the cell costs read off the evaluator -- and does not import that module
or ``brute_force``: a bracket that agrees with the MILP's is evidence only if
nothing but the shared pieces is shared.

CP-SAT takes integers only. Every real number is turned into one by rounding in
the direction that keeps the integer program a **relaxation**: a cost
coefficient is floored, the right-hand side of a ``>=`` constraint is floored,
a coefficient a variable is subtracted with is ceiled. Every feasible schedule
then has an integer point costing at most its true objective, in ticks, plus
the ceiling of each auxiliary -- one tick per auxiliary, ``slack`` below. So

    optimum >= (CP-SAT's bound - slack) / ticks_per_unit + constant

and ``Bracket.lower`` is that. ``Bracket.upper`` is the evaluator at the
schedule, as for every solver.

Under CVaR the weights ``w_s / (1 - alpha)`` are real and multiply ``zeta`` in
every scenario's row. They are moved into the rows, scaled by the largest power of two that fits
and ceiled there, so the objective's coefficients stay exactly one.
"""

from __future__ import annotations

import math
import time

import numpy as np
from ortools.sat.python import cp_model

from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.base import (
    _BRACKET_RTOL,
    HEURISTIC,
    INFEASIBLE,
    OPTIMAL,
    TIME_LIMIT,
    Bracket,
    Budget,
    Result,
)
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.milp_model import (
    cell_costs,
    columns_of,
    loss_model,
    permanent_constant,
    variance_gradient,
)
from quantic.solvers.classical.scoring import score

NAME = "cpsat"

# Ticks per unit of the objective's scale. One tick of slack per auxiliary is
# then about 1e-9 of the objective.
_TICKS = 2.0**30
# Each CVaR row is scaled by the largest power of two, at most 2**_ROW_BITS,
# that keeps it under _ROW_LIMIT; the weight on zeta is resolved to that.
_ROW_BITS = 40
_ROW_LIMIT = 2.0**60
# CP-SAT works in int64; stay well inside it.
_INT_LIMIT = 2**62
_CUT_RTOL = 1e-9
_MAX_ROUNDS = 10_000


class ScaleOverflow(ValueError):
    """The scaled program would not fit in CP-SAT's 64-bit integers."""


def _check(values, what: str) -> None:
    if any(abs(int(v)) >= _INT_LIMIT for v in values):
        raise ScaleOverflow(f"{what} exceeds 2**62 once scaled")


def cpsat(instance: Instance, *, budget: Budget | None = None, seed: int = 0) -> Result:
    """Minimise the objective over the lot grid with CP-SAT, with a proven bound."""
    started = time.perf_counter()
    limit = None if budget is None else budget.wall_clock_s
    risk = instance.risk
    if not isinstance(risk, (VarianceRisk, CVaRRisk)):
        raise NotImplementedError(
            f"cpsat does not know risk spec {risk.name!r} ({type(risk).__name__}). "
            "Dispatch is on type, never on a lam attribute"
        )

    grid = lot_grid(instance)
    columns = columns_of(grid)
    n_y = columns.n
    lam = float(risk.lam)
    cvar = isinstance(risk, CVaRRisk) and lam > 0.0
    variance = isinstance(risk, VarianceRisk) and lam > 0.0

    reference, _ = score(instance, instance.witness)
    ticks = _TICKS / max(1.0, abs(reference.total))
    constant = permanent_constant(instance)

    model = cp_model.CpModel()
    y = [model.NewBoolVar(f"y{c}") for c in range(n_y)]
    _add_coupling(model, y, columns, grid)

    cost = [math.floor(ticks * float(v)) for v in cell_costs(instance, columns)]
    _check(cost, "a cell cost")
    objective = [cost[c] * y[c] for c in range(n_y)]
    slack = 0

    theta = None
    if variance:
        top = _variance_ceiling(instance)
        theta = model.NewIntVar(0, math.ceil(ticks * top) + 1, "theta")
        objective.append(theta)
        slack = 1
    if cvar:
        constant += _add_cvar(model, instance, columns, y, ticks, objective)
        slack = 1 + risk.scenarios.n_scenarios

    model.Minimize(sum(objective))
    for c, value in _hint(instance, columns).items():
        model.AddHint(y[c], value)

    best: tuple[float, Schedule] | None = None
    proven = -math.inf
    rounds = 0
    n_cuts = 0
    cut_at: set[Schedule] = set()
    status = OPTIMAL
    if variance:
        _add_cut(model, instance, columns, y, theta, ticks, instance.witness)
        cut_at.add(instance.witness)
        n_cuts += 1

    while True:
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1
        solver.parameters.random_seed = int(seed)
        if limit is not None:
            left = limit - (time.perf_counter() - started)
            if left <= 0.0:
                status = TIME_LIMIT
                break
            solver.parameters.max_time_in_seconds = left
        rounds += 1
        outcome = solver.Solve(model)

        if outcome == cp_model.MODEL_INVALID:
            raise RuntimeError(
                f"CP-SAT rejected the model for {instance.instance_id}: "
                f"{model.Validate()}"
            )
        if outcome == cp_model.INFEASIBLE:
            if best is not None or rounds > 1:
                raise RuntimeError(
                    f"CP-SAT reported {instance.instance_id} infeasible in round "
                    f"{rounds} after a feasible round; cuts never remove a schedule"
                )
            return Result(
                solver=NAME, status=INFEASIBLE, schedule=None, zeta=None,
                objective=None, bracket=None, feasibility=None,
                timings={"total_s": time.perf_counter() - started},
                detail={"rounds": rounds, "n_cuts": n_cuts, "n_binaries": n_y},
            )

        bound = (solver.BestObjectiveBound() - slack) / ticks + constant
        proven = max(proven, bound)

        if outcome in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            chosen = np.array([solver.Value(v) for v in y], dtype=float)
            schedule = columns.schedule(chosen)
            breakdown, _ = score(instance, schedule)
            if best is None or breakdown.total < best[0]:
                best = (breakdown.total, schedule)

        if outcome != cp_model.OPTIMAL:
            status = TIME_LIMIT
            break
        if not variance:
            break

        reported = solver.Value(theta) / ticks
        gap = breakdown.risk_cost - reported
        if gap <= _CUT_RTOL * max(1.0, abs(breakdown.total)) or schedule in cut_at:
            break
        if rounds >= _MAX_ROUNDS:
            raise RuntimeError(
                f"outer approximation on {instance.instance_id} has not converged "
                f"in {_MAX_ROUNDS} rounds; the last gap was {gap!r}"
            )
        _add_cut(model, instance, columns, y, theta, ticks, schedule)
        model.ClearHints()
        for c, value in _hint_from(columns, schedule).items():
            model.AddHint(y[c], value)
        cut_at.add(schedule)
        n_cuts += 1

    timings = {"total_s": time.perf_counter() - started}
    detail = {
        "rounds": rounds, "n_cuts": n_cuts, "n_binaries": n_y,
        "ticks_per_unit": ticks, "slack_ticks": slack,
    }
    if best is None:
        return Result(
            solver=NAME, status=TIME_LIMIT, schedule=None, zeta=None,
            objective=None, bracket=None, feasibility=None,
            timings=timings, detail=detail,
        )

    total, schedule = best
    breakdown, zeta = score(instance, schedule)
    bracket = Bracket(lower=proven, upper=total)
    if status == OPTIMAL and bracket.width > _BRACKET_RTOL * max(1.0, abs(total)):
        # CP-SAT proved its integer program; the rounding that made it integer
        # left the bracket open. That is a schedule with a bound, not a proof.
        status = HEURISTIC
    return Result(
        solver=NAME, status=status, schedule=schedule, zeta=zeta,
        objective=breakdown, bracket=bracket,
        feasibility=classify(instance, schedule),
        timings=timings, detail=detail,
    )


def _add_coupling(model, y, columns, grid) -> None:
    """One value per cell, full liquidation per name, cardinality per bucket."""
    n_assets, n_buckets = columns.first.shape
    every = np.arange(columns.n)
    for i in range(n_assets):
        for t in range(n_buckets):
            mine = every[(columns.asset == i) & (columns.bucket == t)]
            model.AddExactlyOne(y[int(c)] for c in mine)
    for i in range(n_assets):
        mine = every[columns.asset == i]
        model.Add(
            sum(int(columns.lots[c]) * y[int(c)] for c in mine)
            == int(grid.initial_lots[i])
        )
    if grid.cardinality is not None:
        for t in range(n_buckets):
            mine = every[(columns.bucket == t) & (columns.lots > 0)]
            model.Add(sum(y[int(c)] for c in mine) <= int(grid.cardinality))


def _variance_ceiling(instance: Instance) -> float:
    """No schedule's variance term exceeds ``lam * T * lambda_max * |X|**2``."""
    sigma = np.asarray(instance.params.price_covariance(), dtype=float)
    top_eig = float(np.max(np.linalg.eigvalsh(sigma)))
    shares = np.asarray(instance.initial_lots, dtype=float) * instance.lot_sizes()
    return float(instance.risk.lam) * instance.n_buckets * max(top_eig, 0.0) * float(
        shares @ shares
    )


def _add_cut(model, instance, columns, y, theta, ticks, schedule: Schedule) -> None:
    """``theta >= v(x^) + g . (x - x^)``, floored so it stays valid."""
    breakdown, _ = score(instance, schedule)
    gradient = variance_gradient(instance, schedule)
    at = schedule.as_array().astype(float)
    rhs = breakdown.risk_cost - float((gradient * at).sum())
    coef = gradient[columns.asset, columns.bucket] * columns.lots
    rows = [math.floor(ticks * float(a)) for a in coef]
    floor_rhs = math.floor(ticks * rhs)
    _check([*rows, floor_rhs], "a variance cut")
    model.Add(theta >= floor_rhs + sum(rows[c] * y[c] for c in range(columns.n)))


def _add_cvar(model, instance, columns, y, ticks, objective) -> float:
    """Rockafellar-Uryasev in integers. Returns ``lam * ell_lo``, a constant.

    ``zeta = ell_lo + z`` with ``z >= 0``, where ``ell_lo`` is below every loss
    any schedule on the grid can take, so the minimising ``zeta`` (a loss) is
    never cut off. ``w`` carries ``lam * z`` and ``v_s`` carries
    ``lam * p_s * u_s``, both in objective ticks, each with coefficient one.
    """
    risk = instance.risk
    lam = float(risk.lam)
    p = np.asarray(risk.scenarios.weights, dtype=float) / (1.0 - risk.alpha)
    base, tail = loss_model(instance)
    # t[s, c]: what selling column c's lots takes off scenario s's loss.
    t = tail[:, columns.asset, columns.bucket] * columns.lots[None, :]
    n_assets, n_buckets = columns.first.shape
    lowest = base.copy()
    highest = base.copy()
    for i in range(n_assets):
        for tt in range(n_buckets):
            lo = int(columns.first[i, tt])
            hi = lo + int(columns.count[i, tt])
            lowest -= t[:, lo:hi].max(axis=1)
            highest -= t[:, lo:hi].min(axis=1)
    ell_lo = float(lowest.min())
    span = max(float(highest.max()) - ell_lo, 0.0)

    w_top = math.ceil(ticks * lam * span) + 1
    w = model.NewIntVar(0, w_top, "w")
    objective.append(w)
    for s in range(base.size):
        v_top = math.ceil(ticks * lam * p[s] * span) + 1
        v = model.NewIntVar(0, v_top, f"v{s}")
        objective.append(v)
        # K v_s + ceil(K p_s) w >= K ticks lam p_s (L_s - ell_lo), relaxed. The
        # weight on w is resolved to 1/K, so K is as large as the row allows.
        unit = ticks * lam * p[s]
        size = (
            unit * (abs(float(base[s]) - ell_lo) + float(np.abs(t[s]).sum()))
            + (p[s] + 1.0) * w_top + v_top
        )
        k = 2.0 ** min(_ROW_BITS, math.floor(math.log2(_ROW_LIMIT / max(size, 1.0))))
        if k < 1.0:
            raise ScaleOverflow(
                f"a CVaR row of {instance.instance_id} does not fit in 2**60 "
                "even unscaled"
            )
        rhs = math.floor(k * unit * (float(base[s]) - ell_lo))
        rows = [math.ceil(k * unit * float(a)) for a in t[s]]
        w_coef = math.ceil(k * p[s])
        _check([*rows, rhs, w_coef * w_top, int(k) * v_top], "a CVaR row")
        model.Add(
            int(k) * v + w_coef * w
            >= rhs - sum(rows[c] * y[c] for c in range(columns.n))
        )
    return lam * ell_lo


def _hint(instance: Instance, columns) -> dict[int, int]:
    return _hint_from(columns, instance.witness)


def _hint_from(columns, schedule: Schedule) -> dict[int, int]:
    """The binaries that select ``schedule``: CP-SAT starts from a feasible point."""
    lots = schedule.as_array()
    return {
        c: int(lots[int(columns.asset[c]), int(columns.bucket[c])] == int(columns.lots[c]))
        for c in range(columns.n)
    }


class CpSat:
    """``cpsat`` behind the ``Solver`` protocol."""

    name = NAME

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def solve(self, instance: Instance, budget: Budget) -> Result:
        return cpsat(instance, budget=budget, seed=self.seed)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/solvers/test_cpsat.py -q`
Expected: 24 passed (the Hypothesis property draws 40 examples).

- [ ] **Step 6: Add the import contracts**

In `pyproject.toml`, add `"quantic.solvers.classical.cpsat",` to the `source_modules` of the contract named "The exact baseline does not read the encoder", and append:

```toml
[[tool.importlinter.contracts]]
name = "The three exact methods are independent (M4 design section 8)"
type = "independence"
modules = [
    "quantic.solvers.classical.brute_force",
    "quantic.solvers.classical.exact_pwl_milp",
    "quantic.solvers.classical.cpsat",
]
```

Run: `.venv/Scripts/lint-imports`
Expected: `Contracts: 4 kept, 0 broken.`

- [ ] **Step 7: Check it against the MILP on the ladder (not a test; the gate does this in Task 12)**

Write to the scratchpad and run with `.venv/Scripts/python <file>`:

```python
from quantic.solvers.base import Budget
from quantic.solvers.classical.cpsat import cpsat
from quantic.solvers.classical.exact_pwl_milp import exact_pwl_milp
from tests.ladder import tier

for name in ("T0", "T1"):
    for inst in tier(name):
        ref = exact_pwl_milp(inst).bracket
        got = cpsat(inst, budget=Budget(wall_clock_s=20.0))
        b = got.bracket
        apart = b.lower > ref.upper * (1 + 1e-9) + 1e-9 or ref.lower > b.upper * (1 + 1e-9) + 1e-9
        print(inst.instance_id, got.status, b.width / max(1.0, abs(b.upper)), "APART" if apart else "")
```

Expected: no `APART`; every instance `optimal` with relative width at most about `2.2e-7`, except `T1-D1+D3+D4-2026`, which stops at `time_limit` — HiGHS solves it in 0.3 s and CP-SAT does not in 20 s. That is a finding for the result doc.

- [ ] **Step 8: Lint and commit**

```
feat(solvers): CP-SAT, a third exact method with a bound proven through integer rounding
```

```bash
git add pyproject.toml uv.lock src/quantic/solvers/classical/cpsat.py tests/solvers/test_cpsat.py
git commit -F <message file>
```

---

### Task 5: Almgren–Chriss, the desk reference

Spec §4.1.

**Files:**
- Create: `src/quantic/solvers/classical/almgren_chriss.py`
- Modify: `pyproject.toml`
- Test: `tests/solvers/test_almgren_chriss.py`

**Interfaces:**
- Consumes: `lot_grid`, `scoring.score`, `micro.impact.almgren_chriss.AlmgrenChriss`, `HEURISTIC`.
- Produces: `almgren_chriss(instance) -> Result` (`solver="almgren_chriss"`, status always `HEURISTIC`, no bracket); `AlmgrenChrissSolver()` with `.name == "almgren_chriss"`; `holdings_fraction(n_buckets, kappa) -> ndarray`; `kappa_for(instance, i) -> float`; `nearest_row(values, position, target) -> tuple[int, ...] | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/solvers/test_almgren_chriss.py`:

```python
import dataclasses

import numpy as np
import pytest

from quantic.micro.impact.almgren_chriss import AlmgrenChriss as LinearImpact
from quantic.problem.risk import VarianceRisk
from quantic.solvers.base import HEURISTIC, Budget, Solver
from quantic.solvers.classical.almgren_chriss import (
    AlmgrenChrissSolver,
    almgren_chriss,
    holdings_fraction,
    kappa_for,
    nearest_row,
)
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.scoring import score
from tests.ladder import tier
from tests.solvers.instances import EVERY_SOLVABLE, cvar, discrete, plain


def test_the_linear_arm_costs_a_constant_times_the_square_of_the_trade():
    inst = plain()
    a = inst.params.assets[0]
    model = LinearImpact(eta=a.y_coef, gamma=a.gamma)
    p = inst.params.impact_params(0)
    cost = [model.temporary_cost(q, p) * q * a.price for q in (100.0, 300.0)]
    assert cost[1] / cost[0] == pytest.approx(9.0, rel=1e-12)


def test_the_trajectory_minimises_the_continuous_problem():
    """Impact k q**2 plus lam sigma**2 h**2 per bucket: a quadratic, solved exactly."""
    inst = plain()
    a = inst.params.assets[0]
    k = LinearImpact(eta=a.y_coef, gamma=a.gamma).temporary_cost(
        1.0, inst.params.impact_params(0)
    ) * a.price
    sigma2 = float(np.asarray(inst.params.price_covariance())[0, 0])
    # Chosen so that kappa is 0.5: urgent enough to bend the line, not to empty it.
    lam = 2.0 * k * (np.cosh(0.5) - 1.0) / sigma2
    inst = dataclasses.replace(inst, risk=VarianceRisk(lam=lam))
    n, position = inst.n_buckets, 400.0

    # d/dh_t of sum k (h_{t-1} - h_t)**2 + lam sigma2 h_t**2 = 0 for t = 1 .. n-1.
    m = n - 1
    system = np.zeros((m, m))
    rhs = np.zeros(m)
    for t in range(m):
        system[t, t] = 2.0 * k + lam * sigma2
        if t > 0:
            system[t, t - 1] = -k
        if t < m - 1:
            system[t, t + 1] = -k
    rhs[0] = k * position
    exact = np.linalg.solve(system, rhs)

    closed = holdings_fraction(n, kappa_for(inst, 0))[1:-1] * position
    assert kappa_for(inst, 0) == pytest.approx(0.5, rel=1e-12)
    assert closed == pytest.approx(exact, rel=1e-12)


def test_without_variance_risk_the_trajectory_is_a_straight_line():
    assert kappa_for(cvar(), 0) == 0.0
    assert holdings_fraction(4, 0.0) == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])


@pytest.mark.parametrize("seed", range(5))
def test_the_nearest_row_is_the_nearest_of_every_row(seed):
    rng = np.random.default_rng(seed)
    inst = discrete()
    grid = lot_grid(inst)
    for i in range(inst.n_assets):
        target = rng.uniform(0, 3, size=inst.n_buckets)
        rows = grid.rows(i)
        def distance(row, target=target):
            return sum(abs(v - x) for v, x in zip(row, target, strict=True)), row

        by_hand = min(rows, key=distance)
        assert nearest_row(grid.values[i], grid.initial_lots[i], target) == by_hand


def test_a_position_no_row_reaches_has_no_nearest_row():
    assert nearest_row(((0, 1), (0, 1)), 3, np.zeros(2)) is None


@pytest.mark.parametrize("build", EVERY_SOLVABLE, ids=lambda b: b.__name__)
def test_the_result_is_the_evaluators_and_never_a_proof(build):
    inst = build()
    result = almgren_chriss(inst)
    assert result.status == HEURISTIC
    assert result.bracket is None
    assert result.objective == score(inst, result.schedule)[0]
    assert result.schedule.as_array().sum(axis=1).tolist() == list(inst.initial_lots)


def test_a_breached_cardinality_is_reported_not_repaired():
    """Rows are chosen name by name; somewhere on the ladder that breaches D2."""
    breached = [
        r for inst in tier("T1") if inst.dials.discrete_participation
        for r in [almgren_chriss(inst)] if not r.feasibility.feasible
    ]
    for r in breached:
        assert set(r.feasibility.violations) >= {"cardinality"}
        assert r.status == HEURISTIC


def test_it_is_a_solver():
    assert isinstance(AlmgrenChrissSolver(), Solver)
    assert AlmgrenChrissSolver().solve(plain(), Budget()).status == HEURISTIC


def test_rounding_is_deterministic():
    insts = [i for i in tier("T0")]
    first = [almgren_chriss(i).schedule for i in insts]
    again = [almgren_chriss(i).schedule for i in insts]
    assert first == again
    assert all(s is not None for s in first)
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/solvers/test_almgren_chriss.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.solvers.classical.almgren_chriss'`.

- [ ] **Step 3: Implement**

Create `src/quantic/solvers/classical/almgren_chriss.py`:

```python
"""What a desk does today -- parent spec section 7.2's reference.

The Almgren-Chriss trajectory solves the continuous problem one name at a
time: linear temporary impact, variance risk on that name's own holdings, no
lots, no venue minimum, no cardinality, no blocks. It is then rounded onto the
lot grid. Any result the study calls interesting has to beat this.

Holdings after bucket ``t`` follow ``X sinh(kappa (T - t)) / sinh(kappa T)``,
with ``cosh kappa = 1 + lam sigma**2 / (2 k)``: ``k`` is the temporary cost per
share squared of the linear arm, ``sigma**2`` the name's own price variance per
bucket. The formula knows variance only. Under CVaR it is given no risk
aversion and the trajectory is a straight line, which is what a desk without a
CVaR model would trade.

Rounding takes, per name, the admissible row nearest the trajectory in the sum
of absolute lot differences -- a small dynamic programme over the lot grid.
Rows are chosen name by name, so a cardinality limit can be breached; the
result then says so through its feasibility report. Nothing here is a proof,
so the status is always ``HEURISTIC``.
"""

from __future__ import annotations

import math
import time

import numpy as np

from quantic.micro.impact.almgren_chriss import AlmgrenChriss as LinearImpact
from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.base import HEURISTIC, Budget, Result
from quantic.solvers.classical.lot_grid import lot_grid
from quantic.solvers.classical.scoring import score

NAME = "almgren_chriss"


def holdings_fraction(n_buckets: int, kappa: float) -> np.ndarray:
    """``h_t / X`` for ``t = 0 .. T``: one at the start, zero at the end."""
    t = np.arange(n_buckets + 1, dtype=float)
    if kappa == 0.0:
        return (n_buckets - t) / n_buckets
    return np.sinh(kappa * (n_buckets - t)) / math.sinh(kappa * n_buckets)


def kappa_for(instance: Instance, i: int) -> float:
    """The urgency of name ``i``'s trajectory. Zero without variance risk."""
    risk = instance.risk
    if not isinstance(risk, VarianceRisk) or risk.lam == 0.0:
        return 0.0
    a = instance.params.assets[i]
    k = LinearImpact(eta=a.y_coef, gamma=a.gamma).temporary_cost(
        1.0, instance.params.impact_params(i)
    ) * a.price
    sigma2 = float(np.asarray(instance.params.price_covariance())[i, i])
    return math.acosh(1.0 + float(risk.lam) * sigma2 / (2.0 * k))


def nearest_row(
    values: tuple[tuple[int, ...], ...], position: int, target: np.ndarray
) -> tuple[int, ...] | None:
    """The row of admissible values summing to ``position`` nearest ``target``.

    Nearest in ``sum |v_t - target_t|``; ties go to the row that is
    lexicographically smallest, so the answer does not depend on float order.
    ``None`` when no row reaches the position.
    """
    n = len(values)
    # best[t][r]: (cost, row) for buckets t.. selling exactly r lots.
    best: list[dict[int, tuple[float, tuple[int, ...]]]] = [{} for _ in range(n + 1)]
    best[n][0] = (0.0, ())
    for t in range(n - 1, -1, -1):
        for r in range(position + 1):
            options = []
            for v in values[t]:
                if v <= r and (r - v) in best[t + 1]:
                    cost, rest = best[t + 1][r - v]
                    options.append((cost + abs(v - float(target[t])), (v, *rest)))
            if options:
                best[t][r] = min(options)
    found = best[0].get(position)
    return None if found is None else found[1]


def almgren_chriss(instance: Instance) -> Result:
    """The rounded Almgren-Chriss schedule, scored by the evaluator."""
    started = time.perf_counter()
    grid = lot_grid(instance)
    rows = []
    kappas = []
    for i in range(instance.n_assets):
        kappa = kappa_for(instance, i)
        kappas.append(kappa)
        held = holdings_fraction(instance.n_buckets, kappa) * instance.initial_lots[i]
        row = nearest_row(grid.values[i], grid.initial_lots[i], -np.diff(held))
        if row is None:
            return Result(
                solver=NAME, status=HEURISTIC, schedule=None, zeta=None,
                objective=None, bracket=None, feasibility=None,
                timings={"total_s": time.perf_counter() - started},
                detail={"kappa": kappas, "unreachable_asset": i},
            )
        rows.append(row)

    schedule = Schedule(lots=tuple(rows))
    breakdown, zeta = score(instance, schedule)
    return Result(
        solver=NAME, status=HEURISTIC, schedule=schedule, zeta=zeta,
        objective=breakdown, bracket=None,
        feasibility=classify(instance, schedule),
        timings={"total_s": time.perf_counter() - started},
        detail={"kappa": kappas},
    )


class AlmgrenChrissSolver:
    """``almgren_chriss`` behind the ``Solver`` protocol. The budget is not used."""

    name = NAME

    def solve(self, instance: Instance, budget: Budget) -> Result:
        return almgren_chriss(instance)
```

- [ ] **Step 4: Run the tests and the contract**

Add `"quantic.solvers.classical.almgren_chriss",` to the `source_modules` of the contract named "The exact baseline does not read the encoder".
Run: `.venv/Scripts/python -m pytest tests/solvers/test_almgren_chriss.py -q` and `.venv/Scripts/lint-imports`
Expected: 21 passed; `Contracts: 4 kept, 0 broken.`

On the ladder, every one of the 16 D2 instances at T0–T1 breaches cardinality; the others are feasible. `test_a_breached_cardinality_is_reported_not_repaired` holds that the breach is reported and not repaired.

- [ ] **Step 5: Lint and commit**

```
feat(solvers): Almgren-Chriss, the desk reference, rounded to the nearest admissible row
```

```bash
git add pyproject.toml src/quantic/solvers/classical/almgren_chriss.py tests/solvers/test_almgren_chriss.py
git commit -F <message file>
```

---

### Task 6: The QUBO samplers

Spec §4.2 and §4.3.

**Files:**
- Create: `src/quantic/solvers/qubo/__init__.py`, `sampler.py`, `simulated_annealing.py`, `tabu.py`, `greedy.py`
- Test: `tests/solvers/test_qubo_samplers.py`

**Interfaces:**
- Consumes: `encoding.encode.{encode, EncodedInstance}`, `encoding.decode.{decode, FEASIBLE}`, `encoding.integer.ENCODINGS`, `encoding.qubo.Qubo`, `scoring.score`, `classify`, `HEURISTIC`.
- Produces:
  - `run_sampler(name, enc, budget, seed, draw) -> Result`, where `draw(bqm, reads, seed) -> (bits (m, n) int8 in registry order, energies (m,))`. Raises `ValueError` matching `"sample cap"` when the budget has neither a wall clock nor `max_samples`.
  - `QuboSampler(encoding: str, *, seed=0, zeta_bits=STUDY_ZETA_BITS)` with `.solve(instance, budget)` and `.solve_encoded(enc, budget)`; subclasses `SimulatedAnnealing` (`name="simulated_annealing"`), `Tabu` (`"tabu"`), `Greedy` (`"greedy"`).
  - `Result.detail` keys: `encoding`, `n_vars`, `zeta_bits`, `n_samples`, `n_batches`, `best_energy`, `energy_mismatch`, `samples` (tuple of `Sample(digest, count, energy, classification, total)`), `trace` (tuple of `(seconds, best_energy)`).
  - `to_bqm(qubo)`, `energies(qubo, bits)`, `encoding_named(name)`, `in_variable_order(sampleset, n)`, `batch_seed(seed, batch)`, `STUDY_ZETA_BITS = 8`, `MAX_BATCH = 64`.

- [ ] **Step 1: Add the dependencies**

Run: `uv add dwave-samplers dimod`
Check: `.venv/Scripts/python -c "from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler; import dimod"`

- [ ] **Step 2: Write the failing tests**

Create `tests/solvers/test_qubo_samplers.py`:

```python
import numpy as np
import pytest

from quantic.encoding.decode import FEASIBLE
from quantic.encoding.encode import encode
from quantic.encoding.integer import BinaryEncoding, OneHotEncoding
from quantic.encoding.qubo import Qubo
from quantic.encoding.registry import VarRegistry
from quantic.solvers.base import HEURISTIC, Budget, Solver
from quantic.solvers.classical.qubo_minimum import qubo_minimum
from quantic.solvers.classical.scoring import score
from quantic.solvers.qubo.greedy import Greedy
from quantic.solvers.qubo.sampler import energies, to_bqm
from quantic.solvers.qubo.simulated_annealing import SimulatedAnnealing
from quantic.solvers.qubo.tabu import Tabu
from tests.solvers.instances import cvar, discrete, plain

SAMPLERS = (SimulatedAnnealing, Tabu, Greedy)


def test_the_bqm_has_every_variable_and_the_same_energy():
    registry = VarRegistry(keys=tuple(("x", i) for i in range(4)))
    q = np.zeros((4, 4))
    q[0, 0], q[0, 2], q[2, 2] = 1.5, -2.0, 0.25          # bits 1 and 3 appear nowhere
    qubo = Qubo(Q=q, constant=3.0, registry=registry)
    bqm = to_bqm(qubo)
    assert bqm.num_variables == 4
    for bits in np.ndindex(2, 2, 2, 2):
        assert bqm.energy(dict(enumerate(bits))) == pytest.approx(qubo.energy(bits))
    every = np.array(list(np.ndindex(2, 2, 2, 2)))
    assert energies(qubo, every) == pytest.approx([qubo.energy(b) for b in every])


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda s: s.name)
@pytest.mark.parametrize("build", [plain, discrete, cvar], ids=lambda b: b.__name__)
def test_every_sample_is_counted_judged_and_scored_by_the_evaluator(sampler, build):
    inst = build()
    result = sampler("binary", seed=1).solve(inst, Budget(max_samples=20))
    detail = result.detail
    assert result.status == HEURISTIC
    assert detail["n_samples"] == 20
    assert sum(s.count for s in detail["samples"]) == 20
    assert detail["energy_mismatch"] <= 1e-9
    for s in detail["samples"]:
        assert (s.total is not None) == (s.classification == FEASIBLE)
    if result.schedule is not None:
        assert result.feasibility.feasible
        assert result.objective == score(inst, result.schedule)[0]
        assert result.objective.total == min(
            s.total for s in detail["samples"] if s.total is not None
        )


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda s: s.name)
def test_the_same_seed_draws_the_same_samples_and_a_shorter_run_is_a_prefix(sampler):
    inst = discrete()
    short = sampler("one_hot", seed=4).solve(inst, Budget(max_samples=7))
    again = sampler("one_hot", seed=4).solve(inst, Budget(max_samples=7))
    longer = sampler("one_hot", seed=4).solve(inst, Budget(max_samples=15))
    assert short.detail["samples"] == again.detail["samples"]
    assert {s.digest for s in short.detail["samples"]} <= {
        s.digest for s in longer.detail["samples"]
    }


def test_the_trace_only_ever_goes_down():
    result = SimulatedAnnealing("binary", seed=2).solve(cvar(), Budget(max_samples=31))
    trace = result.detail["trace"]
    assert trace
    assert all(b[1] < a[1] for a, b in zip(trace, trace[1:], strict=False))
    assert trace[-1][1] == result.detail["best_energy"]


def test_annealing_finds_the_qubos_minimum_on_a_small_instance():
    enc = encode(plain(), BinaryEncoding(), n_zeta_bits=8)
    found = SimulatedAnnealing("binary", seed=0).solve_encoded(enc, Budget(max_samples=63))
    lowest = qubo_minimum(enc).energy
    assert found.detail["best_energy"] == pytest.approx(lowest, rel=1e-9)


def test_a_wall_clock_run_draws_at_least_one_batch():
    result = Greedy("binary", seed=0).solve(plain(), Budget(wall_clock_s=1e-6))
    assert result.detail["n_batches"] == 1
    assert result.detail["n_samples"] == 1


def test_a_sampler_with_no_stopping_rule_is_refused():
    with pytest.raises(ValueError, match="sample cap"):
        Greedy("binary").solve(plain(), Budget())


def test_an_instance_encoded_another_way_is_refused():
    enc = encode(plain(), OneHotEncoding())
    with pytest.raises(ValueError, match="one_hot"):
        Greedy("binary").solve_encoded(enc, Budget(max_samples=1))


def test_an_unknown_encoding_is_refused():
    with pytest.raises(KeyError, match="gray"):
        Greedy("gray")


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda s: s.name)
def test_each_is_a_solver(sampler):
    assert isinstance(sampler("binary"), Solver)
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/solvers/test_qubo_samplers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.solvers.qubo'`.

- [ ] **Step 4: Implement the sampling loop**

Create `src/quantic/solvers/qubo/__init__.py`:

```python
"""Samplers that run on the encoder's QUBO -- M4 design section 4.2."""
```

Create `src/quantic/solvers/qubo/sampler.py`:

```python
"""Running a sampler on a QUBO, and judging every sample it returns.

Every sampler in this package -- annealing, tabu, steepest descent, and at M5
the quantum ones -- runs on the same ``Qubo`` object and goes through this
loop, so a difference between two of them is the sampler's and not the
bookkeeping's.

Samples are drawn in batches of 1, 2, 4, ... up to ``MAX_BATCH`` reads, each
batch seeded from ``(seed, batch index)``. The stream of samples is therefore
fixed by the seed alone, and a run stopped by the clock returns a prefix of
the stream a longer run returns. A run always draws at least one batch.

Each distinct bitstring is decoded once and classified by
``encoding.decode`` -- never repaired, parent spec section 6.5. A feasible one
is scored by the evaluator at its own minimising ``zeta``, which is the true
objective; the ``zeta`` bits the sampler chose are the encoding's business,
and what they cost is the regret measured against the QUBO's minimum.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

import dimod
import numpy as np

from quantic.encoding.decode import FEASIBLE, decode
from quantic.encoding.encode import EncodedInstance, encode
from quantic.encoding.integer import ENCODINGS, IntegerEncoding
from quantic.encoding.qubo import Qubo
from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.schedule import Schedule
from quantic.solvers.base import HEURISTIC, Budget, Result
from quantic.solvers.classical.scoring import score

# The CVaR gadget's zeta width for the study (M3b gate result).
STUDY_ZETA_BITS = 8
MAX_BATCH = 64

# (bqm, reads, seed) -> (bits in variable order 0..n-1, the sampler's energies)
Draw = Callable[[dimod.BinaryQuadraticModel, int, int], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Sample:
    """One distinct bitstring a run returned, and how often."""

    digest: str
    count: int
    energy: float                 # as the sampler reported it
    classification: str           # encoding.decode's verdict
    total: float | None           # the evaluator's total, when feasible


def encoding_named(name: str) -> IntegerEncoding:
    for encoding in ENCODINGS:
        if encoding.name == name:
            return encoding
    raise KeyError(f"unknown encoding {name!r}; known: {[e.name for e in ENCODINGS]}")


def to_bqm(qubo: Qubo) -> dimod.BinaryQuadraticModel:
    """Every variable is added, including any with no coefficient at all."""
    bqm = dimod.BinaryQuadraticModel("BINARY")
    q = qubo.Q
    bqm.add_variables_from((i, float(q[i, i])) for i in range(qubo.n_vars))
    rows, cols = np.nonzero(np.triu(q, 1))
    bqm.add_interactions_from(
        (int(i), int(j), float(q[i, j])) for i, j in zip(rows, cols, strict=True)
    )
    bqm.offset = float(qubo.constant)
    return bqm


def in_variable_order(sampleset: dimod.SampleSet, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A sampleset's bits as an ``(m, n)`` 0/1 array in registry order."""
    order = [sampleset.variables.index(v) for v in range(n)]
    record = sampleset.record
    counts = record.num_occurrences
    bits = np.repeat(record.sample[:, order].astype(np.int8), counts, axis=0)
    energies = np.repeat(record.energy.astype(float), counts)
    return bits, energies


def energies(qubo: Qubo, bits: np.ndarray) -> np.ndarray:
    """``z' Q z + constant`` for each row of ``bits``."""
    z = bits.astype(float)
    return ((z @ qubo.Q) * z).sum(axis=1) + qubo.constant


def batch_seed(seed: int, batch: int) -> int:
    """A seed below 2**31: ``dwave.samplers``' annealer refuses anything larger."""
    state = np.random.SeedSequence([int(seed), int(batch)]).generate_state(1)[0]
    return int(state) & 0x7FFF_FFFF


def run_sampler(
    name: str, enc: EncodedInstance, budget: Budget, seed: int, draw: Draw
) -> Result:
    """Draw batches until the budget is spent, then report every sample."""
    if budget.wall_clock_s is None and budget.max_samples is None:
        raise ValueError(
            f"{name} needs a wall clock or a sample cap: a sampler has no other "
            "way to know when to stop"
        )
    started = time.perf_counter()
    bqm = to_bqm(enc.qubo)
    seen: dict[bytes, list] = {}
    best: tuple[float, Schedule] | None = None
    best_energy = np.inf
    trace: list[tuple[float, float]] = []
    mismatch = 0.0
    drawn = 0
    batch = 0
    size = 1

    while True:
        if budget.max_samples is not None:
            size = min(size, budget.max_samples - drawn)
        bits, reported = draw(bqm, size, batch_seed(seed, batch))
        own = energies(enc.qubo, bits)
        scale = np.maximum(1.0, np.abs(own))
        mismatch = max(mismatch, float(np.max(np.abs(reported - own) / scale)))
        now = time.perf_counter() - started
        for row, energy in zip(bits, reported, strict=True):
            key = row.tobytes()
            if key in seen:
                seen[key][1] += 1
                continue
            decoded = decode(enc, row)
            total = None
            if decoded.classification == FEASIBLE:
                total = score(enc.instance, decoded.schedule)[0].total
                if best is None or total < best[0]:
                    best = (total, decoded.schedule)
            seen[key] = [hashlib.blake2b(key, digest_size=8).hexdigest(), 1,
                         float(energy), decoded.classification, total]
            if energy < best_energy:
                best_energy = float(energy)
                trace.append((now, best_energy))
        drawn += len(bits)
        batch += 1
        size = min(2 * size, MAX_BATCH)
        if budget.max_samples is not None and drawn >= budget.max_samples:
            break
        if (
            budget.wall_clock_s is not None
            and time.perf_counter() - started >= budget.wall_clock_s
        ):
            break

    samples = tuple(Sample(*fields) for fields in seen.values())
    detail = {
        "encoding": enc.layout.encoding_name,
        "n_vars": enc.n_vars,
        "zeta_bits": None if enc.gadget is None else enc.gadget.zeta.n_bits,
        "n_samples": drawn,
        "n_batches": batch,
        "best_energy": best_energy,
        "energy_mismatch": mismatch,
        "samples": samples,
        "trace": tuple(trace),
    }
    timings = {"total_s": time.perf_counter() - started}
    if best is None:
        return Result(
            solver=name, status=HEURISTIC, schedule=None, zeta=None, objective=None,
            bracket=None, feasibility=None, timings=timings, detail=detail,
        )
    total, schedule = best
    breakdown, zeta = score(enc.instance, schedule)
    return Result(
        solver=name, status=HEURISTIC, schedule=schedule, zeta=zeta,
        objective=breakdown, bracket=None,
        feasibility=classify(enc.instance, schedule),
        timings=timings, detail=detail,
    )


class QuboSampler:
    """A sampler behind the ``Solver`` protocol, for one encoding and one seed.

    ``solve`` encodes the instance; the bench encodes once per worker and calls
    ``solve_encoded`` instead.
    """

    name: str = ""

    def __init__(
        self, encoding: str, *, seed: int = 0, zeta_bits: int = STUDY_ZETA_BITS
    ) -> None:
        self.encoding = encoding_named(encoding)
        self.seed = seed
        self.zeta_bits = zeta_bits

    def draw(self, bqm: dimod.BinaryQuadraticModel, reads: int, seed: int):
        raise NotImplementedError

    def solve(self, instance: Instance, budget: Budget) -> Result:
        return self.solve_encoded(
            encode(instance, self.encoding, n_zeta_bits=self.zeta_bits), budget
        )

    def solve_encoded(self, enc: EncodedInstance, budget: Budget) -> Result:
        if enc.layout.encoding_name != self.encoding.name:
            raise ValueError(
                f"{self.name} was built for {self.encoding.name!r} and given an "
                f"instance encoded with {enc.layout.encoding_name!r}"
            )
        return run_sampler(self.name, enc, budget, self.seed, self.draw)
```

Two things the dry run found, both in the code above: `dwave.samplers`' annealer refuses a seed of `2**31` or more (its message says `2**32`), hence the mask in `batch_seed`; and a variable with no coefficient would be missing from a BQM built with `dimod.BinaryQuadraticModel.from_qubo`, hence `to_bqm` adds every variable explicitly.

- [ ] **Step 5: Implement the three samplers**

Create `src/quantic/solvers/qubo/simulated_annealing.py`:

```python
"""Simulated annealing on the QUBO (``dwave.samplers``, formerly ``neal``)."""

from __future__ import annotations

from dwave.samplers import SimulatedAnnealingSampler

from quantic.solvers.qubo.sampler import QuboSampler, in_variable_order

NAME = "simulated_annealing"


class SimulatedAnnealing(QuboSampler):
    name = NAME

    def draw(self, bqm, reads, seed):
        found = SimulatedAnnealingSampler().sample(bqm, num_reads=reads, seed=seed)
        return in_variable_order(found, bqm.num_variables)
```

Create `src/quantic/solvers/qubo/tabu.py`:

```python
"""Tabu search on the QUBO (``dwave.samplers``, formerly ``dwave-tabu``).

Tabu's own stopping rule is a wall clock per read, which would make a run's
samples depend on the machine's load. Each read is bounded by restarts
instead, with the timeout set out of reach; the bench's per-run kill is the
backstop.
"""

from __future__ import annotations

from dwave.samplers import TabuSampler

from quantic.solvers.qubo.sampler import QuboSampler, in_variable_order

NAME = "tabu"
RESTARTS_PER_READ = 10
_NEVER_MS = 10**9


class Tabu(QuboSampler):
    name = NAME

    def draw(self, bqm, reads, seed):
        found = TabuSampler().sample(
            bqm, num_reads=reads, seed=seed,
            num_restarts=RESTARTS_PER_READ, timeout=_NEVER_MS,
        )
        return in_variable_order(found, bqm.num_variables)
```

Create `src/quantic/solvers/qubo/greedy.py`:

```python
"""Steepest descent from uniformly random starts: the cheap floor."""

from __future__ import annotations

from dwave.samplers import SteepestDescentSampler

from quantic.solvers.qubo.sampler import QuboSampler, in_variable_order

NAME = "greedy"


class Greedy(QuboSampler):
    name = NAME

    def draw(self, bqm, reads, seed):
        found = SteepestDescentSampler().sample(bqm, num_reads=reads, seed=seed)
        return in_variable_order(found, bqm.num_variables)
```

- [ ] **Step 6: Run the tests and the contracts**

Run: `.venv/Scripts/python -m pytest tests/solvers/test_qubo_samplers.py -q` and `.venv/Scripts/lint-imports`
Expected: 22 passed; `Contracts: 4 kept, 0 broken.`

- [ ] **Step 7: Lint and commit**

```
feat(solvers): annealing, tabu and steepest descent on the encoder's QUBO, one sampling loop
```

```bash
git add pyproject.toml uv.lock src/quantic/solvers/qubo tests/solvers/test_qubo_samplers.py
git commit -F <message file>
```

---

### Task 7: Instances, experiments and keys

Spec §5.1 and §5.2.

**Files:**
- Create: `src/quantic/bench/instances.py`, `src/quantic/bench/config.py`, `src/quantic/bench/keys.py`
- Modify: `tests/ladder.py`
- Create: `tests/bench/__init__.py` (empty)
- Test: `tests/bench/test_config_and_keys.py`

**Interfaces:**
- Consumes: `problem.generator.generate`, `problem.scenarios.generate_scenarios`, `problem.liquidation.by_name`, `problem.dials.Dials`.
- Produces:
  - `instances`: `GATE_SEED = 2026`, `N_RAW = 300`, `N_TARGET = 15`, `synthetic_params(tier) -> MarketParams`, `tier_instances(tier_name, seed) -> tuple[Instance, ...]` (16, in `Dials.combinations(include_cvar=True)` order), `InstanceCache().get(tier, dials_label, seed) -> Instance`.
  - `config`: `load(path) -> Experiment(name, tiers, dials, seeds, qubo_minimum_tiers, solvers)`, `SolverPlan(name, encodings, zeta_bits, seeds, budgets_s, max_samples)`, `ConfigError(ValueError)`, `SOLVERS`, `QUBO_SOLVERS`.
  - `keys`: `Task(kind, tier, dials, instance_seed, solver, encoding=None, zeta_bits=None, solver_seed=None, budget_s=None, max_samples=None)` with `.key(code) -> str`; `CodeVersion(commit, lock_hash, dirty)`; `code_version(root) -> CodeVersion`; `expand(experiment) -> list[Task]` (references first); `RUN`, `REFERENCE`, `REF_BRACKET = "ref:exact_pwl_milp"`, `REF_BRUTE_FORCE = "ref:brute_force"`, `REF_QUBO_MINIMUM = "ref:qubo_minimum"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/bench/__init__.py` empty, and `tests/bench/test_config_and_keys.py`:

```python
import multiprocessing as mp
from pathlib import Path

import pytest

from quantic.bench.config import ConfigError, load
from quantic.bench.instances import GATE_SEED, InstanceCache, tier_instances
from quantic.bench.keys import (
    REF_BRACKET,
    REF_BRUTE_FORCE,
    REF_QUBO_MINIMUM,
    REFERENCE,
    RUN,
    CodeVersion,
    Task,
    code_version,
    expand,
)
from tests.ladder import gate_ladder

ROOT = Path(__file__).resolve().parents[2]
CODE = CodeVersion(commit="c0ffee", lock_hash="10c4", dirty=False)

GOOD = """
[instances]
tiers = ["T0", "T1"]
dials = ["none", "D1+D3"]
seeds = [2026, 2027]

[references]
qubo_minimum_tiers = ["T0"]

[solvers.cpsat]
seeds = [0, 1]
budgets_s = [1, 10]

[solvers.tabu]
encodings = ["binary", "domain_wall"]
seeds = "0..2"
budgets_s = [0.5]
"""


def _load(tmp_path: Path, text: str):
    path = tmp_path / "exp.toml"
    path.write_text(text, encoding="utf-8")
    return load(path)


def test_a_config_reads_into_an_experiment(tmp_path):
    exp = _load(tmp_path, GOOD)
    assert exp.name == "exp"
    assert exp.tiers == ("T0", "T1")
    assert exp.seeds == (2026, 2027)
    tabu = next(p for p in exp.solvers if p.name == "tabu")
    assert tabu.seeds == (0, 1, 2)
    assert tabu.encodings == ("binary", "domain_wall")
    assert tabu.zeta_bits == 8
    cpsat = next(p for p in exp.solvers if p.name == "cpsat")
    assert cpsat.encodings == (None,)
    assert cpsat.zeta_bits is None


def test_all_dials_means_the_sixteen_combinations(tmp_path):
    exp = _load(tmp_path, GOOD.replace('dials = ["none", "D1+D3"]', 'dials = "all"'))
    assert len(exp.dials) == 16


@pytest.mark.parametrize(
    "edit, message",
    [
        (("seeds = [2026, 2027]", "seed = [2026]"), "unknown key"),
        (('budgets_s = [0.5]', 'budgets_s = [0]'), "positive"),
        (('encodings = ["binary", "domain_wall"]\n', ""), "encodings"),
        (("[solvers.cpsat]", "[solvers.cplex]"), "unknown solver"),
        (('dials = ["none", "D1+D3"]', 'dials = ["D5"]'), "dials"),
        (('seeds = "0..2"', 'seeds = "0-2"'), "a..b"),
    ],
)
def test_a_config_that_does_not_say_what_it_means_is_refused(tmp_path, edit, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, GOOD.replace(*edit))


def test_a_config_with_no_solver_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no solver"):
        _load(tmp_path, "[instances]\ntiers = ['T0']\n")


def test_expansion_puts_references_first_and_covers_the_grid(tmp_path):
    tasks = expand(_load(tmp_path, GOOD))
    kinds = [t.kind for t in tasks]
    assert kinds == sorted(kinds, key=lambda k: k != REFERENCE)
    refs = [t for t in tasks if t.kind == REFERENCE]
    runs = [t for t in tasks if t.kind == RUN]
    n_instances = 2 * 2 * 2
    assert sum(t.solver == REF_BRACKET for t in refs) == n_instances
    assert sum(t.solver == REF_BRUTE_FORCE for t in refs) == n_instances // 2  # T0 only
    assert {(t.tier, t.encoding) for t in refs if t.solver == REF_QUBO_MINIMUM} == {
        ("T0", "binary"), ("T0", "domain_wall"),
    }
    assert len(runs) == n_instances * (2 * 2 + 2 * 3 * 1)
    assert len({t.key(CODE) for t in tasks}) == len(tasks)


def test_a_key_changes_with_the_code_and_with_any_coordinate():
    task = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="cpsat",
                solver_seed=0, budget_s=1.0)
    other_code = CodeVersion(commit="beef", lock_hash="10c4", dirty=False)
    assert task.key(CODE) != task.key(other_code)
    assert task.key(CODE) != Task(**{**task.__dict__, "budget_s": 2.0}).key(CODE)
    assert task.key(CODE) == Task(**task.__dict__).key(CODE)


def _key_in_child(queue):
    task = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="cpsat")
    queue.put(task.key(CODE))


def test_a_key_is_the_same_in_another_process():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=_key_in_child, args=(queue,))
    child.start()
    child.join()
    task = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="cpsat")
    assert queue.get(timeout=10) == task.key(CODE)


def test_the_code_version_names_this_checkout():
    code = code_version(ROOT)
    assert len(code.commit) == 40
    assert len(code.lock_hash) == 64


def test_the_gate_seed_gives_the_m3_ladder_exactly():
    for tier in ("T0", "T1"):
        mine = tier_instances(tier, GATE_SEED)
        theirs = tuple(i for i in gate_ladder() if i.tier == tier)
        assert [i.content_hash for i in mine] == [i.content_hash for i in theirs]


def test_the_cache_builds_a_tier_once_and_refuses_an_unknown_label():
    cache = InstanceCache()
    first = cache.get("T0", "none", GATE_SEED)
    assert cache.get("T0", "none", GATE_SEED) is first
    with pytest.raises(KeyError, match="D9"):
        cache.get("T0", "D9", GATE_SEED)
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/bench/test_config_and_keys.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.bench.config'`.

- [ ] **Step 3: Implement**

Create `src/quantic/bench/instances.py`:

```python
"""The instances the bench runs on.

The same generator, parameters and scenario sizes as the M3 gates' ladder
(``tests/ladder.py`` imports from here), so that seed ``GATE_SEED`` gives
exactly the instances M3a and M3b were proven on. Other seeds give new draws
of the same ladder.

The parameters are synthetic. Calibrating them from a real bundle is
``problem.from_bundle``'s job, and M4 does not do it: the study's question is
how solvers behave across the hardness dials, which the synthetic ladder spans.
"""

from __future__ import annotations

import numpy as np

from quantic.problem.dials import Dials
from quantic.problem.generator import generate
from quantic.problem.instance import Instance
from quantic.problem.liquidation import TierSpec, by_name
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.scenarios import generate_scenarios

GATE_SEED = 2026
BUCKET_NS = 1800 * 1_000_000_000
N_RAW = 300
N_TARGET = 15


def synthetic_params(tier: TierSpec) -> MarketParams:
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


def tier_instances(tier_name: str, seed: int) -> tuple[Instance, ...]:
    """All sixteen dial combinations of one tier at one seed.

    One raw scenario set per tier and seed, shared by the CVaR instances, as
    ``generate_ladder`` does.
    """
    tier = by_name(tier_name)
    params = synthetic_params(tier)
    raw = generate_scenarios(params, tier.n_buckets, seed=seed, n_raw=N_RAW)
    return tuple(
        generate(
            tier, dials, params, seed=seed, n_raw=N_RAW, n_target=N_TARGET,
            scenarios=raw if dials.cvar_risk else None,
        )
        for dials in Dials.combinations(include_cvar=True)
    )


class InstanceCache:
    """One worker's instances, built once per (tier, seed)."""

    def __init__(self) -> None:
        self._tiers: dict[tuple[str, int], dict[str, Instance]] = {}

    def get(self, tier: str, dials: str, seed: int) -> Instance:
        key = (tier, seed)
        if key not in self._tiers:
            self._tiers[key] = {i.dials.label: i for i in tier_instances(tier, seed)}
        try:
            return self._tiers[key][dials]
        except KeyError:
            raise KeyError(
                f"no dial combination {dials!r}; known: {sorted(self._tiers[key])}"
            ) from None
```

Create `src/quantic/bench/config.py`:

```python
"""An experiment, read from TOML and expanded into the runs it asks for.

    [instances]
    tiers = ["T0", "T1"]
    dials = "all"                 # or a list of labels such as ["none", "D1+D3"]
    seeds = [2026, 2027]

    [references]
    qubo_minimum_tiers = ["T0"]   # where the QUBO's minimum is enumerated

    [solvers.simulated_annealing]
    encodings = ["binary", "one_hot", "domain_wall"]
    zeta_bits = 8
    seeds = "0..49"               # inclusive range, or a list
    budgets_s = [0.1, 1, 10, 60]
    max_samples = 1000            # optional, applies to every budget

A solver name is one of ``SOLVERS`` or an import path ``"package.module:Class"``
whose class takes ``seed`` (and ``encoding`` and ``zeta_bits`` when the
section names encodings). Unknown keys are refused: a misspelt key silently
ignored is a study that did not run what its author thinks it ran.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from quantic.problem.dials import Dials

SOLVERS = (
    "almgren_chriss", "cpsat", "exact_pwl_milp",
    "simulated_annealing", "tabu", "greedy",
)
QUBO_SOLVERS = ("simulated_annealing", "tabu", "greedy")

_TOP = {"instances", "references", "solvers"}
_INSTANCES = {"tiers", "dials", "seeds"}
_REFERENCES = {"qubo_minimum_tiers"}
_SOLVER = {"encodings", "zeta_bits", "seeds", "budgets_s", "max_samples"}


class ConfigError(ValueError):
    """The experiment file does not say what it means."""


@dataclass(frozen=True)
class SolverPlan:
    name: str
    encodings: tuple[str | None, ...]
    zeta_bits: int | None
    seeds: tuple[int, ...]
    budgets_s: tuple[float, ...]
    max_samples: int | None


@dataclass(frozen=True)
class Experiment:
    name: str
    tiers: tuple[str, ...]
    dials: tuple[str, ...]
    seeds: tuple[int, ...]
    qubo_minimum_tiers: tuple[str, ...]
    solvers: tuple[SolverPlan, ...]


def _unknown(where: str, got: dict, allowed: set[str]) -> None:
    extra = sorted(set(got) - allowed)
    if extra:
        raise ConfigError(f"unknown key(s) {extra} in {where}; allowed: {sorted(allowed)}")


def _seeds(value, where: str) -> tuple[int, ...]:
    if isinstance(value, str):
        lo, sep, hi = value.partition("..")
        if not sep:
            raise ConfigError(f"{where}: a seed range is written 'a..b', got {value!r}")
        return tuple(range(int(lo), int(hi) + 1))
    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return tuple(value)
    raise ConfigError(f"{where}: seeds are a list of integers or 'a..b', got {value!r}")


def _is_qubo(name: str) -> bool:
    return name in QUBO_SOLVERS


def load(path: Path) -> Experiment:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    _unknown("the file", raw, _TOP)
    inst = raw.get("instances", {})
    _unknown("[instances]", inst, _INSTANCES)

    every = tuple(d.label for d in Dials.combinations(include_cvar=True))
    dials = inst.get("dials", "all")
    if dials == "all":
        dials = every
    elif not isinstance(dials, list) or not set(dials) <= set(every):
        raise ConfigError(f"[instances] dials must be 'all' or labels from {every}")

    refs = raw.get("references", {})
    _unknown("[references]", refs, _REFERENCES)

    plans = []
    for name, body in raw.get("solvers", {}).items():
        where = f"[solvers.{name}]"
        _unknown(where, body, _SOLVER)
        if name not in SOLVERS and ":" not in name:
            raise ConfigError(f"{where}: unknown solver; known: {SOLVERS}")
        qubo = _is_qubo(name) or "encodings" in body
        if qubo and not body.get("encodings"):
            raise ConfigError(f"{where}: a QUBO sampler needs its encodings")
        if not qubo and "encodings" in body:
            raise ConfigError(f"{where}: only QUBO samplers take encodings")
        budgets = tuple(float(b) for b in body.get("budgets_s", []))
        if not budgets or any(b <= 0 for b in budgets):
            raise ConfigError(f"{where}: budgets_s must list positive seconds")
        plans.append(SolverPlan(
            name=name,
            encodings=tuple(body["encodings"]) if qubo else (None,),
            zeta_bits=int(body.get("zeta_bits", 8)) if qubo else None,
            seeds=_seeds(body.get("seeds", [0]), where),
            budgets_s=budgets,
            max_samples=body.get("max_samples"),
        ))
    if not plans:
        raise ConfigError("the experiment names no solver")

    return Experiment(
        name=Path(path).stem,
        tiers=tuple(inst.get("tiers", ["T0"])),
        dials=tuple(dials),
        seeds=_seeds(inst.get("seeds", [2026]), "[instances]"),
        qubo_minimum_tiers=tuple(refs.get("qubo_minimum_tiers", [])),
        solvers=tuple(plans),
    )
```

Create `src/quantic/bench/keys.py`:

```python
"""What a run is, and the key that names it.

A key is a SHA-256 over the run's coordinates and the code that runs it: the
git commit and the hash of ``uv.lock``. The same key always means the same
computation, so a store can be resumed by skipping the keys it holds, and rows
from different code never share a key.

The instance enters through its coordinates -- tier, dial combination, seed --
which the commit pins to one instance; its content hash is written on the row.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from quantic.bench.config import Experiment

RUN = "run"
REFERENCE = "reference"

# Reference solvers: computed once per instance, never budgeted.
REF_BRACKET = "ref:exact_pwl_milp"
REF_BRUTE_FORCE = "ref:brute_force"
REF_QUBO_MINIMUM = "ref:qubo_minimum"


@dataclass(frozen=True)
class CodeVersion:
    commit: str
    lock_hash: str
    dirty: bool


@dataclass(frozen=True)
class Task:
    kind: str
    tier: str
    dials: str
    instance_seed: int
    solver: str
    encoding: str | None = None
    zeta_bits: int | None = None
    solver_seed: int | None = None
    budget_s: float | None = None
    max_samples: int | None = None

    def key(self, code: CodeVersion) -> str:
        payload = {**asdict(self), "commit": code.commit, "lock_hash": code.lock_hash}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_version(root: Path) -> CodeVersion:
    """The commit, the lock file's hash, and whether tracked files differ."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    lock = (Path(root) / "uv.lock").read_bytes()
    return CodeVersion(
        commit=git("rev-parse", "HEAD"),
        lock_hash=hashlib.sha256(lock).hexdigest(),
        dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
    )


def expand(experiment: Experiment) -> list[Task]:
    """Every task the experiment asks for: references first, then runs."""
    references: list[Task] = []
    runs: list[Task] = []
    qubo_encodings = sorted({
        (e, p.zeta_bits) for p in experiment.solvers for e in p.encodings if e is not None
    })
    for tier in experiment.tiers:
        for seed in experiment.seeds:
            for dials in experiment.dials:
                at = {"tier": tier, "dials": dials, "instance_seed": seed}
                references.append(Task(kind=REFERENCE, solver=REF_BRACKET, **at))
                if tier == "T0":
                    references.append(Task(kind=REFERENCE, solver=REF_BRUTE_FORCE, **at))
                if tier in experiment.qubo_minimum_tiers:
                    for encoding, bits in qubo_encodings:
                        references.append(Task(
                            kind=REFERENCE, solver=REF_QUBO_MINIMUM,
                            encoding=encoding, zeta_bits=bits, **at,
                        ))
                for plan in experiment.solvers:
                    for encoding in plan.encodings:
                        for solver_seed in plan.seeds:
                            for budget in plan.budgets_s:
                                runs.append(Task(
                                    kind=RUN, solver=plan.name, encoding=encoding,
                                    zeta_bits=plan.zeta_bits, solver_seed=solver_seed,
                                    budget_s=budget, max_samples=plan.max_samples, **at,
                                ))
    return references + runs
```

Replace `tests/ladder.py` so the gates and the bench share one copy of the parameters:

```python
"""The ladder both gates are run on.

M3a's gate and M3b's build the same sixty-four instances, from the same
parameters and the same seed. They are built here so that "the same" is a
fact about one function and not about two copies of it.
"""

from __future__ import annotations

import functools

from quantic.bench.instances import GATE_SEED, N_RAW, N_TARGET, synthetic_params
from quantic.problem.generator import generate_ladder
from quantic.problem.instance import Instance

SEED = GATE_SEED
SMALL = {"n_raw": N_RAW, "n_target": N_TARGET}
params_for = synthetic_params


@functools.cache
def gate_ladder() -> tuple[Instance, ...]:
    """All sixty-four instances. Built once per process: generation is pure."""
    return generate_ladder(params_for, seed=SEED, include_cvar=True, **SMALL)


def tier(name: str) -> tuple[Instance, ...]:
    return tuple(i for i in gate_ladder() if i.tier == name)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/bench/test_config_and_keys.py tests/solvers/test_lot_grid.py -q`
Expected: PASS. `test_the_gate_seed_gives_the_m3_ladder_exactly` is what proves the move of the parameters changed nothing.

- [ ] **Step 5: Lint, contracts, commit**

```
feat(bench): experiments from TOML, keyed tasks, and the M3 ladder as the bench's instances
```

```bash
git add src/quantic/bench/instances.py src/quantic/bench/config.py src/quantic/bench/keys.py tests/ladder.py tests/bench/__init__.py tests/bench/test_config_and_keys.py
git commit -F <message file>
```

---

### Task 8: The results store

Spec §6.

**Files:**
- Create: `src/quantic/bench/store.py`
- Create: `tests/bench/rows.py`
- Test: `tests/bench/test_store.py`

**Interfaces:**
- Consumes: `Task`, `CodeVersion` (Task 7).
- Produces: `Store(path, *, read_only=False)` (context manager) with `.done_keys() -> set[str]`, `.write(row, samples=(), trace=())`, `.runs() -> pl.DataFrame` (ordered by key), `.samples() -> pl.DataFrame`, `.con`; `blank_row(key, task, code, machine, packages) -> dict`; `RUN_COLUMNS`; `SCHEMA_VERSION = 1`; `OK`, `ERROR`, `KILLED`; `SchemaMismatch`, `StoreBusy`. `tests/bench/rows.py`: `CODE`, `row(task, **values)`, `ref(dials, lower, upper, tier="T0")`, `run_row(dials, solver, seed, total, *, budget=1.0, feasible=True, encoding=None, tier="T0", **extra)`, `write(path, rows)`.

- [ ] **Step 1: Add the dependency**

Run: `uv add duckdb`

- [ ] **Step 2: Write the failing tests**

Create `tests/bench/rows.py`:

```python
"""Hand-built store rows, for tests that need a store without running anything."""

from __future__ import annotations

from pathlib import Path

from quantic.bench.keys import REF_BRACKET, REFERENCE, RUN, CodeVersion, Task
from quantic.bench.store import OK, Store, blank_row

CODE = CodeVersion(commit="c0ffee", lock_hash="10c4", dirty=False)


def row(task: Task, **values) -> dict:
    out = blank_row(task.key(CODE), task, CODE, "here", "{}")
    out.update(store_status=OK, instance_hash=f"{task.tier}/{task.dials}", **values)
    return out


def ref(dials: str, lower: float, upper: float, tier: str = "T0") -> dict:
    task = Task(kind=REFERENCE, tier=tier, dials=dials, instance_seed=1, solver=REF_BRACKET)
    return row(task, status="optimal", lower=lower, upper=upper, total=upper)


def run_row(dials: str, solver: str, seed: int, total: float | None, *, budget=1.0,
            feasible=True, encoding=None, tier="T0", **extra) -> dict:
    task = Task(kind=RUN, tier=tier, dials=dials, instance_seed=1, solver=solver,
                encoding=encoding, solver_seed=seed, budget_s=budget)
    return row(task, status="heuristic", total=total, feasible=feasible,
               total_s=0.5, total_bps=total, **extra)


def write(path: Path, rows: list[dict]) -> Path:
    with Store(path) as s:
        for r in rows:
            s.write(r)
    return path
```

Create `tests/bench/test_store.py`:

```python
import multiprocessing as mp

import duckdb
import pytest

from quantic.bench.store import SCHEMA_VERSION, SchemaMismatch, Store, StoreBusy
from tests.bench.rows import run_row, write


def test_a_row_reads_back_as_written_with_its_samples_and_trace(tmp_path):
    row = run_row("none", "tabu", 0, 10.0, encoding="binary", n_samples=3,
                  violations={"cardinality": 0.0}, schedule=[[1, 2], [0, 3]])
    with Store(tmp_path / "s.duckdb") as s:
        s.write(row, [("ab12", 2, -1.5, "feasible", 10.0)], [(0.1, -1.0), (0.2, -1.5)])
        back = s.runs().row(0, named=True)
        assert s.samples()["count"].to_list() == [2]
        assert s.con.execute("SELECT count(*) FROM traces").fetchone() == (2,)
        assert s.done_keys() == {row["key"]}
    assert back["total"] == 10.0
    assert back["schedule"] == "[[1, 2], [0, 3]]"
    assert back["violations"] == '{"cardinality": 0.0}'


def test_a_row_with_a_missing_or_unknown_column_is_refused(tmp_path):
    row = run_row("none", "tabu", 0, 10.0)
    with Store(tmp_path / "s.duckdb") as s:
        with pytest.raises(ValueError, match="unknown"):
            s.write({**row, "colour": "blue"})
        assert s.runs().is_empty()


def test_a_failed_write_leaves_nothing_behind(tmp_path):
    row = run_row("none", "tabu", 0, 10.0)
    with Store(tmp_path / "s.duckdb") as s:
        s.write(row)
        with pytest.raises(duckdb.ConstraintException):
            s.write(row, [("cd34", 1, 0.0, "feasible", 1.0)])
        assert s.samples().is_empty()


def test_a_store_of_another_schema_version_is_refused(tmp_path):
    path = write(tmp_path / "s.duckdb", [])
    con = duckdb.connect(str(path))
    con.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'",
                [str(SCHEMA_VERSION + 1)])
    con.close()
    with pytest.raises(SchemaMismatch):
        Store(path)


def _hold_open(path, opened, release):
    with Store(path):
        opened.set()
        release.wait(30)


def test_a_store_another_process_is_writing_is_refused_with_a_reason(tmp_path):
    path = write(tmp_path / "s.duckdb", [])
    ctx = mp.get_context("spawn")
    opened, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_open, args=(path, opened, release))
    child.start()
    try:
        assert opened.wait(30)
        with pytest.raises(StoreBusy, match="bench run"):
            Store(path, read_only=True)
    finally:
        release.set()
        child.join()


def test_a_store_in_a_directory_that_does_not_exist_yet_is_created(tmp_path):
    path = tmp_path / "studies" / "m4" / "s.duckdb"
    with Store(path) as s:
        assert s.done_keys() == set()
    assert path.exists()
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/bench/test_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.bench.store'`.

- [ ] **Step 4: Implement**

Create `src/quantic/bench/store.py`:

```python
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
        """One finished task, all its rows, in one transaction."""
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
            if samples:
                self.con.executemany(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)",
                    [(row["key"], *s) for s in samples],
                )
            if trace:
                self.con.executemany(
                    "INSERT INTO traces VALUES (?, ?, ?)",
                    [(row["key"], *t) for t in trace],
                )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def runs(self) -> pl.DataFrame:
        return self.con.execute("SELECT * FROM runs ORDER BY key").pl()

    def samples(self) -> pl.DataFrame:
        return self.con.execute("SELECT * FROM samples ORDER BY key, digest").pl()
```

On Windows, a file another process holds fails with the operating system's sharing-violation text, in the system's language — the dry run saw it in French. That is why `StoreBusy` is decided by "the file exists and cannot be opened", not by the message.

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/bench/test_store.py -q`
Expected: 6 passed.

- [ ] **Step 6: Lint, contracts, commit**

```
feat(bench): the DuckDB results store, one transaction per task
```

```bash
git add pyproject.toml uv.lock src/quantic/bench/store.py tests/bench/rows.py tests/bench/test_store.py
git commit -F <message file>
```

---

### Task 9: Running tasks — the executor, the pool and the runner

Spec §5.3 and §5.4.

**Files:**
- Create: `src/quantic/bench/execute.py`, `src/quantic/bench/pool.py`, `src/quantic/bench/runner.py`
- Create: `tests/bench/stand_ins.py`
- Test: `tests/bench/test_runner.py`

**Interfaces:**
- Consumes: every solver of Tasks 4–6, `brute_force`, `exact_pwl_milp`, `qubo_minimum`, `InstanceCache`, `Task`, `CodeVersion`, `code_version`, `expand`, `Store`, `blank_row`.
- Produces:
  - `execute`: `Executor(code)` with `.run(key, task) -> (row: dict, samples: list[tuple], trace: list[tuple])`, never raising for a solver's or an instance's failure; `machine() -> str`; `packages() -> dict[str, str]`.
  - `pool`: `run_tasks(jobs, *, code, n_workers, deadline_s, on_done)` with `on_done(key, task, payload | None, killed: bool)` called in the main process; `WorkerStartError`.
  - `runner`: `plan(experiment, *, root, store=None, workers=None) -> Plan(n_tasks, n_done, n_todo, estimate_s, n_workers)`; `run(experiment, *, root, store, workers=None, allow_dirty=False, stop_after=None, kill_margin_s=30.0) -> Summary(n_written, n_killed, n_errors)`; `deadline_s(task, margin_s=30.0)`; `default_workers()`; `DirtyTree`; `KILL_FACTOR = 3.0`, `KILL_MARGIN_S = 30.0`, `REFERENCE_DEADLINE_S = 3600.0`.

- [ ] **Step 1: Write the failing tests**

Create `tests/bench/stand_ins.py`:

```python
"""Solvers that misbehave on purpose, loaded by import path in a worker."""

from __future__ import annotations

import time


class Hang:
    name = "hang"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def solve(self, instance, budget):
        time.sleep(3600)


class Boom:
    name = "boom"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def solve(self, instance, budget):
        raise ArithmeticError("this solver always fails")
```

Create `tests/bench/test_runner.py`:

```python
from pathlib import Path

import polars as pl
import pytest

from quantic.bench import runner as runner_module
from quantic.bench.config import load
from quantic.bench.keys import CodeVersion
from quantic.bench.runner import DirtyTree, plan, run
from quantic.bench.store import ERROR, KILLED, OK, Store

ROOT = Path(__file__).resolve().parents[2]
TIMING = ["total_s", "timings"]

SMALL = """
[instances]
tiers = ["T0"]
dials = ["none", "D2+D3"]
seeds = [2026]

[references]
qubo_minimum_tiers = ["T0"]

[solvers.almgren_chriss]
budgets_s = [1]

[solvers.cpsat]
budgets_s = [30]

[solvers.simulated_annealing]
encodings = ["binary", "one_hot"]
seeds = "0..1"
budgets_s = [600]
max_samples = 15
"""


def _experiment(tmp_path: Path, text: str, name: str = "exp"):
    path = tmp_path / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return load(path)


def _frame(store: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    with Store(store, read_only=True) as s:
        return s.runs().drop(TIMING), s.samples()


def test_a_stopped_run_resumes_to_the_same_store_as_an_unbroken_one(tmp_path):
    exp = _experiment(tmp_path, SMALL)
    whole, broken = tmp_path / "whole.duckdb", tmp_path / "broken.duckdb"
    run(exp, root=ROOT, store=whole, workers=3, allow_dirty=True)
    first = run(exp, root=ROOT, store=broken, workers=3, allow_dirty=True, stop_after=5)
    assert first.n_written == 5
    assert plan(exp, root=ROOT, store=broken).n_todo == plan(exp, root=ROOT).n_tasks - 5
    run(exp, root=ROOT, store=broken, workers=3, allow_dirty=True)

    runs_whole, samples_whole = _frame(whole)
    runs_broken, samples_broken = _frame(broken)
    assert runs_whole.height == plan(exp, root=ROOT).n_tasks
    assert set(runs_whole["store_status"]) == {OK}
    assert runs_broken.equals(runs_whole)
    assert samples_broken.equals(samples_whole)


def test_running_again_does_nothing_once_the_store_is_complete(tmp_path):
    exp = _experiment(tmp_path, SMALL)
    store = tmp_path / "s.duckdb"
    run(exp, root=ROOT, store=store, workers=3, allow_dirty=True)
    again = run(exp, root=ROOT, store=store, workers=3, allow_dirty=True)
    assert again.n_written == 0


def test_a_hanging_solver_is_killed_and_the_run_carries_on(tmp_path):
    exp = _experiment(tmp_path, """
[instances]
tiers = ["T0"]
dials = ["none"]
seeds = [2026]

[solvers."tests.bench.stand_ins:Hang"]
budgets_s = [0.1]

[solvers.almgren_chriss]
budgets_s = [1]
""")
    store = tmp_path / "s.duckdb"
    summary = run(
        exp, root=ROOT, store=store, workers=2, allow_dirty=True, kill_margin_s=2.0
    )
    rows, _ = _frame(store)
    hung = rows.filter(pl.col("solver") == "tests.bench.stand_ins:Hang")
    assert summary.n_killed == 1
    assert hung["store_status"].to_list() == [KILLED]
    assert hung["error"][0].startswith("killed after")
    assert rows.filter(pl.col("solver") == "almgren_chriss")["store_status"].to_list() == [OK]


def test_a_solver_that_raises_is_a_row_that_says_so(tmp_path):
    exp = _experiment(tmp_path, """
[instances]
tiers = ["T0"]
dials = ["none"]
seeds = [2026]

[solvers."tests.bench.stand_ins:Boom"]
budgets_s = [1]
""")
    store = tmp_path / "s.duckdb"
    summary = run(exp, root=ROOT, store=store, workers=1, allow_dirty=True)
    rows, _ = _frame(store)
    boom = rows.filter(pl.col("solver") == "tests.bench.stand_ins:Boom")
    assert summary.n_errors == 1
    assert boom["store_status"].to_list() == [ERROR]
    assert boom["error"][0] == "ArithmeticError: this solver always fails"


def test_a_dirty_tree_is_refused_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner_module, "code_version",
        lambda root: CodeVersion(commit="abc", lock_hash="def", dirty=True),
    )
    exp = _experiment(tmp_path, SMALL)
    with pytest.raises(DirtyTree, match="allow-dirty"):
        run(exp, root=ROOT, store=tmp_path / "s.duckdb")


def test_the_plan_counts_tasks_and_estimates_from_their_budgets(tmp_path):
    exp = _experiment(tmp_path, SMALL)
    p = plan(exp, root=ROOT, workers=2)
    # 2 instances x (bracket + brute force + 2 QUBO minima) references,
    # and 2 x (1 + 1 + 2 encodings x 2 seeds) runs.
    assert p.n_tasks == 2 * 4 + 2 * 6
    assert p.estimate_s == pytest.approx((8 * 1.0 + 2 * (1 + 30 + 4 * 600)) / 2)


def test_an_instance_that_cannot_be_built_is_an_error_row_not_a_crash():
    from quantic.bench.execute import Executor
    from quantic.bench.keys import RUN, Task

    code = CodeVersion(commit="abc", lock_hash="def", dirty=False)
    task = Task(kind=RUN, tier="T0", dials="D9", instance_seed=2026,
                solver="almgren_chriss", budget_s=1.0)
    row, samples, trace = Executor(code).run(task.key(code), task)
    assert row["store_status"] == ERROR
    assert row["error"].startswith("KeyError")
    assert samples == [] and trace == []
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/bench/test_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.bench.runner'`.

- [ ] **Step 3: Implement the executor**

Create `src/quantic/bench/execute.py`:

```python
"""One task, run in a worker, turned into the rows the store keeps.

A solver that raises is a row with ``store_status = "error"`` and the
exception's type and message; the study carries on. A solver that has to be
killed never reaches this module -- the pool writes that row.
"""

from __future__ import annotations

import importlib
import json
import platform
from collections import OrderedDict
from importlib import metadata

import numpy as np

from quantic.bench.instances import InstanceCache
from quantic.bench.keys import (
    REF_BRACKET,
    REF_BRUTE_FORCE,
    REF_QUBO_MINIMUM,
    CodeVersion,
    Task,
)
from quantic.bench.store import ERROR, OK, blank_row
from quantic.encoding.decode import FEASIBLE, decode
from quantic.encoding.diagnostics import dynamic_range_ratio
from quantic.encoding.encode import EncodedInstance, encode
from quantic.problem.instance import Instance
from quantic.solvers.base import Budget, Result
from quantic.solvers.classical.almgren_chriss import AlmgrenChrissSolver
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.cpsat import CpSat
from quantic.solvers.classical.exact_pwl_milp import ExactPwlMilp, exact_pwl_milp
from quantic.solvers.classical.qubo_minimum import qubo_minimum
from quantic.solvers.classical.scoring import score
from quantic.solvers.qubo.greedy import Greedy
from quantic.solvers.qubo.sampler import QuboSampler, encoding_named
from quantic.solvers.qubo.simulated_annealing import SimulatedAnnealing
from quantic.solvers.qubo.tabu import Tabu

_ENCODED_KEPT = 8
_PACKAGES = ("quantic", "numpy", "scipy", "ortools", "dwave-samplers", "dimod", "duckdb")
_QUBO = {"simulated_annealing": SimulatedAnnealing, "tabu": Tabu, "greedy": Greedy}


def machine() -> str:
    return f"{platform.node()}|{platform.machine()}|{platform.processor()}"


def packages() -> dict[str, str]:
    return {name: metadata.version(name) for name in _PACKAGES}


def _solver(task: Task):
    if task.solver in _QUBO:
        return _QUBO[task.solver](task.encoding, seed=task.solver_seed,
                                  zeta_bits=task.zeta_bits)
    if task.solver == "almgren_chriss":
        return AlmgrenChrissSolver()
    if task.solver == "cpsat":
        return CpSat(seed=task.solver_seed)
    if task.solver == "exact_pwl_milp":
        return ExactPwlMilp()
    module, _, name = task.solver.partition(":")
    return getattr(importlib.import_module(module), name)(seed=task.solver_seed)


class Executor:
    """A worker's state: its caches and the pins every row carries."""

    def __init__(self, code: CodeVersion) -> None:
        self.code = code
        self.where = machine()
        self.versions = json.dumps(packages(), sort_keys=True)
        self.instances = InstanceCache()
        self._encoded: OrderedDict[tuple, EncodedInstance] = OrderedDict()

    def encoded(self, task: Task, instance: Instance) -> EncodedInstance:
        key = (task.tier, task.dials, task.instance_seed, task.encoding, task.zeta_bits)
        if key in self._encoded:
            self._encoded.move_to_end(key)
            return self._encoded[key]
        enc = encode(instance, encoding_named(task.encoding), n_zeta_bits=task.zeta_bits)
        self._encoded[key] = enc
        if len(self._encoded) > _ENCODED_KEPT:
            self._encoded.popitem(last=False)
        return enc

    def run(self, key: str, task: Task) -> tuple[dict, list[tuple], list[tuple]]:
        row = blank_row(key, task, self.code, self.where, self.versions)
        try:
            instance = self.instances.get(task.tier, task.dials, task.instance_seed)
            row["instance_hash"] = instance.content_hash
            row["notional"] = instance.notional()
            return self._run(row, task, instance)
        except Exception as exc:  # the study carries on; the row says what broke
            row["store_status"] = ERROR
            row["error"] = f"{type(exc).__name__}: {exc}"
            return row, [], []

    def _run(self, row, task, instance):
        if task.solver == REF_QUBO_MINIMUM:
            enc = self.encoded(task, instance)
            found = qubo_minimum(enc)
            decoded = decode(enc, found.bits)
            row.update(
                store_status=OK, status="enumerated", qubo_energy=found.energy,
                n_vars=enc.n_vars, dynamic_range=dynamic_range_ratio(enc.qubo),
                detail={"scope": found.scope, "n_states": found.n_states},
            )
            if decoded.classification == FEASIBLE:
                row["qubo_total"] = score(instance, decoded.schedule)[0].total
            return row, [], []

        budget = Budget(wall_clock_s=task.budget_s, max_samples=task.max_samples)
        if task.solver == REF_BRACKET:
            result = exact_pwl_milp(instance)
        elif task.solver == REF_BRUTE_FORCE:
            result = brute_force(instance)
        else:
            solver = _solver(task)
            if isinstance(solver, QuboSampler):
                enc = self.encoded(task, instance)
                row["n_vars"] = enc.n_vars
                row["dynamic_range"] = dynamic_range_ratio(enc.qubo)
                result = solver.solve_encoded(enc, budget)
            else:
                result = solver.solve(instance, budget)
        return self._fill(row, result, instance)

    def _fill(self, row, result: Result, instance: Instance):
        row["store_status"] = OK
        row["status"] = result.status
        detail = dict(result.detail)
        samples = detail.pop("samples", ())
        trace = detail.pop("trace", ())
        if result.schedule is not None:
            row["schedule"] = [list(r) for r in result.schedule.lots]
            row["zeta"] = result.zeta
        if result.objective is not None:
            o = result.objective
            row.update(
                spread=o.spread_cost, temporary=o.temporary_impact_cost,
                permanent=o.permanent_impact_cost, risk=o.risk_cost, total=o.total,
                total_bps=o.total * 1e4 / instance.notional(),
            )
        if result.bracket is not None:
            row["lower"] = result.bracket.lower
            row["upper"] = result.bracket.upper
        if result.feasibility is not None:
            row["feasible"] = result.feasibility.feasible
            row["violations"] = dict(result.feasibility.violations)
        if samples:
            row["n_samples"] = detail.get("n_samples")
            row["n_feasible_samples"] = sum(
                s.count for s in samples if s.classification == FEASIBLE
            )
            row["best_energy"] = detail.get("best_energy")
            row["energy_mismatch"] = detail.get("energy_mismatch")
        row["total_s"] = result.timings.get("total_s")
        row["timings"] = dict(result.timings)
        row["detail"] = json.loads(json.dumps(detail, default=_plain))
        return (
            row,
            [(s.digest, s.count, s.energy, s.classification, s.total) for s in samples],
            [tuple(point) for point in trace],
        )


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
```

- [ ] **Step 4: Implement the pool**

Create `src/quantic/bench/pool.py`:

```python
"""Worker processes that can be killed one at a time.

``concurrent.futures`` cannot stop one task without stopping the pool, and a
sampler written in Python can ignore its budget. Each worker here is a
long-lived process with its own pair of pipes: the main process sends it one
task at a time, notes the deadline, and terminates and replaces it if the
deadline passes. A terminated process can corrupt the pipe it was writing to,
so every worker's pipes are its own and are thrown away with it.

Workers start with ``spawn`` on every platform, so a test on Linux sees what a
run on Windows sees.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import time
from collections import deque
from collections.abc import Callable, Iterable
from multiprocessing.connection import wait

from quantic.bench.keys import CodeVersion, Task

_READY = "ready"
_POLL_S = 0.2
_MAX_FAILED_STARTS = 3


class WorkerStartError(RuntimeError):
    """Workers keep dying before they can take a task."""


def _worker(conn, code: CodeVersion) -> None:
    from quantic.bench.execute import Executor

    executor = Executor(code)
    conn.send(_READY)
    while True:
        item = conn.recv()
        if item is None:
            return
        key, task = item
        conn.send((key, executor.run(key, task)))


class _Slot:
    def __init__(self, ctx, code: CodeVersion) -> None:
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, code), daemon=True)
        self.process.start()
        child.close()
        self.ready = False
        self.job: tuple[str, Task, float] | None = None

    def stop(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join()
        self.conn.close()


# (key, task, payload or None, killed)
OnDone = Callable[[str, Task, tuple | None, bool], None]


def run_tasks(
    jobs: Iterable[tuple[str, Task]],
    *,
    code: CodeVersion,
    n_workers: int,
    deadline_s: Callable[[Task], float],
    on_done: OnDone,
) -> None:
    """Run every job; call ``on_done`` in the main process as each one ends."""
    ctx = mp.get_context("spawn")
    pending = deque(jobs)
    slots = [_Slot(ctx, code) for _ in range(n_workers)]
    failed_starts = 0
    try:
        while pending or any(s.job for s in slots):
            for slot in slots:
                if slot.ready and slot.job is None and pending:
                    key, task = pending.popleft()
                    slot.job = (key, task, time.monotonic() + deadline_s(task))
                    slot.conn.send((key, task))

            by_conn = {s.conn: s for s in slots}
            for conn in wait(list(by_conn), timeout=_POLL_S):
                slot = by_conn[conn]
                try:
                    message = conn.recv()
                except (EOFError, OSError):
                    message = None
                if message == _READY:
                    slot.ready = True
                elif message is not None and slot.job is not None:
                    key, task, _ = slot.job
                    sent_key, payload = message
                    if sent_key == key:
                        slot.job = None
                        on_done(key, task, payload, False)

            now = time.monotonic()
            for i, slot in enumerate(slots):
                died = not slot.process.is_alive()
                late = slot.job is not None and now > slot.job[2]
                if died and not slot.ready:
                    failed_starts += 1
                    if failed_starts > _MAX_FAILED_STARTS:
                        raise WorkerStartError(
                            f"{failed_starts} workers died before they were ready; "
                            f"the last exited with code {slot.process.exitcode}"
                        )
                if late or died:
                    job = slot.job
                    slot.stop()
                    slots[i] = _Slot(ctx, code)
                    if job is not None:
                        on_done(job[0], job[1], None, True)
    finally:
        for slot in slots:
            if slot.ready and slot.job is None and slot.process.is_alive():
                with contextlib.suppress(OSError):
                    slot.conn.send(None)
            slot.stop()
```

- [ ] **Step 5: Implement the runner**

Create `src/quantic/bench/runner.py`:

```python
"""Plan and run an experiment into a store, resumably.

``plan`` says how many tasks an experiment holds, how many a store already
has, and roughly how long the rest will take. ``run`` executes the rest. Each
finished task is written as it arrives; a run interrupted for any reason is
finished by running the same command again.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from quantic.bench.config import Experiment
from quantic.bench.execute import machine, packages
from quantic.bench.keys import REFERENCE, CodeVersion, Task, code_version, expand
from quantic.bench.pool import run_tasks
from quantic.bench.store import KILLED, Store, blank_row

# A budgeted run is killed at KILL_FACTOR * budget + KILL_MARGIN_S seconds.
KILL_FACTOR = 3.0
KILL_MARGIN_S = 30.0
# References are unbudgeted proofs; this only stops one that has hung.
REFERENCE_DEADLINE_S = 3600.0
# For the estimate only: a reference's typical cost.
_REFERENCE_ESTIMATE_S = 1.0


class DirtyTree(RuntimeError):
    """Tracked files differ from the commit the rows would name."""


@dataclass(frozen=True)
class Plan:
    n_tasks: int
    n_done: int
    n_todo: int
    estimate_s: float
    n_workers: int


@dataclass(frozen=True)
class Summary:
    n_written: int
    n_killed: int
    n_errors: int


def deadline_s(task: Task, margin_s: float = KILL_MARGIN_S) -> float:
    if task.kind == REFERENCE or task.budget_s is None:
        return REFERENCE_DEADLINE_S
    return KILL_FACTOR * task.budget_s + margin_s


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def _todo(experiment: Experiment, code: CodeVersion, store: Path | None):
    tasks = [(t.key(code), t) for t in expand(experiment)]
    done: set[str] = set()
    if store is not None and Path(store).exists():
        with Store(store, read_only=True) as s:
            done = s.done_keys()
    return tasks, [(k, t) for k, t in tasks if k not in done]


def plan(
    experiment: Experiment, *, root: Path, store: Path | None = None,
    workers: int | None = None,
) -> Plan:
    code = code_version(root)
    n = workers or default_workers()
    tasks, todo = _todo(experiment, code, store)
    seconds = sum(
        _REFERENCE_ESTIMATE_S if t.kind == REFERENCE else t.budget_s for _, t in todo
    )
    return Plan(
        n_tasks=len(tasks), n_done=len(tasks) - len(todo), n_todo=len(todo),
        estimate_s=seconds / n, n_workers=n,
    )


def run(
    experiment: Experiment,
    *,
    root: Path,
    store: Path,
    workers: int | None = None,
    allow_dirty: bool = False,
    stop_after: int | None = None,
    kill_margin_s: float = KILL_MARGIN_S,
) -> Summary:
    """Run every task the store does not hold yet.

    ``stop_after`` and ``kill_margin_s`` exist for the tests: the first
    interrupts a run partway, the second makes a kill take seconds, not 30.
    """
    code = code_version(root)
    if code.dirty and not allow_dirty:
        raise DirtyTree(
            "tracked files differ from HEAD, so rows would name a commit that did "
            "not produce them. Commit, or pass --allow-dirty (recorded on every row)"
        )
    _, todo = _todo(experiment, code, store)
    if stop_after is not None:
        todo = todo[:stop_after]

    where = machine()
    versions = json.dumps(packages(), sort_keys=True)
    counts = {"written": 0, "killed": 0, "errors": 0}

    with Store(store) as out:
        def on_done(key: str, task: Task, payload, killed: bool) -> None:
            if killed:
                row = blank_row(key, task, code, where, versions)
                row["store_status"] = KILLED
                row["error"] = f"killed after {deadline_s(task, kill_margin_s):.1f} s"
                out.write(row)
                counts["killed"] += 1
            else:
                row, samples, trace = payload
                out.write(row, samples, trace)
                if row["store_status"] != "ok":
                    counts["errors"] += 1
            counts["written"] += 1

        run_tasks(
            todo, code=code, n_workers=workers or default_workers(),
            deadline_s=lambda task: deadline_s(task, kill_margin_s), on_done=on_done,
        )
    return Summary(
        n_written=counts["written"], n_killed=counts["killed"], n_errors=counts["errors"]
    )
```

- [ ] **Step 6: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/bench/test_runner.py -q`
Expected: 7 passed, in about a minute: every test starts worker processes, and each `spawn` re-imports NumPy, SciPy and OR-Tools.

- [ ] **Step 7: Lint, contracts, commit**

```
feat(bench): run an experiment on killable workers, resumably, into the store
```

```bash
git add src/quantic/bench/execute.py src/quantic/bench/pool.py src/quantic/bench/runner.py tests/bench/stand_ins.py tests/bench/test_runner.py
git commit -F <message file>
```

---

### Task 10: Metrics and the report

Spec §7.

**Files:**
- Create: `src/quantic/bench/metrics.py`, `src/quantic/bench/report.py`
- Test: `tests/bench/test_metrics_report.py`

**Interfaces:**
- Consumes: `Store`, `REF_BRACKET`, `REF_QUBO_MINIMUM`, `RUN`, `OK`.
- Produces: `metrics.tts99(t_run, p) -> float | None`; `metrics.scored(runs) -> pl.DataFrame` (adds `ref_lower`, `ref_upper`, `ref_closed`, `gap`, `energy_gap`, `encoding_regret`, `ok_<eps>`); `metrics.cells(runs, *, eps=0.01, seed=0) -> pl.DataFrame`; `metrics.EPSILONS`, `metrics.CELL`. `report.report(store, out, *, commit=None) -> list[Path]`; `report.best_encoding(table)`; `report.crossover(best, budget)`; `report.dial_effects(best)`; `report.bracket_check(runs)`; `report.CENSORED = "censored"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/bench/test_metrics_report.py`:

```python
import math

import polars as pl
import pytest

from quantic.bench.keys import REF_QUBO_MINIMUM, REFERENCE, RUN, CodeVersion, Task
from quantic.bench.metrics import cells, scored, tts99
from quantic.bench.report import bracket_check, dial_effects, report
from quantic.bench.store import OK, Store, blank_row
from quantic.problem.dials import Dials
from tests.bench.rows import ref, row, run_row, write


@pytest.mark.parametrize(
    "p, expected",
    [(1.0, 2.0), (0.0, None), (0.5, 2.0 * math.log(0.01) / math.log(0.5))],
)
def test_time_to_solution(p, expected):
    got = tts99(2.0, p)
    assert got == expected if expected is None else got == pytest.approx(expected)


def test_tts_on_a_known_success_rate_from_runs(tmp_path):
    """Two of four seeds reach the optimum: p = 0.5."""
    rows = [ref("none", 10.0, 10.0)] + [
        run_row("none", "sa", s, total) for s, total in enumerate([10.0, 10.0, 11.0, 12.0])
    ]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        table = cells(s.runs(), eps=0.0)
    got = table.row(0, named=True)
    assert got["p_success"] == 0.5
    assert got["tts99_s"] == pytest.approx(0.5 * math.log(0.01) / math.log(0.5))
    assert got["gap_median"] == pytest.approx(0.05)
    assert got["censored_above_s"] is None


def test_an_infeasible_run_never_succeeds_and_a_cell_with_no_success_is_censored(tmp_path):
    rows = [ref("none", 10.0, 10.0),
            run_row("none", "sa", 0, 9.0, feasible=False),
            run_row("none", "sa", 1, None, feasible=None)]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        runs = s.runs()
    assert not scored(runs)["ok_0.05"].any()
    got = cells(runs).row(0, named=True)
    assert got["tts99_s"] is None
    assert got["censored_above_s"] == 2.0
    assert got["gap_median"] is None


def test_an_open_reference_gives_no_gap_and_no_success(tmp_path):
    rows = [ref("none", 9.0, 10.0), run_row("none", "sa", 0, 10.0)]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        out = scored(s.runs())
    assert out["gap"].to_list() == [None]
    assert not out["ok_0.0"].any()


def test_regret_is_read_against_the_qubos_minimum(tmp_path):
    qmin = Task(kind=REFERENCE, tier="T0", dials="none", instance_seed=1,
                solver=REF_QUBO_MINIMUM, encoding="binary", zeta_bits=8)
    run = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="sa",
               encoding="binary", zeta_bits=8, solver_seed=0, budget_s=1.0)
    rows = [
        ref("none", 10.0, 10.0),
        row(qmin, qubo_energy=20.0, qubo_total=10.5),
        row(run, total=11.0, feasible=True, best_energy=21.0, total_s=1.0),
    ]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        out = scored(s.runs()).row(0, named=True)
    assert out["energy_gap"] == pytest.approx(0.05)
    assert out["encoding_regret"] == pytest.approx(0.05)


def test_dial_effects_are_the_factorial_contrasts():
    """A gap that is D1's indicator gives D1 an effect of one and nothing else."""
    labels = [d.label for d in Dials.combinations(include_cvar=True)]
    best = pl.DataFrame({
        "tier": ["T0"] * 16, "dials": labels, "solver": ["sa"] * 16,
        "budget_s": [1.0] * 16,
        "gap_median": [1.0 if "D1" in label.split("+") else 0.0 for label in labels],
    })
    effects = {r["effect"]: r["value"] for r in dial_effects(best).iter_rows(named=True)}
    assert effects["D1"] == pytest.approx(1.0)
    assert all(abs(v) < 1e-12 for k, v in effects.items() if k != "D1")


def test_brackets_that_do_not_meet_are_listed(tmp_path):
    rows = [
        ref("none", 10.0, 10.0), ref("D1", 20.0, 20.0),
        run_row("none", "cpsat", 0, 10.0, lower=9.99, upper=10.0),
        run_row("D1", "cpsat", 0, 20.5, lower=20.4, upper=20.5),
    ]
    with Store(write(tmp_path / "s.duckdb", rows), read_only=True) as s:
        apart = bracket_check(s.runs())
    assert apart["dials"].to_list() == ["D1"]


def test_the_report_writes_every_table_and_refuses_mixed_commits(tmp_path):
    rows = [ref("none", 10.0, 10.0), run_row("none", "sa", 0, 10.0),
            run_row("none", "cpsat", 0, 10.0, lower=10.0, upper=10.0)]
    store = write(tmp_path / "s.duckdb", rows)
    written = report(store, tmp_path / "out")
    names = {p.name for p in written}
    assert {"cells.csv", "crossover_1s.md", "dial_effects.csv", "bracket_check.md"} <= names
    assert "_none_" in (tmp_path / "out" / "bracket_check.md").read_text(encoding="utf-8")

    other = CodeVersion(commit="beef", lock_hash="10c4", dirty=False)
    task = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="sa",
                solver_seed=9, budget_s=1.0)
    with Store(store) as s:
        s.write({**blank_row(task.key(other), task, other, "here", "{}"),
                 "store_status": OK})
    with pytest.raises(ValueError, match="--commit"):
        report(store, tmp_path / "out2")
    assert report(store, tmp_path / "out3", commit="c0ffee")


def test_a_report_on_a_store_with_no_references_says_so_cell_by_cell(tmp_path):
    rows = [run_row("none", "sa", 0, 10.0), run_row("none", "sa", 1, 11.0)]
    store = write(tmp_path / "s.duckdb", rows)
    report(store, tmp_path / "out")
    text = (tmp_path / "out" / "crossover_1s.md").read_text(encoding="utf-8")
    assert "- / 100% / censored" in text
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/bench/test_metrics_report.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.bench.metrics'`.

- [ ] **Step 3: Implement**

Create `src/quantic/bench/metrics.py`:

```python
"""From rows to the numbers of parent spec section 9.1.

A run succeeds at tolerance ``eps`` when its schedule is feasible without
repair and its total is within ``eps`` of the instance's optimum, measured
against the proven lower bound. ``eps = 0`` means the optimum to the
reference bracket's own width. Instances whose reference bracket is open have
no gap and no success; they report the width instead.

Pooling is over solver seeds and instance seeds together, per cell of tier x
dial combination x solver x encoding x budget.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from quantic.bench.keys import REF_BRACKET, REF_QUBO_MINIMUM, RUN
from quantic.bench.store import OK

EPSILONS = (0.0, 0.001, 0.01, 0.05)
# Two evaluator calls at one schedule agree to round-off; this is not a tolerance.
_SAME = 1e-9
_BOOTSTRAP = 1000
CELL = ["tier", "dials", "solver", "encoding", "budget_s"]


def tts99(t_run: float, p: float) -> float | None:
    """Time to reach the target with 99% confidence; ``None`` when censored."""
    if p >= 1.0:
        return t_run
    if p <= 0.0:
        return None
    return t_run * math.log(0.01) / math.log(1.0 - p)


def _rounded(relative: pl.Expr) -> pl.Expr:
    """Round-off either side of zero is zero; anything larger is kept, sign and all."""
    return pl.when(relative.abs() <= _SAME).then(0.0).otherwise(relative)


def references(runs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per instance: the proven bracket, and per encoding the QUBO's minimum."""
    bracket = runs.filter(
        (pl.col("solver") == REF_BRACKET) & (pl.col("store_status") == OK)
    ).select(
        "instance_hash",
        pl.col("lower").alias("ref_lower"),
        pl.col("upper").alias("ref_upper"),
    )
    qubo = runs.filter(
        (pl.col("solver") == REF_QUBO_MINIMUM) & (pl.col("store_status") == OK)
    ).select(
        "instance_hash", "encoding", "zeta_bits",
        pl.col("qubo_energy").alias("ref_qubo_energy"),
        pl.col("qubo_total").alias("ref_qubo_total"),
    )
    return bracket, qubo


def scored(runs: pl.DataFrame) -> pl.DataFrame:
    """Every solver run with its gap, its success at each eps, and its regret."""
    bracket, qubo = references(runs)
    out = (
        runs.filter(pl.col("kind") == RUN)
        .join(bracket, on="instance_hash", how="left")
        .join(qubo, on=["instance_hash", "encoding", "zeta_bits"], how="left")
    )
    # No reference row, or an open one, is no success and no gap -- never a null
    # that turns a success rate into NaN.
    closed = (
        (pl.col("ref_upper") - pl.col("ref_lower"))
        <= _SAME * pl.max_horizontal(pl.lit(1.0), pl.col("ref_upper").abs())
    ).fill_null(False)
    feasible = (pl.col("store_status") == OK) & pl.col("feasible").fill_null(False)
    out = out.with_columns(
        closed.alias("ref_closed"),
        pl.when(feasible & closed)
        .then(_rounded((pl.col("total") - pl.col("ref_lower")) / pl.col("ref_lower").abs()))
        .alias("gap"),
        _rounded(
            (pl.col("best_energy") - pl.col("ref_qubo_energy"))
            / pl.max_horizontal(pl.lit(1.0), pl.col("ref_qubo_energy").abs())
        ).alias("energy_gap"),
        _rounded(
            (pl.col("ref_qubo_total") - pl.col("ref_lower")) / pl.col("ref_lower").abs()
        ).alias("encoding_regret"),
    )
    for eps in EPSILONS:
        target = pl.col("ref_upper") + pl.max_horizontal(
            eps * pl.col("ref_lower").abs(), _SAME * pl.col("ref_upper").abs()
        )
        out = out.with_columns(
            (feasible & closed & (pl.col("total") <= target)).alias(f"ok_{eps}")
        )
    return out


def _bootstrap_tts(ok: np.ndarray, t_run: float, rng: np.random.Generator):
    if ok.size == 0:
        return None, None
    draws = rng.choice(ok, size=(_BOOTSTRAP, ok.size), replace=True).mean(axis=1)
    values = np.array([np.inf if (v := tts99(t_run, p)) is None else v for p in draws])
    # inverted_cdf returns observed values, so a censored draw stays infinite
    # instead of turning the interpolation into inf - inf.
    lo, hi = np.quantile(values, [0.025, 0.975], method="inverted_cdf")
    return (None if np.isinf(lo) else float(lo)), (None if np.isinf(hi) else float(hi))


def cells(runs: pl.DataFrame, *, eps: float = 0.01, seed: int = 0) -> pl.DataFrame:
    """One row per cell: gap, feasibility, success, TTS with a bootstrap interval."""
    rows = scored(runs)
    rng = np.random.default_rng(seed)
    out = []
    for keys, group in rows.group_by(CELL, maintain_order=True):
        ok = group[f"ok_{eps}"].to_numpy().astype(float)
        p = float(ok.mean()) if ok.size else 0.0
        t_run = float(group["total_s"].drop_nulls().mean() or 0.0)
        lo, hi = _bootstrap_tts(ok, t_run, rng)
        sampled = group["n_samples"].drop_nulls()
        if sampled.len():
            feasibility = float(group["n_feasible_samples"].sum() / sampled.sum())
        else:
            feasibility = float(group["feasible"].fill_null(False).mean())
        gaps = group["gap"].drop_nulls()
        out.append({
            **dict(zip(CELL, keys, strict=True)),
            "n_runs": group.height,
            "n_killed": int((group["store_status"] == "killed").sum()),
            "n_errors": int((group["store_status"] == "error").sum()),
            "gap_median": float(gaps.median()) if gaps.len() else None,
            "gap_q1": float(gaps.quantile(0.25)) if gaps.len() else None,
            "gap_q3": float(gaps.quantile(0.75)) if gaps.len() else None,
            "feasibility": feasibility,
            "p_success": p,
            "t_run_s": t_run,
            "tts99_s": tts99(t_run, p),
            "tts99_lo_s": lo,
            "tts99_hi_s": hi,
            "censored_above_s": None if p > 0 else (group["budget_s"][0] or 0) * group.height,
            "bps_best": group.filter(pl.col("feasible").fill_null(False))["total_bps"].min(),
            "energy_gap_median": group["energy_gap"].drop_nulls().median(),
            "encoding_regret": group["encoding_regret"].drop_nulls().median(),
            "dynamic_range": group["dynamic_range"].drop_nulls().median(),
        })
    return pl.DataFrame(out, infer_schema_length=None)
```

Create `src/quantic/bench/report.py`:

```python
"""The crossover report: CSV for machines, Markdown for people.

Three products, each written as both:

* ``crossover_<budget>s``: tier x dial rows, one column per solver -- each
  sampler at its best encoding, named in the cell -- showing median gap,
  feasibility rate and TTS at 99% for eps = 1%.
* ``dial_effects``: each dial's main effect and every pairwise interaction on
  each solver's median gap, at the largest budget. Sixteen combinations is the
  full 2**4 factorial, so these are exact contrasts, not a screening estimate.
* ``bracket_check``: every instance where a CP-SAT bracket and the reference
  bracket do not overlap. Expected empty.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import polars as pl

from quantic.bench.keys import REF_BRACKET, RUN
from quantic.bench.metrics import cells
from quantic.bench.store import OK, Store

DIALS = ("D1", "D2", "D3", "D4")
_OVERLAP_RTOL = 1e-6
CENSORED = "censored"


def _has(label: str, dial: str) -> bool:
    return dial in label.split("+")


def best_encoding(table: pl.DataFrame) -> pl.DataFrame:
    """Per (tier, dials, solver, budget), the encoding with the lowest median gap."""
    return (
        table.with_columns(pl.col("gap_median").fill_null(float("inf")).alias("_gap"),
                           pl.col("tts99_s").fill_null(float("inf")).alias("_tts"))
        .sort(["_gap", "_tts", "encoding"], nulls_last=True)
        .group_by(["tier", "dials", "solver", "budget_s"], maintain_order=True)
        .first()
        .drop("_gap", "_tts")
    )


def _cell_text(row: dict) -> str:
    gap = "-" if row["gap_median"] is None else f"{100 * row['gap_median']:.2f}%"
    tts = CENSORED if row["tts99_s"] is None else f"{row['tts99_s']:.3g}s"
    enc = f" ({row['encoding']})" if row["encoding"] else ""
    return f"{gap} / {100 * row['feasibility']:.0f}% / {tts}{enc}"


def crossover(best: pl.DataFrame, budget: float) -> pl.DataFrame:
    at = best.filter(pl.col("budget_s") == budget)
    texts = at.with_columns(
        pl.Series("cell", [_cell_text(r) for r in at.iter_rows(named=True)])
    )
    return (
        texts.pivot(on="solver", index=["tier", "dials"], values="cell")
        .sort(["tier", "dials"])
    )


def dial_effects(best: pl.DataFrame) -> pl.DataFrame:
    top = best.filter(pl.col("budget_s") == best["budget_s"].max())
    out = []
    for (tier, solver), group in top.group_by(["tier", "solver"], maintain_order=True):
        gaps = {r["dials"]: r["gap_median"] for r in group.iter_rows(named=True)}
        if len(gaps) < 16 or any(g is None for g in gaps.values()):
            continue

        def mean(test, gaps=gaps) -> float:
            values = [g for label, g in gaps.items() if test(label)]
            return sum(values) / len(values)

        for d in DIALS:
            out.append({"tier": tier, "solver": solver, "effect": d,
                        "value": mean(lambda x, d=d: _has(x, d))
                        - mean(lambda x, d=d: not _has(x, d))})
        for a, b in itertools.combinations(DIALS, 2):
            same = mean(lambda x, a=a, b=b: _has(x, a) == _has(x, b))
            differ = mean(lambda x, a=a, b=b: _has(x, a) != _has(x, b))
            out.append({"tier": tier, "solver": solver, "effect": f"{a}x{b}",
                        "value": same - differ})
    return pl.DataFrame(out, schema={"tier": pl.String, "solver": pl.String,
                                     "effect": pl.String, "value": pl.Float64})


def bracket_check(runs: pl.DataFrame) -> pl.DataFrame:
    ref = runs.filter((pl.col("solver") == REF_BRACKET) & (pl.col("store_status") == OK))
    cp = runs.filter(
        (pl.col("kind") == RUN) & (pl.col("solver") == "cpsat")
        & pl.col("lower").is_not_null()
    )
    joined = cp.join(
        ref.select("instance_hash", pl.col("lower").alias("ref_lower"),
                   pl.col("upper").alias("ref_upper")),
        on="instance_hash",
    )
    slack = _OVERLAP_RTOL * pl.max_horizontal(pl.lit(1.0), pl.col("ref_upper").abs())
    apart = (pl.col("lower") > pl.col("ref_upper") + slack) | (
        pl.col("ref_lower") > pl.col("upper") + slack
    )
    return joined.filter(apart).select(
        "tier", "dials", "instance_seed", "solver_seed", "budget_s",
        "lower", "upper", "ref_lower", "ref_upper",
    )


def _markdown(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "_none_\n"
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([head, rule, *body]) + "\n"


def _write(frame: pl.DataFrame, out: Path, stem: str) -> list[Path]:
    csv, md = out / f"{stem}.csv", out / f"{stem}.md"
    frame.write_csv(csv)
    md.write_text(_markdown(frame), encoding="utf-8")
    return [csv, md]


def report(store: Path, out: Path, *, commit: str | None = None) -> list[Path]:
    with Store(store, read_only=True) as s:
        runs = s.runs()
    if commit is not None:
        runs = runs.filter(pl.col("commit") == commit)
    commits = runs["commit"].unique().to_list()
    if len(commits) > 1:
        raise ValueError(
            f"the store holds rows from {len(commits)} commits; pass --commit to "
            "choose one, since rows from different code do not belong in one table"
        )
    out.mkdir(parents=True, exist_ok=True)
    table = cells(runs)
    best = best_encoding(table)
    written = _write(table, out, "cells")
    for budget in sorted(best["budget_s"].unique().to_list()):
        written += _write(crossover(best, budget), out, f"crossover_{budget:g}s")
    written += _write(dial_effects(best), out, "dial_effects")
    written += _write(bracket_check(runs), out, "bracket_check")
    return written
```

Two things the dry run found, both in the code above: `np.quantile` interpolating between two infinite bootstrap draws gives `inf - inf = nan`, hence `method="inverted_cdf"`; and with no reference row the success flag was null, which made `p` and TTS `nan`, hence `fill_null(False)` on `closed`.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/bench/test_metrics_report.py -q`
Expected: 11 passed.

- [ ] **Step 5: Lint, contracts, commit**

```
feat(bench): gaps, success, TTS with bootstrap intervals, and the crossover report
```

```bash
git add src/quantic/bench/metrics.py src/quantic/bench/report.py tests/bench/test_metrics_report.py
git commit -F <message file>
```

---

### Task 11: The CLI

Spec §5.1 and §7.3.

**Files:**
- Modify: `src/quantic/cli.py`
- Test: `tests/bench/test_cli_bench.py`

**Interfaces:**
- Consumes: `bench.config.load`, `bench.runner.{plan, run, DirtyTree}`, `bench.report.report`.
- Produces: `quantic bench plan CONFIG [--store PATH] [--workers N]`, `quantic bench run CONFIG --store PATH [--workers N] [--allow-dirty]`, `quantic bench report STORE --out DIR [--commit SHA]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/bench/test_cli_bench.py`:

```python
from typer.testing import CliRunner

from quantic.cli import app
from tests.bench.rows import ref, run_row, write


def test_the_cli_plans_without_running(tmp_path):
    config = tmp_path / "exp.toml"
    config.write_text(
        "[instances]\ntiers = ['T0']\ndials = ['none']\nseeds = [2026]\n"
        "[solvers.almgren_chriss]\nbudgets_s = [1]\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["bench", "plan", str(config), "--workers", "2"])
    assert result.exit_code == 0, result.output
    assert "3 tasks, 0 already in the store, 3 to run" in result.output


def test_the_cli_refuses_a_bad_config(tmp_path):
    config = tmp_path / "exp.toml"
    config.write_text("[instances]\ncolour = 'blue'\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["bench", "plan", str(config)])
    assert result.exit_code != 0
    assert "unknown key" in result.output


def test_the_cli_reports_a_store(tmp_path):
    store = write(tmp_path / "s.duckdb", [ref("none", 10.0, 10.0),
                                         run_row("none", "sa", 0, 10.0)])
    result = CliRunner().invoke(app, ["bench", "report", str(store), "--out",
                                      str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "crossover_1s.md").exists()
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m pytest tests/bench/test_cli_bench.py -q`
Expected: FAIL — exit code 2, `No such command 'bench'`.

- [ ] **Step 3: Implement**

In `src/quantic/cli.py`, add to the imports, before `from quantic.core.session import ...`:

```python
from quantic.bench import config as bench_config
from quantic.bench import report as bench_report
from quantic.bench import runner as bench_runner
```

after `app.add_typer(data_app, name="data")`:

```python
bench_app = typer.Typer(help="The benchmark: plan, run and report an experiment.")
app.add_typer(bench_app, name="bench")
```

and append to the end of the module:

```python
# The repository whose commit every bench row names: the one this package runs from.
_REPO_ROOT = Path(__file__).resolve().parents[2]


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


@bench_app.command("run")
def bench_run(
    config: Annotated[Path, typer.Argument(help="Experiment TOML.")],
    store: Annotated[Path, typer.Option("--store")],
    workers: Annotated[int | None, typer.Option("--workers")] = None,
    allow_dirty: Annotated[
        bool, typer.Option("--allow-dirty", help="Run on uncommitted changes; recorded.")
    ] = False,
) -> None:
    """Run every task the store does not hold yet. Safe to interrupt and rerun."""
    try:
        s = bench_runner.run(
            _experiment(config), root=_REPO_ROOT, store=store, workers=workers,
            allow_dirty=allow_dirty,
        )
    except bench_runner.DirtyTree as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"wrote {s.n_written} rows: {s.n_killed} killed, {s.n_errors} errors"
    )


@bench_app.command("report")
def bench_report_command(
    store: Annotated[Path, typer.Argument(help="The DuckDB store.")],
    out: Annotated[Path, typer.Option("--out")],
    commit: Annotated[str | None, typer.Option("--commit")] = None,
) -> None:
    """Write the crossover tables, dial effects and bracket check."""
    for path in bench_report.report(store, out, commit=commit):
        console.print(str(path))
```

- [ ] **Step 4: Run the tests, and the CLI by hand**

Run: `.venv/Scripts/python -m pytest tests/bench/test_cli_bench.py tests/test_cli.py -q`
Expected: PASS.
Run: `.venv/Scripts/quantic.exe bench plan experiments/m4-gate.toml` once Task 12 has written that file; until then any small TOML will do.

- [ ] **Step 5: Lint, contracts, commit**

```
feat(cli): quantic bench plan, run and report
```

```bash
git add src/quantic/cli.py tests/bench/test_cli_bench.py
git commit -F <message file>
```

---

### Task 12: The gate

Spec §9. Read the module docstring of the gate before running it.

**Files:**
- Create: `experiments/m4-gate.toml`, `experiments/m4-crossover.toml`
- Test: `tests/bench/test_m4_gate.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the gate; the study's experiment file for Task 13.

- [ ] **Step 1: Write the experiments**

Create `experiments/m4-gate.toml`:

```toml
# The M4 gate (tests/bench/test_m4_gate.py): the M3 ladder's 32 T0-T1 instances.
# Samplers are capped by samples, not by the clock, so every row is
# reproducible and the gate can check that it is.

[instances]
tiers = ["T0", "T1"]
dials = "all"
seeds = [2026]

[references]
qubo_minimum_tiers = ["T0"]

[solvers.almgren_chriss]
budgets_s = [1]

[solvers.exact_pwl_milp]
budgets_s = [60]

[solvers.cpsat]
seeds = [0]
budgets_s = [60]

[solvers.simulated_annealing]
encodings = ["binary", "one_hot", "domain_wall"]
zeta_bits = 8
seeds = "0..4"
budgets_s = [600]
max_samples = 256

[solvers.tabu]
encodings = ["binary", "one_hot", "domain_wall"]
zeta_bits = 8
seeds = "0..2"
budgets_s = [600]
max_samples = 32

[solvers.greedy]
encodings = ["binary", "one_hot", "domain_wall"]
zeta_bits = 8
seeds = "0..2"
budgets_s = [600]
max_samples = 256
```

Create `experiments/m4-crossover.toml`:

```toml
# The M4 crossover study: every free classical solver on T0-T1, against proven
# brackets. M4 design section 10. Run it with
#   quantic bench plan experiments/m4-crossover.toml --store <path>
#   quantic bench run  experiments/m4-crossover.toml --store <path>
# and rerun the second command after any interruption.

[instances]
tiers = ["T0", "T1"]
dials = "all"
seeds = "2026..2030"

[references]
qubo_minimum_tiers = ["T0"]

[solvers.almgren_chriss]
budgets_s = [1]

[solvers.exact_pwl_milp]
budgets_s = [0.1, 1, 10, 60]

[solvers.cpsat]
seeds = "0..4"
budgets_s = [0.1, 1, 10, 60]

[solvers.simulated_annealing]
encodings = ["binary", "one_hot", "domain_wall"]
zeta_bits = 8
seeds = "0..49"
budgets_s = [0.1, 1, 10, 60]

[solvers.tabu]
encodings = ["binary", "one_hot", "domain_wall"]
zeta_bits = 8
seeds = "0..49"
budgets_s = [0.1, 1, 10, 60]

[solvers.greedy]
encodings = ["binary", "one_hot", "domain_wall"]
zeta_bits = 8
seeds = "0..49"
budgets_s = [0.1, 1, 10, 60]
```

Run: `.venv/Scripts/quantic.exe bench plan experiments/m4-crossover.toml`
Expected: `292480 tasks, 0 already in the store, 292480 to run; about 102.9 h on 14 workers ...`

- [ ] **Step 2: Write the gate**

Create `tests/bench/test_m4_gate.py`:

```python
"""The M4 gate -- M4 design section 9.

The harness is correct and complete on the M3 ladder's 32 T0-T1 instances.
Every check is about correctness or integrity; none says a solver is good.
A slow sampler is a finding for the result doc, not a failure here.

The experiment runs once per session into a temporary store
(``experiments/m4-gate.toml``, about ten minutes on 14 workers); every check
reads that store. Checks 6 and 7 run small experiments of their own.

Never weaken a check to make it pass. If one fails, stop and report which
check, which instance, which solver and which encoding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantic.bench.config import load
from quantic.bench.execute import Executor
from quantic.bench.instances import InstanceCache
from quantic.bench.keys import (
    REF_BRACKET,
    REF_BRUTE_FORCE,
    REF_QUBO_MINIMUM,
    RUN,
    code_version,
    expand,
)
from quantic.bench.metrics import cells
from quantic.bench.report import CENSORED, best_encoding, bracket_check, crossover
from quantic.bench.runner import plan, run
from quantic.bench.store import KILLED, OK, Store
from quantic.encoding.encode import encode
from quantic.problem.schedule import Schedule
from quantic.solvers.base import _BRACKET_RTOL, OPTIMAL, Budget
from quantic.solvers.classical.qubo_minimum import qubo_minimum
from quantic.solvers.classical.scoring import score
from quantic.solvers.qubo.sampler import encoding_named, energies, run_sampler

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "experiments" / "m4-gate.toml"
# Two evaluator calls at one schedule: round-off only.
SAME = 1e-9
# The same energy accumulated in two orders at dynamic ranges up to 1e10: the
# tolerance M3b set for the closed form against the dense z'Qz.
DENSE = 1e-6
SOLVERS = ("almgren_chriss", "exact_pwl_milp", "cpsat", "simulated_annealing", "tabu",
           "greedy")


@pytest.fixture(scope="module")
def gate(tmp_path_factory) -> pl.DataFrame:
    store = tmp_path_factory.mktemp("gate") / "gate.duckdb"
    summary = run(load(GATE), root=ROOT, store=store, allow_dirty=True)
    assert summary.n_errors == 0 and summary.n_killed == 0, summary
    with Store(store, read_only=True) as s:
        return s.runs()


def _schedule(row) -> Schedule:
    return Schedule(lots=tuple(tuple(r) for r in json.loads(row["schedule"])))


def test_the_store_holds_every_task_and_nothing_failed(gate):
    assert gate.height == plan(load(GATE), root=ROOT).n_tasks
    assert set(gate["store_status"]) == {OK}
    assert set(gate.filter(pl.col("kind") == RUN)["solver"]) == set(SOLVERS)
    assert gate["instance_hash"].n_unique() == 32


def test_check_1_every_objective_is_the_evaluators_and_every_optimum_is_proven(gate):
    cache = InstanceCache()
    wrong = []
    for row in gate.filter(pl.col("schedule").is_not_null()).iter_rows(named=True):
        inst = cache.get(row["tier"], row["dials"], row["instance_seed"])
        total = score(inst, _schedule(row))[0].total
        if abs(total - row["total"]) > SAME * max(1.0, abs(total)):
            wrong.append((row["tier"], row["dials"], row["solver"], row["encoding"]))
        if row["status"] == OPTIMAL:
            width = row["upper"] - row["lower"]
            assert row["feasible"], row["key"]
            assert width <= _BRACKET_RTOL * max(1.0, abs(row["upper"])), row["key"]
    assert not wrong


def test_check_2_cpsat_meets_the_reference_bracket_and_t0_meets_brute_force(gate):
    assert bracket_check(gate).is_empty()
    cp = gate.filter((pl.col("solver") == "cpsat") & pl.col("lower").is_not_null())
    assert cp["instance_hash"].n_unique() == 32

    by = {
        (r["instance_hash"], r["solver"]): r
        for r in gate.filter(pl.col("tier") == "T0").iter_rows(named=True)
        if r["solver"] in (REF_BRACKET, REF_BRUTE_FORCE, "cpsat")
    }
    for (instance, solver), row in by.items():
        if solver != REF_BRUTE_FORCE:
            continue
        optimum = row["total"]
        for other in (REF_BRACKET, "cpsat"):
            mine = by[(instance, other)]
            if mine["status"] == OPTIMAL:
                assert mine["total"] == pytest.approx(optimum, rel=SAME), (instance, other)


def test_check_3_every_sampler_reports_the_qubos_own_energy(gate):
    """A sampler that read the bits in another order would report other energies."""
    sampled = gate.filter(pl.col("n_samples").is_not_null())
    assert sampled.height == 32 * 3 * (5 + 3 + 3)
    assert sampled["energy_mismatch"].max() <= DENSE
    assert (sampled["n_samples"] == sampled["max_samples"]).all()


def test_check_4_the_sampler_pipeline_returns_the_qubos_minimum_intact(gate):
    """Fed the enumerated minimum, the pipeline reports its energy and its total.

    Whether a sampler *finds* the minimum is a finding for the result doc, not a
    gate condition: on the CVaR instances the QUBO has 260-296 variables and a
    dynamic range near 1e7, and the dry run's annealer ended 12-45 times above
    it. What the gate can demand is that a sampler which does return the
    minimum is credited with it -- the right energy, the right decoded total.
    """
    cache = InstanceCache()
    refs = gate.filter(pl.col("solver") == REF_QUBO_MINIMUM)
    assert refs.height == 16 * 3
    for row in refs.iter_rows(named=True):
        inst = cache.get(row["tier"], row["dials"], row["instance_seed"])
        enc = encode(inst, encoding_named(row["encoding"]), n_zeta_bits=row["zeta_bits"])
        lowest = qubo_minimum(enc)

        def draw(bqm, reads, seed, bits=lowest.bits, enc=enc):
            stack = np.tile(bits, (reads, 1))
            return stack, energies(enc.qubo, stack)

        result = run_sampler("stand_in", enc, Budget(max_samples=1), 0, draw)
        where = (row["dials"], row["encoding"])
        assert result.detail["best_energy"] == pytest.approx(
            row["qubo_energy"], rel=DENSE
        ), where
        if row["qubo_total"] is None:
            assert result.schedule is None, where
        else:
            assert result.objective.total == pytest.approx(row["qubo_total"], rel=SAME), where


def test_check_5_rerunning_a_task_reproduces_its_row(gate):
    code = code_version(ROOT)
    executor = Executor(code)
    picked = (
        gate.filter(pl.col("kind") == RUN)
        .group_by(["tier", "solver"]).agg(pl.col("key").min())
    )
    tasks = {t.key(code): t for t in expand(load(GATE))}
    ignore = {"total_s", "timings"}
    for key in picked["key"]:
        again, _, _ = executor.run(key, tasks[key])
        row = gate.filter(pl.col("key") == key).row(0, named=True)
        for column, value in again.items():
            if column in ignore:
                continue
            stored = row[column]
            if isinstance(value, (dict, list)):
                value = json.loads(json.dumps(value))
                stored = json.loads(stored)
            assert stored == value, (key, column)


SMALL = """
[instances]
tiers = ["T0"]
dials = ["none", "D2+D3"]
seeds = [2026]

[solvers.cpsat]
budgets_s = [30]

[solvers.simulated_annealing]
encodings = ["binary", "one_hot"]
seeds = "0..1"
budgets_s = [600]
max_samples = 15
"""


def test_check_6_a_stopped_run_resumes_to_the_same_store(tmp_path):
    path = tmp_path / "small.toml"
    path.write_text(SMALL, encoding="utf-8")
    exp = load(path)
    whole, broken = tmp_path / "whole.duckdb", tmp_path / "broken.duckdb"
    run(exp, root=ROOT, store=whole, workers=3, allow_dirty=True)
    run(exp, root=ROOT, store=broken, workers=3, allow_dirty=True, stop_after=4)
    run(exp, root=ROOT, store=broken, workers=3, allow_dirty=True)
    with Store(whole, read_only=True) as a, Store(broken, read_only=True) as b:
        assert b.runs().drop("total_s", "timings").equals(a.runs().drop("total_s", "timings"))
        assert b.samples().equals(a.samples())


def test_check_7_a_hanging_solver_is_killed_and_the_run_carries_on(tmp_path):
    path = tmp_path / "hang.toml"
    path.write_text(
        "[instances]\ntiers = ['T0']\ndials = ['none']\nseeds = [2026]\n"
        "[solvers.'tests.bench.stand_ins:Hang']\nbudgets_s = [0.1]\n"
        "[solvers.almgren_chriss]\nbudgets_s = [1]\n",
        encoding="utf-8",
    )
    store = tmp_path / "s.duckdb"
    run(load(path), root=ROOT, store=store, workers=2, allow_dirty=True, kill_margin_s=2.0)
    with Store(store, read_only=True) as s:
        status = dict(s.runs().select("solver", "store_status").iter_rows())
    assert status["tests.bench.stand_ins:Hang"] == KILLED
    assert status["almgren_chriss"] == OK


def test_check_8_the_report_fills_every_cell(gate):
    best = best_encoding(cells(gate))
    for budget in best["budget_s"].unique().to_list():
        table = crossover(best, budget)
        solvers = [c for c in table.columns if c not in ("tier", "dials")]
        assert table.height == 32, budget
        for column in solvers:
            assert table[column].null_count() == 0, (budget, column)
            assert all(t.endswith(("s", ")", CENSORED)) for t in table[column]), column
```

- [ ] **Step 3: Run the gate**

Run, redirecting to a file (a piped `tail` shows nothing until the end): `.venv/Scripts/python -m pytest tests/bench/test_m4_gate.py -q > <scratchpad>/gate.log 2>&1`
Expected: 9 passed, about 15 minutes (the experiment takes 13.5 of them).

If a check fails, stop. Report the check, the instance, the solver and the encoding. Do not change a tolerance, a budget or a sample cap to make it pass.

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q > <scratchpad>/full.log 2>&1`
Expected: 1314 passed and 1 skipped, about 34 minutes (the M3a gate, the M3b gate and the M4 gate included). `.venv/Scripts/ruff check src tests` clean; `.venv/Scripts/lint-imports` `Contracts: 4 kept, 0 broken.`

- [ ] **Step 5: Commit**

```
test(bench): the M4 gate -- the harness is correct and complete on the M3 ladder
```

```bash
git add experiments/m4-gate.toml experiments/m4-crossover.toml tests/bench/test_m4_gate.py
git commit -F <message file>
```

---

### Task 13: The study, and its result

Spec §10. This task needs the machine for about four days and the owner's go-ahead to start; ask before Step 2. The store is not committed: it is an output, regenerable from the commit, and too large for the repository.

- [ ] **Step 1: Start from a clean tree**

Run: `git status --porcelain --untracked-files=no`
Expected: nothing. The runner refuses a tree whose tracked files differ from HEAD, so every row names a commit that produced it.

- [ ] **Step 2: Run the study** (owner's go-ahead first)

Run, in the background, with output to a file:
`.venv/Scripts/quantic.exe bench run experiments/m4-crossover.toml --store <outside the repo>/m4-crossover.duckdb`
If it stops for any reason, run the same command again; it skips what the store holds. `quantic bench plan experiments/m4-crossover.toml --store <path>` says how much is left — but not while the run holds the store (`StoreBusy`).

- [ ] **Step 3: Report**

Run: `.venv/Scripts/quantic.exe bench report <path>/m4-crossover.duckdb --out <scratchpad>/m4-report`

- [ ] **Step 4: Write the result doc**

Create `docs/superpowers/specs/<date>-m4-crossover-result.md`, in the form of the M3a and M3b result docs: what was run (commit, lock hash, machine, task count, wall-clock, kills, errors), the crossover tables at each budget, the dial effects, the bracket check (expected empty), and what they say. It must address, with the numbers:

- how often each sampler reaches the QUBO's minimum on T0 (`energy_gap_median` in `cells.csv`), per encoding and with CVaR on and off — the dry run saw annealing end 12 to 45 times above it on every CVaR instance;
- the encoding's regret at 8 zeta bits (`encoding_regret`) against M3b's 0.23% to 3.46%;
- Almgren–Chriss's feasibility under D2 (zero in the dry run) and its gap elsewhere;
- CP-SAT against HiGHS: where each is faster, and `T1-D1+D3+D4`;
- what M5 should take from it: the zeta bit budget, which encodings, which samplers to compare a QPU against.

- [ ] **Step 5: Commit the result doc**

```
docs: record the M4 crossover result
```
