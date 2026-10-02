# M4 — The Classical Slate, the Bench Runner and the Results Store

**Status:** Design approved in conversation; awaiting review of this written spec
**Date:** 2026-10-02
**Builds on:** `2026-09-29-m3b-exact-baseline-design.md`, `2026-09-30-m3b-gate-result.md`;
parent spec `2026-09-22-quantum-liquidation-lvar-design.md` §7.1, §7.2, §9, §12, §14
**Gate:** the harness is correct and complete on the 32 T0–T1 instances at
instance seed 0 (§9). Solver *performance* is a finding, never a gate condition.
**Deliverable:** a complete classical crossover table on T0–T1 (parent spec §14),
recorded in a result doc
**Hands off to:** M5 — quantum solvers, embedding cache, quota guard

---

## 1. Purpose

Measure, on T0 and T1, how every free classical solver does on the liquidation
problem, against proven optima, in a store from which every number regenerates.

Parent spec §14: "M4 already produces a publishable result — a classical-only
crossover study across the hardness dials — before a single qubit is touched."
M4 is that result, and the harness M5 plugs its quantum solvers into.

## 2. Scope

**In:**

- The rest of the classical slate behind the existing `Solver` protocol:
  `almgren_chriss`, `cpsat`, `simulated_annealing`, `tabu`, `greedy` (§4).
- The bench runner: experiment config, run keys, a process pool with a single
  DuckDB writer, resume, per-run kill (§5).
- The results store schema (§6).
- Metrics and the crossover report (§7).
- The M3b final-review backlog (§8), done first.

**Out:**

- Figures. M6 plots from the same store.
- Quantum solvers and the QPU budget guard. M5.
- T2 and T3. Their reference is best-known, not a bracket; they enter with M5.
- `gurobi_nonconvex`. It needs a licence (parent spec §7.2, "optional").
- The zeta bit budget for the study. It stays M5's choice; M4's samplers use the
  default 8 bits (§4.3).
- Advisory output (allocation or risk recommendations). The user chose on
  2026-10-02 to keep M4 the benchmark as specced; advisory work is a separate
  vertical with its own spec.

## 3. Decisions

| # | Decision | Chosen | Why |
|---|---|---|---|
| D1 | M4 scope | Benchmark as specced | Advisory output waits until it is known which solvers can be relied on |
| D2 | Runner architecture | Process pool, single writer in the main process | DuckDB has one writer; 16 cores; resume and kill fall out of it |
| D3 | Total wall-clock of a study run | Unlimited | Size is set by the statistics, not by a clock. Each *run* keeps its own budget, which is an experimental variable |
| D4 | Gurobi | Out | Licence |
| D5 | Zeta bits for the samplers | 8 (study default) | Scored against the optimum *and* the QUBO's own minimum, so encoding resolution is separated from sampler failure |
| D6 | Repair of sampler output | None | A sample is decoded and classified as returned. A repaired variant would be a separately named solver |

## 4. The solver slate

Every solver implements `solve(instance, budget) -> Result` from
`solvers/base.py`. `Result.objective` is always `problem.objective.evaluate` at
`Result.schedule`, as in M3b.

### 4.1 On the problem — `solvers/classical/`

**`almgren_chriss.py` — `AlmgrenChriss`.** The closed-form continuous
Almgren–Chriss trajectory per asset, rounded to lots, then a deterministic repair
toward the instance's constraints (full liquidation, cardinality, minimum
participation, blocks). If repair cannot reach a feasible schedule, the result is
`INFEASIBLE` with the rounded schedule and its feasibility report: nothing is
hidden. It never claims optimality and carries no bracket. This is the "what a
desk does today" reference every interesting result must beat.

**`cpsat.py` — `CpSat`.** OR-Tools CP-SAT over the same model as the MILP:

- the lot grid and one binary `y[i,t,k]` per admissible lot value, from
  `lot_grid.py` and `milp_model.columns_of`;
- the per-cell costs from `milp_model.cell_costs`, so no second objective model
  exists to drift;
- variance and CVaR through the same outer-approximation rounds as
  `exact_pwl_milp` (`variance_gradient`, `loss_model`).

CP-SAT is integer-only. Costs are scaled by a factor `S` and rounded. The total
rounding error is bounded by `R = (number of rounded coefficients) * 0.5 / S`
weighted by the largest absolute value each binary can take, and the proven lower
bound is widened by `R` before it becomes a `Bracket`. `S` is chosen so that `R`
is below `1e-7 * |objective scale|`. CP-SAT returns `OPTIMAL` only when the
widened bracket closes within `base._BRACKET_RTOL`.

CP-SAT is a third exact method. It must not import `brute_force` or
`exact_pwl_milp`, and neither may import it (§8, item 8). It may import the
shared model pieces in `milp_model.py` and `lot_grid.py`.

### 4.2 On the QUBO — `solvers/qubo/`

`simulated_annealing.py` (neal), `tabu.py` (dwave-tabu), `greedy.py`
(steepest descent from uniformly random starts, the cheap floor).

All three:

- take an `EncodedInstance` built by the runner (encoding and zeta bits are part
  of the run key) and wrap the `Solver` protocol around it;
- run on the same `Qubo` object, so differences between them are the sampler's,
  not the formulation's;
- report every sample in `Result.detail["samples"]`: bitstring hash, energy,
  decoded objective terms, feasibility class;
- report the best *feasible* decoded schedule as `Result.schedule`, or none, with
  status `TIME_LIMIT` (they never prove optimality);
- record an energy trace, a list of `(seconds_since_start, best_energy)`, in
  `Result.detail["trace"]` — the trace `base.py`'s docstring deferred to M4.

Each sample is decoded with `encoding.decode` and classified by
`problem.feasibility.classify`, never repaired (D6).

### 4.3 The two references for a sampler

At 8 zeta bits, the QUBO's minimum is not the optimum on five of the eight T0 CVaR
cells (M3b gate result, regret 0.23% to 3.46%). A sampler is therefore scored
against:

1. the optimum, through the instance's `Bracket` (the optimality gap of §7);
2. the QUBO's own minimum, where `qubo_minimum` can enumerate it (the regret).

"The sampler failed" and "8 bits cost this much" are different findings, and the
report keeps them apart.

## 5. The bench runner — `bench/`

### 5.1 Experiment config — `bench/config.py`

A TOML file under `experiments/` declares the factor grid:

```toml
[instances]
tiers = ["T0", "T1"]
dials = "all"            # the 16 combinations
seeds = [0, 1, 2, 3, 4]

[solvers.cpsat]
seeds = [0, 1, 2, 3, 4]
budgets_s = [0.1, 1, 10, 60]

[solvers.simulated_annealing]
encodings = ["binary", "one_hot", "domain_wall"]
zeta_bits = 8
seeds = "0..49"
budgets_s = [0.1, 1, 10, 60]
```

`quantic bench plan <config>` expands the grid, prints the run count and an
estimated wall-clock, and runs nothing. `quantic bench run <config> --store
<path>` executes.

### 5.2 Run key

A SHA-256 over: instance hash, solver name, encoding, zeta bits, solver seed,
budget, the git commit, and a hash of `uv.lock`. The same key always means the
same computation. A new commit yields new keys, so rows from old code are never
mixed silently with new ones; the report filters on a commit.

A dirty working tree refuses to start a run unless `--allow-dirty`, which is
recorded in every row it writes.

### 5.3 Execution — `bench/runner.py`

- A process pool of `cpu_count - 2` workers (14 on the 64 GB machine).
- Workers rebuild instances from `(tier, dials, seed)` with
  `problem.generator.generate` and encode locally. Generation is deterministic;
  nothing large crosses a process boundary. Workers cache instances and
  encodings per process.
- Each run has a hard kill at `3 * budget + 30 s`. A killed run is written with
  store status `killed`. `killed` is a store status, not a `Result` status: a
  solver that had to be killed did not answer.
- An exception inside a solver is written as store status `error` with its type
  and message. The runner carries on.
- Only the main process writes to DuckDB, one transaction per completed run.

### 5.4 Resume and interruption

On start the runner reads the keys already in `runs` and skips them. Ctrl-C stops
dispatching, lets in-flight writes finish, and exits. A crash loses only the runs
in flight. Re-running the same command finishes the study.

## 6. The results store — `bench/store.py`

One DuckDB file. Tables:

| Table | One row per | Holds |
|---|---|---|
| `instances` | instance | tier, dials, seed, instance hash, dataset hash, notional, bracket (lower, upper, method), QUBO minimum per (encoding, zeta bits) where enumerable |
| `runs` | run key | the pins of parent spec §9.3 (dataset hash, seeds, package versions, machine fingerprint, penalty, encoding and repair strategy, commit, lock hash, dirty flag); store status; `Result.status`; objective terms in currency and in bps of notional; bracket; feasibility class; full timings |
| `samples` | sampler sample | run key, energy, decoded objective, feasibility class |
| `traces` | trace point | run key, seconds, best energy |

The store carries a schema version and refuses to open a file of another version.

## 7. Metrics and the crossover report — `bench/metrics.py`, `bench/report.py`

### 7.1 Success

A run succeeds at tolerance ε when its schedule is classified feasible without
repair and its evaluated total is at most `lower + ε * |lower|`, with `lower` the
instance's `Bracket.lower`. ε ∈ {0, 0.1%, 1%, 5%}; ε = 0 means the optimum to the
bracket's own width. An instance whose bracket is open reports its width instead
of a gap; on T0–T1 none is expected (M3b closed all 16 at T1).

### 7.2 Metrics per (instance, solver, encoding, budget), over solver seeds

- **Optimality gap** against `Bracket.lower`: median and interquartile range.
- **Feasibility rate.** For samplers, over all samples, not runs.
- **TTS at 99%:** `t_run * ln(0.01) / ln(1 - p)`, with `p` the fraction of seeds
  that succeed and `t_run` the measured wall-clock. `p = 1` gives `t_run`. `p = 0`
  is reported as censored, `> budget * seeds`, never as infinity or an invented
  number. Bootstrap confidence intervals over seeds.
- **Cost in bps of notional** of the best feasible schedule.
- **Regret against the QUBO's minimum**, samplers only (§4.3).
- **Dynamic-range ratio**, from the encoder's diagnostics.

### 7.3 Report

`quantic bench report <store> [--commit <sha>]` writes CSV and a Markdown
rendering:

- **Crossover table**, one per budget: rows are tier × dial combination, columns
  are solvers (each sampler at its best encoding, named in the cell); a cell shows
  median gap / feasibility rate / TTS at ε = 1%.
- **Dial effects:** each dial's main effect and the pairwise interactions on each
  solver's gap. Sixteen combinations is the full 2⁴ factorial, so parent spec
  §9.2's screening is computed exactly.
- **Bracket cross-check:** every instance where the CP-SAT and MILP brackets do
  not overlap, which should be an empty list.

## 8. The M3b backlog — done first

The bench trusts `Result` and `Bracket`, so these land before any new solver.

**Fixed:**

1. `Result` enforces `OPTIMAL`: a bracket with width within `_BRACKET_RTOL` and a
   schedule that `classify` calls feasible, not merely both present.
2. The `TIME_LIMIT` path drops the `min(proven, total)` clamp; a lower bound above
   the incumbent raises, as `Bracket` already does.
3. `qubo_minimum` raises a typed `EnumerationTooLarge`, not the built-in
   `MemoryError`. The runner catches errors by type.
4. The witness threshold widens from `1e-9` to `1e-6` (measured margin ≥ 5e-3).
5. The presolve retry gets the remaining time, not the original limit.
6. `detail["rounds"]` stops counting a round that never ran.
7. scipy status 1 is `TIME_LIMIT` only when time ran out; otherwise it is recorded
   as an iteration or node limit in `detail`.
8. An import-linter contract keeps `brute_force`, `exact_pwl_milp` and `cpsat`
   mutually independent.
9. Duplicate `MinParticipation` or `BlockTrades` descriptors raise instead of
   overwriting.
10. The M3b result doc's stale "go to the final review" bullet is corrected.

**Recorded, not fixed** (none changes an M4 result):

- `lot_grid` takes block lots unclamped to the cap. This is correct; its docstring
  will say so.
- `max_single_cell_gain` is per cell.
- The zeta bit budget (M5).

**Closed as a side effect:** "nothing checks HiGHS's lower bound at T1". CP-SAT's
independent bracket on the same instances checks it (§7.3, §9 check 2).

## 9. The gate — `tests/bench/test_m4_gate.py`

On the 32 T0–T1 instances at instance seed 0. Each check is about correctness or
harness integrity; none asserts that a solver is fast or good.

1. Every solver's `Result.objective` equals `evaluate(Result.schedule)`, and every
   `OPTIMAL` result meets §8 item 1.
2. CP-SAT's bracket overlaps the MILP's on all 32 instances; on T0 both equal
   `brute_force`.
3. For every sampler sample, the reported energy equals the QUBO energy of its
   bitstring, and for feasible samples the decoded objective equals the
   evaluator's.
4. Sanity: at the largest budget, simulated annealing reaches the enumerated QUBO
   minimum on every T0 instance for at least one seed. On about 24 variables plus
   auxiliaries, never reaching it means a bug, not weak annealing.
5. Determinism: re-running a key reproduces its row, timings excepted.
6. Resume: a run interrupted midway and restarted yields the same store as an
   uninterrupted one, timings excepted.
7. Kill: a deliberately hanging test solver yields a `killed` row and the run
   carries on.
8. The report has a value or an explicit censored marker in every
   (tier, dial, solver) cell.

Unit tests come first for each component (TDD): each solver against the
evaluator on T0; CP-SAT's rounding bound as a Hypothesis property (the widened
lower bound never exceeds the true optimum); Almgren–Chriss rounding and repair
on hand-built cases; TTS at `p = 0`, `p = 1` and a known binomial case; store
round-trip and run-key stability across processes.

## 10. The study run

`experiments/m4-crossover.toml`, starting size:

| Setting | Value |
|---|---|
| Instances | T0, T1 × 16 dials × 5 seeds = 160 |
| Samplers | SA, tabu, greedy × 3 encodings × 50 seeds |
| Exact | CP-SAT and the MILP × 5 seeds; `almgren_chriss` once (deterministic) |
| Budgets | 0.1, 1, 10, 60 s |

Rough estimate: 4–5 days on 14 workers, dominated by the samplers. `bench plan`
prints the measured estimate before anything runs. There is no total wall-clock
limit (D3). The table and its reading go to
`docs/superpowers/specs/<date>-m4-crossover-result.md`.

## 11. Dependencies

Added to `pyproject.toml`: `duckdb`, `ortools`, `dwave-neal`, `dwave-tabu`,
`dimod`. HiGHS stays reached through `scipy.optimize.milp`, as in M3b.

## 12. File structure

```
src/quantic/
  solvers/
    base.py                     §8 items 1–2
    classical/
      almgren_chriss.py         new
      cpsat.py                  new
      exact_pwl_milp.py         §8 items 2, 5, 6, 7
      qubo_minimum.py           §8 items 3, 4
    qubo/
      __init__.py
      sampler.py                shared: decode, classify, samples, trace
      simulated_annealing.py
      tabu.py
      greedy.py
  bench/
    config.py
    keys.py
    runner.py
    store.py
    metrics.py
    report.py
  cli.py                        `bench plan | run | report`
experiments/
  m4-crossover.toml
  m4-gate.toml
tests/
  solvers/  (unit tests per new solver)
  bench/    (unit tests, test_m4_gate.py)
```

Import contracts: `bench` already sits above `solvers` and `encoding` in the
layers contract, and is the only source package that calls `problem.generator`.
The "solvers never build the problem" contract is unchanged. `cpsat` and
`almgren_chriss` join the "exact baseline does not read the encoder" contract;
`solvers/qubo/` is the only part of `solvers` that reads `encoding`.

### Deviation from the parent spec

Parent spec §12 lists `simulated_annealing`, `tabu` and `greedy` under
`solvers/classical/`. M4 puts them in `solvers/qubo/`: they share a decode,
classify and sample-recording layer the problem-level solvers do not, and the
encoder contract above is then a statement about a directory rather than a list
of files. M5's QPU and hybrid samplers sit in the same layer.

## 13. Risks

| Risk | Mitigation |
|---|---|
| CP-SAT's integer scaling loses precision on wide-ranged costs | Rounding bound widens the lower bound; property-tested; `OPTIMAL` only on a closed widened bracket |
| A pure-Python sampler ignores its budget and stalls the study | Hard kill per run, recorded as `killed` |
| A multi-day run dies midway | Resume by run key |
| Rows from different code versions mixed in one table | Commit and lock hash in the run key; dirty tree refused by default |
| Greedy or tabu dominate SA and the study reads as "annealing loses" | That is a finding. The report states it against proven brackets, which is the point of M4 |

## 14. What M5 picks up

- The `Solver` protocol and the `qubo/sampler.py` scaffolding: a QPU or hybrid
  sampler is one more `qubo/` module, scored identically.
- The store and report unchanged; M5 adds QPU-access time to timings, which
  parent spec §9.1 requires next to the end-to-end wall-clock.
- The zeta bit budget, chosen with the M4 regret numbers in hand.
