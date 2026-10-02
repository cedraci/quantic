# M4 — The Classical Slate, the Bench Runner and the Results Store

**Status:** Approved 2026-10-02; revised the same day by the plan's dry run (§15).
Plan at `docs/superpowers/plans/2026-10-02-m4-classical-bench.md`
**Date:** 2026-10-02
**Builds on:** `2026-09-29-m3b-exact-baseline-design.md`, `2026-09-30-m3b-gate-result.md`;
parent spec `2026-09-22-quantum-liquidation-lvar-design.md` §7.1, §7.2, §9, §12, §14
**Gate:** the harness is correct and complete on the 32 T0–T1 instances of the
M3 ladder, instance seed 2026 (§9). Solver *performance* is a finding, never a gate condition.
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
| D7 | Status of a method that never proves | New `Result` status `HEURISTIC` | `TIME_LIMIT` says more time could have produced a proof; a sampler or a rounding never could |

## 4. The solver slate

Every solver implements `solve(instance, budget) -> Result` from
`solvers/base.py`. `Result.objective` is always `problem.objective.evaluate` at
`Result.schedule`, as in M3b.

### 4.1 On the problem — `solvers/classical/`

**`almgren_chriss.py` — `AlmgrenChrissSolver`.** The closed-form continuous
Almgren–Chriss trajectory per asset: linear temporary impact, the name's own
price variance, holdings `X sinh(kappa (T - t)) / sinh(kappa T)` with
`cosh kappa = 1 + lam sigma**2 / (2 k)`. The formula knows variance only, so
under CVaR it gets no risk aversion and trades a straight line. Each name's
trajectory is rounded to the admissible row of the lot grid nearest it in summed
absolute lots (a small dynamic programme), which respects the cap, the venue
minimum and the blocks. Rows are chosen name by name, so a cardinality limit can
be breached: the dry run found it breached on all sixteen D2 instances at T0–T1.
The result says so through its feasibility report and is never repaired further.
Status is always `HEURISTIC`, with no bracket. This is the "what a desk does
today" reference every interesting result must beat.

**`cpsat.py` — `CpSat`.** OR-Tools CP-SAT over the same model as the MILP:

- the lot grid and one binary `y[i,t,k]` per admissible lot value, from
  `lot_grid.py` and `milp_model.columns_of`;
- the per-cell costs from `milp_model.cell_costs`, so no second objective model
  exists to drift;
- variance and CVaR through the same outer-approximation rounds as
  `exact_pwl_milp` (`variance_gradient`, `loss_model`).

CP-SAT is integer-only. Every real number is made an integer by rounding in the
direction that keeps the integer program a **relaxation**: cost coefficients are
floored, the right-hand side of each `>=` row is floored, a coefficient that is
subtracted is ceiled. Each auxiliary (the variance `theta`, CVaR's `w` and `v_s`)
is the ceiling of its true value at the true optimum, which costs at most one tick
each. So `optimum >= (CP-SAT's bound - slack) / ticks + constant`, with `slack`
one tick per auxiliary, and that is `Bracket.lower`. Ticks are `2**30` per unit of
the witness's objective. Under CVaR the weights `w_s / (1 - alpha)` are moved into
the scenario rows, each row scaled by the largest power of two (at most `2**40`)
that keeps it under `2**60`, so the objective's coefficients stay exactly one.
CP-SAT runs on one worker with the run's seed and starts from the witness as a
hint. When CP-SAT proves its integer program but the rounding leaves the bracket
wider than `base._BRACKET_RTOL`, the status is `HEURISTIC`, not `OPTIMAL`.

Measured in the dry run on the 32 gate instances: every CP-SAT bracket overlaps
the MILP's, the optimum agrees where both finish, the widest closed CVaR bracket
is `2.2e-7` relative, and one instance (`T1-D1+D3+D4`) does not finish in 20 s
where HiGHS takes 0.3 s. That last is a finding, not a defect.

CP-SAT is a third exact method. It must not import `brute_force` or
`exact_pwl_milp`, and neither may import it (§8, item 8). It may import the
shared model pieces in `milp_model.py` and `lot_grid.py`.

### 4.2 On the QUBO — `solvers/qubo/`

`simulated_annealing.py`, `tabu.py` and `greedy.py` (steepest descent from
uniformly random starts, the cheap floor), all from `dwave-samplers`, which
replaces the separate `dwave-neal` and `dwave-tabu` packages. Tabu's own stopping
rule is a wall clock per read, which would make its samples depend on machine
load; each read is bounded by 10 restarts instead.

All three:

- take an `EncodedInstance` built by the runner (encoding and zeta bits are part
  of the run key) and wrap the `Solver` protocol around it;
- run on the same `Qubo` object, so differences between them are the sampler's,
  not the formulation's;
- draw batches of 1, 2, 4, ... up to 64 reads, each seeded from
  `(seed, batch index)` and kept below `2**31`, so the sample stream is fixed by
  the seed and a run stopped by the clock returns a prefix of a longer run's; a
  run always draws at least one batch;
- report every distinct bitstring once, with its count, in
  `Result.detail["samples"]`: digest, count, the sampler's energy, feasibility
  class, and the evaluator's total when feasible;
- report the largest relative difference between the sampler's energies and
  `Qubo.energy` as `detail["energy_mismatch"]`;
- report the best *feasible* decoded schedule as `Result.schedule`, or none, with
  status `HEURISTIC` (they never prove optimality);
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
seeds = "2026..2030"     # 2026 is the M3 gates' ladder

[references]
qubo_minimum_tiers = ["T0"]

[solvers.cpsat]
seeds = [0, 1, 2, 3, 4]
budgets_s = [0.1, 1, 10, 60]

[solvers.simulated_annealing]
encodings = ["binary", "one_hot", "domain_wall"]
zeta_bits = 8
seeds = "0..49"
budgets_s = [0.1, 1, 10, 60]
```

`max_samples` may cap a sampler's runs as well. Unknown keys are refused. A
solver may also be named by import path, `"package.module:Class"`, which the
tests use for solvers that hang or raise on purpose.

`quantic bench plan <config>` expands the grid, prints the task count and an
upper estimate of wall-clock (every budget spent in full), and runs nothing.
`quantic bench run <config> --store <path>` executes.

The instance parameters are the synthetic ones the M3 gates use, moved from
`tests/ladder.py` to `bench/instances.py`. Calibrating from a real bundle is
`problem.from_bundle`'s job and is not part of M4.

### 5.2 Run key

A SHA-256 over: the instance's coordinates (tier, dial combination, instance
seed), solver name, encoding, zeta bits, solver seed, budget, sample cap, the git
commit, and a hash of `uv.lock`. The commit pins coordinates to one instance; the
instance's content hash is written on the row. The same key always means the
same computation. A new commit yields new keys, so rows from old code are never
mixed silently with new ones; the report filters on a commit.

A working tree whose tracked files differ from HEAD refuses to start a run
unless `--allow-dirty`, which is recorded in every row it writes. Untracked files
do not count, so a store file inside the checkout does not block its own run.

### 5.3 Execution — `bench/runner.py`

- `cpu_count - 2` long-lived worker processes (14 on the 64 GB machine), each
  with its own pipe, started with `spawn`. `concurrent.futures` cannot stop one
  task without stopping the pool, and terminating a process can corrupt a queue
  it shares, so each worker's pipe is its own and is discarded with it.
- Workers rebuild instances from `(tier, dials, seed)` with
  `problem.generator.generate` and encode locally. Generation is deterministic;
  nothing large crosses a process boundary. Workers cache instances, and the
  eight most recent encodings, per process.
- Each run has a hard kill at `3 * budget + 30 s`; a reference, which is
  unbudgeted, at one hour. A killed run is written with
  store status `killed`. `killed` is a store status, not a `Result` status: a
  solver that had to be killed did not answer.
- An exception inside a solver is written as store status `error` with its type
  and message. The runner carries on.
- Only the main process writes to DuckDB, one transaction per completed run.

### 5.4 Resume and interruption

On start the runner reads the keys already in `runs` and skips them. Each
finished task is written in its own transaction as it arrives, so Ctrl-C or a
crash loses only the tasks in flight; workers are terminated. Re-running the same
command finishes the study.

## 6. The results store — `bench/store.py`

One DuckDB file. Tables:

| Table | One row per | Holds |
|---|---|---|
| `meta` | setting | the schema version |
| `runs` | task key | the task's coordinates and pins (instance hash, notional, commit, lock hash, dirty flag, machine, package versions); store status (`ok`, `error`, `killed`) and error text; `Result.status`; schedule and zeta; objective terms, total, total in bps of notional; bracket; feasibility and violations; for samplers the sample counts, best energy and energy mismatch; for QUBO-minimum references the minimum energy and the decoded schedule's total; dynamic-range ratio and variable count; timings; detail |
| `samples` | distinct bitstring per sampler run | task key, digest, count, the sampler's energy, feasibility class, evaluator total |
| `traces` | trace point | task key, seconds, best energy |

References are rows of `runs` with `kind = "reference"`, one task each, so they
are written, resumed and keyed exactly like solver runs:
`ref:exact_pwl_milp` (unbudgeted, the bracket every gap is measured against),
`ref:brute_force` (T0 only), and `ref:qubo_minimum` per encoding on the tiers
named in `[references]`. The parent spec's dataset hash is the instance hash
here: the parameters are synthetic and the instance carries them.

The store carries a schema version and refuses to open a file of another version.

## 7. Metrics and the crossover report — `bench/metrics.py`, `bench/report.py`

### 7.1 Success

A run succeeds at tolerance ε when its schedule is classified feasible without
repair and its evaluated total is at most
`upper + max(ε * |lower|, 1e-9 * |upper|)`, with `[lower, upper]` the reference
bracket. ε ∈ {0, 0.1%, 1%, 5%}; ε = 0 means the optimum to round-off. An instance whose bracket is open reports its width instead
of a gap; on T0–T1 none is expected (M3b closed all 16 at T1).

### 7.2 Metrics per (tier, dials, solver, encoding, budget), over solver and instance seeds

- **Optimality gap** against the reference `lower`: median and interquartile
  range. Round-off within `1e-9` either side of zero is reported as zero; a
  larger negative gap is kept, because it means a bound is wrong.
- **Feasibility rate.** For samplers, over all samples, not runs.
- **TTS at 99%:** `t_run * ln(0.01) / ln(1 - p)`, with `p` the fraction of seeds
  that succeed and `t_run` the measured wall-clock. `p = 1` gives `t_run`. `p = 0`
  is reported as censored, `> budget * seeds`, never as infinity or an invented
  number. Bootstrap confidence intervals over seeds.
- **Cost in bps of notional** of the best feasible schedule.
- **Against the QUBO's minimum**, samplers only (§4.3): the energy gap of the
  best sample to the enumerated minimum, and the encoding's regret, the decoded
  minimum's true total against the reference `lower`.
- **Dynamic-range ratio**, from the encoder's diagnostics.

### 7.3 Report

`quantic bench report <store> --out <dir> [--commit <sha>]` writes CSV and a
Markdown rendering. A store holding rows from more than one commit is refused
unless `--commit` picks one. Besides the tables below, `cells.csv` holds every
metric per cell.

- **Crossover table**, one per budget: rows are tier × dial combination, columns
  are solvers (each sampler at its best encoding, named in the cell); a cell shows
  median gap / feasibility rate / TTS at ε = 1%.
- **Dial effects:** each dial's main effect and the pairwise interactions on each
  solver's median gap, at the largest budget. Sixteen combinations is the full 2⁴ factorial, so parent spec
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
7. scipy status 1 is `TIME_LIMIT` only when a wall clock was set. No iteration or
   node limit is ever set, so status 1 without a clock raises rather than being
   recorded under a label that hides it.
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

On the 32 T0–T1 instances of the M3 ladder, instance seed 2026, from
`experiments/m4-gate.toml`. Samplers there are capped by samples (256 for
annealing and steepest descent, 32 for tabu), not by the clock, so every row is
reproducible. Each check is about correctness or harness integrity; none asserts
that a solver is fast or good. The dry run measured the gate experiment at 1,248
tasks in 13.5 minutes on 14 workers.

1. Every stored objective equals the evaluator's at the stored schedule
   (`1e-9`), and every `OPTIMAL` row has a closed bracket and a feasible
   schedule.
2. CP-SAT's bracket overlaps the reference MILP's on all 32 instances; at T0 every
   `OPTIMAL` total equals brute force's.
3. Every sampler run's energies equal `Qubo.energy` of the bits it returned, to
   `1e-6` relative: M3b's tolerance for one energy accumulated in two orders,
   here at dynamic ranges up to `1e10` (measured worst `6.9e-7`). A sampler that
   read the bits in another order would fail this.
4. The sampler pipeline returns the QUBO's minimum intact: fed the enumerated
   minimum of every T0 instance under every encoding, it reports that energy and,
   when the minimum decodes to a feasible schedule, that schedule's true total.
   Whether a sampler *finds* the minimum is not gated (§15).
5. Determinism: re-executing one task per (tier, solver) reproduces its row,
   timings excepted.
6. Resume: a small experiment stopped after four tasks and restarted yields the
   same store as an unbroken run, timings excepted.
7. Kill: a deliberately hanging solver yields a `killed` row and the run carries
   on.
8. The report has a value or an explicit censored marker in every
   (tier, dial, solver) cell of every crossover table.

Unit tests come first for each component (TDD): each solver against brute force
on the shared small instances; CP-SAT's scaled bound as a Hypothesis property (it
never exceeds the true optimum); the Almgren–Chriss trajectory against an exact
solve of the continuous problem, and its rounding against every row of the grid;
TTS at `p = 0`, `p = 1` and a known success rate; store round-trip, refusal of a
busy store, and run-key stability across processes.

## 10. The study run

`experiments/m4-crossover.toml`, starting size:

| Setting | Value |
|---|---|
| Instances | T0, T1 × 16 dials × seeds 2026–2030 = 160 |
| Samplers | SA, tabu, greedy × 3 encodings × 50 seeds |
| Exact | CP-SAT × 5 seeds; the MILP × 1 seed (it has no seed); `almgren_chriss` once |
| Budgets | 0.1, 1, 10, 60 s (Almgren–Chriss: one, it ignores budgets) |

Upper estimate: about 4.3 days on 14 workers, dominated by the samplers.
`bench plan` prints it before anything runs. There is no total wall-clock limit
(D3). The table and its reading go to
`docs/superpowers/specs/<date>-m4-crossover-result.md`.

## 11. Dependencies

Added to `pyproject.toml`: `duckdb`, `ortools`, `dwave-samplers`, `dimod`. All
four install from wheels on Windows under Python 3.11 (checked 2026-10-02:
duckdb 1.5.6, ortools 9.15.6755, dwave-samplers 1.8.0, dimod 0.12.22). HiGHS
stays reached through `scipy.optimize.milp`, as in M3b.

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
      lot_grid.py               §8 item 9
  bench/
    instances.py                the ladder's parameters and per-tier generation
    config.py
    keys.py
    store.py
    execute.py                  one task in a worker, into rows
    pool.py                     killable worker processes
    runner.py
    metrics.py
    report.py
  cli.py                        `bench plan | run | report`
experiments/
  m4-crossover.toml
  m4-gate.toml
tests/
  ladder.py                     imports its parameters from bench/instances.py
  solvers/  (unit tests per new solver)
  bench/    (unit tests, stand_ins.py, test_m4_gate.py)
```

Import contracts: `bench` already sits above `solvers` and `encoding` in the
layers contract, and is the only source package that calls `problem.generator`.
The "solvers never build the problem" contract is unchanged. `cpsat` and
`almgren_chriss` join the "exact baseline does not read the encoder" contract,
and a new independence contract keeps `brute_force`, `exact_pwl_milp` and
`cpsat` from importing one another.

### Deviation from the parent spec

Parent spec §12 lists `simulated_annealing`, `tabu` and `greedy` under
`solvers/classical/`. M4 puts them in `solvers/qubo/`: they share a decode,
classify and sample-recording layer the problem-level solvers do not, and the
encoder contract above is then a statement about a directory rather than a list
of files. M5's QPU and hybrid samplers sit in the same layer.

## 13. Risks

| Risk | Mitigation |
|---|---|
| CP-SAT's integer scaling loses precision on wide-ranged costs | The integer program is a relaxation by construction; property-tested; `OPTIMAL` only on a closed bracket, `HEURISTIC` otherwise; a program that cannot fit in 64 bits raises `ScaleOverflow` |
| A store read while a run writes it | DuckDB allows one process; `Store` raises `StoreBusy` saying so |
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

## 15. Changed by the plan's dry run (2026-10-02)

The plan was written from code that was run, in a throwaway worktree, against
`bfd5ca5`. What the run changed in this spec:

- **`HEURISTIC` status (D7).** Almgren–Chriss, the samplers, and CP-SAT when
  rounding leaves its bracket open, report `HEURISTIC`; the spec had them as
  `INFEASIBLE` or `TIME_LIMIT`, which claim things they cannot.
- **CP-SAT's bound (§4.1).** The "widen by R" sketch became a relaxation built
  by directed rounding. A fixed CVaR row scale of `2**30` overflowed 64 bits on
  some instances and `2**20` left brackets open by `1e-6`; the scale is now
  chosen per row.
- **Almgren–Chriss (§4.1).** Rounded to the nearest admissible row per name; it
  breaches cardinality on all sixteen D2 instances, which the table will show.
- **Samplers (§4.2).** `dwave-samplers` replaces `dwave-neal` and `dwave-tabu`;
  tabu is bounded by restarts, not its wall-clock timeout; the annealer refuses
  seeds of `2**31` and above, so batch seeds are masked to 31 bits.
- **Store (§6).** References are rows of `runs`, not a separate `instances`
  table; samples are stored once per distinct bitstring with a count.
- **Run key (§5.2).** Instance coordinates, not the instance hash, enter the key;
  the hash is on the row.
- **Gate (§9).** Instance seed 2026 (the M3 ladder), not 0. Check 3's tolerance
  is M3b's `1e-6`, measured worst `6.9e-7`. **Check 4 was replaced, at the
  owner's decision:** the spec's premise of "about 24 variables" holds only
  without CVaR. With the gadget, T0 QUBOs have 260–296 variables and dynamic
  ranges near `1e7`, and in the dry run simulated annealing ended 12 to 45 times
  above the minimum on every CVaR instance; even on the small one-hot QUBOs no
  sampler reached it on three instances in about 2,000 reads. That is sampler
  capability, a finding for the result doc, so the gate now checks that the
  pipeline credits the minimum correctly when given it.
- **Review focus.** A store opened by another process raises `StoreBusy` (the
  operating system's message is localised on Windows, so the test is that the
  file exists and cannot be opened); a missing reference gave a NaN success rate
  and is now no success.
