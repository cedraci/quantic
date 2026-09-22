# Quantic — Quantum vs Classical Benchmark Platform for Combinatorial Liquidation and L-VaR

**Status:** Design approved, pending implementation plan
**Date:** 2026-09-22
**Vertical:** B — Combinatorial liquidation / L-VaR (first of three planned verticals)

---

## 1. Purpose

Build a research and benchmarking platform that answers one question rigorously:

> **Where, as a function of problem structure and size, do quantum and hybrid solvers compete with the best free classical solvers on realistic multi-asset discrete liquidation problems?**

The deliverable is an *honest crossover study*. A negative result — "quantum does not compete in any regime we can reach" — is a valid and publishable outcome, and the platform is designed so that outcome is as well-evidenced as a positive one.

Success criteria:

1. A reproducible benchmark harness where any published figure regenerates exactly from a pinned dataset hash, seed set, and environment fingerprint.
2. A QUBO/Ising encoding of the liquidation problem whose correctness is *proven* against exact classical optima on small instances, for every combination of hardness features.
3. A crossover table spanning classical, hybrid, and real-QPU solvers across a defined instance ladder, reporting solution quality, feasibility rate, and time-to-target.
4. Results expressed in basis points of notional execution cost, so they are legible to practitioners and not only to physicists.

## 2. Scope and Non-Goals

### In scope

- Microstructure analytics core: order book reconstruction, liquidity metrics, market impact model calibration, cross-asset covariance estimation.
- Discrete multi-asset liquidation problem with four composable sources of combinatorial hardness.
- QUBO/Ising encoding with pluggable integer encodings and first-class penalty derivation.
- Solver layer spanning exact classical, classical metaheuristic, quantum annealing, hybrid, and gate-based (QAOA).
- Benchmark harness, metrics, results store, publication-quality figures.
- Local Streamlit dashboard for browsing runs.

### Explicitly out of scope (deferred to later specs)

- **Vertical A — Multi-asset optimal routing.** Shares the QUBO solver substrate but requires multi-venue fee, rebate, queue, and latency modelling. Separate spec.
- **Vertical C — High-dimensional non-linear PDE** (Sircar–Papanicolaou multiscale stochastic volatility, multi-asset derivatives). Different mathematics, different quantum algorithms, no viable free-tier QPU path today. Its honest framing is tensor-network / quantum-inspired methods benchmarked against deep BSDE solvers. Separate spec.
- Live trading, order management, or exchange connectivity. This is a research platform, not an execution system.
- Production-grade latency engineering.

### Non-goals within vertical B

- Proving quantum advantage. The platform must be capable of *detecting* it if present, and equally capable of demonstrating its absence.
- Supporting paid QPU access as a requirement. Paid backends may be added, but every core result must be reproducible on free tiers.

## 3. Context and Hard Constraints

### 3.1 Quantum backend access

No existing accounts. The platform targets free tiers only:

| Backend | Free allowance | Role |
|---|---|---|
| D-Wave Leap developer tier | ~1 min/month Advantage2 QPU; unlimited local classical solvers | Primary real-QPU target. Native QUBO/Ising fit. |
| D-Wave Leap Hybrid | ~20 min/month | Large-instance tier (T3). CQM sampler accepts constraints natively. |
| IBM Quantum Open | ~10 min/month on 100+ qubit devices | QAOA on hardware, small instances only. |
| Qiskit Aer / PennyLane Lightning | Unlimited, local | Exact statevector to ~30 qubits; MPS beyond. |
| Azure Quantum | ~$500 first-time credits per provider | Optional one-shot trapped-ion comparison point, spent only on a finished experiment. |

**The ~1 minute/month real QPU budget is an architectural constraint, not an operational detail.** See section 7.5.

### 3.2 Hardware sizing ceiling

The liquidation QUBO is **dense** — the risk term couples nearly all variables. Dense (clique) minor-embedding limits:

- D-Wave Advantage2, Zephyr topology: **~230 logical variables**
- D-Wave Advantage, Pegasus topology: ~180 logical variables
- IBM Open + QAOA, realistically meaningful: ~20–30 variables
- Exact classical ground truth: ~30–40 variables by brute force, further by MILP

Variable count is `N_assets × T_buckets × B_bits` plus auxiliaries. This directly determines the instance ladder in section 5.5.

### 3.3 Market data

A market data database exists on a separate machine and can be fed with specific data on request. **Assumption: it is not directly network-reachable from this project.** Data therefore arrives as exports, and the project consumes them through a versioned, content-hashed bundle format (section 8).

Available granularities: L1 TAQ, L2 depth snapshots, L3 message-level, and daily OHLCV.
First benchmark universe: **single-market equities.**

### 3.4 Local compute

Workstation, 32+ cores, 64GB+ RAM. Sufficient for statevector simulation to ~32 qubits, long classical baseline runs, and large scenario sets for CVaR.

## 4. The Central Design Problem

**The textbook discrete Almgren–Chriss problem is convex.** Quadratic temporary impact plus quadratic risk with a PSD covariance is a convex MIQP, which commercial and open solvers solve to proven optimality at the sizes reachable here. Benchmarking a quantum annealer against that produces a foregone conclusion.

**Hardness must therefore come from the microstructure, not from the size.** Four **dials** are defined, each realistic and each destroying convexity or adding combinatorial structure. A dial is a single on/off switch in the experiment design; note that dial D2 governs two related constraint modules together, since cardinality without a minimum lot size is not meaningful on a real desk.

| Dial | Lives in | Realism | Source of hardness |
|---|---|---|---|
| **D1** — square-root / concave impact law (vs. quadratic) | Objective, impact term | Empirically correct impact scaling; exponent calibrated per asset from L2/L3 | Concave objective, so minimisation is NP-hard; MIQP solvers lose their bound |
| **D2** — discrete participation: cardinality **and** minimum lot | Constraints (`cardinality`, `min_participation`) | "Touch at most K names per bucket" plus venue minimum quantities — both standard desk practice | Combinatorial with no convex relaxation; semi-continuous variables, so big-M binaries |
| **D3** — L-VaR as CVaR (vs. variance) | Objective, risk term | The actual liquidity-adjusted risk metric of interest | Scenario-based big-M; scales badly classically |
| **D4** — all-or-nothing block trades | Constraints (`block_trades`) | Block and dark crossing is discrete by construction | Pure binary structure |

All four dials are **independently togglable**. The 2^4 = 16 combinations form a factor in the experiment design: the study measures *which* hardness features move the crossover, not merely whether a crossover exists.

## 5. Domain Model

### 5.1 Decision variables

`x[i,t]` = lots of asset *i* sold in bucket *t*, for assets *i* = 1..N and buckets *t* = 1..T.

Remaining holding: `h[i,t] = X[i] − sum over s<=t of x[i,s]`, where `X[i]` is the initial position in lots.

### 5.2 Objective

Four additive terms, each an ablatable module:

1. **Spread cost** — `sum over (i,t) of (s_i / 2) * x[i,t]`, linear, from L1 quotes.
2. **Temporary impact** — `sum over (i,t) of sigma_i * Y_i * (v[i,t] / V_i)^delta_i`, where `v[i,t]` is traded notional volume and `V_i` bucket volume. **`delta_i` and `Y_i` are calibrated per asset from L2/L3 data, not assumed.** `delta ≈ 0.5` recovers the square-root law. This term supplies the concavity.
3. **Permanent impact** — linear in cumulative traded volume, preserving the standard no-arbitrage structure.
4. **Risk** — either variance `lambda * sum over t of tau * h_t' * Sigma * h_t`, or CVaR per section 5.4.

### 5.3 Base constraints

- **Full liquidation:** `sum over t of x[i,t] = X[i]` for all *i*.
- **Non-negativity:** `x[i,t] >= 0`, implicit in the binary expansion.
- **Integrality:** `x[i,t]` in whole lots.

### 5.4 L-VaR via CVaR (Rockafellar–Uryasev)

`CVaR_alpha = min over zeta of { zeta + 1/((1 − alpha) * S) * sum over s of max(L_s − zeta, 0) }`

where `L_s` is realised liquidation shortfall in scenario *s*. In QUBO form this costs a binary-expanded `zeta` plus one auxiliary variable per scenario, so *S* directly inflates the variable count.

**Scenario reduction is therefore a required preprocessing step**: generate S_raw ≈ 2000 price/liquidity paths, cluster to S ≈ 20–30 representative scenarios with probability weights. The reduction error (Wasserstein distance between the raw and reduced distributions, and the resulting CVaR discrepancy) is computed and reported alongside every CVaR-enabled result.

### 5.5 Instance ladder

| Tier | Shape (N × T × B) | Approx. variables | Ground truth | Runs on |
|---|---|---|---|---|
| **T0 — validate** | 3 × 4 × 2 | ~24 + aux | Brute force **and** exact MILP | Everything |
| **T1 — QPU-native** | 5 × 8 × 3 | ~120–180 | Exact MILP bracket | Real Advantage2 QPU |
| **T2 — QPU-edge** | 8 × 8 × 3 | ~190–250 | Best-known | Borderline embedding |
| **T3 — hybrid-only** | 30 × 12 × 4 | ~1,500+ | Best-known | Leap Hybrid, classical |

## 6. Encoding to QUBO / Ising

### 6.1 Pluggable integer encodings

The mapping from integer `x[i,t]` to binaries is a strategy, selectable per run, and **a study dimension in its own right**:

- **Binary (log) encoding** — `x = sum over b of 2^b * z_b`. Minimal variable count, largest coupler magnitudes.
- **One-hot** — precision-friendly, expensive in variables.
- **Domain-wall encoding** (Chancellor) — one fewer variable than one-hot, strictly local couplings, empirically superior on annealers.

Comparing these on a realistic finance problem is itself a contribution; the literature has not published it.

### 6.2 Keeping the hard constraints quadratic

**Minimum participation** (`x` is either 0 or at least `m_i`) naively requires `y * (m + sum over b of 2^b * z_b)`, which is cubic. The formulation used instead:

```
x[i,t] = m_i * y[i,t] + sum over b of 2^b * z[i,t,b]
penalty: sum over b of z[i,t,b] * (1 − y[i,t])
```

The penalty forces every `z` to zero when `y = 0`. All terms remain quadratic.

**Cardinality** (`sum over i of y[i,t] <= K`) uses binary slack:

```
penalty: (sum over i of y[i,t] + sum over j of 2^j * s_j − K)^2
```

**Block trades** introduce binaries `b_k` per indivisible block, contributing their fixed quantity to the full-liquidation constraint.

Any residual higher-order terms are quadratised via `dimod.make_quadratic`, and the encoder asserts a quadratic result before emitting.

### 6.3 Penalty derivation — a first-class component

`encoding/penalty.py` is not a helper. Incorrect penalty coefficients are the single most common cause of invalid quantum benchmark results:

- Too small: constraint-violating solutions score as cheap and are reported as wins.
- Too large: the penalty term swamps the objective's dynamic range. On real hardware the objective differences fall below the analogue precision floor, and the annealer effectively solves a feasibility problem while being reported as an optimiser.

The module provides two strategies, both recorded in the results row:

1. **Derived bounds** — penalty exceeds the maximum objective improvement obtainable from a one-unit constraint violation, computed from problem structure.
2. **Adaptive search** — increase penalties until the feasibility rate on a reference sampler crosses a threshold.

### 6.4 Dynamic-range diagnostic

D-Wave couplers carry roughly 4–5 bits of effective precision. Every encoded instance therefore records:

```
dynamic_range_ratio = max|coefficient| / (minimum meaningful objective gap)
```

The runner emits a warning when this exceeds the hardware precision budget, and the value is stored on every results row. No result is interpretable without it.

### 6.5 Decoding and feasibility policy

`encoding/decode.py` classifies every returned sample as **feasible**, **repairable**, or **rejected**.

- Repair is an explicit, named strategy recorded in the results row — never a silent fix.
- **Feasibility rate is a headline metric reported alongside cost.**
- Scoring an infeasible quantum solution against a feasible classical one is the error that invalidates most published comparisons in this area. The architecture makes it impossible to do accidentally.

## 7. Solver Layer

### 7.1 Uniform protocol

```
solve(instance: Instance, budget: Budget) -> Result
```

`Budget` carries wall-clock limit, target energy (for time-to-target), and a sample cap.
`Result` carries best solution, energy trace, feasibility classification, and a full timing breakdown.

### 7.2 Classical slate

| Solver | Role |
|---|---|
| `almgren_chriss_analytic` | Closed-form continuous solution, rounded to lots. The "what a desk does today" reference. Any interesting result must beat this. |
| `exact_pwl_milp` (HiGHS) | **Rigorous optimality bracket.** See section 7.3. |
| `cpsat` (OR-Tools) | The real opponent. CP-SAT is very strong on this structure and free. Benchmarking against weak classical baselines is a primary failure mode of this literature. |
| `simulated_annealing` (neal) | Runs on the *same QUBO* — isolates "is annealing helping?" from "is the QUBO formulation helping?" |
| `tabu` (dwave-tabu) | As above. |
| `greedy` / steepest descent | Cheap floor. |
| `gurobi_nonconvex` | Optional, licence-dependent. Exactness cross-check against the PWL bracket. |

### 7.3 The exact baseline, on free software

The concave square-root impact makes this a **nonconvex** MIQP. HiGHS cannot solve nonconvex MIQP; Gurobi's spatial branch-and-bound requires a licence.

The free route: **piecewise-linearise the concave impact term and solve the resulting MILP exactly.** Using both PWL under- and over-estimators yields a *rigorous optimality bracket*, not merely a good heuristic reference. The discretisation error is bounded analytically and reported.

This matters because it means the crossover study rests on proven bounds rather than on "the best solution anyone happened to find."

### 7.4 Quantum slate

| Solver | Notes |
|---|---|
| `dwave_qpu` | Advantage2 via `DWaveSampler` plus cached minor-embeddings. The scarce resource. |
| `dwave_hybrid_cqm` | `LeapHybridCQMSampler` accepts constraints natively — **no penalty tuning at all.** This makes CQM-versus-BQM-with-penalties a clean, self-contained comparison. |
| `dwave_hybrid_bqm` | `LeapHybridBQMSampler` for the T3 tier. |
| `qaoa` | Aer to ~28 variables, IBM Runtime on hardware. Uses **CVaR-QAOA** aggregation for the classical outer loop. |

### 7.5 QPU budget guard — architectural

With ~1 minute of real QPU per month, the runner enforces:

1. **Embedding cache.** Minor-embedding is expensive and reusable across instances sharing a topology. Embeddings are computed once, hashed by problem topology, and persisted.
2. **Simulator gate.** No QPU call is permitted until the same experiment has passed against `neal` and a simulator.
3. **Quota ledger.** A running record of consumed QPU-seconds that **hard-fails** rather than silently exhausting the month's allowance.

## 8. Data Ingestion

### 8.1 Bundle format

A `DatasetBundle` is a directory containing Parquet partitioned by symbol and date, plus `manifest.json` carrying content hashes, schema version, provenance, date range, symbol list, and granularity.

Every benchmark run pins the bundle's content hash. Any figure is therefore traceable to exact input data.

### 8.2 Schemas

- `l1_taq` — ts, symbol, bid, ask, bid_size, ask_size, last_px, last_size
- `l2_depth` — ts, symbol, side, level, px, size
- `l3_messages` — ts, symbol, order_id, action (add/cancel/execute/replace), side, px, size
- `daily_bars` — date, symbol, open, high, low, close, volume, adv

### 8.3 CLI verbs

- `quantic data request` — emits a precise request spec (symbols, dates, granularity, fields) to hand to the market data machine. Removes guesswork about what to export.
- `quantic data ingest <path>` — validates against schema, computes hashes, registers in the local catalog.
- `quantic data synth` — generates a synthetic bundle with **known ground-truth impact parameters**. Unblocks M0–M3 from the external machine and doubles as the calibration test fixture.

### 8.4 First data request

- ~25 liquid single-market names
- L1 TAQ plus L2 (10 levels) for ~20 trading days
- Daily bars for one year (covariance and ADV estimation)
- L3 for a 3–5 name subset over 5 days (impact calibration and reconstruction validation)

## 9. Benchmark Harness and Metrics

### 9.1 Metrics

- **Execution cost in basis points of notional** — the translation that makes results legible to practitioners.
- **Feasibility rate** — fraction of returned samples satisfying all constraints without repair.
- **Optimality gap** — against the proven MILP bracket on T0/T1, against best-known on T2/T3.
- **TTS / time-to-target at 99% success**: `TTS = t_run * ln(0.01) / ln(1 − p_success)`. Correctly penalises a solver that reaches the optimum 2% of the time.
- **Two wall-clocks, both always reported**: pure QPU access time, *and* end-to-end including embedding, compilation, and network round-trip to Leap. Quoting only the former is the most common overclaim in this literature; quoting only the latter buries the physics. Both are always shown.
- **Dynamic-range ratio** (section 6.4).
- **Scenario reduction error**, where CVaR is enabled.

### 9.2 Experiment design

The factor space — hardness dials (2^4) × tier (4) × encoding (3) × solver (~9) × seeds — is too large for full factorial. The approach:

1. **Screening design** over the four hardness dials to identify which features actually move the crossover.
2. **Focused sweeps** on the features that do, across tiers and encodings.

### 9.3 Results store and reproducibility

DuckDB. Every row pins: dataset content-hash, RNG seeds, package versions, machine fingerprint, penalty strategy, encoding strategy, repair strategy, and full timing breakdown. Any figure regenerates exactly.

## 10. Testing Strategy

Four layers:

1. **Property tests (Hypothesis) on encodings.** The load-bearing invariant: *for any feasible assignment, the QUBO energy equals the directly-evaluated objective.* If that holds and penalties are correct, the encoding is correct by construction. Also: `decode(encode(x)) == x`.

2. **Golden test / correctness gate.** On T0, the decoded QUBO optimum must equal the exact MILP optimum, **for all 16 hardness-dial combinations.** No QPU time is spent until this passes. This is the gate that separates the platform from most quantum-finance repositories.

3. **Book reconstruction cross-validation.** Both L3 and L2 are available, so the L3 message stream is replayed and the reconstructed book asserted to match the independent L2 snapshots. This converts reconstruction from "hopefully correct" to "verified against an independent source."

4. **Calibration recovery.** Fit the impact model on synthetic data with known `delta` and assert recovery within tolerance.

Plus determinism tests: identical seed and bundle hash produce identical results.

## 11. Visualisation and Dashboard

`viz/figures.py` is the durable artifact — publication-quality matplotlib figures: solution quality versus time-to-solution, crossover curves by hardness dial, feasibility-rate panels, encoding comparisons.

The dashboard is **Streamlit for v1 and deliberately disposable.** It reads the DuckDB store directly and provides:

- Run browser, filtered by experiment, solver, tier, and dial combination
- Crossover plots
- Schedule inspector: a liquidation schedule overlaid on book depth through time
- Encoding diagnostics: dynamic range, embedding statistics, feasibility

**The dashboard and the paper figures call the same plotting functions**, so they cannot diverge. If the dashboard later becomes a deliverable in its own right it is rebuilt properly on FastAPI; that cost is not paid before the need is demonstrated.

## 12. Repository Layout

```
src/quantic/
  core/        domain types (Asset, Lot, Portfolio, BookSnapshot). No deps.
  data/        bundle.py, manifest.py, loaders/{l1,l2,l3,daily}.py,
               request.py, synth.py, catalog.py
  micro/       book_reconstruct.py, liquidity.py, covariance.py,
               impact/{base,almgren_chriss,sqrt_law,depth_walk,calibrate}.py
  problem/     liquidation.py, objective.py, instance.py, generator.py,
               constraints/{full_liquidation,cardinality,min_participation,
                            block_trades,cvar_lvar}.py
  encoding/    qubo.py, penalty.py, integer_encodings.py, ising.py,
               miqp.py, pwl.py, decode.py, diagnostics.py
  solvers/     base.py
               classical/{exact_pwl_milp,cpsat,simulated_annealing,tabu,
                          greedy,almgren_chriss_analytic,gurobi_nonconvex}.py
               quantum/{dwave_qpu,dwave_hybrid_cqm,dwave_hybrid_bqm,qaoa}.py
               embedding_cache.py, quota.py
  bench/       runner.py, metrics.py, store.py, config.py, reproduce.py
  viz/         figures.py
  dashboard/   app.py
  cli.py
tests/
docs/
notebooks/
```

**Boundary rules, enforced by import-linting in CI:**

- `micro/` must not import from `problem/`, `encoding/`, or `solvers/`.
- `problem/` must not import from `encoding/` or `solvers/`.
- `solvers/` must not import from `problem/` internals — only from the public `Instance` type.

These rules are what allow hardness features and solvers to vary independently, which is the entire experiment.

## 13. Technology Stack

- Python 3.11+, environment managed with `uv`
- Polars plus PyArrow for data; Parquet bundles
- NumPy plus Numba for hot loops
- `dwave-ocean-sdk` (dimod, neal, dwave-tabu, dwave-system) for annealing
- Qiskit plus Qiskit Aer for gate-based
- `highspy` for MILP; OR-Tools CP-SAT; Gurobi optional
- DuckDB for the results store
- Typer for the CLI
- pytest plus Hypothesis for tests
- Streamlit for the dashboard
- matplotlib for figures

## 14. Build Order

Each milestone ends at a point where work could stop with something of value in hand.

| | Milestone | Done when |
|---|---|---|
| **M0** | Scaffold, bundle format, synthetic generator | `quantic data synth` produces a valid, schema-conformant bundle |
| **M1** | Microstructure core: reconstruction, liquidity, impact calibration | L3 replay matches L2 snapshots; `delta` recovered on synthetic data within tolerance |
| **M2** | Problem model, constraint modules, instance ladder generator | All 16 dial combinations generate valid instances across all four tiers |
| **M3** | Encoding, penalty derivation, exact PWL-MILP baseline | **GATE: decoded QUBO optimum equals exact MILP optimum on T0, all 16 combinations** |
| **M4** | Classical solvers, bench runner, DuckDB store | A complete classical crossover table on T0–T1 |
| **M5** | Quantum solvers, embedding cache, quota guard | First real Advantage2 result obtained with monthly quota intact |
| **M6** | Metrics, figures, crossover study v1 | Publication-quality figure set regenerable from a single command |
| **M7** | Streamlit dashboard | Runs browsable, schedule inspector working |

**M4 already produces a publishable result** — a classical-only crossover study across the hardness dials — before a single qubit is touched. If quantum hardware disappoints, the project still has an outcome.

## 15. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Quantum solvers lose decisively in every regime | This is an accepted possible outcome, not a failure. The study is framed as a crossover investigation; M4 guarantees a result independent of quantum performance. |
| Penalty tuning silently invalidates results | Penalty derivation is a first-class module with its own tests; feasibility rate and dynamic-range ratio are mandatory reported metrics. |
| Free QPU quota exhausted early | Embedding cache, simulator gate, and hard-failing quota ledger (section 7.5). |
| Dense embedding fails above ~230 variables | Instance ladder is designed around this ceiling; T2 is explicitly the "edge" tier and T3 is hybrid-only by construction. |
| External market data machine blocks progress | Synthetic bundle generator (section 8.3) unblocks M0–M3 entirely; real data is needed first at M1 validation. |
| Weak classical baselines inflate quantum results | CP-SAT and a rigorous PWL-MILP bracket are in the baseline slate specifically to prevent this. |
| CVaR scenario count explodes the QUBO | Mandatory scenario reduction with reported reduction error (section 5.4). |
| Scope creep into verticals A and C | Both are explicitly out of scope with their own future specs (section 2). |

## 16. Decisions Log

| Decision | Rationale |
|---|---|
| Research/benchmark platform, not production tool | Chosen purpose; prioritises correctness and reproducibility over latency. |
| Vertical B first | Self-contained, needs only an impact model, has a rigorous classical baseline, best signal-to-effort ratio. |
| Vertical C deferred and reframed | No viable free-tier quantum path; honest framing is tensor-network versus deep BSDE, which is a different project. |
| Free-tier backends only | No existing account access; also forces results that others can reproduce without budget. |
| Single-market equities | One calendar, one currency, homogeneous impact dynamics; easiest to defend scientifically. |
| All four hardness features included | Not scope creep — each is a dial in the screening design, and the study is specifically about which features move the crossover. |
| Data arrives as exported bundles, not a live connection | Assumed non-reachable database; also yields content-hashed reproducibility as a side effect. |
| Streamlit dashboard, disposable | Dashboard is not the scientific deliverable; figures module is. Upgrade path exists if that changes. |
