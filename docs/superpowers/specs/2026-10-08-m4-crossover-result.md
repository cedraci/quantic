# M4 crossover — result

**Written:** 2026-10-08
**Branch:** `feat/m4-classical-bench`
**Spec:** `2026-10-02-m4-classical-bench-design.md`
**Plan:** `docs/superpowers/plans/2026-10-02-m4-classical-bench.md`, Task 13

---

## Verdict

**The exact classical methods win every cell with cardinality (D2) or CVaR (D3) on. The QUBO samplers win a few cells without them, by small factors, at the smallest budget.**

HiGHS (`exact_pwl_milp`) proves the optimum on all 160 instances within the 60 s budget. Its TTS99 at ε = 1% (60 s budget) is under 35 s on every T1 cell and under 2 s on 25 of the 32 cells.

The crossover is in the TTS99 tables:
- **The cells samplers win.** Taking each solver's best TTS99 over the four budgets, a sampler beats both HiGHS and the sound CP-SAT on 5 of the 32 cells, all at the 0.1 s budget:
  - T1: `none`, `D1` and `D4`, all tabu on domain-wall (0.27, 0.19, 0.47 s, against 0.66, 0.76, 0.51 s for the best exact method).
  - T0: `none` and `D4`, both greedy (0.11 and 0.089 s, against 0.16 and 0.094 s).
- **The common thread.** None of those cells has the cardinality dial (D2) or CVaR (D3). Against the original, unsound CP-SAT the count is 7.
- **From 1 s up** no sampler wins any cell against the sound exact methods.
- **On every cell with D2 or D3** the exact methods win, by factors from 2 (T0 cells with D2 but no CVaR) to 10^5 (T1 cells with CVaR), and at T1 the samplers often never reach 1% at all: averaged over the 16 T1 cells, the best sampler and encoding (tabu, domain-wall) is within 1% on 25% of its runs.
- **At T0 the samplers do reach the optimum,** but only slowly: simulated annealing on binary has 93% of runs within 1% at 60 s.

So the crossover at T0–T1 sits where neither D2 nor D3 is on. There a fast heuristic finds the optimum before an exact solver has finished proving it. Turning on D2 or D3 moves every cell back to the exact side. D1 (concave impact) and D4 (block trades) split: the samplers keep `T1-D1`, `T1-D4` and `T0-D4`, while the exact methods take `T0-D1` and both `D1+D4` cells.

That fixes the bar for M5:
- A QPU running the same QUBOs at T0–T1 is compared with the samplers, its classical analogues, and with the exact methods' TTS99 as the bar.
- On the hard cells there is nothing to win at this size.
- Where quantum might compete on the hard cells is the question of T2–T3, where exact brackets stop closing.

**The CVaR dial is what breaks the samplers.** Without CVaR, every sampler under every encoding reaches the QUBO's minimum on every T0 cell at 60 s. With CVaR, none does on any: they end 21 to 50 times the minimum's magnitude above it (cell medians). The encoding is not the cause: at eight zeta bits it costs at most 1.98% of the optimum (median of seeds, worst cell), and nothing without CVaR.

**One defect was found, and fixed.** The bracket check was not empty: on `T0-D2-2030`, CP-SAT reported OPTIMAL at 17.98 where the true optimum is 14.34. The cause is OR-Tools 9.15's inclusion presolve, the current release. The fix is `78fc519`, and CP-SAT was run again in its own store (`feb00ff`), where the bracket check is empty. The fix costs CP-SAT speed on T1 instances that combine CVaR with block trades; see "CP-SAT and the presolve defect" below. No other solver's numbers depend on CP-SAT: every metric is scored against the reference brackets.

## What was run

| what | crossover study | CP-SAT re-run |
|---|---|---|
| config | `experiments/m4-crossover.toml` | `experiments/m4-cpsat-rerun.toml` |
| commit | `4d420cf` (clean) | `feb00ff` (clean) |
| lock hash | `207fa6f1aca2477e6e0beea0d557bf3717453658d872437afbaf58757b0c919d` | same |
| tasks | 292,480 (292,000 runs, 480 references) | 3,440 (3,200 runs, 240 references) |
| killed | 1 | 0 |
| errors | 0 | 0 |
| wall clock | 110.6 h of running, 2026-10-02 20:21 → 2026-10-08 05:40, in three stretches (paused twice by the owner) | 8 min, 2026-10-08 18:21 → 18:29 |
| throughput | 44 tasks/min on 14 workers | 470 tasks/min |
| store | 37.7 GB DuckDB, not committed | small, not committed |

Machine: Windows 11 (10.0.26200), Intel Core i9-9900 at 3.10 GHz, 63.9 GB, Python 3.11, OR-Tools 9.15.6755.

Instances: T0 (3 assets × 4 buckets) and T1 (5 × 8), all 16 dial combinations, seeds 2026–2030, so 160 instances. Runs:

| solver | runs | per instance |
|---|---|---|
| `simulated_annealing`, `tabu`, `greedy` | 96,000 each | 3 encodings × 50 seeds × 4 budgets, zeta 8 bits |
| `cpsat` | 3,200 | 5 seeds × 4 budgets |
| `exact_pwl_milp` | 640 | 4 budgets |
| `almgren_chriss` | 160 | once, 1 s |
| `ref:brute_force` (T0), `ref:exact_pwl_milp`, `ref:qubo_minimum` (T0, per encoding) | 80, 160, 240 | no budget, killed at 3,600 s |

**The references.** All 160 MILP brackets close: the widest is 1.0e-6 absolute, 3.5e-8 relative. Brute force agrees with the MILP on all 80 T0 instances. No reference is missing, so every cell is scored against a proven optimum. The re-run computed the references again from the same instances (the same hashes) and got identical brackets, to the last bit.

**The kill.** One run reached its hard limit (3 × budget + 30 s = 210 s): tabu, domain-wall, `T1-D3-2029`, sampler seed 49, at the 60 s budget. It counts as a failed run in its cell (`n_killed`); the cell's TTS uses the runs that ended.

**Reproduce.** `quantic bench report <store> --out <dir>` regenerates every table below from a store. Either store regenerates from its commit with `quantic bench run <config> --store <path>`.

### How to read a crossover cell

`median gap / feasibility / TTS99`, at ε = 1%, each sampler at its best encoding (named in brackets):

- **median gap** is over feasible runs on instances whose reference is closed: (total − lower) / |lower|.
- **feasibility** is the share of runs that return a schedule satisfying every constraint. Nothing is repaired.
- **TTS99** is t_run · ln 0.01 / ln(1 − p), with p the share of runs within 1% of the optimum. When p = 0 it is censored, and the cell gives the lower bound the runs imply.

Notes on the definitions:
- The best encoding is ranked by TTS at 1% first, then by median gap.
- t_run excludes killed runs.
- A sampler's trace precision does not enter TTS.
- An infinite bootstrap bound is left blank in `cells.csv`.

## Findings the plan asked for

### 1. Do the samplers reach the QUBO's minimum on T0?

`energy_gap_median`, (best energy − QUBO minimum) / max(1, |minimum|), per sampler and encoding, at 60 s. The table gives the median over the eight cells with CVaR off or on:

| sampler | encoding | CVaR off: median (cells at the minimum) | CVaR on: median (worst cell) |
|---|---|---|---|
| greedy | binary | 0 (8 of 8) | 20.9 (27.2) |
| greedy | domain-wall | 0 (8 of 8) | 22.0 (30.2) |
| greedy | one-hot | 0 (8 of 8) | 25.1 (35.5) |
| simulated annealing | binary | 0 (8 of 8) | 24.4 (33.5) |
| simulated annealing | domain-wall | 0 (8 of 8) | 24.9 (32.8) |
| simulated annealing | one-hot | 0 (8 of 8) | 27.2 (34.4) |
| tabu | binary | 0 (8 of 8) | 42.7 (61.6) |
| tabu | domain-wall | 0 (8 of 8) | 43.6 (62.6) |
| tabu | one-hot | 0 (5 of 8, worst 0.065) | 50.2 (66.8) |

**Without CVaR, the samplers reach the QUBO minimum on every T0 cell.** Greedy does so even at 0.1 s; annealing and tabu do by 1 s. **With CVaR, none reaches it on any cell, at any budget.** Over budgets, the CVaR-on median falls slowly:

| sampler | 0.1 s | 1 s | 10 s | 60 s |
|---|---|---|---|---|
| greedy | 52.4 | 33.0 | 26.8 | 23.2 |
| simulated annealing | 49.2 | 36.4 | 30.5 | 26.2 |
| tabu | 71.4 | 63.5 | 54.7 | 45.6 |

This confirms the dry run's observation (annealing ending 12 to 45 times above the minimum on every CVaR instance) across all three samplers and every seed. The CVaR QUBOs carry the auxiliary ζ register and one slack per scenario (15 scenarios), so they have several hundred variables at T0. Single-flip local search does not cross their landscape.

### 2. The encoding's regret at eight zeta bits

`encoding_regret`, (true objective at the QUBO's minimum − optimum) / |optimum|. Each cell is the median over its five seeds; the table gives the median and worst cell:

| encoding | CVaR off | CVaR on: median | CVaR on: worst cell |
|---|---|---|---|
| binary | 0 | 0.57% | 1.98% |
| domain-wall | 0 | 0.57% | 0.97% |
| one-hot | 0 | 0.57% | 0.97% |

M3b measured 0.23% to 3.46% on seed 2026's five CVaR cells with non-zero regret. Over five seeds the median-of-seeds is inside that range and no cell exceeds 2%. **The encoding loses at most about 2% of the optimum; the samplers lose 20 to 50 times the minimum's magnitude.** On the CVaR cells, sampler failure dominates encoding loss by orders of magnitude.

### 3. Almgren–Chriss

The desk reference (a closed-form trajectory rounded to whole lots, then checked against the constraints) **is never feasible when D2 is on**, at T0 or T1: 0 of 80 instances. Its rounded trajectory spreads every name over every bucket, which a per-bucket cardinality cap of 1 rules out, and nothing is repaired. With D2 off, it is always feasible:

| tier | median gap (worst cell) | median cost |
|---|---|---|
| T0 | 17.6% (47.1%, `T0-D3`) | 2.3 bps |
| T1 | 18.4% (22.8%, `T1-D3`) | 2.2 bps |

So a desk following the textbook schedule pays about 18% more than the optimum on these instances, about 0.35 bps of notional. Its gap is smallest under D4 (2.7% at T0, 8.7% at T1) and largest under CVaR, whose tail risk the variance-based trajectory does not see.

### 4. CP-SAT against HiGHS, and `T1-D1+D3+D4`

From the sound re-run, TTS99 at ε = 1% at the 60 s budget:

| tier | where CP-SAT is faster | where HiGHS is faster |
|---|---|---|
| T0 | 15 of 16 cells, up to 26× (e.g. `D1+D2` 0.049 s against 1.26 s, `D2` 0.074 against 1.26 s) | `D1+D3` only (0.041 against 0.053 s) |
| T1 | all 8 cells without D3, 2.4–8× (e.g. `none` 1.28 s against 9.72 s, `D1+D2` 4.33 against 34.4 s); and `D1+D2+D3+D4`, `D2+D3+D4`, `D3` | 5 of the 8 cells with D3, four by one to four orders of magnitude (`D1+D2+D3` 0.60 against 18.6 s, `D1+D3` 0.92 against 123 s, `D3+D4` 0.20 against 3,052 s, `D1+D3+D4` 0.44 s against never), and `D2+D3` narrowly (0.38 against 0.40 s) |

The split mostly follows the risk model:
- **CP-SAT is faster on variance risk.** It handles the quadratic term with a few outer-approximation cuts.
- **HiGHS is faster on most CVaR cells.** CP-SAT has to carry the scenario rows in scaled integers.

Mean success rate at ε = 1% over the 16 cells of each tier:

| tier | budget | HiGHS | CP-SAT, original | CP-SAT, re-run |
|---|---|---|---|---|
| T0 | 0.1 s | 80% | 76% | 97% |
| T0 | 1 s+ | 100% | 99% | 100% |
| T1 | 0.1 s | 25% | 9% | 13% |
| T1 | 1 s | 88% | 72% | 81% |
| T1 | 10 s | 99% | 94% | 85% |
| T1 | 60 s | 100% | 95% | 85% |

**`T1-D1+D3+D4`:** HiGHS proves it in 0.44 s TTS99. The original CP-SAT took 41.8 s (80% success). The re-run never reaches 1% within 60 s (0%, median gap 9.7%). The other cell the fix hurt is `T1-D3+D4`: 84% → 8%, TTS 30 s → 3,052 s.

### 5. Feasibility of the samplers' output

Infeasible samples count as failures, never repaired. At 60 s, mean over cells:

| tier | binary | domain-wall | one-hot |
|---|---|---|---|
| T0 (SA / tabu / greedy) | 61% / 56% / 36% | 56% / 54% / 27% | 30% / 42% / 14% |
| T1 (SA / tabu / greedy) | 51% / 49% / 46% | 13% / 27% / 0.6% | 2.8% / 1.4% / 1.9% |

At T1, one-hot samples are almost never feasible. Domain-wall samples rarely are, except under tabu.

## CP-SAT and the presolve defect

**What the bracket check found.** On `T0-D2-2030` (variance risk, cardinality cap 1), 18 of the 20 CP-SAT runs closed a bracket at [17.981995, 17.981995], disjoint from the reference's 14.343347:
- Brute force, the MILP and the samplers all find the schedule `[[3,2,0,0],[0,0,0,1],[0,0,2,0]]` at 14.343347, feasible under the evaluator.
- CP-SAT returned `[[2,3,0,0],…]` at 17.98 and claimed it optimal.

**The root cause.** The investigation went step by step:
1. CP-SAT's final integer program was rebuilt with the true optimum forced in. It is feasible there at 879,962,264 ticks, below the 1,189,247,599 that CP-SAT called optimal.
2. Every outer-approximation cut was checked at the optimum. All are valid, so the model is a correct relaxation, and the solver's claim is false.
3. The exported model alone, with none of quantic's code, reproduces the false claim on OR-Tools 9.15.6755, the latest release. With presolve off, it finds the cheaper point.
4. A sweep of presolve parameters isolates the cause: only `presolve_inclusion_work_limit = 0` removes it.
5. The defect depends on our scale. At 2^30 ticks per unit the cut coefficients reach about 5 × 10^9 and the defect appears; at 2^24 it does not.

The repro model and scripts are kept outside the repository. They are worth an upstream report.

**The fix.** Commit `78fc519` sets `presolve_inclusion_work_limit = 0` in `cpsat.py`. A regression test (`test_presolve_never_proves_a_wrong_optimum`) builds this exact instance and asserts that CP-SAT's lower bound is not above the brute-force optimum. It failed before the fix and passes after. The solver and bench suites pass. A lower tick scale was rejected: at 2^20 the T0 brackets already stay open (HEURISTIC), and at T1, with three times as many cells, 2^24 would leave them open too.

**What it changed.** The re-run's bracket check compared 2,963 CP-SAT brackets on all 160 instances and found none disjoint from the reference.
- **Faster:** CP-SAT is faster on most cells without the inclusion pass, at T0 and on T1 without CVaR (e.g. `T1-none` TTS 7.92 → 1.28 s).
- **Slower:** it is much slower on T1 CVaR cells with block trades (`D3+D4`, `D1+D3+D4`).
- **Overall:** at T1 and 60 s its mean success falls from 95% to 85%.

The original numbers stand as what default CP-SAT does, including one false proof in 160 instances. The re-run's are what a sound CP-SAT does.

**For M5.** Inclusion presolve was earning its keep on CVaR. A fix that keeps it is to scale each cut row to smaller integers, as the CVaR rows already are, rather than disable the pass. That is follow-up work, not part of this result.

## Dial effects

Main effects and pairwise interactions of each dial on each solver's median gap at 60 s. These are exact 2^4 factorial contrasts over the 16 cells. Exact solvers show zero everywhere, because their median gap at 60 s is zero in every cell.

The effects that matter are at T1:
- **D2 (cardinality plus minimum lot) is the largest effect for every sampler.** Tabu +0.30, annealing +0.078, greedy +0.047.
- **D3 (CVaR) is next.** Annealing +0.046, greedy +0.057.

At T0 every effect is under 2 points, because the samplers' median gaps at 60 s are mostly zero there. CVaR's damage at T0 shows in success rate and energy gap (section 1), not in median gap. This is a limit of reading effects off the median.

| tier | solver | effect | value | note |
|---|---|---|---|---|
| T0 | cpsat | D1 | 0.0 |  |
| T0 | cpsat | D2 | 0.0 |  |
| T0 | cpsat | D3 | 0.0 |  |
| T0 | cpsat | D4 | 0.0 |  |
| T0 | cpsat | D1xD2 | 0.0 |  |
| T0 | cpsat | D1xD3 | 0.0 |  |
| T0 | cpsat | D1xD4 | 0.0 |  |
| T0 | cpsat | D2xD3 | 0.0 |  |
| T0 | cpsat | D2xD4 | 0.0 |  |
| T0 | cpsat | D3xD4 | 0.0 |  |
| T0 | exact_pwl_milp | D1 | 0.0 |  |
| T0 | exact_pwl_milp | D2 | 0.0 |  |
| T0 | exact_pwl_milp | D3 | 0.0 |  |
| T0 | exact_pwl_milp | D4 | 0.0 |  |
| T0 | exact_pwl_milp | D1xD2 | 0.0 |  |
| T0 | exact_pwl_milp | D1xD3 | 0.0 |  |
| T0 | exact_pwl_milp | D1xD4 | 0.0 |  |
| T0 | exact_pwl_milp | D2xD3 | 0.0 |  |
| T0 | exact_pwl_milp | D2xD4 | 0.0 |  |
| T0 | exact_pwl_milp | D3xD4 | 0.0 |  |
| T1 | exact_pwl_milp | D1 | 0.0 |  |
| T1 | exact_pwl_milp | D2 | 0.0 |  |
| T1 | exact_pwl_milp | D3 | 0.0 |  |
| T1 | exact_pwl_milp | D4 | 0.0 |  |
| T1 | exact_pwl_milp | D1xD2 | 0.0 |  |
| T1 | exact_pwl_milp | D1xD3 | 0.0 |  |
| T1 | exact_pwl_milp | D1xD4 | 0.0 |  |
| T1 | exact_pwl_milp | D2xD3 | 0.0 |  |
| T1 | exact_pwl_milp | D2xD4 | 0.0 |  |
| T1 | exact_pwl_milp | D3xD4 | 0.0 |  |
| T1 | cpsat | D1 | 0.0 |  |
| T1 | cpsat | D2 | 0.0 |  |
| T1 | cpsat | D3 | 0.0 |  |
| T1 | cpsat | D4 | 0.0 |  |
| T1 | cpsat | D1xD2 | 0.0 |  |
| T1 | cpsat | D1xD3 | 0.0 |  |
| T1 | cpsat | D1xD4 | 0.0 |  |
| T1 | cpsat | D2xD3 | 0.0 |  |
| T1 | cpsat | D2xD4 | 0.0 |  |
| T1 | cpsat | D3xD4 | 0.0 |  |
| T0 | simulated_annealing | D1 | -0.0015729636192201167 |  |
| T0 | simulated_annealing | D2 | -0.002017180806218488 |  |
| T0 | simulated_annealing | D3 | 0.002017180806218488 |  |
| T0 | simulated_annealing | D4 | -0.0015729636192201167 |  |
| T0 | simulated_annealing | D1xD2 | 0.0015729636192201167 |  |
| T0 | simulated_annealing | D1xD3 | -0.0015729636192201167 |  |
| T0 | simulated_annealing | D1xD4 | 0.002017180806218488 |  |
| T0 | simulated_annealing | D2xD3 | -0.002017180806218488 |  |
| T0 | simulated_annealing | D2xD4 | 0.0015729636192201167 |  |
| T0 | simulated_annealing | D3xD4 | -0.0015729636192201167 |  |
| T0 | tabu | D1 | -0.011691473117605915 |  |
| T0 | tabu | D2 | -0.015313957997546373 |  |
| T0 | tabu | D3 | 0.017963648359213845 |  |
| T0 | tabu | D4 | -0.007315511100859408 |  |
| T0 | tabu | D1xD2 | 0.009041782755938442 |  |
| T0 | tabu | D1xD3 | -0.011691473117605915 |  |
| T0 | tabu | D1xD4 | 0.011743128252248682 |  |
| T0 | tabu | D2xD3 | -0.015313957997546373 |  |
| T0 | tabu | D2xD4 | 0.0046658207391919344 |  |
| T0 | tabu | D3xD4 | -0.007315511100859408 |  |
| T0 | greedy | D1 | -0.0025338113295287103 |  |
| T0 | greedy | D2 | -0.004278172948236011 |  |
| T0 | greedy | D3 | 0.004278172948236011 |  |
| T0 | greedy | D4 | -0.0025338113295287103 |  |
| T0 | greedy | D1xD2 | 0.0025338113295287103 |  |
| T0 | greedy | D1xD3 | -0.0025338113295287103 |  |
| T0 | greedy | D1xD4 | 0.004278172948236011 |  |
| T0 | greedy | D2xD3 | -0.004278172948236011 |  |
| T0 | greedy | D2xD4 | 0.0025338113295287103 |  |
| T0 | greedy | D3xD4 | -0.0025338113295287103 |  |
| T1 | tabu | D1 | -0.03158950710361069 |  |
| T1 | tabu | D2 | 0.297841740286348 |  |
| T1 | tabu | D3 | -0.021202109078058085 |  |
| T1 | tabu | D4 | -0.016581169519529648 |  |
| T1 | tabu | D1xD2 | -0.043386226627870506 |  |
| T1 | tabu | D1xD3 | 0.009346190868753601 |  |
| T1 | tabu | D1xD4 | -0.040081492608299296 |  |
| T1 | tabu | D2xD3 | -0.09610102891762662 |  |
| T1 | tabu | D2xD4 | -0.014973728714834156 |  |
| T1 | tabu | D3xD4 | 0.024318775202877396 |  |
| T1 | greedy | D1 | 0.006414252004796901 |  |
| T1 | greedy | D2 | 0.04720312809772805 |  |
| T1 | greedy | D3 | 0.05731923485397991 |  |
| T1 | greedy | D4 | 0.0016839554066059512 |  |
| T1 | greedy | D1xD2 | 0.0035940127943669756 |  |
| T1 | greedy | D1xD3 | 0.015954415729355437 |  |
| T1 | greedy | D1xD4 | -0.0032541807452012295 |  |
| T1 | greedy | D2xD3 | 0.023845013015329847 |  |
| T1 | greedy | D2xD4 | 0.011253842749474295 |  |
| T1 | greedy | D3xD4 | 0.00668191649126159 |  |
| T1 | simulated_annealing | D1 | 0.007168008515629529 |  |
| T1 | simulated_annealing | D2 | 0.07812940413391933 |  |
| T1 | simulated_annealing | D3 | 0.045766649127332484 |  |
| T1 | simulated_annealing | D4 | -0.01393112329505504 |  |
| T1 | simulated_annealing | D1xD2 | -0.00560044721386449 |  |
| T1 | simulated_annealing | D1xD3 | 0.029198896750058462 |  |
| T1 | simulated_annealing | D1xD4 | -0.006895290551528779 |  |
| T1 | simulated_annealing | D2xD3 | 0.026862608457653037 |  |
| T1 | simulated_annealing | D2xD4 | -0.001359658000938166 |  |
| T1 | simulated_annealing | D3xD4 | 0.014995340175774058 |  |

## What M5 should take from it

- **Zeta bits: keep 8.** At eight bits the encoding costs ≤ 2% of the optimum (median of seeds, worst cell) and nothing without CVaR. The samplers' loss on CVaR is 20–50 times the minimum's magnitude. More bits would grow the QUBO without touching the failure that matters.
- **Encodings: binary first, domain-wall second, drop one-hot.**
  - Binary has the best feasibility at both tiers and the best sampler success (SA binary 93% at T0 within 1%; greedy binary 9% at T1).
  - Domain-wall is the encoding under which tabu reaches 25% at T1.
  - One-hot is the least feasible (1–3% at T1) and has the largest dynamic range: a median 3.2 × 10^9 at T1 without CVaR, against 1.8 × 10^8 for domain-wall and 5.0 × 10^8 for binary.
  - All T1 dynamic ranges (10^8–10^9) far exceed what an annealer's analogue precision resolves. **Expect the QPU to see a coarsened QUBO at T1, and measure that before blaming the sampler.**
- **Samplers to compare a QPU against:**
  - **Simulated annealing on binary**: the strongest at T0 and the QPU's direct classical analogue.
  - **Tabu on domain-wall and greedy on binary**: the strongest at T1.
  - Report the QPU against these at matched wall-clock budgets. Report it against HiGHS's TTS99 only as the bar.
- **The T0–T1 crossover is on the easy cells.** On `none`, `D1` and `D4`, a fast sampler beats the exact methods' TTS99 at the 0.1 s budget. That is where a QPU at T0–T1 has a realistic comparison to make, and it should be measured there at sub-second budgets. Report it against tabu on domain-wall at T1 and greedy at T0. Do not treat a win there as a win on the hard problem.
- **Where to look for a crossover on the hard cells: T2–T3, not T0–T1.** Inside this ladder, exact methods prove optima in seconds. The CVaR and D2 cells are where the samplers fail hardest, and they are also where the exact methods' times grow fastest (HiGHS up to 34 s on `T1-D1+D2`; CP-SAT up to 3,052 s on `T1-D3+D4`). M5's T2–T3 runs should start there.
- **CP-SAT:** run the sound version (`78fc519` or later). Fix its CVaR speed by scaling rather than by turning a presolve pass off, before relying on it at T2–T3.

## Crossover tables: the study

Each table is at one budget. Samplers and HiGHS come from `4d420cf`. The CP-SAT column is the original, unsound one; the next section has the sound CP-SAT.

### 0.1 s budget

| tier | dials | cpsat | exact_pwl_milp | greedy | tabu | simulated_annealing |
|---|---|---|---|---|---|---|
| T0 | D1 | 0.17% / 100% / 0.787s | 0.77% / 100% / 1.41s | 0.00% / 51% / 0.106s (domain_wall) | 0.00% / 100% / 0.181s (domain_wall) | 2.96% / 100% / 3.85s (binary) |
| T0 | D1+D2 | 0.00% / 100% / 0.147s | 37.59% / 100% / 6.49s | 0.00% / 29% / 0.121s (one_hot) | 2.59% / 97% / 1.2s (domain_wall) | 0.00% / 66% / 0.223s (domain_wall) |
| T0 | D1+D2+D3 | 0.00% / 100% / 0.0265s | 0.00% / 100% / 0.0406s | 42.48% / 3% / 45.7s (binary) | 55.12% / 11% / 113s (binary) | 48.99% / 30% / 27.2s (one_hot) |
| T0 | D1+D2+D3+D4 | 0.00% / 76% / 0.432s | 0.00% / 100% / 0.12s | 13.61% / 2% / 82.1s (binary) | 27.58% / 10% / 23.7s (one_hot) | 20.24% / 22% / 11.8s (one_hot) |
| T0 | D1+D2+D4 | 0.00% / 100% / 0.0579s | 0.00% / 100% / 0.093s | 0.00% / 48% / 0.114s (binary) | 19.07% / 95% / 2.72s (binary) | 0.00% / 63% / 0.246s (domain_wall) |
| T0 | D1+D3 | 0.00% / 100% / 0.0406s | 0.00% / 100% / 0.0371s | 9.31% / 4% / 57s (binary) | 7.01% / 15% / 33.5s (binary) | 12.15% / 50% / 23.3s (binary) |
| T0 | D1+D3+D4 | 0.00% / 40% / 1.29s | 0.00% / 100% / 0.192s | 17.03% / 3% / 219s (binary) | 17.46% / 3% / censored, > 25 s (one_hot) | 11.26% / 20% / 35.9s (domain_wall) |
| T0 | D1+D4 | 0.00% / 100% / 0.0511s | 0.00% / 100% / 0.191s | 0.00% / 89% / 0.104s (domain_wall) | 0.00% / 100% / 0.134s (domain_wall) | 0.38% / 100% / 0.442s (binary) |
| T0 | D2 | 3.55% / 100% / 1.13s | 21.38% / 100% / 5.75s | 0.00% / 23% / 0.193s (domain_wall) | 12.33% / 97% / 1.67s (binary) | 0.00% / 72% / 0.36s (domain_wall) |
| T0 | D2+D3 | 0.00% / 100% / 0.0556s | 0.00% / 100% / 0.0651s | 17.58% / 4% / 37.6s (one_hot) | 19.93% / 12% / 236s (binary) | 24.97% / 16% / 10.6s (one_hot) |
| T0 | D2+D3+D4 | 0.00% / 76% / 0.425s | 0.00% / 100% / 0.529s | 28.07% / 2% / 49.8s (one_hot) | 33.20% / 17% / 119s (domain_wall) | 20.67% / 20% / 28.4s (domain_wall) |
| T0 | D2+D4 | 0.00% / 100% / 0.13s | 0.00% / 100% / 0.298s | 0.00% / 39% / 0.195s (binary) | 29.00% / 100% / 8.31s (domain_wall) | 0.00% / 87% / 0.761s (domain_wall) |
| T0 | D3 | 0.00% / 100% / 0.0884s | 0.00% / 100% / 0.0597s | 25.35% / 4% / 462s (binary) | 7.57% / 7% / censored, > 25 s (domain_wall) | 46.02% / 5% / 189s (one_hot) |
| T0 | D3+D4 | 0.00% / 60% / 0.769s | 0.00% / 100% / 0.112s | 0.00% / 62% / 11.4s (domain_wall) | 0.00% / 17% / 3.66s (one_hot) | 14.95% / 71% / 3.72s (binary) |
| T0 | D4 | 0.00% / 100% / 0.0724s | 0.00% / 100% / 0.106s | 0.00% / 91% / 0.0887s (binary) | 0.00% / 100% / 0.164s (domain_wall) | 1.78% / 100% / 0.932s (binary) |
| T0 | none | 1.36% / 100% / 0.808s | 0.47% / 100% / 1.16s | 0.00% / 65% / 0.107s (domain_wall) | 0.00% / 100% / 0.147s (domain_wall) | 1.36% / 100% / 0.88s (binary) |
| T1 | D1 | - / 0% / censored, > 2.5 s | 18.53% / 40% / censored, > 0.5 s | 4.77% / 100% / 46.9s (binary) | 0.03% / 100% / 0.185s (domain_wall) | 15.55% / 14% / censored, > 25 s (domain_wall) |
| T1 | D1+D2 | 1.39% / 84% / censored, > 2.5 s | 22.12% / 80% / censored, > 0.5 s | 10.82% / 3% / 62.6s (domain_wall) | 54.47% / 3% / censored, > 25 s (binary) | 22.02% / 12% / censored, > 25 s (domain_wall) |
| T1 | D1+D2+D3 | 1.70% / 68% / 1.48s | 0.00% / 100% / 0.427s | 26.45% / 0% / censored, > 25 s (binary) | 29.81% / 2% / censored, > 25 s (domain_wall) | 31.48% / 4% / censored, > 25 s (binary) |
| T1 | D1+D2+D3+D4 | - / 0% / censored, > 2.5 s | 0.00% / 60% / 5.54s | 24.86% / 1% / censored, > 25 s (one_hot) | 83.38% / 1% / censored, > 25 s (domain_wall) | 21.48% / 0% / censored, > 25 s (one_hot) |
| T1 | D1+D2+D4 | 1.75% / 100% / 2.07s | 4.91% / 100% / 2.16s | 5.23% / 2% / 5.21s (domain_wall) | 70.88% / 2% / censored, > 25 s (binary) | 47.78% / 1% / 148s (one_hot) |
| T1 | D1+D3 | - / 0% / censored, > 2.5 s | 0.24% / 60% / 1.24s | 18.25% / 71% / censored, > 25 s (binary) | 41.49% / 96% / censored, > 25 s (binary) | 22.01% / 99% / censored, > 25 s (binary) |
| T1 | D1+D3+D4 | - / 0% / censored, > 2.5 s | 0.04% / 40% / 2.06s | 17.74% / 90% / censored, > 25 s (binary) | 34.76% / 87% / censored, > 25 s (binary) | 18.74% / 24% / censored, > 25 s (domain_wall) |
| T1 | D1+D4 | 23.37% / 8% / censored, > 2.5 s | 32.25% / 60% / 4.33s | 4.44% / 99% / 6.53s (binary) | 0.74% / 100% / 0.992s (domain_wall) | 13.64% / 100% / censored, > 25 s (binary) |
| T1 | D2 | 6.07% / 56% / censored, > 2.5 s | 38.23% / 80% / censored, > 0.5 s | 19.02% / 3% / censored, > 25 s (one_hot) | 72.11% / 1% / censored, > 25 s (binary) | 47.62% / 1% / censored, > 25 s (one_hot) |
| T1 | D2+D3 | 16.60% / 56% / 2.04s | 0.10% / 100% / 0.433s | 31.32% / 1% / censored, > 25 s (one_hot) | 123.04% / 0% / censored, > 25 s (domain_wall) | 31.84% / 2% / censored, > 25 s (binary) |
| T1 | D2+D3+D4 | - / 0% / censored, > 2.5 s | 12.61% / 40% / censored, > 0.5 s | 47.34% / 0% / censored, > 25 s (binary) | 55.86% / 1% / censored, > 25 s (domain_wall) | 41.37% / 3% / censored, > 25 s (binary) |
| T1 | D2+D4 | 0.58% / 100% / 0.409s | 11.12% / 100% / 2.2s | 23.17% / 0% / censored, > 25 s (domain_wall) | - / 0% / censored, > 25 s (binary) | 15.47% / 0% / censored, > 25 s (domain_wall) |
| T1 | D3 | - / 0% / censored, > 2.5 s | 12.62% / 40% / 13.9s | 19.10% / 97% / censored, > 25 s (binary) | 15.37% / 2% / censored, > 25 s (domain_wall) | 18.62% / 100% / censored, > 25 s (binary) |
| T1 | D3+D4 | - / 0% / censored, > 2.5 s | 0.00% / 60% / 3.17s | 14.71% / 90% / censored, > 25 s (binary) | 34.99% / 84% / censored, > 25 s (binary) | 16.43% / 96% / censored, > 25 s (binary) |
| T1 | D4 | - / 0% / censored, > 2.5 s | 9.70% / 40% / 15.2s | 2.51% / 0% / 237s (domain_wall) | 0.27% / 96% / 0.474s (domain_wall) | 11.50% / 99% / censored, > 25 s (binary) |
| T1 | none | - / 0% / censored, > 2.5 s | 34.40% / 60% / censored, > 0.5 s | 6.55% / 100% / censored, > 25 s (binary) | 0.00% / 100% / 0.273s (domain_wall) | 18.82% / 100% / censored, > 25 s (binary) |

### 1 s budget

| tier | dials | almgren_chriss | cpsat | exact_pwl_milp | simulated_annealing | greedy | tabu |
|---|---|---|---|---|---|---|---|
| T0 | D1 | 13.22% / 100% / censored, > 5 s | 0.00% / 100% / 0.413s | 0.00% / 100% / 0.515s | 0.67% / 100% / 5.01s (binary) | 0.00% / 95% / 1.01s (binary) | 0.00% / 100% / 1.52s (domain_wall) |
| T0 | D1+D2 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.122s | 0.00% / 100% / 0.628s | 0.00% / 68% / 1.01s (domain_wall) | 0.00% / 38% / 1s (domain_wall) | 0.00% / 96% / 2.89s (domain_wall) |
| T0 | D1+D2+D3 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.0268s | 0.00% / 100% / 0.0399s | 18.24% / 27% / 19.6s (domain_wall) | 20.84% / 6% / 21s (domain_wall) | 47.05% / 10% / 106s (binary) |
| T0 | D1+D2+D3+D4 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.116s | 0.00% / 100% / 0.267s | 5.63% / 20% / 15.6s (one_hot) | 8.49% / 3% / 21.9s (one_hot) | 19.70% / 11% / 47.7s (one_hot) |
| T0 | D1+D2+D4 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.0646s | 0.00% / 100% / 0.131s | 0.00% / 62% / 1.01s (binary) | 0.00% / 45% / 1s (domain_wall) | 0.00% / 94% / 5.97s (domain_wall) |
| T0 | D1+D3 | 17.44% / 100% / censored, > 5 s | 0.00% / 100% / 0.0409s | 0.00% / 100% / 0.0423s | 1.98% / 50% / 14.9s (binary) | 4.49% / 4% / 30s (binary) | 4.28% / 16% / 43s (binary) |
| T0 | D1+D3+D4 | 29.35% / 100% / censored, > 5 s | 0.00% / 100% / 0.138s | 0.00% / 100% / 0.117s | 6.22% / 39% / 44.6s (binary) | 6.07% / 4% / 73.7s (binary) | 10.62% / 16% / 695s (binary) |
| T0 | D1+D4 | 12.52% / 100% / censored, > 5 s | 0.00% / 100% / 0.0971s | 0.00% / 100% / 0.145s | 0.00% / 100% / 1.25s (binary) | 0.00% / 96% / 1.01s (binary) | 0.00% / 100% / 1.25s (domain_wall) |
| T0 | D2 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.472s | 0.00% / 100% / 0.791s | 0.00% / 74% / 0.98s (binary) | 0.00% / 21% / 1.01s (binary) | 0.00% / 83% / 6.12s (one_hot) |
| T0 | D2+D3 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.0412s | 0.00% / 100% / 0.0927s | 3.70% / 21% / 12.5s (binary) | 6.97% / 3% / 17.1s (one_hot) | 15.06% / 11% / 200s (binary) |
| T0 | D2+D3+D4 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.139s | 0.00% / 100% / 0.432s | 7.17% / 20% / 18.4s (one_hot) | 8.09% / 4% / 22.6s (binary) | 20.85% / 17% / 138s (domain_wall) |
| T0 | D2+D4 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.0557s | 0.00% / 100% / 0.156s | 0.00% / 87% / 1.13s (domain_wall) | 0.00% / 45% / 1.01s (domain_wall) | 1.53% / 100% / 12.3s (binary) |
| T0 | D3 | 47.07% / 100% / censored, > 5 s | 0.00% / 100% / 0.0793s | 0.00% / 100% / 0.0854s | 9.26% / 47% / 114s (binary) | 16.21% / 3% / 243s (binary) | 29.71% / 9% / 743s (binary) |
| T0 | D3+D4 | 19.48% / 100% / 0.0105s | 0.00% / 100% / 0.149s | 0.00% / 100% / 0.176s | 5.64% / 61% / 21.5s (binary) | 7.47% / 15% / 25.8s (binary) | 17.66% / 27% / 26.5s (binary) |
| T0 | D4 | 2.70% / 100% / 0.0117s | 0.00% / 100% / 0.0876s | 0.00% / 100% / 0.181s | 0.00% / 100% / 1.71s (domain_wall) | 0.00% / 95% / 1.01s (binary) | 0.00% / 100% / 1.31s (domain_wall) |
| T0 | none | 17.68% / 100% / censored, > 5 s | 0.00% / 100% / 0.364s | 0.00% / 100% / 0.519s | 0.00% / 100% / 2.58s (binary) | 0.00% / 98% / 1.01s (binary) | 0.00% / 100% / 1.35s (domain_wall) |
| T1 | D1 | 17.68% / 100% / censored, > 5 s | 0.02% / 88% / 4.17s | 0.00% / 100% / 2.67s | 10.81% / 100% / censored, > 250 s (binary) | 3.61% / 100% / 53.8s (binary) | 0.01% / 100% / 1.43s (domain_wall) |
| T1 | D1+D2 | - / 0% / censored, > 5 s | 0.75% / 100% / 6s | 0.46% / 100% / 3.38s | 12.02% / 12% / censored, > 250 s (domain_wall) | 7.88% / 4% / 104s (domain_wall) | 55.63% / 2% / censored, > 250 s (binary) |
| T1 | D1+D2+D3 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.934s | 0.00% / 100% / 0.373s | 25.59% / 7% / censored, > 250 s (binary) | 22.59% / 1% / censored, > 250 s (binary) | 38.85% / 2% / censored, > 250 s (binary) |
| T1 | D1+D2+D3+D4 | - / 0% / censored, > 5 s | 0.00% / 96% / 1.95s | 0.00% / 100% / 1.83s | 26.87% / 2% / censored, > 250 s (binary) | 19.76% / 1% / censored, > 250 s (one_hot) | 31.95% / 1% / censored, > 250 s (binary) |
| T1 | D1+D2+D4 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.798s | 0.00% / 100% / 2.33s | 30.35% / 2% / 1.28e+03s (one_hot) | 5.47% / 2% / 20.6s (binary) | 54.74% / 2% / censored, > 250 s (binary) |
| T1 | D1+D3 | 20.90% / 100% / censored, > 5 s | 37.60% / 56% / 20s | 0.00% / 100% / 0.64s | 15.21% / 99% / censored, > 250 s (binary) | 11.10% / 67% / censored, > 250 s (binary) | 22.79% / 97% / censored, > 250 s (binary) |
| T1 | D1+D3+D4 | 20.66% / 100% / censored, > 5 s | 0.13% / 60% / 7.44s | 0.00% / 100% / 0.46s | 13.00% / 100% / censored, > 250 s (binary) | 8.93% / 87% / censored, > 250 s (binary) | 23.13% / 88% / censored, > 250 s (binary) |
| T1 | D1+D4 | 14.75% / 100% / censored, > 5 s | 0.22% / 100% / 1.94s | 0.01% / 100% / 2.11s | 9.44% / 100% / censored, > 250 s (binary) | 2.81% / 99% / 14.6s (binary) | 0.14% / 100% / 2.79s (domain_wall) |
| T1 | D2 | - / 0% / censored, > 5 s | 0.36% / 100% / 3.16s | 1.38% / 100% / 9.34s | 27.37% / 0% / censored, > 250 s (domain_wall) | 18.37% / 0% / 1.17e+03s (binary) | 44.25% / 0% / censored, > 250 s (domain_wall) |
| T1 | D2+D3 | - / 0% / censored, > 5 s | 0.00% / 100% / 0.283s | 0.00% / 100% / 0.384s | 25.71% / 2% / censored, > 250 s (domain_wall) | 20.87% / 1% / censored, > 250 s (one_hot) | 44.95% / 0% / censored, > 250 s (domain_wall) |
| T1 | D2+D3+D4 | - / 0% / censored, > 5 s | 0.00% / 92% / 2.04s | 0.00% / 100% / 0.537s | 36.04% / 1% / censored, > 250 s (one_hot) | 31.47% / 1% / censored, > 250 s (one_hot) | 35.90% / 0% / censored, > 250 s (one_hot) |
| T1 | D2+D4 | - / 0% / censored, > 5 s | 0.03% / 100% / 0.861s | 0.03% / 100% / 0.97s | 27.09% / 0% / censored, > 250 s (binary) | 14.80% / 0% / censored, > 250 s (domain_wall) | 7.60% / 0% / censored, > 250 s (binary) |
| T1 | D3 | 22.82% / 100% / censored, > 5 s | 0.00% / 68% / 3.7s | 0.00% / 100% / 0.616s | 12.57% / 100% / censored, > 250 s (binary) | 7.95% / 95% / censored, > 250 s (binary) | 12.50% / 97% / censored, > 250 s (binary) |
| T1 | D3+D4 | 19.19% / 100% / censored, > 5 s | 0.00% / 60% / 8s | 0.00% / 100% / 0.322s | 11.06% / 97% / censored, > 250 s (binary) | 7.83% / 91% / censored, > 250 s (binary) | 14.94% / 87% / censored, > 250 s (binary) |
| T1 | D4 | 8.75% / 100% / censored, > 5 s | 0.00% / 96% / 1.34s | 0.00% / 100% / 3.48s | 7.36% / 99% / censored, > 250 s (binary) | 2.22% / 0% / 455s (domain_wall) | 0.00% / 95% / 1.58s (domain_wall) |
| T1 | none | 13.49% / 100% / censored, > 5 s | 0.00% / 80% / 2.81s | 0.03% / 100% / 3.54s | 12.79% / 100% / censored, > 250 s (binary) | 4.79% / 100% / censored, > 250 s (binary) | 0.00% / 100% / 1.29s (domain_wall) |

### 10 s budget

| tier | dials | cpsat | exact_pwl_milp | simulated_annealing | greedy | tabu |
|---|---|---|---|---|---|---|
| T0 | D1 | 0.00% / 100% / 0.455s | 0.00% / 100% / 0.471s | 0.00% / 100% / 17.6s (binary) | 0.00% / 95% / 10s (binary) | 0.00% / 100% / 10.9s (binary) |
| T0 | D1+D2 | 0.00% / 100% / 0.125s | 0.00% / 100% / 1.24s | 0.00% / 69% / 10s (domain_wall) | 0.00% / 37% / 10s (binary) | 0.00% / 95% / 11.5s (domain_wall) |
| T0 | D1+D2+D3 | 0.00% / 100% / 0.027s | 0.00% / 100% / 0.0399s | 0.00% / 28% / 41.4s (binary) | 0.00% / 7% / 60.8s (one_hot) | 5.71% / 11% / 122s (domain_wall) |
| T0 | D1+D2+D3+D4 | 0.00% / 100% / 0.108s | 0.00% / 100% / 0.217s | 0.00% / 21% / 27.2s (one_hot) | 0.00% / 3% / 49.7s (one_hot) | 2.56% / 11% / 131s (one_hot) |
| T0 | D1+D2+D4 | 0.00% / 100% / 0.0668s | 0.00% / 100% / 0.128s | 0.00% / 63% / 10s (domain_wall) | 0.00% / 43% / 10s (domain_wall) | 0.00% / 94% / 25.6s (domain_wall) |
| T0 | D1+D3 | 0.00% / 100% / 0.0396s | 0.00% / 100% / 0.0406s | 0.32% / 50% / 28.2s (binary) | 0.77% / 5% / 53.3s (binary) | 0.95% / 16% / 78.2s (binary) |
| T0 | D1+D3+D4 | 0.00% / 100% / 0.114s | 0.00% / 100% / 0.105s | 1.28% / 38% / 91.6s (binary) | 2.29% / 4% / 128s (binary) | 4.38% / 16% / 784s (binary) |
| T0 | D1+D4 | 0.00% / 100% / 0.0876s | 0.00% / 100% / 0.138s | 0.00% / 100% / 10s (binary) | 0.00% / 96% / 10s (binary) | 0.00% / 100% / 11.1s (binary) |
| T0 | D2 | 0.00% / 100% / 0.507s | 0.00% / 100% / 1.12s | 0.00% / 72% / 10s (domain_wall) | 0.00% / 20% / 10s (binary) | 0.00% / 98% / 27.1s (binary) |
| T0 | D2+D3 | 0.00% / 100% / 0.0443s | 0.00% / 100% / 0.0982s | 0.00% / 20% / 21.7s (domain_wall) | 0.00% / 3% / 37.9s (binary) | 7.09% / 12% / 322s (domain_wall) |
| T0 | D2+D3+D4 | 0.00% / 100% / 0.146s | 0.00% / 100% / 0.504s | 0.00% / 23% / 25.8s (domain_wall) | 0.00% / 5% / 34s (one_hot) | 5.97% / 18% / 133s (one_hot) |
| T0 | D2+D4 | 0.00% / 100% / 0.0523s | 0.00% / 100% / 0.164s | 0.00% / 88% / 10s (binary) | 0.00% / 45% / 10s (domain_wall) | 0.00% / 100% / 22.3s (binary) |
| T0 | D3 | 0.00% / 100% / 0.0773s | 0.00% / 100% / 0.115s | 4.22% / 47% / 151s (binary) | 6.40% / 3% / 502s (binary) | 12.87% / 8% / 1.28e+03s (binary) |
| T0 | D3+D4 | 0.00% / 100% / 0.148s | 0.00% / 100% / 0.122s | 0.75% / 55% / 65s (binary) | 2.26% / 11% / 99.4s (binary) | 6.91% / 26% / 222s (binary) |
| T0 | D4 | 0.00% / 100% / 0.0877s | 0.00% / 100% / 0.191s | 0.00% / 100% / 10s (binary) | 0.00% / 97% / 10s (binary) | 0.00% / 100% / 11.1s (binary) |
| T0 | none | 0.00% / 100% / 0.381s | 0.00% / 100% / 1.02s | 0.00% / 100% / 8.36s (domain_wall) | 0.00% / 98% / 10s (binary) | 0.00% / 100% / 11.2s (domain_wall) |
| T1 | D1 | 0.00% / 100% / 3.61s | 0.00% / 100% / 3.94s | 8.44% / 100% / censored, > 2.5e+03 s (binary) | 2.61% / 100% / 203s (binary) | 0.00% / 100% / 11.5s (domain_wall) |
| T1 | D1+D2 | 0.00% / 100% / 8.6s | 0.10% / 100% / 28.8s | 12.41% / 13% / 3.86e+03s (domain_wall) | 7.45% / 4% / 214s (domain_wall) | 47.00% / 2% / censored, > 2.5e+03 s (binary) |
| T1 | D1+D2+D3 | 0.00% / 100% / 3.41s | 0.00% / 100% / 0.792s | 20.14% / 4% / censored, > 2.5e+03 s (domain_wall) | 13.46% / 1% / censored, > 2.5e+03 s (binary) | 36.01% / 2% / censored, > 2.5e+03 s (binary) |
| T1 | D1+D2+D3+D4 | 0.00% / 100% / 1.29s | 0.00% / 100% / 2.46s | 22.20% / 0% / censored, > 2.5e+03 s (one_hot) | 11.82% / 1% / censored, > 2.5e+03 s (one_hot) | 23.93% / 1% / censored, > 2.5e+03 s (domain_wall) |
| T1 | D1+D2+D4 | 0.00% / 100% / 2.92s | 0.00% / 100% / 4.74s | 9.45% / 3% / 622s (binary) | 2.23% / 12% / 112s (one_hot) | 31.85% / 2% / censored, > 2.5e+03 s (binary) |
| T1 | D1+D3 | 0.00% / 100% / 36.7s | 0.00% / 100% / 0.902s | 10.45% / 99% / censored, > 2.5e+03 s (binary) | 7.54% / 66% / censored, > 2.5e+03 s (binary) | 11.51% / 97% / censored, > 2.5e+03 s (binary) |
| T1 | D1+D3+D4 | 0.00% / 100% / 13.1s | 0.00% / 100% / 0.507s | 12.50% / 26% / 1.72e+04s (domain_wall) | 6.42% / 86% / censored, > 2.5e+03 s (binary) | 11.12% / 88% / censored, > 2.5e+03 s (binary) |
| T1 | D1+D4 | 0.00% / 100% / 4.42s | 0.00% / 100% / 4.45s | 7.47% / 100% / 3.9e+03s (binary) | 1.89% / 99% / 89.5s (binary) | 0.04% / 100% / 12.1s (domain_wall) |
| T1 | D2 | 0.00% / 100% / 6.27s | 0.00% / 100% / 10.1s | 21.68% / 1% / censored, > 2.5e+03 s (one_hot) | 4.11% / 3% / 1.64e+03s (one_hot) | 47.13% / 0% / censored, > 2.5e+03 s (domain_wall) |
| T1 | D2+D3 | 0.00% / 100% / 0.282s | 0.00% / 100% / 0.389s | 18.98% / 1% / censored, > 2.5e+03 s (domain_wall) | 15.25% / 0% / censored, > 2.5e+03 s (binary) | 40.79% / 1% / censored, > 2.5e+03 s (binary) |
| T1 | D2+D3+D4 | 0.00% / 100% / 0.947s | 0.00% / 100% / 1.12s | 24.85% / 2% / censored, > 2.5e+03 s (binary) | 15.91% / 1% / censored, > 2.5e+03 s (one_hot) | 44.53% / 0% / censored, > 2.5e+03 s (one_hot) |
| T1 | D2+D4 | 0.00% / 100% / 4.29s | 0.00% / 100% / 6.51s | 19.24% / 0% / censored, > 2.5e+03 s (domain_wall) | 11.00% / 0% / 1.16e+04s (binary) | 32.05% / 0% / censored, > 2.5e+03 s (domain_wall) |
| T1 | D3 | 0.00% / 100% / 1.6s | 0.00% / 100% / 0.729s | 9.64% / 100% / censored, > 2.5e+03 s (binary) | 6.22% / 95% / censored, > 2.5e+03 s (binary) | 7.29% / 98% / censored, > 2.5e+03 s (binary) |
| T1 | D3+D4 | 0.00% / 100% / 10.1s | 0.00% / 100% / 0.337s | 8.18% / 98% / censored, > 2.5e+03 s (binary) | 5.56% / 91% / censored, > 2.5e+03 s (binary) | 8.87% / 87% / 1.6e+04s (binary) |
| T1 | D4 | 0.00% / 100% / 1.34s | 0.00% / 100% / 1.56s | 5.49% / 99% / censored, > 2.5e+03 s (binary) | 1.02% / 0% / 294s (domain_wall) | 0.00% / 95% / 11.7s (domain_wall) |
| T1 | none | 0.00% / 100% / 3.54s | 0.00% / 100% / 2.97s | 9.93% / 100% / censored, > 2.5e+03 s (binary) | 3.65% / 100% / censored, > 2.5e+03 s (binary) | 0.00% / 100% / 12.2s (domain_wall) |

### 60 s budget

| tier | dials | cpsat | exact_pwl_milp | simulated_annealing | tabu | greedy |
|---|---|---|---|---|---|---|
| T0 | D1 | 0.00% / 100% / 0.479s | 0.00% / 100% / 0.455s | 0.00% / 100% / 60s (binary) | 0.00% / 100% / 61.1s (binary) | 0.00% / 96% / 60s (binary) |
| T0 | D1+D2 | 0.00% / 100% / 0.105s | 0.00% / 100% / 1.26s | 0.00% / 68% / 60s (domain_wall) | 0.00% / 95% / 61.5s (domain_wall) | 0.00% / 37% / 60s (domain_wall) |
| T0 | D1+D2+D3 | 0.00% / 100% / 0.0262s | 0.00% / 100% / 0.0389s | 0.00% / 28% / 78.3s (binary) | 0.00% / 11% / 316s (binary) | 0.00% / 7% / 167s (domain_wall) |
| T0 | D1+D2+D3+D4 | 0.00% / 100% / 0.111s | 0.00% / 100% / 0.267s | 0.00% / 21% / 50.7s (one_hot) | 0.00% / 11% / 282s (one_hot) | 0.00% / 3% / 86.5s (one_hot) |
| T0 | D1+D2+D4 | 0.00% / 100% / 0.0666s | 0.00% / 100% / 0.143s | 0.00% / 63% / 60s (binary) | 0.00% / 94% / 51.3s (binary) | 0.00% / 42% / 60s (domain_wall) |
| T0 | D1+D3 | 0.00% / 100% / 0.0398s | 0.00% / 100% / 0.0412s | 0.00% / 50% / 60.8s (binary) | 0.37% / 16% / 213s (binary) | 0.00% / 5% / 101s (binary) |
| T0 | D1+D3+D4 | 0.00% / 100% / 0.132s | 0.00% / 100% / 0.0976s | 0.18% / 38% / 257s (binary) | 2.14% / 16% / 947s (binary) | 0.70% / 4% / 335s (binary) |
| T0 | D1+D4 | 0.00% / 100% / 0.0918s | 0.00% / 100% / 0.136s | 0.00% / 100% / 60s (binary) | 0.00% / 100% / 61.1s (binary) | 0.00% / 96% / 60s (binary) |
| T0 | D2 | 0.00% / 100% / 0.572s | 0.00% / 100% / 1.26s | 0.00% / 72% / 60s (binary) | 0.00% / 82% / 63.1s (one_hot) | 0.00% / 20% / 60s (domain_wall) |
| T0 | D2+D3 | 0.00% / 100% / 0.0416s | 0.00% / 100% / 0.096s | 0.00% / 20% / 50.7s (domain_wall) | 1.06% / 11% / 500s (one_hot) | 0.00% / 3% / 138s (domain_wall) |
| T0 | D2+D3+D4 | 0.00% / 100% / 0.148s | 0.00% / 100% / 0.452s | 0.00% / 23% / 50.6s (binary) | 0.00% / 18% / 202s (one_hot) | 0.00% / 5% / 67.3s (one_hot) |
| T0 | D2+D4 | 0.00% / 100% / 0.0529s | 0.00% / 100% / 0.158s | 0.00% / 89% / 60s (binary) | 0.00% / 100% / 61.9s (binary) | 0.00% / 45% / 60s (domain_wall) |
| T0 | D3 | 0.00% / 100% / 0.0794s | 0.00% / 100% / 0.113s | 1.44% / 47% / 424s (binary) | 8.68% / 9% / 1.79e+03s (binary) | 2.72% / 3% / 713s (binary) |
| T0 | D3+D4 | 0.00% / 100% / 0.173s | 0.00% / 100% / 0.125s | 0.00% / 54% / 144s (binary) | 2.12% / 26% / 759s (binary) | 0.00% / 10% / 214s (binary) |
| T0 | D4 | 0.00% / 100% / 0.0865s | 0.00% / 100% / 0.206s | 0.00% / 100% / 60s (binary) | 0.00% / 100% / 61.2s (binary) | 0.00% / 97% / 60s (binary) |
| T0 | none | 0.00% / 100% / 0.381s | 0.00% / 100% / 0.746s | 0.00% / 100% / 60s (binary) | 0.00% / 100% / 61.1s (binary) | 0.00% / 98% / 60s (binary) |
| T1 | D1 | 0.00% / 100% / 3.62s | 0.00% / 100% / 3.99s | 7.05% / 100% / censored, > 1.5e+04 s (binary) | 0.00% / 100% / 62.6s (domain_wall) | 2.07% / 100% / 1.1e+03s (binary) |
| T1 | D1+D2 | 0.00% / 100% / 12.8s | 0.00% / 100% / 34.4s | 12.38% / 13% / 3.16e+03s (binary) | 42.84% / 3% / 7.08e+04s (domain_wall) | 5.84% / 3% / 1.2e+03s (domain_wall) |
| T1 | D1+D2+D3 | 0.00% / 100% / 16.5s | 0.00% / 100% / 0.599s | 22.36% / 2% / censored, > 1.5e+04 s (one_hot) | 27.88% / 2% / censored, > 1.5e+04 s (binary) | 11.29% / 1% / censored, > 1.5e+04 s (binary) |
| T1 | D1+D2+D3+D4 | 0.00% / 100% / 1.36s | 0.00% / 100% / 1.78s | 20.08% / 1% / censored, > 1.5e+04 s (domain_wall) | 20.65% / 1% / censored, > 1.5e+04 s (binary) | 17.03% / 1% / 2.31e+04s (one_hot) |
| T1 | D1+D2+D4 | 0.00% / 100% / 2.95s | 0.00% / 100% / 5.47s | 6.00% / 3% / 1.27e+03s (binary) | 27.76% / 2% / censored, > 1.5e+04 s (binary) | 0.76% / 10% / 314s (one_hot) |
| T1 | D1+D3 | 0.00% / 100% / 173s | 0.00% / 100% / 0.92s | 8.69% / 99% / censored, > 1.5e+04 s (binary) | 8.86% / 97% / censored, > 1.5e+04 s (binary) | 5.87% / 64% / censored, > 1.5e+04 s (binary) |
| T1 | D1+D3+D4 | 0.00% / 100% / 41.8s | 0.00% / 100% / 0.436s | 10.17% / 26% / 7.7e+04s (domain_wall) | 8.48% / 88% / censored, > 1.5e+04 s (binary) | 5.23% / 86% / censored, > 1.5e+04 s (binary) |
| T1 | D1+D4 | 0.00% / 100% / 5.04s | 0.00% / 100% / 6.62s | 5.89% / 100% / 2.87e+03s (binary) | 0.01% / 100% / 61.9s (domain_wall) | 1.44% / 99% / 510s (binary) |
| T1 | D2 | 0.00% / 100% / 6.81s | 0.00% / 100% / 18.6s | 13.81% / 1% / censored, > 1.5e+04 s (one_hot) | 44.14% / 0% / censored, > 1.5e+04 s (domain_wall) | 2.20% / 3% / 2.11e+03s (one_hot) |
| T1 | D2+D3 | 0.00% / 100% / 0.29s | 0.00% / 100% / 0.376s | 15.02% / 1% / censored, > 1.5e+04 s (domain_wall) | 25.58% / 0% / censored, > 1.5e+04 s (one_hot) | 11.00% / 0% / censored, > 1.5e+04 s (binary) |
| T1 | D2+D3+D4 | 0.00% / 100% / 0.936s | 0.00% / 100% / 1.28s | 17.58% / 2% / censored, > 1.5e+04 s (binary) | 36.55% / 1% / censored, > 1.5e+04 s (binary) | 9.84% / 1% / censored, > 1.5e+04 s (one_hot) |
| T1 | D2+D4 | 0.00% / 100% / 4.53s | 0.00% / 100% / 10.5s | 13.79% / 0% / censored, > 1.5e+04 s (binary) | 42.85% / 0% / censored, > 1.5e+04 s (domain_wall) | 7.89% / 0% / 3.49e+04s (domain_wall) |
| T1 | D3 | 0.00% / 100% / 1.69s | 0.00% / 100% / 0.705s | 7.75% / 100% / censored, > 1.5e+04 s (binary) | 6.46% / 98% / 7.29e+04s (binary) | 5.10% / 95% / censored, > 1.5e+04 s (binary) |
| T1 | D3+D4 | 0.00% / 100% / 30.1s | 0.00% / 100% / 0.202s | 6.41% / 98% / censored, > 1.5e+04 s (binary) | 6.17% / 88% / 3.63e+04s (binary) | 4.53% / 90% / 6.95e+04s (binary) |
| T1 | D4 | 0.00% / 100% / 1.43s | 0.00% / 100% / 1.41s | 4.26% / 99% / censored, > 1.5e+04 s (binary) | 0.00% / 95% / 61.8s (domain_wall) | 0.93% / 0% / 875s (domain_wall) |
| T1 | none | 0.00% / 100% / 7.92s | 0.00% / 100% / 9.72s | 8.28% / 100% / censored, > 1.5e+04 s (binary) | 0.00% / 100% / 62.2s (domain_wall) | 2.91% / 100% / censored, > 1.5e+04 s (binary) |

## Crossover tables: CP-SAT re-run

### 0.1 s budget

| tier | dials | cpsat |
|---|---|---|
| T0 | D1 | 0.00% / 100% / 0.122s |
| T0 | D1+D2 | 0.00% / 100% / 0.0448s |
| T0 | D1+D2+D3 | 0.00% / 100% / 0.0241s |
| T0 | D1+D2+D3+D4 | 0.00% / 100% / 0.0262s |
| T0 | D1+D2+D4 | 0.00% / 100% / 0.0572s |
| T0 | D1+D3 | 0.00% / 100% / 0.0892s |
| T0 | D1+D3+D4 | 0.00% / 100% / 0.0381s |
| T0 | D1+D4 | 0.00% / 100% / 0.053s |
| T0 | D2 | 0.00% / 100% / 0.0516s |
| T0 | D2+D3 | 0.00% / 100% / 0.0335s |
| T0 | D2+D3+D4 | 0.00% / 100% / 0.0262s |
| T0 | D2+D4 | 0.00% / 100% / 0.0847s |
| T0 | D3 | 0.00% / 100% / 0.132s |
| T0 | D3+D4 | 0.00% / 100% / 0.0588s |
| T0 | D4 | 0.00% / 100% / 0.12s |
| T0 | none | 0.00% / 100% / 0.231s |
| T1 | D1 | - / 0% / censored, > 2.5 s |
| T1 | D1+D2 | 1.39% / 100% / 2.66s |
| T1 | D1+D2+D3 | 48.71% / 44% / 5.24s |
| T1 | D1+D2+D3+D4 | 31.57% / 64% / 1.52s |
| T1 | D1+D2+D4 | 3.99% / 100% / 2.59s |
| T1 | D1+D3 | - / 0% / censored, > 2.5 s |
| T1 | D1+D3+D4 | - / 0% / censored, > 2.5 s |
| T1 | D1+D4 | 0.06% / 28% / 2.65s |
| T1 | D2 | 2.63% / 96% / 1.4s |
| T1 | D2+D3 | 1.79% / 36% / 5.62s |
| T1 | D2+D3+D4 | 1.41% / 52% / 1.51s |
| T1 | D2+D4 | 6.35% / 100% / 0.93s |
| T1 | D3 | - / 0% / censored, > 2.5 s |
| T1 | D3+D4 | - / 0% / censored, > 2.5 s |
| T1 | D4 | 0.03% / 16% / 2.68s |
| T1 | none | 0.05% / 16% / 3.06s |

### 1 s budget

| tier | dials | cpsat |
|---|---|---|
| T0 | D1 | 0.00% / 100% / 0.106s |
| T0 | D1+D2 | 0.00% / 100% / 0.0426s |
| T0 | D1+D2+D3 | 0.00% / 100% / 0.0255s |
| T0 | D1+D2+D3+D4 | 0.00% / 100% / 0.0254s |
| T0 | D1+D2+D4 | 0.00% / 100% / 0.0746s |
| T0 | D1+D3 | 0.00% / 100% / 0.0605s |
| T0 | D1+D3+D4 | 0.00% / 100% / 0.0373s |
| T0 | D1+D4 | 0.00% / 100% / 0.0804s |
| T0 | D2 | 0.00% / 100% / 0.0591s |
| T0 | D2+D3 | 0.00% / 100% / 0.0305s |
| T0 | D2+D3+D4 | 0.00% / 100% / 0.0268s |
| T0 | D2+D4 | 0.00% / 100% / 0.0575s |
| T0 | D3 | 0.00% / 100% / 0.0772s |
| T0 | D3+D4 | 0.00% / 100% / 0.0458s |
| T0 | D4 | 0.00% / 100% / 0.0955s |
| T0 | none | 0.00% / 100% / 0.162s |
| T1 | D1 | 0.00% / 100% / 0.764s |
| T1 | D1+D2 | 0.10% / 100% / 2.86s |
| T1 | D1+D2+D3 | 0.00% / 100% / 1.01s |
| T1 | D1+D2+D3+D4 | 0.00% / 100% / 0.306s |
| T1 | D1+D2+D4 | 0.00% / 100% / 0.83s |
| T1 | D1+D3 | 0.15% / 100% / 4.82s |
| T1 | D1+D3+D4 | 11.23% / 100% / censored, > 25 s |
| T1 | D1+D4 | 0.00% / 100% / 0.601s |
| T1 | D2 | 0.26% / 100% / 1.43s |
| T1 | D2+D3 | 0.00% / 100% / 0.392s |
| T1 | D2+D3+D4 | 0.00% / 100% / 0.5s |
| T1 | D2+D4 | 0.39% / 100% / 2.44s |
| T1 | D3 | 0.00% / 100% / 0.59s |
| T1 | D3+D4 | 4.21% / 100% / 54.4s |
| T1 | D4 | 0.00% / 100% / 0.507s |
| T1 | none | 0.00% / 100% / 0.657s |

### 10 s budget

| tier | dials | cpsat |
|---|---|---|
| T0 | D1 | 0.00% / 100% / 0.0919s |
| T0 | D1+D2 | 0.00% / 100% / 0.0514s |
| T0 | D1+D2+D3 | 0.00% / 100% / 0.0234s |
| T0 | D1+D2+D3+D4 | 0.00% / 100% / 0.0281s |
| T0 | D1+D2+D4 | 0.00% / 100% / 0.0721s |
| T0 | D1+D3 | 0.00% / 100% / 0.0492s |
| T0 | D1+D3+D4 | 0.00% / 100% / 0.0408s |
| T0 | D1+D4 | 0.00% / 100% / 0.084s |
| T0 | D2 | 0.00% / 100% / 0.0549s |
| T0 | D2+D3 | 0.00% / 100% / 0.0311s |
| T0 | D2+D3+D4 | 0.00% / 100% / 0.0265s |
| T0 | D2+D4 | 0.00% / 100% / 0.0577s |
| T0 | D3 | 0.00% / 100% / 0.0746s |
| T0 | D3+D4 | 0.00% / 100% / 0.0474s |
| T0 | D4 | 0.00% / 100% / 0.0943s |
| T0 | none | 0.00% / 100% / 0.19s |
| T1 | D1 | 0.00% / 100% / 1.21s |
| T1 | D1+D2 | 0.00% / 100% / 4.6s |
| T1 | D1+D2+D3 | 0.00% / 100% / 4.73s |
| T1 | D1+D2+D3+D4 | 0.00% / 100% / 0.304s |
| T1 | D1+D2+D4 | 0.00% / 100% / 2.55s |
| T1 | D1+D3 | 0.00% / 100% / 32s |
| T1 | D1+D3+D4 | 10.34% / 100% / censored, > 250 s |
| T1 | D1+D4 | 0.00% / 100% / 1.4s |
| T1 | D2 | 0.00% / 100% / 3.29s |
| T1 | D2+D3 | 0.00% / 100% / 0.393s |
| T1 | D2+D3+D4 | 0.00% / 100% / 0.49s |
| T1 | D2+D4 | 0.00% / 100% / 4.24s |
| T1 | D3 | 0.00% / 100% / 0.645s |
| T1 | D3+D4 | 4.21% / 100% / 511s |
| T1 | D4 | 0.00% / 100% / 0.621s |
| T1 | none | 0.00% / 100% / 1.25s |

### 60 s budget

| tier | dials | cpsat |
|---|---|---|
| T0 | D1 | 0.00% / 100% / 0.103s |
| T0 | D1+D2 | 0.00% / 100% / 0.0492s |
| T0 | D1+D2+D3 | 0.00% / 100% / 0.0246s |
| T0 | D1+D2+D3+D4 | 0.00% / 100% / 0.0261s |
| T0 | D1+D2+D4 | 0.00% / 100% / 0.0745s |
| T0 | D1+D3 | 0.00% / 100% / 0.0528s |
| T0 | D1+D3+D4 | 0.00% / 100% / 0.0383s |
| T0 | D1+D4 | 0.00% / 100% / 0.0817s |
| T0 | D2 | 0.00% / 100% / 0.074s |
| T0 | D2+D3 | 0.00% / 100% / 0.0296s |
| T0 | D2+D3+D4 | 0.00% / 100% / 0.0265s |
| T0 | D2+D4 | 0.00% / 100% / 0.0513s |
| T0 | D3 | 0.00% / 100% / 0.0712s |
| T0 | D3+D4 | 0.00% / 100% / 0.0462s |
| T0 | D4 | 0.00% / 100% / 0.114s |
| T0 | none | 0.00% / 100% / 0.16s |
| T1 | D1 | 0.00% / 100% / 1.11s |
| T1 | D1+D2 | 0.00% / 100% / 4.33s |
| T1 | D1+D2+D3 | 0.00% / 100% / 18.6s |
| T1 | D1+D2+D3+D4 | 0.00% / 100% / 0.326s |
| T1 | D1+D2+D4 | 0.00% / 100% / 2.24s |
| T1 | D1+D3 | 0.00% / 100% / 123s |
| T1 | D1+D3+D4 | 9.72% / 100% / censored, > 1.5e+03 s |
| T1 | D1+D4 | 0.00% / 100% / 1.52s |
| T1 | D2 | 0.00% / 100% / 3.21s |
| T1 | D2+D3 | 0.00% / 100% / 0.404s |
| T1 | D2+D3+D4 | 0.00% / 100% / 0.492s |
| T1 | D2+D4 | 0.00% / 100% / 4.36s |
| T1 | D3 | 0.00% / 100% / 0.684s |
| T1 | D3+D4 | 4.20% / 100% / 3.05e+03s |
| T1 | D4 | 0.00% / 100% / 0.576s |
| T1 | none | 0.00% / 100% / 1.28s |

## Bracket check

### The study (`4d420cf`)

Compared 2830 CP-SAT runs on 160 instances with the reference bracket. Runs whose brackets do not meet it:

| tier | dials | instance_seed | solver_seed | budget_s | lower | upper | ref_lower | ref_upper |
|---|---|---|---|---|---|---|---|---|
| T0 | D2 | 2030 | 3 | 60.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 2 | 60.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 4 | 60.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 3 | 1.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 0 | 10.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 0 | 60.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 1 | 1.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 1 | 10.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 2 | 1.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 2 | 10.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 4 | 1.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 3 | 0.1 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 4 | 0.1 | 15.525815957125278 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 3 | 10.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 4 | 10.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 0 | 1.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 2 | 0.1 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |
| T0 | D2 | 2030 | 1 | 60.0 | 17.981994981817508 | 17.981995052738604 | 14.343346668677787 | 14.343346668677789 |

### The re-run (`feb00ff`)

Compared 2963 CP-SAT runs on 160 instances with the reference bracket. Runs whose brackets do not meet it:

_none_
