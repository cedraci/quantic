# M5 QAOA on IBM Quantum — result

**Written:** 2026-10-27, after two hardware passes; the third pass and the reps = 1 rung are added below as they finish
**Branch:** `feat/m5-qaoa-ibm`
**Spec:** `2026-10-09-m5-qaoa-ibm-design.md`
**Plan:** `docs/superpowers/plans/2026-10-09-m5-qaoa-ibm.md`, Task 14

---

## Verdict

**QAOA on IBM's `ibm_fez` does not compete anywhere on T0. The device's output is barely distinguishable from uniform random bits, and even the noiseless QAOA loses to every classical sampler at the same number of reads.**

- **On hardware,** 60 jobs of 1,000 shots (two per instance, 30 instances, 132 s of QPU time) returned a feasible schedule in 0.47% of shots; 18 of the 60 jobs returned none. The best feasible sample was within 1% of the optimum in 8 jobs. A 1,000‑shot job of *uniform random bitstrings* would have managed about 4 of those 8 (section 3): the hardware's feasible rate is 1.8 times the uniform rate (Poisson p ≈ 4 × 10⁻¹⁹, so not noise alone), against 50 times for the same circuit simulated without noise. All 8 successes are on the 10 shallowest circuits (≤ 1,048 two‑qubit gates); the 40 jobs above 1,100 gates found nothing.
- **Noiselessly,** the reps = 2 CVaR‑QAOA at its trained angles reaches 1% of the optimum in 46% of 1,000‑shot runs on the same 30 instances, with 7.5% feasible samples. At 1,000 reads, simulated annealing reaches it in 54–100% of runs with 69–100% feasible samples, greedy descent in 88–100%, tabu in 98–100%. HiGHS proves the optimum on every cell in 0.24–0.40 s.
- **Time to solution.** On the four cells where any hardware job succeeded, TTS99 at 1% is 13–87 s of QPU time (D1+D2+D4 13.3 s, D1+D2 24.5 s, D2 40 s, D1+D4 87 s); on the other four it is censored. That counts QPU seconds only: each instance also cost 6 s to 90 min of classical training (17.3 h in all) and 1 s to 11 min of queue. Greedy at 1,000 reads: 0.17–0.74 s. Annealing: 0.6–6.7 s.
- **The noise model predicts this.** Aer with the `fake_fez` twin puts the feasible rate at 0.2–4.9% and, on the 17 instances it could simulate in full, the success rate at 0.37 against 0.68 noiseless and 0.24 on the device.
- **Depth is the limit, not the dynamic range.** Routed circuits carry 735–2,044 CZ gates (depth 665–1,563) on a device whose median CZ error is 0.25%: at those counts 16% down to 0.6% of shots are free of a two‑qubit error, before readout. The QUBOs' dynamic range (10⁴ to 4.5 × 10⁶) shows no measurable effect on the noiseless success once the qubit count is accounted for (section 5).
- **Two gaps in the simulations, both reported, neither affecting the hardware result:** 30 full noisy runs and one gate were killed at the 2 h limit, all circuits whose routing spread them over 23–29 physical qubits (section 7); and the one hardware task whose gate was killed (`T0‑D2+D4‑2030`) never ran.

The parent question — where does quantum compete with the best free classical solvers on discrete liquidation — gets, for gate‑model QAOA on a 2026 Heron processor, the answer "nowhere we can reach": not at 16 qubits, where the feasible search space holds 12 schedules and random sampling already finds the optimum a fifth of the time; and not at 24, where the circuit is lost in noise.

## What was run

| what | main study | smoke | reps = 1 rung |
|---|---|---|---|
| config | `experiments/m5-qaoa.toml` | `results/m5-smoke.toml` (one task, 20 s cap) | `experiments/m5-qaoa-p1.toml` |
| commit | `b8c1244` (clean) | `55c1009` (clean) | `d9a3de6` (clean) |
| lock hash | `dd3493365dfe6fc5bff5f6431c95b9cff8af7dfd830c571aecd496a17a6e610f` | same | same |
| tasks | 6,184: 6,153 simulated, 31 hardware | 6 | 1,334: 1,303 simulated, 31 hardware |
| killed | 31 (30 full noisy runs, 1 gate) | 0 | pending |
| errors | 0 | 0 | pending |
| wall clock, simulation | about 56 h of running, 2026‑10‑12 13:35 → 2026‑10‑24 18:13, in three stretches (stopped twice by the machine running out of memory, once by the owner); 14 workers for the classical phase, 2 for training and sampling | 2026‑10‑14 | started 2026‑10‑27 17:05 |
| hardware lane | 2026‑10‑27 16:13–16:52 (pass 1, 30 jobs) and 16:55–17:14 (pass 2, 30 jobs) | 2026‑10‑14 20:04 submitted, collected 2026‑10‑22 | pending |
| QPU seconds | 132 (2–3 s per job) | 2 | pending |
| store | 80 MB DuckDB, not committed | small, not committed | not committed |

Machine: Windows 11 (10.0.26200), Intel Core i9‑9900 at 3.10 GHz, 63.9 GB, Python 3.11; qiskit 2.5.2, qiskit‑aer 0.17.2, qiskit‑ibm‑runtime 0.50.0, numpy 2.4.6, scipy 1.17.1, OR‑Tools 9.15.6755, DuckDB 1.5.6.

Device: `ibm_fez`, Heron r2, 156 qubits, CZ/RZ/SX/X basis, Open plan. Calibration as read after the lane (2026‑10‑27 16:32): median CZ error 0.25%, SX 0.03%, readout 1.1%. The gate routes on the fake twin (`FakeFez`), the lane on the live backend; where both were checked the routed counts agree within a few gates (`T0‑D2+D4‑2029`: 735 CZ and depth 770 on the twin, 735 and 765 live).

Instances: T0 (3 assets × 4 buckets), the eight dial combinations without CVaR (D3), seeds 2026–2030: 40 instances, encoded in binary and domain‑wall at eight zeta bits (no zeta register without CVaR). Caps: `max_qubits = 24` keeps 31 of the 40 binary QUBOs and 16 of the 40 domain‑wall ones (876 quantum tasks left out); `noisy_max_qubits = 20` keeps 27 instances for full‑shot noisy runs (100 left out). Angles: CVaR‑QAOA, reps 2, α = 0.25, COBYLA 200 evaluations on an exact statevector, one training per distinct QUBO.

| solver | runs | per instance |
|---|---|---|
| `qaoa_ideal` | 940 | 20 seeds × (31 binary + 16 domain‑wall QUBOs), 1,000 shots |
| `qaoa_noisy`, full | 135 (105 ok, 30 killed) | 5 seeds × 27 QUBOs of ≤ 20 qubits, 1,000 shots |
| `qaoa_noisy`, gate | 31 (30 ok, 1 killed) | one 32‑shot run per hardware task |
| `qaoa_ibm` | 60 (one task skipped) | 2 jobs × 30 instances, 1,000 shots, binary |
| `simulated_annealing`, `tabu`, `greedy` | 1,600 each | 2 encodings × 20 seeds × 40 instances, 1,000 reads, no clock |
| `exact_pwl_milp` | 40 | 1 s budget |
| `ref:qaoa_angles` | 47 | one per distinct QUBO |
| `ref:brute_force`, `ref:exact_pwl_milp`, `ref:qubo_minimum` | 40, 40, 80 | no budget |

**The references.** No reference is missing, every bracket is closed, and `encoding_regret` is 0 on every QAOA cell: the encodings represent the optimum, so a gap is the sampler's. There is no CP‑SAT in this study, so the bracket check compares nothing.

**The quota.** The ledger (`~/.quantic/ibm-quota.json`) holds 61 jobs, all `done`, 134 s used of the 300 s cap (the 60 study jobs and the smoke job). IBM's own counter agrees. The 60 s per‑job limit never bound: every job used 2 or 3 s. Queue times were 1–8 s for 59 jobs and 11 min for one; the first two jobs of the lane took about 14 min each from submission to collection, the rest about half a minute. All 60 usage figures were final when read, so the report's QPU sums are complete.

**Reproduce.** `quantic bench report <store> --out <dir>` regenerates every table below from a store; either store regenerates from its commit with `quantic bench run <config> --store <path>` (the hardware lane needs a saved IBM account and the owner's go‑ahead). `experiments/m5-uniform-baseline.py <store> <out.csv>` regenerates section 3.

### How to read the numbers

- **Success at ε** is the share of runs whose best feasible sample is within ε of the proven optimum: (total − optimum) / |optimum| ≤ ε. A run with no feasible sample fails.
- **Feasibility** is the share of samples that decode to a schedule satisfying every constraint. Nothing is repaired.
- **TTS99** is t_run · ln 0.01 / ln(1 − p). For `qaoa_ibm`, t_run is the job's QPU seconds; for the simulators it is not reported (decision D6: simulation time is not quantum time).
- Each QAOA run is one 1,000‑shot circuit at fixed trained angles; its seed changes only the shot sampling.

## Findings the plan asked for

### 1. Matched reads: QAOA against the samplers at 1,000 reads

Success at ε = 1% / feasible samples, binary encoding, every solver at 1,000 reads. The samplers are averaged over 100 runs per cell (5 instances × 20 seeds), the noiseless QAOA over 20 runs per instance, the noise model over 5, the hardware over 2.

| cell | greedy | simulated annealing | tabu | QAOA, noiseless | QAOA, noise model | QAOA, `ibm_fez` |
|---|---|---|---|---|---|---|
| `D1` | 0.97 / 96% | 0.54 / 100% | 1.00 / 100% | 0.03 / 7.6% | — (24 qubits) | 0 of 10 / 0.07% |
| `D1+D2` | 0.90 / 35% | 0.99 / 69% | 1.00 / 96% | 0.58 / 2.3% | 0.10 / 1.2% (10 of 20 runs killed) | 3 of 8 / 0.73% |
| `D1+D2+D4` | 1.00 / 41% | 1.00 / 69% | 0.99 / 95% | 1.00 / 14% | 0.93 / 3.3% | 3 of 6 / 1.0% |
| `D1+D4` | 1.00 / 96% | 0.94 / 100% | 1.00 / 100% | 0.56 / 8.3% | 0.60 / 1.6% | 1 of 10 / 0.33% |
| `D2` | 0.88 / 20% | 1.00 / 76% | 0.98 / 98% | 0.73 / 5.8% | 0.60 / 1.3% | 1 of 4 / 0.53% |
| `D2+D4` | 0.95 / 45% | 0.90 / 92% | 1.00 / 100% | 0.58 / 5.6% | all 5 runs killed | 0 of 2 / 1.2% |
| `D4` | 1.00 / 96% | 0.76 / 100% | 1.00 / 100% | 0.27 / 6.8% | 0.00 / 2.0% | 0 of 10 / 0.59% |
| `none` | 1.00 / 98% | 0.75 / 100% | 1.00 / 100% | 0.32 / 8.5% | 0.20 / 1.3% | 0 of 10 / 0.16% |

TTS99 at ε = 1%, seconds, binary encoding:

| cell | greedy | simulated annealing | tabu | HiGHS (proof, 1 s budget) | QAOA, `ibm_fez` (QPU seconds) |
|---|---|---|---|---|---|
| `D1` | 0.74 | 6.65 | 41.6 | 0.24 | censored |
| `D1+D2` | 0.30 | 0.63 | 58.7 | 0.37 | 24.5 |
| `D1+D2+D4` | 0.17 | 0.93 | 61.4 | 0.35 | 13.3 |
| `D1+D4` | 0.36 | 1.44 | 42.2 | 0.26 | 87.4 |
| `D2` | 0.47 | 0.83 | 74.7 | 0.40 | 40.0 |
| `D2+D4` | 0.42 | 1.84 | 63.9 | 0.38 | censored |
| `D4` | 0.35 | 3.45 | 41.8 | 0.36 | censored |
| `none` | 0.34 | 3.52 | 42.3 | 0.39 | censored |

**The noiseless QAOA already loses.** Its weakness is feasibility: the penalty‑encoded constraints leave 2–14% of its shots feasible where annealing has 69–100%, so even where it finds the optimum (every `D1+D2+D4` run) it does so from a handful of feasible samples. It is competitive only on the three 16‑qubit cells with the cardinality dial, whose feasible search space is tiny (section 3), and weakest on `D1`, `D4` and `none` at 20–24 qubits.

**The hardware loses to the noiseless circuit by another factor of 3 to 10** in both success and feasibility, and its QPU‑only TTS99, where it exists at all, is 10–100 times greedy's wall‑clock TTS99 before counting training and queue.

### 2. The noise ladder: noiseless → noise model → device

Per instance, binary encoding, success at ε = 1% (and feasible samples), over the 17 hardware instances the noise model could simulate in full:

| instance | qubits | CZ, routed | noiseless (20 runs) | noise model (5 runs) | `ibm_fez` (2 jobs) |
|---|---|---|---|---|---|
| `D1+D2‑2026` | 16 | 758 | 0.70 (1.0%) | 0.20 (0.4%) | 1.00 (0.55%) |
| `D1+D2‑2027` | 20 | 1,262 | 0.70 (2.5%) | 0 (all killed) | 0 (0.05%) |
| `D1+D2‑2028` | 20 | 1,268 | 0.60 (1.1%) | 0 (all killed) | 0 (0%) |
| `D1+D2‑2030` | 20 | 915 | 0.30 (4.5%) | 0.20 (2.1%) | 0.50 (2.3%) |
| `D1+D2+D4‑2026` | 16 | 736 | 1.00 (9.8%) | 1.00 (2.3%) | 0.50 (0.3%) |
| `D1+D2+D4‑2027` | 16 | 774 | 1.00 (21%) | 1.00 (4.9%) | 0 (1.65%) |
| `D1+D2+D4‑2028` | 16 | 736 | 1.00 (11%) | 0.80 (2.7%) | 1.00 (1.1%) |
| `D1+D4‑2027` | 18 | 873 | 1.00 (5.6%) | 0.80 (1.4%) | 0 (0.2%) |
| `D1+D4‑2030` | 19 | 1,048 | 1.00 (11%) | 0.40 (1.9%) | 0.50 (0.6%) |
| `D2‑2026` | 16 | 736 | 1.00 (10%) | 1.00 (2.3%) | 0.50 (0.9%) |
| `D2‑2030` | 20 | 1,168 | 0.45 (1.7%) | 0.20 (0.2%) | 0 (0.15%) |
| `D2+D4‑2029` | 16 | 735 | 1.00 (6.1%) | 0 (all killed) | 0 (1.2%) |
| `D4‑2027` | 19 | 1,233 | 0.15 (8.7%) | 0.00 (2.8%) | 0 (1.75%) |
| `D4‑2030` | 18 | 962 | 0.00 (3.0%) | 0.00 (1.2%) | 0 (0.6%) |
| `none‑2027` | 20 | 1,219 | 0.50 (7.2%) | 0.20 (0.6%) | 0 (0.05%) |
| `none‑2028` | 20 | 1,261 | 0.15 (15%) | 0.00 (2.2%) | 0 (0.6%) |
| `none‑2030` | 20 | 1,232 | 0.95 (6.5%) | 0.40 (1.1%) | 0 (0.15%) |
| **mean** | | | **0.68** | **0.37** (killed cells as 0) | **0.24** |

Over all 30 hardware instances: noiseless 0.46 success and 7.5% feasibility, device 0.13 and 0.47%. The 13 instances the noise model could not run in full are the 24‑qubit ones (`D1` × 5, `none‑2026`, `none‑2029`) and the 21–22‑qubit ones (`D1+D4` × 3, `D4` × 3): the device succeeded on none of their 26 jobs.

The noise model is a fair guide to the device: it puts the feasible rate in the right band (0.2–4.9% against 0–2.3% on the device) and, instance by instance, its success rate is above the device's on 7, below on 4 and equal on 6. With two jobs per instance the per‑instance comparison is coarse; the ladder's means and the pooled numbers of section 3 are the robust reading.

**Depth decides.** Every one of the 8 hardware successes sits on the 10 circuits with at most 1,048 CZ gates (20 jobs); the 20 circuits with 1,100–2,044 gates (40 jobs) produced none. At the device's median CZ error of 0.25% a shot passes 735 gates without a two‑qubit error 16% of the time and 2,044 gates 0.6% of the time, before the 1.1% readout error on each of 16–24 qubits. The design's risk table expected about 2% error‑free shots at 20 qubits; that is what the device delivers.

### 3. Against uniform random bits

Is the device's output distinguishable from random bitstrings? For each hardware instance, every one of the 2¹⁶ bitstrings (16 qubits) or 100,000 uniform random ones (18–24 qubits) was passed through the study's own decoder and scorer, giving the share of bitstrings that decode to a feasible schedule and, where exact, how many sit within 1% of the optimum.

- **Feasibility.** The 60 hardware jobs returned 279 feasible samples in 60,000 shots. Uniform bitstrings would return 156 (per‑instance rates 0.001%–2.5%). The ratio is 1.8 (Poisson tail p ≈ 4 × 10⁻¹⁹), so the output is not uniform: a trace of the circuit survives. The noiseless circuit's ratio is 50 (median over instances; 1.8 to 4,500). Per instance, the device's ratio is 0–13 (median 1.3), the noiseless circuit's 1.8–4,500.
- **Success.** The six 16‑qubit instances have 192 or 384 feasible bitstrings out of 65,536, which decode to just 12 or 24 distinct feasible schedules, with one optimum (16 bitstrings). A uniform 1,000‑shot job therefore contains the optimum with probability 1 − (1 − 16/65,536)¹⁰⁰⁰ = 22%. The device managed 6 of its 12 jobs there, against 2.6 expected by chance (binomial p = 0.03); the noiseless circuit 70–100%. On the 24 instances of 18–24 qubits the device succeeded in 2 of 48 jobs, against about 1.1 expected by chance, almost all of it on `D1+D2‑2030`, where 2.5% of all bitstrings are feasible and a uniform 1,000‑shot job finds the optimum 39% of the time (the noiseless circuit: 30%, the device: 1 of 2).

So the device keeps a little of the circuit's structure (feasibility 1.8 × uniform) and almost none of its optimisation power (noiseless: 50 × uniform; 46% success against 13% on the device, of which about 6 points are what chance gives on these small instances). The per‑instance table is in the appendix.

### 4. Training cost against qubit count

47 angle sets (31 binary, 16 domain‑wall QUBOs), 17.3 h of CPU in all. Each COBYLA evaluation is an exact statevector CVaR at α = 0.25:

| qubits | trainings | median | max |
|---|---|---|---|
| 16 | 12 | 6 s | 7 s |
| 18 | 2 | 55 s | 66 s |
| 19 | 2 | 128 s | 153 s |
| 20 | 11 | 214 s | 333 s |
| 21 | 3 | 603 s | 604 s |
| 22 | 4 | 904 s | 933 s |
| 23 | 1 | 2,629 s | — |
| 24 | 12 | 3,961 s (66 min) | 5,421 s (90 min) |

About 2.3× per added qubit, 1.4 GB at 24 qubits. The dry run measured 28 qubits at 413 s per evaluation and 20 GB (23 h per QUBO), which is why the study stops at 24. COBYLA used all 200 evaluations on 45 of the 47 trainings (two stopped at 181), so the angles are where the budget ran out, not where the search converged; `reps`, `maxiter` and `alpha` are the first things a follow‑up would vary. Against this, HiGHS proves the optimum of every instance in under 0.4 s: the classical pre‑processing alone costs 10 to 10⁴ times the classical solve.

### 5. The dynamic‑range effect

The QUBOs' dynamic range (largest to smallest non‑zero coefficient) runs from 1.0 × 10⁴ to 4.5 × 10⁶, the scaled Ising forms' from 2.5 × 10⁴ to 5.0 × 10⁶. The design's risk was that a range near 5 × 10⁵ hides small terms from QAOA. The data do not show it:

- Over the 31 binary QUBOs, Spearman's correlation between log range and noiseless success is −0.16 (p = 0.38); within the 16‑, 20‑ and 24‑qubit groups it is +0.39, +0.11 and +0.11, none significant. Qubit count alone correlates at −0.77 (p = 4 × 10⁻⁷).
- Per dial, `D1+D4` has the largest median range (3.1 × 10⁶) and succeeds in 56% of noiseless runs at 20 qubits; `D4` has a range four times smaller and succeeds in 27%; `D1` (1.9 × 10⁶) succeeds in 3%, but it is also the one dial at 24 qubits in every seed.
- Against the device's success the correlation is −0.35 (p = 0.05), which is the gate count in disguise: the widest ranges belong to the `D1` instances at 24 qubits, which are also the deepest circuits.

The range enters the angles' training (an exact Ising energy) rather than the device, which runs the same circuit whatever the coefficients. Where a range of 10⁶ would matter is on an analogue annealer, which M5 could not run.

### 6. How far the hardware lane got within 300 s

Two full passes over the 30 runnable tasks, 60 jobs, 132 s of QPU time, 2 or 3 s per job (1,000 shots, `max_execution_time` 60 s). The cap never bound; the lane stopped because its list was done. The third pass is pending with 166 s of headroom. Wall time was 39 min for the first pass (its first two jobs took 14 min each, one of them queued for 11 min) and 19 min for the second. The smoke job of 2026‑10‑14 (`db7s898r1c5c738l6su0`, the `none‑2026` instance, 2 s) sits in the same ledger; it ran against its own store, so that instance was paid for twice, 4 s in all.

One task never ran: `T0‑D2+D4‑2030` (24 qubits), because its gate was killed (section 7) and the lane refuses a task without a passed gate.

### 7. The kills

31 tasks reached the 2 h limit of a quantum run: the 30 full noisy runs on `D1+D2‑2027`, `D1+D2‑2028` and `D2+D4‑2029` (binary and domain‑wall, five seeds each) and the gate of `D2+D4‑2030`. All are circuits whose routing on `FakeFez` spread them over more physical qubits than they have logical ones: `D2+D4‑2029` touches 23 physical qubits for 16 logical, `D1+D2‑2027` 24 for 20, `D2+D4‑2030` 29 for 24, where every circuit that ran touches exactly its logical count. The noisy simulator carries every physical qubit the circuit touches, so its cost grows exponentially with that spread: the 32‑shot gate of `D2+D4‑2029` took 250 s, which puts its 1,000‑shot run near 7,800 s. These cells are missing from the noise ladder; their hardware jobs ran regardless, since their gates had passed. They cost 62 worker‑hours, about 31 of the simulation's 56 h of wall time.

### 8. What the QAOA columns are not

The QAOA solvers sit outside the budget‑indexed crossover tables (section 10 of the design): a QAOA run is a shot count at fixed angles, not a time budget, so it has no 0.1 s, 1 s, 10 s or 60 s column and no crossover row. The comparison that holds is section 1, at matched reads, and the TTS99 of `qaoa_ibm` is QPU seconds only. A wall‑clock TTS99 would add the training (6 s to 90 min per instance) and the queue (1 s to 14 min per job) and would be larger by one to four orders of magnitude.

## The reps = 1 rung

Pending: `experiments/m5-qaoa-p1.toml` at `d9a3de6` is running. This section will report the same circuits at one QAOA layer (about half the CZ count after routing), on the simulators and on `ibm_fez`.

## What next should take from it

- **Gate‑model QAOA with penalty‑encoded constraints is not a solver for these problems, with or without noise.** Feasibility is the failure: 2–14% feasible shots noiselessly, under 1% on the device, against 69–100% for annealing. Any further gate‑model attempt should change the algorithm before the device: constraint‑preserving circuits (XY or ring mixers on one‑hot or domain‑wall registers, Grover‑type mixers) that never leave the feasible subspace, so every shot is a candidate.
- **735 two‑qubit gates is already too deep at a 0.25% CZ error.** The reps = 1 rung (pending) says whether half the depth helps. Beyond that, nothing in this study's reach — not more shots, not more replicates, not error mitigation designed for expectation values — turns a near‑uniform sample distribution into a solver.
- **The classical side of QAOA is itself the bottleneck.** 66 min of exact‑statevector training at 24 qubits, 23 h at 28, against 0.4 s for a proof. Any larger instance needs a trainer that is not a statevector: fixed angle schedules, transfer of angles from smaller instances, or training on the device, each of which gives up the exact CVaR.
- **The annealing hardware the parent spec aimed at remains untested.** The QUBOs, the three encodings, the matched‑reads harness and the quota ledger are ready for it; a D‑Wave Leap account is what is missing. There, unlike here, the dynamic range of 10⁴–10⁶ would matter (analogue precision), and the 20–50× energy gaps the M4 samplers show on CVaR cells are the thing to measure.
- **T1 and beyond are out of reach for a gate device at this fidelity.** T1 QUBOs exceed 40 qubits and would route to several thousand CZ gates; the M4 result's "look at T2–T3" applies to annealers and classical methods, not to QAOA on Heron.
- **For the crossover question,** M4's answer stands and this study adds the quantum column to it: at T0, HiGHS proves the optimum in under 0.4 s, greedy finds it in under 1 s, and QAOA on hardware finds it, where it finds it at all, in 13–87 s of QPU time plus hours of preparation.

## Appendix: the uniform‑random baseline per instance

`uniform_frac` is the share of all bitstrings that decode to a feasible schedule (exact at 16 qubits, else from 100,000 draws); `exp_per_1000` the feasible samples a uniform 1,000‑shot job would give; `hw_feasible` what the two hardware jobs gave; `ratio_hw` and `ratio_ideal` the device's and the noiseless circuit's feasible rate over the uniform one; `k_within_1pct` the bitstrings within 1% of the optimum among those examined; `p_success_uniform` the chance a uniform 1,000‑shot job contains one.

<!-- uniform-table -->
| instance | qubits | bitstrings examined | uniform feasible | expected per 1,000 shots | hardware feasible per job | hardware within 1% | jobs | device / uniform | noiseless / uniform | k within 1% | p(success) uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `D1+D2-2026` | 16 | exact | 0.0029 | 2.93 | 7+4 | 2 | 2 | 1.88 | 3.45 | 16 | 0.217 |
| `D1+D2+D4-2026` | 16 | exact | 0.0029 | 2.93 | 5+1 | 1 | 2 | 1.02 | 33.5 | 16 | 0.217 |
| `D1+D2+D4-2027` | 16 | exact | 0.0059 | 5.86 | 14+19 | 0 | 2 | 2.82 | 35.4 | 16 | 0.217 |
| `D1+D2+D4-2028` | 16 | exact | 0.0029 | 2.93 | 8+14 | 2 | 2 | 3.75 | 38.6 | 16 | 0.217 |
| `D2-2026` | 16 | exact | 0.0029 | 2.93 | 9+9 | 1 | 2 | 3.07 | 34 | 16 | 0.217 |
| `D2+D4-2029` | 16 | exact | 0.0029 | 2.93 | 12+12 | 0 | 2 | 4.1 | 20.8 | 16 | 0.217 |
| `D1+D4-2027` | 18 | 100,000 draws | 0.00015 | 0.15 | 0+4 | 0 | 2 | 13.3 | 373 | 2 | 0.0198 |
| `D4-2030` | 18 | 100,000 draws | 0.0075 | 7.51 | 3+9 | 0 | 2 | 0.799 | 3.98 | 0 | 0 |
| `D1+D4-2030` | 19 | 100,000 draws | 0.00063 | 0.63 | 6+6 | 1 | 2 | 9.52 | 168 | 3 | 0.0296 |
| `D4-2027` | 19 | 100,000 draws | 0.005 | 4.99 | 14+21 | 0 | 2 | 3.51 | 17.3 | 0 | 0 |
| `D1+D2-2027` | 20 | 100,000 draws | 0.00046 | 0.46 | 1+0 | 0 | 2 | 1.09 | 53.7 | 2 | 0.0198 |
| `D1+D2-2028` | 20 | 100,000 draws | 0.00022 | 0.22 | 0+0 | 0 | 2 | 0 | 50.2 | 2 | 0.0198 |
| `D1+D2-2030` | 20 | 100,000 draws | 0.0253 | 25.3 | 26+20 | 1 | 2 | 0.908 | 1.79 | 49 | 0.387 |
| `D2-2030` | 20 | 100,000 draws | 0.00045 | 0.45 | 1+2 | 0 | 2 | 3.33 | 37.6 | 2 | 0.0198 |
| `none-2027` | 20 | 100,000 draws | 0.0002 | 0.2 | 0+1 | 0 | 2 | 2.5 | 359 | 0 | 0 |
| `none-2028` | 20 | 100,000 draws | 0.0046 | 4.56 | 8+4 | 0 | 2 | 1.32 | 32.3 | 0 | 0 |
| `none-2030` | 20 | 100,000 draws | 0.00053 | 0.53 | 1+2 | 0 | 2 | 2.83 | 122 | 2 | 0.0198 |
| `D1+D4-2026` | 21 | 100,000 draws | 0.0016 | 1.59 | 2+2 | 0 | 2 | 1.26 | 50.3 | 1 | 0.01 |
| `D1+D4-2029` | 21 | 100,000 draws | 0.00049 | 0.49 | 0+1 | 0 | 2 | 1.02 | 168 | 0 | 0 |
| `D4-2026` | 21 | 100,000 draws | 1e-05 | 0.01 | 0+0 | 0 | 2 | 0 | 4.52e+03 | 0 | 0 |
| `D1+D4-2028` | 22 | 100,000 draws | 0.0023 | 2.3 | 5+7 | 0 | 2 | 2.61 | 40.4 | 0 | 0 |
| `D4-2028` | 22 | 100,000 draws | 0.00053 | 0.53 | 6+1 | 0 | 2 | 6.6 | 158 | 0 | 0 |
| `D4-2029` | 22 | 100,000 draws | 0.0023 | 2.32 | 3+2 | 0 | 2 | 1.08 | 40 | 0 | 0 |
| `D1-2026` | 24 | 100,000 draws | 0.00029 | 0.29 | 0+0 | 0 | 2 | 0 | 229 | 0 | 0 |
| `D1-2027` | 24 | 100,000 draws | 0.0025 | 2.49 | 3+1 | 0 | 2 | 0.803 | 42.2 | 0 | 0 |
| `D1-2028` | 24 | 100,000 draws | 0.00033 | 0.33 | 0+0 | 0 | 2 | 0 | 215 | 1 | 0.01 |
| `D1-2029` | 24 | 100,000 draws | 0.00034 | 0.34 | 0+0 | 0 | 2 | 0 | 204 | 0 | 0 |
| `D1-2030` | 24 | 100,000 draws | 0.00047 | 0.47 | 2+1 | 0 | 2 | 3.19 | 142 | 0 | 0 |
| `none-2026` | 24 | 100,000 draws | 0.00018 | 0.18 | 0+0 | 0 | 2 | 0 | 385 | 0 | 0 |
| `none-2029` | 24 | 100,000 draws | 0.00089 | 0.89 | 0+0 | 0 | 2 | 0 | 80.6 | 0 | 0 |
<!-- /uniform-table -->

<!-- tables -->
## Appendix: the report's tables

Regenerated by `quantic bench report <store> --out <dir>` from the main study's store.

### Matched reads (`matched_reads.md`)

Every solver at 1,000 reads; `tts` is QPU seconds for `qaoa_ibm`, wall-clock seconds for the samplers, and "simulated" for the simulators.

| tier | dials | solver | encoding | max_samples | n_runs | p_success_0.01 | p_success_0.05 | gap_median | feasibility | tts |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | D1 | greedy | binary | 1000 | 100 | 0.97 | 1.0 | 0.0 | 0.95892 | 0.735s |
| T0 | D1 | greedy | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.5171 | 0.353s |
| T0 | D1 | qaoa_ibm | binary | 1000 | 10 | 0.0 | 0.0 | 0.3192624238476382 | 0.0007 | censored |
| T0 | D1 | qaoa_ideal | binary | 1000 | 100 | 0.03 | 0.52 | 0.04739289237474535 | 0.07568 | simulated |
| T0 | D1 | simulated_annealing | binary | 1000 | 100 | 0.54 | 0.96 | 0.009283043068350016 | 1.0 | 6.65s |
| T0 | D1 | simulated_annealing | domain_wall | 1000 | 100 | 0.55 | 0.9 | 0.008839746229817019 | 1.0 | 7.09s |
| T0 | D1 | tabu | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 41.6s |
| T0 | D1 | tabu | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 65.3s |
| T0 | D1+D2 | greedy | binary | 1000 | 100 | 0.9 | 0.99 | 0.0 | 0.35235 | 0.296s |
| T0 | D1+D2 | greedy | domain_wall | 1000 | 100 | 0.9 | 0.99 | 0.0 | 0.35259 | 0.301s |
| T0 | D1+D2 | qaoa_ibm | binary | 1000 | 8 | 0.375 | 0.375 | 0.0 | 0.00725 | 24.5s |
| T0 | D1+D2 | qaoa_ideal | binary | 1000 | 80 | 0.575 | 0.625 | 3.4737553137705954e-08 | 0.0228 | simulated |
| T0 | D1+D2 | qaoa_ideal | domain_wall | 1000 | 80 | 0.6 | 0.625 | 3.4737553137705954e-08 | 0.0225875 | simulated |
| T0 | D1+D2 | qaoa_noisy | binary | 1000 | 20 | 0.1 | 0.1 | 0.09073065530750446 | 0.0121 | simulated |
| T0 | D1+D2 | qaoa_noisy | domain_wall | 1000 | 20 | 0.1 | 0.1 | 0.09073065530750446 | 0.0121 | simulated |
| T0 | D1+D2 | simulated_annealing | binary | 1000 | 100 | 0.99 | 1.0 | 0.0 | 0.69319 | 0.63s |
| T0 | D1+D2 | simulated_annealing | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.69337 | 0.568s |
| T0 | D1+D2 | tabu | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.96031 | 58.7s |
| T0 | D1+D2 | tabu | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.95969 | 56.8s |
| T0 | D1+D2+D4 | greedy | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.4069 | 0.17s |
| T0 | D1+D2+D4 | greedy | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.41582 | 0.159s |
| T0 | D1+D2+D4 | qaoa_ibm | binary | 1000 | 6 | 0.5 | 0.5 | 0.02541473215309317 | 0.010166666666666666 | 13.3s |
| T0 | D1+D2+D4 | qaoa_ideal | binary | 1000 | 60 | 1.0 | 1.0 | 0.0 | 0.1396 | simulated |
| T0 | D1+D2+D4 | qaoa_ideal | domain_wall | 1000 | 60 | 1.0 | 1.0 | 0.0 | 0.1396 | simulated |
| T0 | D1+D2+D4 | qaoa_noisy | binary | 1000 | 15 | 0.9333333333333333 | 0.9333333333333333 | 0.0 | 0.033 | simulated |
| T0 | D1+D2+D4 | qaoa_noisy | domain_wall | 1000 | 15 | 0.9333333333333333 | 0.9333333333333333 | 0.0 | 0.033 | simulated |
| T0 | D1+D2+D4 | simulated_annealing | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.68902 | 0.934s |
| T0 | D1+D2+D4 | simulated_annealing | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.68859 | 0.818s |
| T0 | D1+D2+D4 | tabu | binary | 1000 | 100 | 0.99 | 0.99 | 0.0 | 0.94923 | 61.4s |
| T0 | D1+D2+D4 | tabu | domain_wall | 1000 | 100 | 0.99 | 0.99 | 0.0 | 0.94923 | 59.5s |
| T0 | D1+D4 | greedy | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.96325 | 0.356s |
| T0 | D1+D4 | greedy | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.78946 | 0.217s |
| T0 | D1+D4 | qaoa_ibm | binary | 1000 | 10 | 0.1 | 0.3 | 0.054096452093820274 | 0.0033 | 87.4s |
| T0 | D1+D4 | qaoa_ideal | binary | 1000 | 100 | 0.56 | 0.96 | 0.008669052890741467 | 0.08345 | simulated |
| T0 | D1+D4 | qaoa_ideal | domain_wall | 1000 | 40 | 0.9 | 1.0 | 0.0 | 0.0579 | simulated |
| T0 | D1+D4 | qaoa_noisy | binary | 1000 | 10 | 0.6 | 1.0 | 0.007070811174385927 | 0.0162 | simulated |
| T0 | D1+D4 | simulated_annealing | binary | 1000 | 100 | 0.94 | 1.0 | 0.0 | 1.0 | 1.44s |
| T0 | D1+D4 | simulated_annealing | domain_wall | 1000 | 100 | 0.95 | 1.0 | 0.0 | 1.0 | 1.53s |
| T0 | D1+D4 | tabu | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 42.2s |
| T0 | D1+D4 | tabu | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 41.6s |
| T0 | D2 | greedy | binary | 1000 | 100 | 0.88 | 0.9 | 0.0 | 0.19756 | 0.474s |
| T0 | D2 | greedy | domain_wall | 1000 | 100 | 0.88 | 0.9 | 0.0 | 0.19786 | 0.442s |
| T0 | D2 | qaoa_ibm | binary | 1000 | 4 | 0.25 | 0.25 | 0.14429685321061442 | 0.00525 | 40s |
| T0 | D2 | qaoa_ideal | binary | 1000 | 40 | 0.725 | 0.725 | 0.0 | 0.058225 | simulated |
| T0 | D2 | qaoa_ideal | domain_wall | 1000 | 40 | 0.725 | 0.725 | 0.0 | 0.05795 | simulated |
| T0 | D2 | qaoa_noisy | binary | 1000 | 10 | 0.6 | 0.6 | 0.0 | 0.0127 | simulated |
| T0 | D2 | qaoa_noisy | domain_wall | 1000 | 10 | 0.6 | 0.6 | 0.0 | 0.013 | simulated |
| T0 | D2 | simulated_annealing | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.76198 | 0.833s |
| T0 | D2 | simulated_annealing | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.76352 | 0.756s |
| T0 | D2 | tabu | binary | 1000 | 100 | 0.98 | 0.98 | 0.0 | 0.97973 | 74.7s |
| T0 | D2 | tabu | domain_wall | 1000 | 100 | 0.98 | 0.98 | 0.0 | 0.97831 | 74.3s |
| T0 | D2+D4 | greedy | binary | 1000 | 100 | 0.95 | 1.0 | 0.0 | 0.44526 | 0.416s |
| T0 | D2+D4 | greedy | domain_wall | 1000 | 100 | 0.95 | 1.0 | 0.0 | 0.44708 | 0.451s |
| T0 | D2+D4 | qaoa_ibm | binary | 1000 | 2 | 0.0 | 1.0 | 0.03296351479499938 | 0.012 | censored |
| T0 | D2+D4 | qaoa_ideal | binary | 1000 | 40 | 0.575 | 0.95 | 0.0 | 0.0557 | simulated |
| T0 | D2+D4 | qaoa_ideal | domain_wall | 1000 | 40 | 0.625 | 0.875 | 0.0 | 0.055375 | simulated |
| T0 | D2+D4 | qaoa_noisy | binary | 1000 | 5 | 0.0 | 0.0 |  | 0.0 | simulated |
| T0 | D2+D4 | qaoa_noisy | domain_wall | 1000 | 5 | 0.0 | 0.0 |  | 0.0 | simulated |
| T0 | D2+D4 | simulated_annealing | binary | 1000 | 100 | 0.9 | 1.0 | 0.0 | 0.92152 | 1.84s |
| T0 | D2+D4 | simulated_annealing | domain_wall | 1000 | 100 | 0.92 | 1.0 | 0.0 | 0.92052 | 1.4s |
| T0 | D2+D4 | tabu | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.99851 | 63.9s |
| T0 | D2+D4 | tabu | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.99851 | 63s |
| T0 | D4 | greedy | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.96438 | 0.351s |
| T0 | D4 | greedy | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.65363 | 0.235s |
| T0 | D4 | qaoa_ibm | binary | 1000 | 10 | 0.0 | 0.0 | 0.21757477677303394 | 0.0059 | censored |
| T0 | D4 | qaoa_ideal | binary | 1000 | 100 | 0.27 | 0.37 | 0.06995521088580015 | 0.06761 | simulated |
| T0 | D4 | qaoa_ideal | domain_wall | 1000 | 40 | 0.025 | 0.125 | 0.19571369588210724 | 0.0106 | simulated |
| T0 | D4 | qaoa_noisy | binary | 1000 | 10 | 0.0 | 0.0 | 0.27703706242808257 | 0.0199 | simulated |
| T0 | D4 | simulated_annealing | binary | 1000 | 100 | 0.76 | 0.96 | 0.0015235685530318106 | 1.0 | 3.45s |
| T0 | D4 | simulated_annealing | domain_wall | 1000 | 100 | 0.84 | 0.97 | 0.0015235685530318106 | 1.0 | 2.32s |
| T0 | D4 | tabu | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 41.8s |
| T0 | D4 | tabu | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 54.9s |
| T0 | none | greedy | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.97943 | 0.343s |
| T0 | none | greedy | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.59869 | 0.254s |
| T0 | none | qaoa_ibm | binary | 1000 | 10 | 0.0 | 0.0 | 0.16875504569445107 | 0.0016 | censored |
| T0 | none | qaoa_ideal | binary | 1000 | 100 | 0.32 | 0.61 | 0.03080923966770832 | 0.08503 | simulated |
| T0 | none | qaoa_ideal | domain_wall | 1000 | 20 | 0.0 | 0.0 | 0.18125706074518239 | 0.0138 | simulated |
| T0 | none | qaoa_noisy | binary | 1000 | 15 | 0.2 | 0.6666666666666666 | 0.03585092305517504 | 0.013 | simulated |
| T0 | none | simulated_annealing | binary | 1000 | 100 | 0.75 | 1.0 | 0.0 | 1.0 | 3.52s |
| T0 | none | simulated_annealing | domain_wall | 1000 | 100 | 0.8 | 1.0 | 0.0 | 1.0 | 3.04s |
| T0 | none | tabu | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 42.3s |
| T0 | none | tabu | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 46.5s |

### The noise ladder per instance (`qaoa_noise.md`)

Routed depth and two-qubit count are medians over the runs; `qpu_s` sums the hardware jobs.

| tier | dials | instance_seed | encoding | solver | n_runs | p_success_0.01 | feasibility | routed_depth | routed_2q | qpu_s |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | D1 | 2026 | binary | qaoa_ibm | 2 | 0.0 | 0.0 | 1303.0 | 1789.0 | 6.0 |
| T0 | D1 | 2026 | binary | qaoa_ideal | 20 | 0.05 | 0.06645 |  |  |  |
| T0 | D1 | 2027 | binary | qaoa_ibm | 2 | 0.0 | 0.002 | 1056.0 | 1815.0 | 4.0 |
| T0 | D1 | 2027 | binary | qaoa_ideal | 20 | 0.0 | 0.1051 |  |  |  |
| T0 | D1 | 2028 | binary | qaoa_ibm | 2 | 0.0 | 0.0 | 1092.0 | 1822.0 | 4.0 |
| T0 | D1 | 2028 | binary | qaoa_ideal | 20 | 0.0 | 0.0708 |  |  |  |
| T0 | D1 | 2029 | binary | qaoa_ibm | 2 | 0.0 | 0.0 | 1306.0 | 1865.0 | 6.0 |
| T0 | D1 | 2029 | binary | qaoa_ideal | 20 | 0.1 | 0.0695 |  |  |  |
| T0 | D1 | 2030 | binary | qaoa_ibm | 2 | 0.0 | 0.0015 | 1106.0 | 1767.0 | 4.0 |
| T0 | D1 | 2030 | binary | qaoa_ideal | 20 | 0.0 | 0.06655 |  |  |  |
| T0 | D1+D2 | 2026 | binary | qaoa_ibm | 2 | 1.0 | 0.0055 | 852.0 | 758.0 | 4.0 |
| T0 | D1+D2 | 2026 | binary | qaoa_ideal | 20 | 0.7 | 0.0101 |  |  |  |
| T0 | D1+D2 | 2026 | binary | qaoa_noisy | 5 | 0.2 | 0.0036 | 852.0 | 758.0 |  |
| T0 | D1+D2 | 2026 | domain_wall | qaoa_ideal | 20 | 0.7 | 0.0101 |  |  |  |
| T0 | D1+D2 | 2026 | domain_wall | qaoa_noisy | 5 | 0.2 | 0.0036 | 852.0 | 758.0 |  |
| T0 | D1+D2 | 2027 | binary | qaoa_ibm | 2 | 0.0 | 0.0005 | 1146.0 | 1262.0 | 6.0 |
| T0 | D1+D2 | 2027 | binary | qaoa_ideal | 20 | 0.7 | 0.0247 |  |  |  |
| T0 | D1+D2 | 2027 | binary | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D1+D2 | 2027 | domain_wall | qaoa_ideal | 20 | 0.75 | 0.024 |  |  |  |
| T0 | D1+D2 | 2027 | domain_wall | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D1+D2 | 2028 | binary | qaoa_ibm | 2 | 0.0 | 0.0 | 1148.0 | 1268.0 | 6.0 |
| T0 | D1+D2 | 2028 | binary | qaoa_ideal | 20 | 0.6 | 0.01105 |  |  |  |
| T0 | D1+D2 | 2028 | binary | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D1+D2 | 2028 | domain_wall | qaoa_ideal | 20 | 0.65 | 0.0109 |  |  |  |
| T0 | D1+D2 | 2028 | domain_wall | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D1+D2 | 2030 | binary | qaoa_ibm | 2 | 0.5 | 0.023 | 946.0 | 915.0 | 4.0 |
| T0 | D1+D2 | 2030 | binary | qaoa_ideal | 20 | 0.3 | 0.04535 |  |  |  |
| T0 | D1+D2 | 2030 | binary | qaoa_noisy | 5 | 0.2 | 0.0206 | 954.0 | 915.0 |  |
| T0 | D1+D2 | 2030 | domain_wall | qaoa_ideal | 20 | 0.3 | 0.04535 |  |  |  |
| T0 | D1+D2 | 2030 | domain_wall | qaoa_noisy | 5 | 0.2 | 0.0206 | 954.0 | 915.0 |  |
| T0 | D1+D2+D4 | 2026 | binary | qaoa_ibm | 2 | 0.5 | 0.003 | 817.0 | 736.0 | 4.0 |
| T0 | D1+D2+D4 | 2026 | binary | qaoa_ideal | 20 | 1.0 | 0.0981 |  |  |  |
| T0 | D1+D2+D4 | 2026 | binary | qaoa_noisy | 5 | 1.0 | 0.023 | 813.0 | 736.0 |  |
| T0 | D1+D2+D4 | 2026 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.0981 |  |  |  |
| T0 | D1+D2+D4 | 2026 | domain_wall | qaoa_noisy | 5 | 1.0 | 0.023 | 813.0 | 736.0 |  |
| T0 | D1+D2+D4 | 2027 | binary | qaoa_ibm | 2 | 0.0 | 0.0165 | 871.0 | 774.0 | 4.0 |
| T0 | D1+D2+D4 | 2027 | binary | qaoa_ideal | 20 | 1.0 | 0.20755 |  |  |  |
| T0 | D1+D2+D4 | 2027 | binary | qaoa_noisy | 5 | 1.0 | 0.0494 | 867.0 | 774.0 |  |
| T0 | D1+D2+D4 | 2027 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.20755 |  |  |  |
| T0 | D1+D2+D4 | 2027 | domain_wall | qaoa_noisy | 5 | 1.0 | 0.0494 | 867.0 | 774.0 |  |
| T0 | D1+D2+D4 | 2028 | binary | qaoa_ibm | 2 | 1.0 | 0.011 | 811.0 | 736.0 | 4.0 |
| T0 | D1+D2+D4 | 2028 | binary | qaoa_ideal | 20 | 1.0 | 0.11315 |  |  |  |
| T0 | D1+D2+D4 | 2028 | binary | qaoa_noisy | 5 | 0.8 | 0.0266 | 807.0 | 736.0 |  |
| T0 | D1+D2+D4 | 2028 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.11315 |  |  |  |
| T0 | D1+D2+D4 | 2028 | domain_wall | qaoa_noisy | 5 | 0.8 | 0.0266 | 807.0 | 736.0 |  |
| T0 | D1+D4 | 2026 | binary | qaoa_ibm | 2 | 0.0 | 0.002 | 1069.0 | 1400.0 | 4.0 |
| T0 | D1+D4 | 2026 | binary | qaoa_ideal | 20 | 0.4 | 0.08005 |  |  |  |
| T0 | D1+D4 | 2027 | binary | qaoa_ibm | 2 | 0.0 | 0.002 | 777.0 | 873.0 | 4.0 |
| T0 | D1+D4 | 2027 | binary | qaoa_ideal | 20 | 1.0 | 0.056 |  |  |  |
| T0 | D1+D4 | 2027 | binary | qaoa_noisy | 5 | 0.8 | 0.0136 | 783.0 | 873.0 |  |
| T0 | D1+D4 | 2027 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.08525 |  |  |  |
| T0 | D1+D4 | 2028 | binary | qaoa_ibm | 2 | 0.0 | 0.006 | 1045.0 | 1471.0 | 4.0 |
| T0 | D1+D4 | 2028 | binary | qaoa_ideal | 20 | 0.0 | 0.0929 |  |  |  |
| T0 | D1+D4 | 2029 | binary | qaoa_ibm | 2 | 0.0 | 0.0005 | 1060.0 | 1379.0 | 4.0 |
| T0 | D1+D4 | 2029 | binary | qaoa_ideal | 20 | 0.4 | 0.0822 |  |  |  |
| T0 | D1+D4 | 2030 | binary | qaoa_ibm | 2 | 0.5 | 0.006 | 697.0 | 1048.0 | 4.0 |
| T0 | D1+D4 | 2030 | binary | qaoa_ideal | 20 | 1.0 | 0.1061 |  |  |  |
| T0 | D1+D4 | 2030 | binary | qaoa_noisy | 5 | 0.4 | 0.0188 | 700.0 | 1048.0 |  |
| T0 | D1+D4 | 2030 | domain_wall | qaoa_ideal | 20 | 0.8 | 0.03055 |  |  |  |
| T0 | D2 | 2026 | binary | qaoa_ibm | 2 | 0.5 | 0.009 | 814.0 | 736.0 | 4.0 |
| T0 | D2 | 2026 | binary | qaoa_ideal | 20 | 1.0 | 0.09955 |  |  |  |
| T0 | D2 | 2026 | binary | qaoa_noisy | 5 | 1.0 | 0.0234 | 813.0 | 736.0 |  |
| T0 | D2 | 2026 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.09955 |  |  |  |
| T0 | D2 | 2026 | domain_wall | qaoa_noisy | 5 | 1.0 | 0.0234 | 813.0 | 736.0 |  |
| T0 | D2 | 2030 | binary | qaoa_ibm | 2 | 0.0 | 0.0015 | 1128.0 | 1168.0 | 6.0 |
| T0 | D2 | 2030 | binary | qaoa_ideal | 20 | 0.45 | 0.0169 |  |  |  |
| T0 | D2 | 2030 | binary | qaoa_noisy | 5 | 0.2 | 0.002 | 1137.0 | 1166.0 |  |
| T0 | D2 | 2030 | domain_wall | qaoa_ideal | 20 | 0.45 | 0.01635 |  |  |  |
| T0 | D2 | 2030 | domain_wall | qaoa_noisy | 5 | 0.2 | 0.0026 | 1171.0 | 1227.0 |  |
| T0 | D2+D4 | 2029 | binary | qaoa_ibm | 2 | 0.0 | 0.012 | 765.0 | 735.0 | 4.0 |
| T0 | D2+D4 | 2029 | binary | qaoa_ideal | 20 | 1.0 | 0.0609 |  |  |  |
| T0 | D2+D4 | 2029 | binary | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D2+D4 | 2029 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.0609 |  |  |  |
| T0 | D2+D4 | 2029 | domain_wall | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D2+D4 | 2030 | binary | qaoa_ideal | 20 | 0.15 | 0.0505 |  |  |  |
| T0 | D2+D4 | 2030 | domain_wall | qaoa_ideal | 20 | 0.25 | 0.04985 |  |  |  |
| T0 | D4 | 2026 | binary | qaoa_ibm | 2 | 0.0 | 0.0 | 1100.0 | 1391.0 | 4.0 |
| T0 | D4 | 2026 | binary | qaoa_ideal | 20 | 1.0 | 0.04515 |  |  |  |
| T0 | D4 | 2027 | binary | qaoa_ibm | 2 | 0.0 | 0.0175 | 969.0 | 1233.0 | 4.0 |
| T0 | D4 | 2027 | binary | qaoa_ideal | 20 | 0.15 | 0.08655 |  |  |  |
| T0 | D4 | 2027 | binary | qaoa_noisy | 5 | 0.0 | 0.0278 | 978.0 | 1233.0 |  |
| T0 | D4 | 2027 | domain_wall | qaoa_ideal | 20 | 0.0 | 0.006 |  |  |  |
| T0 | D4 | 2028 | binary | qaoa_ibm | 2 | 0.0 | 0.0035 | 942.0 | 1507.0 | 4.0 |
| T0 | D4 | 2028 | binary | qaoa_ideal | 20 | 0.2 | 0.0836 |  |  |  |
| T0 | D4 | 2029 | binary | qaoa_ibm | 2 | 0.0 | 0.0025 | 1086.0 | 1512.0 | 4.0 |
| T0 | D4 | 2029 | binary | qaoa_ideal | 20 | 0.0 | 0.09285 |  |  |  |
| T0 | D4 | 2030 | binary | qaoa_ibm | 2 | 0.0 | 0.006 | 665.0 | 962.0 | 4.0 |
| T0 | D4 | 2030 | binary | qaoa_ideal | 20 | 0.0 | 0.0299 |  |  |  |
| T0 | D4 | 2030 | binary | qaoa_noisy | 5 | 0.0 | 0.012 | 663.0 | 962.0 |  |
| T0 | D4 | 2030 | domain_wall | qaoa_ideal | 20 | 0.05 | 0.0152 |  |  |  |
| T0 | none | 2026 | binary | qaoa_ibm | 2 | 0.0 | 0.0 | 1173.0 | 1930.0 | 4.0 |
| T0 | none | 2026 | binary | qaoa_ideal | 20 | 0.0 | 0.06935 |  |  |  |
| T0 | none | 2027 | binary | qaoa_ibm | 2 | 0.0 | 0.0005 | 1037.0 | 1219.0 | 4.0 |
| T0 | none | 2027 | binary | qaoa_ideal | 20 | 0.5 | 0.07175 |  |  |  |
| T0 | none | 2027 | binary | qaoa_noisy | 5 | 0.2 | 0.0062 | 1050.0 | 1219.0 |  |
| T0 | none | 2028 | binary | qaoa_ibm | 2 | 0.0 | 0.006 | 867.0 | 1261.0 | 4.0 |
| T0 | none | 2028 | binary | qaoa_ideal | 20 | 0.15 | 0.1474 |  |  |  |
| T0 | none | 2028 | binary | qaoa_noisy | 5 | 0.0 | 0.0222 | 869.0 | 1261.0 |  |
| T0 | none | 2028 | domain_wall | qaoa_ideal | 20 | 0.0 | 0.0138 |  |  |  |
| T0 | none | 2029 | binary | qaoa_ibm | 2 | 0.0 | 0.0 | 1563.0 | 2044.0 | 6.0 |
| T0 | none | 2029 | binary | qaoa_ideal | 20 | 0.0 | 0.07175 |  |  |  |
| T0 | none | 2030 | binary | qaoa_ibm | 2 | 0.0 | 0.0015 | 1084.0 | 1232.0 | 4.0 |
| T0 | none | 2030 | binary | qaoa_ideal | 20 | 0.95 | 0.0649 |  |  |  |
| T0 | none | 2030 | binary | qaoa_noisy | 5 | 0.4 | 0.0106 | 1084.0 | 1232.0 |  |

### Training (`qaoa_training.md`)

| tier | dials | instance_seed | encoding | n_vars | dynamic_range | ising_dynamic_range | train_s | cvar | n_evals |
|---|---|---|---|---|---|---|---|---|---|
| T0 | D2 | 2026 | binary | 16 | 131297.85161264712 | 188035.44698694485 | 6.660222799982876 | 521.4686658390647 | 200 |
| T0 | D1+D2+D4 | 2026 | domain_wall | 16 | 150051.34276485513 | 213667.51291527192 | 5.681276399991475 | 1829.6055199894388 | 200 |
| T0 | D2+D4 | 2030 | domain_wall | 24 | 306965.8451784672 | 184067.0607447073 | 3698.116124299995 | 633.4073509808604 | 200 |
| T0 | D1+D4 | 2030 | binary | 19 | 3123334.8234805446 | 3127437.828172062 | 102.95197229999758 | 323.90269977530716 | 200 |
| T0 | D1 | 2026 | binary | 24 | 2281246.5692745014 | 2280923.8863076526 | 5285.188360900036 | 959.314333020961 | 200 |
| T0 | D2 | 2030 | domain_wall | 20 | 842077.7856412798 | 988810.772013221 | 213.10987990000285 | 2293.9008634372244 | 200 |
| T0 | D4 | 2030 | domain_wall | 24 | 177115.24514701654 | 126159.60561394728 | 3967.8384686000063 | 567.2926346973743 | 200 |
| T0 | D4 | 2026 | binary | 21 | 4338327.144770745 | 4957998.170000287 | 604.1193476000335 | 447.02546016143214 | 200 |
| T0 | none | 2028 | binary | 20 | 759064.6218446441 | 255179.95173478458 | 268.5130847999826 | 182.16136164970763 | 181 |
| T0 | D1 | 2027 | binary | 24 | 1155514.452811122 | 661335.4871877294 | 5421.16087219998 | 440.41191776456014 | 200 |
| T0 | D2+D4 | 2029 | domain_wall | 16 | 267622.30004372145 | 381076.852458577 | 3.8588179999933345 | 1286.279297966958 | 200 |
| T0 | D1+D4 | 2029 | binary | 21 | 3478300.797060273 | 3482215.2581682666 | 449.8586338000023 | 547.6407507704503 | 200 |
| T0 | D1+D2 | 2027 | domain_wall | 20 | 111480.19666454612 | 131295.74656935147 | 291.2639223000151 | 1920.5750194739112 | 200 |
| T0 | D1+D2 | 2027 | binary | 20 | 112260.87061133442 | 132214.18650674945 | 285.0019840999739 | 1935.3055569761868 | 200 |
| T0 | D2 | 2026 | domain_wall | 16 | 131297.85161264712 | 188035.44698694485 | 7.101933200028725 | 521.4686658390647 | 200 |
| T0 | D1+D2+D4 | 2027 | domain_wall | 16 | 9986.09276216544 | 25005.841946902932 | 5.686394999967888 | 143.3888175803449 | 200 |
| T0 | D1+D2+D4 | 2027 | binary | 16 | 9986.09276216544 | 25005.841946902932 | 6.22707000002265 | 143.3888175803449 | 200 |
| T0 | D1 | 2028 | binary | 24 | 1928805.5868174913 | 1930087.064344834 | 3848.6914867000014 | 505.7587041288621 | 200 |
| T0 | D1+D2+D4 | 2028 | binary | 16 | 147059.70760238502 | 209846.12670868673 | 3.2847720999998273 | 1318.44980640671 | 200 |
| T0 | D1+D4 | 2026 | binary | 21 | 2200583.8228090606 | 1928974.798081254 | 602.8713632000145 | 643.4166803715292 | 200 |
| T0 | D4 | 2027 | domain_wall | 23 | 171989.91258671423 | 220617.6749040006 | 2629.0052727999864 | 1381.133674789587 | 200 |
| T0 | D4 | 2029 | binary | 22 | 869907.6219097628 | 579612.9528968993 | 932.720645399997 | 367.25680006631194 | 200 |
| T0 | none | 2027 | binary | 20 | 2858732.848559634 | 3118430.7722666548 | 333.0398005999741 | 438.58478189443315 | 200 |
| T0 | D4 | 2030 | binary | 18 | 277577.722909354 | 184599.58710221457 | 43.61334120000538 | 464.0209354247634 | 181 |
| T0 | D1+D4 | 2027 | binary | 18 | 4540095.092686086 | 4956665.292469511 | 65.62662120000459 | 397.560428020335 | 200 |
| T0 | D1+D2 | 2026 | binary | 16 | 48650.67434373239 | 34876.03798732215 | 7.431677700020373 | 3285.7197724702833 | 200 |
| T0 | none | 2030 | binary | 20 | 3847243.570422343 | 3851380.5088098226 | 216.67118879999907 | 287.55241190926364 | 200 |
| T0 | D1+D2 | 2030 | domain_wall | 20 | 370735.0187875911 | 222939.59640466218 | 211.67147970000224 | 3023.745672842467 | 200 |
| T0 | D4 | 2028 | binary | 22 | 1592191.5631031005 | 1417342.726187961 | 897.6687227999937 | 317.4983614089844 | 200 |
| T0 | D2+D4 | 2029 | binary | 16 | 267622.30004372145 | 381076.852458577 | 3.4982738999970024 | 1286.279297966958 | 200 |
| T0 | D1+D4 | 2028 | binary | 22 | 1053836.0490469376 | 604064.3677823368 | 897.3274392000021 | 338.9568293751446 | 200 |
| T0 | none | 2029 | binary | 24 | 1696165.8983049456 | 1510376.3538750745 | 3886.136924300008 | 380.5317149498862 | 200 |
| T0 | none | 2026 | binary | 24 | 1476218.4710141397 | 1476036.1387551464 | 5278.669033300015 | 486.8862303893887 | 200 |
| T0 | D1+D4 | 2027 | domain_wall | 24 | 2445860.0374534237 | 2660340.668263771 | 4132.122526899999 | 322.38343335801784 | 200 |
| T0 | D1 | 2030 | binary | 24 | 1680347.2915862605 | 1680410.2600117684 | 3953.7137340000045 | 561.9803919189794 | 200 |
| T0 | D1+D2+D4 | 2028 | domain_wall | 16 | 147059.70760238502 | 209846.12670868673 | 3.3798901999980444 | 1318.44980640671 | 200 |
| T0 | D4 | 2027 | binary | 19 | 268734.73895857023 | 357515.4611453831 | 153.13842380000278 | 253.53180219072775 | 200 |
| T0 | D1 | 2029 | binary | 24 | 2652154.140299909 | 2651742.8486133995 | 3921.440889199992 | 670.0359233259853 | 200 |
| T0 | D1+D2 | 2028 | domain_wall | 20 | 46122.1444294258 | 102092.60822759196 | 213.92269609999494 | 973.920477053748 | 200 |
| T0 | D2 | 2030 | binary | 20 | 842270.1358897206 | 989037.0664231512 | 212.9275115000055 | 2465.3000780941825 | 200 |
| T0 | none | 2028 | domain_wall | 24 | 439393.50796450884 | 305196.3961698304 | 4133.672887400011 | 720.6994854651006 | 200 |
| T0 | D1+D2+D4 | 2026 | binary | 16 | 150051.34276485513 | 213667.51291527192 | 6.253213500021957 | 1829.6055199894388 | 200 |
| T0 | D1+D2 | 2030 | binary | 20 | 370735.0187875911 | 222939.59640466218 | 214.1368189999921 | 3023.745672842467 | 200 |
| T0 | D2+D4 | 2030 | binary | 24 | 307257.07928725815 | 184241.80120998184 | 3896.3238394999935 | 634.066310964684 | 200 |
| T0 | D1+D2 | 2028 | binary | 20 | 46814.696485712346 | 103631.61279711765 | 208.9032819999993 | 987.9093759336841 | 200 |
| T0 | D1+D4 | 2030 | domain_wall | 22 | 3250721.6530754687 | 3251713.450842411 | 911.2222880000045 | 593.4693196712088 | 200 |
| T0 | D1+D2 | 2026 | domain_wall | 16 | 48650.67434373239 | 34876.03798732215 | 6.783304800046608 | 3285.7197724702833 | 200 |

### HiGHS at the 1 s budget (`crossover_1s.md`)

median gap / feasibility / TTS99 at 1%.

| tier | dials | exact_pwl_milp |
|---|---|---|
| T0 | D1 | 0.00% / 100% / 0.239s |
| T0 | D1+D2 | 0.00% / 100% / 0.365s |
| T0 | D1+D2+D4 | 0.00% / 100% / 0.353s |
| T0 | D1+D4 | 0.00% / 100% / 0.255s |
| T0 | D2 | 0.00% / 100% / 0.399s |
| T0 | D2+D4 | 0.00% / 100% / 0.381s |
| T0 | D4 | 0.00% / 100% / 0.361s |
| T0 | none | 0.00% / 100% / 0.388s |

### The quota ledger (`quota.md`)

| job_id | device | task_key | submitted_at | max_execution_time_s | state | usage_s | task_identity | used_in_window_s | window_s |
|---|---|---|---|---|---|---|---|---|---|
| db7s898r1c5c738l6su0 | ibm_fez | 742d57fe71c67e7a73735322f49c5def796762825eb5a316db83840235f93fb9 | 1792001084.8477473 | 20 | done | 2.0 | fbc951b5364d554bf914bc72d0f82f075585265146f337113a74b3b31cc4cbe6 | 134.0 | 2419200 |
| dbgbv75rd2mc73cudk90 | ibm_fez | 5bc3d287212e2e1e17259ecd59ee49f76107e9e5d547e82aa7a050ee66ca13c1 | 1793113996.5260794 | 60 | done | 2.0 | fbc951b5364d554bf914bc72d0f82f075585265146f337113a74b3b31cc4cbe6 | 134.0 | 2419200 |
| dbgc5ldjm0ks73egb0rg | ibm_fez | e92f40884ac7af04721acdd33b32460d5109d2c93f75876744a8ae2ea95b578c | 1793114822.607945 | 60 | done | 3.0 | c3be47838500533c10cd046f5b1efa0edfed642a773377f4945f0398550e84de | 134.0 | 2419200 |
| dbgccm5jm0ks73egbc20 | ibm_fez | 95d6fe98aa75b00917a11e84a2b0e0374dbc7decda83dba697447d521e775d1c | 1793115721.6063027 | 60 | done | 2.0 | 7537ce30959bf918ceb95e2564b009294b0169249d499764aed963be59d0d5ca | 134.0 | 2419200 |
| dbgccs1j4t8s73ac2olg | ibm_fez | 8331d523af8dc6435f7b8ca72e0ba0a50ab659080106b227a6d684e5bd5cc691 | 1793115745.690666 | 60 | done | 2.0 | 87ae89a615eaaacf44016360be905e4c6b44ddf6264f8f355aa65d60241f63ba | 134.0 | 2419200 |
| dbgcd0giudkc73e47240 | ibm_fez | e4c05d58a2e7f9a30a9198d48fef0c3ef9cb4c26254f017f0ca707196b7786eb | 1793115762.8045678 | 60 | done | 2.0 | 632c57adecf4a113a9e89573811ce2eb3b2a3d5a09b997730a47b77cb5ca3337 | 134.0 | 2419200 |
| dbgcd48iudkc73e472c0 | ibm_fez | e1eac56bbe53a159be4aba222df756890da91e62617f6851d4c16bf599beb5b5 | 1793115777.918979 | 60 | done | 2.0 | a7d7a3e78b2dc54fc014d8049c3cc7ffa14476ed5626023305005b9715e110c5 | 134.0 | 2419200 |
| dbgcd7trd2mc73cuecj0 | ibm_fez | 8a5e7adf42bc77aa2643280666d768753c208890900665d77c74ce07e2576e51 | 1793115792.39108 | 60 | done | 2.0 | fcbf8884339dace8f0e6ea5be87a9682719f06b39898e3f3155ded49a3e6664f | 134.0 | 2419200 |
| dbgcdblrd2mc73cuecr0 | ibm_fez | 388a13eba89ffa5025c45a0e24417bdff1b4cfcd2267d8e92269134c73e7a5c4 | 1793115807.2992496 | 60 | done | 2.0 | ba3e3154f083a0889ba1a3eadbb39d987cdee330d0d735d02c09a366cc09c31c | 134.0 | 2419200 |
| dbgcdedrd2mc73cued00 | ibm_fez | 916336a5e532221fd011eba264bf9b2ed69b6c304b74a239370fcb7582d77221 | 1793115818.722859 | 60 | done | 2.0 | dbe90f8a2bb63ac767e302d5698967c2764b597dfe8a3b5973fc7e1a92237dd5 | 134.0 | 2419200 |
| dbgcdh5rd2mc73cued60 | ibm_fez | ee2938b66b4dfec18246b6e94f7a672bf50a544332e72e8d89dcba21a99f10d5 | 1793115829.1212795 | 60 | done | 2.0 | ab89f27188758c46c0fedb0cd36f5b28d750d94806d931707c34da5503eb4e30 | 134.0 | 2419200 |
| dbgcdklrd2mc73cuede0 | ibm_fez | c27497bd5c69a09c73231b8aedc9c564b3da7ea5c284b41c5422b6b760e45bcb | 1793115843.5479982 | 60 | done | 3.0 | 6fbac9fa4821cb3f147d932864e232895a9a73fb77b0a81e9d596d6e395e2fa1 | 134.0 | 2419200 |
| dbgcdodrd2mc73cuedm0 | ibm_fez | 567b8db6d7121308b3bad33cebf60a2555b2b9bcc0a67f1699b1ac04ef9f8bb9 | 1793115858.9553695 | 60 | done | 2.0 | c4d23b6bd33490989858c92241b3e90e016af28c106226dbefca31365070972d | 134.0 | 2419200 |
| dbgcds8iudkc73e473qg | ibm_fez | d8bc4dd9cd40f83e27b51061b2a35d943c6e2545e03f2904f7a84646a651e113 | 1793115874.8482885 | 60 | done | 2.0 | 2d6b2374d6d3cfeafeb5d3a52808bac2ce5ecfd7765630b5839abdb9cd3fba54 | 134.0 | 2419200 |
| dbgce09j4t8s73ac2qo0 | ibm_fez | 0ce03d3de8191c7b906cbe336dbe68f93660e5df58240f0de2ec80b45e4ddda7 | 1793115890.174759 | 60 | done | 2.0 | c749485370a1d66bdbac96120fd695fd2b640b14622eead4e56db4d62c41b338 | 134.0 | 2419200 |
| dbgce48iudkc73e474a0 | ibm_fez | ec2d2c8aba84c168287568f2c18ff9af4823dd0e3bb511ab4dcbfb0e549f7514 | 1793115905.941306 | 60 | done | 2.0 | 01a6971836f3757e4df32decd6508c4daf9cb14e9f4e6674a97682d6b12e3f74 | 134.0 | 2419200 |
| dbgcei5rd2mc73cuef2g | ibm_fez | ed1d575c702154737dcaa9b17a6e8b2bd64fceef2d7812f8552e6daa0865583d | 1793115960.8358324 | 60 | done | 2.0 | 8b4afac2e4963359355f7f4e865a412fe34c3cccfa75889fd14105b5c5110ead | 134.0 | 2419200 |
| dbgceltjm0ks73egbfq0 | ibm_fez | 93ba32cab15fcd1b6895f36f1973dee13bd99e3a25634664d9c1b5f4c02d6552 | 1793115976.2779145 | 60 | done | 3.0 | 44ca8366fc52e607d7ec7b969e5cff700a055589062607b6871d5729a0aab3c9 | 134.0 | 2419200 |
| dbgcet5jm0ks73egbg9g | ibm_fez | aa354faed4b19f5e2131493b00037230ae75e84a1e4d7bdb8d42ac174ecf879c | 1793116005.7969882 | 60 | done | 2.0 | 0c2c1924707b5db2e969c36c8e3f45f0957ddab30f150a44d4d91714f172f65a | 134.0 | 2419200 |
| dbgcf45rd2mc73cueg50 | ibm_fez | 0274c043eeded17b1acd54f02ab3fac1ad64a23949410bc0ce41b4ff3a250cb4 | 1793116033.4641268 | 60 | done | 2.0 | 6297629cd6a10b500a050e061ba2dbb9a125f4e192470ebd399a00839a5bc126 | 134.0 | 2419200 |
| dbgcf80iudkc73e476g0 | ibm_fez | a668ada777c96208fcd0236c52c08f1f4f9b80296e850310b0491c13a836299c | 1793116049.4762332 | 60 | done | 3.0 | 5bf67bb74fb6d2a99f0e0b6c3d6535c34ac41bb5a3f8e70b56327e09051d486f | 134.0 | 2419200 |
| dbgcfbtrd2mc73cuegig | ibm_fez | 7b1fcb0da8f1edce95ed1e10f3f4df760d8726454de673619fd2f6912f66b744 | 1793116064.16791 | 60 | done | 3.0 | 326968951029947e4a3f30b7a3695953190752022ea9b56d0785fbc2d9536f49 | 134.0 | 2419200 |
| dbgcfftrd2mc73cuegs0 | ibm_fez | ea95854634899cfc527f0862674cdea1b04e490f0a1de5a50832e53b4722ead0 | 1793116080.6001348 | 60 | done | 2.0 | 97997839b7929e096d1a64b5e85f6526253c4ce9588201b8797cf1b8b777f075 | 134.0 | 2419200 |
| dbgcfpljm0ks73egbi4g | ibm_fez | 0d38f87e8ad423f0eb01b2015fcbec815a1faebfbb3597f89f0be2a2cb2da551 | 1793116119.0960314 | 60 | done | 2.0 | cd608edce37a02058568c7f3d9066c5f30d2c19447517ea17520794b5a9a215c | 134.0 | 2419200 |
| dbgcg6drd2mc73cueid0 | ibm_fez | 07c2c765611c5080733e0aedd59310a6ee12a7c89b31cb50365a549751492836 | 1793116170.13173 | 60 | done | 2.0 | 9debf60ca6e504020f041b302a0507d341386148cecacb67253a3aa3ca63d101 | 134.0 | 2419200 |
| dbgcgeoiudkc73e47950 | ibm_fez | 6d3f45878639ddc9ca7edf0b8695c5b2970092ba8bdfc33240bc150a8e664111 | 1793116204.2955897 | 60 | done | 2.0 | a4ea59eaa7db41a2de7eeafa2ff582b213b12b2f2d74270728269cbef7b9f092 | 134.0 | 2419200 |
| dbgcgilrd2mc73cuej70 | ibm_fez | 5cec9d6e893764e461b0896e52d8088284afd07d5ffcedd05220fc2688305dae | 1793116219.3613398 | 60 | done | 2.0 | e533be17a0f4914c054736d29fde5c7bd5955527a9de5661dd1cf6c6fb863a4a | 134.0 | 2419200 |
| dbgcgmlrd2mc73cuejig | ibm_fez | a33a5640009b0eff2daca6d4271ca30a8455a3de15678866010903afffc9406b | 1793116235.1101367 | 60 | done | 3.0 | 5ff4af87885688b47152adb2495dd0bd0aa00f93334d4296f57f5c88f2af2beb | 134.0 | 2419200 |
| dbgcgqlrd2mc73cuejsg | ibm_fez | 4d73be5d153e64583e9d57b3d71fbf6e7628d0612707b1356ca46b9d51335884 | 1793116251.2938747 | 60 | done | 2.0 | efe3d423bb6668c04b543a9adaa8638f179532dd13b3fc4fc90aeed928e1a372 | 134.0 | 2419200 |
| dbgcgulrd2mc73cuek70 | ibm_fez | c9916dddbd08a16802791890a41f12242c864a150734fe5e63c8688e940259a0 | 1793116267.2847464 | 60 | done | 2.0 | b241ea45e08e924b6d0ef5786c3ae11a4b3c4f894a2c19abd67283ad1febfc9c | 134.0 | 2419200 |
| dbgch8djm0ks73egblc0 | ibm_fez | cdf8156c820ca8bda27909a40bfa2e5c7d78fb9d66842d5c6c6c5188d05c1c14 | 1793116306.3281107 | 60 | done | 2.0 | b8e5c6b059541eebc9dd7a965816f3ccace674bd425e700de041bcd74ebed238 | 134.0 | 2419200 |
| dbgcm58iudkc73e47p00 | ibm_fez | 5bc3d287212e2e1e17259ecd59ee49f76107e9e5d547e82aa7a050ee66ca13c1#r1 | 1793116933.648006 | 60 | done | 2.0 | fbc951b5364d554bf914bc72d0f82f075585265146f337113a74b3b31cc4cbe6 | 134.0 | 2419200 |
| dbgcm99j4t8s73ac3flg | ibm_fez | e92f40884ac7af04721acdd33b32460d5109d2c93f75876744a8ae2ea95b578c#r1 | 1793116950.141263 | 60 | done | 3.0 | c3be47838500533c10cd046f5b1efa0edfed642a773377f4945f0398550e84de | 134.0 | 2419200 |
| dbgcmd5jm0ks73egc2pg | ibm_fez | 95d6fe98aa75b00917a11e84a2b0e0374dbc7decda83dba697447d521e775d1c#r1 | 1793116965.496341 | 60 | done | 2.0 | 7537ce30959bf918ceb95e2564b009294b0169249d499764aed963be59d0d5ca | 134.0 | 2419200 |
| dbgcmglrd2mc73cuf2u0 | ibm_fez | 8331d523af8dc6435f7b8ca72e0ba0a50ab659080106b227a6d684e5bd5cc691#r1 | 1793116979.9566784 | 60 | done | 2.0 | 87ae89a615eaaacf44016360be905e4c6b44ddf6264f8f355aa65d60241f63ba | 134.0 | 2419200 |
| dbgcmppj4t8s73ac3gvg | ibm_fez | e4c05d58a2e7f9a30a9198d48fef0c3ef9cb4c26254f017f0ca707196b7786eb#r1 | 1793117016.3798535 | 60 | done | 2.0 | 632c57adecf4a113a9e89573811ce2eb3b2a3d5a09b997730a47b77cb5ca3337 | 134.0 | 2419200 |
| dbgcmt5jm0ks73egc41g | ibm_fez | e1eac56bbe53a159be4aba222df756890da91e62617f6851d4c16bf599beb5b5#r1 | 1793117029.7511861 | 60 | done | 2.0 | a7d7a3e78b2dc54fc014d8049c3cc7ffa14476ed5626023305005b9715e110c5 | 134.0 | 2419200 |
| dbgcn0oiudkc73e47qlg | ibm_fez | 8a5e7adf42bc77aa2643280666d768753c208890900665d77c74ce07e2576e51#r1 | 1793117044.1241412 | 60 | done | 2.0 | fcbf8884339dace8f0e6ea5be87a9682719f06b39898e3f3155ded49a3e6664f | 134.0 | 2419200 |
| dbgcn98iudkc73e47r9g | ibm_fez | 388a13eba89ffa5025c45a0e24417bdff1b4cfcd2267d8e92269134c73e7a5c4#r1 | 1793117078.3597074 | 60 | done | 2.0 | ba3e3154f083a0889ba1a3eadbb39d987cdee330d0d735d02c09a366cc09c31c | 134.0 | 2419200 |
| dbgcnf5jm0ks73egc5ag | ibm_fez | 916336a5e532221fd011eba264bf9b2ed69b6c304b74a239370fcb7582d77221#r1 | 1793117101.564257 | 60 | done | 2.0 | dbe90f8a2bb63ac767e302d5698967c2764b597dfe8a3b5973fc7e1a92237dd5 | 134.0 | 2419200 |
| dbgcni5rd2mc73cuf590 | ibm_fez | ee2938b66b4dfec18246b6e94f7a672bf50a544332e72e8d89dcba21a99f10d5#r1 | 1793117112.9782853 | 60 | done | 2.0 | ab89f27188758c46c0fedb0cd36f5b28d750d94806d931707c34da5503eb4e30 | 134.0 | 2419200 |
| dbgcnotjm0ks73egc5ug | ibm_fez | c27497bd5c69a09c73231b8aedc9c564b3da7ea5c284b41c5422b6b760e45bcb#r1 | 1793117140.1349123 | 60 | done | 3.0 | 6fbac9fa4821cb3f147d932864e232895a9a73fb77b0a81e9d596d6e395e2fa1 | 134.0 | 2419200 |
| dbgcnt0iudkc73e47su0 | ibm_fez | 567b8db6d7121308b3bad33cebf60a2555b2b9bcc0a67f1699b1ac04ef9f8bb9#r1 | 1793117157.4361577 | 60 | done | 2.0 | c4d23b6bd33490989858c92241b3e90e016af28c106226dbefca31365070972d | 134.0 | 2419200 |
| dbgco1lrd2mc73cuf6cg | ibm_fez | d8bc4dd9cd40f83e27b51061b2a35d943c6e2545e03f2904f7a84646a651e113#r1 | 1793117174.6576374 | 60 | done | 2.0 | 2d6b2374d6d3cfeafeb5d3a52808bac2ce5ecfd7765630b5839abdb9cd3fba54 | 134.0 | 2419200 |
| dbgco95rd2mc73cuf73g | ibm_fez | 0ce03d3de8191c7b906cbe336dbe68f93660e5df58240f0de2ec80b45e4ddda7#r1 | 1793117205.8220124 | 60 | done | 2.0 | c749485370a1d66bdbac96120fd695fd2b640b14622eead4e56db4d62c41b338 | 134.0 | 2419200 |
| dbgcodpj4t8s73ac3kl0 | ibm_fez | ec2d2c8aba84c168287568f2c18ff9af4823dd0e3bb511ab4dcbfb0e549f7514#r1 | 1793117224.6398473 | 60 | done | 2.0 | 01a6971836f3757e4df32decd6508c4daf9cb14e9f4e6674a97682d6b12e3f74 | 134.0 | 2419200 |
| dbgcohpj4t8s73ac3kvg | ibm_fez | ed1d575c702154737dcaa9b17a6e8b2bd64fceef2d7812f8552e6daa0865583d#r1 | 1793117240.7507868 | 60 | done | 2.0 | 8b4afac2e4963359355f7f4e865a412fe34c3cccfa75889fd14105b5c5110ead | 134.0 | 2419200 |
| dbgcoldrd2mc73cuf7s0 | ibm_fez | 93ba32cab15fcd1b6895f36f1973dee13bd99e3a25634664d9c1b5f4c02d6552#r1 | 1793117254.082179 | 60 | done | 3.0 | 44ca8366fc52e607d7ec7b969e5cff700a055589062607b6871d5729a0aab3c9 | 134.0 | 2419200 |
| dbgcop8iudkc73e47usg | ibm_fez | aa354faed4b19f5e2131493b00037230ae75e84a1e4d7bdb8d42ac174ecf879c#r1 | 1793117270.1570163 | 60 | done | 2.0 | 0c2c1924707b5db2e969c36c8e3f45f0957ddab30f150a44d4d91714f172f65a | 134.0 | 2419200 |
| dbgcot0iudkc73e47v4g | ibm_fez | 0274c043eeded17b1acd54f02ab3fac1ad64a23949410bc0ce41b4ff3a250cb4#r1 | 1793117285.7387915 | 60 | done | 2.0 | 6297629cd6a10b500a050e061ba2dbb9a125f4e192470ebd399a00839a5bc126 | 134.0 | 2419200 |
| dbgcpi5rd2mc73cufa2g | ibm_fez | a668ada777c96208fcd0236c52c08f1f4f9b80296e850310b0491c13a836299c#r1 | 1793117369.3673759 | 60 | done | 3.0 | 5bf67bb74fb6d2a99f0e0b6c3d6535c34ac41bb5a3f8e70b56327e09051d486f | 134.0 | 2419200 |
| dbgcpmdrd2mc73cufac0 | ibm_fez | 7b1fcb0da8f1edce95ed1e10f3f4df760d8726454de673619fd2f6912f66b744#r1 | 1793117385.869328 | 60 | done | 3.0 | 326968951029947e4a3f30b7a3695953190752022ea9b56d0785fbc2d9536f49 | 134.0 | 2419200 |
| dbgcqb0iudkc73e482e0 | ibm_fez | ea95854634899cfc527f0862674cdea1b04e490f0a1de5a50832e53b4722ead0#r1 | 1793117469.5702145 | 60 | done | 2.0 | 97997839b7929e096d1a64b5e85f6526253c4ce9588201b8797cf1b8b777f075 | 134.0 | 2419200 |
| dbgcqdoiudkc73e482n0 | ibm_fez | 0d38f87e8ad423f0eb01b2015fcbec815a1faebfbb3597f89f0be2a2cb2da551#r1 | 1793117480.640393 | 60 | done | 2.0 | cd608edce37a02058568c7f3d9066c5f30d2c19447517ea17520794b5a9a215c | 134.0 | 2419200 |
| dbgcqi0iudkc73e48300 | ibm_fez | 07c2c765611c5080733e0aedd59310a6ee12a7c89b31cb50365a549751492836#r1 | 1793117497.7684212 | 60 | done | 2.0 | 9debf60ca6e504020f041b302a0507d341386148cecacb67253a3aa3ca63d101 | 134.0 | 2419200 |
| dbgcql5jm0ks73egccd0 | ibm_fez | 6d3f45878639ddc9ca7edf0b8695c5b2970092ba8bdfc33240bc150a8e664111#r1 | 1793117509.0702953 | 60 | done | 2.0 | a4ea59eaa7db41a2de7eeafa2ff582b213b12b2f2d74270728269cbef7b9f092 | 134.0 | 2419200 |
| dbgcqq5rd2mc73cufcrg | ibm_fez | 5cec9d6e893764e461b0896e52d8088284afd07d5ffcedd05220fc2688305dae#r1 | 1793117529.3443906 | 60 | done | 2.0 | e533be17a0f4914c054736d29fde5c7bd5955527a9de5661dd1cf6c6fb863a4a | 134.0 | 2419200 |
| dbgcqsoiudkc73e483q0 | ibm_fez | a33a5640009b0eff2daca6d4271ca30a8455a3de15678866010903afffc9406b#r1 | 1793117540.9104047 | 60 | done | 3.0 | 5ff4af87885688b47152adb2495dd0bd0aa00f93334d4296f57f5c88f2af2beb | 134.0 | 2419200 |
| dbgcr6lrd2mc73cufdqg | ibm_fez | 4d73be5d153e64583e9d57b3d71fbf6e7628d0612707b1356ca46b9d51335884#r1 | 1793117579.3348777 | 60 | done | 2.0 | efe3d423bb6668c04b543a9adaa8638f179532dd13b3fc4fc90aeed928e1a372 | 134.0 | 2419200 |
| dbgcra8iudkc73e484l0 | ibm_fez | c9916dddbd08a16802791890a41f12242c864a150734fe5e63c8688e940259a0#r1 | 1793117594.6198447 | 60 | done | 2.0 | b241ea45e08e924b6d0ef5786c3ae11a4b3c4f894a2c19abd67283ad1febfc9c | 134.0 | 2419200 |
| dbgcrgtrd2mc73cufedg | ibm_fez | cdf8156c820ca8bda27909a40bfa2e5c7d78fb9d66842d5c6c6c5188d05c1c14#r1 | 1793117620.6596184 | 60 | done | 2.0 | b8e5c6b059541eebc9dd7a965816f3ccace674bd425e700de041bcd74ebed238 | 134.0 | 2419200 |
<!-- /tables -->
