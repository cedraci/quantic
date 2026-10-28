# M5 QAOA on IBM Quantum — result

**Written:** 2026-10-27 after two hardware passes; completed 2026-10-28 with the third pass and the reps = 1 rung
**Branch:** `feat/m5-qaoa-ibm`
**Spec:** `2026-10-09-m5-qaoa-ibm-design.md`
**Plan:** `docs/superpowers/plans/2026-10-09-m5-qaoa-ibm.md`, Task 14
**Compared with M4 in:** `2026-10-28-m4-m5-comparison.md`

---

## Verdict

**QAOA on IBM's `ibm_fez` does not compete anywhere on T0. The device's output is barely distinguishable from uniform random bits, and even the noiseless QAOA loses to every classical sampler at the same number of reads. Halving the circuit's depth lets the device keep a little more of the circuit's structure but makes the algorithm itself weaker, and changes nothing in the comparison.**

- **On hardware,** 90 jobs of 1,000 shots (three per instance, 30 instances, 198 s of QPU time) returned a feasible schedule in 0.51% of shots; 23 of the 90 jobs returned none. The best feasible sample was within 1% of the optimum in 11 jobs, on 8 instances. Jobs of 1,000 *uniform random bitstrings* would have managed about 5.5 of those 11 (section 3; exact tail p ≈ 0.01, a lower bound). The hardware's feasible rate is 2.0 times the uniform rate (Poisson p ≈ 7 × 10⁻⁴⁰, so not noise alone), against 50 times for the same circuit simulated without noise. Every success is on a circuit of at most 1,233 two‑qubit gates, 10 of the 11 on the ten shallowest (≤ 1,048); the 48 jobs on circuits of 1,261 gates or more found nothing.
- **Noiselessly,** the reps = 2 CVaR‑QAOA at its trained angles reaches 1% of the optimum in 46% of 1,000‑shot runs on the same 30 instances, with 7.5% feasible samples. At 1,000 reads, simulated annealing reaches it in 54–100% of runs with 69–100% feasible samples, greedy descent in 88–100%, tabu in 98–100%. HiGHS proves the optimum on every cell in 0.24–0.40 s.
- **Time to solution.** On the six cells where any hardware job succeeded, TTS99 at 1% is 23–133 s of QPU time (`D1+D2+D4` and `D2+D4` 22.7 s, `D1+D2` 28.4 s, `D2` 63.1 s, `D1+D4` and `D4` 133 s); on `none` and `D1` it is censored. That counts QPU seconds only: each instance also cost 6 s to 90 min of classical training (17.3 h in all) and 1 s to 11 min of queue. Greedy at 1,000 reads: 0.17–0.74 s. Annealing: 0.63–6.7 s.
- **The noise model predicts this.** Aer with the `fake_fez` twin puts the feasible rate at 0.2–4.9%, and on the 17 hardware instances of at most 20 qubits (three of them killed, counted as 0) the success rate at 0.36, between 0.68 noiseless and 0.22 on the device.
- **Depth is the limit, not the dynamic range.** Routed circuits carry 735–2,044 CZ gates (depth 665–1,563) on a device whose median CZ error is 0.25%: at those counts 16% down to 0.6% of shots are free of a two‑qubit error, before readout. The QUBOs' dynamic range (10⁴ to 4.5 × 10⁶) shows no measurable effect on the noiseless success once the qubit count is accounted for (section 5).
- **Half the depth: more structure kept, less algorithm.** At reps = 1 the routed circuits carry 313–952 CZ gates, 0.41 to 0.59 of the reps = 2 count instance by instance, and the device's feasible rate rises to 3.0 times uniform (242 feasible samples in 31,000 shots); 5 of 31 jobs reached 1% of the optimum against about 1.9 by chance, four of them on 16‑qubit instances. But the shallower circuit is a weaker optimiser: over the 31 binary QUBOs, noiseless success falls from 0.45 to 0.35 and the feasible share from 7.4% to 3.4%. Neither depth comes near the samplers ("The reps = 1 rung").
- **Gaps in the simulations, all reported, none affecting the hardware result.** At reps = 2, 30 full noisy runs and one gate were killed at the 2 h limit, all circuits whose routing spread them over 23–29 physical qubits (section 7); the one hardware task whose gate was killed (`T0‑D2+D4‑2030`) never ran at reps = 2, and ran at reps = 1. At reps = 1, the five noisy runs of `D1+D2‑2030` were killed for the same reason (25 physical qubits for 20 logical).

The parent question — where does quantum compete with the best free classical solvers on discrete liquidation — gets, for gate‑model QAOA on a 2026 Heron processor, the answer "nowhere we can reach": not at 16 qubits, where the feasible search space holds 12 or 24 schedules and random sampling already finds the optimum a fifth of the time; and not at 24, where the circuit is lost in noise.

## What was run

| what | main study, reps = 2 | smoke | reps = 1 rung |
|---|---|---|---|
| config | `experiments/m5-qaoa.toml` | `results/m5-smoke.toml` (one task, 20 s cap) | `experiments/m5-qaoa-p1.toml` |
| commit | `b8c1244` (clean) | `55c1009` (clean) | `d9a3de6` (clean) |
| lock hash | `dd3493365dfe6fc5bff5f6431c95b9cff8af7dfd830c571aecd496a17a6e610f` | same | same |
| tasks | 6,184: 6,153 simulated, 31 hardware | 6 | 1,334: 1,303 simulated, 31 hardware |
| killed | 31 (30 full noisy runs, 1 gate) | 0 | 5 (the full noisy runs of `D1+D2‑2030`) |
| errors | 0 | 0 | 0 |
| simulation | about 56 h of running, 2026‑10‑12 13:35 → 2026‑10‑24 18:13, in three stretches (stopped twice by the machine running out of memory, once by the owner) | 2026‑10‑14 | one run, 2026‑10‑27 about 17:04 → 2026‑10‑28 06:55 (about 13.9 h) |
| hardware lane, job submissions | pass 1 2026‑10‑27 16:13–16:52, pass 2 17:02–17:14, pass 3 2026‑10‑28 09:52–09:59; 30 jobs each | submitted 2026‑10‑14 20:04, collected 2026‑10‑22 | 2026‑10‑28 09:35–09:50, 31 jobs |
| QPU seconds | 198 (2 or 3 s per job) | 2 | 62 (2 s per job) |
| store | 80 MB DuckDB, not committed | 3 MB, not committed | 29 MB, not committed |

Both studies ran 14 workers for the classical phase and 2 for training and sampling. Machine: Windows 11 (10.0.26200), Intel Core i9‑9900 at 3.10 GHz, 63.9 GB, Python 3.11; qiskit 2.5.2, qiskit‑aer 0.17.2, qiskit‑ibm‑runtime 0.50.0, numpy 2.4.6, scipy 1.17.1, OR‑Tools 9.15.6755, DuckDB 1.5.6.

Device: `ibm_fez`, Heron r2, 156 qubits, CZ/RZ/SX/X basis, Open plan. Calibration as read after the first pass (properties last updated 2026‑10‑27 16:32): median CZ error 0.25%, SX 0.03%, readout 1.1%. The next morning's calibration, read after the last lane, was last updated at 09:49, during the reps = 1 lane, and has the same medians (CZ 0.25%, SX 0.03%, readout 1.0%); the third pass ran under it, the reps = 1 jobs before 09:49 under the one before it. The gate routes on the fake twin (`FakeFez`), the lane on the live backend; where both were checked the routed counts agree within a few gates (`T0‑D2+D4‑2029`: 735 CZ and depth 770 on the twin, 735 and 765 live).

Instances: T0 (3 assets × 4 buckets), the eight dial combinations without CVaR (D3), seeds 2026–2030: 40 instances, encoded in binary and domain‑wall at eight zeta bits (no zeta register without CVaR). Caps: `max_qubits = 24` keeps 31 of the 40 binary QUBOs and 16 of the 40 domain‑wall ones (876 quantum tasks left out); `noisy_max_qubits = 20` keeps 27 QUBOs for full‑shot noisy runs (100 left out). Angles: CVaR‑QAOA, α = 0.25, COBYLA 200 evaluations on an exact statevector, one training per distinct QUBO and depth.

| solver | runs, reps = 2 | per instance |
|---|---|---|
| `qaoa_ideal` | 940 | 20 seeds × (31 binary + 16 domain‑wall QUBOs), 1,000 shots |
| `qaoa_noisy`, full | 135 (105 ok, 30 killed) | 5 seeds × 27 QUBOs of ≤ 20 qubits, 1,000 shots |
| `qaoa_noisy`, gate | 31 (30 ok, 1 killed) | one 32‑shot run per hardware task |
| `qaoa_ibm` | 90 (one task skipped) | 3 jobs × 30 instances, 1,000 shots, binary |
| `simulated_annealing`, `tabu`, `greedy` | 1,600 each | 2 encodings × 20 seeds × 40 instances, 1,000 reads, no clock |
| `exact_pwl_milp` | 40 | 1 s budget |
| `ref:qaoa_angles` | 47 | one per distinct QUBO |
| `ref:brute_force`, `ref:exact_pwl_milp`, `ref:qubo_minimum` | 40, 40, 80 | no budget |

The reps = 1 rung's runs are listed in its own section.

**The references.** In both stores no reference is missing, every bracket is closed, and `encoding_regret` is 0 on every QAOA cell: the encodings represent the optimum, so a gap is the sampler's. There is no CP‑SAT in this study, so the bracket check compares nothing.

**The quota.** The ledger (`~/.quantic/ibm-quota.json`) holds 122 jobs, all `done`, 262 s used of the 300 s cap: the smoke job (2 s), the main study's three passes (198 s) and the reps = 1 lane (62 s). IBM's own counter read 262 s after the last job. The 60 s per‑job limit never bound: every job used 2 or 3 s. Queue times were under 8 s for 120 of the 121 study jobs and 11 min for one (pass 1, `D1‑2026`); the smoke job waited 11 h behind a maintenance window. All usage figures were final when read, so the report's QPU sums are complete. 38 s remain under the cap until the first study jobs leave the 28‑day window on 2026‑11‑24.

**Reproduce.** `quantic bench report <store> --out <dir> [--ledger <path>]` regenerates every table below from a store; either store regenerates from its commit with `quantic bench run <config> --store <path>` (the hardware lane needs a saved IBM account and the owner's go‑ahead, and is new device data, not the same numbers). `experiments/m5-uniform-baseline.py <store> <out.csv>` regenerates section 3, `experiments/m5-hardware-jobs.py` the job table, and `experiments/m5-result-appendix.py` splices the appendix into this document. The figures of the comparison document are built from the same tables (`experiments/m5-figures.py`).

### How to read the numbers

- **Success at ε** is the share of runs whose best feasible sample is within ε of the proven optimum: (total − optimum) / |optimum| ≤ ε. A run with no feasible sample fails.
- **Feasibility** is the share of samples that decode to a schedule satisfying every constraint. Nothing is repaired.
- **TTS99** is t_run · ln 0.01 / ln(1 − p). For `qaoa_ibm`, t_run is the job's QPU seconds; for the simulators it is not reported (decision D6: simulation time is not quantum time).
- Each QAOA run is one 1,000‑shot circuit at fixed trained angles; its seed changes only the shot sampling. A repeated hardware task is stored under the key `<key>#r<n>` beside the earlier rows, and the report pools all of an instance's jobs.

## Findings the plan asked for

### 1. Matched reads: QAOA against the samplers at 1,000 reads

Success at ε = 1% / feasible samples, binary encoding, every solver at 1,000 reads. The samplers are averaged over 100 runs per cell (5 instances × 20 seeds), the noiseless QAOA over 20 runs per instance, the noise model over 5, the hardware over 3 jobs per instance (`m5-p2-matched_reads.csv`):

| cell | greedy | simulated annealing | tabu | QAOA, noiseless | QAOA, noise model | QAOA, `ibm_fez` |
|---|---|---|---|---|---|---|
| `D1` | 0.97 / 96% | 0.54 / 100% | 1.00 / 100% | 0.03 / 7.6% | — (24 qubits) | 0 of 15 / 0.093% |
| `D1+D2` | 0.90 / 35% | 0.99 / 69% | 1.00 / 96% | 0.57 / 2.3% | 0.10 / 1.2% (10 of 20 runs killed) | 4 of 12 / 0.81% |
| `D1+D2+D4` | 1.00 / 41% | 1.00 / 69% | 0.99 / 95% | 1.00 / 14% | 0.93 / 3.3% | 3 of 9 / 1.1% |
| `D1+D4` | 1.00 / 96% | 0.94 / 100% | 1.00 / 100% | 0.56 / 8.3% | 0.60 / 1.6% | 1 of 15 / 0.36% |
| `D2` | 0.88 / 20% | 1.00 / 76% | 0.98 / 98% | 0.72 / 5.8% | 0.60 / 1.3% | 1 of 6 / 0.45% |
| `D2+D4` | 0.95 / 45% | 0.90 / 92% | 1.00 / 100% | 0.57 / 5.6% | all 5 runs killed | 1 of 3 / 1.1% |
| `D4` | 1.00 / 96% | 0.76 / 100% | 1.00 / 100% | 0.27 / 6.8% | 0.00 / 2.0% | 1 of 15 / 0.71% |
| `none` | 1.00 / 98% | 0.75 / 100% | 1.00 / 100% | 0.32 / 8.5% | 0.20 / 1.3% | 0 of 15 / 0.19% |

TTS99 at ε = 1%, seconds, binary encoding (`m5-p2-cells.csv`, `m5-p1-cells.csv`):

| cell | greedy | simulated annealing | tabu | HiGHS (proof, 1 s budget) | QAOA, `ibm_fez`, reps = 2 (QPU seconds) | QAOA, `ibm_fez`, reps = 1 (QPU seconds) |
|---|---|---|---|---|---|---|
| `D1` | 0.735 | 6.65 | 41.6 | 0.24 | censored (15 jobs) | censored (5 jobs) |
| `D1+D2` | 0.296 | 0.63 | 58.7 | 0.37 | 28.4 (4 of 12 jobs) | censored (4 jobs) |
| `D1+D2+D4` | 0.17 | 0.934 | 61.4 | 0.35 | 22.7 (3 of 9 jobs) | 8.38 (2 of 3 jobs) |
| `D1+D4` | 0.356 | 1.44 | 42.2 | 0.25 | 133 (1 of 15 jobs) | 41.3 (1 of 5 jobs) |
| `D2` | 0.474 | 0.833 | 74.7 | 0.40 | 63.1 (1 of 6 jobs) | 13.3 (1 of 2 jobs) |
| `D2+D4` | 0.416 | 1.84 | 63.9 | 0.38 | 22.7 (1 of 3 jobs) | 13.3 (1 of 2 jobs) |
| `D4` | 0.351 | 3.45 | 41.8 | 0.36 | 133 (1 of 15 jobs) | censored (5 jobs) |
| `none` | 0.343 | 3.52 | 42.3 | 0.39 | censored (15 jobs) | censored (5 jobs) |

**The noiseless QAOA already loses.** Its weakness is feasibility: the penalty‑encoded constraints leave 2–14% of its shots feasible where annealing has 69–100%, so even where it finds the optimum (every `D1+D2+D4` run) it does so from a handful of feasible samples. It is competitive only on the three cells with the cardinality dial whose instances are 16 qubits, where the feasible search space is tiny (section 3), and weakest on `D1`, `D4` and `none` at 20–24 qubits.

**The hardware loses to the noiseless circuit by another factor of about 4 in success and 15 in feasibility** (0.46 against 0.12, 7.5% against 0.51%, pooled over the 30 instances), and its QPU‑only TTS99, where it exists at all, is 50 to 400 times greedy's wall‑clock TTS99 at the same reads, before counting training and queue.

### 2. The noise ladder: noiseless → noise model → device

Per instance, binary encoding, success at ε = 1% (and feasible samples), over the 17 hardware instances of at most 20 qubits, the ones the noise model was planned for (`m5-p2-qaoa_noise.csv`):

| instance | qubits | CZ, routed | noiseless (20 runs) | noise model (5 runs) | `ibm_fez` (3 jobs) |
|---|---|---|---|---|---|
| `D1+D2-2026` | 16 | 758 | 0.70 (1.0%) | 0.20 (0.36%) | 2 of 3 (0.63%) |
| `D1+D2-2027` | 20 | 1,262 | 0.70 (2.5%) | 0 (all killed) | 0 of 3 (0.067%) |
| `D1+D2-2028` | 20 | 1,268 | 0.60 (1.1%) | 0 (all killed) | 0 of 3 (0.033%) |
| `D1+D2-2030` | 20 | 915 | 0.30 (4.5%) | 0.20 (2.1%) | 2 of 3 (2.5%) |
| `D1+D2+D4-2026` | 16 | 736 | 1.00 (9.8%) | 1.00 (2.3%) | 1 of 3 (0.33%) |
| `D1+D2+D4-2027` | 16 | 774 | 1.00 (21%) | 1.00 (4.9%) | 0 of 3 (1.8%) |
| `D1+D2+D4-2028` | 16 | 736 | 1.00 (11%) | 0.80 (2.7%) | 2 of 3 (1.2%) |
| `D1+D4-2027` | 18 | 873 | 1.00 (5.6%) | 0.80 (1.4%) | 0 of 3 (0.2%) |
| `D1+D4-2030` | 19 | 1,048 | 1.00 (11%) | 0.40 (1.9%) | 1 of 3 (0.63%) |
| `D2-2026` | 16 | 736 | 1.00 (10.0%) | 1.00 (2.3%) | 1 of 3 (0.8%) |
| `D2-2030` | 20 | 1,168 | 0.45 (1.7%) | 0.20 (0.2%) | 0 of 3 (0.1%) |
| `D2+D4-2029` | 16 | 735 | 1.00 (6.1%) | 0 (all killed) | 1 of 3 (1.1%) |
| `D4-2027` | 19 | 1,233 | 0.15 (8.7%) | 0.00 (2.8%) | 1 of 3 (1.9%) |
| `D4-2030` | 18 | 962 | 0.00 (3.0%) | 0.00 (1.2%) | 0 of 3 (1.0%) |
| `none-2027` | 20 | 1,219 | 0.50 (7.2%) | 0.20 (0.62%) | 0 of 3 (0.067%) |
| `none-2028` | 20 | 1,261 | 0.15 (15%) | 0.00 (2.2%) | 0 of 3 (0.67%) |
| `none-2030` | 20 | 1,232 | 0.95 (6.5%) | 0.40 (1.1%) | 0 of 3 (0.13%) |
| **mean** | | | **0.68** | **0.36** (killed instances as 0) | **0.22** |

Over all 30 hardware instances: noiseless 0.46 success and 7.5% feasibility, device 0.12 and 0.51%. The 13 instances the noise model was not planned for are the 24‑qubit ones (`D1` × 5, `none‑2026`, `none‑2029`) and the 21–22‑qubit ones (`D1+D4` × 3, `D4` × 3): the device succeeded on none of their 39 jobs.

The noise model is a fair guide to the device: it puts the feasible rate in the right band (0.2–4.9% against 0.03–2.5% on the device) and, on the 14 instances it simulated in full, its success rate is above the device's on 9, below on 3 and equal on 2. With three jobs per instance the per‑instance comparison is still coarse; the ladder's means and the pooled numbers of section 3 are the robust reading.

**Depth decides.** Every one of the 11 hardware successes sits on a circuit of at most 1,233 CZ gates (14 instances, 42 jobs), 10 of them on the ten shallowest (at most 1,048 gates, 30 jobs); the 16 circuits with 1,261–2,044 gates (48 jobs) produced none. The one success above 1,048 gates came in the third pass, on `D4‑2027` (1,233 CZ, 19 qubits), where the noiseless circuit itself succeeds in 15% of runs and none of 100,000 uniform bitstrings is within 1%. At the device's median CZ error of 0.25% a shot passes 735 gates without a two‑qubit error 16% of the time and 2,044 gates 0.6% of the time, before the 1.1% readout error on each of 16–24 qubits. The design's risk table expected about 2% error‑free shots at 20 qubits; that is what the device delivers.

### 3. Against uniform random bits

Is the device's output distinguishable from random bitstrings? For each hardware instance, every one of the 2¹⁶ bitstrings (16 qubits) or 100,000 uniform random ones (18–24 qubits) was passed through the study's own decoder and scorer, giving the share of bitstrings that decode to a feasible schedule and, where exact, how many sit within 1% of the optimum (`m5-p2-uniform.csv`).

- **Feasibility.** The 90 hardware jobs returned 462 feasible samples in 90,000 shots. Uniform bitstrings would return 233 (per‑instance rates 0.001%–2.5%). The ratio is 2.0 (Poisson tail p ≈ 7 × 10⁻⁴⁰), so the output is not uniform: a trace of the circuit survives. The noiseless circuit's ratio is 50 (median over instances; 1.8 to 4,500). Per instance, the device's ratio is 0–33 (median 2.1), the noiseless circuit's 1.8–4,500.
- **Success.** The six 16‑qubit instances have 192 or 384 feasible bitstrings out of 65,536, which decode to just 12 or 24 distinct feasible schedules, with one optimum (16 bitstrings). A uniform 1,000‑shot job therefore contains the optimum with probability 1 − (1 − 16/65,536)¹⁰⁰⁰ = 22%. The device managed 7 of its 18 jobs there, against 3.9 expected by chance (binomial p = 0.07); the noiseless circuit 70–100%. On the 24 instances of 18–24 qubits the device succeeded in 4 of 72 jobs, against about 1.6 expected by chance: two on `D1+D2‑2030`, where 2.5% of all bitstrings are feasible and a uniform 1,000‑shot job finds the optimum 39% of the time (the noiseless circuit: 30%, the device: 2 of 3); one on `D1+D4‑2030`, where chance gives 3% per job; and one on `D4‑2027`, where none of the 100,000 draws was within 1%. In all, 11 successes where chance gives about 5.5 (exact Poisson‑binomial tail p = 0.011; a lower bound on the tail, since where the draws saw no bitstring within 1% the chance is counted as zero).

So the device keeps a little of the circuit's structure (feasibility 2.0 × uniform) and almost none of its optimisation power (noiseless: 50 × uniform; 46% success against 12% on the device, of which about 6 points are what chance gives on these instances). With the third pass the evidence on the 16‑qubit instances weakened (two passes: 6 of 12 against 2.6, p = 0.03; three: 7 of 18 against 3.9, p = 0.07), while the pooled feasibility excess strengthened. The per‑instance table is in the appendix.

### 4. Training cost against qubit count

47 angle sets (31 binary, 16 domain‑wall QUBOs), 17.3 h of CPU in all. Each COBYLA evaluation is an exact statevector CVaR at α = 0.25 (`m5-p2-qaoa_training.csv`):

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

About 2.3× per added qubit, 1.4 GB at 24 qubits. The dry run measured 28 qubits at 413 s per evaluation and 20 GB (23 h per QUBO), which is why the study stops at 24. COBYLA used all 200 evaluations on 45 of the 47 trainings (two stopped at 181), so the angles are where the budget ran out, not where the search converged; `reps`, `maxiter` and `alpha` are the first things a follow‑up would vary, and the reps = 1 rung varied the first. Against this, HiGHS proves the optimum of every instance in under 0.4 s: the classical pre‑processing alone costs 10 to 10⁴ times the classical solve.

### 5. The dynamic‑range effect

The QUBOs' dynamic range (largest to smallest non‑zero coefficient) runs from 1.0 × 10⁴ to 4.5 × 10⁶, the scaled Ising forms' from 2.5 × 10⁴ to 5.0 × 10⁶. The design's risk was that a range near 5 × 10⁵ hides small terms from QAOA. The data do not show it:

- Over the 31 binary QUBOs, Spearman's correlation between log range and noiseless success is −0.16 (p = 0.38); within the 16‑, 20‑ and 24‑qubit groups it is +0.39, +0.11 and +0.11, none significant. Qubit count alone correlates at −0.77 (p = 4 × 10⁻⁷). At reps = 1 the same correlations are −0.24 (p = 0.19) and −0.82 (p = 2 × 10⁻⁸).
- Per dial, `D1+D4` has the largest median range (3.1 × 10⁶) and succeeds in 56% of noiseless runs at 18–22 qubits; `D4` has a range four times smaller and succeeds in 27%; `D1` (1.9 × 10⁶) succeeds in 3%, but it is also the one dial at 24 qubits in every seed.
- Against the device's success the correlation is −0.44 (p = 0.015, 30 instances), which is the gate count in disguise: the widest ranges belong to the `D1` instances at 24 qubits, which are also the deepest circuits.

The range enters the angles' training (an exact Ising energy) rather than the device, which runs the same circuit whatever the coefficients. Where a range of 10⁶ would matter is on an analogue annealer, which M5 could not run.

### 6. How far the hardware lane got within 300 s

Three full passes over the 30 runnable tasks, 90 jobs, 198 s of QPU time, 2 or 3 s per job (72 jobs at 2 s, 18 at 3 s; 1,000 shots, `max_execution_time` 60 s), and the reps = 1 lane's 31 jobs at 2 s each (62 s). The cap never bound; each lane run stopped because its list was done. Job submissions spanned 39 min in the first pass (one job queued for 11 min), 11 min in the second, 7 min in the third and 15 min in the reps = 1 lane. The smoke job of 2026‑10‑14 (`db7s898r1c5c738l6su0`, the `none‑2026` instance, 2 s) sits in the same ledger; it ran against its own store, so that instance was paid for four times, 8 s in all.

One task never ran at reps = 2: `T0‑D2+D4‑2030` (24 qubits), because its gate was killed (section 7) and the lane refuses a task without a passed gate. At reps = 1 its gate passed in 484 s and the task ran.

### 7. The kills

At reps = 2, 31 tasks reached the 2 h limit of a quantum run: the 30 full noisy runs on `D1+D2‑2027`, `D1+D2‑2028` and `D2+D4‑2029` (binary and domain‑wall, five seeds each) and the gate of `D2+D4‑2030`. All are circuits whose routing on `FakeFez` spread them over more physical qubits than they have logical ones: `D2+D4‑2029` touches 23 physical qubits for 16 logical, `D1+D2‑2027` and `D1+D2‑2028` 24 for 20, `D2+D4‑2030` 29 for 24, where every circuit that ran touches exactly its logical count. The noisy simulator carries every physical qubit the circuit touches, so its cost grows exponentially with that spread: the 32‑shot gate of `D2+D4‑2029` took 250 s, which puts its 1,000‑shot run near 7,800 s. These cells are missing from the noise ladder; their hardware jobs ran regardless, since their gates had passed. They cost 62 worker‑hours, about 31 of the simulation's 56 h of wall time. The reps = 1 rung's five kills have the same cause (its section).

### 8. What the QAOA columns are not

The QAOA solvers sit outside the budget‑indexed crossover tables (section 10 of the design): a QAOA run is a shot count at fixed angles, not a time budget, so it has no 0.1 s, 1 s, 10 s or 60 s column and no crossover row. The comparison that holds is section 1, at matched reads, and the TTS99 of `qaoa_ibm` is QPU seconds only. A wall‑clock TTS99 would add the training (6 s to 90 min per instance at reps = 2) and the queue (1 s to 11 min per job) and would be larger by one to four orders of magnitude.

## The reps = 1 rung

### Why, and what changed

On 2026‑10‑27, with 166 s left under the cap after two passes, the owner chose to spend part of it on a depth rung (decision "A + B": a third pass of the main study, and this rung). The question: does a shallower circuit survive the device better? At reps = 1 the routed circuits should carry about half the CZ gates, which at the device's median CZ error raises the error‑free share of shots from 0.6–16% to roughly 9–46%. The configuration `experiments/m5-qaoa-p1.toml` (commit `d9a3de6`) is the main study's with `reps = 1`, the noise model on binary only, and no samplers (at matched reads they do not depend on `reps`; the main store already holds them on these instances). Everything else is the same: instances, encodings, caps, α, COBYLA's 200 evaluations, shots, device, gate and ledger.

### What ran

| solver | runs | per instance |
|---|---|---|
| `qaoa_ideal` | 940 | 20 seeds × (31 binary + 16 domain‑wall QUBOs), 1,000 shots |
| `qaoa_noisy`, full | 85 (80 ok, 5 killed) | 5 seeds × 17 binary QUBOs of ≤ 20 qubits, 1,000 shots |
| `qaoa_noisy`, gate | 31 (all passed) | one 32‑shot run per hardware task |
| `qaoa_ibm` | 31 | one job × 31 instances, 1,000 shots, binary |
| `exact_pwl_milp` | 40 | 1 s budget |
| `ref:qaoa_angles` | 47 | one per distinct QUBO |
| `ref:brute_force`, `ref:exact_pwl_milp`, `ref:qubo_minimum` | 40, 40, 80 | no budget |

1,334 tasks, 5 killed, 0 errors. The simulation ran in one stretch, 2026‑10‑27 about 17:04 to 2026‑10‑28 06:55 (about 13.9 h); the hardware lane submitted its 31 jobs on 2026‑10‑28 between 09:35 and 09:50, each 2 s of QPU time, queued 1.2–1.8 s. All 31 instances reached the device, including `D2+D4‑2030`, whose 24‑qubit gate was killed at reps = 2.

### The circuits: half the gates

Routed on the live device, the reps = 1 circuits carry 313–952 CZ gates (depth 359–864), against 735–2,044 (665–1,563) at reps = 2. Instance by instance the CZ count is 0.41 to 0.59 of the reps = 2 count (median 0.49), the depth 0.53 of it (median): one layer instead of two, plus a routing overhead that does not halve exactly. At a 0.25% CZ error, 46% of shots pass 313 gates without a two‑qubit error and 9% pass 952. The six 16‑qubit instances route to 313 or 325 gates.

### Noiselessly, a weaker optimiser

Per cell, binary encoding, reps = 2 → reps = 1 (`m5-p2-matched_reads.csv`, `m5-p1-matched_reads.csv`):

| cell | noiseless success | noiseless feasible | noise model success | device, reps 2 | device, reps 1 | device feasible |
|---|---|---|---|---|---|---|
| `D1` | 0.03 → 0.00 | 7.6% → 2.5% | — → — | 0 of 15 | 0 of 5 | 0.093% → 0.16% |
| `D1+D2` | 0.57 → 0.51 | 2.3% → 2.4% | 0.10 (10 of 20 killed) → 0.45 (5 of 20 killed) | 4 of 12 | 0 of 4 | 0.81% → 1.0% |
| `D1+D2+D4` | 1.00 → 0.97 | 14% → 5.7% | 0.93 → 0.73 | 3 of 9 | 2 of 3 | 1.1% → 2.4% |
| `D1+D4` | 0.56 → 0.41 | 8.3% → 3.2% | 0.60 → 0.60 | 1 of 15 | 1 of 5 | 0.36% → 0.38% |
| `D2` | 0.72 → 0.53 | 5.8% → 2.4% | 0.60 → 0.60 | 1 of 6 | 1 of 2 | 0.45% → 0.65% |
| `D2+D4` | 0.57 → 0.40 | 5.6% → 1.6% | killed → 0.60 | 1 of 3 | 1 of 2 | 1.1% → 0.9% |
| `D4` | 0.27 → 0.24 | 6.8% → 5.7% | 0.00 → 0.00 | 1 of 15 | 0 of 5 | 0.71% → 1.0% |
| `none` | 0.32 → 0.14 | 8.5% → 2.9% | 0.20 → 0.07 | 0 of 15 | 0 of 5 | 0.19% → 0.36% |

Without noise, one layer is worse than two on every cell but `D1+D2`'s feasibility: over the 31 binary QUBOs the mean success falls from 0.45 to 0.35 and the mean feasible share from 7.4% to 3.4%. The fall is largest where reps = 2 was weakest, on `none` (0.32 → 0.14) and the 20–24‑qubit cells; on the 16‑qubit cells with the cardinality dial both depths succeed in nearly every run. Domain‑wall behaves the same (appendix).

### The noise ladder and the device, instance by instance

Binary encoding, sorted by the reps = 2 CZ count; device successes are jobs within 1% of the optimum (`m5-p2-qaoa_noise.csv`, `m5-p1-qaoa_noise.csv`):

| instance | qubits | CZ, reps 2 → 1 | noiseless, 2 → 1 | noise model, 2 → 1 | device, reps 2 | device, reps 1 | device feasible, 2 → 1 |
|---|---|---|---|---|---|---|---|
| `D2+D4-2029` | 16 | 735 → 313 | 1.00 → 0.80 | killed → 0.60 | 1 of 3 | 1 of 1 | 1.1% → 1.7% |
| `D1+D2+D4-2026` | 16 | 736 → 313 | 1.00 → 0.95 | 1.00 → 0.40 | 1 of 3 | 1 of 1 | 0.33% → 1.7% |
| `D2-2026` | 16 | 736 → 313 | 1.00 → 0.90 | 1.00 → 1.00 | 1 of 3 | 1 of 1 | 0.8% → 1.3% |
| `D1+D2+D4-2028` | 16 | 736 → 313 | 1.00 → 1.00 | 0.80 → 1.00 | 2 of 3 | 0 of 1 | 1.2% → 1.5% |
| `D1+D2-2026` | 16 | 758 → 313 | 0.70 → 1.00 | 0.20 → 1.00 | 2 of 3 | 0 of 1 | 0.63% → 2.4% |
| `D1+D2+D4-2027` | 16 | 774 → 325 | 1.00 → 0.95 | 1.00 → 0.80 | 0 of 3 | 1 of 1 | 1.8% → 4.1% |
| `D1+D4-2027` | 18 | 873 → 378 | 1.00 → 0.95 | 0.80 → 1.00 | 0 of 3 | 1 of 1 | 0.2% → 0.5% |
| `D1+D2-2030` | 20 | 915 → 405 | 0.30 → 0.55 | 0.20 → killed | 2 of 3 | 0 of 1 | 2.5% → 1.3% |
| `D4-2030` | 18 | 962 → 487 | 0.00 → 0.15 | 0.00 → 0.00 | 0 of 3 | 0 of 1 | 1.0% → 2.5% |
| `D1+D4-2030` | 19 | 1,048 → 559 | 1.00 → 0.55 | 0.40 → 0.20 | 1 of 3 | 0 of 1 | 0.63% → 0.1% |
| `D2-2030` | 20 | 1,168 → 508 | 0.45 → 0.15 | 0.20 → 0.20 | 0 of 3 | 0 of 1 | 0.1% → 0% |
| `none-2027` | 20 | 1,219 → 621 | 0.50 → 0.15 | 0.20 → 0.00 | 0 of 3 | 0 of 1 | 0.067% → 0.1% |
| `none-2030` | 20 | 1,232 → 534 | 0.95 → 0.30 | 0.40 → 0.20 | 0 of 3 | 0 of 1 | 0.13% → 0% |
| `D4-2027` | 19 | 1,233 → 546 | 0.15 → 0.00 | 0.00 → 0.00 | 1 of 3 | 0 of 1 | 1.9% → 1.8% |
| `none-2028` | 20 | 1,261 → 681 | 0.15 → 0.20 | 0.00 → 0.00 | 0 of 3 | 0 of 1 | 0.67% → 1.6% |
| `D1+D2-2027` | 20 | 1,262 → 574 | 0.70 → 0.15 | killed → 0.40 | 0 of 3 | 0 of 1 | 0.067% → 0.4% |
| `D1+D2-2028` | 20 | 1,268 → 574 | 0.60 → 0.35 | killed → 0.40 | 0 of 3 | 0 of 1 | 0.033% → 0% |
| `D1+D4-2029` | 21 | 1,379 → 695 | 0.40 → 0.25 | — → — | 0 of 3 | 0 of 1 | 0.1% → 0.3% |
| `D4-2026` | 21 | 1,391 → 711 | 1.00 → 0.95 | — → — | 0 of 3 | 0 of 1 | 0.033% → 0% |
| `D1+D4-2026` | 21 | 1,400 → 675 | 0.40 → 0.25 | — → — | 0 of 3 | 0 of 1 | 0.23% → 0.4% |
| `D1+D4-2028` | 22 | 1,471 → 864 | 0.00 → 0.05 | — → — | 0 of 3 | 0 of 1 | 0.63% → 0.6% |
| `D4-2028` | 22 | 1,507 → 739 | 0.20 → 0.10 | — → — | 0 of 3 | 0 of 1 | 0.27% → 0.3% |
| `D4-2029` | 22 | 1,512 → 860 | 0.00 → 0.00 | — → — | 0 of 3 | 0 of 1 | 0.37% → 0.6% |
| `D1-2030` | 24 | 1,767 → 946 | 0.00 → 0.00 | — → — | 0 of 3 | 0 of 1 | 0.1% → 0.1% |
| `D1-2026` | 24 | 1,789 → 903 | 0.05 → 0.00 | — → — | 0 of 3 | 0 of 1 | 0% → 0.1% |
| `D1-2027` | 24 | 1,815 → 952 | 0.00 → 0.00 | — → — | 0 of 3 | 0 of 1 | 0.3% → 0.4% |
| `D1-2028` | 24 | 1,822 → 911 | 0.00 → 0.00 | — → — | 0 of 3 | 0 of 1 | 0.067% → 0.2% |
| `D1-2029` | 24 | 1,865 → 909 | 0.10 → 0.00 | — → — | 0 of 3 | 0 of 1 | 0% → 0% |
| `none-2026` | 24 | 1,930 → 946 | 0.00 → 0.05 | — → — | 0 of 3 | 0 of 1 | 0.1% → 0% |
| `none-2029` | 24 | 2,044 → 946 | 0.00 → 0.00 | — → — | 0 of 3 | 0 of 1 | 0% → 0.1% |
| `D2+D4-2030` | 24 | — → 635 | 0.15 → 0.00 | — → — | — | 0 of 1 | — → 0.1% |

- **The device at reps = 1:** 31 jobs, 242 feasible samples in 31,000 shots (0.78%), 6 jobs with none, 5 jobs within 1% of the optimum: `D2+D4‑2029`, `D2‑2026` and `D1+D2+D4‑2026` (313 CZ each), `D1+D2+D4‑2027` (325) and `D1+D4‑2027` (378, 18 qubits). Every success is on one of the seven shallowest circuits (at most 378 gates); the 24 circuits of 405 gates or more found nothing. On the 30 instances that ran at both depths the device's mean success is 0.17 with one job each, against 0.12 with three at reps = 2, and its feasible share 0.80% against 0.51%.
- **The ladder at reps = 1**, over the same 17 instances (killed as 0): noiseless 0.54, noise model 0.42, device 0.29 (one job each). On the 16 instances simulated in full the noise model's success is above the device's on 7, below on 3, equal on 6. Its feasible rate, 0.22–4.9%, again brackets the device's 0–4.1%.

### Against uniform random bits at reps = 1

(`m5-p1-uniform.csv`.) The 31 jobs returned 242 feasible samples where uniform bitstrings would give 81: a ratio of 3.0 (Poisson tail p ≈ 4 × 10⁻⁴⁷), against 2.0 at reps = 2. Per instance the device's ratio has a median of 3.3 (0–33), the noiseless circuit's a median of 20 (1.4–1,855), against 50 at reps = 2. Within 1% of the optimum: 5 jobs where chance gives about 1.9 (exact Poisson‑binomial tail 0.020, a lower bound): 4 of the 6 jobs on the 16‑qubit instances against 1.3 expected (binomial p = 0.023), and 1 of 25 on 18–24 qubits against 0.56.

So the shallower circuit carries more of its structure through the device (3.0 against 2.0 times uniform in feasibility), but it has less structure to carry (20 against 50 times uniform noiselessly), and the device's successes stay on the 16‑qubit instances, where chance already finds the optimum in 22% of jobs.

### Time to solution and training at reps = 1

Where the device succeeded at reps = 1, TTS99 at 1% is 8.4 s of QPU time on `D1+D2+D4` (2 of 3 jobs), 13.3 s on `D2` and `D2+D4` (1 of 2 each) and 41.3 s on `D1+D4` (1 of 5); `none`, `D1`, `D4` and `D1+D2` are censored (section 1's table). These rest on one to five jobs per cell, against greedy's 0.17–0.74 s on 100 runs.

Training is the one cost the shallower circuit cuts by much: 3.6 h for the 47 angle sets against 17.3 h, with half the parameters to fit (`m5-p1-qaoa_training.csv`):

| qubits | trainings | reps 2: median | reps 2: max | reps 1: median | reps 1: max | reps 1: median evaluations |
|---|---|---|---|---|---|---|
| 16 | 12 | 6.0 s | 7.4 s | 0.8 s | 15 s | 72 |
| 18 | 2 | 55 s | 66 s | 5.6 s | 5.9 s | 48 |
| 19 | 2 | 128 s | 153 s | 39 s | 57 s | 140 |
| 20 | 11 | 214 s | 333 s | 35 s | 61 s | 64 |
| 21 | 3 | 603 s | 604 s | 89 s | 162 s | 76 |
| 22 | 4 | 904 s | 933 s | 236 s | 279 s | 92 |
| 23 | 1 | 2,629 s | 2,629 s | 402 s | 402 s | 86 |
| 24 | 12 | 3,961 s | 5,421 s | 941 s | 1,431 s | 74 |

At reps = 1 COBYLA stopped before its 200 evaluations on 46 of the 47 trainings (31 to 146 evaluations), so these angles are converged where the reps = 2 angles were not: the reps = 1 rung's weaker noiseless success is the shallower circuit's, not an artefact of the budget. Per evaluation the cost still grows about 2.4× per qubit, from a median 0.8 s at 16 qubits to 16 min at 24.

### The kills at reps = 1

The five killed tasks are the five full noisy runs of `D1+D2‑2030`, 10 worker‑hours in all. They have the reps = 2 kills' cause: routed on `FakeFez` at reps = 1, this 20‑qubit circuit touches 25 physical qubits, against exactly 20 at reps = 2, where it ran. Conversely `D1+D2‑2027`, `D1+D2‑2028` and `D2+D4‑2029`, killed at reps = 2 (24, 24 and 23 physical qubits), route onto 20, 20 and 17 at reps = 1 and ran (scratch measurement of the routed circuits of both rungs from their angle caches, kept with the comparison document's provenance). Which circuits the noisy simulator can afford depends on the router's layout, not on the depth.

### Verdict on the rung

Halving the depth did for the device roughly what the error arithmetic predicts: the feasible share rose from 0.51% to 0.78%, from 2.0 to 3.0 times uniform, and the per‑job success on the shared instances from 0.12 to 0.17. It did the opposite for the algorithm: noiselessly the one‑layer circuit is a weaker optimiser on every cell, 0.45 against 0.35 in mean success and 7.4% against 3.4% feasible over the 31 QUBOs. The two effects nearly cancel on the device, whose successes stay where chance does a fifth of the work, on the 16‑qubit instances. Neither depth makes QAOA competitive with a classical sampler at the same reads (88–100% for greedy), and the depth that would let a circuit keep most of its shots error‑free is not available on this device: half the shots free of a two‑qubit error needs at most about 280 CZ gates at 0.25% per gate, or a CZ error of about 0.07% at 1,000 gates. The comparison document develops this.

## What next should take from it

- **Gate‑model QAOA with penalty‑encoded constraints is not a solver for these problems, with or without noise.** Feasibility is the failure: 2–14% feasible shots noiselessly at reps = 2 (1.6–5.7% at reps = 1), under 1% on the device, against 69–100% for annealing. Any further gate‑model attempt should change the algorithm before the device: constraint‑preserving circuits (XY or ring mixers on one‑hot or domain‑wall registers, Grover‑type mixers) that never leave the feasible subspace, so every shot is a candidate.
- **735 two‑qubit gates is already too deep at a 0.25% CZ error, and halving the depth does not rescue it.** The reps = 1 rung cut the gates to 313–952 and lifted the device's feasible rate from 2.0 to 3.0 times uniform, but the shallower circuit loses as much noiselessly as it gains on the device. Beyond that, nothing in this study's reach — not more shots, not more replicates, not error mitigation designed for expectation values — turns a near‑uniform sample distribution into a solver.
- **The classical side of QAOA is itself the bottleneck.** 66 min of exact‑statevector training at 24 qubits (16 min at reps = 1), 23 h at 28, against 0.4 s for a proof. Any larger instance needs a trainer that is not a statevector: fixed angle schedules, transfer of angles from smaller instances, or training on the device, each of which gives up the exact CVaR.
- **The annealing hardware the parent spec aimed at remains untested.** The QUBOs, the three encodings, the matched‑reads harness and the quota ledger are ready for it; a D‑Wave Leap account is what is missing. There, unlike here, the dynamic range of 10⁴–10⁶ would matter (analogue precision), and the 20–50× energy gaps the M4 samplers show on CVaR cells are the thing to measure.
- **T1 and beyond are out of reach for a gate device at this fidelity.** T1 QUBOs exceed 50 qubits and would route to several thousand CZ gates; the M4 result's "look at T2–T3" applies to annealers and classical methods, not to QAOA on Heron.
- **The quota left is 38 s until 2026‑11‑24.** The ledger refuses a job when less than 10 s would remain, so that is about 14 more 2 s jobs; not enough for a new pass over the 30 instances.
- **For the crossover question,** M4's answer stands and this study adds the quantum column to it: at T0, HiGHS proves the optimum in under 0.4 s, greedy finds it in under 1 s, and QAOA on hardware finds it, where it finds it at all, in 23–133 s of QPU time at reps = 2 (8–41 s at reps = 1) plus hours of preparation.

## Appendix: the uniform‑random baseline per instance, reps = 2

`uniform_frac` is the share of all bitstrings that decode to a feasible schedule (exact at 16 qubits, else from 100,000 draws); `exp_per_1000` the feasible samples a uniform 1,000‑shot job would give; `hw_feasible` what the three hardware jobs gave; `ratio_hw` and `ratio_ideal` the device's and the noiseless circuit's feasible rate over the uniform one; `k_within_1pct` the bitstrings within 1% of the optimum among those examined; `p_success_uniform` the chance a uniform 1,000‑shot job contains one (`m5-p2-uniform.csv`).

<!-- uniform-table -->
| instance | qubits | bitstrings examined | uniform feasible | expected per 1,000 shots | hardware feasible per job | hardware within 1% | jobs | device / uniform | noiseless / uniform | k within 1% | p(success) uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `D1+D2-2026` | 16 | exact | 0.0029 | 2.93 | 7+4+8 | 2 | 3 | 2.16 | 3.45 | 16 | 0.217 |
| `D1+D2+D4-2026` | 16 | exact | 0.0029 | 2.93 | 5+1+4 | 1 | 3 | 1.14 | 33.5 | 16 | 0.217 |
| `D1+D2+D4-2027` | 16 | exact | 0.0059 | 5.86 | 14+19+21 | 0 | 3 | 3.07 | 35.4 | 16 | 0.217 |
| `D1+D2+D4-2028` | 16 | exact | 0.0029 | 2.93 | 8+14+15 | 2 | 3 | 4.21 | 38.6 | 16 | 0.217 |
| `D2-2026` | 16 | exact | 0.0029 | 2.93 | 9+9+6 | 1 | 3 | 2.73 | 34 | 16 | 0.217 |
| `D2+D4-2029` | 16 | exact | 0.0029 | 2.93 | 12+12+9 | 1 | 3 | 3.75 | 20.8 | 16 | 0.217 |
| `D1+D4-2027` | 18 | 100,000 draws | 0.00015 | 0.15 | 0+4+2 | 0 | 3 | 13.3 | 373 | 2 | 0.0198 |
| `D4-2030` | 18 | 100,000 draws | 0.0075 | 7.51 | 3+9+19 | 0 | 3 | 1.38 | 3.98 | 0 | 0 |
| `D1+D4-2030` | 19 | 100,000 draws | 0.00063 | 0.63 | 6+6+7 | 1 | 3 | 10.1 | 168 | 3 | 0.0296 |
| `D4-2027` | 19 | 100,000 draws | 0.005 | 4.99 | 14+21+21 | 1 | 3 | 3.74 | 17.3 | 0 | 0 |
| `D1+D2-2027` | 20 | 100,000 draws | 0.00046 | 0.46 | 1+0+1 | 0 | 3 | 1.45 | 53.7 | 2 | 0.0198 |
| `D1+D2-2028` | 20 | 100,000 draws | 0.00022 | 0.22 | 0+0+1 | 0 | 3 | 1.52 | 50.2 | 2 | 0.0198 |
| `D1+D2-2030` | 20 | 100,000 draws | 0.0253 | 25.3 | 26+20+29 | 2 | 3 | 0.987 | 1.79 | 49 | 0.387 |
| `D2-2030` | 20 | 100,000 draws | 0.00045 | 0.45 | 1+2+0 | 0 | 3 | 2.22 | 37.6 | 2 | 0.0198 |
| `none-2027` | 20 | 100,000 draws | 0.0002 | 0.2 | 0+1+1 | 0 | 3 | 3.33 | 359 | 0 | 0 |
| `none-2028` | 20 | 100,000 draws | 0.0046 | 4.56 | 8+4+8 | 0 | 3 | 1.46 | 32.3 | 0 | 0 |
| `none-2030` | 20 | 100,000 draws | 0.00053 | 0.53 | 1+2+1 | 0 | 3 | 2.52 | 122 | 2 | 0.0198 |
| `D1+D4-2026` | 21 | 100,000 draws | 0.0016 | 1.59 | 2+2+3 | 0 | 3 | 1.47 | 50.3 | 1 | 0.01 |
| `D1+D4-2029` | 21 | 100,000 draws | 0.00049 | 0.49 | 0+1+2 | 0 | 3 | 2.04 | 168 | 0 | 0 |
| `D4-2026` | 21 | 100,000 draws | 1e-05 | 0.01 | 0+0+1 | 0 | 3 | 33.3 | 4.52e+03 | 0 | 0 |
| `D1+D4-2028` | 22 | 100,000 draws | 0.0023 | 2.3 | 5+7+7 | 0 | 3 | 2.75 | 40.4 | 0 | 0 |
| `D4-2028` | 22 | 100,000 draws | 0.00053 | 0.53 | 6+1+1 | 0 | 3 | 5.03 | 158 | 0 | 0 |
| `D4-2029` | 22 | 100,000 draws | 0.0023 | 2.32 | 3+2+6 | 0 | 3 | 1.58 | 40 | 0 | 0 |
| `D1-2026` | 24 | 100,000 draws | 0.00029 | 0.29 | 0+0+0 | 0 | 3 | 0 | 229 | 0 | 0 |
| `D1-2027` | 24 | 100,000 draws | 0.0025 | 2.49 | 3+1+5 | 0 | 3 | 1.2 | 42.2 | 0 | 0 |
| `D1-2028` | 24 | 100,000 draws | 0.00033 | 0.33 | 0+0+2 | 0 | 3 | 2.02 | 215 | 1 | 0.01 |
| `D1-2029` | 24 | 100,000 draws | 0.00034 | 0.34 | 0+0+0 | 0 | 3 | 0 | 204 | 0 | 0 |
| `D1-2030` | 24 | 100,000 draws | 0.00047 | 0.47 | 2+1+0 | 0 | 3 | 2.13 | 142 | 0 | 0 |
| `none-2026` | 24 | 100,000 draws | 0.00018 | 0.18 | 0+0+3 | 0 | 3 | 5.56 | 385 | 0 | 0 |
| `none-2029` | 24 | 100,000 draws | 0.00089 | 0.89 | 0+0+0 | 0 | 3 | 0 | 80.6 | 0 | 0 |
<!-- /uniform-table -->

## Appendix: the uniform‑random baseline per instance, reps = 1

The same columns for the reps = 1 rung's one job per instance; `ratio_ideal` is the reps = 1 noiseless circuit's (`m5-p1-uniform.csv`).

<!-- uniform-table-p1 -->
| instance | qubits | bitstrings examined | uniform feasible | expected per 1,000 shots | hardware feasible per job | hardware within 1% | jobs | device / uniform | noiseless / uniform | k within 1% | p(success) uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `D1+D2-2026` | 16 | exact | 0.0029 | 2.93 | 24 | 0 | 1 | 8.19 | 16.8 | 16 | 0.217 |
| `D1+D2+D4-2026` | 16 | exact | 0.0029 | 2.93 | 17 | 1 | 1 | 5.8 | 14.7 | 16 | 0.217 |
| `D1+D2+D4-2027` | 16 | exact | 0.0059 | 5.86 | 41 | 1 | 1 | 7 | 13.1 | 16 | 0.217 |
| `D1+D2+D4-2028` | 16 | exact | 0.0029 | 2.93 | 15 | 0 | 1 | 5.12 | 17.7 | 16 | 0.217 |
| `D2-2026` | 16 | exact | 0.0029 | 2.93 | 13 | 1 | 1 | 4.44 | 14.5 | 16 | 0.217 |
| `D2+D4-2029` | 16 | exact | 0.0029 | 2.93 | 17 | 1 | 1 | 5.8 | 9.35 | 16 | 0.217 |
| `D1+D4-2027` | 18 | 100,000 draws | 0.00015 | 0.15 | 5 | 1 | 1 | 33.3 | 203 | 2 | 0.0198 |
| `D4-2030` | 18 | 100,000 draws | 0.0075 | 7.51 | 25 | 0 | 1 | 3.33 | 16.1 | 0 | 0 |
| `D1+D4-2030` | 19 | 100,000 draws | 0.00063 | 0.63 | 1 | 0 | 1 | 1.59 | 43.3 | 3 | 0.0296 |
| `D4-2027` | 19 | 100,000 draws | 0.005 | 4.99 | 18 | 0 | 1 | 3.61 | 11 | 0 | 0 |
| `D1+D2-2027` | 20 | 100,000 draws | 0.00046 | 0.46 | 4 | 0 | 1 | 8.7 | 17.2 | 2 | 0.0198 |
| `D1+D2-2028` | 20 | 100,000 draws | 0.00022 | 0.22 | 0 | 0 | 1 | 0 | 19.1 | 2 | 0.0198 |
| `D1+D2-2030` | 20 | 100,000 draws | 0.0253 | 25.3 | 13 | 0 | 1 | 0.513 | 1.4 | 49 | 0.387 |
| `D2-2030` | 20 | 100,000 draws | 0.00045 | 0.45 | 0 | 0 | 1 | 0 | 11.4 | 2 | 0.0198 |
| `none-2027` | 20 | 100,000 draws | 0.0002 | 0.2 | 1 | 0 | 1 | 5 | 80 | 0 | 0 |
| `none-2028` | 20 | 100,000 draws | 0.0046 | 4.56 | 16 | 0 | 1 | 3.51 | 14 | 0 | 0 |
| `none-2030` | 20 | 100,000 draws | 0.00053 | 0.53 | 0 | 0 | 1 | 0 | 38.4 | 2 | 0.0198 |
| `D1+D4-2026` | 21 | 100,000 draws | 0.0016 | 1.59 | 4 | 0 | 1 | 2.52 | 22.7 | 1 | 0.01 |
| `D1+D4-2029` | 21 | 100,000 draws | 0.00049 | 0.49 | 3 | 0 | 1 | 6.12 | 46.9 | 0 | 0 |
| `D4-2026` | 21 | 100,000 draws | 1e-05 | 0.01 | 0 | 0 | 1 | 0 | 1.86e+03 | 0 | 0 |
| `D1+D4-2028` | 22 | 100,000 draws | 0.0023 | 2.3 | 6 | 0 | 1 | 2.61 | 19.3 | 0 | 0 |
| `D4-2028` | 22 | 100,000 draws | 0.00053 | 0.53 | 3 | 0 | 1 | 5.66 | 66.8 | 0 | 0 |
| `D4-2029` | 22 | 100,000 draws | 0.0023 | 2.32 | 6 | 0 | 1 | 2.59 | 24.2 | 0 | 0 |
| `D1-2026` | 24 | 100,000 draws | 0.00029 | 0.29 | 1 | 0 | 1 | 3.45 | 60.5 | 0 | 0 |
| `D1-2027` | 24 | 100,000 draws | 0.0025 | 2.49 | 4 | 0 | 1 | 1.61 | 20.4 | 0 | 0 |
| `D1-2028` | 24 | 100,000 draws | 0.00033 | 0.33 | 2 | 0 | 1 | 6.06 | 55.3 | 1 | 0.01 |
| `D1-2029` | 24 | 100,000 draws | 0.00034 | 0.34 | 0 | 0 | 1 | 0 | 56.8 | 0 | 0 |
| `D1-2030` | 24 | 100,000 draws | 0.00047 | 0.47 | 1 | 0 | 1 | 2.13 | 37.4 | 0 | 0 |
| `D2+D4-2030` | 24 | 100,000 draws | 0.0034 | 3.38 | 1 | 0 | 1 | 0.296 | 1.42 | 2 | 0.0198 |
| `none-2026` | 24 | 100,000 draws | 0.00018 | 0.18 | 0 | 0 | 1 | 0 | 98.3 | 0 | 0 |
| `none-2029` | 24 | 100,000 draws | 0.00089 | 0.89 | 1 | 0 | 1 | 1.12 | 31.3 | 0 | 0 |
<!-- /uniform-table-p1 -->

## Appendix: every hardware job

The 122 jobs of the quota ledger in submission order, joined with the stores' hardware rows by `experiments/m5-hardware-jobs.py` (`m5-hardware-jobs.csv`): lane (`smoke`, `reps2`, `reps1`), pass within the lane, instance, IBM's QPU charge, queue seconds, routed CZ count, feasible samples, and whether the best feasible sample is within 1% of the optimum. The ledger itself is regenerated as the report's `quota` table by `quantic bench report <store> --out <dir> --ledger <path>`.

<!-- jobs -->
| # | job id | submitted (local) | lane | pass | instance | QPU s | queue s | routed CZ | feasible / shots | within 1% |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `db7s898r1c5c738l6su0` | 2026-10-14 20:04:44 | smoke | 1 | `none-2026` | 2 | 40,980.4 | 1,930 | 1 / 1,000 | no |
| 2 | `dbgbv75rd2mc73cudk90` | 2026-10-27 16:13:16 | reps2 | 1 | `none-2026` | 2 | 2.4 | 1,930 | 0 / 1,000 | no |
| 3 | `dbgc5ldjm0ks73egb0rg` | 2026-10-27 16:27:02 | reps2 | 1 | `D1-2026` | 3 | 668.3 | 1,789 | 0 / 1,000 | no |
| 4 | `dbgccm5jm0ks73egbc20` | 2026-10-27 16:42:01 | reps2 | 1 | `D2-2026` | 2 | 6.2 | 736 | 9 / 1,000 | yes |
| 5 | `dbgccs1j4t8s73ac2olg` | 2026-10-27 16:42:25 | reps2 | 1 | `D4-2026` | 2 | 1.9 | 1,391 | 0 / 1,000 | no |
| 6 | `dbgcd0giudkc73e47240` | 2026-10-27 16:42:42 | reps2 | 1 | `D1+D2-2026` | 2 | 1.1 | 758 | 7 / 1,000 | yes |
| 7 | `dbgcd48iudkc73e472c0` | 2026-10-27 16:42:57 | reps2 | 1 | `D1+D4-2026` | 2 | 1.6 | 1,400 | 2 / 1,000 | no |
| 8 | `dbgcd7trd2mc73cuecj0` | 2026-10-27 16:43:12 | reps2 | 1 | `D1+D2+D4-2026` | 2 | 1.1 | 736 | 5 / 1,000 | yes |
| 9 | `dbgcdblrd2mc73cuecr0` | 2026-10-27 16:43:27 | reps2 | 1 | `none-2027` | 2 | 1.9 | 1,219 | 0 / 1,000 | no |
| 10 | `dbgcdedrd2mc73cued00` | 2026-10-27 16:43:38 | reps2 | 1 | `D1-2027` | 2 | 1.4 | 1,815 | 3 / 1,000 | no |
| 11 | `dbgcdh5rd2mc73cued60` | 2026-10-27 16:43:49 | reps2 | 1 | `D4-2027` | 2 | 1.1 | 1,233 | 14 / 1,000 | no |
| 12 | `dbgcdklrd2mc73cuede0` | 2026-10-27 16:44:03 | reps2 | 1 | `D1+D2-2027` | 3 | 1.9 | 1,262 | 1 / 1,000 | no |
| 13 | `dbgcdodrd2mc73cuedm0` | 2026-10-27 16:44:18 | reps2 | 1 | `D1+D4-2027` | 2 | 1.2 | 873 | 0 / 1,000 | no |
| 14 | `dbgcds8iudkc73e473qg` | 2026-10-27 16:44:34 | reps2 | 1 | `D1+D2+D4-2027` | 2 | 2.0 | 774 | 14 / 1,000 | no |
| 15 | `dbgce09j4t8s73ac2qo0` | 2026-10-27 16:44:50 | reps2 | 1 | `none-2028` | 2 | 1.8 | 1,261 | 8 / 1,000 | no |
| 16 | `dbgce48iudkc73e474a0` | 2026-10-27 16:45:05 | reps2 | 1 | `D1-2028` | 2 | 1.9 | 1,822 | 0 / 1,000 | no |
| 17 | `dbgcei5rd2mc73cuef2g` | 2026-10-27 16:46:00 | reps2 | 1 | `D4-2028` | 2 | 1.4 | 1,507 | 6 / 1,000 | no |
| 18 | `dbgceltjm0ks73egbfq0` | 2026-10-27 16:46:16 | reps2 | 1 | `D1+D2-2028` | 3 | 1.2 | 1,268 | 0 / 1,000 | no |
| 19 | `dbgcet5jm0ks73egbg9g` | 2026-10-27 16:46:45 | reps2 | 1 | `D1+D4-2028` | 2 | 1.5 | 1,471 | 5 / 1,000 | no |
| 20 | `dbgcf45rd2mc73cueg50` | 2026-10-27 16:47:13 | reps2 | 1 | `D1+D2+D4-2028` | 2 | 1.4 | 736 | 8 / 1,000 | yes |
| 21 | `dbgcf80iudkc73e476g0` | 2026-10-27 16:47:29 | reps2 | 1 | `none-2029` | 3 | 1.1 | 2,044 | 0 / 1,000 | no |
| 22 | `dbgcfbtrd2mc73cuegig` | 2026-10-27 16:47:44 | reps2 | 1 | `D1-2029` | 3 | 2.0 | 1,865 | 0 / 1,000 | no |
| 23 | `dbgcfftrd2mc73cuegs0` | 2026-10-27 16:48:00 | reps2 | 1 | `D4-2029` | 2 | 3.9 | 1,512 | 3 / 1,000 | no |
| 24 | `dbgcfpljm0ks73egbi4g` | 2026-10-27 16:48:39 | reps2 | 1 | `D1+D4-2029` | 2 | 1.6 | 1,379 | 0 / 1,000 | no |
| 25 | `dbgcg6drd2mc73cueid0` | 2026-10-27 16:49:30 | reps2 | 1 | `D2+D4-2029` | 2 | 5.5 | 735 | 12 / 1,000 | no |
| 26 | `dbgcgeoiudkc73e47950` | 2026-10-27 16:50:04 | reps2 | 1 | `none-2030` | 2 | 6.8 | 1,232 | 1 / 1,000 | no |
| 27 | `dbgcgilrd2mc73cuej70` | 2026-10-27 16:50:19 | reps2 | 1 | `D1-2030` | 2 | 7.2 | 1,767 | 2 / 1,000 | no |
| 28 | `dbgcgmlrd2mc73cuejig` | 2026-10-27 16:50:35 | reps2 | 1 | `D2-2030` | 3 | 7.6 | 1,168 | 1 / 1,000 | no |
| 29 | `dbgcgqlrd2mc73cuejsg` | 2026-10-27 16:50:51 | reps2 | 1 | `D4-2030` | 2 | 7.1 | 962 | 3 / 1,000 | no |
| 30 | `dbgcgulrd2mc73cuek70` | 2026-10-27 16:51:07 | reps2 | 1 | `D1+D2-2030` | 2 | 2.7 | 915 | 26 / 1,000 | yes |
| 31 | `dbgch8djm0ks73egblc0` | 2026-10-27 16:51:46 | reps2 | 1 | `D1+D4-2030` | 2 | 1.4 | 1,048 | 6 / 1,000 | no |
| 32 | `dbgcm58iudkc73e47p00` | 2026-10-27 17:02:13 | reps2 | 2 | `none-2026` | 2 | 1.2 | 1,930 | 0 / 1,000 | no |
| 33 | `dbgcm99j4t8s73ac3flg` | 2026-10-27 17:02:30 | reps2 | 2 | `D1-2026` | 3 | 1.3 | 1,789 | 0 / 1,000 | no |
| 34 | `dbgcmd5jm0ks73egc2pg` | 2026-10-27 17:02:45 | reps2 | 2 | `D2-2026` | 2 | 1.4 | 736 | 9 / 1,000 | no |
| 35 | `dbgcmglrd2mc73cuf2u0` | 2026-10-27 17:02:59 | reps2 | 2 | `D4-2026` | 2 | 4.3 | 1,391 | 0 / 1,000 | no |
| 36 | `dbgcmppj4t8s73ac3gvg` | 2026-10-27 17:03:36 | reps2 | 2 | `D1+D2-2026` | 2 | 1.7 | 758 | 4 / 1,000 | yes |
| 37 | `dbgcmt5jm0ks73egc41g` | 2026-10-27 17:03:49 | reps2 | 2 | `D1+D4-2026` | 2 | 3.5 | 1,400 | 2 / 1,000 | no |
| 38 | `dbgcn0oiudkc73e47qlg` | 2026-10-27 17:04:04 | reps2 | 2 | `D1+D2+D4-2026` | 2 | 1.1 | 736 | 1 / 1,000 | no |
| 39 | `dbgcn98iudkc73e47r9g` | 2026-10-27 17:04:38 | reps2 | 2 | `none-2027` | 2 | 1.2 | 1,219 | 1 / 1,000 | no |
| 40 | `dbgcnf5jm0ks73egc5ag` | 2026-10-27 17:05:01 | reps2 | 2 | `D1-2027` | 2 | 1.3 | 1,815 | 1 / 1,000 | no |
| 41 | `dbgcni5rd2mc73cuf590` | 2026-10-27 17:05:12 | reps2 | 2 | `D4-2027` | 2 | 1.2 | 1,233 | 21 / 1,000 | no |
| 42 | `dbgcnotjm0ks73egc5ug` | 2026-10-27 17:05:40 | reps2 | 2 | `D1+D2-2027` | 3 | 1.7 | 1,262 | 0 / 1,000 | no |
| 43 | `dbgcnt0iudkc73e47su0` | 2026-10-27 17:05:57 | reps2 | 2 | `D1+D4-2027` | 2 | 1.4 | 873 | 4 / 1,000 | no |
| 44 | `dbgco1lrd2mc73cuf6cg` | 2026-10-27 17:06:14 | reps2 | 2 | `D1+D2+D4-2027` | 2 | 1.1 | 774 | 19 / 1,000 | no |
| 45 | `dbgco95rd2mc73cuf73g` | 2026-10-27 17:06:45 | reps2 | 2 | `none-2028` | 2 | 1.9 | 1,261 | 4 / 1,000 | no |
| 46 | `dbgcodpj4t8s73ac3kl0` | 2026-10-27 17:07:04 | reps2 | 2 | `D1-2028` | 2 | 3.4 | 1,822 | 0 / 1,000 | no |
| 47 | `dbgcohpj4t8s73ac3kvg` | 2026-10-27 17:07:20 | reps2 | 2 | `D4-2028` | 2 | 1.2 | 1,507 | 1 / 1,000 | no |
| 48 | `dbgcoldrd2mc73cuf7s0` | 2026-10-27 17:07:34 | reps2 | 2 | `D1+D2-2028` | 3 | 1.5 | 1,268 | 0 / 1,000 | no |
| 49 | `dbgcop8iudkc73e47usg` | 2026-10-27 17:07:50 | reps2 | 2 | `D1+D4-2028` | 2 | 1.2 | 1,471 | 7 / 1,000 | no |
| 50 | `dbgcot0iudkc73e47v4g` | 2026-10-27 17:08:05 | reps2 | 2 | `D1+D2+D4-2028` | 2 | 1.5 | 736 | 14 / 1,000 | yes |
| 51 | `dbgcpi5rd2mc73cufa2g` | 2026-10-27 17:09:29 | reps2 | 2 | `none-2029` | 3 | 1.2 | 2,044 | 0 / 1,000 | no |
| 52 | `dbgcpmdrd2mc73cufac0` | 2026-10-27 17:09:45 | reps2 | 2 | `D1-2029` | 3 | 1.3 | 1,865 | 0 / 1,000 | no |
| 53 | `dbgcqb0iudkc73e482e0` | 2026-10-27 17:11:09 | reps2 | 2 | `D4-2029` | 2 | 1.6 | 1,512 | 2 / 1,000 | no |
| 54 | `dbgcqdoiudkc73e482n0` | 2026-10-27 17:11:20 | reps2 | 2 | `D1+D4-2029` | 2 | 1.8 | 1,379 | 1 / 1,000 | no |
| 55 | `dbgcqi0iudkc73e48300` | 2026-10-27 17:11:37 | reps2 | 2 | `D2+D4-2029` | 2 | 1.1 | 735 | 12 / 1,000 | no |
| 56 | `dbgcql5jm0ks73egccd0` | 2026-10-27 17:11:49 | reps2 | 2 | `none-2030` | 2 | 1.7 | 1,232 | 2 / 1,000 | no |
| 57 | `dbgcqq5rd2mc73cufcrg` | 2026-10-27 17:12:09 | reps2 | 2 | `D1-2030` | 2 | 1.1 | 1,767 | 1 / 1,000 | no |
| 58 | `dbgcqsoiudkc73e483q0` | 2026-10-27 17:12:20 | reps2 | 2 | `D2-2030` | 3 | 5.7 | 1,168 | 2 / 1,000 | no |
| 59 | `dbgcr6lrd2mc73cufdqg` | 2026-10-27 17:12:59 | reps2 | 2 | `D4-2030` | 2 | 1.5 | 962 | 9 / 1,000 | no |
| 60 | `dbgcra8iudkc73e484l0` | 2026-10-27 17:13:14 | reps2 | 2 | `D1+D2-2030` | 2 | 1.1 | 915 | 20 / 1,000 | no |
| 61 | `dbgcrgtrd2mc73cufedg` | 2026-10-27 17:13:40 | reps2 | 2 | `D1+D4-2030` | 2 | 1.2 | 1,048 | 6 / 1,000 | yes |
| 62 | `dbgr7m1t9uhs73c8qj90` | 2026-10-28 09:35:24 | reps1 | 1 | `none-2026` | 2 | 1.4 | 946 | 0 / 1,000 | no |
| 63 | `dbgr800te5oc738iq0eg` | 2026-10-28 09:36:04 | reps1 | 1 | `D1-2026` | 2 | 1.5 | 903 | 1 / 1,000 | no |
| 64 | `dbgr8mht9uhs73c8qkp0` | 2026-10-28 09:37:34 | reps1 | 1 | `D2-2026` | 2 | 1.5 | 313 | 13 / 1,000 | yes |
| 65 | `dbgr8tsdgr2s73cs4sg0` | 2026-10-28 09:38:03 | reps1 | 1 | `D4-2026` | 2 | 1.2 | 711 | 0 / 1,000 | no |
| 66 | `dbgr974dgr2s73cs4sug` | 2026-10-28 09:38:40 | reps1 | 1 | `D1+D2-2026` | 2 | 1.2 | 313 | 24 / 1,000 | no |
| 67 | `dbgr9gkdgr2s73cs4te0` | 2026-10-28 09:39:18 | reps1 | 1 | `D1+D4-2026` | 2 | 1.5 | 675 | 4 / 1,000 | no |
| 68 | `dbgr9psdgr2s73cs4tvg` | 2026-10-28 09:39:55 | reps1 | 1 | `D1+D2+D4-2026` | 2 | 1.4 | 313 | 17 / 1,000 | yes |
| 69 | `dbgra41q260s73bttk2g` | 2026-10-28 09:40:36 | reps1 | 1 | `none-2027` | 2 | 1.6 | 621 | 1 / 1,000 | no |
| 70 | `dbgra6pq260s73bttk90` | 2026-10-28 09:40:47 | reps1 | 1 | `D1-2027` | 2 | 1.4 | 952 | 4 / 1,000 | no |
| 71 | `dbgrao1q260s73bttl9g` | 2026-10-28 09:41:56 | reps1 | 1 | `D4-2027` | 2 | 1.6 | 546 | 18 / 1,000 | no |
| 72 | `dbgrar1q260s73bttle0` | 2026-10-28 09:42:07 | reps1 | 1 | `D1+D2-2027` | 2 | 1.4 | 574 | 4 / 1,000 | no |
| 73 | `dbgrb91t9uhs73c8qph0` | 2026-10-28 09:43:04 | reps1 | 1 | `D1+D4-2027` | 2 | 1.7 | 378 | 5 / 1,000 | yes |
| 74 | `dbgrbe1q260s73bttmrg` | 2026-10-28 09:43:24 | reps1 | 1 | `D1+D2+D4-2027` | 2 | 1.4 | 325 | 41 / 1,000 | yes |
| 75 | `dbgrbgsdgr2s73cs515g` | 2026-10-28 09:43:35 | reps1 | 1 | `none-2028` | 2 | 1.6 | 681 | 16 / 1,000 | no |
| 76 | `dbgrbk0te5oc738iq7bg` | 2026-10-28 09:43:47 | reps1 | 1 | `D1-2028` | 2 | 1.5 | 911 | 2 / 1,000 | no |
| 77 | `dbgrcopq260s73bttphg` | 2026-10-28 09:46:15 | reps1 | 1 | `D4-2028` | 2 | 1.5 | 739 | 3 / 1,000 | no |
| 78 | `dbgrcrhq260s73bttpp0` | 2026-10-28 09:46:26 | reps1 | 1 | `D1+D2-2028` | 2 | 1.7 | 574 | 0 / 1,000 | no |
| 79 | `dbgrd29q260s73bttqbg` | 2026-10-28 09:46:53 | reps1 | 1 | `D1+D4-2028` | 2 | 1.5 | 864 | 6 / 1,000 | no |
| 80 | `dbgrd51t9uhs73c8qtog` | 2026-10-28 09:47:04 | reps1 | 1 | `D1+D2+D4-2028` | 2 | 1.5 | 313 | 15 / 1,000 | no |
| 81 | `dbgrd7pq260s73bttqo0` | 2026-10-28 09:47:15 | reps1 | 1 | `none-2029` | 2 | 1.6 | 946 | 1 / 1,000 | no |
| 82 | `dbgrdapt9uhs73c8qu50` | 2026-10-28 09:47:27 | reps1 | 1 | `D1-2029` | 2 | 1.8 | 909 | 0 / 1,000 | no |
| 83 | `dbgrdigte5oc738iqbug` | 2026-10-28 09:47:58 | reps1 | 1 | `D4-2029` | 2 | 1.2 | 860 | 6 / 1,000 | no |
| 84 | `dbgrdl9t9uhs73c8qv30` | 2026-10-28 09:48:09 | reps1 | 1 | `D1+D4-2029` | 2 | 1.4 | 695 | 3 / 1,000 | no |
| 85 | `dbgrdo0te5oc738iqcfg` | 2026-10-28 09:48:20 | reps1 | 1 | `D2+D4-2029` | 2 | 1.6 | 313 | 17 / 1,000 | yes |
| 86 | `dbgrdr1q260s73btts70` | 2026-10-28 09:48:31 | reps1 | 1 | `none-2030` | 2 | 1.4 | 534 | 0 / 1,000 | no |
| 87 | `dbgre2ote5oc738iqd8g` | 2026-10-28 09:49:03 | reps1 | 1 | `D1-2030` | 2 | 1.3 | 946 | 1 / 1,000 | no |
| 88 | `dbgre5sdgr2s73cs57ag` | 2026-10-28 09:49:15 | reps1 | 1 | `D2-2030` | 2 | 1.4 | 508 | 0 / 1,000 | no |
| 89 | `dbgre89t9uhs73c8r0kg` | 2026-10-28 09:49:25 | reps1 | 1 | `D4-2030` | 2 | 1.7 | 487 | 25 / 1,000 | no |
| 90 | `dbgreb9t9uhs73c8r0qg` | 2026-10-28 09:49:37 | reps1 | 1 | `D1+D2-2030` | 2 | 1.5 | 405 | 13 / 1,000 | no |
| 91 | `dbgreikdgr2s73cs5860` | 2026-10-28 09:50:06 | reps1 | 1 | `D1+D4-2030` | 2 | 1.3 | 559 | 1 / 1,000 | no |
| 92 | `dbgrel8te5oc738iqekg` | 2026-10-28 09:50:17 | reps1 | 1 | `D2+D4-2030` | 2 | 1.7 | 635 | 1 / 1,000 | no |
| 93 | `dbgrfbcdgr2s73cs5a90` | 2026-10-28 09:51:44 | reps2 | 3 | `none-2026` | 2 | 1.3 | 1,930 | 3 / 1,000 | no |
| 94 | `dbgrfe4dgr2s73cs5ai0` | 2026-10-28 09:51:56 | reps2 | 3 | `D1-2026` | 3 | 1.2 | 1,789 | 0 / 1,000 | no |
| 95 | `dbgrfhsdgr2s73cs5aqg` | 2026-10-28 09:52:11 | reps2 | 3 | `D2-2026` | 2 | 1.0 | 736 | 6 / 1,000 | no |
| 96 | `dbgrfkht9uhs73c8r3rg` | 2026-10-28 09:52:22 | reps2 | 3 | `D4-2026` | 2 | 1.3 | 1,391 | 1 / 1,000 | no |
| 97 | `dbgrfn1q260s73btu0v0` | 2026-10-28 09:52:32 | reps2 | 3 | `D1+D2-2026` | 2 | 1.2 | 758 | 8 / 1,000 | no |
| 98 | `dbgrfq1t9uhs73c8r4dg` | 2026-10-28 09:52:44 | reps2 | 3 | `D1+D4-2026` | 2 | 1.3 | 1,400 | 3 / 1,000 | no |
| 99 | `dbgrfsgte5oc738iqhhg` | 2026-10-28 09:52:54 | reps2 | 3 | `D1+D2+D4-2026` | 2 | 1.4 | 736 | 4 / 1,000 | no |
| 100 | `dbgrfvgte5oc738iqhq0` | 2026-10-28 09:53:06 | reps2 | 3 | `none-2027` | 2 | 1.5 | 1,219 | 1 / 1,000 | no |
| 101 | `dbgrg28te5oc738iqi0g` | 2026-10-28 09:53:18 | reps2 | 3 | `D1-2027` | 2 | 1.3 | 1,815 | 5 / 1,000 | no |
| 102 | `dbgrg59t9uhs73c8r5ag` | 2026-10-28 09:53:28 | reps2 | 3 | `D4-2027` | 2 | 1.2 | 1,233 | 21 / 1,000 | yes |
| 103 | `dbgrg7pt9uhs73c8r5gg` | 2026-10-28 09:53:39 | reps2 | 3 | `D1+D2-2027` | 3 | 1.8 | 1,262 | 1 / 1,000 | no |
| 104 | `dbgrgapt9uhs73c8r5n0` | 2026-10-28 09:53:51 | reps2 | 3 | `D1+D4-2027` | 2 | 1.3 | 873 | 2 / 1,000 | no |
| 105 | `dbgrgn1t9uhs73c8r6c0` | 2026-10-28 09:54:40 | reps2 | 3 | `D1+D2+D4-2027` | 2 | 1.4 | 774 | 21 / 1,000 | no |
| 106 | `dbgrgpote5oc738iqjn0` | 2026-10-28 09:54:51 | reps2 | 3 | `none-2028` | 2 | 1.1 | 1,261 | 8 / 1,000 | no |
| 107 | `dbgrgsgte5oc738iqjvg` | 2026-10-28 09:55:02 | reps2 | 3 | `D1-2028` | 2 | 1.4 | 1,822 | 2 / 1,000 | no |
| 108 | `dbgrgv9t9uhs73c8r6s0` | 2026-10-28 09:55:13 | reps2 | 3 | `D4-2028` | 2 | 1.2 | 1,507 | 1 / 1,000 | no |
| 109 | `dbgrh21t9uhs73c8r70g` | 2026-10-28 09:55:24 | reps2 | 3 | `D1+D2-2028` | 3 | 1.5 | 1,268 | 1 / 1,000 | no |
| 110 | `dbgrh4pt9uhs73c8r760` | 2026-10-28 09:55:35 | reps2 | 3 | `D1+D4-2028` | 2 | 1.5 | 1,471 | 7 / 1,000 | no |
| 111 | `dbgrh7ht9uhs73c8r7e0` | 2026-10-28 09:55:46 | reps2 | 3 | `D1+D2+D4-2028` | 2 | 1.1 | 736 | 15 / 1,000 | no |
| 112 | `dbgrhbote5oc738iql10` | 2026-10-28 09:56:03 | reps2 | 3 | `none-2029` | 3 | 1.3 | 2,044 | 0 / 1,000 | no |
| 113 | `dbgrhn4dgr2s73cs5ffg` | 2026-10-28 09:56:48 | reps2 | 3 | `D1-2029` | 3 | 2.6 | 1,865 | 0 / 1,000 | no |
| 114 | `dbgrhq9t9uhs73c8r8rg` | 2026-10-28 09:57:01 | reps2 | 3 | `D4-2029` | 2 | 3.5 | 1,512 | 6 / 1,000 | no |
| 115 | `dbgrhtpt9uhs73c8r950` | 2026-10-28 09:57:15 | reps2 | 3 | `D1+D4-2029` | 2 | 2.3 | 1,379 | 2 / 1,000 | no |
| 116 | `dbgri0ht9uhs73c8r9cg` | 2026-10-28 09:57:27 | reps2 | 3 | `D2+D4-2029` | 2 | 1.5 | 735 | 9 / 1,000 | yes |
| 117 | `dbgri3hq260s73btu6a0` | 2026-10-28 09:57:38 | reps2 | 3 | `none-2030` | 2 | 1.4 | 1,232 | 1 / 1,000 | no |
| 118 | `dbgri61q260s73btu6f0` | 2026-10-28 09:57:48 | reps2 | 3 | `D1-2030` | 2 | 1.1 | 1,767 | 0 / 1,000 | no |
| 119 | `dbgri8sdgr2s73cs5gl0` | 2026-10-28 09:57:58 | reps2 | 3 | `D2-2030` | 3 | 1.6 | 1,168 | 0 / 1,000 | no |
| 120 | `dbgribhq260s73btu6rg` | 2026-10-28 09:58:10 | reps2 | 3 | `D4-2030` | 2 | 1.0 | 962 | 19 / 1,000 | no |
| 121 | `dbgrie9t9uhs73c8rae0` | 2026-10-28 09:58:21 | reps2 | 3 | `D1+D2-2030` | 2 | 1.4 | 915 | 29 / 1,000 | yes |
| 122 | `dbgrigpq260s73btu780` | 2026-10-28 09:58:31 | reps2 | 3 | `D1+D4-2030` | 2 | 1.9 | 1,048 | 7 / 1,000 | no |
<!-- /jobs -->

<!-- tables -->
## Appendix: the report's tables, reps = 2

Regenerated by `quantic bench report <store> --out <dir>` from the main study's store.

### Matched reads (`matched_reads.md`)

Every solver at 1,000 reads; `tts` is QPU seconds for `qaoa_ibm`, wall-clock seconds for the samplers, and "simulated" for the simulators.

| tier | dials | solver | encoding | max_samples | n_runs | p_success_0.01 | p_success_0.05 | gap_median | feasibility | tts |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | D1 | greedy | binary | 1000 | 100 | 0.97 | 1.0 | 0.0 | 0.95892 | 0.735s |
| T0 | D1 | greedy | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.5171 | 0.353s |
| T0 | D1 | qaoa_ibm | binary | 1000 | 15 | 0.0 | 0.0 | 0.3192624238476382 | 0.0009333333333333333 | censored |
| T0 | D1 | qaoa_ideal | binary | 1000 | 100 | 0.03 | 0.52 | 0.04739289237474535 | 0.07568 | simulated |
| T0 | D1 | simulated_annealing | binary | 1000 | 100 | 0.54 | 0.96 | 0.009283043068350016 | 1.0 | 6.65s |
| T0 | D1 | simulated_annealing | domain_wall | 1000 | 100 | 0.55 | 0.9 | 0.008839746229817019 | 1.0 | 7.09s |
| T0 | D1 | tabu | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 41.6s |
| T0 | D1 | tabu | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 65.3s |
| T0 | D1+D2 | greedy | binary | 1000 | 100 | 0.9 | 0.99 | 0.0 | 0.35235 | 0.296s |
| T0 | D1+D2 | greedy | domain_wall | 1000 | 100 | 0.9 | 0.99 | 0.0 | 0.35259 | 0.301s |
| T0 | D1+D2 | qaoa_ibm | binary | 1000 | 12 | 0.3333333333333333 | 0.3333333333333333 | 0.07158557961671795 | 0.008083333333333333 | 28.4s |
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
| T0 | D1+D2+D4 | qaoa_ibm | binary | 1000 | 9 | 0.3333333333333333 | 0.3333333333333333 | 0.05082946430618634 | 0.011222222222222222 | 22.7s |
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
| T0 | D1+D4 | qaoa_ibm | binary | 1000 | 15 | 0.06666666666666667 | 0.4 | 0.05181150662412858 | 0.0036 | 133s |
| T0 | D1+D4 | qaoa_ideal | binary | 1000 | 100 | 0.56 | 0.96 | 0.008669052890741467 | 0.08345 | simulated |
| T0 | D1+D4 | qaoa_ideal | domain_wall | 1000 | 40 | 0.9 | 1.0 | 0.0 | 0.0579 | simulated |
| T0 | D1+D4 | qaoa_noisy | binary | 1000 | 10 | 0.6 | 1.0 | 0.007070811174385927 | 0.0162 | simulated |
| T0 | D1+D4 | simulated_annealing | binary | 1000 | 100 | 0.94 | 1.0 | 0.0 | 1.0 | 1.44s |
| T0 | D1+D4 | simulated_annealing | domain_wall | 1000 | 100 | 0.95 | 1.0 | 0.0 | 1.0 | 1.53s |
| T0 | D1+D4 | tabu | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 42.2s |
| T0 | D1+D4 | tabu | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 41.6s |
| T0 | D2 | greedy | binary | 1000 | 100 | 0.88 | 0.9 | 0.0 | 0.19756 | 0.474s |
| T0 | D2 | greedy | domain_wall | 1000 | 100 | 0.88 | 0.9 | 0.0 | 0.19786 | 0.442s |
| T0 | D2 | qaoa_ibm | binary | 1000 | 6 | 0.16666666666666666 | 0.16666666666666666 | 0.12329275466771841 | 0.0045 | 63.1s |
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
| T0 | D2+D4 | qaoa_ibm | binary | 1000 | 3 | 0.3333333333333333 | 1.0 | 0.03296351479499938 | 0.011 | 22.7s |
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
| T0 | D4 | qaoa_ibm | binary | 1000 | 15 | 0.06666666666666667 | 0.06666666666666667 | 0.1776057986501159 | 0.0071333333333333335 | 133s |
| T0 | D4 | qaoa_ideal | binary | 1000 | 100 | 0.27 | 0.37 | 0.06995521088580015 | 0.06761 | simulated |
| T0 | D4 | qaoa_ideal | domain_wall | 1000 | 40 | 0.025 | 0.125 | 0.19571369588210724 | 0.0106 | simulated |
| T0 | D4 | qaoa_noisy | binary | 1000 | 10 | 0.0 | 0.0 | 0.27703706242808257 | 0.0199 | simulated |
| T0 | D4 | simulated_annealing | binary | 1000 | 100 | 0.76 | 0.96 | 0.0015235685530318106 | 1.0 | 3.45s |
| T0 | D4 | simulated_annealing | domain_wall | 1000 | 100 | 0.84 | 0.97 | 0.0015235685530318106 | 1.0 | 2.32s |
| T0 | D4 | tabu | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 41.8s |
| T0 | D4 | tabu | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 1.0 | 54.9s |
| T0 | none | greedy | binary | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.97943 | 0.343s |
| T0 | none | greedy | domain_wall | 1000 | 100 | 1.0 | 1.0 | 0.0 | 0.59869 | 0.254s |
| T0 | none | qaoa_ibm | binary | 1000 | 15 | 0.0 | 0.0 | 0.19706210630161325 | 0.0019333333333333333 | censored |
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
| T0 | D1 | 2026 | binary | qaoa_ibm | 3 | 0.0 | 0.0 | 1303.0 | 1789.0 | 9.0 |
| T0 | D1 | 2026 | binary | qaoa_ideal | 20 | 0.05 | 0.06645 |  |  |  |
| T0 | D1 | 2027 | binary | qaoa_ibm | 3 | 0.0 | 0.003 | 1056.0 | 1815.0 | 6.0 |
| T0 | D1 | 2027 | binary | qaoa_ideal | 20 | 0.0 | 0.1051 |  |  |  |
| T0 | D1 | 2028 | binary | qaoa_ibm | 3 | 0.0 | 0.0006666666666666666 | 1092.0 | 1822.0 | 6.0 |
| T0 | D1 | 2028 | binary | qaoa_ideal | 20 | 0.0 | 0.0708 |  |  |  |
| T0 | D1 | 2029 | binary | qaoa_ibm | 3 | 0.0 | 0.0 | 1306.0 | 1865.0 | 9.0 |
| T0 | D1 | 2029 | binary | qaoa_ideal | 20 | 0.1 | 0.0695 |  |  |  |
| T0 | D1 | 2030 | binary | qaoa_ibm | 3 | 0.0 | 0.001 | 1106.0 | 1767.0 | 6.0 |
| T0 | D1 | 2030 | binary | qaoa_ideal | 20 | 0.0 | 0.06655 |  |  |  |
| T0 | D1+D2 | 2026 | binary | qaoa_ibm | 3 | 0.6666666666666666 | 0.006333333333333333 | 852.0 | 758.0 | 6.0 |
| T0 | D1+D2 | 2026 | binary | qaoa_ideal | 20 | 0.7 | 0.0101 |  |  |  |
| T0 | D1+D2 | 2026 | binary | qaoa_noisy | 5 | 0.2 | 0.0036 | 852.0 | 758.0 |  |
| T0 | D1+D2 | 2026 | domain_wall | qaoa_ideal | 20 | 0.7 | 0.0101 |  |  |  |
| T0 | D1+D2 | 2026 | domain_wall | qaoa_noisy | 5 | 0.2 | 0.0036 | 852.0 | 758.0 |  |
| T0 | D1+D2 | 2027 | binary | qaoa_ibm | 3 | 0.0 | 0.0006666666666666666 | 1146.0 | 1262.0 | 9.0 |
| T0 | D1+D2 | 2027 | binary | qaoa_ideal | 20 | 0.7 | 0.0247 |  |  |  |
| T0 | D1+D2 | 2027 | binary | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D1+D2 | 2027 | domain_wall | qaoa_ideal | 20 | 0.75 | 0.024 |  |  |  |
| T0 | D1+D2 | 2027 | domain_wall | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D1+D2 | 2028 | binary | qaoa_ibm | 3 | 0.0 | 0.0003333333333333333 | 1148.0 | 1268.0 | 9.0 |
| T0 | D1+D2 | 2028 | binary | qaoa_ideal | 20 | 0.6 | 0.01105 |  |  |  |
| T0 | D1+D2 | 2028 | binary | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D1+D2 | 2028 | domain_wall | qaoa_ideal | 20 | 0.65 | 0.0109 |  |  |  |
| T0 | D1+D2 | 2028 | domain_wall | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D1+D2 | 2030 | binary | qaoa_ibm | 3 | 0.6666666666666666 | 0.025 | 946.0 | 915.0 | 6.0 |
| T0 | D1+D2 | 2030 | binary | qaoa_ideal | 20 | 0.3 | 0.04535 |  |  |  |
| T0 | D1+D2 | 2030 | binary | qaoa_noisy | 5 | 0.2 | 0.0206 | 954.0 | 915.0 |  |
| T0 | D1+D2 | 2030 | domain_wall | qaoa_ideal | 20 | 0.3 | 0.04535 |  |  |  |
| T0 | D1+D2 | 2030 | domain_wall | qaoa_noisy | 5 | 0.2 | 0.0206 | 954.0 | 915.0 |  |
| T0 | D1+D2+D4 | 2026 | binary | qaoa_ibm | 3 | 0.3333333333333333 | 0.0033333333333333335 | 817.0 | 736.0 | 6.0 |
| T0 | D1+D2+D4 | 2026 | binary | qaoa_ideal | 20 | 1.0 | 0.0981 |  |  |  |
| T0 | D1+D2+D4 | 2026 | binary | qaoa_noisy | 5 | 1.0 | 0.023 | 813.0 | 736.0 |  |
| T0 | D1+D2+D4 | 2026 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.0981 |  |  |  |
| T0 | D1+D2+D4 | 2026 | domain_wall | qaoa_noisy | 5 | 1.0 | 0.023 | 813.0 | 736.0 |  |
| T0 | D1+D2+D4 | 2027 | binary | qaoa_ibm | 3 | 0.0 | 0.018 | 871.0 | 774.0 | 6.0 |
| T0 | D1+D2+D4 | 2027 | binary | qaoa_ideal | 20 | 1.0 | 0.20755 |  |  |  |
| T0 | D1+D2+D4 | 2027 | binary | qaoa_noisy | 5 | 1.0 | 0.0494 | 867.0 | 774.0 |  |
| T0 | D1+D2+D4 | 2027 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.20755 |  |  |  |
| T0 | D1+D2+D4 | 2027 | domain_wall | qaoa_noisy | 5 | 1.0 | 0.0494 | 867.0 | 774.0 |  |
| T0 | D1+D2+D4 | 2028 | binary | qaoa_ibm | 3 | 0.6666666666666666 | 0.012333333333333333 | 811.0 | 736.0 | 6.0 |
| T0 | D1+D2+D4 | 2028 | binary | qaoa_ideal | 20 | 1.0 | 0.11315 |  |  |  |
| T0 | D1+D2+D4 | 2028 | binary | qaoa_noisy | 5 | 0.8 | 0.0266 | 807.0 | 736.0 |  |
| T0 | D1+D2+D4 | 2028 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.11315 |  |  |  |
| T0 | D1+D2+D4 | 2028 | domain_wall | qaoa_noisy | 5 | 0.8 | 0.0266 | 807.0 | 736.0 |  |
| T0 | D1+D4 | 2026 | binary | qaoa_ibm | 3 | 0.0 | 0.0023333333333333335 | 1069.0 | 1400.0 | 6.0 |
| T0 | D1+D4 | 2026 | binary | qaoa_ideal | 20 | 0.4 | 0.08005 |  |  |  |
| T0 | D1+D4 | 2027 | binary | qaoa_ibm | 3 | 0.0 | 0.002 | 777.0 | 873.0 | 6.0 |
| T0 | D1+D4 | 2027 | binary | qaoa_ideal | 20 | 1.0 | 0.056 |  |  |  |
| T0 | D1+D4 | 2027 | binary | qaoa_noisy | 5 | 0.8 | 0.0136 | 783.0 | 873.0 |  |
| T0 | D1+D4 | 2027 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.08525 |  |  |  |
| T0 | D1+D4 | 2028 | binary | qaoa_ibm | 3 | 0.0 | 0.006333333333333333 | 1045.0 | 1471.0 | 6.0 |
| T0 | D1+D4 | 2028 | binary | qaoa_ideal | 20 | 0.0 | 0.0929 |  |  |  |
| T0 | D1+D4 | 2029 | binary | qaoa_ibm | 3 | 0.0 | 0.001 | 1060.0 | 1379.0 | 6.0 |
| T0 | D1+D4 | 2029 | binary | qaoa_ideal | 20 | 0.4 | 0.0822 |  |  |  |
| T0 | D1+D4 | 2030 | binary | qaoa_ibm | 3 | 0.3333333333333333 | 0.006333333333333333 | 697.0 | 1048.0 | 6.0 |
| T0 | D1+D4 | 2030 | binary | qaoa_ideal | 20 | 1.0 | 0.1061 |  |  |  |
| T0 | D1+D4 | 2030 | binary | qaoa_noisy | 5 | 0.4 | 0.0188 | 700.0 | 1048.0 |  |
| T0 | D1+D4 | 2030 | domain_wall | qaoa_ideal | 20 | 0.8 | 0.03055 |  |  |  |
| T0 | D2 | 2026 | binary | qaoa_ibm | 3 | 0.3333333333333333 | 0.008 | 814.0 | 736.0 | 6.0 |
| T0 | D2 | 2026 | binary | qaoa_ideal | 20 | 1.0 | 0.09955 |  |  |  |
| T0 | D2 | 2026 | binary | qaoa_noisy | 5 | 1.0 | 0.0234 | 813.0 | 736.0 |  |
| T0 | D2 | 2026 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.09955 |  |  |  |
| T0 | D2 | 2026 | domain_wall | qaoa_noisy | 5 | 1.0 | 0.0234 | 813.0 | 736.0 |  |
| T0 | D2 | 2030 | binary | qaoa_ibm | 3 | 0.0 | 0.001 | 1128.0 | 1168.0 | 9.0 |
| T0 | D2 | 2030 | binary | qaoa_ideal | 20 | 0.45 | 0.0169 |  |  |  |
| T0 | D2 | 2030 | binary | qaoa_noisy | 5 | 0.2 | 0.002 | 1137.0 | 1166.0 |  |
| T0 | D2 | 2030 | domain_wall | qaoa_ideal | 20 | 0.45 | 0.01635 |  |  |  |
| T0 | D2 | 2030 | domain_wall | qaoa_noisy | 5 | 0.2 | 0.0026 | 1171.0 | 1227.0 |  |
| T0 | D2+D4 | 2029 | binary | qaoa_ibm | 3 | 0.3333333333333333 | 0.011 | 765.0 | 735.0 | 6.0 |
| T0 | D2+D4 | 2029 | binary | qaoa_ideal | 20 | 1.0 | 0.0609 |  |  |  |
| T0 | D2+D4 | 2029 | binary | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D2+D4 | 2029 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.0609 |  |  |  |
| T0 | D2+D4 | 2029 | domain_wall | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D2+D4 | 2030 | binary | qaoa_ideal | 20 | 0.15 | 0.0505 |  |  |  |
| T0 | D2+D4 | 2030 | domain_wall | qaoa_ideal | 20 | 0.25 | 0.04985 |  |  |  |
| T0 | D4 | 2026 | binary | qaoa_ibm | 3 | 0.0 | 0.0003333333333333333 | 1100.0 | 1391.0 | 6.0 |
| T0 | D4 | 2026 | binary | qaoa_ideal | 20 | 1.0 | 0.04515 |  |  |  |
| T0 | D4 | 2027 | binary | qaoa_ibm | 3 | 0.3333333333333333 | 0.018666666666666668 | 969.0 | 1233.0 | 6.0 |
| T0 | D4 | 2027 | binary | qaoa_ideal | 20 | 0.15 | 0.08655 |  |  |  |
| T0 | D4 | 2027 | binary | qaoa_noisy | 5 | 0.0 | 0.0278 | 978.0 | 1233.0 |  |
| T0 | D4 | 2027 | domain_wall | qaoa_ideal | 20 | 0.0 | 0.006 |  |  |  |
| T0 | D4 | 2028 | binary | qaoa_ibm | 3 | 0.0 | 0.0026666666666666666 | 942.0 | 1507.0 | 6.0 |
| T0 | D4 | 2028 | binary | qaoa_ideal | 20 | 0.2 | 0.0836 |  |  |  |
| T0 | D4 | 2029 | binary | qaoa_ibm | 3 | 0.0 | 0.0036666666666666666 | 1086.0 | 1512.0 | 6.0 |
| T0 | D4 | 2029 | binary | qaoa_ideal | 20 | 0.0 | 0.09285 |  |  |  |
| T0 | D4 | 2030 | binary | qaoa_ibm | 3 | 0.0 | 0.010333333333333333 | 665.0 | 962.0 | 6.0 |
| T0 | D4 | 2030 | binary | qaoa_ideal | 20 | 0.0 | 0.0299 |  |  |  |
| T0 | D4 | 2030 | binary | qaoa_noisy | 5 | 0.0 | 0.012 | 663.0 | 962.0 |  |
| T0 | D4 | 2030 | domain_wall | qaoa_ideal | 20 | 0.05 | 0.0152 |  |  |  |
| T0 | none | 2026 | binary | qaoa_ibm | 3 | 0.0 | 0.001 | 1173.0 | 1930.0 | 6.0 |
| T0 | none | 2026 | binary | qaoa_ideal | 20 | 0.0 | 0.06935 |  |  |  |
| T0 | none | 2027 | binary | qaoa_ibm | 3 | 0.0 | 0.0006666666666666666 | 1037.0 | 1219.0 | 6.0 |
| T0 | none | 2027 | binary | qaoa_ideal | 20 | 0.5 | 0.07175 |  |  |  |
| T0 | none | 2027 | binary | qaoa_noisy | 5 | 0.2 | 0.0062 | 1050.0 | 1219.0 |  |
| T0 | none | 2028 | binary | qaoa_ibm | 3 | 0.0 | 0.006666666666666667 | 867.0 | 1261.0 | 6.0 |
| T0 | none | 2028 | binary | qaoa_ideal | 20 | 0.15 | 0.1474 |  |  |  |
| T0 | none | 2028 | binary | qaoa_noisy | 5 | 0.0 | 0.0222 | 869.0 | 1261.0 |  |
| T0 | none | 2028 | domain_wall | qaoa_ideal | 20 | 0.0 | 0.0138 |  |  |  |
| T0 | none | 2029 | binary | qaoa_ibm | 3 | 0.0 | 0.0 | 1563.0 | 2044.0 | 9.0 |
| T0 | none | 2029 | binary | qaoa_ideal | 20 | 0.0 | 0.07175 |  |  |  |
| T0 | none | 2030 | binary | qaoa_ibm | 3 | 0.0 | 0.0013333333333333333 | 1084.0 | 1232.0 | 6.0 |
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
<!-- /tables -->

<!-- tables-p1 -->
## Appendix: the report's tables, reps = 1

Regenerated by `quantic bench report <store> --out <dir>` from the reps = 1 store.

### Matched reads (`matched_reads.md`)

Every solver at 1,000 reads; `tts` is QPU seconds for `qaoa_ibm`, wall-clock seconds for the samplers, and "simulated" for the simulators.

| tier | dials | solver | encoding | max_samples | n_runs | p_success_0.01 | p_success_0.05 | gap_median | feasibility | tts |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | D1 | qaoa_ibm | binary | 1000 | 5 | 0.0 | 0.0 | 0.22535589334434997 | 0.0016 | censored |
| T0 | D1 | qaoa_ideal | binary | 1000 | 100 | 0.0 | 0.22 | 0.08315828867146792 | 0.02472 | simulated |
| T0 | D1+D2 | qaoa_ibm | binary | 1000 | 4 | 0.0 | 0.0 | 0.15964757426886844 | 0.01025 | censored |
| T0 | D1+D2 | qaoa_ideal | binary | 1000 | 80 | 0.5125 | 0.5625 | 0.0013062842025486935 | 0.024175 | simulated |
| T0 | D1+D2 | qaoa_ideal | domain_wall | 1000 | 80 | 0.5125 | 0.575 | 0.0013062842025486935 | 0.0242125 | simulated |
| T0 | D1+D2 | qaoa_noisy | binary | 1000 | 20 | 0.45 | 0.5 | 3.4737553137705954e-08 | 0.009533333333333333 | simulated |
| T0 | D1+D2+D4 | qaoa_ibm | binary | 1000 | 3 | 0.6666666666666666 | 0.6666666666666666 | 0.0 | 0.024333333333333332 | 8.38s |
| T0 | D1+D2+D4 | qaoa_ideal | binary | 1000 | 60 | 0.9666666666666667 | 0.9666666666666667 | 0.0 | 0.0573 | simulated |
| T0 | D1+D2+D4 | qaoa_ideal | domain_wall | 1000 | 60 | 0.9666666666666667 | 0.9666666666666667 | 0.0 | 0.0573 | simulated |
| T0 | D1+D2+D4 | qaoa_noisy | binary | 1000 | 15 | 0.7333333333333333 | 0.7333333333333333 | 0.0 | 0.032466666666666665 | simulated |
| T0 | D1+D4 | qaoa_ibm | binary | 1000 | 5 | 0.2 | 0.2 | 0.08111115812225829 | 0.0038 | 41.3s |
| T0 | D1+D4 | qaoa_ideal | binary | 1000 | 100 | 0.41 | 0.9 | 0.012656013207228504 | 0.03221 | simulated |
| T0 | D1+D4 | qaoa_ideal | domain_wall | 1000 | 40 | 0.475 | 0.925 | 0.01121687547012913 | 0.01225 | simulated |
| T0 | D1+D4 | qaoa_noisy | binary | 1000 | 10 | 0.6 | 0.9 | 0.007693352625174092 | 0.0155 | simulated |
| T0 | D2 | qaoa_ibm | binary | 1000 | 2 | 0.5 | 0.5 | 0.0 | 0.0065 | 13.3s |
| T0 | D2 | qaoa_ideal | binary | 1000 | 40 | 0.525 | 0.525 | 0.0 | 0.02375 | simulated |
| T0 | D2 | qaoa_ideal | domain_wall | 1000 | 40 | 0.6 | 0.6 | 0.0 | 0.024 | simulated |
| T0 | D2 | qaoa_noisy | binary | 1000 | 10 | 0.6 | 0.6 | 0.0 | 0.0134 | simulated |
| T0 | D2+D4 | qaoa_ibm | binary | 1000 | 2 | 0.5 | 0.5 | 0.2218686506519677 | 0.009 | 13.3s |
| T0 | D2+D4 | qaoa_ideal | binary | 1000 | 40 | 0.4 | 0.625 | 0.03296351479499938 | 0.0161 | simulated |
| T0 | D2+D4 | qaoa_ideal | domain_wall | 1000 | 40 | 0.4 | 0.575 | 0.03296351479499938 | 0.016375 | simulated |
| T0 | D2+D4 | qaoa_noisy | binary | 1000 | 5 | 0.6 | 0.6 | 0.0 | 0.0166 | simulated |
| T0 | D4 | qaoa_ibm | binary | 1000 | 5 | 0.0 | 0.0 | 0.17527632751163638 | 0.0104 | censored |
| T0 | D4 | qaoa_ideal | binary | 1000 | 100 | 0.24 | 0.37 | 0.07165515648019757 | 0.05717 | simulated |
| T0 | D4 | qaoa_ideal | domain_wall | 1000 | 40 | 0.025 | 0.1 | 0.2239275810321238 | 0.0045 | simulated |
| T0 | D4 | qaoa_noisy | binary | 1000 | 10 | 0.0 | 0.1 | 0.1108062129225236 | 0.0349 | simulated |
| T0 | none | qaoa_ibm | binary | 1000 | 5 | 0.0 | 0.0 | 0.36906518566825547 | 0.0036 | censored |
| T0 | none | qaoa_ideal | binary | 1000 | 100 | 0.14 | 0.47 | 0.056939738226353405 | 0.02916 | simulated |
| T0 | none | qaoa_ideal | domain_wall | 1000 | 20 | 0.0 | 0.0 | 0.24566546583236093 | 0.00405 | simulated |
| T0 | none | qaoa_noisy | binary | 1000 | 15 | 0.06666666666666667 | 0.5333333333333333 | 0.04128222397794128 | 0.016533333333333334 | simulated |

### The noise ladder per instance (`qaoa_noise.md`)

Routed depth and two-qubit count are medians over the runs; `qpu_s` sums the hardware jobs.

| tier | dials | instance_seed | encoding | solver | n_runs | p_success_0.01 | feasibility | routed_depth | routed_2q | qpu_s |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | D1 | 2026 | binary | qaoa_ibm | 1 | 0.0 | 0.001 | 677.0 | 903.0 | 2.0 |
| T0 | D1 | 2026 | binary | qaoa_ideal | 20 | 0.0 | 0.01755 |  |  |  |
| T0 | D1 | 2027 | binary | qaoa_ibm | 1 | 0.0 | 0.004 | 849.0 | 952.0 | 2.0 |
| T0 | D1 | 2027 | binary | qaoa_ideal | 20 | 0.0 | 0.0509 |  |  |  |
| T0 | D1 | 2028 | binary | qaoa_ibm | 1 | 0.0 | 0.002 | 683.0 | 911.0 | 2.0 |
| T0 | D1 | 2028 | binary | qaoa_ideal | 20 | 0.0 | 0.01825 |  |  |  |
| T0 | D1 | 2029 | binary | qaoa_ibm | 1 | 0.0 | 0.0 | 682.0 | 909.0 | 2.0 |
| T0 | D1 | 2029 | binary | qaoa_ideal | 20 | 0.0 | 0.0193 |  |  |  |
| T0 | D1 | 2030 | binary | qaoa_ibm | 1 | 0.0 | 0.001 | 862.0 | 946.0 | 2.0 |
| T0 | D1 | 2030 | binary | qaoa_ideal | 20 | 0.0 | 0.0176 |  |  |  |
| T0 | D1+D2 | 2026 | binary | qaoa_ibm | 1 | 0.0 | 0.024 | 371.0 | 313.0 | 2.0 |
| T0 | D1+D2 | 2026 | binary | qaoa_ideal | 20 | 1.0 | 0.0491 |  |  |  |
| T0 | D1+D2 | 2026 | binary | qaoa_noisy | 5 | 1.0 | 0.023 | 371.0 | 313.0 |  |
| T0 | D1+D2 | 2026 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.0491 |  |  |  |
| T0 | D1+D2 | 2027 | binary | qaoa_ibm | 1 | 0.0 | 0.004 | 588.0 | 574.0 | 2.0 |
| T0 | D1+D2 | 2027 | binary | qaoa_ideal | 20 | 0.15 | 0.0079 |  |  |  |
| T0 | D1+D2 | 2027 | binary | qaoa_noisy | 5 | 0.4 | 0.0034 | 593.0 | 574.0 |  |
| T0 | D1+D2 | 2027 | domain_wall | qaoa_ideal | 20 | 0.25 | 0.0084 |  |  |  |
| T0 | D1+D2 | 2028 | binary | qaoa_ibm | 1 | 0.0 | 0.0 | 584.0 | 574.0 | 2.0 |
| T0 | D1+D2 | 2028 | binary | qaoa_ideal | 20 | 0.35 | 0.0042 |  |  |  |
| T0 | D1+D2 | 2028 | binary | qaoa_noisy | 5 | 0.4 | 0.0022 | 587.0 | 574.0 |  |
| T0 | D1+D2 | 2028 | domain_wall | qaoa_ideal | 20 | 0.25 | 0.00385 |  |  |  |
| T0 | D1+D2 | 2030 | binary | qaoa_ibm | 1 | 0.0 | 0.013 | 469.0 | 405.0 | 2.0 |
| T0 | D1+D2 | 2030 | binary | qaoa_ideal | 20 | 0.55 | 0.0355 |  |  |  |
| T0 | D1+D2 | 2030 | binary | qaoa_noisy | 5 | 0.0 | nan |  |  |  |
| T0 | D1+D2 | 2030 | domain_wall | qaoa_ideal | 20 | 0.55 | 0.0355 |  |  |  |
| T0 | D1+D2+D4 | 2026 | binary | qaoa_ibm | 1 | 1.0 | 0.017 | 359.0 | 313.0 | 2.0 |
| T0 | D1+D2+D4 | 2026 | binary | qaoa_ideal | 20 | 0.95 | 0.043 |  |  |  |
| T0 | D1+D2+D4 | 2026 | binary | qaoa_noisy | 5 | 0.4 | 0.0234 | 359.0 | 313.0 |  |
| T0 | D1+D2+D4 | 2026 | domain_wall | qaoa_ideal | 20 | 0.95 | 0.043 |  |  |  |
| T0 | D1+D2+D4 | 2027 | binary | qaoa_ibm | 1 | 1.0 | 0.041 | 399.0 | 325.0 | 2.0 |
| T0 | D1+D2+D4 | 2027 | binary | qaoa_ideal | 20 | 0.95 | 0.077 |  |  |  |
| T0 | D1+D2+D4 | 2027 | binary | qaoa_noisy | 5 | 0.8 | 0.0488 | 402.0 | 325.0 |  |
| T0 | D1+D2+D4 | 2027 | domain_wall | qaoa_ideal | 20 | 0.95 | 0.077 |  |  |  |
| T0 | D1+D2+D4 | 2028 | binary | qaoa_ibm | 1 | 0.0 | 0.015 | 364.0 | 313.0 | 2.0 |
| T0 | D1+D2+D4 | 2028 | binary | qaoa_ideal | 20 | 1.0 | 0.0519 |  |  |  |
| T0 | D1+D2+D4 | 2028 | binary | qaoa_noisy | 5 | 1.0 | 0.0252 | 363.0 | 313.0 |  |
| T0 | D1+D2+D4 | 2028 | domain_wall | qaoa_ideal | 20 | 1.0 | 0.0519 |  |  |  |
| T0 | D1+D4 | 2026 | binary | qaoa_ibm | 1 | 0.0 | 0.004 | 623.0 | 675.0 | 2.0 |
| T0 | D1+D4 | 2026 | binary | qaoa_ideal | 20 | 0.25 | 0.03605 |  |  |  |
| T0 | D1+D4 | 2027 | binary | qaoa_ibm | 1 | 1.0 | 0.005 | 405.0 | 378.0 | 2.0 |
| T0 | D1+D4 | 2027 | binary | qaoa_ideal | 20 | 0.95 | 0.0304 |  |  |  |
| T0 | D1+D4 | 2027 | binary | qaoa_noisy | 5 | 1.0 | 0.0174 | 407.0 | 384.0 |  |
| T0 | D1+D4 | 2027 | domain_wall | qaoa_ideal | 20 | 0.75 | 0.01615 |  |  |  |
| T0 | D1+D4 | 2028 | binary | qaoa_ibm | 1 | 0.0 | 0.006 | 745.0 | 864.0 | 2.0 |
| T0 | D1+D4 | 2028 | binary | qaoa_ideal | 20 | 0.05 | 0.04435 |  |  |  |
| T0 | D1+D4 | 2029 | binary | qaoa_ibm | 1 | 0.0 | 0.003 | 559.0 | 695.0 | 2.0 |
| T0 | D1+D4 | 2029 | binary | qaoa_ideal | 20 | 0.25 | 0.023 |  |  |  |
| T0 | D1+D4 | 2030 | binary | qaoa_ibm | 1 | 0.0 | 0.001 | 581.0 | 559.0 | 2.0 |
| T0 | D1+D4 | 2030 | binary | qaoa_ideal | 20 | 0.55 | 0.02725 |  |  |  |
| T0 | D1+D4 | 2030 | binary | qaoa_noisy | 5 | 0.2 | 0.0136 | 580.0 | 559.0 |  |
| T0 | D1+D4 | 2030 | domain_wall | qaoa_ideal | 20 | 0.2 | 0.00835 |  |  |  |
| T0 | D2 | 2026 | binary | qaoa_ibm | 1 | 1.0 | 0.013 | 359.0 | 313.0 | 2.0 |
| T0 | D2 | 2026 | binary | qaoa_ideal | 20 | 0.9 | 0.04235 |  |  |  |
| T0 | D2 | 2026 | binary | qaoa_noisy | 5 | 1.0 | 0.0236 | 363.0 | 315.0 |  |
| T0 | D2 | 2026 | domain_wall | qaoa_ideal | 20 | 0.9 | 0.04235 |  |  |  |
| T0 | D2 | 2030 | binary | qaoa_ibm | 1 | 0.0 | 0.0 | 507.0 | 508.0 | 2.0 |
| T0 | D2 | 2030 | binary | qaoa_ideal | 20 | 0.15 | 0.00515 |  |  |  |
| T0 | D2 | 2030 | binary | qaoa_noisy | 5 | 0.2 | 0.0032 | 517.0 | 514.0 |  |
| T0 | D2 | 2030 | domain_wall | qaoa_ideal | 20 | 0.3 | 0.00565 |  |  |  |
| T0 | D2+D4 | 2029 | binary | qaoa_ibm | 1 | 1.0 | 0.017 | 360.0 | 313.0 | 2.0 |
| T0 | D2+D4 | 2029 | binary | qaoa_ideal | 20 | 0.8 | 0.0274 |  |  |  |
| T0 | D2+D4 | 2029 | binary | qaoa_noisy | 5 | 0.6 | 0.0166 | 360.0 | 313.0 |  |
| T0 | D2+D4 | 2029 | domain_wall | qaoa_ideal | 20 | 0.8 | 0.0274 |  |  |  |
| T0 | D2+D4 | 2030 | binary | qaoa_ibm | 1 | 0.0 | 0.001 | 714.0 | 635.0 | 2.0 |
| T0 | D2+D4 | 2030 | binary | qaoa_ideal | 20 | 0.0 | 0.0048 |  |  |  |
| T0 | D2+D4 | 2030 | domain_wall | qaoa_ideal | 20 | 0.0 | 0.00535 |  |  |  |
| T0 | D4 | 2026 | binary | qaoa_ibm | 1 | 0.0 | 0.0 | 680.0 | 711.0 | 2.0 |
| T0 | D4 | 2026 | binary | qaoa_ideal | 20 | 0.95 | 0.01855 |  |  |  |
| T0 | D4 | 2027 | binary | qaoa_ibm | 1 | 0.0 | 0.018 | 475.0 | 546.0 | 2.0 |
| T0 | D4 | 2027 | binary | qaoa_ideal | 20 | 0.0 | 0.05495 |  |  |  |
| T0 | D4 | 2027 | binary | qaoa_noisy | 5 | 0.0 | 0.0294 | 472.0 | 546.0 |  |
| T0 | D4 | 2027 | domain_wall | qaoa_ideal | 20 | 0.05 | 0.00485 |  |  |  |
| T0 | D4 | 2028 | binary | qaoa_ibm | 1 | 0.0 | 0.003 | 598.0 | 739.0 | 2.0 |
| T0 | D4 | 2028 | binary | qaoa_ideal | 20 | 0.1 | 0.0354 |  |  |  |
| T0 | D4 | 2029 | binary | qaoa_ibm | 1 | 0.0 | 0.006 | 740.0 | 860.0 | 2.0 |
| T0 | D4 | 2029 | binary | qaoa_ideal | 20 | 0.0 | 0.0562 |  |  |  |
| T0 | D4 | 2030 | binary | qaoa_ibm | 1 | 0.0 | 0.025 | 458.0 | 487.0 | 2.0 |
| T0 | D4 | 2030 | binary | qaoa_ideal | 20 | 0.15 | 0.12075 |  |  |  |
| T0 | D4 | 2030 | binary | qaoa_noisy | 5 | 0.0 | 0.0404 | 457.0 | 487.0 |  |
| T0 | D4 | 2030 | domain_wall | qaoa_ideal | 20 | 0.0 | 0.00415 |  |  |  |
| T0 | none | 2026 | binary | qaoa_ibm | 1 | 0.0 | 0.0 | 864.0 | 946.0 | 2.0 |
| T0 | none | 2026 | binary | qaoa_ideal | 20 | 0.05 | 0.0177 |  |  |  |
| T0 | none | 2027 | binary | qaoa_ibm | 1 | 0.0 | 0.001 | 553.0 | 621.0 | 2.0 |
| T0 | none | 2027 | binary | qaoa_ideal | 20 | 0.15 | 0.016 |  |  |  |
| T0 | none | 2027 | binary | qaoa_noisy | 5 | 0.0 | 0.007 | 563.0 | 621.0 |  |
| T0 | none | 2028 | binary | qaoa_ibm | 1 | 0.0 | 0.016 | 737.0 | 681.0 | 2.0 |
| T0 | none | 2028 | binary | qaoa_ideal | 20 | 0.2 | 0.06385 |  |  |  |
| T0 | none | 2028 | binary | qaoa_noisy | 5 | 0.0 | 0.0346 | 740.0 | 681.0 |  |
| T0 | none | 2028 | domain_wall | qaoa_ideal | 20 | 0.0 | 0.00405 |  |  |  |
| T0 | none | 2029 | binary | qaoa_ibm | 1 | 0.0 | 0.001 | 861.0 | 946.0 | 2.0 |
| T0 | none | 2029 | binary | qaoa_ideal | 20 | 0.0 | 0.0279 |  |  |  |
| T0 | none | 2030 | binary | qaoa_ibm | 1 | 0.0 | 0.0 | 581.0 | 534.0 | 2.0 |
| T0 | none | 2030 | binary | qaoa_ideal | 20 | 0.3 | 0.02035 |  |  |  |
| T0 | none | 2030 | binary | qaoa_noisy | 5 | 0.2 | 0.008 | 577.0 | 534.0 |  |

### Training (`qaoa_training.md`)

| tier | dials | instance_seed | encoding | n_vars | dynamic_range | ising_dynamic_range | train_s | cvar | n_evals |
|---|---|---|---|---|---|---|---|---|---|
| T0 | D2+D4 | 2029 | binary | 16 | 267622.30004372145 | 381076.852458577 | 0.7718648000154644 | 1985.494260326062 | 86 |
| T0 | none | 2028 | domain_wall | 24 | 439393.50796450884 | 305196.3961698304 | 1038.1376214000047 | 1046.5727664599988 | 109 |
| T0 | D1+D2+D4 | 2028 | domain_wall | 16 | 147059.70760238502 | 209846.12670868673 | 0.47386540001025423 | 2334.6894341333664 | 48 |
| T0 | D4 | 2028 | binary | 22 | 1592191.5631031005 | 1417342.726187961 | 205.51838890003273 | 543.542342326934 | 93 |
| T0 | none | 2030 | binary | 20 | 3847243.570422343 | 3851380.5088098226 | 44.28248719999101 | 531.929395069256 | 72 |
| T0 | D1+D4 | 2030 | binary | 19 | 3123334.8234805446 | 3127437.828172062 | 20.606674699985888 | 645.0378640018575 | 81 |
| T0 | D4 | 2030 | domain_wall | 24 | 177115.24514701654 | 126159.60561394728 | 723.5561116999597 | 845.4965790228351 | 67 |
| T0 | D1+D4 | 2028 | binary | 22 | 1053836.0490469376 | 604064.3677823368 | 193.16798249998828 | 501.3232728287066 | 88 |
| T0 | D1 | 2030 | binary | 24 | 1680347.2915862605 | 1680410.2600117684 | 997.0079573999974 | 1076.9706122821112 | 94 |
| T0 | D2 | 2030 | domain_wall | 20 | 842077.7856412798 | 988810.772013221 | 44.803603400010616 | 3311.7041094055653 | 74 |
| T0 | D1+D2 | 2030 | domain_wall | 20 | 370735.0187875911 | 222939.59640466218 | 16.587249499978498 | 3614.4480438777455 | 31 |
| T0 | D1+D2+D4 | 2026 | domain_wall | 16 | 150051.34276485513 | 213667.51291527192 | 0.6788144999882206 | 3380.7191582056616 | 63 |
| T0 | D1+D4 | 2026 | binary | 21 | 2200583.8228090606 | 1928974.798081254 | 162.15508280001814 | 1018.173245454537 | 128 |
| T0 | D1+D2+D4 | 2026 | binary | 16 | 150051.34276485513 | 213667.51291527192 | 0.6687180000008084 | 3380.7191582056616 | 63 |
| T0 | D1 | 2029 | binary | 24 | 2652154.140299909 | 2651742.8486133995 | 912.5982104999712 | 1308.953318538786 | 96 |
| T0 | D2+D4 | 2030 | domain_wall | 24 | 306965.8451784672 | 184067.0607447073 | 605.4944026999874 | 1445.5591772925313 | 39 |
| T0 | D1+D4 | 2027 | binary | 18 | 4540095.092686086 | 4956665.292469511 | 5.378843400045298 | 566.7895204189404 | 45 |
| T0 | D1+D2 | 2027 | domain_wall | 20 | 111480.19666454612 | 131295.74656935147 | 33.49930210004095 | 2982.157240075641 | 64 |
| T0 | D1+D2 | 2028 | domain_wall | 20 | 46122.1444294258 | 102092.60822759196 | 38.989999099983834 | 1487.5527081517203 | 75 |
| T0 | D4 | 2026 | binary | 21 | 4338327.144770745 | 4957998.170000287 | 86.95916169998236 | 743.3278514885242 | 43 |
| T0 | D1 | 2027 | binary | 24 | 1155514.452811122 | 661335.4871877294 | 1024.2622193999705 | 623.1752451664244 | 106 |
| T0 | D4 | 2030 | binary | 18 | 277577.722909354 | 184599.58710221457 | 5.877642499981448 | 217.67400671463074 | 50 |
| T0 | D2 | 2026 | domain_wall | 16 | 131297.85161264712 | 188035.44698694485 | 15.334619499975815 | 957.3234959869116 | 70 |
| T0 | D4 | 2027 | domain_wall | 23 | 171989.91258671423 | 220617.6749040006 | 402.35990809998475 | 909.0773140806971 | 86 |
| T0 | D1 | 2028 | binary | 24 | 1928805.5868174913 | 1930087.064344834 | 728.9706603999948 | 969.5140749139425 | 77 |
| T0 | none | 2027 | binary | 20 | 2858732.848559634 | 3118430.7722666548 | 34.86619899998186 | 1097.169407794703 | 58 |
| T0 | D4 | 2027 | binary | 19 | 268734.73895857023 | 357515.4611453831 | 57.11890340002719 | 314.76699086180076 | 200 |
| T0 | D1+D2+D4 | 2027 | domain_wall | 16 | 9986.09276216544 | 25005.841946902932 | 1.1297454999876209 | 310.0142627277446 | 115 |
| T0 | D4 | 2029 | binary | 22 | 869907.6219097628 | 579612.9528968993 | 279.1708850999712 | 454.50123407012745 | 107 |
| T0 | D1+D2+D4 | 2028 | binary | 16 | 147059.70760238502 | 209846.12670868673 | 0.4328730999841355 | 2334.6894341333664 | 48 |
| T0 | D1+D4 | 2027 | domain_wall | 24 | 2445860.0374534237 | 2660340.668263771 | 494.56294019997586 | 700.4289573348219 | 52 |
| T0 | D1+D2+D4 | 2027 | binary | 16 | 9986.09276216544 | 25005.841946902932 | 1.083164100011345 | 310.0142627277446 | 115 |
| T0 | none | 2026 | binary | 24 | 1476218.4710141397 | 1476036.1387551464 | 968.6610013000318 | 967.0894166146109 | 58 |
| T0 | D1 | 2026 | binary | 24 | 2281246.5692745014 | 2280923.8863076526 | 1184.0245470000082 | 2060.1477498130675 | 72 |
| T0 | D2 | 2030 | binary | 20 | 842270.1358897206 | 989037.0664231512 | 60.543921900040004 | 3312.437992018579 | 100 |
| T0 | D1+D4 | 2029 | binary | 21 | 3478300.797060273 | 3482215.2581682666 | 88.85518549999688 | 1054.4403566642577 | 76 |
| T0 | D1+D2 | 2027 | binary | 20 | 112260.87061133442 | 132214.18650674945 | 32.20524179999484 | 3002.3510822483795 | 61 |
| T0 | D1+D2 | 2026 | domain_wall | 16 | 48650.67434373239 | 34876.03798732215 | 0.8281248999992386 | 914.9812728513224 | 75 |
| T0 | D2+D4 | 2030 | binary | 24 | 307257.07928725815 | 184241.80120998184 | 713.5457926000236 | 1446.8904032895337 | 42 |
| T0 | D1+D2 | 2026 | binary | 16 | 48650.67434373239 | 34876.03798732215 | 0.8384931000182405 | 914.9812728513224 | 75 |
| T0 | none | 2029 | binary | 24 | 1696165.8983049456 | 1510376.3538750745 | 1430.840192899981 | 643.6456532814194 | 146 |
| T0 | D2 | 2026 | binary | 16 | 131297.85161264712 | 188035.44698694485 | 13.430496499990113 | 957.3234959869116 | 70 |
| T0 | D2+D4 | 2029 | domain_wall | 16 | 267622.30004372145 | 381076.852458577 | 0.7978227999992669 | 1985.494260326062 | 86 |
| T0 | D1+D2 | 2030 | binary | 20 | 370735.0187875911 | 222939.59640466218 | 17.469541599974036 | 3614.4480438777455 | 31 |
| T0 | D1+D4 | 2030 | domain_wall | 22 | 3250721.6530754687 | 3251713.450842411 | 265.6480501999613 | 912.8386543395164 | 91 |
| T0 | none | 2028 | binary | 20 | 759064.6218446441 | 255179.95173478458 | 25.846344299963675 | 320.6235197756889 | 50 |
| T0 | D1+D2 | 2028 | binary | 20 | 46814.696485712346 | 103631.61279711765 | 42.65650929999538 | 1508.9949062374408 | 82 |

### HiGHS at the 1 s budget (`crossover_1s.md`)

median gap / feasibility / TTS99 at 1%.

| tier | dials | exact_pwl_milp |
|---|---|---|
| T0 | D1 | 0.00% / 100% / 0.157s |
| T0 | D1+D2 | 0.00% / 100% / 0.234s |
| T0 | D1+D2+D4 | 0.00% / 100% / 0.166s |
| T0 | D1+D4 | 0.00% / 100% / 0.171s |
| T0 | D2 | 0.00% / 100% / 0.213s |
| T0 | D2+D4 | 0.00% / 100% / 0.217s |
| T0 | D4 | 0.00% / 100% / 0.265s |
| T0 | none | 0.00% / 100% / 0.336s |
<!-- /tables-p1 -->
