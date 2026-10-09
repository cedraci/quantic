# M5 — QAOA on IBM Quantum: design

**Written:** 2026-10-09
**Branch:** `feat/m5-qaoa-ibm`
**Parent spec:** `2026-09-22-quantum-liquidation-lvar-design.md` (sections 3.1, 3.2, 7.4, 7.5, 14)
**Builds on:** `2026-10-08-m4-crossover-result.md`

---

## 1. Purpose

Put the first quantum solver into the M4 bench harness and measure it on the instances where a gate-model quantum computer can run at all. Then obtain a first result from real IBM hardware without exceeding a hard quota.

The question M5 answers: **on the T0 liquidation QUBOs that fit in a gate-model circuit, how does QAOA compare with the classical QUBO samplers and the exact methods?** It is measured noiseless, under a simulated device noise model, and on a real IBM device.

M4 already fixed the bar. HiGHS proves every T0 optimum in about 0.1 s, and the samplers win time-to-solution only on the cells without D2 or D3, at the 0.1 s budget. M5 measures where QAOA sits against that bar. It does not assume a quantum advantage.

## 2. Scope

### In scope

- **Instances:** T0 cells without CVaR: the eight dial combinations `none`, `D1`, `D2`, `D4`, `D1+D2`, `D1+D4`, `D2+D4`, `D1+D2+D4`, on seeds 2026–2030. These are the M4 study's instances, built by `bench.instances`.
- **Encodings:** binary (16–28 qubits) and domain-wall (16–36 qubits). A QUBO above the qubit cap (`max_qubits`, default 28) is not planned. It is left out of the task list, not recorded as an error.
- **One algorithm:** CVaR-QAOA (parent spec section 7.4).
- **Three backends for the same trained circuit:** `qaoa_ideal`, `qaoa_noisy` and `qaoa_ibm` (section 6).
- **The parent spec's QPU budget guard (section 7.5)** recast for IBM: an angle cache, a simulator gate and a hard-failing quota ledger.
- **Sampler runs at matched read counts,** for a like-for-like quality comparison.

### Out of scope

- **D-Wave** (`dwave_qpu`, `dwave_hybrid_cqm`, `dwave_hybrid_bqm`): there is no Leap account. Deferred until one exists, with its own spec.
- **CVaR instances (260–296 variables at T0) and T1 and above (88 or more variables):** too large for a gate-model circuit.
- **The variational loop on hardware:** angles are trained only on the noiseless simulator (decision D3).
- **The one-hot encoding:** M4 found it the least feasible encoding, with the largest dynamic range.
- **Error mitigation** beyond what the IBM Sampler applies by default.

## 3. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | IBM Quantum is the only hardware backend. D-Wave is deferred. | It is the only account the owner has (2026-10-09). The parent spec's M5 "done" criterion, a first Advantage2 result, is replaced by a first IBM result. |
| D2 | QAOA on T0 cells without CVaR, binary and domain-wall encodings | These are the only QUBOs within about 30 qubits. They are also M4's crossover cells. |
| D3 | Train angles on the noiseless simulator; sample once on hardware | The full loop on hardware would cost 50–200 jobs per instance, so 5 minutes would cover one or two instances. Sharing the angles isolates what noise costs. |
| D4 | Quota cap: 300 s of QPU time per rolling 28 days | Half the Open plan's 10 minutes, leaving margin for retries and the owner's own use. |
| D5 | The simulator gate is a short noisy run (32 shots) of the exact routed circuit. Full noisy statistics only at 20 qubits or fewer. | Measured on fake Fez: noisy simulation costs 1.4 s per shot at 20 qubits and grows about six-fold per 4 qubits. |
| D6 | Time-to-solution for `qaoa_ibm` uses IBM's reported QPU usage seconds. `qaoa_ideal` and `qaoa_noisy` report no TTS. | Simulation time on a laptop is not quantum time. The comparison with samplers is quality at equal read counts. |
| D7 | A stored `qaoa_ibm` result is never repeated by a later commit | M4's resume key includes the commit, and re-running would spend quota again. |

## 4. Components

| Unit | Path | Does | Depends on |
|---|---|---|---|
| Ising map | `solvers/quantum/ising.py` | QUBO → `SparsePauliOp` plus constant, scaled so the largest coefficient is 1. Returns the scale, the constant and the dynamic range. | `encoding.qubo` |
| Circuit | `solvers/quantum/circuit.py` | p-layer QAOA ansatz for an Ising operator, with the linear-ramp initial angles | qiskit |
| Trainer | `solvers/quantum/train.py` | CVaR-QAOA on the noiseless statevector: an exact CVaR from the probability vector, COBYLA, deterministic. Returns angles and training cost. | Aer, scipy |
| Angle cache | `bench/angles.py` | Store and fetch trained angles by key (section 5.4), in the study's store | `bench.store` |
| Backends | `solvers/quantum/backends.py` | Run one batch of shots at given angles on `ideal`, `noisy` or `ibm`; return bitstrings, counts and backend metadata | Aer, qiskit-ibm-runtime |
| QAOA sampler | `solvers/quantum/qaoa.py` | A `QuboSampler`. `draw` fetches the angles and runs one batch on its backend, so decoding, feasibility and scoring are the M4 samplers' own. | the above |
| Quota ledger | `solvers/quantum/quota.py` | Records QPU seconds per job outside any store, enforces the cap, and holds pending job ids | qiskit-ibm-runtime |
| Simulator gate | `bench/gate.py` | Decides whether a `qaoa_ibm` task may run: requires a stored, successful gate run (section 7.2) | `bench.store` |
| Hardware lane | `bench/hardware.py` | Runs `qaoa_ibm` tasks one at a time in the main process. Owns submit, record id, wait, collect, store. | ledger, gate, store |

The import contracts gain `quantic.solvers.quantum` as a layer beside `quantic.solvers.qubo`. The classical solvers must not import it, and the encoder still does not read the solvers.

## 5. The algorithm

### 5.1 QUBO to circuit

- **The map:** `x_i = (1 − z_i)/2` turns the QUBO `xᵀQx + c` into `Σ h_i Z_i + Σ J_ij Z_i Z_j + c'`.
- **Scaling:** coefficients are divided by `s = max(|h|, |J|)`. A bitstring's QUBO energy is recovered exactly as `s · ⟨z|H|z⟩ + c'`, and a test enumerates every bitstring of small QUBOs to check this.
- **Dynamic range:** `max|coef| / min|nonzero coef|` is recorded per QUBO. At T0 it is about 5 × 10⁵ for binary. Below roughly 10⁻⁴ of the largest term, a term's rotation is beyond what hardware resolves. This is reported as a property of the problem, not corrected.

### 5.2 Circuit

- **Layers:** p layers of `exp(−iγ_k H)` then `exp(−iβ_k Σ X_i)`, starting from `|+⟩^n`, with p = 2 by default (config `reps`).
- **Initial angles:** the linear ramp `γ_k = (k − ½)/p · Δ`, `β_k = (1 − (k − ½)/p) · Δ`, with Δ = 0.75. The start is deterministic, with no random restarts.

### 5.3 Training

- **Objective:** CVaR_α of the energy, with α = 0.25: the mean of the lowest-energy quarter of the probability mass.
- **Computed exactly, not sampled:** the energy of every basis state is computed once per QUBO, as a vector of 2ⁿ floats. The statevector gives probabilities; sort by energy and integrate to α.
  - Memory at 28 qubits is about 8 GB: 4 GB statevector, 2 GB probabilities, 2 GB energies. That is within the 64 GB machine.
  - The cap `max_qubits` protects it.
- **Optimiser:** COBYLA, `maxiter` 200 by default, fixed seed.
- **Recorded:** final CVaR, number of evaluations, simulator seconds, Aer and qiskit versions.

### 5.4 Angle cache

- **Key:** SHA-256 of (QUBO content hash, p, α, `maxiter`, initial-angle rule, qiskit and Aer versions).
- **Shared by all three backends,** so they run exactly the same circuit.
- **Store table:** `angles(key, qubo_hash, reps, alpha, gammas, betas, cvar, n_evals, train_s, versions)`.
- **Training on demand:** the first QAOA task on a QUBO trains its angles, and later tasks read them. Training runs in the worker that needs it. Two workers needing the same angles both train; the first insert wins and the results are identical, because training is deterministic.

## 6. Backends and what a run measures

**A run** is one batch of `shots` measurements at the trained angles, 1,000 by default (config `shots`). Each distinct bitstring is decoded, classified and scored exactly as a sampler's read is (`solvers/qubo/sampler.py`). The `Result` is the best feasible sample, `HEURISTIC` status. `n_samples` is the shot count.

| Backend | Solver name | Circuit | Run seed varies | Time recorded |
|---|---|---|---|---|
| Noiseless | `qaoa_ideal` | logical circuit, Aer statevector | shot sampling | simulation seconds (not TTS) |
| Simulated noise | `qaoa_noisy` | routed for the target device, Aer with `AerSimulator.from_backend(fake twin)` | shot sampling and noise trajectories | simulation seconds (not TTS) |
| IBM hardware | `qaoa_ibm` | routed for the device, `qiskit_ibm_runtime.executor_sampler.Sampler` | none; one job per task | `job.usage()` QPU seconds, plus queue and wall seconds |

- **Target device:** set in config (`device`, for example `ibm_fez`). It must have a fake twin in `qiskit_ibm_runtime.fake_provider`, so `qaoa_noisy` and the gate route for the same coupling map and basis gates.
- **Routing:** `optimization_level` 3, fixed `seed_transpiler`. The routed circuit's depth and two-qubit count are stored on every routed run.
- **`qaoa_noisy` limits:** planned at full shots only when the QUBO has at most `noisy_max_qubits` variables, 20 by default (D5).
- **Metrics:**
  - `qaoa_ideal` and `qaoa_noisy` report median gap, feasibility, and success rate per run at each ε. Their TTS column reads "simulated".
  - `qaoa_ibm` reports the same, plus TTS99 computed with t_run equal to the job's QPU usage seconds.
  - Training cost has its own column.
- **Matched reads:** the M5 study config also runs `simulated_annealing`, `tabu` and `greedy` on the same instances with `max_samples = shots` and no effective wall-clock limit. A new report table compares QAOA and samplers at equal read counts: success rate per run at each ε, median gap, feasibility.

## 7. The hardware lane

### 7.1 Quota ledger

- **Location:** `%USERPROFILE%\.quantic\ibm-quota.json`, overridable by `QUANTIC_QUOTA_PATH`. It lives outside any store, because the quota belongs to the account. Writes are atomic: write a temporary file, then replace.
- **Entries:** one per job: job id, device, submitted at, `max_execution_time`, state (`pending` or `done`), usage seconds once known.
- **Before each submission:**
  1. `used` = max(IBM's `service.usage()` consumed seconds over the window, if the call succeeds; the ledger's own sum over the last 28 days, with pending jobs counted at their `max_execution_time`).
  2. The job is submitted with `max_execution_time = min(per_job_cap, cap − used)`.
  3. If `cap − used < min_job_s` (default 10 s), the submission is refused with `QuotaExhausted`.
- **Effect:** no single job, even one that runs away, can take the account past the cap.
- **`cap`:** default 300 s, config `quota_cap_s`. The config may lower the cap but never raise it above 300 s without editing the code constant (D4).

### 7.2 Simulator gate

A `qaoa_ibm` task may run only when the store holds a successful gate row for the same instance, encoding and angle key, routed for the same device. A gate row is a `qaoa_noisy` run with `shots = gate_shots` (32), status `HEURISTIC`, no error. The planner adds one gate task for every planned `qaoa_ibm` task, and gate tasks run in the process pool before the hardware lane starts. Gate rows are kept out of `qaoa_noisy`'s measurement cells, which are distinguished by `shots`.

The gate proves the full path before any quota is spent: training, routing for that device, the Sampler call shape, decoding and scoring. It runs at any qubit count up to `max_qubits`, a few minutes at 28 qubits. A refused task stays to do. The runner prints which gate run is missing.

### 7.3 Job lifecycle

1. Check the gate, then the ledger. Either can refuse.
2. Route the circuit, then submit.
3. **Before waiting,** write the job id to the ledger (`pending`) and the store (a `pending` row).
4. Wait for the result, then write the samples and metadata to the store in one transaction, and mark the ledger entry `done` with `job.usage()`.
5. **On resume:** a pending job id is collected, never re-submitted. A job IBM reports as failed becomes an `error` row with IBM's message. Its usage is still recorded.

The hardware lane runs after the process pool has drained, one job at a time, in the main process. `quantic bench run` takes `--no-hardware` to skip it.

### 7.4 No repeats

A `qaoa_ibm` task counts as done if any store row for the same task fields, ignoring commit and lock hash, is `ok`, `pending` or `error`. `--repeat-hardware` overrides this, and the planner says how much quota the repeats would need.

## 8. Store changes

- **New table:** `angles` (section 5.4).
- **`runs` gains nullable columns:**
  - `backend`: `ideal`, `noisy` or `ibm`
  - `device`
  - `job_id`
  - `qpu_s`
  - `queue_s`
  - `routed_depth`
  - `routed_2q`
  - `angle_key`
  - `reps`
  - `shots`
- **`store_status` gains `pending`,** for hardware rows only. Metrics and the report skip pending rows and count them in the `quota` table.
- **Migration:** existing stores open unchanged, and new columns read as null.

## 9. Configuration

`experiments/m5-qaoa.toml`:

```toml
[instances]
tiers = ["T0"]
dials = ["none", "D1", "D2", "D4", "D1+D2", "D1+D4", "D2+D4", "D1+D2+D4"]
seeds = "2026..2030"

[quantum]
reps = 2
alpha = 0.25
maxiter = 200
shots = 1000
max_qubits = 28
noisy_max_qubits = 20
gate_shots = 32
device = "ibm_fez"
quota_cap_s = 300

[solvers.qaoa_ideal]
encodings = ["binary", "domain_wall"]
seeds = "0..19"

[solvers.qaoa_noisy]
encodings = ["binary", "domain_wall"]
seeds = "0..4"

[solvers.qaoa_ibm]
encodings = ["binary"]
seeds = "0..0"

[solvers.simulated_annealing]   # and tabu, greedy: matched reads
encodings = ["binary", "domain_wall"]
seeds = "0..19"
max_samples = 1000
budgets_s = [3600]
```

The QAOA solvers take no `budgets_s`: a run is its shot count. The device name is checked against the fake provider at load time.

## 10. Report

- **Crossover tables:** gain the three QAOA columns. A QAOA cell's TTS reads "simulated" for `ideal` and `noisy`.
- **New tables:**
  - `matched_reads`: QAOA and the samplers at `max_samples = shots`, per cell and encoding.
  - `qaoa_noise`: per instance and encoding, the same angles under `ideal`, `noisy` and `ibm`, with success at ε = 1%, feasibility, routed depth and two-qubit count, and `qpu_s` for hardware.
  - `quota`: the ledger's jobs, QPU seconds, and seconds left in the window, at report time.
- **Bracket check:** unchanged. QAOA never claims a bound.

## 11. Errors

| What | Result |
|---|---|
| QUBO over `max_qubits` | not planned; the planner reports the count left out |
| `noisy_max_qubits` exceeded at full shots | not planned, as above; gate runs still planned |
| Gate run missing | hardware task stays to do; message names the gate key |
| `QuotaExhausted` | hardware lane stops, tasks stay to do, exit status non-zero with the seconds used and left |
| No IBM account saved | hardware lane stops before any submission, with how to save the token; simulator lanes unaffected |
| IBM job failed or cancelled | `error` row with IBM's message; usage recorded |
| Network failure while waiting | job stays `pending`; the next run collects it |
| Training did not converge in `maxiter` | not an error: the angles are what COBYLA reached, and `n_evals` = `maxiter` is recorded |

## 12. Testing

No test touches the network. The IBM path runs in Qiskit Runtime's local testing mode, against a fake device and a fake service.

- **Ising map:** on random QUBOs of up to 10 variables, `s · ⟨z|H|z⟩ + c'` equals the QUBO energy for every bitstring.
- **Training:** deterministic, so the same inputs give the same angles. On 2–6 qubit QUBOs, the noiseless sampler's best sample is the QUBO minimum. The exact CVaR equals a brute-force CVaR on small cases.
- **QAOA sampler:** goes through `sampler.run_sampler` like the M4 samplers; infeasible samples are counted, never repaired.
- **Ledger:**
  - window arithmetic, including pending jobs counted at their cap
  - the larger of IBM's figure and the ledger's own
  - refusal below `min_job_s`
  - `max_execution_time` never pushes `used` past the cap
  - atomic writes
  - the cap cannot be configured above 300 s
- **Gate:** refuses without a gate row, accepts with one, rejects a gate row for another device or another angle key.
- **Lifecycle:** submit then crash before collect leaves a pending row and a pending ledger entry, and resume collects instead of re-submitting. A failed job becomes an `error` row. No-repeats holds across commits, and `--repeat-hardware` overrides it.
- **Live smoke test:** one test runs a 4-qubit QAOA on real hardware. It is skipped unless `QUANTIC_IBM_LIVE=1` and an account is saved, and it is never run by default.

## 13. Done

1. **Suite and checks:** the full suite passes; ruff and the import contracts pass.
2. **Simulator study:** `experiments/m5-qaoa.toml` has run every simulator task, and every hardware task's gate has passed.
3. **Hardware:** at least one real `qaoa_ibm` result is stored, with the ledger under 300 s for the window. The hardware lane has run until done or until the ledger stopped it.
4. **Result document:** `docs/superpowers/specs/<date>-m5-qaoa-result.md` reports:
   - the matched-reads comparison
   - the noise ladder (`ideal`, `noisy`, `ibm`)
   - the effect of the dynamic range
   - QPU seconds spent
   - how far the hardware runs got

## 14. Risks

| Risk | Mitigation |
|---|---|
| Hardware output is mostly noise: about 1,000–1,300 two-qubit gates at 20 qubits, error-free fraction about 2% | Expected and reported as the finding. The `qaoa_noise` table shows where the signal goes. |
| Training at 28 qubits is slow, about 200 evaluations × several seconds | Measured in the plan's dry run. `max_qubits` can be lowered, and the cache means each QUBO trains once. |
| Dynamic range of about 5 × 10⁵ hides small terms from QAOA | Reported per QUBO; the result says how much of the gap it explains. |
| IBM changes the Open plan, the API or the Sampler (`SamplerV2` deprecated in qiskit-ibm-runtime 0.50) | Versions are pinned in the lock file. The executor `Sampler` is used from the start. The ledger takes the larger of IBM's figure and its own. |
| A queued job outlives the session | Job ids persist before waiting; resume collects. |

## 15. To verify in the plan's dry run

- Training time per QUBO at 16, 20, 24 and 28 qubits, so the study's wall clock can be estimated.
- `job.usage()` for a 1,000-shot routed T0 circuit on the target device. One live job of a few seconds needs the owner's go-ahead.
- The shape of `service.usage()` on the owner's plan: which field holds consumed seconds, and over what window.
