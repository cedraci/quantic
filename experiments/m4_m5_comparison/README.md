# The M4/M5 comparison's computations

Every number in `docs/superpowers/specs/2026-10-28-m4-m5-comparison.md` that is not quoted from a
result document comes from one of these scripts. They read the report tables committed under
`docs/figures/m5/data/` (paths are resolved from the repository root), and a few read the study
stores, which are not committed, from paths given on the command line. Run them from the
repository root with the project's interpreter, for example:

    .venv/Scripts/python.exe -I experiments/m4_m5_comparison/compare.py time

| script | arguments | what it computes | comparison sections |
|---|---|---|---|
| `same_instances.py` | `<M4 store> <M5 store>` | whether the 40 shared T0 instances carry the same content hash and the same bracket in both stores | 4.1 |
| `highs_load.py` | — | HiGHS's run times on the eight shared cells in the M4 and M5 stores, at the same budget | 4.6 |
| `m4_t0.py` | — | the M4 crossover's T0 rows at 0.1 s and 60 s with the sound CP-SAT, and each solver's best TTS99 over budgets | 5.2, 5.3 |
| `m4_best.py` | `T0` or `T1` | each solver's best TTS99 over budgets and encodings per cell, the winner and its margin | 5.2 |
| `facts.py` | `p2` or `p1` | the derived numbers of one QAOA rung: hardware totals, successes against chance, the noise ladder, training, dynamic-range correlations | 6, 7 |
| `m5_tables.py` | `matched`, `tts`, `ladder`, `cellsrung`, `perinstance` or `training` | the M5 result's and sections 6–7's in-text tables | 6, 7 |
| `compare.py` | `time`, `quality`, `factors`, `dials`, `encodings` or `cost` | the cross-study tables: time on every clock, quality at matched reads, factors, dial effects, encodings, costs | 8 |
| `samplers_on_hw.py` | `<M5 store> <out.csv>` | the samplers at 1,000 reads restricted to the instances QAOA ran on the device, scored with `bench.metrics.scored` | 6.6, 8.1, 8.4 |
| `mixing_model.py` | — | the parameter-free error-fraction model of the device's feasibility | 7.8, 8.5 |
| `couplings.py` | — | QUBO couplings at T0 and T1 (five seeds each), the routing overhead of the device circuits, and what a T1 circuit would need | 1, 3.7, 8.9, 10.2 |
| `sizes.py` | — | the variable (qubit) count of every QUBO per tier, dials and encoding, over the five seeds | 3.7 |
| `qualifications.py` | `<reps-2 store> <reps-1 store>` | HiGHS per instance and each training against it; the device's successes split by depth against chance; rank correlations with the qubit count held fixed; the rung's and the third pass's significance; the noise model's bias; what ranks the device's feasibility; the device's QPU TTS99 against every classical clock; the lane's per-job limits and wall time | 1, 6, 7, 8, 9, 10 |
| `routecheck_p1.py` | `<reps-2 angle cache> <reps-1 angle cache>` | the physical qubits each rung's routed circuit touches on `FakeFez`, for the killed noise-model runs | 7.3 |

The figures are built separately by `experiments/m5-figures.py`, the uniform baselines by
`experiments/m5-uniform-baseline.py`, the per-job table by `experiments/m5-hardware-jobs.py`.
