# M3b gate — result

**Written:** 2026-09-30
**Branch:** `feat/m3b-exact-baseline`
**Spec:** `2026-09-29-m3b-exact-baseline-design.md`
**Plan:** `docs/superpowers/plans/2026-09-30-m3b-exact-baseline.md`

---

## Verdict

**The gate passes.** On all sixteen T0 dial combinations two exact methods agree on the optimum, and under every encoding, with the CVaR cells at twenty zeta bits, the QUBO's minimum decodes to a feasible schedule whose true objective is that optimum. "Minimum" is over every state in 31 of the 48 cases, and over the states that decode in the other 17 (`DECODABLE`: one-hot 11, domain-wall 6; see below). Under the binary encoding all 16 combinations are proven over every state. The report finds nothing on any check.

| what | result |
|---|---|
| gate | 18 of 18 pass |
| T1 measurement | 6 of 6 tests pass; 16 of 16 instances OPTIMAL |
| full suite | 1181 pass, 1 skipped, 0 fail |
| `ruff check src tests` | clean (`All checks passed!`) |
| `lint-imports` | 3 kept, 0 broken |
| gate wall time | gate checks 81 s, T1 measurement 53 s (the report's last line, run with nothing else on the machine) |

The skip is `tests/micro/test_reconstruction_gate.py`, which needs a real paired L2/L3 bundle named by `QUANTIC_REAL_BUNDLE`. It is unrelated to this milestone.

The full suite took 677 s (11 min 17 s). Its slowest item is the M3a gate's setup, 449.7 s; the M3b gate's setup is 93.1 s and the T1 measurement's 50.3 s. The gate and the T1 measurement together, run as their own two files, took 138.9 s.

Machine: Windows 11 (10.0.26200), Intel Core i9-9900 at 3.10 GHz, 63.9 GB, Python 3.11.15, pytest 9.1.1.

## What passing establishes, and what it does not

**What is proven.** On sixteen combinations, under three encodings, with the CVaR cells encoded at twenty zeta bits (`GATE_ZETA_BITS`), the QUBO's minimum -- over every state in 31 cases, over the states that decode in the other 17 (`DECODABLE`: one-hot 11, domain-wall 6; states that `decode` rejects are not examined, spec section 8) -- is a feasible schedule, and its true objective equals the optimum that brute force and the MILP agree on. Every one of the 16 combinations is proven over every state under the binary encoding. The two exact methods agree to `1e-9` relative, both being `evaluate` at a schedule. The MILP's bracket closes on all sixteen (widest 4.4e-13 at T0). The QUBO's regret against the optimum is 0 in all 48 cases. The width is fixed, not derived from the instance, so the assertion could have failed.

**What is not proven: the study's own QUBOs.** The study runs at eight zeta bits, and there the QUBO's minimum is not the optimum on five cells. The regret of each, from the fourth table below:

| cell | regret | regret / optimum |
|---|---|---|
| `T0-D3` | 0.7485 | 3.46% |
| `T0-D2+D3` | 0.7650 | 2.48% |
| `T0-D2+D3+D4` | 0.0954 | 0.56% |
| `T0-D1+D3` | 1.4777 | 1.98% |
| `T0-D1+D2+D3+D4` | 0.0717 | 0.23% |

The other three CVaR cells, `T0-D3+D4`, `T0-D1+D3+D4` and `T0-D1+D2+D3`, have zero regret at eight bits. The gate asserts that check 6's findings at eight bits are exactly these five cells under each of the three encodings, and no others.

**What is not proven: the rejected states.** In seventeen of forty-eight cases the minimum is taken over the states that decode, not over every state: eleven under one-hot and six under domain-wall (none under binary, where every case is over every state). In those cases a state that `decode` rejects, with two one-hot bits set in a cell or a hole in a domain wall, was not examined, and nothing in this milestone shows that none of them lies below the minimum. The validity penalty's strength is derived to prevent it, and the fifteen `EVERY_STATE` cases under the table encodings at T0 are where that derivation is checked. Every one of the sixteen combinations is covered over every state under the binary encoding.

## The exact optimum, two ways

| instance | feasible schedules | optimum | brute force, s | MILP, s | rounds | binaries | presolve retries | bracket width |
|---|---|---|---|---|---|---|---|---|
| `T0-none-2026` | 3200 | 29.534182 | 0.27 | 0.08 | 7 | 48 | 0 | 0.0e+00 |
| `T0-D4-2026` | 40 | 64.637495 | 0.00 | 0.05 | 5 | 42 | 0 | 2.8e-14 |
| `T0-D3-2026` | 38440 | 21.626217 | 5.21 | 0.13 | 1 | 48 | 0 | 0.0e+00 |
| `T0-D3+D4-2026` | 64 | 3.017044 | 0.01 | 0.00 | 1 | 24 | 0 | 4.4e-16 |
| `T0-D2-2026` | 12 | 20.445419 | 0.00 | 0.17 | 7 | 24 | 0 | 2.5e-14 |
| `T0-D2+D4-2026` | 456 | 32.553612 | 0.04 | 0.07 | 6 | 32 | 0 | 0.0e+00 |
| `T0-D2+D3-2026` | 456 | 30.807496 | 0.08 | 0.16 | 1 | 32 | 0 | 4.4e-13 |
| `T0-D2+D3+D4-2026` | 108 | 17.031364 | 0.01 | 0.09 | 1 | 31 | 0 | 1.1e-14 |
| `T0-D1-2026` | 1760 | 49.432296 | 0.13 | 0.06 | 7 | 44 | 0 | 7.1e-15 |
| `T0-D1+D4-2026` | 3080 | 64.418935 | 0.31 | 0.12 | 7 | 42 | 0 | 0.0e+00 |
| `T0-D1+D3-2026` | 100 | 74.456636 | 0.01 | 0.01 | 1 | 44 | 0 | 0.0e+00 |
| `T0-D1+D3+D4-2026` | 1200 | 26.142061 | 0.14 | 0.01 | 1 | 40 | 0 | 0.0e+00 |
| `T0-D1+D2-2026` | 12 | 30.919391 | 0.00 | 0.18 | 7 | 24 | 1 | 0.0e+00 |
| `T0-D1+D2+D4-2026` | 12 | 35.860684 | 0.00 | 0.04 | 4 | 24 | 0 | 2.1e-14 |
| `T0-D1+D2+D3-2026` | 12 | 24.068671 | 0.00 | 0.01 | 1 | 24 | 0 | 3.6e-15 |
| `T0-D1+D2+D3+D4-2026` | 162 | 31.168557 | 0.02 | 0.06 | 1 | 27 | 0 | 0.0e+00 |

## The QUBO's minimum at twenty zeta bits

| instance | encoding | variables | scope | states | scanned | widest window | ties | regret | dense error |
|---|---|---|---|---|---|---|---|---|---|
| `T0-none-2026` | binary | 24 | every_state | 16777216 | 0 | 0 | 1 | 0.0e+00 | 2.5e-13 |
| `T0-none-2026` | one_hot | 48 | decodable | 16777216 | 0 | 0 | 1 | 0.0e+00 | 4.9e-13 |
| `T0-none-2026` | domain_wall | 36 | decodable | 16777216 | 0 | 0 | 1 | 0.0e+00 | 4.9e-13 |
| `T0-D4-2026` | binary | 21 | every_state | 2097152 | 0 | 0 | 1 | 0.0e+00 | 1.1e-13 |
| `T0-D4-2026` | one_hot | 39 | decodable | 2097152 | 0 | 0 | 1 | 0.0e+00 | 1.1e-13 |
| `T0-D4-2026` | domain_wall | 30 | decodable | 2097152 | 0 | 0 | 1 | 0.0e+00 | 5.6e-14 |
| `T0-D3-2026` | binary | 644 | every_state | 16777216 | 38440 | 129 | 1 | 0.0e+00 | 4.7e-08 |
| `T0-D3-2026` | one_hot | 668 | decodable | 16777216 | 38440 | 129 | 1 | 0.0e+00 | 6.9e-08 |
| `T0-D3-2026` | domain_wall | 656 | decodable | 16777216 | 38440 | 129 | 1 | 0.0e+00 | 6.9e-08 |
| `T0-D3+D4-2026` | binary | 632 | every_state | 4096 | 64 | 1 | 64 | 0.0e+00 | 5.9e-16 |
| `T0-D3+D4-2026` | one_hot | 643 | every_state | 8388608 | 64 | 1 | 64 | 0.0e+00 | 7.1e-15 |
| `T0-D3+D4-2026` | domain_wall | 632 | every_state | 4096 | 64 | 1 | 64 | 0.0e+00 | 5.9e-16 |
| `T0-D2-2026` | binary | 16 | every_state | 65536 | 0 | 0 | 1 | 0.0e+00 | 6.8e-14 |
| `T0-D2-2026` | one_hot | 28 | every_state | 268435456 | 0 | 0 | 1 | 0.0e+00 | 1.1e-13 |
| `T0-D2-2026` | domain_wall | 16 | every_state | 65536 | 0 | 0 | 1 | 0.0e+00 | 6.8e-14 |
| `T0-D2+D4-2026` | binary | 28 | every_state | 268435456 | 0 | 0 | 4 | 0.0e+00 | 1.1e-12 |
| `T0-D2+D4-2026` | one_hot | 39 | decodable | 26873856 | 0 | 0 | 4 | 0.0e+00 | 1.2e-12 |
| `T0-D2+D4-2026` | domain_wall | 28 | every_state | 268435456 | 0 | 0 | 4 | 0.0e+00 | 3.4e-13 |
| `T0-D2+D3-2026` | binary | 648 | every_state | 268435456 | 1536 | 129 | 4 | 0.0e+00 | 8.0e-08 |
| `T0-D2+D3-2026` | one_hot | 660 | decodable | 26873856 | 1536 | 129 | 4 | 0.0e+00 | 4.9e-08 |
| `T0-D2+D3-2026` | domain_wall | 648 | every_state | 268435456 | 1536 | 129 | 4 | 0.0e+00 | 8.0e-08 |
| `T0-D2+D3+D4-2026` | binary | 647 | every_state | 134217728 | 504 | 129 | 2 | 0.0e+00 | 7.5e-08 |
| `T0-D2+D3+D4-2026` | one_hot | 658 | decodable | 17915904 | 504 | 129 | 2 | 0.0e+00 | 4.7e-08 |
| `T0-D2+D3+D4-2026` | domain_wall | 647 | every_state | 134217728 | 504 | 129 | 2 | 0.0e+00 | 7.5e-08 |
| `T0-D1-2026` | binary | 24 | every_state | 16777216 | 0 | 0 | 1 | 0.0e+00 | 0.0e+00 |
| `T0-D1-2026` | one_hot | 44 | decodable | 5308416 | 0 | 0 | 1 | 0.0e+00 | 2.9e-12 |
| `T0-D1-2026` | domain_wall | 32 | decodable | 5308416 | 0 | 0 | 1 | 0.0e+00 | 1.2e-12 |
| `T0-D1+D4-2026` | binary | 21 | every_state | 2097152 | 0 | 0 | 1 | 0.0e+00 | 2.0e-13 |
| `T0-D1+D4-2026` | one_hot | 39 | decodable | 2097152 | 0 | 0 | 1 | 0.0e+00 | 1.1e-13 |
| `T0-D1+D4-2026` | domain_wall | 30 | decodable | 2097152 | 0 | 0 | 1 | 0.0e+00 | 1.4e-13 |
| `T0-D1+D3-2026` | binary | 644 | every_state | 16777216 | 280 | 129 | 1 | 0.0e+00 | 2.3e-08 |
| `T0-D1+D3-2026` | one_hot | 664 | decodable | 5308416 | 100 | 129 | 1 | 0.0e+00 | 2.7e-08 |
| `T0-D1+D3-2026` | domain_wall | 652 | decodable | 5308416 | 100 | 129 | 1 | 0.0e+00 | 2.7e-08 |
| `T0-D1+D3+D4-2026` | binary | 642 | every_state | 4194304 | 3360 | 129 | 4 | 0.0e+00 | 2.8e-09 |
| `T0-D1+D3+D4-2026` | one_hot | 658 | decodable | 1327104 | 1200 | 129 | 1 | 0.0e+00 | 1.1e-08 |
| `T0-D1+D3+D4-2026` | domain_wall | 648 | every_state | 268435456 | 1200 | 129 | 1 | 0.0e+00 | 6.8e-09 |
| `T0-D1+D2-2026` | binary | 16 | every_state | 65536 | 0 | 0 | 1 | 0.0e+00 | 1.5e-14 |
| `T0-D1+D2-2026` | one_hot | 28 | every_state | 268435456 | 0 | 0 | 1 | 0.0e+00 | 5.9e-14 |
| `T0-D1+D2-2026` | domain_wall | 16 | every_state | 65536 | 0 | 0 | 1 | 0.0e+00 | 1.5e-14 |
| `T0-D1+D2+D4-2026` | binary | 16 | every_state | 65536 | 0 | 0 | 1 | 0.0e+00 | 6.3e-14 |
| `T0-D1+D2+D4-2026` | one_hot | 27 | every_state | 134217728 | 0 | 0 | 1 | 0.0e+00 | 6.4e-14 |
| `T0-D1+D2+D4-2026` | domain_wall | 16 | every_state | 65536 | 0 | 0 | 1 | 0.0e+00 | 6.3e-14 |
| `T0-D1+D2+D3-2026` | binary | 636 | every_state | 65536 | 12 | 129 | 1 | 0.0e+00 | 3.4e-08 |
| `T0-D1+D2+D3-2026` | one_hot | 648 | every_state | 268435456 | 12 | 129 | 1 | 0.0e+00 | 3.4e-08 |
| `T0-D1+D2+D3-2026` | domain_wall | 636 | every_state | 65536 | 12 | 129 | 1 | 0.0e+00 | 3.4e-08 |
| `T0-D1+D2+D3+D4-2026` | binary | 643 | every_state | 8388608 | 756 | 129 | 8 | 0.0e+00 | 4.1e-09 |
| `T0-D1+D2+D3+D4-2026` | one_hot | 654 | decodable | 3538944 | 756 | 129 | 8 | 0.0e+00 | 1.1e-08 |
| `T0-D1+D2+D3+D4-2026` | domain_wall | 643 | every_state | 8388608 | 756 | 129 | 8 | 0.0e+00 | 1.1e-08 |

## Regret at the study's eight zeta bits

| instance | optimum | regret | regret / optimum | declared resolution |
|---|---|---|---|---|
| `T0-D3-2026` | 21.6262 | 0.7485 | 3.46% | 17.28 |
| `T0-D3+D4-2026` | 3.0170 | 0.0000 | 0.00% | 0.00 |
| `T0-D2+D3-2026` | 30.8075 | 0.7650 | 2.48% | 25.31 |
| `T0-D2+D3+D4-2026` | 17.0314 | 0.0954 | 0.56% | 11.47 |
| `T0-D1+D3-2026` | 74.4566 | 1.4777 | 1.98% | 24.09 |
| `T0-D1+D3+D4-2026` | 26.1421 | 0.0000 | 0.00% | 14.89 |
| `T0-D1+D2+D3-2026` | 24.0687 | 0.0000 | 0.00% | 16.06 |
| `T0-D1+D2+D3+D4-2026` | 31.1686 | 0.0717 | 0.23% | 8.81 |

The largest regret as a share of its optimum is 3.46%, on `T0-D3` (0.7485 against an optimum of 21.6262); the largest in absolute terms is 1.4777, on `T0-D1+D3` (1.98%).

## The exact program at T1

| instance | status | upper | bracket width | rounds | binaries | presolve retries | seconds |
|---|---|---|---|---|---|---|---|
| `T1-none-2026` | optimal | 266.599721 | 0.0e+00 | 5 | 312 | 0 | 0.1 |
| `T1-D4-2026` | optimal | 191.500640 | 6.5e-11 | 15 | 245 | 0 | 1.4 |
| `T1-D3-2026` | optimal | 198.969937 | 0.0e+00 | 1 | 304 | 0 | 0.7 |
| `T1-D3+D4-2026` | optimal | 247.255491 | 1.7e-13 | 1 | 248 | 0 | 0.3 |
| `T1-D2-2026` | optimal | 92.451944 | 1.8e-13 | 21 | 120 | 0 | 4.8 |
| `T1-D2+D4-2026` | optimal | 185.640236 | 3.9e-11 | 37 | 116 | 0 | 20.4 |
| `T1-D2+D3-2026` | optimal | 101.720072 | 2.4e-13 | 1 | 112 | 0 | 0.2 |
| `T1-D2+D3+D4-2026` | optimal | 115.589574 | 0.0e+00 | 1 | 117 | 0 | 0.1 |
| `T1-D1-2026` | optimal | 459.653689 | 5.7e-14 | 12 | 280 | 0 | 0.4 |
| `T1-D1+D4-2026` | optimal | 62.375380 | 0.0e+00 | 34 | 241 | 0 | 7.8 |
| `T1-D1+D3-2026` | optimal | 442.635542 | 8.6e-11 | 1 | 320 | 0 | 0.6 |
| `T1-D1+D3+D4-2026` | optimal | 217.111569 | 0.0e+00 | 1 | 254 | 0 | 0.2 |
| `T1-D1+D2-2026` | optimal | 287.642869 | 0.0e+00 | 44 | 96 | 0 | 15.7 |
| `T1-D1+D2+D4-2026` | optimal | 170.167723 | 8.5e-14 | 11 | 118 | 0 | 0.3 |
| `T1-D1+D2+D3-2026` | optimal | 127.850176 | 4.3e-14 | 1 | 104 | 0 | 0.1 |
| `T1-D1+D2+D3+D4-2026` | optimal | 198.974003 | 1.1e-11 | 1 | 117 | 0 | 0.2 |

All sixteen were solved to optimality. The slowest is `T1-D2+D4`, at 20.4 s in 37 rounds; the most rounds are `T1-D1+D2`, 44, in 15.7 s. There were no presolve retries at T1. At T0 there was one, on `T0-D1+D2`. The widest bracket at T1 is 8.6e-11 (`T1-D1+D3`). The 120 s budget was never reached.

## What the dry run and the build found

| finding | where | disposition |
|---|---|---|
| the outer-approximation loop did not end when HiGHS returned a schedule twice | `exact_pwl_milp`, on a three-bucket instance with cardinality | fixed before the plan was written: the loop stops at a schedule it has already cut |
| HiGHS's presolve fails on small programs that carry cuts | `T0-D1+D2`, fourth round | retried without presolve; counted in `Result.detail` |
| with `optimal_aux`'s rule the closed form still crowned the right winner | `gadget_minimum`, on the eight-scenario instance | the closed form is tested at every state, not at the winner |
| the plan's code tested the variance gradient at the witness only, though its prose asks for the loss model and the gradient at the schedule that sells nothing | `milp_model`, Task 4 | a test of the gradient against the evaluator at zero holdings was added, beyond the plan's code |
| the out-of-grid test never reached the window-widening branch (five bits, top level 31 below the first window of 64), though the spec names it as the evidence for the claim | `gadget_minimum`, Task 6 | the test now exercises the doubling branch and is compared with an exhaustive scan of the levels |
| the plan's defect tests asserted that some finding was reported, where spec §9 names checks 1 to 3; the eight-bit claim was asserted through a regret proxy; check 8 asserted only the `EVERY_STATE` counts | `tests/solvers/test_m3b_gate.py`, Task 8 | tightened to the spec: the defect tests name checks 1 to 3; at eight bits check 6's own findings are asserted to be exactly the five cells under every encoding (the proxy agreed with it); the `DECODABLE` counts 0, 6 and 11 are asserted alongside the `EVERY_STATE` ones |

## What is still open

- **The rejected states in the seventeen `DECODABLE` cases.** Closing this takes a branch and bound over cells, or a bound on what a rejected cell can gain. Neither is here (spec §8).
- **The study's zeta bit budget.** At eight bits the encoding is correct and the regret above is what it costs; at T2 and T3 the declared resolution is the size of the objective (M3a gate result). Choosing the budget is M5's.
- **`max_single_cell_gain` is per cell.** The gain in spread, impact and variance is bounded by what one cell's bits can do; a two-cell move could in principle buy twice that. Unchanged by this milestone (M3a gate result, "What is still open").
- **Nothing checks HiGHS's lower bound at T1.** At T0 enumeration would disagree with a wrong bound. At T1 the upper bound is the evaluator's and the schedule is classified, but the lower bound is HiGHS's word.
- **A future ladder with a smaller gap than twenty bits resolves** would fail check 6 and name the cell. The width is a named constant, and agreement is not monotone in it.
- **Opened by this build, deferred from review, none of them changing a result here.**
  The final review fixed two items that were listed here: `Bracket` now rejects NaN bounds, and a numerical infeasible status after a feasible round now raises instead of discarding the incumbent. `lot_grid` takes block lots unclamped to the cap. Duplicate `MinParticipation` or `BlockTrades` descriptors overwrite silently. The presolve retry reuses a stale `time_limit`, so a run can overrun `wall_clock_s` by one attempt. `detail["rounds"]` counts a round that never ran when the budget is exhausted. scipy status 1 is always labelled `TIME_LIMIT`. `qubo_minimum` raises the built-in `MemoryError` for a logic condition, not out-of-memory. Gate check 9's lower bound and check 10's `lower <= upper` carry solver-sized slack. `tests/test_architecture.py` emitted a threading or Unicode warning during Task 1 that was not shown to be pre-existing; the full-suite run above shows no warning in its summary. The final review passed these to M4, whose design (`2026-10-02-m4-classical-bench-design.md` §8) fixes the duplicate descriptors, the stale retry limit, the round count, the status-1 label and the `MemoryError`, and records the block-lot clamp as correct.

## What M4 picks up

`Solver`, `Result` and `Bracket` are the interface; M4 adds the rest of the classical slate behind the same protocol, then the bench runner and the store.

- **The regret to compare a sampler against.** A sampler that returns the QUBO's minimum at eight zeta bits has not failed when its schedule is not the baseline's. On five of the eight CVaR cells the minimum is not the optimum, and the regret table above (0.0717 to 1.4777, 0.23% to 3.46% of the optimum) is the number to compare against.
- **The approximand's cost table.** Under the binary encoding with D1 on, the QUBO minimises the approximand, not the power law. `exact_pwl_milp(instance, cell_cost=report.cell_cost)`, and `brute_force` with the same argument, give the optimum of the function the encoder declares.
- **The optimality gap** of parent spec §9.1 is stated against `Bracket.lower`, which is proven, and not against the best schedule anyone found. At T1 the bracket closed on all sixteen instances in 53 s in total, so the lower bound is available there.
- **Cost.** The gate's QUBOs at twenty bits have 16 to 668 variables; they are objects for enumeration, not for a sampler.
