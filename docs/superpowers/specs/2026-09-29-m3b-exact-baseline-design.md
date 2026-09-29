# M3b — Exact Baseline and the Parent Gate

**Status:** Design approved, pending implementation plan
**Date:** 2026-09-29
**Builds on:** `2026-09-25-m3a-encoding-design.md`, `2026-09-29-m3a-gate-result.md`;
parent spec `2026-09-22-quantum-liquidation-lvar-design.md` §7.1, §7.3, §10.2, §14
**Gate:** decoded QUBO optimum equals the exact optimum on T0, all 16 dial
combinations, under all three encodings
**Hands off to:** M4 — the classical slate, the bench runner and the results store

---

## 1. Purpose

Prove the encoder against exact optima, and deliver the exact classical baseline
every later milestone is measured against.

M3a showed that the QUBO's energy equals the objective at a feasible schedule. It
could not show where the QUBO's *minimum* is: that takes an optimiser, and M3a
shipped none. M3b ships three, each exact, and asks them the same question.

Done when, on all 16 T0 dial combinations:

1. two independent exact methods agree on the optimum of the true objective, and
2. the QUBO's minimum decodes to a feasible schedule whose true objective is that
   optimum, under every encoding.

## 2. Scope

### In scope

- `solvers/base.py` — `Budget`, `Bracket`, `Result`, the `Solver` protocol of
  parent §7.1, at the size M3b's solvers need.
- The lot grid: which values each cell may take, read from the constraint set.
- `brute_force` — the exact optimum by enumeration of feasible schedules.
- `exact_pwl_milp` — the exact optimum through HiGHS, with a proven bracket.
- `qubo_minimum` — the exact minimum of an `EncodedInstance`'s energy.
- The parent §14 gate.
- A `forbidden` import contract for `solvers/`.
- Two measurements that are recorded and not gated: regret at the study's bit
  width, and the MILP bracket at T1.

### Out of scope

- CP-SAT, simulated annealing, tabu, greedy and Almgren-Chriss analytic. M4.
- The bench runner, metrics and the DuckDB store. M4.
- The adaptive penalty strategy and repair strategies. M4, as in M3a.
- `dimod` and any sampler. M5.
- Gurobi.
- Choosing the study's zeta bit budget. M5, as the M3a gate result records.
- An exact baseline at T2 and T3. Parent §5.5 gives those tiers best-known
  values, not proven ones.

## 3. What was measured before designing

All 16 T0 instances of the M3a gate's ladder (seed 2026, `n_raw = 300`,
`n_target = 15`), under all three encodings. The probes were read-only and are
not part of the repository.

| fact | value |
|---|---|
| feasible schedules per instance | 12 to 38,440 |
| D3-off cells where QUBO and true optimum coincide | 8 of 8, all encodings |
| D3-on cells where they coincide, at 8 zeta bits | 3 of 8 |
| regret where they differ, at 8 bits | 0.07 to 1.48, against objectives of 17 to 74 |
| bits outside the gadget, binary | 12 to 28 |
| bits outside the gadget, domain-wall | 12 to 36 |
| bits outside the gadget, one-hot | 23 to 48 |

Under the binary encoding with D1 on, the QUBO minimises the approximand of M3a
§5 and not the power law. Its worst error per cell is 0.02 at T0, on four of the
eight D1 cells and zero on the rest. The optimum was unmoved on all eight; that
is a measurement on this ladder and not a property of the fit.

Every figure in this section is over feasible schedules, with `ζ` scanned near
each loss. The gate of §9 takes the minimum over every state, with the candidate
levels of §7.3.

### 3.1 The parent gate does not hold at the study's bit width

With 8 zeta bits the QUBO's best feasible schedule is not the true optimum on
five of the eight CVaR cells. This is the M3a gate result's open item seen from
the other side: the gadget charges each scenario the distance from its loss to
the grid, that charge differs from schedule to schedule, and at 8 bits it is
larger than the gap between the best schedules.

Three ways to state the gate were put to the owner: equality where the encoding
is exact and a bounded regret where a substitution is declared; equality on the
D3-off cells only; or a wider zeta grid. **The third was chosen.**

### 3.2 The width equality needs

`Y` marks a width at which the QUBO's best feasible schedule is the true
optimum. `T0-D3+D4` has `λ = 0`, carries no CVaR, and agrees at any width.

| instance | gap, best to second | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 to 24 |
|---|---|---|---|---|---|---|---|---|---|---|
| `T0-D3` | 0.1065 | . | Y | Y | Y | Y | Y | Y | Y | Y |
| `T0-D2+D3` | 0.0069 | . | . | Y | Y | . | . | Y | . | Y |
| `T0-D2+D3+D4` | 0.0301 | . | Y | Y | Y | . | Y | Y | Y | Y |
| `T0-D1+D3` | 0.3826 | . | Y | . | Y | Y | Y | Y | Y | Y |
| `T0-D1+D3+D4` | 1.0222 | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `T0-D1+D2+D3` | 2.6136 | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `T0-D1+D2+D3+D4` | 0.0717 | . | . | . | . | Y | Y | Y | Y | Y |

**Agreement is not monotone in the width.** A cell that agrees at 10 bits can
disagree at 12. Below 16 bits a pass is partly where the losses happen to fall
on the grid.

The distortion is the energy less the true objective. What decides the ranking
is its spread across schedules, which halves with each bit:

| width | spread, worst cell | against the smallest gap, 0.0069 |
|---|---|---|
| 8 | 7.3 | 1,060 × |
| 16 | 0.027 | 3.9 × |
| 18 | 0.0067 | 0.97 × |
| 20 | 0.0017 | 0.24 × |

Every cell agrees from 16 bits up. From 18 the spread is below the smallest gap,
so agreement follows from the arithmetic and not from the draw. At 20 it is
below a quarter of it.

At 20 bits, under each encoding's own costs — the approximand included — the
best feasible schedule is the true optimum in all 48 cases.

### 3.3 What a wide gadget costs

| | 8 bits | 20 bits |
|---|---|---|
| variables, `T0-D3`, binary | 272 | 644 |
| dynamic range | 1.5e6 | 2.4e12 |
| round-off in the dense energy, relative | 1e-11 | 1e-8 to 7e-8 |

A 20-bit QUBO is an object for enumeration. No sampler and no QPU could run it:
at 8 bits no CVaR cell fits the hardware already.

## 4. Decisions

| Decision | Rationale |
|---|---|
| **The gate encodes CVaR cells at 20 zeta bits, fixed** | The owner's choice of §3.1, at the smallest width with margin. Sixteen passes today with none. A width derived per instance from its own gap would make the assertion true by construction; a fixed width can fail. |
| **Regret at 8 bits is measured on every CVaR cell and recorded** | The gate shows the encoding converges on the exact optimum. It does not show that the QUBOs the study runs find it, and the number that says how far off they are belongs next to the gate. It is asserted only to sit inside the bound the gadget declares. |
| **The MILP carries one binary per admissible value of each cell** | Any per-cell cost is then exactly linear in the binaries, as under the table encodings of M3a §3. The under- and over-estimators of parent §7.3 coincide on the lot grid, and the bracket's width is the solver's gap and nothing else. |
| **Variance goes through outer approximation** | HiGHS solves linear programs. The variance term is convex quadratic, which parent §7.3 did not address. A tangent cut at an integer point is tight there, so the loop ends at the exact optimum in finitely many rounds. |
| **CVaR in the MILP is continuous Rockafellar-Uryasev** | It is a linear program as written. No grid, no gadget, no resolution: the baseline is exact where the QUBO is not. |
| **The upper bound is `problem.objective.evaluate` at the returned schedule** | One evaluator, as in M3a §5. A MILP that misstated its own objective would report a bracket that does not close. |
| **The minimum is taken over every bit state where the bits outside the gadget number 28 or fewer, and over decodable states otherwise** | `2^48` states cannot be enumerated. The gate records which scope each case had. §8 sets out what that leaves unproven. |
| **The auxiliaries are solved with `cheapest_aux`** | `optimal_aux` minimises the penalty, not the energy. M3a gate result, second review, I2. |
| **Brute force and the MILP share the lot grid and the evaluator, and nothing else** | Two methods that share a search or a bound are one method. |

## 5. `solvers/base.py`

```python
@dataclass(frozen=True, slots=True)
class Budget:
    wall_clock_s: float | None = None
    target: float | None = None        # time-to-target; unused until M4
    max_samples: int | None = None     # unused until M4


@dataclass(frozen=True, slots=True)
class Bracket:
    lower: float    # proven: no feasible schedule costs less
    upper: float    # attained: evaluate() at the returned schedule

    @property
    def width(self) -> float: ...


@dataclass(frozen=True)
class Result:
    solver: str
    status: str                          # OPTIMAL | TIME_LIMIT | INFEASIBLE
    schedule: Schedule | None
    zeta: float | None
    objective: ObjectiveBreakdown | None
    bracket: Bracket | None
    feasibility: FeasibilityReport | None
    timings: Mapping[str, float]
    detail: Mapping[str, Any]


class Solver(Protocol):
    name: str

    def solve(self, instance: Instance, budget: Budget) -> Result: ...
```

`Bracket` refuses `lower > upper` beyond round-off: a lower bound above an
attained value means one of the two is wrong.

Parent §7.1's energy trace is not here. Neither exact solver has one, and M4's
samplers add it with the first solver that does.

## 6. The lot grid — `solvers/classical/lot_grid.py`

```python
@dataclass(frozen=True)
class LotGrid:
    values: tuple[tuple[tuple[int, ...], ...], ...]   # [i][t] -> admissible lots
    cardinality: int | None                           # names per bucket, or None

    def schedules(self) -> Iterator[Schedule]:
        """Every schedule that liquidates in full, on admissible values."""
```

`lot_grid(instance)` reads the constraint descriptors:

| constraint | effect on the grid |
|---|---|
| `max_lots_per_bucket`, `initial_lots` | a cell ranges over `0 .. min(cap, X_i)` |
| `MinParticipation` | values in `1 .. min_lots − 1` are removed |
| `BlockTrades` | a block cell takes `0` or the block size |
| `Cardinality` | not a cell property; carried as `cardinality` |
| `FullLiquidation` | enforced by `schedules()`, row by row |

A constraint type the function does not know raises. A grid that silently
ignored a constraint would make both exact methods agree on the wrong answer.

`encoding/layout.py` derives the same sets for its own purpose. `lot_grid` does
not import it: the gate compares the QUBO with the baseline, and a baseline that
took its search space from the encoder would inherit the encoder's mistakes.
A test asserts the two agree cell by cell on all 64 instances.

`problem.feasibility.classify` stays the only verdict on feasibility. `schedules()`
yields candidates; every schedule either method returns is classified.

## 7. The exact methods

### 7.1 `brute_force`

Enumerates `LotGrid.schedules()`, drops those `classify` rejects, and returns
the minimum of the objective.

```python
def brute_force(
    instance: Instance, *, cell_cost: CellCost | None = None
) -> Result
```

`CellCost` is `(i, t, lots) -> float`, the spread and temporary impact of one
cell. `None` means the true cost, from `evaluate`. Passing the table of an
`ApproximandReport` gives the optimum of the function the encoder declares it
implements, which M3a's hand-off asks for and which M4's comparison at T1 under
the binary encoding will need.

Under CVaR the schedule's `ζ` is `optimal_zeta`. The bracket has zero width:
both ends are the same evaluation.

It refuses a grid of more than `10^7` schedules rather than run for hours.
T0's largest has `4^12 ≈ 1.7 × 10^7` before full liquidation and 38,440 after.

### 7.2 `exact_pwl_milp`

Through `scipy.optimize.milp`, which is HiGHS. No new dependency.

**Variables.** `y[i,t,k] ∈ {0,1}` for each admissible value `k` of cell `(i,t)`;
under CVaR a free continuous `ζ` and one `u_s ≥ 0` per scenario; under variance
one `θ ≥ 0`.

**Constraints.**

| | |
|---|---|
| one value per cell | `Σ_k y[i,t,k] = 1` |
| full liquidation | `Σ_t Σ_k k · y[i,t,k] = X_i` |
| cardinality | `Σ_i (1 − y[i,t,0]) ≤ K` for each bucket |
| Rockafellar-Uryasev | `u_s ≥ L_s(y) − ζ`, with `L_s` linear in the holdings |

Minimum participation and blocks are already in the grid.

**Objective.**

| term | form |
|---|---|
| spread, temporary impact | `Σ c[i,t,k] · y[i,t,k]`, exact for any cost table |
| permanent impact | a constant |
| CVaR | `λ (ζ + Σ_s w_s u_s / (1 − α))` |
| variance | `θ`, bounded below by cuts |

Permanent impact is `(γ σ p / V) Σ_t (Σ_{s<t} q_s + q_t / 2) q_t`, and the sum
is `(Σ_t q_t)² / 2`. Under full liquidation that is a constant of the position.
The MILP carries it as one, and a test checks the identity against `evaluate`
on every feasible schedule of a T0 instance.

**Outer approximation.** With `v(x) = λ Σ_t h_t' Σ h_t`:

1. Solve the MILP. Read the schedule `x̂` and the value `θ̂`.
2. Evaluate `v(x̂)` exactly.
3. If `v(x̂) − θ̂` is within tolerance, stop.
4. Otherwise add `θ ≥ v(x̂) + ∇v(x̂)' (x − x̂)` and return to 1.

`v` is convex, so every cut is valid everywhere and the MILP's value is a lower
bound at every round. The cut at `x̂` is tight at `x̂`, so no schedule is
returned twice with a gap, and the grid is finite.

**The bracket.** `lower` is the MILP's dual bound at the last round. `upper` is
`evaluate` at the schedule it returned. `mip_rel_gap` is set to zero; the
default of `1e-4` would leave a bracket wider than the gap between schedules
on `T0-D2+D3`.

**Budget.** `wall_clock_s` is split across rounds. On running out, the result
carries `TIME_LIMIT`, the best schedule found and the bracket as it stands.

### 7.3 `qubo_minimum`

```python
@dataclass(frozen=True)
class QuboMinimum:
    bits: np.ndarray          # the full vector, auxiliaries included
    energy: float
    scope: str                # EVERY_STATE | DECODABLE
    n_states: int             # how many were examined
    n_ties: int               # states within round-off of the minimum


def qubo_minimum(encoded: EncodedInstance) -> QuboMinimum
```

**Scope.** Let `n` be the bits outside the gadget: lot bits and cardinality
slack.

- `n ≤ 28`: every one of the `2^n` states. `EVERY_STATE`.
- `n > 28`: every state that decodes — each cell on a value the layout admits,
  feasible or not, with every setting of the slack bits. `DECODABLE`.

**Without a gadget** the energy of a state is `z' Q z`. The bits are split into
two halves so that one half's energy is computed once and the cross term is a
matrix product, which is what makes `2^28` a matter of minutes.

**With a gadget** the scenarios separate once the schedule and `ζ` are fixed, as
in `tests/encoding/landscape.py`. For each state:

1. The losses `L_s` follow from the holdings.
2. For each candidate level `j` of the `ζ` grid, the auxiliaries are
   `cheapest_aux` in closed form, vectorised.
3. The state's energy is the least over the candidates.

**The candidate levels.** A scan of `2^20` levels per state is not affordable,
and a scan of the levels nearest each loss is not a proof. Write the gadget's
charge at level `j` as `G(j) = R(j) + e(j)`, where `R(j)` is `λ` times
continuous Rockafellar-Uryasev at `ζ_j`. With `c_s = λ w_s δ / (1 − α)` the cost
of one step of `u_s`, and `P` the gadget's strength:

```
−A ≤ e(j) ≤ B,    A = Σ_s c_s² / (4 P δ²),    B = Σ_s (c_s / 2 + P δ² / 4)
```

The lower bound relaxes the step count to a real; the upper puts each `u_s` on
the nearest grid point. `R` is convex in `j`, so the levels with
`R(j) ≤ min R + A + B` form an interval, and no level outside it can hold the
minimum of `G`. That interval is the candidate set. Both `A` and `B` scale with
`δ`, as does the slope of `R`, so its length does not grow with the width.

**Pruning.** CVaR is at least the mean loss, so a state's energy is at least

```
costs + hard penalties + λ Σ_s w_s L_s − A
```

which is linear in the state and costs one matrix product. Only states whose
bound is below the incumbent go through the scan. The incumbent starts at the
energy of the instance's witness.

The vectorised closed form is a second implementation of `cheapest_aux`. The
winning state is rebuilt through the production method and its energy checked
against `Qubo.energy`, as `Landscape.state` does.

**Ties.** `n_ties` counts states within round-off of the minimum. The gate
asserts on the decoded schedule's objective value, never on which schedule it
is, so a tie between two schedules of equal cost cannot fail it.

## 8. What the scope leaves unproven

Under the binary encoding every case has 28 bits or fewer outside the gadget.
**Every one of the 16 dial combinations is therefore covered over every state
under at least one encoding.**

| encoding | `EVERY_STATE` | `DECODABLE` |
|---|---|---|
| binary | 16 | 0 |
| domain-wall | 10 | 6 |
| one-hot | 5 | 11 |

In the 17 `DECODABLE` cases the states not examined are those `decode` rejects:
two one-hot bits set in a cell, a hole in a domain wall. Such a state is not a
schedule, and nothing in this milestone shows that none of them has an energy
below the minimum. The validity penalty's strength is derived to prevent it;
the derivation is what M3a's enumeration tests check on small instances, and
what the 15 `EVERY_STATE` cases under the table encodings check at T0.

Closing it would take a branch and bound over cells, or a bound on what a
rejected cell can gain. Neither is here.

## 9. The gate — `tests/solvers/test_m3b_gate.py`

The ladder is M3a's: `generate_ladder(_params_for, seed=2026, include_cvar=True,
n_raw=300, n_target=15)`, T0 only. The ladder's fixture moves to a shared
`tests/ladder.py` so both gates build the same instances.

As in M3a, one pass collects findings per check and the named tests assert on
them.

**For each of the 16 instances:**

1. `brute_force` and `exact_pwl_milp` both return `OPTIMAL`, and both schedules
   classify as feasible.
2. Their objective values are equal, to `1e-9` relative. Both are `evaluate` at
   a schedule, so this is round-off and not a solver tolerance.
3. The MILP's bracket closes: `width ≤ 1e-6 · max(1, |upper|)`. This one *is*
   the solver's tolerance — HiGHS works to `1e-7` in feasibility — and at an
   objective of 30 it is two orders below the smallest gap between schedules.
4. Permanent impact at the returned schedule equals the MILP's constant.

**For each instance under each of the three encodings,** with CVaR cells encoded
at `GATE_ZETA_BITS = 20` and the rest as they are:

5. The minimum state decodes as `feasible`.
6. **The decoded schedule's true objective equals the exact optimum**, to `1e-9`
   relative. This is parent §14.
7. The closed-form energy of the winning state equals `Qubo.energy` of the same
   vector, to `1e-6` relative. At a dynamic range of `2.4e12` the dense product
   carries `7e-8`; the bound is that, with room, and is two orders below the
   smallest gap.
8. The scope and the number of states examined are recorded, and match §8's
   table.

**Recorded, and asserted only as stated:**

9. At 8 zeta bits, on every CVaR cell under every encoding: the regret of the
   QUBO's minimum. Asserted non-negative and at most `2 · resolution +
   cost_gap`. The energy of a schedule is at most its true objective plus
   `resolution + cost_gap`, which the canonical state attains, and at least its
   true objective less `A` of §7.3, which is below `cost_gap`; the regret of
   the energy's minimum is bounded by the sum.
10. On the 16 T1 instances: `exact_pwl_milp` with a budget of 120 s each.
    Asserted to return `OPTIMAL` or `TIME_LIMIT` with `lower ≤ upper`. The
    statuses, brackets and wall times go into the result document. This is
    `tests/solvers/test_t1_bracket.py`, under a `slow` marker the default run
    deselects: sixteen budgets of 120 s are not part of the gate's wall time.

**Mutation checks.** Each of these is put in by monkeypatch and the gate is
watched failing, as M3a's second review required. The second column is the
expectation; the result document records what each one did.

| defect | expected to fail |
|---|---|
| hard strengths scaled down until an infeasible state wins | check 5 |
| the vectorised closed form uses `optimal_aux`'s rule | check 7 |
| CVaR cells encoded at 8 bits | check 6, on five cells |
| cardinality dropped from the MILP | check 1 |
| a cut with the wrong sign | check 2 or 3 |
| the MILP's cost table shifted on one cell | check 2 |

## 10. Deviations from the parent spec

| § | parent | M3b | why |
|---|---|---|---|
| 14, 10.2 | QUBO optimum equals MILP optimum | the same, with CVaR cells at 20 zeta bits, not the study's 8 | §3.1. At 8 bits it is false on five cells. |
| 7.3 | under- and over-estimators bracket the optimum | one exact table on the lot grid | The grid is integer and small; the estimators coincide on it. |
| 7.3 | piecewise-linearise and solve | and outer approximation for variance | HiGHS is linear. |
| 12 | `solvers/` imports only `Instance` | also `objective`, `feasibility`, `schedule`, `risk`, `scenarios` and the constraint descriptors | M2a §3 already has the MILP expressing constraints natively and calling `classify`. A `forbidden` contract bars `generator`, `from_bundle` and `liquidation`. |
| 12 | `encoding/pwl.py`, `encoding/miqp.py` | neither exists | The linearisation lives with the solver that uses it. |

## 11. File structure

| File | Responsibility |
|---|---|
| `solvers/base.py` | `Budget`, `Bracket`, `Result`, `Solver` |
| `solvers/classical/lot_grid.py` | admissible values per cell; schedule enumeration |
| `solvers/classical/brute_force.py` | the optimum by enumeration |
| `solvers/classical/exact_pwl_milp.py` | the MILP and the outer-approximation loop |
| `solvers/classical/qubo_minimum.py` | the exact minimum of an encoded instance |
| `tests/ladder.py` | the gate ladder, shared by M3a and M3b |
| `tests/solvers/test_m3b_gate.py` | the gate |
| `tests/solvers/test_t1_bracket.py` | the T1 measurement, marked `slow` |
| `pyproject.toml` | the `forbidden` import contract; the `slow` marker |

`solvers/` sits above `encoding/` in the layer contract. `qubo_minimum` imports
`encoding/`; `lot_grid`, `brute_force` and `exact_pwl_milp` do not, and a test
asserts it.

## 12. Risks

| Risk | Mitigation |
|---|---|
| The gate takes too long | The split enumeration and the pruning of §7.3. The plan carries a target of 30 minutes for the whole gate on the 64 GB machine, and measures before it optimises. |
| The MILP does not finish at T1 | Recorded, not gated. If it does not, M4 inherits a measured fact instead of an assumption. |
| The candidate interval is long where `R` is flat | Its length is reported per case. A flat `R` needs the tail weight to equal `1 − α` exactly, which fifteen clustered weights do not give. |
| A future ladder has a smaller gap than 20 bits resolves | The gate would fail on check 6 and say which cell. The width is a named constant. |
| Round-off at 20 bits hides a real discrepancy | Check 7's bound is three orders below the smallest gap, and checks 2 and 6 compare `evaluate` with `evaluate`. |

## 13. What M4 picks up

`Solver`, `Result` and `Bracket` are the interface. M4 adds the rest of the
classical slate behind the same protocol, the bench runner and the store.

Three things M4 should not have to rediscover:

- The study's CVaR QUBOs at 8 bits do not have the true optimum as their
  minimum on five of eight T0 cells. A sampler that returns the QUBO's minimum
  has not failed when its schedule is not the baseline's; check 9's regret is
  the number to compare against.
- Under the binary encoding with D1 on, the QUBO minimises the approximand.
  `brute_force` and `exact_pwl_milp` take its cost table for that comparison.
- The optimality gap of parent §9.1 is stated against `Bracket.lower`, which is
  proven, and not against the best schedule anyone found.
