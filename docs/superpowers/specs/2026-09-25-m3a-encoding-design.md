# M3a — QUBO Encoding, Penalty Derivation and Decode

**Status:** Design approved, pending implementation plan
**Date:** 2026-09-25
**Builds on:** `2026-09-24-m2a-problem-model-design.md`, `2026-09-24-m2b-cvar-design.md`;
parent spec `2026-09-22-quantum-liquidation-lvar-design.md` §3.2, §6
**Gate:** exact energy equality on 64 instances × 3 encodings
**Hands off to:** M3b — the exact PWL-MILP baseline and the parent spec §14 gate

---

## 1. Purpose

Turn an `Instance` into a QUBO, and prove the QUBO *is* the objective.

M3a ships no solver, and its gate needs none. If the QUBO's energy at an
encoded feasible schedule equals `objective.evaluate` on that schedule, the
encoder is correct — and it is shown correct *before* any optimiser exists to
confound the question. M3b then measures optima against a proven encoder
rather than debugging two unknowns at once.

## 2. Scope

### In scope

- `VarRegistry` — every binary carries a typed, printable name.
- The three integer encodings of §6.1: binary/log, one-hot, domain-wall.
- A layout planner: how each `x[i,t]` is represented depends on the constraints.
- `Qubo` — an owned type, dense upper-triangular, quadratic by construction.
- Objective expansion, term by term, mirroring `ObjectiveBreakdown`.
- Hard constraints as penalties (§6.2) and derived penalty bounds (§6.3).
- The dynamic-range diagnostic (§6.4).
- Decode and feasibility classification (§6.5).
- The variable-budget table across tiers × dials × encodings.

### Out of scope

- Any solver, and therefore the **adaptive** penalty strategy of §6.3. That
  strategy is defined as "increase penalties until the feasibility rate on a
  reference sampler crosses a threshold", and no sampler exists until M4.
  M3a ships derived bounds; adaptive search lands with M4's `neal`.
- `dimod`, and any conversion to it. That belongs to `solvers/quantum/` at M5.
- Repair strategies. Classification is pure logic and ships here; repair acts
  on returned samples, which arrive at M4.
- PWL linearisation and MILP. Those are M3b.

## 3. Decisions

| Decision | Rationale |
|---|---|
| **Own `Qubo` type; no `dimod` inside `encoding/`** | Every formulation below is quadratic by construction, so §6.2's `dimod.make_quadratic` would be dead code. The gate's arithmetic then rides on our own accumulation order rather than a third party's. Conversion lands at M5, where a sampler actually needs it. Deviation from §6.2, recorded in §11. |
| **The gate proves the QUBO equals the function the encoder *declares it implements*, exactly. The gap between that function and the continuous ideal is a separate, reported, separately-asserted quantity.** | Two objective pieces are not exactly representable: concave impact under log encoding (§5), and `max(L_s − ζ, 0)` under any encoding (§7). Folding either into a tolerance on the correctness gate is exactly the slack M2b §3 refused. Splitting them into **two exact assertions** keeps the correctness claim tolerance-free and turns the discretisation into a reported number, as `ReductionReport` already is for scenario reduction. |
| **Per-term QUBOs, not one blob** | `objective.py` already returns its terms separately, for the stated reason that "a single scalar says nothing about which term is wrong". The encoder mirrors that: `encode_objective` returns a `QuboBreakdown` with one `Qubo` per term, and the gate compares field by field. |
| **Single-cell costs use an exact cost table under one-hot and domain-wall** | Exactly one binary is set per cell, so *any* `f: {0..M} → ℝ` is exactly linear in those bits. This is what makes D1 free under both encodings, and it uses fewer couplers than squaring the value form. |
| **Binary/log uses bounded-coefficient expansion, not powers of two** | `x = Σ 2^b z_b` over-represents whenever `M+1` is not a power of two, making `x > M` reachable and requiring a range penalty to tune. Coefficients `1, 2, …, 2^{k−2}, M − 2^{k−1} + 1` cover exactly `[0, M]` and nothing else, so no range penalty exists. Cost: the map is not injective, so `encode` picks a canonical bitstring and only `decode ∘ encode` round-trips. |
| **Derived penalty bounds only** | See scope. |

## 4. Representability, term by term

This table is the spine of the design. `x = x[i,t]` is in lots and
`q = x · lot_size` in shares, matching `objective.evaluate`.

| Objective term | Algebraic form in `x` | log | one-hot / domain-wall |
|---|---|---|---|
| Spread | `half_spread · q` — **linear** | exact | exact |
| Temporary, D1 **off** (Almgren–Chriss) | `(η σ p / 2V) · q²` — **quadratic** | exact | exact (cost table) |
| Temporary, D1 **on** (power law) | `[Y σ /(δ+1)] · (q/V)^δ · q · p` — **non-polynomial** | **approximand, §5** | exact (cost table) |
| Permanent | `(γ σ p / V) · (Σ_{s<t} q_s + q_t/2) · q_t` — **quadratic** | exact | exact |
| Risk, variance | `λ Σ_t h_t' Σ_price h_t`, with `h` linear in `x` — **quadratic** | exact | exact |
| Risk, CVaR | `λ (ζ + 1/(1−α) Σ_s w_s max(L_s − ζ, 0))`, `L_s` **linear** in `h` | **gadget, §7** | **gadget, §7** |

Only two cells are inexact, and both are handled by the same pattern: a named
substitution, an exact gate against that substitution, and a reported bound.

`evaluate` skips cells where `traded == 0`. Every term above vanishes at
`q = 0`, so that skip introduces no discrepancy the encoder must reproduce.

## 5. The D1 × log approximand

For each cell, fit a quadratic `q̂(x) = b·x + c·x²` to the true per-cell cost

```
f(x) = [Y_i · σ_i / (δ_i + 1)] · (x·lot_i / V_i)^{δ_i} · x · lot_i · p_i
```

over the integer grid `x ∈ {0..M}`, with no constant term: `f(0) = 0` exactly,
and a constant would shift every energy including the empty schedule's.

The domain is finite and small, so the error is not bounded asymptotically —
it is **computed exactly** over the grid:

```python
@dataclass(frozen=True)
class ApproximandReport:
    coefficients: dict[tuple[int, int], tuple[float, float]]   # (i, t) -> (b, c)
    max_abs_error: float        # max over cells and lot values of |f - q̂|
    max_rel_error: float        # the same, relative to f, skipping f == 0
    worst_cell: tuple[int, int]

    def temporary_cost(self, schedule: Schedule) -> float:
        """The substituted term, summed over cells. The gate's expected value."""
```

`encode_objective` returns `None` here for one-hot and domain-wall. The
**absence** of a report is the claim that the encoding was exact, and a test
pins that absence rather than letting it pass unnoticed.

The gate's expected value reuses `evaluate` with one field swapped:

```python
expected = dataclasses.replace(
    evaluate(instance, schedule, zeta=z),
    temporary_impact_cost=report.temporary_cost(schedule),
)
```

There is deliberately no parallel evaluator, so nothing can drift out of sync
with `problem/objective.py`.

## 6. Variable layout — `encoding/layout.py`

The representation of `x[i,t]` is **not** a property of the encoding alone.
§6.2's min-participation formulation is a re-expansion of `x`, not a penalty
bolted onto a fixed one, and a D4 block cell collapses to a single bit. The
planner therefore reads `instance.constraints` first and emits one `CellPlan`
per `(i,t)`:

| Situation | `CellPlan` | Bits | Penalty |
|---|---|---|---|
| Plain | `x ∈ [0, M]`, encoded | `n_bits(M)` | validity only (one-hot / DW) |
| D4 block on this cell | `x = lots_k · b_k` | **1** | none — exactness is structural |
| D2, log encoding | `x = m_i·y + Σ_b c_b z_b`, `z` over `[0, M − m_i]` | `1 + n_bits(M − m_i)` | `Σ_b z_b (1 − y)` (§6.2) |
| D2, one-hot / DW | cost table over `{0} ∪ {m_i..M}` | `|{0} ∪ {m_i..M}|` | validity only |

**D2 is free under one-hot and domain-wall** — the disallowed values are
simply absent from the table — and costs a penalty only under log. That
asymmetry is a result for the §6.1 encoding study, not an implementation
detail, and belongs in the results row.

The generator attaches `Cardinality` **only** alongside `MinParticipation`
(`generator.py:235-247`), so the `y[i,t]` indicator cardinality needs always
exists under D2. Under one-hot it is the linear form `1 − z_{v=0}`; under
domain-wall it is `z_1`. No extra variable and no gadget in either case.

Cardinality itself is §6.2's binary-slack square:
`P_card · Σ_t (Σ_i y[i,t] + Σ_j 2^j s[t,j] − K)²`.

Full liquidation is `P_full · Σ_i (Σ_t x[i,t] − X_i)²`.

## 7. The CVaR gadget, and why it is a declared substitution

Rockafellar–Uryasev requires `u_s ≥ L_s − ζ` with `u_s` minimised. Remaining
quadratic requires an equality plus a slack:

```
cost:     λ ( ζ + 1/(1−α) · Σ_s w_s · u_s )
penalty:  P_cvar · Σ_s ( u_s − L_s + ζ − g_s )²      u_s, g_s ≥ 0, binary-expanded
```

`L_s` is linear in `x` with **real** coefficients, while `ζ`, `u_s` and `g_s`
live on finite binary grids. So `u_s − g_s = L_s − ζ` cannot be satisfied
exactly — for the same reason `ζ` cannot land exactly on the empirical
quantile, which M2b §3 already named and accepted as the encoding's
discretisation cost.

The design therefore declares precisely what it encodes:

```python
@dataclass(frozen=True)
class CVaRGadget:
    zeta_grid: Grid
    u_grid: Grid
    g_grid: Grid
    penalty: float

    def encoded_cvar(self, instance, schedule, *, zeta, aux) -> float:
        """The exact algebraic value the QUBO implements, penalty included."""

    def optimal_aux(self, instance, schedule, *, zeta) -> Aux:
        """Penalty-minimising u, g on their grids, for a given x and zeta."""

    @property
    def resolution(self) -> float:
        """Reported worst-case gap to continuous Rockafellar-Uryasev."""
```

`ζ`'s range is derived from the **instance**, never from a schedule, so it is
deterministic and independent of which schedule is being scored:

```
L_s^max = Σ_{t,i} X_i · max(0, −lot_i · p_i · returns[s,t,i])
L_s^min = Σ_{t,i} X_i · min(0, −lot_i · p_i · returns[s,t,i])
```

`h[i,t] ∈ [0, X_i]`, so the true `ζ` optimum always lies inside
`[min_s L_s^min, max_s L_s^max]`.

**Two assertions, both exact:**

1. `Qubo.energy(bits) == gadget.encoded_cvar(...)` — the encoder is correct.
2. `|encoded_cvar(at optimal_aux) − evaluate(..., zeta=ζ_snapped)| ≤ gadget.resolution`
   — the discretisation is no worse than it claims to be.

Neither is a tolerance on correctness. `resolution` joins `wasserstein` and
`cvar_discrepancy` on the results row.

## 8. Penalty derivation — `encoding/penalty.py`

Derived bounds, §6.3's first strategy. For each constraint, `P` must exceed
the largest objective improvement a one-unit violation can buy:

```
P_c  >  max_gain_per_unit(instance) / min_penalty_increment(c)
```

`max_gain_per_unit` comes from problem structure — the largest per-lot
objective coefficient anywhere in the instance, which is readable off the
expanded objective QUBO itself. `min_penalty_increment` is `1` for the
integer-residual squares (full liquidation, cardinality) and is derived per
constraint otherwise. A safety factor is applied, named, and recorded on the
results row alongside the strategy name.

Dispatch is on **type**, never on an attribute name — M2b's lesson, applied to
constraints this time, with the same style of impostor test guarding it.

## 9. Dynamic range — `encoding/diagnostics.py`

```
dynamic_range_ratio = max|coefficient| / min_meaningful_gap
```

Recorded on every encoded instance. §6.4 is explicit that "no result is
interpretable without it". M3a computes and reports it; the runner's warning
threshold belongs to M4.

## 10. Decode — `encoding/decode.py`

`decode(qubo, bits) -> DecodedSample(schedule, zeta, classification)`, where
`classification ∈ {feasible, repairable, rejected}` comes from
`problem.feasibility.classify`. Repair strategies are M4's.

Round-trip property: `decode(encode(schedule)) == schedule` for every feasible
schedule under all three encodings. The reverse does **not** hold under
bounded-coefficient log expansion, by design (§3), and a test pins that
asymmetry so nobody later "fixes" it into a false claim.

## 11. Deviations from the parent spec

| § | Spec says | M3a does | Why |
|---|---|---|---|
| 6.2 | quadratise residual higher-order terms via `dimod.make_quadratic` | owns `Qubo`, asserts quadratic by construction | nothing higher-order arises from the formulations in §4–§7; see §3 |
| 6.3 | two penalty strategies | derived bounds only | adaptive needs a reference sampler, which arrives at M4 |
| 5.4 | CVaR costs "one auxiliary variable per scenario" | two binary-expanded auxiliaries, `u_s` and `g_s` | a one-sided constraint cannot stay quadratic without a slack; §7 |
| 6.1 | three encodings, all first-class | all three, but D1 is approximated under log | `x^δ` has no exact quadratic form; §5 |

## 12. The M3a gate — `tests/encoding/test_m3a_gate.py`

**Tier coverage.** A dense QUBO matrix is `n²` doubles. At T3 the one-hot
layout names roughly `30 × 12 × 16 ≈ 5,760` bits, which is 265 MB *per term* —
so materialising the breakdown there would need gigabytes, and the gate could
not run. Energy equality therefore covers **T0, T1 and T2 under all three
encodings** (48 instances × 3) and **T3 under the log encoding**, whose
`30 × 12 × 4` layout stays tractable. T3 under one-hot and domain-wall is
covered by §13's variable budget, which counts keys rather than building
matrices — and which is the artifact showing those cells cannot reach the QPU
in any case. Every one of the 64 instances is therefore encoded and gated;
what varies is how many encodings can be materialised at the largest tier.

For every instance × materialisable encoding:

1. `encode(instance, encoding)` succeeds and emits a `Qubo` that asserts
   quadratic.
2. Term by term, `Qubo.energy(encode_schedule(witness)) == expected`, where
   `expected` is `evaluate(instance, witness, zeta=ζ_snapped)` with the
   temporary-impact field swapped per §5 when an `ApproximandReport` is
   present. Equality is exact up to floating-point round-off only — a relative
   tolerance at the scale of double-precision accumulation, never a modelling
   tolerance.
3. The penalty block contributes **zero** at the feasible witness — exactly
   zero algebraically, and in floating point zero to within a bound scaled to
   the penalty strength. `Qubo.energy` evaluates the residual as a dense
   `z'Qz` product, so double-precision cancellation leaves ~1e-13 against
   strengths of ~24–62 (~5e-15 relative). This is not a modelling tolerance:
   `derive_penalties` floors every strength at 1.0 and every residual is an
   integer square, so the cheapest genuine breach costs a whole strength —
   nine orders of magnitude above the bound asserted.
4. Perturbing the witness by one lot into infeasibility **increases** the total
   energy, on the instances where a one-lot edit is representable — D2 and D4
   restrict a cell's admissible values, so the perturbed schedule need not be
   encodable there. This is the testable form of the penalty guarantee; the
   stronger claim — that the QUBO's global minimum is feasible — requires
   enumeration and is M3b's to make.
5. `decode(encode(witness)) == witness`.
6. `ApproximandReport.max_abs_error` and `CVaRGadget.resolution` are finite,
   reported, and within their asserted bounds.
7. The variable budget is recorded.

## 13. The variable budget — a deliverable, not a diagnostic

§3.2 caps dense embedding at roughly 230 variables. §7 makes CVaR dominate the
count: `S` scenarios × two binary expansions each, against roughly
`N · T · n_bits` for the schedule itself. At T0 the auxiliaries outnumber the
decision variables by an order of magnitude.

M3a therefore emits a table — tier × dials × encoding → variable count — that
answers a question M5 would otherwise discover on the QPU: **which cells of the
study actually fit Advantage2.** The expectation going in is that D3-on cells
do not fit at T1 or T2 under any encoding, which would make `n_zeta_bits` and
the scenario count the levers to pull. That is worth knowing now rather than in
six weeks.

It also shapes M3b. With D3 on, T0's QUBO has hundreds of variables and cannot
be brute-forced. But `optimal_aux` is analytic for fixed `x` and `ζ`, so M3b
enumerates `x` over the T0 lot grid × `ζ` over its grid and solves the
auxiliaries in closed form. The §14 gate stays reachable.

## 14. File structure

| File | Responsibility |
|---|---|
| `encoding/registry.py` | `VarRegistry` — typed keys, stable order |
| `encoding/integer.py` | the three `IntegerEncoding` strategies |
| `encoding/layout.py` | `CellPlan` per `(i,t)`, derived from the constraint set |
| `encoding/qubo.py` | `Qubo`, `QuboBuilder`, `energy` |
| `encoding/expand.py` | objective → `QuboBreakdown`; `ApproximandReport` |
| `encoding/cvar.py` | `CVaRGadget` and the ζ / u / g grids |
| `encoding/penalty.py` | constraint quadratisation and derived bounds |
| `encoding/encode.py` | `EncodedInstance`, `encode` — the orchestrator |
| `encoding/decode.py` | bits → `Schedule`, classification |
| `encoding/diagnostics.py` | dynamic range, variable budget |
| `tests/encoding/test_m3a_gate.py` | the 64 × 3 gate |

`encoding/` sits directly above `problem/` in the §12 layer contract, so it may
import `problem/`, `micro/`, `data/` and `core/`, and must never be imported by
any of them.

## 15. What M3b picks up

`Qubo` and `decode` are the whole interface. M3b adds `solvers/classical/`:
brute force at T0, the PWL under- and over-estimator MILPs through
`scipy.optimize.milp` (HiGHS — already a dependency, so no new one), the
rigorous optimality bracket of §7.3, and the parent spec §14 gate.
