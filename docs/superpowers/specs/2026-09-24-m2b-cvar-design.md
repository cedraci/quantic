# M2b — CVaR / L-VaR Risk and the Scenario Machinery (dial D3)

**Status:** Design approved, pending implementation plan
**Date:** 2026-09-24
**Builds on:** `2026-09-24-m2a-problem-model-design.md`; parent spec `2026-09-22-quantum-liquidation-lvar-design.md` §5.2, §5.4
**Completes:** M2 — `Dials.combinations(include_cvar=True)` and the 64-instance gate

---

## 1. Purpose

Add the fourth hardness dial, **D3**: replace the variance risk term with CVaR
(Rockafellar–Uryasev), backed by generated price scenarios reduced to a
tractable weighted set with its reduction error reported.

M2b is done when **16 dial combinations × 4 tiers = 64 instances** generate,
each with a provably feasible witness and a finite objective — the parent
spec §14 criterion in full.

## 2. Scope

### In scope

- `ScenarioSet` — weighted price-shock scenarios and the loss they induce.
- Scenario generation from `MarketParams.covariance`, and reduction to 20–30
  weighted representatives with a reported Wasserstein error.
- `CVaRRisk` as the second `RiskSpec` variant.
- The CVaR branch in `objective.evaluate`, in Rockafellar–Uryasev form.
- The `isinstance` dispatch fix in `objective._risk_cost` (see §7).
- Generator support so `include_cvar=True` produces valid instances.

### Out of scope

- Any QUBO or binary expansion of `zeta`. That is M3.
- Stressing liquidity. Scenarios perturb prices only — see §3, decision 2.

## 3. Decisions

| Decision | Rationale |
|---|---|
| **The evaluator uses Rockafellar–Uryasev with an explicit `zeta`**, not the sorted closed form | M3's load-bearing gate asserts that QUBO energy equals this function's output. M3 encodes a binary-expanded `zeta` plus one auxiliary per scenario. If the evaluator used the closed form, the QUBO could match only when `zeta`'s expansion landed exactly on the empirical quantile, so the gate would need a tolerance — and a tolerance on the gate that is supposed to *prove* the encoding correct is exactly the slack that hides real bugs. |
| **Scenarios perturb prices only; `L_s` is the mark-to-market loss on remaining holdings** | The variance term is already the variance of precisely this quantity, so D3 on-versus-off becomes a clean swap of how one loss distribution is summarised, and D3 changes exactly one objective term. That is what lets the §9.2 screening design attribute a crossover shift to D3 rather than to a D1×D3 interaction. It remains liquidity-adjusted in the Almgren–Chriss sense because `h_t` depends on how fast the schedule liquidates. |
| **`zeta` is a required keyword on `evaluate`, never a field on `Instance`** | M2a's central structural claim is that adding this dial reshapes no other type. A `witness_zeta` field would break that. Requiring it explicitly — rather than defaulting to the optimum — also means every CVaR call site states which `zeta` it is scoring at, which is the difference the gate is measuring. |
| **Reduction keeps actual scenarios, not centroids** | A centroid is an averaged path: smoother than any real path, with a thinner tail. CVaR is a tail statistic, so averaging is precisely the wrong operation. Fast-forward selection keeps real paths and directly minimises the Wasserstein distance the spec asks us to report. |

## 4. `ScenarioSet` — `problem/scenarios.py`

```python
@dataclass(frozen=True)
class ScenarioSet:
    returns: np.ndarray     # (S, T, N) fractional price shocks per bucket per asset
    weights: np.ndarray     # (S,) probabilities, summing to 1
    seed: int
    reduction: ReductionReport | None = None    # None if unreduced

    @property
    def n_scenarios(self) -> int
    @property
    def n_buckets(self) -> int
    @property
    def n_assets(self) -> int
    def losses(self, instance: Instance, schedule: Schedule) -> np.ndarray   # (S,)
```

`__post_init__` validates: shapes agree; weights are non-negative, finite and
sum to 1 within tolerance; returns are finite.

### 4.1 The loss

```
L_s(x) = - sum over t, i of  h[i,t] * lot_size[i] * price[i] * returns[s,t,i]
```

`h[i,t]` is the holding in **lots** remaining after bucket `t`
(`Instance.remaining_lots`), converted to shares by `lot_size` exactly as the
other terms do. The leading minus makes `L_s` a **loss**: a negative return on
a long remaining position is a positive loss. The result is currency, matching
every other term in `ObjectiveBreakdown`.

### 4.2 Correspondence with the variance term

With shocks drawn independently across buckets,

```
Var(L_s)  =  sum over t of  h_t' Sigma_price h_t
```

which is exactly the variance term's quadratic form. This is not a
coincidence to be noted in passing — it is the load-bearing check that the
scenario machinery and the variance term describe the same risk, and §10
makes it a test.

## 5. Generation — `generate_scenarios`

```python
def generate_scenarios(
    params: MarketParams, n_buckets: int, *, seed: int, n_raw: int = 2000
) -> ScenarioSet
```

Draws `returns[s,t,:] ~ N(0, params.covariance)` independently across `s` and
`t`, seeded explicitly. `params.covariance` is already at the bucket horizon
(M2a §4), so no rescaling happens here — a second rescale would be the same
class of silent error the M2a units work exists to prevent.

Equal weights `1/n_raw` before reduction.

## 6. Reduction — `reduce_scenarios`

```python
def reduce_scenarios(raw: ScenarioSet, *, n_target: int = 25) -> ScenarioSet
```

**Dupačová fast-forward selection.** Each scenario is flattened to a vector in
`R^(T*N)`. Greedily select the scenario that most reduces

```
D(selected) = sum over s of  p_s * min over j in selected of  ||xi_s - xi_j||
```

until `n_target` are chosen. Every raw scenario is then assigned to its
nearest selected representative, and each representative's weight is the total
probability assigned to it.

`D(selected)` at termination **is** the Wasserstein-1 (Kantorovich) distance
between the raw and reduced distributions under that assignment, so the error
the parent spec §5.4 demands comes out of the algorithm rather than needing a
separate optimal-transport solve.

```python
@dataclass(frozen=True)
class ReductionReport:
    n_raw: int
    n_reduced: int
    wasserstein: float
    cvar_raw: float
    cvar_reduced: float
    cvar_discrepancy: float      # cvar_reduced - cvar_raw
```

The CVaR figures are evaluated at the instance's witness, at the same
`alpha`, on the raw and reduced sets respectively. Reduction error is thus
reported both in the abstract (Wasserstein) and in the units the study
actually cares about (currency of CVaR).

**Cost.** The pairwise distance matrix is `n_raw**2` entries; selection is
`n_target` vectorised passes over it. At `n_raw=2000`, `n_target=25`, T3's
`T*N = 360` dimensions, this is seconds, not minutes. Scenarios depend only on
`(covariance, n_buckets, seed, n_raw, n_target)`, so **all eight dial
combinations within a tier share one scenario set** — the ladder generates
four, not thirty-two.

## 7. The objective — `problem/objective.py`

### 7.1 The dispatch fix comes first

`_risk_cost` currently selects the variance branch with
`getattr(instance.risk, "lam", None)`. `CVaRRisk` carries a `lam` too, so this
milestone is the exact coincidence M2a's deferred finding predicted: a CVaR
instance would be silently scored as variance. **Converting to `isinstance`
is the first task of M2b, before `CVaRRisk` exists**, so the fix lands on a
green suite and is verifiable in isolation rather than tangled with new
behaviour.

### 7.2 Rockafellar–Uryasev

```
CVaR_alpha(x, zeta) = zeta + 1/(1 - alpha) * sum over s of w_s * max(L_s(x) - zeta, 0)
risk_cost           = lam * CVaR_alpha(x, zeta)
```

Note the **weighted** form. The parent spec §5.4 writes `1/((1-alpha)*S) *
sum_s`, which assumes uniform scenarios; after reduction the weights are not
uniform, and using `1/S` would silently mis-weight every reduced scenario set.
The two agree when `w_s = 1/S`.

`lam` multiplies CVaR exactly as it multiplies the variance quadratic, so the
generator's existing auto-balancing (risk ≈ cost at the witness) works
unchanged.

### 7.3 `zeta`

```python
def optimal_zeta(instance: Instance, schedule: Schedule) -> float
def evaluate(instance, schedule, *, zeta: float | None = None) -> ObjectiveBreakdown
```

`optimal_zeta` is the weighted empirical VaR at level `alpha`: the smallest
`zeta` with `sum of w_s over {s : L_s <= zeta} >= alpha`. R–U attains its
minimum there, so `CVaR(x, optimal_zeta(x))` equals the sorted-tail closed
form — which §10 asserts.

`evaluate` **raises** when the risk spec is `CVaRRisk` and `zeta` is omitted.
No silent default: the whole point of the explicit form is that the call site
declares which `zeta` it is scoring.

## 8. `CVaRRisk` — `problem/risk.py`

```python
@dataclass(frozen=True)
class CVaRRisk:
    lam: float
    alpha: float                 # tail level, e.g. 0.95
    scenarios: ScenarioSet
    name: str = "cvar"

    def describe(self) -> dict[str, Any]
```

`__post_init__` rejects `lam < 0` and `alpha` outside `(0, 1)`.

`describe()` must stay JSON-safe for `Instance.content_hash`: it emits
`alpha`, `lam`, `n_scenarios`, the scenario `seed`, and the reduction
report's scalars — **not** the returns array. Two instances differing in their
scenario set must still hash differently, which the seed and scenario count
provide without embedding megabytes of floats in a hash payload.

## 9. Generator — `problem/generator.py`

When `dials.cvar_risk` is set, `generate` builds a `CVaRRisk` instead of a
`VarianceRisk`, using a scenario set cached per tier (§6). Everything else —
positions, the witness, the D2/D4 constraints, the feasibility assertion — is
untouched. `_balanced_lam` works unchanged because it goes through the public
`evaluate`; it passes `optimal_zeta(...)` for the CVaR case.

## 10. Testing

**The M2b gate.** `generate_ladder(include_cvar=True)` produces 16 dial
combinations × 4 tiers = 64 instances; every witness is feasible and every
objective finite. The 32 non-CVaR instances must be **byte-identical** to
M2a's, verified by `content_hash`, proving D3 added rather than perturbed.

**The variational property.** For any `zeta`, `CVaR(x, zeta) >=
CVaR(x, optimal_zeta(x))`. This single property validates the whole R–U
formulation, and is exactly what M3's binary-expanded `zeta` will trade
against. Verified during design over a 400-point grid spanning the loss
range: every point satisfied it.

**R–U at the optimum equals the closed form**, to within the discreteness of
a finite sample: measured `23412.5942` against a sorted-tail mean of
`23412.1216`, a relative difference of `2.0e-5`. The two agree exactly only
when `alpha * S` is an integer, so this test needs a small tolerance rather
than exact equality — and that residual is *not* the same quantity as M3's
`zeta`-discretisation gap, which is a separate and much larger effect.

**Variance correspondence.** On a generated (unreduced) scenario set,
`Var(L_s)` matches `sum_t h_t' Sigma_price h_t` within sampling tolerance.
This ties the scenario machinery to the variance term the parent spec says
CVaR replaces. Measured during design at N=3, T=4, S=200,000: empirical
`1.2901e8` against analytic `1.2947e8`, a ratio of `0.9965`. The
correspondence is exact only in expectation, so the test needs a Monte Carlo
tolerance — roughly `rel=0.02` at S=20,000, tightening as sqrt(S).

**CVaR monotonicity.** `CVaR` is non-decreasing in `alpha`, and
`CVaR_alpha >= E[L]` for any `alpha`.

**Reduction quality.** Wasserstein distance decreases monotonically as
`n_target` grows; `cvar_discrepancy` at `n_target=25` is small relative to
the CVaR itself; weights sum to 1 and are non-negative.

**Determinism.** Same seed reproduces identical scenarios and identical
instance `content_hash` values.

**Rejections.** A weight vector not summing to 1, a non-finite return, an
`alpha` outside `(0, 1)`, a mismatched scenario shape, and a missing `zeta`
under `CVaRRisk` each raise a typed error naming the offending value.

## 11. Deviations from the parent spec

1. **CVaR lives in `problem/scenarios.py` and `problem/risk.py`, not
   `problem/constraints/cvar_lvar.py`.** Parent spec §12 files it under
   constraints, but CVaR is a risk *term* in the objective, not a constraint
   on the decision variables. Filing it under `constraints/` would put a
   non-constraint behind the `Constraint` protocol.
2. **`zeta` is a keyword on `evaluate`, not a field on `Instance`.** Keeps
   M2a's structural promise (§3).
3. **The R–U sum is weighted (`w_s`) rather than uniform (`1/S`).** Required
   once scenarios are reduced; equivalent before reduction.
4. **Reduction keeps real scenarios, not centroids** (§3).

## 12. What M3 picks up

`zeta` as a binary-expanded decision variable plus one auxiliary per scenario,
per parent spec §5.4 — which is why scenario count directly inflates the
variable count and why reduction is mandatory rather than an optimisation.
M3's energy-equality gate compares its QUBO against `evaluate(..., zeta=...)`
at the same `zeta`, which §3's first decision exists to make possible.
