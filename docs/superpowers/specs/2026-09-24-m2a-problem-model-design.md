# M2a — Problem Model, Hardness Dials D1/D2/D4, and the Instance Ladder

**Status:** Design approved, pending implementation plan
**Date:** 2026-09-24
**Builds on:** `2026-09-22-quantum-liquidation-lvar-design.md` (§4, §5, §7.1, §12, §14)
**Consumes:** `2026-09-24-m2-prerequisites-resolved.md`

---

## 1. Purpose

Build `src/quantic/problem/`: the discrete multi-asset liquidation problem, three
of its four hardness dials, and a generator that produces provably valid
instances across the spec's four-tier ladder.

M2a is done when **8 dial combinations × 4 tiers = 32 instances** generate, each
carrying a feasible witness schedule and evaluating to a finite objective.

## 2. Scope

### In scope

- `Schedule`, `MarketParams`, `Dials`, `Instance` — the types that cross into M3 and M4.
- The four-term objective (spec §5.2) with a **variance** risk term, and a
  reference evaluator.
- Dials **D1** (concave impact), **D2** (cardinality + minimum lot), **D4**
  (all-or-nothing blocks).
- Constraint modules with problem-level feasibility logic, and one aggregating
  feasibility classifier.
- The instance ladder generator, feasible by construction.
- An adapter building `MarketParams` from a `DatasetBundle`.

### Deferred to M2b

- Dial **D3** — CVaR / L-VaR risk (spec §5.4): scenario path generation,
  clustering to 20–30 representatives, Wasserstein reduction error and its
  reporting.

M2b adds a `CVaRRisk` variant of `RiskSpec` and the scenario machinery behind
it. **No other M2a type changes.** §5 below is written to guarantee that.

### Out of scope

- Any QUBO, penalty or binary expansion. That is M3, and spec §12 forbids
  `problem/` importing `encoding/`.
- Solvers of any kind. That is M4.

## 3. Decisions

| Decision | Rationale |
|---|---|
| Split M2 into M2a (D1/D2/D4) and M2b (D3) | D3 alone drags in scenario generation, clustering and a reduction-error measurement problem. Splitting reaches M3's encoding gate on 8 combinations sooner without stalling the modelling layer. The cost — that `Instance` must absorb per-scenario auxiliaries later — is paid once, in §5, by making the risk term a variant point. |
| Generator takes an explicit `MarketParams`; a separate adapter builds one from a bundle | Measured on 2026-09-24: calibration **raises** for 8 of 25 symbols at spec §8.4's 20-day shape, and returns median delta error 0.34 for the rest. Calling calibration inside instance generation would make a third of the ladder fail to generate for reasons unrelated to the problem model, and would stop instance identity being reproducible from a seed alone. |
| Constraints are frozen descriptors carrying feasibility logic; encoding pattern-matches to quadratise | Feasibility rate is a headline metric (spec §6.5) and must be computable with no encoder present — including for M4's CP-SAT and MILP solvers, which express constraints natively and never build a QUBO. Each constraint also has a genuinely different quadratisation (spec §6.2's min-participation trick is not derivable from a generic interface), so a uniform term-emitting interface would be encoding's concern wearing a disguise. |
| Instances are feasible by construction and carry a witness schedule | It turns §14's done-criterion into something a test asserts rather than hopes for; it gives M3's decode/repair a known-good target and M4 a guaranteed-feasible baseline for the optimality gap; and it makes an infeasible solver result unambiguously the solver's fault. |

## 4. Units

The single largest source of plausible-but-wrong results in this layer, and the
subject of the prerequisite work in `2026-09-24-m2-prerequisites-resolved.md` §2.2.

| Quantity | Unit | Where it is fixed |
|---|---|---|
| `x[i,t]`, `X[i]`, everything in `Schedule` | **lots** | spec §5.1 |
| Everything in `micro/`, including `ImpactParams` | **shares** | `micro/impact/base.py` |
| `MarketParams.covariance` | dimensionless log-return covariance, at the **bucket** horizon | §5.2 below |
| `half_spread` | currency per share | §5.2 below |
| Every `ObjectiveBreakdown` field | currency | §6 below |

**Lots to shares** converts exactly once, at the objective boundary, via
`Asset.lot_size`. Nothing else in `problem/` performs the conversion.

**Return covariance to price covariance.** `CovarianceEstimate` is a covariance
of log returns and is dimensionless; Almgren–Chriss's risk term `h' Σ h` needs a
covariance of price changes with `h` in shares. The evaluator converts
explicitly:

```
Sigma_price = diag(price) @ Sigma_returns @ diag(price)
```

Left implicit, this is a silent error of `price**2` — four orders of magnitude at
a $100 stock — and entirely plausible in the output.

**Horizon.** `MarketParams.covariance` is stored already at the bucket horizon,
so the `tau` in spec §5.2's risk term is 1 per bucket and does not appear again.
The adapter in §5.3 performs the rescale via
`CovarianceEstimate.at_horizon(days=bucket_ns / session.length_ns)`.

## 5. Types

### 5.1 `Schedule` — `problem/schedule.py`

A frozen `N × T` integer grid in lots, plus the operations every constraint and
the evaluator need.

```python
@dataclass(frozen=True, slots=True)
class Schedule:
    lots: tuple[tuple[int, ...], ...]     # N rows of T columns

    @property
    def n_assets(self) -> int
    @property
    def n_buckets(self) -> int
    def sold(self, i: int) -> int                     # sum over t
    def active_names(self, t: int) -> int             # count of x[i,t] > 0
    def cumulative(self) -> np.ndarray                # (N, T), INCLUSIVE of t
    def cumulative_before(self) -> np.ndarray         # (N, T), EXCLUSIVE; column 0 is zero
    def as_array(self) -> np.ndarray                  # (N, T) int64
```

A tuple of tuples rather than an array: `Instance` is frozen and content-hashed,
and T3 is 30 × 12 = 360 integers, so there is nothing to gain from mutability.
`as_array()` exists for the numeric paths.

`__post_init__` rejects negative entries and ragged rows. Non-negativity is spec
§5.3's constraint and is enforced by the type rather than checked later.

### 5.2 `MarketParams` — `problem/params.py`

```python
@dataclass(frozen=True, slots=True)
class AssetParams:
    symbol: str
    delta: float                   # (0, 1], the D1 exponent
    y_coef: float
    gamma: float                   # permanent impact coefficient
    sigma_bucket: float
    bucket_volume_shares: float
    price: float
    half_spread: float             # currency per share, spec 5.2 term 1's s_i / 2

@dataclass(frozen=True)
class MarketParams:
    assets: tuple[AssetParams, ...]
    covariance: np.ndarray         # (N, N) log-return covariance at the BUCKET horizon
    bucket_ns: int

    def impact_params(self, i: int) -> ImpactParams      # builds micro's type
    def price_covariance(self) -> np.ndarray             # diag(p) @ cov @ diag(p)
    def prices(self) -> np.ndarray
```

`__post_init__` validates: covariance is square, symmetric and positive
semi-definite; its dimension matches the asset count; `delta` lies in `(0, 1]`;
`bucket_volume_shares`, `price` and `half_spread` are positive; `bucket_ns` is
positive. A non-PSD covariance makes the risk term unbounded below and every
downstream optimum meaningless, so it is rejected at construction.

`impact_params(i)` is the only place `micro.ImpactParams` is built, which keeps
the unit contract from §4 in one function.

### 5.3 `params_from_bundle` — `problem/from_bundle.py`

```python
def market_params_from_bundle(
    bundle: DatasetBundle,
    *,
    session: TradingSession,
    bucket_ns: int,
    symbols: Sequence[str] | None = None,
    gamma: Mapping[str, float] | None = None,
    assume_own_participation: bool = False,
) -> MarketParams
```

Composes what M1 produced: `calibrate_bundle` for `delta` and `y_coef`,
`estimate_covariance(...).at_horizon(...)` for the covariance, `liquidity.adv`
for `bucket_volume_shares`, and `books.bucket_spreads` for `half_spread`.

`assume_own_participation` is passed straight through to
`CalibrationResult.to_model()` and is **not** defaulted to `True`: the M2
objective evaluates our own participation while the fit was made on aggregate
net order-flow imbalance, and that identification assumption stays at the call
site where a human chose it.

A `CalibrationError` for any requested symbol propagates, naming the symbol.
Measured behaviour makes this the common case at spec §8.4's data shape, which
is exactly why this function is separate from the generator.

### 5.4 `Dials` — `problem/dials.py`

```python
@dataclass(frozen=True, slots=True)
class Dials:
    concave_impact: bool = False           # D1
    discrete_participation: bool = False   # D2 -- cardinality AND minimum lot
    cvar_risk: bool = False                # D3 -- M2b
    block_trades: bool = False             # D4

    @classmethod
    def combinations(cls, *, include_cvar: bool = False) -> tuple[Dials, ...]
    @property
    def label(self) -> str                 # "none", "D1", "D1+D4", ...
```

`combinations(include_cvar=False)` yields the 8 M2a combinations;
`include_cvar=True` yields all 16 once M2b lands. One switch, and the M2b
change is a default flip in the experiment config rather than a code change.

`label` exists because every results row in spec §9.3 needs a stable, readable
dial identifier.

D2 governs cardinality and minimum lot **together**, per spec §4: cardinality
without a minimum lot size is not meaningful on a real desk.

### 5.5 `RiskSpec` — `problem/risk.py`, the M2b extension point

```python
@runtime_checkable
class RiskSpec(Protocol):
    name: str

@dataclass(frozen=True, slots=True)
class VarianceRisk:
    lam: float                 # risk aversion
    name: str = "variance"
```

M2b adds `CVaRRisk(alpha, scenarios, weights, reduction_error)` alongside it and
a matching branch in the evaluator. `Instance.risk` is typed `RiskSpec`, so
nothing else in §5 changes.

### 5.6 Constraints — `problem/constraints/`

```python
@runtime_checkable
class Constraint(Protocol):
    name: str
    def violation(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> float: ...
    def is_satisfied(self, schedule: Schedule, initial_lots: tuple[int, ...]) -> bool: ...
    def describe(self) -> dict[str, Any]: ...
```

The signature takes `initial_lots` rather than the whole `Instance`, which would
be circular — `Instance` holds constraints. Only `FullLiquidation` uses it; a
uniform signature keeps aggregation trivial.

| Module | Type | Constraint |
|---|---|---|
| `full_liquidation.py` | `FullLiquidation` | `sum over t of x[i,t] == X[i]` for every `i` (spec §5.3) |
| `cardinality.py` | `Cardinality(k)` | at most `k` names have `x[i,t] > 0` in each bucket `t` (D2) |
| `min_participation.py` | `MinParticipation(min_lots)` | for each `(i,t)`, `x[i,t] == 0` or `x[i,t] >= min_lots[i]` (D2) |
| `block_trades.py` | `BlockTrades(blocks)` | each block `(i, t, lots)` is traded whole or not at all (D4) |

`violation` returns a non-negative magnitude, zero exactly when satisfied — not
a boolean — because M3's penalty derivation (spec §6.3) needs the size of a
one-unit violation, and M4's repair strategies need to know how far off a
sample is.

`describe()` returns the constraint's parameters as a dict for the results row.

### 5.7 `feasibility.py` — one implementation, no more

```python
@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    feasible: bool
    violations: dict[str, float]        # constraint name -> magnitude
    total_violation: float

def classify(instance: Instance, schedule: Schedule) -> FeasibilityReport
```

This is the **single** answer to "is this schedule feasible". M3's decoder, M4's
CP-SAT and MILP solvers, and M6's metrics all call it. Three independent copies
of this test is how a headline feasibility-rate metric silently disagrees with
itself.

### 5.8 `Instance` — `problem/instance.py`

```python
@dataclass(frozen=True, slots=True)
class Instance:
    instance_id: str
    tier: str                          # "T0".."T3"
    assets: tuple[Asset, ...]          # N, ordered; carries lot_size and tick_size
    n_buckets: int                     # T
    initial_lots: tuple[int, ...]      # X[i]
    max_lots_per_bucket: int
    params: MarketParams
    dials: Dials
    constraints: tuple[Constraint, ...]
    risk: RiskSpec
    witness: Schedule
    seed: int
    content_hash: str

    def remaining_lots(self, schedule: Schedule) -> np.ndarray    # (N, T), h[i,t]
    def notional(self) -> float                                   # sum X_i * lot * price
```

It carries `max_lots_per_bucket`, **not** the bit width `B`. Bit width is a
property of an encoding, and spec §6.1 makes the integer encoding a study
dimension with three competing choices — binary, one-hot and domain-wall — which
need different variable counts for the same problem. The problem's shape is
`N × T` plus an integer range. Spec §5.5's tier table is reproduced exactly by
`max_lots_per_bucket = 2**B - 1`, stated in the tier constants.

`content_hash` is a deterministic hash over every field except itself, so a
results row can pin an instance the way it pins a dataset.

## 6. The objective — `problem/objective.py`

```python
@dataclass(frozen=True, slots=True)
class ObjectiveBreakdown:
    spread_cost: float
    temporary_impact_cost: float
    permanent_impact_cost: float
    risk_cost: float

    @property
    def total(self) -> float
    def in_bps(self, notional: float) -> ObjectiveBreakdown

def evaluate(instance: Instance, schedule: Schedule) -> ObjectiveBreakdown
```

The four terms are returned **separately**, not just summed. M3's load-bearing
property test (spec §10.1) asserts that QUBO energy equals the directly
evaluated objective; when that fails, a single scalar says nothing about which
term is wrong.

Three derived quantities, all in shares, defined once here because an off-by-one
in any of them is invisible in the output:

- `q[i,t]` — shares traded in bucket `t`, i.e. `x[i,t] * lot_size[i]`.
- `cum_before[i,t]` — shares traded **strictly before** `t`, so
  `cum_before[i,0] == 0`.
- `h[i,t]` — shares remaining **after** `t`, i.e.
  `X[i] * lot_size[i] - sum over s <= t of q[i,s]`. Under full liquidation
  `h[i,T-1] == 0`, so the final bucket contributes nothing to risk, which is
  correct: nothing is held over it.

1. **Spread.** `sum over (i,t) of half_spread[i] * q[i,t]`.
2. **Temporary impact.** `sum over (i,t) of model_i.temporary_cost(q, p_i) * q[i,t] * price[i]`,
   where `model_i` is `PowerLawImpact(delta_i, Y_i)` when D1 is on and
   `AlmgrenChriss` linear impact when it is off. This term is the only thing D1
   changes, and a test asserts that.
3. **Permanent impact.**
   `sum over (i,t) of gamma[i] * sigma[i] * ((cum_before[i,t] + q[i,t]/2) / V[i]) * q[i,t] * price[i]`

   The **midpoint** convention: we pay the displacement already caused *plus
   half of our own*, because our order trades through its own permanent
   impact. This is standard Almgren-Chriss, and it is what makes the term
   path-independent, summing to `0.5 * gamma * sigma * X**2 / V * price` for
   full liquidation regardless of the schedule.

   Using `cum_before` alone would be path-**dependent** and understate the
   term by `sum over t of q**2 / 2`. Measured on a 400-share liquidation over
   two buckets: the exclusive form gives 0.0050 for an even schedule and
   0.0000 for a front-loaded one, against the closed form's 0.0100; the
   midpoint form gives 0.0100 for both.
4. **Risk.** `lam * sum over t of h_t @ Sigma_price @ h_t`, with `h_t` in shares.
   The bucket horizon is already inside `Sigma_price` (§4), so spec §5.2's `tau`
   does not appear.

`evaluate` does not check feasibility. Scoring an infeasible schedule is a
legitimate thing to want — it is how penalty magnitudes are calibrated in M3 —
and conflating the two is how an infeasible quantum solution gets compared
against a feasible classical one, which spec §6.5 names as the error that
invalidates most published work in this area. Feasibility is `classify`'s job
and is reported separately.

## 7. The generator — `problem/generator.py`

### 7.1 The ladder

```python
@dataclass(frozen=True, slots=True)
class TierSpec:
    name: str
    n_assets: int
    n_buckets: int
    bits: int                                    # spec 5.5's B

    @property
    def max_lots_per_bucket(self) -> int         # 2**bits - 1
    @property
    def approx_variables(self) -> int            # n_assets * n_buckets * bits

T0 = TierSpec("T0", n_assets=3,  n_buckets=4,  bits=2)   # ~24 variables
T1 = TierSpec("T1", n_assets=5,  n_buckets=8,  bits=3)   # ~120
T2 = TierSpec("T2", n_assets=8,  n_buckets=8,  bits=3)   # ~192
T3 = TierSpec("T3", n_assets=30, n_buckets=12, bits=4)   # ~1440

LADDER = (T0, T1, T2, T3)
```

These reproduce spec §5.5. `bits` is recorded here — where the ladder is
*defined* in the spec's own terms — and converted to `max_lots_per_bucket` for
the `Instance`, which is encoding-agnostic (§5.8).

### 7.2 Feasible by construction

```python
def generate(
    tier: TierSpec,
    dials: Dials,
    params: MarketParams,
    *,
    seed: int,
    lam: float,
) -> Instance
```

1. **Positions.** Draw `X[i]` seeded, bounded so that a feasible base schedule
   exists *under the dials in force*. Asset `i` needs
   `ceil(X[i] / max_lots_per_bucket)` buckets, so with D2 off the bound is

   ```
   X[i] <= n_buckets * max_lots_per_bucket
   ```

   With D2 on, step 3 caps each bucket at `active = ceil(n_assets / 2)` names,
   giving a total capacity of `n_buckets * active` asset-bucket slots shared
   across `n_assets` assets. The bound tightens to

   ```
   buckets_per_asset = max(1, (n_buckets * active) // n_assets)
   X[i] <= buckets_per_asset * max_lots_per_bucket
   ```

   **The looser bound is not sufficient** and fails on every tier. Worst-case
   slots demanded against capacity available, computed over the ladder:

   | Tier | N | T | capacity | loose bound demands | tightened bound demands |
   |---|---|---|---|---|---|
   | T0 | 3  | 4  | 8   | 12 — infeasible  | 6   |
   | T1 | 5  | 8  | 24  | 40 — infeasible  | 20  |
   | T2 | 8  | 8  | 32  | 64 — infeasible  | 32  |
   | T3 | 30 | 12 | 180 | 360 — infeasible | 180 |

   T2 and T3 sit exactly at capacity, so the tightened bound is not merely
   sufficient but tight. A test asserts the inequality holds for every tier
   rather than trusting this table.

2. **Base schedule.** Spread each `X[i]` over buckets, no cell exceeding
   `max_lots_per_bucket`.
3. **Concentrate, if D2.** Assign each asset a *subset* of buckets rather than
   all of them, sized to fit its quantity, chosen so each bucket hosts at most
   `active = ceil(n_assets / 2)` names. Step 1's tightened bound is exactly the
   condition under which this assignment exists.
4. **Bind cardinality, if D2.** Set `k = max over t of active_names(t)` in the
   concentrated base schedule. This is the tightest `k` the witness satisfies.
   **A loose `k` would make D2 a no-op**, and a dial that does not bind
   contributes nothing to the screening design in spec §9.2 while appearing in
   every results row as though it did.
5. **Bind minimum lot, if D2.** Set `min_lots[i]` to the smallest non-zero entry
   of row `i` in the base schedule — again the tightest value the witness
   satisfies.
6. **Carve blocks, if D4.** Select a seeded subset of non-zero `(i, t)` cells and
   declare each an indivisible block of exactly the size already there. The
   witness satisfies them by construction.
7. **Witness and assertion.** The base schedule becomes `Instance.witness`, and
   `generate` asserts `classify(instance, witness).feasible` before returning.
   A generator that cannot produce a feasible witness raises
   `InstanceGenerationError` naming the tier and dial combination, rather than
   returning an instance nobody can satisfy.

Steps 4 and 5 are what make D2 real hardness rather than decoration.

### 7.3 Ladder generation

```python
def generate_ladder(
    params_for: Callable[[TierSpec], MarketParams],
    *,
    seed: int,
    include_cvar: bool = False,
) -> tuple[Instance, ...]
```

Every tier × every dial combination. `include_cvar=False` gives M2a's 32
instances; `True` gives M2b's 64. `params_for` is a callable because each tier
needs a different asset count.

## 8. Module layout

| File | Responsibility |
|---|---|
| `problem/schedule.py` | `Schedule` |
| `problem/params.py` | `AssetParams`, `MarketParams` |
| `problem/from_bundle.py` | `market_params_from_bundle` — the only `micro/` consumer |
| `problem/dials.py` | `Dials` |
| `problem/risk.py` | `RiskSpec` protocol, `VarianceRisk` |
| `problem/constraints/base.py` | `Constraint` protocol |
| `problem/constraints/full_liquidation.py` | `FullLiquidation` |
| `problem/constraints/cardinality.py` | `Cardinality` |
| `problem/constraints/min_participation.py` | `MinParticipation` |
| `problem/constraints/block_trades.py` | `BlockTrades`, `Block` |
| `problem/feasibility.py` | `FeasibilityReport`, `classify` |
| `problem/instance.py` | `Instance`, content hashing |
| `problem/objective.py` | `ObjectiveBreakdown`, `evaluate` |
| `problem/liquidation.py` | `TierSpec`, the four tier constants, `LADDER` |
| `problem/generator.py` | `generate`, `generate_ladder`, `InstanceGenerationError` |

`problem/` imports `core/` and `micro/` and nothing else from the project. The
existing import-linter contract already enforces this.

## 9. Testing

Following M0/M1: every task is test-first, and every gate is mutation-checked.

**The M2a gate.** `generate_ladder` produces 8 dial combinations × 4 tiers = 32
instances; every one has a witness that `classify` reports feasible, and every
one evaluates to a finite objective. This is spec §14's done-criterion, asserted.

**Property tests (Hypothesis).**
- For any tier and any dial combination, the generated witness is feasible.
- `ObjectiveBreakdown` fields sum to `total`, for any schedule.
- `violation(s) == 0` if and only if `is_satisfied(s)`, for every constraint and
  any schedule.
- A schedule violating a constraint more has a larger `violation`.

**Unit-discipline tests.** These exist because §4 is where this layer is most
likely to be wrong while looking right.
- Doubling every `lot_size` while halving every entry in the schedule leaves
  every objective term unchanged.
- The permanent impact term is path-independent: two different full-liquidation
  schedules give the same permanent cost, equal to the closed form in §6.
- Scaling every price by `c` scales the risk term by `c**2`, because the
  return-to-price covariance conversion is quadratic in price.
- D1 on versus off changes the temporary impact term and no other term.

**Binding-dial tests.** For every tier, the `k` chosen with D2 on is strictly
less than `n_assets` — a cardinality equal to the asset count constrains nothing
— and `min_lots[i]` is strictly greater than zero.

**Capacity test.** For every tier, §7.2 step 1's tightened bound satisfies
`n_assets * buckets_per_asset <= n_buckets * ceil(n_assets / 2)`. T2 and T3 sit
exactly at equality, so this is asserted rather than assumed; a change to any
tier's shape that broke it would otherwise surface as an intermittent
generator failure.

**Two cumulative conventions.** `cumulative()` is inclusive and
`cumulative_before()` is exclusive with a zero first column. A test pins
`cumulative() - cumulative_before() == as_array()`, because the permanent
impact term uses the exclusive form and silently reading the inclusive one
would charge us for impact we had not yet caused.

**Determinism.** Same seed, same `MarketParams`, same tier and dials produce an
identical `content_hash`.

**Rejection tests.** A non-PSD covariance, a `delta` outside `(0, 1]`, a
negative schedule entry, and an `X[i]` too large for the tier shape each raise a
typed error naming the offending value.

## 10. Deviations from the main spec

Recorded so a reviewer does not flag them as drift.

1. **M2 split into M2a and M2b.** Spec §14 defines a single M2 whose
   done-criterion is all 16 dial combinations. M2a covers 8; M2b completes the
   set. The split is contained by making the risk term a variant point (§5.5).
2. **`Instance` carries `max_lots_per_bucket`, not `B`.** Spec §5.5 sizes tiers
   by `N × T × B`. Bit width belongs to an encoding, and spec §6.1 makes the
   integer encoding a study dimension with three choices of differing width.
   `TierSpec` keeps `bits` so the spec's table is reproduced exactly.
3. **Constraints take `initial_lots`, not `Instance`.** `Instance` holds the
   constraints, so passing it back would be circular. Only `FullLiquidation`
   needs it.
4. **No `problem/liquidation.py` god-object.** Spec §12 lists
   `liquidation.py`, `objective.py`, `instance.py`, `generator.py`.
   `liquidation.py` here holds the tier ladder rather than a `LiquidationProblem`
   class: with `Instance` frozen and the objective a free function, a problem
   object would hold no state that `Instance` does not already hold.

## 11. What M2b picks up

`CVaRRisk` as a second `RiskSpec` variant, a branch in `evaluate`, and the
scenario machinery behind it: generation of ~2000 price and liquidity paths,
clustering to 20–30 weighted representatives, and the Wasserstein reduction
error reported alongside every CVaR-enabled result (spec §5.4). `Dials.combinations(include_cvar=True)`
then yields all 16, and the M2 gate becomes 64 instances.
