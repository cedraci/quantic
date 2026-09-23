# M0/M1 Final Review — Open Findings and M2 Prerequisites

**Date:** 2026-09-23
**Branch reviewed:** `feat/m0-m1-data-microstructure-core`, `db75268..7e0c15e` (29 commits, 183 tests)
**Verdict:** Ready to merge. Four Critical blockers were found and fixed in a final fix wave; the findings below are the ones deliberately left open.

This document exists so the review's output survives the session that produced it. Nothing here blocks the merge. Everything here should be read before M2 builds the optimisation problem model on this foundation.

---

## 1. The three gates and what they do not cover

M0/M1 ships three mutation-verified gates:

| Gate | Test | Verified by |
|---|---|---|
| L3 replay reproduces L2 | `test_reconstruction_matches_l2*` | 4 injected reconstructor bugs, all caught |
| Message stream fully explains the book | `test_l3_replay_reproduces_l2_at_default_scale` | Suppressed `add` emission, caught |
| Impact parameter recovery | `test_recovers_ground_truth_delta_and_y_in_low_noise` | 5 injected estimator bugs, all caught |

Three independently-written order book implementations must agree for the first two to pass, which is a genuinely strong design. Three classes of defect nonetheless sit outside all of them:

1. **`l1_taq` and `daily_bars` as data are ungated.** Gates 1 and 2 never read them. Gate 3 reads L1 only for `(bid+ask)/2` and daily only through `estimate_bucket_sigma`, which the gate bypasses by passing the true sigma.
2. **Shared misconceptions about message semantics.** Both sides of gates 1 and 2 originate in `build_l3_and_l2`. Three implementations agreeing about a wrong `replace` semantic is three agreements, not three checks. Real feeds vary (in-place size modify, price-unchanged replace, priority retained on size decrease).
3. **Anything the synthetic generator cannot express.** Synthetic data is uniform, never crossed, never duplicated, never gapped, always full depth. A gate can only test what its fixture can produce.

Cheapest coverage gains, in order: assert `calibrate_bundle` (not `fit_power_law`) recovers ground truth; add L1/L2/daily cross-consistency assertions; add a ragged/crossed/duplicated fixture and run the existing gates against it.

---

## 2. Top 5 to address before M2

### 2.1 Build the real-data ingest path, value validation first

There is currently **no code path from a real data export to a `DatasetBundle`**. `quantic data ingest` calls `DatasetBundle.load`, which requires a `manifest.json` that only `DatasetBundle.write` produces, and the only CLI route to `write` is `synth`. Pointed at a directory of exchange parquet, ingest raises `FileNotFoundError`.

The plan narrowed spec §8.3's ingest without recording it as a deviation. Spec §15's own risk table says real data is needed first at M1 validation.

Needed: a CLI verb that turns an export directory into a bundle. The value-level validation added as blocker C4 (`schemas.validate_values`) is the foundation; it must run on that path.

### 2.2 Pin the M1 to M2 semantic contract before the objective is written

This is the finding most likely to produce an M2 objective that is wrong but plausible.

- **`sigma` has no declared time base.** `ImpactParams.sigma` carries no unit. Calibration feeds it a per-bucket volatility, `SynthConfig.daily_vol` is daily, and `CovarianceEstimate.matrix` is annualised. Three volatility time bases on one branch, none named in a type.
- **`bucket_volume` unit is unpinned.** Spec §5.2 defines participation as traded notional over bucket notional. The code uses shares throughout while `ImpactParams` also carries `price`, so mixing a notional `q` with a share `V` is one keyword away and nothing detects it.
- **The calibrated regressor is not own-participation.** `signed_order_flow` returns aggregate signed order-flow imbalance in [-1, 1]. `fit_power_law` regresses impact on its absolute value. M2's objective uses *our own* one-sided participation `x[i,t]/V_i` in [0, 1]. A `Y` fitted on imbalance is not a `Y` for own participation, and nothing says so.
- **`Side` carries two contradictory meanings.** In the data layer and `liquidity.signed_order_flow`, `side` is the *resting* order's side. In `depth_walk.walk`, `Side.BUY` means the *aggressor* is buying. Same enum, inverted meaning.

### 2.3 Replace the calibration estimator to remove its selection bias

The binned log-log fit drops any bin whose mean signed impact is non-positive, because it then takes a logarithm. That conditions on the outcome variable, biasing surviving bins upward and making *which* bins survive draw-dependent. Measured consequence: delta error does not fall monotonically with sample size (4800 obs/symbol was worse than 3600; 1440 worse than 480 at lower noise). The estimator is not consistent under this filter.

This is a limitation of the estimator **chosen by the plan**, not one the spec prescribes. The spec prescribes the impact law (§5.2) and the recovery test (§10.4). Fitting `mean_impact = Y*sigma*f^delta` directly on signed bin means by non-linear least squares needs no logarithm, therefore no positivity filter, therefore no selection mechanism.

Until this is done, `Y` is fitted through a biased filter.

### 2.4 Harden the covariance path for ragged real data

`log_returns` pivots wide then calls `drop_nulls()`, so a null for any one symbol drops that date for **every** symbol. Measured: 5 symbols over 60 days with one symbol missing 10 scattered days loses 18 of 59 rows — 30% of the cross-section, including for symbols with complete coverage. Real coverage is ragged through halts, staggered listings and venue holidays.

Recommended: a `max_dropped_fraction` keyword (default ~0.10) raising `InsufficientHistoryError` and naming the dropped count. Fold in a typed error for duplicate `(date, symbol)` rows, which currently raise a raw polars `ComputeError`.

### 2.5 Replace epoch-floor `bucket_id` with session-aware bucketing

`liquidity.py` uses `ts_ns // bucket_ns`; `calibrate.py` uses `(ts_ns - 1) // bucket_ns`. Two half-open conventions share one `bucket_id` namespace, reconciled only by a comment. They align today only because the synthetic session opens at 09:30 UTC and 34,200s is an exact multiple of 1,800s.

Measured failure: at 09:30 America/New_York with 20-minute buckets, the session open falls mid-bucket, so the first bucket of each day straddles the overnight gap. The gap filter (`bucket_id - prev_bucket == 1`) cannot see it, and the overnight return is silently attributed to the day's flow.

Also `calibrate_bundle` hardcodes a second copy of the session length and derives `buckets_per_day` by rounding — at a 1-hour bucket that gives 6 instead of 6.5, a silent 4.1% error in every estimated sigma.

Needed: one `session_bucket_id(ts_ns, calendar, bucket_ns)` helper, one source of truth for session length, one half-open convention.

---

## 3. Other open findings

### Important

- **Reconstruction gate is self-consistency, not cross-validation, until real paired L2/L3 arrives.** `validation.py`'s docstring and spec §10.3 both claim independent-source verification. Today both sides come from the same `build_l3_and_l2` call. Amend the claims and keep a named holdout test for the day real data lands.
- **`fit_power_law` can return a structurally unusable result.** `delta` comes from an unconstrained `polyfit`. A convex generator yielded `delta = 1.922` with `r_squared = 0.9996`; the `CalibrationResult` constructs fine and `to_model()` then raises. Validate `delta` at the fit and raise `CalibrationError` naming the value.
- **The M1 gate never exercises `estimate_bucket_sigma`.** The gate passes the injected true sigma. `estimate_bucket_sigma` is covered only at `rel=0.5`, so a 49% sigma error passes, and sigma bias maps one-to-one onto `y_coef` bias. Add a low-noise assertion on `calibrate_bundle` with sigma unsupplied.
- **`strict=False` drops messages with no observability**, and is asymmetric: it tolerates unknown-order cancels but still raises on a duplicate `order_id`, which is the most common real-export defect. Add a drop counter surfaced from `snapshots_at`; decide deliberately about duplicate adds.
- **`compare_to_l2` is superlinear.** It filters the symbol's whole L2 frame inside the per-timestamp loop: 65 snapshots took 0.124s, 520 took 1.326s (10.7x for 8x the data). `snapshots_at` iterates rows at ~17k/s. One liquid US name is O(10^6) messages/day. A `partition_by("ts_ns")` before the loop removes the quadratic term.
- **Crossed and degenerate L1 quotes reach the calibration path.** `calibrate.py` computes `(bid+ask)/2` with no validity check while `BookSnapshot.mid` raises on the same condition. An injected crossed quote produced a -3.66% bucket impact observation against a normal range of +/-0.9%; a `bid = 0.0` row gives roughly -50%. Blocker C4 now rejects these at the bundle boundary, but the calibration path itself is still unguarded for data that arrives another way.
- **No `BookSnapshot` can be built from L1 or L2 — only from L3.** Every book-based liquidity metric therefore reaches only the 3-5 symbols that have L3. Under spec §8.4's data shape (~25 names with L1+L2, 3-5 with L3), no liquidity metric is computable for ~20 of 25 symbols, and spec §5.2's spread-cost term has no route from `l1_taq` to a number.
- **`resiliency_halflife` has no caller and no path from data.** Nothing extracts an `(elapsed_ns, spread)` series from a bundle.

### Minor

- Zero-row partitioned table: `write()` succeeds and `validate()` reports clean, but `table()` then raises a polars `ComputeError`. Unreachable from current producers; a real export routinely has a symbol with zero rows for a granularity.
- `build_l1` sets `last_size = |net_flow|`, so summing it gives 8.6% of true volume. Either rename the semantics or set something defensible.
- L1 is stamped at `ts_end_ns`, L2 at `ts_end_ns - 1`; a naive join yields zero rows. Intentional and load-bearing, but undocumented at both stamp sites.
- No test cross-checks L1 against L2 or daily.
- Four tests are tautological or near-tautological, notably `test_annualisation_scales_the_matrix`, which restates the implementation.
- The R22 replenish guard duplicates the generator's `buy_vol` formula by hand rather than instrumenting it; if the generator's formula changes, the test's copy diverges silently.
- `test_layer_contract_holds` resolves `lint-imports` via PATH, so it fails without venv activation. Use `Path(sys.executable).parent`.
- The layer contract is a `layers` contract, which permits `solvers` to import `problem` internals; spec §12 forbids it. Needs a `forbidden` contract at M3/M4. `exhaustive = false` leaves future subpackages unconstrained.
- Two ADV implementations: `synth.build_daily_bars` writes an `adv` column, `liquidity.adv` recomputes and discards it.
- `round_to_tick` is dead code with its own test.
- `BookBuilder._levels` keys on raw floats with no quantisation, so `10.0` and `10.000000000000002` are separate levels; `compare_to_l2` uses a `1e-9` tolerance on the other side of the same gate.
- Error types split across `ValueError` and `RuntimeError` with no common base, so a caller cannot catch all project errors.
- `.gitignore` misses `data/catalog.json` (the default catalog path), `.import_linter_cache/` and `.hypothesis/`.
- `Catalog.register` returns a stale entry on idempotent re-register if the bundle moved.
- `bundle.py` carries a `# type: ignore` with no type checker configured.
- `_execute_against`'s `if not progressed: break` under-executes silently; it should raise.

---

## 4. Findings from the first data-forging exercise (2026-09-23)

Four synthetic bundles were generated and the full analytics stack run against them. Everything passed, but two structural problems surfaced that only appear at realistic scale.

### 4.1 No single synthetic config can validate both `delta` and `Y` end-to-end

The generator injects impact as `Y * sigma_bucket * sign(f) * |f|^delta`, where `sigma_bucket` is the **configured** `daily_vol / sqrt(buckets_per_day)`. But the realised volatility of the emitted mid series only equals that when `noise_frac ~ 1.0`, because `noise_frac` scales the diffusion term without compensating elsewhere. Measured:

| bundle | `noise_frac` | configured `sigma_bucket` | realised | `estimate_bucket_sigma` | est/configured |
|---|---|---|---|---|---|
| `calib` | 0.05 | 0.005547 | 0.001131 | 0.001158 | **0.21x** |
| `bench25` | 1.0 | 0.005547 | 0.005671 | 0.005265 | 0.95x |

`estimate_bucket_sigma` is **correct in both cases** — it measures realised volatility to within 2%. The fiction is `ground_truth()["sigma_bucket"]`, which reports the configured value regardless of `noise_frac`.

Consequence, measured on `calib`: supplying the (correct) estimated sigma inflates the recovered `Y` by 357–464%, while `delta` is unaffected — sigma is a constant divisor inside `log(mean_impact / sigma)`, so it shifts the intercept but not the slope.

So `Y` is only recoverable through the production path at `noise_frac ~ 1.0`, which is precisely the regime where `delta` is **not** recoverable (see 4.2). The two cannot currently be validated in the same bundle.

**Recommended fix:** hold total bucket variance at `sigma_bucket^2` and let `noise_frac` control the split between impact and diffusion, rather than scaling diffusion alone. Realised volatility then equals the configured value at every noise level, and both parameters become validatable anywhere.

### 4.2 The calibration limitation is more severe at realistic noise than the ledger's ranges suggested

`bench25` (25 symbols, 20 days, `noise_frac=1.0`, 240 observations/symbol), sigma supplied as the true value:

- max `|delta error|` = **1.763**; 6 of 25 symbols recovered a **negative** delta
- `r_squared` below 0.05 for 8 of 25 symbols
- SYN07 recovered `delta = 2.263`, which is outside `PowerLawImpact`'s valid `(0, 1]` — `CalibrationResult` constructs happily and `to_model()` then raises. This is finding I6 occurring on ordinary generated data rather than a crafted fixture.

Earlier measurements put realistic-noise recovery at 0.13–0.26 error; those used 1800–3600 observations per symbol. At the spec's own 20-day shape (240 observations) the estimator is not merely imprecise, it is unusable. Calibration at realistic noise needs a substantially longer history than spec section 8.4 requests, or the direct non-linear fit from section 2.3.

### 4.3 Quantified: `compare_to_l2` throughput

Confirmed finding I9 at scale: 6,500 snapshots across 25 symbols took **23.4s (278 snapshots/s)**, with zero mismatches. The per-timestamp filter inside the loop dominates. One liquid US name is O(10^6) messages/day, so this needs the `partition_by` fix before real L3 arrives.

### 4.4 Generation cost is superlinear in days

3 symbols x 20 days ~ 1.7s; 25 x 20 ~ 7s; 25 x 252 took **8m47s** — roughly 10x the time for 12.6x the buckets, but measured against a 25 x 20 baseline it is materially worse than linear. Acceptable for fixtures; worth knowing before generating multi-year data.

### 4.5 The CLI cannot reach the knobs that matter

`quantic data synth` exposes 5 of `SynthConfig`'s 16 fields. `noise_frac`, `base_price`, `daily_vol`, `adv_shares`, `level_size` and `flow_sd` are unreachable, so the calibration-grade fixture had to be built with a Python script. `noise_frac` in particular is the single knob that decides whether impact calibration works at all.
