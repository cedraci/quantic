# M2 Prerequisites — Resolution Record

**Date:** 2026-09-24
**Resolves:** `2026-09-23-m1-review-findings.md` sections 2 and 4
**Suite:** 183 tests → 288 collected, 287 passing, 1 named skip. Lint clean.

Every finding in section 2 ("Top 5 to address before M2") is closed, together
with four of the five section 4 items that blocked realistic fixture work.
Section 3 findings are addressed where they lie on the same code paths; the
remainder are listed as still open at the end.

---

## Section 2 — the five M2 prerequisites

### 2.1 Real-data ingest path — **closed**

`data/ingest.py` plus a polymorphic `quantic data ingest`. Without `--out` the
verb validates and registers an existing bundle exactly as before; with
`--out`, `path` is a raw export and is normalised into a new bundle.

It normalises what a market-data machine actually sends: flat
`<export>/<name>.parquet|csv|tsv` or a partitioned `<export>/<name>/**` tree;
any column order; narrower dtypes widened; extra vendor columns dropped **and
reported**, in the `IngestReport` and in the manifest. It will not invent a
missing column, coerce a value or repair a row — a missing required column
names itself and stops.

`validate_values` (blocker C4) runs on this path, as the finding required, and
`DatasetBundle.write` runs both validators before touching the filesystem, so
a bad export cannot leave a half-written bundle behind.

`--session` is required in export mode and is never inferred.

### 2.2 M1 → M2 semantic contract — **closed**

| Was | Now |
|---|---|
| `ImpactParams.sigma`, no declared time base | `sigma_bucket`, carried with the `bucket_ns` it was measured over, plus `rescale_to()` |
| `CovarianceEstimate.matrix` silently annualised | `horizon_days` field and `at_horizon()`; covariance scales linearly in time, volatility as its square root, and neither conversion happens at a call site |
| `bucket_volume`, unit unpinned | `bucket_volume_shares`; `participation()` is the only sanctioned way to form abs(q)/V, so a notional q cannot meet a share V |
| `Y` fitted on imbalance, used as own participation, nothing said so | `CalibrationResult.basis`; `to_model()` refuses to cross bases unless the caller passes `assume_own_participation=True` |
| `Side` meant resting in the data layer, aggressor in `depth_walk` | `Aggressor` is a separate type with `.consumes` and `.from_resting_side()`; `walk()` rejects a `Side` at runtime |

`Aggressor` is a plain `Enum`, not a `StrEnum`: `StrEnum` members compare
equal across classes by value, so `Aggressor.BUY == Side.BUY` would have been
`True` and the separation would have bought nothing.

### 2.3 Calibration estimator — **closed**

Replaced the binned log-log fit with direct non-linear least squares on
`mean_impact / sigma = Y * f**delta`. No logarithm, therefore no positivity
requirement, therefore no filter and no selection on the outcome variable.

Measured on the review's own section 4.2 shape — 25 symbols, 20 days,
`noise_frac=1.0`, 240 observations per symbol, true sigma supplied:

| | old log-log | new direct NLS |
|---|---|---|
| usable fits | 25 of 25 | 17 of 25 |
| reported failures | 0 | 8 |
| max abs delta error | 1.763 | **0.490** |
| median abs delta error | not recorded | 0.344 |
| negative deltas | 6 of 25 | **0** |
| delta outside (0, 1] | SYN07 = 2.263 | **0** |

The eight that now fail are the ones that were previously returning garbage. A
delta outside (0, 1] or a negative `Y` is a failed calibration, raised where
the evidence is rather than deferred to `to_model()`. That closes finding I6,
which section 4.2 showed occurring on ordinary generated data.

Delta is fitted over [0.01, 3.0] rather than clamped, so a convex result is
observed and reported instead of squeezed into looking plausible.

**This does not make calibration usable at 240 observations per symbol.** A
median error of 0.34 is still too large. Section 4.2's conclusion stands.

### 2.4 Ragged covariance coverage — **closed**

`log_returns(daily, *, max_dropped_fraction=0.10)` raises
`InsufficientHistoryError` naming the dropped count, the percentage and the
per-symbol gaps responsible. Row 0 is null by construction and no longer
counts against the budget. Duplicate `(date, symbol)` rows get a typed
`DuplicateObservationError` instead of the raw polars `ComputeError`.

### 2.5 Session-aware bucketing — **closed**

`core/session.py` is the single source of truth.
`TradingSession.buckets_per_day` is exact and rejects a width that does not
divide the session, replacing the `int(round(...))` that reported 6 buckets
per day for a 1-hour bucket in a 6.5-hour session — a silent 4.1% error in
every sigma derived from it.

`Boundary.OPENING` and `Boundary.CLOSING` name the two half-open conventions.
Contiguity is now tested on `session_index`, never on `bucket_id`: consecutive
sessions are adjacent in `bucket_id`, which is precisely how overnight returns
were leaking in.

Measured on a dense L1 series spanning the extended session, 2026-03-03/04
NYSE, 1170s buckets: epoch-floored bucketing makes **all 92** transitions
contiguous, so a 20% overnight jump is scored as intraday impact.
`test_overnight_gap_is_excluded_on_a_real_nyse_calendar` pins this and fails
when the contiguity test is mutated back onto `bucket_id`.

Rows outside the session are kept as nulls by the helper and raise
`OutOfSessionError` at each consumer, with an explicit `allow_out_of_session`
escape hatch. A bundle now records its trading calendar
(`DatasetBundle.session`) and raises rather than defaulting.

---

## Section 4 — findings from the data-forging exercise

### 4.1 Generator variance fiction — **closed**

Total bucket variance is now held at `sigma_bucket**2` by dividing both the
impact and diffusion terms by `sqrt(Y**2 * E[abs(f)**(2*delta)] + noise_frac**2)`,
with the second moment estimated by fixed-seed Monte Carlo (0.14s for 25
symbols; generation cost is unchanged at 7.2s for 25 × 20 days).

`noise_frac` keeps its meaning as the impact/diffusion split.
`ground_truth()["impact_Y"]` now reports the *effective* coefficient actually
injected — the one calibration can recover — with `impact_Y_configured` kept
for traceability.

**Both `delta` and `Y` are now recoverable from the same bundle**, which the
review recorded as impossible for any single synthetic config. Measured end to
end through the production path with sigma *estimated, not supplied*,
3 symbols × 40 days at `noise_frac=0.05`:

| | before | after |
|---|---|---|
| max abs delta error | — | 0.020 |
| max abs Y error | 357–464% | **8.0%** |
| min r-squared | — | 0.996 |

Locked in as `test_calibrate_bundle_recovers_delta_and_y_with_sigma_estimated`,
which also closes two section 3 findings: the gate now goes through
`calibrate_bundle` rather than `fit_power_law`, and leaves sigma unsupplied so
`estimate_bucket_sigma` is under test rather than bypassed.

`noise_frac=0` is now rejected: a mid with no diffusion leaves the
renormalisation nothing to trade off against impact.

### 4.2 Calibration at realistic noise — **improved, not solved**

See 2.3. Max error 1.763 → 0.490 and no silently invalid results, but a
median of 0.344 at 240 observations per symbol is still unusable.

**Carry into M2.** The first data request (spec 8.4) asks for ~20 trading days
of L2 and 5 days of L3. That is not enough history to calibrate impact. Either
lengthen the request, or treat calibrated `delta` as an assumption rather than
a measurement and run the crossover study across a delta grid.

### 4.3 `compare_to_l2` throughput — **closed**

Grouped L2 once into a `(ts_ns, side) -> levels` map instead of filtering
inside the loop, and replaced `iter_rows(named=True)` in `snapshots_at` and
`final_book` with a single columnar pass. On the review's own fixture, 6,500
snapshots across 25 symbols:

| | before | after |
|---|---|---|
| wall time | 23.4s | **0.47s** |
| throughput | 278 snap/s | **13,773 snap/s** |

Guarded by a 3,000 snap/s throughput floor rather than a scaling ratio — a
ratio between two small timings is dominated by fixed overheads and flakes.

**Correction to the original measurement.** The review's 10.7x-for-8x
superlinearity did not reproduce. Per-snapshot cost on a single symbol was
flat across 65 / 260 / 780 / 1560 snapshots both before and after the fix, so
the dominant cost was the per-message dict allocation and not the quadratic
filter. The quadratic term was real and is gone, but it was not what the
original figure was measuring.

### 4.4 Generation cost — **not addressed**

Still superlinear in days. 25 × 20 days is 7.2s, as before. Acceptable for
fixtures, worth knowing before generating multi-year data.

### 4.5 CLI knobs — **closed**

All 16 `SynthConfig` fields are now flags, with a test asserting set equality
against `dataclasses.fields(SynthConfig)` so a new field cannot be added
without a way to reach it. `--impact-delta` and `--impact-y` take
`SYMBOL=VALUE` pairs and report unknown and missing symbols without a
traceback.

---

## Also closed, from section 3

- **Books from L1 and L2.** A `BookSnapshot` could only come from L3 replay,
  so under spec 8.4's shape no liquidity metric was computable for ~20 of 25
  symbols and spec 5.2's spread-cost term had no route from `l1_taq` at all.
  `micro/books.py` adds `books_from_l1`, `books_from_l2` and `bucket_spreads`,
  the last giving `s_i / 2` a direct route from data for every symbol with L1.
- **`resiliency_halflife` had no caller and no data path.** `spread_series` is
  that path.
- **`fit_power_law` could return a structurally unusable result.** Closed by 2.3.
- **The M1 gate never exercised `estimate_bucket_sigma`.** Closed by 4.1.
- **Crossed and degenerate L1 quotes reached the calibration path.** Guarded at
  `observations_from_buckets` and at `bucket_spreads`, not only at the bundle
  boundary.
- **The reconstruction gate claimed cross-validation it did not have.** The
  docstring now says what the gate actually demonstrates on synthetic data —
  three independently written book implementations agreeing — and
  `test_l3_l2_holdout_against_real_data` is the named, skipped holdout that
  makes spec 10.3's claim true once real paired L2/L3 exists.
- **Zero-row partitioned table.** `table()` raised a raw polars `ComputeError`;
  it now returns the declared schema with zero rows.
- **`test_layer_contract_holds` resolved `lint-imports` via PATH.** Now
  resolved next to `sys.executable`.

---

## Still open

Carried forward unchanged from section 3.

**Important**

- `strict=False` drops messages with no observability, and is asymmetric: it
  tolerates unknown-order cancels but raises on a duplicate `order_id`, the
  most common real-export defect. Needs a drop counter surfaced from
  `snapshots_at`, and a deliberate decision about duplicate adds.

**Minor**

- `build_l1` sets `last_size = abs(net_flow)`, so summing it gives 8.6% of
  true volume.
- L1 is stamped at `ts_end_ns`, L2 at `ts_end_ns - 1`; intentional and
  load-bearing, still undocumented at both stamp sites. `bucket_spreads` now
  documents the consequence for callers.
- No test cross-checks L1 against L2 or daily.
- Four tests are tautological, notably `test_annualisation_scales_the_matrix`.
- The R22 replenish guard duplicates the generator's `buy_vol` formula by hand.
- The layer contract is a `layers` contract, which permits `solvers` to import
  `problem` internals; spec 12 forbids it. Needs a `forbidden` contract at
  M3/M4. `exhaustive = false` leaves future subpackages unconstrained.
- Two ADV implementations: `synth.build_daily_bars` writes an `adv` column,
  `liquidity.adv` recomputes and discards it.
- `round_to_tick` is dead code with its own test.
- `BookBuilder._levels` keys on raw floats with no quantisation.
- Error types split across `ValueError` and `RuntimeError` with no common base.
- `bundle.py` carries a `# type: ignore` with no type checker configured.
- `Catalog.register` returns a stale entry on idempotent re-register if the
  bundle moved.
- `_execute_against` has an `if not progressed: break` that under-executes
  silently.
- Generation cost superlinear in days (4.4).
