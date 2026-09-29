# M3a gate — first complete run, what it found, and two reviews

**Written:** 2026-09-29
**Branch:** `feat/m3a-encoding`
**Follows:** `2026-09-28-m3a-gate-handoff.md`
**Plan:** `docs/superpowers/plans/2026-09-25-m3a-encoding.md`, Task 12 Steps 2–5

---

## Verdict

**The gate passes.** Sixteen tests, no finding on any of the 160 cases it encodes.

It did not pass when it first ran to completion. It failed three checks on 33 cases,
for two reasons, both of them real: a reference that was wrong, and a CVaR encoding
whose energy did not rank schedules the way the objective does. Two reviews then
found that the gate could not see a wrong gadget, and that the first repair held only
at the default safety factor.

| what | first complete run | now |
|---|---|---|
| gate | 10 of 13 pass | **16 of 16 pass** |
| energy equality, total | fails, 16 cases | passes |
| energy equality, term by term | fails, 8 cases | passes |
| one-lot violation raises the energy | fails, 17 cases | passes |
| full suite | 906 pass, 1 skipped, 1 fails | **941 pass, 1 skipped, 0 fail** |
| `ruff check src tests` | clean | clean |
| `lint-imports` | 1 kept, 0 broken | 1 kept, 0 broken |
| gate wall time | 5 min 40 s | 9 min 28 s, with other work on the machine |

The first-run column is the gate as handed off, with only the fold of `595c69c`
applied so that it could finish; the full-suite figure in it was taken one commit
later, after Finding A was fixed. The gate is slower now because it checks more: a
second energy at every CVaR case, and a lot left unsold at every cell the witness
holds rather than at one.

The skip is `tests/micro/test_reconstruction_gate.py:104`, which needs a real paired
L2/L3 bundle named by `QUANTIC_REAL_BUNDLE`. It is unrelated to this milestone.

Machine: Windows 11, i9-9900, 63.9 GB, Python 3.11.15, pytest 9.1.1.

**What passing does and does not establish** is set out under "What is still open".
In short: the encoder equals the objective it declares, on every case. That the
QUBO's global minimum is feasible is proved by enumeration on instances small enough
to enumerate, and is not something the gate can show at the ladder's size.

## What changed on the branch

| commit | what | found by |
|---|---|---|
| `595c69c` | perf: fold repeated keys before expanding a product | the handoff's question |
| `0d3e316` | fix: the substituted impact term dropped the cells it never fitted | the gate |
| `6890720` | fix: decode refuses a vector that is not binary | first review |
| `502ee2b` | fix: size the CVaR auxiliary grids at zeta's width | first review |
| `f4d49a3` | test: the gate ties the CVaR gadget back to `problem/` | first review |
| `7ae04ff` | fix: derive the CVaR strength from the gadget's own trade-off | the gate, and first review |
| `683592c` | fix: the hard strengths cover the gain in risk under D3 | the gate, and first review |
| `af5f2df` | feat: the auxiliaries the QUBO's own minimum takes | second review |
| `9367baf` | fix: bound the risk a unit of violation buys, not the risk a cell buys | second review |
| `cd723a2` | test: the gate sees a gadget that is wrong but agrees with itself | second review |

Each fix was written test-first, and each test was watched failing for the stated
reason before the code changed. Where a test was later moved to a different
instance, each defect was put back by monkeypatch and the test run against it.

## The handoff's question, answered

The handoff asked whether the encode cost was inherent or a hot spot. **It was a hot
spot.**

`cProfile` on `encode(T2-D3, binary)` put 99.9% of 44.7 s inside `add_square`, called
from `expand_cvar`. The Rockafellar–Uryasev residual sums one holdings form per
(asset, bucket), and a holdings form names every cell traded up to that bucket — so
each lot bit appears once per bucket it is still held through. `add_product` expanded
those repeats pairwise. Counted without building anything, that came to about 14
billion `add_quadratic` calls across the gate's ladder, 12.3 billion of them in the
eight T3 cells with D3 on.

`LinearForm.folded()` sums repeated keys; `add_product` folds both operands first.

| `encode`, binary | before | after |
|---|---|---|
| `T2-D3` | 14.8 s | 0.9 s |
| `T3-D1+D2+D3+D4` | 448.5 s | 12.2 s |
| `T3-D1+D3` | not measured | 43.1 s |

The handoff's probe of `T3-D1+D2+D3+D4` gave up at 590 s on a contended 15.7 GB
machine; 448.5 s is the same call, uncontended, on this one.

**What it changes numerically.** Nothing where no key repeats: `T0-none` and `T1-D1`
produce bit-identical `Q` under all three encodings. Where keys do repeat — only the
CVaR penalty — a coefficient is now a product of sums rather than a sum of products,
and `Q` differs by at most 1.1e-16 of its largest entry. Derived penalty strengths
are bit-identical. The reviewer checked this independently over 18 instance and
encoding pairs, by disabling the fold, and reached the same figures.

The handoff's estimate of 1.5–2 hours for the unfixed gate was a soft lower bound; the
call count puts it nearer three. That run was started, and stopped once the gate's
findings on T0 and T1 had been shown identical, key for key, with and without the
fold.

## Finding A — the reference dropped the cells it never fitted. Fixed.

`0d3e316`. Energy equality failed on all sixteen cells with D1 and D4 both on under
the log encoding — four dial combinations at each of four tiers — and the
term-by-term check named `temporary` in each of the eight it then covered.

The QUBO was right; the reference was wrong. A block cell is one bit, so its cost is
exact and the quadratic fit never sees it. `ApproximandReport` recorded only fitted
cells, so `temporary_cost` — which `reference_breakdown` swaps in for the evaluator's
term — summed over those alone.

| instance | QUBO − reference | exact cost of the block cells |
|---|---|---|
| `T0-D1+D4` | 2.839060329543 | 2.839060329543 (3 cells) |
| `T0-D1+D2+D4` | 2.190890230021 | 2.190890230021 (1 cell) |
| `T1-D1+D4` | 3.817197986764 | 3.817197986764 (7 cells) |
| `T1-D1+D2+D4` | 16.086183443495 | 16.086183443495 (2 cells) |

The report now carries the exact table of every cell it did not fit. No unit test had
combined D1 with D4 under the log encoding; three do now.

This matters beyond the gate: M3b's optimality bracket is stated against
`reference_breakdown`.

## Finding B — with D3 on, the energy did not rank schedules. Fixed, in three steps.

### What the first run showed

`test_a_one_lot_violation_raises_the_total_energy` failed on 17 of the 20 cells where
it applies with D3 on. Leaving a lot unsold *lowered* the energy:

| instance | encoding | energy at witness | one lot short | change |
|---|---|---|---|---|
| `T0-D1+D3` | binary | 10,561.0 | 3,381.0 | −7,180.0 |
| `T1-D3` | binary | 511,042.7 | 407,988.2 | −103,054.4 |
| `T1-D1+D3` | one_hot | 8,218,010.5 | 6,736,048.9 | −1,481,961.6 |
| `T2-D3` | one_hot | 27,941,999.8 | 19,614,510.7 | −8,327,489.1 |
| `T3-D3` | binary | 1,280,729,509.1 | 1,259,606,062.7 | −21,123,446.4 |

(Five of the seventeen; the rest follow the same pattern.)

### Why

The gadget's penalty is `P · Σ_s (u_s − L_s + ζ − g_s)²`. `u`, `g` and `ζ` live on a
grid of step `δ`; `L_s` is linear in the holdings with real coefficients. So at the
best auxiliaries each residual is the distance from `L_s − ζ` to the nearest grid
point — anywhere in `[0, δ/2]`, effectively arbitrary in the schedule, and **never
zero**. Design §7 says as much, and `CVaRGadget.resolution` reported the worst case
honestly, which is why the resolution check passed throughout.

What was not drawn out is the size of that leftover. `δ` is the instance-wide loss
span over 255 — 1.7 to 15 at T0, 1,618 to 3,951 at T3 — and the leftover scales as
`P · S · δ²`. Decomposed at `T0-D1+D3`, binary, as first run:

| | spread | temporary | permanent | CVaR cost | **CVaR penalty** | hard penalties |
|---|---|---|---|---|---|---|
| witness | 24.00 | 16.48 | 0.16 | 40.34 | **10,480.07** | 0.00 |
| one lot short | 23.00 | 15.45 | 0.14 | 48.04 | **3,264.12** | 30.24 |

The hard constraint charged its 30.24 exactly as derived. The CVaR penalty moved by
7,216 underneath it. Across every D3 cell, binary, the leftover was 13 to 129 times
the whole objective at T0 and 484 to 4,120 times at T1.

The cause is a departure from the design. §8: "`min_penalty_increment` is `1` for the
integer-residual squares (full liquidation, cardinality) and is **derived per
constraint otherwise**." `derive_penalties` gave all five families one strength, on
the stated ground that every residual is an integer-valued square. The CVaR residual
is in currency.

### Step one — the CVaR strength

`7ae04ff`. The CVaR strength is now the one at which dropping `u_s` a step below a
target on the grid costs more than it saves, by the safety factor and no more:

```
P_cvar = safety · λ · max_s w_s / ((1 − α) · δ)
```

Enumerating all 38,440 feasible schedules of `T0-D3`, minimising over `ζ` and the
auxiliaries for each:

| | before | after |
|---|---|---|
| Spearman, energy against objective | 0.018 | 0.985 |
| rank of the energy's cheapest schedule, under the objective | 34,968 | 10 |
| its regret in the objective | 21.80 | 0.75 |

Before, the QUBO's best schedule was worse than nine in ten of the alternatives. The
enumeration is the reviewer's; it was re-run here, before and after.

Every D3 cell, binary, at the feasible witness:

| tier | objective | leftover, before | leftover, now | now ÷ objective | resolution ÷ objective |
|---|---|---|---|---|---|
| T0 | 3 – 81 | 40 – 10,480 | 0 – 5.7 | 0.00 – 0.14 | 0.00 – 0.79 |
| T1 | 134 – 498 | 91,785 – 2,053,295 | 7.7 – 38 | 0.04 – 0.09 | 0.23 – 0.30 |
| T2 | 268 – 1,195 | not measured | 24 – 273 | 0.09 – 0.24 | 0.60 – 1.34 |
| T3 | 3,095 – 14,372 | not measured | 817 – 3,903 | 0.18 – 0.45 | 0.99 – 1.81 |

### Step two — the hard strengths

Step one left the one-lot check failing on 2 cells, and passing on the rest by margin
rather than by construction:

| instance | encoding | energy at witness | one lot short | change |
|---|---|---|---|---|
| `T2-D3` | binary | 815.93 | 811.65 | −4.28 |
| `T2-D3` | domain_wall | 815.93 | 811.93 | −4.00 |

`encode` derives the hard strengths from an objective built with the CVaR penalty at
zero. No lot bit is coupled to the risk term in it, so the hard strengths bounded the
gain in spread and impact and **not the gain in risk**; they did not move with `λ` at
all. The first reviewer measured the consequence: at `T0-D1+D3` the strength stayed at
30.2 while the worst one-lot gain in the true objective was 5.8 at the balanced `λ`,
45.1 at ten times it and 438 at a hundred.

Three ways out were put to the owner: bound the risk gain and add it to the hard
strengths; spend more `ζ` bits; or scope the guarantee to D3-off cells. **The first
was chosen.** A fourth — reading the bound off the QUBO's own coefficients, as
`max_single_cell_gain` does for the other terms — had been tried and is not an
option: it gives strengths of 1.7e6 to 8.9e6 at T0 and T1, at which size round-off in
the penalty block alone breaks energy equality at `1e-9` on 32 cases.

`683592c`. `CVaRGadget.risk_gain_bound` bounds the gain from the problem. CVaR is
monotone and translation-equivariant, so it moves by no more than the largest change
in any one scenario's loss; two resolutions and a cost gap carry the comparison from
continuous Rockafellar–Uryasev to the encoded value at a snapped `ζ`.
`derive_penalties` adds it to the gain it reads off the matrix.

### Step three — a unit of violation is not a cell

`9367baf`. The second reviewer found that step two held only because the safety
factor defaults to exactly two, and showed it.

As first written, the bound was on what changing *one cell* can do to a loss: the
cell's reach times the tail sum of its coefficients. But a breach of cardinality is
repaired by moving lots *between two buckets* of one name, which moves only the
holdings in between — a partial sum. A partial sum can be twice the largest tail sum,
because the buckets after the second, whose coefficients the tail sum nets off, are
not touched.

Two names, two buckets, one name a bucket, and a tail scenario in which both names
fall 6% in the first bucket and recover 3% in the second. Every state of the lot bits
enumerated, the gadget minimised exactly over `ζ` and the auxiliaries for each:

| safety | hard strength | best feasible | best infeasible | global minimum |
|---|---|---|---|---|
| 2.0 | 2,900.4 | 1,843.88 | 2,906.48 | feasible |
| 1.2 | 1,638.1 | 1,823.86 | 1,644.17 | **infeasible** |

The winning state sells both names at once. It reproduces under all three encodings.
`encode(safety_factor=...)` is public, and M4's adaptive strategy exists to lower it.

The bound now takes the largest partial sum over any run of buckets. A tail sum is
the partial sum that runs to the end, so the one bound covers a lot left unsold and a
lot moved. **The lowest energy in the landscape is feasible at safety factors of 2.0,
1.5, 1.2 and 1.05, under all three encodings.**

On the ladder the two bounds nearly coincide — its scenarios rarely reverse — so the
strengths are those of step two to within 0.2%:

| tier | hard strength under D3, before | now | ratio |
|---|---|---|---|
| T0 | 6 – 32 | 6 – 281 | 1.0 – 15.3 |
| T1 | 44 – 90 | 278 – 829 | 4.5 – 11.4 |
| T2 | 44 – 93 | 915 – 8,044 | 20.5 – 86.8 |
| T3 | 106 – 285 | 17,117 – 87,611 | 116 – 337 |

**The dynamic range did not move**, on any of the 32 D3 instances. Its extremes are
set by the gadget's own coefficients, not by the hard constraints. The cost that was
expected of this option when it was chosen did not arise.

### What is still open

**With 8 `ζ` bits, the declared resolution at T2 and T3 is the size of the
objective** — the last column of the table under step one. The encoding is correct
there and says what it costs; what it costs is a great deal. Each further bit halves
it, at 31 variables a bit with 15 scenarios, in a budget where no D3 cell fits
already. This is a choice of bit budget for M5 and was not made here.

**The gate cannot prove the hard strengths sufficient.** The one-lot check is the
testable form of the guarantee at the ladder's size: with the risk bound removed
altogether it notices nothing at T0 or T1, and 30 cases at T2. The proof is the
enumeration in `tests/encoding/test_encode.py`, on instances of 14 to 18 lot bits,
which fails at every safety factor with the bound removed.

**`max_single_cell_gain` is per cell too.** The gain in spread, impact and variance
is still bounded by what one cell's bits can do, and a two-cell move could in
principle buy twice that. It is 130 to 500 times looser than the worst true gain on
the variance instances, so nothing is known to break — the first reviewer brute-forced
every T0 variance QUBO of 30 bits or fewer and found the global minimum feasible in
all 17. But at a safety factor below two that slack is what the guarantee would be
resting on. Pre-existing, not changed here.

**`optimal_aux` is not the QUBO's minimum**, and M3b must not use it as one. See the
second review, I2.

## First review — the whole branch

One reviewer, range `8244a9f..93f37b5` — `main` to the branch head plus the fold. It
read the diff in passes, ran twelve probes, and brute-forced every T0 variance QUBO
of 30 bits or fewer: 17 instance and encoding pairs, up to 2^30 states, global
minimum feasible in all of them. Its verdict was **ready to merge with fixes**.

`93f37b5` has the same tree as `595c69c`; the commit was re-made to correct a timing
in its message.

| finding | severity | disposition |
|---|---|---|
| C1 — with D3 on the QUBO does not rank schedules as the objective does | critical | fixed, `7ae04ff`; enumeration re-run |
| C2 — the reference drops exact cells under binary with D1 and D4 | critical | fixed, `0d3e316`, found independently |
| I1 — the gate's CVaR checks pass with a wrong loss model | important | fixed, `f4d49a3` and `cd723a2`; reproduced first |
| I2 — under D3 the hard strengths ignore the risk term | important | fixed, `683592c` and `9367baf` |
| I3 — `decode` accepts non-binary input | important | fixed, `6890720` |
| I4 — the `u` and `g` grid width depends on float rounding | important | fixed, `502ee2b` |
| I5 — the penalty-zero floor has 12–22× headroom at T3, not six orders | important | **not taken** |
| I6 — `encode` expands the objective twice; pairwise loops remain | important | **not taken** |
| M1–M8 | minor | **not taken** |

**On I1.** With the gadget's loss coefficients sign-flipped, the gate reported exactly
what it reports with them right. Every CVaR check was stated against the gadget's own
losses. The gate now compares them with `ScenarioSet.losses` on every CVaR case, and
compares spread, temporary and permanent impact term by term under every risk spec,
where before all 32 CVaR instances were skipped.

**On I5.** The reviewer measured round-off of 2.4e-7 to 6.0e-7 in the penalty block
at T3 against a bound of 2.9e-6 to 1.0e-5, and suggests `1e-6 × strength`. The bound
and its reasoning are written into design §12.3, so changing it is a change to the
design. It passes today.

**On I6.** The risk term is still over 95% of encode time, in pure Python. A
vectorised `add_product` would remove it.
`test_a_product_costs_the_distinct_keys_not_the_repeats` counts `add_quadratic` calls
and will need restating when that is done.

**Minor, not taken.** M1: `layout.py` conflates "log encoding" with "table encoding
with zero bits", so a zero position under domain-wall with D1 on emits a report for a
table encoding; not reached by the ladder. M2: a block larger than the per-bucket cap
is accepted. M3: `ζ` and auxiliaries outside the grid are clamped silently. M4: the
gain bound counts the diagonal three times, so strengths are 130 to 500 times the
worst true gain on variance instances. M5: the one-lot check covers one cell and
under-liquidation only — it now covers every held cell (`cd723a2`), still
under-liquidation only. M6: the approximand is checked finite, not bounded. M7: a
`# noqa: BLE001` for a rule that is not selected. M8: `EncodedInstance.qubo` rebuilds
the dense sum on every access.

## Second review — the fixes

One reviewer, range `595c69c..683592c`: the six commits made after the first review.
It was asked to look hardest at the two derivations, and to try to break them. It
built an exact minimiser over `ζ` and the auxiliaries, validated against `Qubo.energy`
to 3e-15, and ran it exhaustively over all 15,625 lot-grid schedules of a 2×3
instance, ten random heavy-tailed instances and samples of `T0-D3` and `T0-D1+D3`.

**It found no failure at the default settings.** The single-cell risk bound held
everywhere it was tested, on the canonical states and on the auxiliary-minimised
landscape, and `resolution` remained a valid bound at the new strength. Its verdict
was **ready to merge with fixes**.

| finding | severity | disposition |
|---|---|---|
| I1 — a unit of violation can buy two cells of risk; the guarantee holds only at a safety factor of two | important | fixed, `9367baf`; reproduced first |
| I2 — at the derived strength `optimal_aux` is not the energy's minimum | important | fixed, `af5f2df` |
| I3 — the gate cannot see a wrong gadget in several forms | important | fixed, `cd723a2`; reproduced first |
| I4 — the tests pinning the derivations run where the excess is identically zero | important | fixed, `9367baf` |
| M1 — `ApproximandReport`'s docstring says a zero-capped cell goes in `exact`; under binary it is fitted as `(0, 0)` | minor | **not taken** |
| M2 — first review's M1, now with a correct `temporary_cost` | minor | **not taken** |
| M3 — `temporary_cost` raises on an exact cell's inadmissible value, not on a fitted one's | minor | **not taken** |
| M4 — `T0-D3+D4` has `λ = 0`, so it is a D3 case with no CVaR in it | minor | **not taken** |
| M5 — `PenaltyStrengths.describe()` does not record the risk-gain component | minor | **not taken** |
| M6 — the energy at a feasible state depends on the safety factor | minor | **not taken** |
| M7 — a stale comment in the gate | minor | fixed, `cd723a2` |
| M8 — `risk_gain_bound` is looser than the argument needs | minor | **not taken** |

The three commits that answer this review have not themselves been reviewed.

**On I2.** `optimal_aux` puts each `u_s` on the grid point nearest its target, which
minimises the penalty. A target is rarely on the grid, and a step below the nearest
point can save more in cost than it adds in penalty. The reviewer measured it off the
minimum on 175 of 225 schedules of a 20-scenario instance and 48 of 4,000 sampled
from `T0-D3`. Nothing in the gate depends on which is used. **M3b does:** design §13
has it enumerate `x` and `ζ` and solve the auxiliaries in closed form, and with
`optimal_aux` that is not the QUBO's minimum. `CVaRGadget.cheapest_aux` is the closed
form for the energy. `optimal_aux` stays the canonical encoding, and the one
`resolution` is stated at.

**On I3.** The reviewer ran the gate's own check loop against a gadget broken eight
ways and it passed all eight. On T0 and T1, 96 cases, the gate now reports:

| the gadget, broken | the gate now reports |
|---|---|
| loss coefficients ×(−3) on cells the witness does not hold | `loss_model`, 48 |
| `derived_penalty` × 0, × 1e-6, × 1000 | `gadget_inputs`, 45 each |
| scenario weights halved inside the gadget | `gadget_inputs`, 48 |
| gadget `α` 0.95 changed to 0.99 | `gadget_inputs`, 48 |
| gadget `λ` doubled | `gadget_inputs`, 45 |
| `risk_gain_bound` returns 0 | nothing at T0 or T1; `one_lot`, 30 at T2 |

The loss model is compared at the schedule that sells nothing as well as at the
witness, because a loss is a coefficient times a holding and the witness no longer
holds most cells. The gadget's inputs and its strength are compared with the
instance's. And energy equality is checked a second time at the smallest loss: at the
optimal `ζ`, with `α` at 0.95 and fifteen scenarios, the tail holds one scenario or
none, no `u` bit is set, and the first check never touches the excess term.

**On I4.** Six equal-weight scenarios at `α = 0.9` make CVaR the largest loss. Only 3
of 15,625 canonical states set any `u` bit, which is how I2 went unnoticed. The tests
now run on twenty unevenly weighted fat-tailed scenarios at `α = 0.7`, where 222 of
the 225 feasible schedules carry excess in four or more. One test asserted that the
strength exceeded the safety factor times the bound, which is true of any bound; it
is gone.

**On M4.** `T0-D3+D4`'s witness holds nothing after the first bucket, so the balanced
`λ` is zero. Its QUBO carries about 240 auxiliary variables with all-zero
coefficients, and the variable budget counts them. That is the generator's doing and
`problem/` was outside both reviews.

## Variable budget — Task 12 Step 3

`variable_budget(ladder)`, `SMALL = {"n_raw": 300, "n_target": 15}`, 8 `ζ` bits, against
`DENSE_EMBEDDING_CEILING = 230`. **58 of 192 cells fit.**

| tier | encoding | n_vars | fit | the cells that fit |
|---|---|---|---|---|
| T0 | binary | 16 – 276 | 8/16 | every D3-off combination |
| T0 | domain_wall | 16 – 284 | 8/16 | every D3-off combination |
| T0 | one_hot | 27 – 296 | 8/16 | every D3-off combination |
| T1 | binary | 72 – 368 | 8/16 | every D3-off combination |
| T1 | domain_wall | 72 – 528 | 6/16 | D4, D2, D2+D4, D1+D4, D1+D2, D1+D2+D4 |
| T1 | one_hot | 112 – 568 | 4/16 | D2, D2+D4, D1+D2, D1+D2+D4 |
| T2 | binary | 112 – 440 | 8/16 | every D3-off combination |
| T2 | domain_wall | 112 – 696 | 4/16 | D2, D2+D4, D1+D2, D1+D2+D4 |
| T2 | one_hot | 176 – 760 | 4/16 | D2, D2+D4, D1+D2, D1+D2+D4 |
| T3 | binary | 894 – 1,688 | 0/16 | — |
| T3 | domain_wall | 1,037 – 5,516 | 0/16 | — |
| T3 | one_hot | 1,370 – 5,876 | 0/16 | — |

Three things follow.

- **No D3 cell fits, at any tier, under any encoding** — including T0. Design §13
  expected this at T1 and T2.
- **No T3 cell fits.**
- **Under a table encoding at T1 and T2, only the D2-on cells fit.** D2 removes the
  values a venue minimum forbids, and under one-hot and domain-wall that removes
  bits. It is the §6.1 result — D2 is free under a table encoding — showing up as
  the difference between reaching the hardware and not.

The T0 maxima are lower than in the handoff's table: 276, 284 and 296 against 306,
306 and 318. The grid-width fix (`502ee2b`) took a bit off every auxiliary grid of
two instances, `T0-D2+D3` and `T0-D3+D4`, thirty variables each; `T0-D2+D3` had been
the maximum under all three encodings. No cell changed sides of the ceiling. Every
other row agrees with the handoff.
