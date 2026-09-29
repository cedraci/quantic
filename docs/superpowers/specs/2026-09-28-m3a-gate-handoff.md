# M3a gate — handoff notice

**Written:** 2026-09-28
**Branch:** `feat/m3a-encoding` @ `91f316c`, pushed to `origin`
**For:** the next agent, running on a 64 GB machine
**Plan:** `docs/superpowers/plans/2026-09-25-m3a-encoding.md`, Task 12

---

## What you are picking up

Task 12 Step 1 is done: `tests/encoding/test_m3a_gate.py` is written, committed and
pushed. Steps 2–5 are not. **The gate has never been observed to pass.** Your job is
to run it, deal with whatever it says, and finish the task.

Do not assume it passes. Do not assume it fails. It has not been measured.

## State of the repository

Everything is on `origin`, nothing is stashed, the tree is clean.

| branch | head | remote |
|---|---|---|
| `feat/m3a-encoding` | `91f316c` | tracks `origin/feat/m3a-encoding` |
| `main` | `8244a9f` | tracks `origin/main`, fast-forwarded this session |

`feat/m3a-encoding` contains all of `main` and is 18 commits ahead of it.

## What is actually verified

| check | result |
|---|---|
| `ruff check tests/encoding/test_m3a_gate.py` | clean |
| `test_the_ladder_is_still_sixteen_combinations_on_four_tiers` | passes |
| `test_the_variable_budget_covers_every_cell_of_the_study` | passes, all 192 rows |
| `test_encoding_is_reproducible_across_calls` | passes |
| the ten `gate`-fixture tests | **never completed a single run** |

The three that pass are the ones that do not touch the module-scoped `gate` fixture.
They run in 11.2s together:

```bash
.venv/Scripts/python.exe -m pytest tests/encoding/test_m3a_gate.py -v \
  -k "sixteen_combinations or variable_budget or reproducible"
```

## Why it never ran here, and why your machine changes that

Two full attempts were killed by Claude Code's background-shell watchdog, which reaps
background tasks when *system-wide* free memory runs low while the session is idle.

| run | last observed before the kill | gate tests completed |
|---|---|---|
| 1 | ≥ 1121 CPU-seconds (≈19 CPU-min) | 0 |
| 2 | ≥ 2489 CPU-seconds (≈41.5 CPU-min, 45.2 wall-min, ~92% of one core) | 0 |

**The test was not the cause of the memory pressure.** pytest held steady at 92–113 MB
resident across both runs. The pressure came from the desktop — two `msedgewebview2`
processes at ~1 GB each, plus Teams and Chrome — on a 15.7 GB machine that was down to
1.2 GB free. The watchdog reaps by system state, not by the task's own footprint, so a
100 MB pytest process was killed for something else's allocation.

On 64 GB this should simply not arise. **But note what that does and does not fix:**
it removes the reaper as an obstacle. It does not make the run fast. The run is
CPU-bound and single-core; more RAM buys you completion, not speed.

> The commit message on `91f316c` says the two runs were killed "at 41 and 48
> CPU-minutes". That is wrong — it conflates run 2's CPU time with its wall time and
> omits run 1. The table above is correct. The commit was already pushed, so it was
> left rather than force-corrected.

## How long to expect

Measured on the machine above (uncontended unless noted). `encode()` dominates
completely; every per-case check is ≤ 0.08s.

| tier | encodings | n_vars | `encode` for the *first* instance of the tier |
|---|---|---|---|
| T0 | all three | 24–48 | ~0.06s total |
| T1 | all three | 120–312 | ~2.7s total |
| T2 | all three | 192–496 | ~6.9s total |
| T3 | binary only | 1440 | 14.2s |

That table is misleading on its own, and this is the single most important thing in
this notice: **encode cost varies by more than 40× *within* a tier depending on the
dials.** `t3[0]` is `T3-none` — every dial off, the cheapest instance in the tier.
`t3[15]` is `T3-D1+D2+D3+D4` — every dial on. A probe of `t3[15]` **did not finish a
single `encode()` in 590 seconds.**

`_cases()` walks the ladder in order, so the heaviest instances come last. A run that
dies late loses everything. Expect **1.5–2 hours**, and treat that as a soft lower
bound, since the tail was never measured to completion.

## The performance question worth asking

The plan's Task 12 rationale asserts that encoding once rather than once per test "is
the difference between minutes and the better part of an hour", and that T3 under the
log encoding "stays tractable". The observed behaviour does not match: one T3 binary
instance at 1688 variables exceeding ten minutes is far outside "minutes".

So there are two possibilities, and they need different responses:

1. **This is inherent cost.** Then the gate is correct but expensive, and the right
   follow-up is to mark the heavy T3 dial combinations so the fast majority can be
   selected separately — without weakening the gate, which is M3a's load-bearing
   artifact.
2. **There is a hot spot in the encoder.** Then the gate is telling you about a real
   defect in `src/quantic/encoding/`, and fixing it is worth more than any test
   restructuring, because M3b and M5 will pay the same cost repeatedly.

Get the gate green first, then profile one `encode(t3[15], BinaryEncoding())` before
deciding. Do not restructure the test to make the symptom go away.

## Variable budget, already measured

From `variable_budget(ladder)` with `SMALL = {"n_raw": 300, "n_target": 15}`:

| tier | encoding | min n_vars | max n_vars | dense MB at max |
|---|---|---|---|---|
| T0 | binary | 16 | 306 | 0.7 |
| T0 | domain_wall | 16 | 306 | 0.7 |
| T0 | one_hot | 27 | 318 | 0.8 |
| T1 | binary | 72 | 368 | 1.1 |
| T1 | domain_wall | 72 | 528 | 2.2 |
| T1 | one_hot | 112 | 568 | 2.6 |
| T2 | binary | 112 | 440 | 1.5 |
| T2 | domain_wall | 112 | 696 | 3.9 |
| T2 | one_hot | 176 | 760 | 4.6 |
| T3 | binary | 894 | 1688 | 22.8 |
| T3 | domain_wall | 1037 | 5516 | 243.4 |
| T3 | one_hot | 1370 | 5876 | 276.2 |

`ENCODINGS` order is `['binary', 'one_hot', 'domain_wall']`; `MATERIALISABLE` slices it
to `[:3]` for T0–T2 and `[:1]` for T3. The T3 domain-wall and one-hot cells are
excluded from energy equality by design and covered by the budget instead — that is
the artifact showing they cannot reach Advantage2.

This does not discharge Task 12 Step 3, which wants the full `describe()` dump and a
statement of which cells fall under `DENSE_EMBEDDING_CEILING` reported in the commit
message. Re-run it yourself.

## What to do

1. **Run the gate.** Give it hours, not minutes.

   ```bash
   .venv/Scripts/python.exe -u -m pytest tests/encoding/test_m3a_gate.py -v --durations=0
   ```

   On POSIX the interpreter is `.venv/bin/python`. Redirect to a file if you want to
   watch it — **do not pipe through `tail`**, which buffers until EOF and hides
   everything. That mistake cost a probe here.

2. **Record the variable budget** — Task 12 Step 3. Throwaway script under the
   scratchpad, importing `_params_for` from the gate module. Do not commit the script.

3. **Full suite, lint, layer contract** — Task 12 Step 4.

   ```bash
   .venv/Scripts/python.exe -m pytest -q
   .venv/Scripts/python.exe -m ruff check src tests
   .venv/Scripts/lint-imports.exe
   ```

   Expected: green, and `Contracts: 1 kept, 0 broken.`

4. **Whole-branch review**, then finish the branch. This was the original request and
   is the one step no one has started.

## Environment

Windows 11, Python 3.11.16, pytest 9.1.1, hypothesis 6.168.1, numpy/scipy/polars per
`pyproject.toml`. `.venv/` is checked out in place. The `tzdata` dependency is
deliberate — Windows ships no system IANA database and `core.session` needs one.

Git will warn `LF will be replaced by CRLF` on the gate file. Benign.
