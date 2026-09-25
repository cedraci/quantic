# Quantic M3a: QUBO Encoding, Penalty Derivation and Decode

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an `Instance` into a QUBO and prove the QUBO *is* the objective — exact energy equality on 64 instances × 3 encodings, with no solver in sight.

**Architecture:** A `VarRegistry` names every binary. A layout planner reads the constraint set and decides how each `x[i,t]` is represented, because §6.2's min-participation trick is a re-expansion of `x` rather than a penalty on a fixed one. Three `IntegerEncoding` strategies supply the bits. Everything then accumulates into an owned `Qubo` through a `QuboBuilder` whose workhorse is `add_product(LinearForm, LinearForm)` — spread, impact, variance, and every penalty square are all products of linear forms.

**Tech Stack:** Python 3.11+, NumPy, pytest, Hypothesis, ruff, import-linter. **No new dependencies** — in particular no `dimod`.

**Spec:** `docs/superpowers/specs/2026-09-25-m3a-encoding-design.md`

## Global Constraints

- **Python `>=3.11`.** Tests run as `.venv/Scripts/python.exe -m pytest` from the repo root.
- **Layer boundaries, enforced by import-linter:** `encoding/` may import `problem/`, `micro/`, `data/`, `core/`. It must **never** import `solvers/`, `bench/`, `viz/` or `dashboard/`, and none of those lower layers may import `encoding/`.
- **No new dependencies.** `dimod` conversion belongs to `solvers/quantum/` at M5.
- **Determinism.** Identical inputs produce identical output. No implicit global RNGs, no module-level mutable caches.
- **Strictness over silent repair.** Malformed input raises a typed exception. Nothing is silently dropped, clamped or patched.
- **Units.** `Schedule` and `initial_lots` are in **lots**; everything in `micro/` is in **shares**. `objective.evaluate` is the only place that converts, via `Asset.lot_size`. This plan's expansions convert explicitly and say so at each site.
- **Array-holding frozen dataclasses need explicit `__eq__`/`__hash__`.** A `@dataclass(frozen=True)` with an ndarray field has a generated `__eq__` that raises `ValueError` for arrays of 2+ elements and a `__hash__` that raises `TypeError`. `MarketParams`, `Instance` and `ScenarioSet` all carry this fix; `Qubo` must too.
- **Dispatch on type, never on an attribute name.** M2b's lesson. Applies to constraints in Task 7 exactly as it applied to risk specs.
- **Line length 100**, ruff rules `E,F,I,UP,B,SIM`, passing **without suppressions**. A `# noqa` added to make a rule pass is a defect — report the conflict instead.
- **Test-first.** Write the failing test, watch it fail for the right reason, then implement.
- **Expected-result lines are advisory.** Judge success by whether tests pass, not by matching a count.

## File Structure

| File | Responsibility |
|---|---|
| `encoding/registry.py` | `VarKey`, `VarRegistry`, `RegistryBuilder` |
| `encoding/qubo.py` | `LinearForm`, `Qubo`, `QuboBuilder` |
| `encoding/integer.py` | `IntegerEncoding` protocol + binary/one-hot/domain-wall |
| `encoding/layout.py` | `CellPlan`, `Layout`, `plan_layout` |
| `encoding/expand.py` | `QuboBreakdown`, `ApproximandReport`, `expand_objective` |
| `encoding/cvar.py` | `Grid`, `CVaRGadget` |
| `encoding/penalty.py` | `PenaltyStrengths`, `derive_penalties`, `expand_penalties` |
| `encoding/decode.py` | `DecodedSample`, `decode` |
| `encoding/encode.py` | `EncodedInstance`, `encode` — the orchestrator |
| `encoding/diagnostics.py` | `dynamic_range_ratio`, `variable_budget` |
| `tests/encoding/test_m3a_gate.py` | the 64 × 3 gate |

**Refinement of the spec's §14 table.** The spec lists nine modules; this plan adds `encoding/encode.py`. The spec's table names responsibilities, not an orchestration seam, and every one of the nine still needs a single caller that ties layout → expansion → penalties → diagnostics into one artifact. Splitting that caller out keeps `expand.py` about the objective alone.

## Task Order

Tasks 1–4 build the substrate with no reference to the objective. Task 5 lands the exact objective terms and **deliberately raises** for D1-on and for `CVaRRisk`, so those two arrive later on a green suite and can each be verified alone — the same staging M2b used for its `isinstance` fix. Task 12's gate is the last thing written and the first thing that can fail for an interesting reason.

---

## Task 1: `VarRegistry` — every binary has a name

**Files:**
- Create: `src/quantic/encoding/registry.py`
- Create: `tests/encoding/__init__.py` (empty)
- Test: `tests/encoding/test_registry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `VarKey = tuple[str | int, ...]`; `VarRegistry` with `keys: tuple[VarKey, ...]`, `__len__`, `index(key) -> int`, `__contains__`; `RegistryBuilder` with `add(key) -> int` and `freeze() -> VarRegistry`.

**Why names rather than bare indices.** A QUBO is an opaque matrix. When Task 12's gate reports a mismatch in the permanent-impact term, the difference between "coefficient 47 is wrong" and "coefficient `('x', 1, 2, 0)` is wrong" is the difference between a debuggable failure and a rewrite.

- [ ] **Step 1: Write the failing test**

Create `tests/encoding/__init__.py` as an empty file, then create `tests/encoding/test_registry.py`:

```python
import pytest

from quantic.encoding.registry import RegistryBuilder, VarRegistry


def test_keys_index_in_insertion_order():
    b = RegistryBuilder()
    assert b.add(("x", 0, 0, 0)) == 0
    assert b.add(("x", 0, 0, 1)) == 1
    assert b.add(("y", 0, 0)) == 2
    reg = b.freeze()
    assert len(reg) == 3
    assert reg.index(("y", 0, 0)) == 2
    assert reg.keys[0] == ("x", 0, 0, 0)


def test_a_duplicate_key_is_refused_at_add():
    b = RegistryBuilder()
    b.add(("zeta", 0))
    with pytest.raises(ValueError, match="already registered"):
        b.add(("zeta", 0))


def test_an_unknown_key_names_itself_in_the_error():
    reg = RegistryBuilder().freeze()
    with pytest.raises(KeyError, match=r"\('u', 3, 1\)"):
        reg.index(("u", 3, 1))


def test_membership_does_not_raise():
    b = RegistryBuilder()
    b.add(("g", 0, 0))
    reg = b.freeze()
    assert ("g", 0, 0) in reg
    assert ("g", 0, 1) not in reg


def test_a_registry_built_with_duplicates_directly_is_refused():
    """freeze() is not the only door; the type itself must hold the invariant."""
    with pytest.raises(ValueError, match="duplicate"):
        VarRegistry(keys=(("x", 0), ("x", 0)))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.encoding.registry'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/encoding/registry.py`:

```python
"""Names for every binary in a QUBO.

A QUBO is an opaque matrix, and the gate that proves it correct compares
energies term by term. When a term disagrees, "coefficient 47 is wrong" sends
an implementer back to the drawing board; ``('x', 1, 2, 0)`` sends them to one
line. The registry is the difference.

Key conventions, fixed here because every module below depends on them:

    ("x", i, t, b)      expansion bit b of the lots traded in cell (i, t)
    ("y", i, t)         the min-participation indicator for cell (i, t)
    ("block", k)        the all-or-nothing bit for block k
    ("card", t, j)      cardinality slack bit j in bucket t
    ("zeta", b)         expansion bit b of the Rockafellar-Uryasev zeta
    ("u", s, b)         expansion bit b of the tail excess in scenario s
    ("g", s, b)         expansion bit b of the slack in scenario s
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

VarKey = tuple[str | int, ...]


@dataclass(frozen=True)
class VarRegistry:
    """An ordered, duplicate-free naming of a QUBO's binaries.

    Order is the matrix's column order and is therefore load-bearing: two
    registries with the same keys in different orders describe different
    matrices, and a results row pins the order through the instance hash.
    """

    keys: tuple[VarKey, ...]
    _index: Mapping[VarKey, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        counts = Counter(self.keys)
        duplicates = sorted(k for k, n in counts.items() if n > 1)
        if duplicates:
            raise ValueError(
                f"duplicate variable key(s) {duplicates}: two bits sharing a name "
                "would silently alias to one column"
            )
        object.__setattr__(
            self, "_index", MappingProxyType({k: i for i, k in enumerate(self.keys)})
        )

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: VarKey) -> bool:
        return key in self._index

    def index(self, key: VarKey) -> int:
        try:
            return self._index[key]
        except KeyError:
            raise KeyError(
                f"{key!r} is not a variable in this registry of {len(self)} bits"
            ) from None


class RegistryBuilder:
    """Accumulates keys in order, refusing duplicates at the point of insertion."""

    def __init__(self) -> None:
        self._keys: list[VarKey] = []
        self._seen: set[VarKey] = set()

    def add(self, key: VarKey) -> int:
        if key in self._seen:
            raise ValueError(f"{key!r} is already registered at index {self._keys.index(key)}")
        self._seen.add(key)
        self._keys.append(key)
        return len(self._keys) - 1

    def freeze(self) -> VarRegistry:
        return VarRegistry(keys=tuple(self._keys))
```

Create `src/quantic/encoding/__init__.py` if it does not already carry a docstring; it exists from M0 scaffolding, so leave it alone if present.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_registry.py -q`
Expected: 5 passed.

- [ ] **Step 5: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 6: Commit**

```bash
git add src/quantic/encoding/registry.py tests/encoding/
git commit -m "feat(encoding): name every QUBO binary with a typed key"
```

---

## Task 2: `LinearForm`, `Qubo` and `QuboBuilder`

**Files:**
- Create: `src/quantic/encoding/qubo.py`
- Test: `tests/encoding/test_qubo.py`

**Interfaces:**
- Consumes: `VarKey`, `VarRegistry` (Task 1).
- Produces:
  - `LinearForm(constant: float, terms: tuple[tuple[VarKey, float], ...])` with `scaled(s) -> LinearForm`, `__add__`, and classmethods `constant_form(c)` and `of(key, coef=1.0)`.
  - `Qubo(Q: np.ndarray, constant: float, registry: VarRegistry)` with `n_vars`, `energy(bits) -> float`, `__add__`.
  - `QuboBuilder(registry)` with `add_constant`, `add_linear`, `add_quadratic`, `add_form`, `add_product`, `add_square`, `build() -> Qubo`.

**Why `add_product` is the workhorse.** Every quadratic thing in this milestone — permanent impact, variance, full liquidation, cardinality, the CVaR penalty — is a product of two linear forms. Writing that operation once, correctly (including the `z² = z` fold that a naive implementation gets wrong), is most of the encoder's correctness.

- [ ] **Step 1: Write the failing test**

Create `tests/encoding/test_qubo.py`:

```python
import numpy as np
import pytest

from quantic.encoding.qubo import LinearForm, Qubo, QuboBuilder
from quantic.encoding.registry import RegistryBuilder


def _registry(*keys):
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    return b.freeze()


def test_a_linear_form_scales_and_adds():
    f = LinearForm.of(("a",), 2.0) + LinearForm.constant_form(3.0)
    g = f.scaled(0.5)
    assert g.constant == 1.5
    assert g.terms == ((("a",), 1.0),)


def test_energy_is_the_quadratic_form_plus_the_constant():
    reg = _registry(("a",), ("b",))
    b = QuboBuilder(reg)
    b.add_constant(5.0)
    b.add_linear(("a",), 2.0)
    b.add_quadratic(("a",), ("b",), 3.0)
    q = b.build()
    assert q.energy([0, 0]) == 5.0
    assert q.energy([1, 0]) == 7.0
    assert q.energy([1, 1]) == 10.0


def test_a_product_folds_z_squared_to_z():
    """z**2 == z for a binary, so a self-product is linear, not quadratic."""
    reg = _registry(("a",),)
    b = QuboBuilder(reg)
    b.add_square(LinearForm.of(("a",), 3.0))
    q = b.build()
    assert q.energy([1]) == pytest.approx(9.0)
    assert q.energy([0]) == 0.0


def test_a_product_of_two_forms_expands_every_cross_term():
    reg = _registry(("a",), ("b",))
    f = LinearForm.of(("a",), 2.0) + LinearForm.constant_form(1.0)
    g = LinearForm.of(("b",), 5.0) + LinearForm.constant_form(-1.0)
    b = QuboBuilder(reg)
    b.add_product(f, g)
    q = b.build()
    for za, zb in ((0, 0), (0, 1), (1, 0), (1, 1)):
        expected = (2.0 * za + 1.0) * (5.0 * zb - 1.0)
        assert q.energy([za, zb]) == pytest.approx(expected)


def test_a_square_of_a_three_term_form_matches_direct_arithmetic():
    """The case a naive implementation gets wrong: cross terms counted once."""
    reg = _registry(("a",), ("b",), ("c",))
    f = (
        LinearForm.of(("a",), 1.0)
        + LinearForm.of(("b",), 2.0)
        + LinearForm.of(("c",), 3.0)
        + LinearForm.constant_form(-4.0)
    )
    q = QuboBuilder(reg).add_square(f) or None
    builder = QuboBuilder(reg)
    builder.add_square(f)
    q = builder.build()
    for za in (0, 1):
        for zb in (0, 1):
            for zc in (0, 1):
                expected = (za + 2 * zb + 3 * zc - 4) ** 2
                assert q.energy([za, zb, zc]) == pytest.approx(expected)


def test_the_matrix_is_upper_triangular():
    reg = _registry(("a",), ("b",))
    b = QuboBuilder(reg)
    b.add_quadratic(("b",), ("a",), 7.0)
    q = b.build()
    assert q.Q[1, 0] == 0.0
    assert q.Q[0, 1] == 7.0


def test_energy_refuses_a_non_binary_vector():
    reg = _registry(("a",))
    q = QuboBuilder(reg).build()
    with pytest.raises(ValueError, match="0 or 1"):
        q.energy([2])


def test_energy_refuses_a_wrong_length_vector():
    reg = _registry(("a",), ("b",))
    q = QuboBuilder(reg).build()
    with pytest.raises(ValueError, match="2 variables"):
        q.energy([1])


def test_two_qubos_over_one_registry_add():
    reg = _registry(("a",))
    x = QuboBuilder(reg)
    x.add_linear(("a",), 1.0)
    y = QuboBuilder(reg)
    y.add_constant(4.0)
    total = x.build() + y.build()
    assert total.energy([1]) == 5.0


def test_adding_qubos_over_different_registries_is_refused():
    a = QuboBuilder(_registry(("a",))).build()
    b = QuboBuilder(_registry(("b",))).build()
    with pytest.raises(ValueError, match="different registries"):
        _ = a + b


def test_a_non_finite_coefficient_is_refused_at_build():
    reg = _registry(("a",))
    b = QuboBuilder(reg)
    b.add_linear(("a",), float("inf"))
    with pytest.raises(ValueError, match="finite"):
        b.build()


def test_equality_is_defined_despite_the_array_field():
    reg = _registry(("a",))
    b = QuboBuilder(reg)
    b.add_linear(("a",), 1.0)
    assert b.build() == QuboBuilder(reg).build() + b.build()
    assert np.array_equal(b.build().Q, b.build().Q)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_qubo.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.encoding.qubo'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/encoding/qubo.py`:

```python
"""The QUBO this project emits, and the algebra that builds it.

Spec section 6.2 reaches for ``dimod.make_quadratic`` to flatten residual
higher-order terms. No formulation in M3a produces one -- min participation
and cardinality are quadratic by construction, block trades contribute
linearly, the CVaR loss is linear in holdings, and concave impact is either a
per-value table or an explicit quadratic approximand -- so that call would be
dead code, and it would put a third party's float accumulation order
underneath the gate that proves this encoder correct. The type is owned here
instead; conversion to ``dimod`` lands in ``solvers/quantum/`` at M5, where a
sampler needs it.

``LinearForm`` is the unit everything is written in. A binary's square is
itself, so the product of two linear forms is quadratic, and every quadratic
object in the milestone -- permanent impact, variance, and every penalty
square -- is exactly such a product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.encoding.registry import VarKey, VarRegistry


@dataclass(frozen=True, slots=True)
class LinearForm:
    """``constant + sum of coef * z`` over named binaries.

    Repeated keys are permitted and are summed at use; callers build these by
    addition, and forcing uniqueness at every intermediate step would cost
    more than folding once at the end.
    """

    constant: float = 0.0
    terms: tuple[tuple[VarKey, float], ...] = ()

    @classmethod
    def constant_form(cls, value: float) -> LinearForm:
        return cls(constant=float(value))

    @classmethod
    def of(cls, key: VarKey, coef: float = 1.0) -> LinearForm:
        return cls(constant=0.0, terms=((key, float(coef)),))

    def scaled(self, factor: float) -> LinearForm:
        f = float(factor)
        return LinearForm(
            constant=self.constant * f,
            terms=tuple((k, c * f) for k, c in self.terms),
        )

    def __add__(self, other: LinearForm) -> LinearForm:
        if not isinstance(other, LinearForm):
            return NotImplemented
        return LinearForm(
            constant=self.constant + other.constant,
            terms=self.terms + other.terms,
        )


@dataclass(frozen=True, eq=False)
class Qubo:
    """An upper-triangular quadratic form over named binaries.

    ``eq=False`` with an explicit ``__eq__``: a frozen dataclass holding an
    ndarray gets a generated ``__eq__`` that raises ``ValueError`` for arrays
    of two or more elements. ``MarketParams``, ``Instance`` and ``ScenarioSet``
    all carry the same fix.
    """

    Q: np.ndarray
    constant: float
    registry: VarRegistry

    def __post_init__(self) -> None:
        q = np.asarray(self.Q, dtype=float)
        n = len(self.registry)
        if q.shape != (n, n):
            raise ValueError(
                f"Q has shape {q.shape} but the registry names {n} variables"
            )
        if not np.isfinite(q).all() or not np.isfinite(self.constant):
            raise ValueError("every QUBO coefficient must be finite")
        if np.any(np.tril(q, -1) != 0.0):
            raise ValueError(
                "Q must be upper triangular: a coefficient below the diagonal would "
                "be counted twice by energy()"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Qubo):
            return NotImplemented
        return (
            self.registry == other.registry
            and self.constant == other.constant
            and np.array_equal(self.Q, other.Q)
        )

    def __hash__(self) -> int:
        return hash((self.registry.keys, self.constant, self.Q.tobytes()))

    @property
    def n_vars(self) -> int:
        return len(self.registry)

    def energy(self, bits) -> float:
        """``z' Q z + constant`` for a 0/1 vector in the registry's order."""
        z = np.asarray(bits, dtype=float)
        if z.shape != (self.n_vars,):
            raise ValueError(
                f"energy needs {self.n_vars} variables, got a vector of shape {z.shape}"
            )
        if not np.all((z == 0.0) | (z == 1.0)):
            raise ValueError("every entry must be 0 or 1; this is a QUBO, not a QP")
        return float(z @ self.Q @ z + self.constant)

    def __add__(self, other: Qubo) -> Qubo:
        if not isinstance(other, Qubo):
            return NotImplemented
        if self.registry != other.registry:
            raise ValueError(
                "cannot add QUBOs over different registries: column k would mean a "
                "different bit in each"
            )
        return Qubo(
            Q=self.Q + other.Q,
            constant=self.constant + other.constant,
            registry=self.registry,
        )


class QuboBuilder:
    """Accumulates terms, then freezes into a :class:`Qubo`."""

    def __init__(self, registry: VarRegistry) -> None:
        self._registry = registry
        self._q = np.zeros((len(registry), len(registry)), dtype=float)
        self._constant = 0.0

    def add_constant(self, value: float) -> None:
        self._constant += float(value)

    def add_linear(self, key: VarKey, coef: float) -> None:
        i = self._registry.index(key)
        self._q[i, i] += float(coef)

    def add_quadratic(self, key_a: VarKey, key_b: VarKey, coef: float) -> None:
        """Accumulate ``coef * z_a * z_b``, folding ``z**2 = z`` when they coincide."""
        i = self._registry.index(key_a)
        j = self._registry.index(key_b)
        if i == j:
            self._q[i, i] += float(coef)
        else:
            lo, hi = (i, j) if i < j else (j, i)
            self._q[lo, hi] += float(coef)

    def add_form(self, form: LinearForm, scale: float = 1.0) -> None:
        """Accumulate ``scale * form`` as a linear contribution."""
        s = float(scale)
        self.add_constant(form.constant * s)
        for key, coef in form.terms:
            self.add_linear(key, coef * s)

    def add_product(self, a: LinearForm, b: LinearForm, scale: float = 1.0) -> None:
        """Accumulate ``scale * a * b``.

        Both forms may repeat keys; repeats simply accumulate. The ``i == j``
        case is delegated to :meth:`add_quadratic`, which folds it onto the
        diagonal -- the step a naive expansion gets wrong, because a binary's
        square is the binary itself rather than a separate quadratic term.
        """
        s = float(scale)
        self.add_constant(a.constant * b.constant * s)
        for key, coef in b.terms:
            self.add_linear(key, a.constant * coef * s)
        for key, coef in a.terms:
            self.add_linear(key, b.constant * coef * s)
        for key_a, coef_a in a.terms:
            for key_b, coef_b in b.terms:
                self.add_quadratic(key_a, key_b, coef_a * coef_b * s)

    def add_square(self, form: LinearForm, scale: float = 1.0) -> None:
        self.add_product(form, form, scale)

    def build(self) -> Qubo:
        return Qubo(Q=self._q.copy(), constant=self._constant, registry=self._registry)
```

- [ ] **Step 4: Fix the one malformed test line**

`test_a_square_of_a_three_term_form_matches_direct_arithmetic` above contains a stray first assignment to `q`. Replace the body's first two statements with just the `builder` lines so the test reads:

```python
    builder = QuboBuilder(reg)
    builder.add_square(f)
    q = builder.build()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_qubo.py -q`
Expected: every test in the file passes.

- [ ] **Step 6: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 7: Commit**

```bash
git add src/quantic/encoding/qubo.py tests/encoding/test_qubo.py
git commit -m "feat(encoding): add the owned Qubo type and its linear-form algebra"
```

---

## Task 3: The three integer encodings

**Files:**
- Create: `src/quantic/encoding/integer.py`
- Test: `tests/encoding/test_integer.py`

**Interfaces:**
- Consumes: `VarKey` (Task 1), `LinearForm` (Task 2).
- Produces:
  - `bounded_coefficients(max_value: int) -> tuple[int, ...]`
  - `IntegerEncoding` protocol: `name`, `supports_exact_table`, `bit_keys(prefix, values) -> tuple[VarKey, ...]`, `value_form(keys, values) -> LinearForm`, `cost_form(keys, values, costs) -> LinearForm | None`, `validity_forms(keys, values) -> tuple[LinearForm, ...]`, `indicator_form(keys, values) -> LinearForm | None`, `encode_value(keys, values, x) -> dict[VarKey, int]`, `decode_value(keys, values, bits) -> int`
  - `BinaryEncoding()`, `OneHotEncoding()`, `DomainWallEncoding()`
  - `ENCODINGS: tuple[IntegerEncoding, ...]` — the three, in the spec §6.1 order.

**The `values` contract.** `values` is the ascending tuple of every integer the cell may take. `OneHotEncoding` and `DomainWallEncoding` accept any such tuple, which is what makes D2 free under them. `BinaryEncoding` accepts only a contiguous range starting at zero and raises otherwise; the layout planner (Task 4) is responsible for never handing it anything else, by re-expanding a D2 cell as `m·y + residual`.

**`validity_forms` returns forms to be squared, not penalties.** Each returned `LinearForm` is a residual that must be zero at every valid state; Task 7 squares them and scales by a derived strength. Domain-wall's wall condition is the exception and is expressed as a product, so it is returned by `validity_products` instead — see the implementation.

- [ ] **Step 1: Write the failing test**

Create `tests/encoding/test_integer.py`:

```python
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantic.encoding.integer import (
    ENCODINGS,
    BinaryEncoding,
    DomainWallEncoding,
    OneHotEncoding,
    bounded_coefficients,
)
from quantic.encoding.qubo import QuboBuilder
from quantic.encoding.registry import RegistryBuilder


@given(st.integers(min_value=0, max_value=512))
def test_bounded_coefficients_reach_every_value_and_no_more(m):
    coeffs = bounded_coefficients(m)
    reachable = {0}
    for c in coeffs:
        reachable |= {r + c for r in reachable}
    assert max(reachable) == m
    assert set(range(m + 1)) <= reachable


@given(st.integers(min_value=0, max_value=512))
def test_bounded_coefficients_use_the_minimal_bit_count(m):
    assert len(bounded_coefficients(m)) == (m.bit_length() if m else 0)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize("values", [(0, 1), (0, 1, 2, 3, 4, 5), tuple(range(9))])
def test_every_value_round_trips(encoding, values):
    keys = encoding.bit_keys(("x", 0, 0), values)
    for v in values:
        bits = encoding.encode_value(keys, values, v)
        assert encoding.decode_value(keys, values, bits) == v


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_value_form_agrees_with_the_encoded_bits(encoding):
    values = tuple(range(6))
    keys = encoding.bit_keys(("x", 0, 0), values)
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    reg = b.freeze()
    form = encoding.value_form(keys, values)
    for v in values:
        bits = encoding.encode_value(keys, values, v)
        builder = QuboBuilder(reg)
        builder.add_form(form)
        vector = [bits[k] for k in reg.keys]
        assert builder.build().energy(vector) == pytest.approx(float(v))


@pytest.mark.parametrize(
    "encoding", [OneHotEncoding(), DomainWallEncoding()], ids=lambda e: e.name
)
def test_a_table_encoding_represents_an_arbitrary_cost_exactly(encoding):
    """This is what makes D1 free: any f over the value set is linear in the bits."""
    values = (0, 2, 3, 7)
    costs = {0: 0.0, 2: 1.9, 3: -4.25, 7: 11.0}
    keys = encoding.bit_keys(("x", 0, 0), values)
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    reg = b.freeze()
    form = encoding.cost_form(keys, values, costs)
    assert form is not None
    for v in values:
        bits = encoding.encode_value(keys, values, v)
        builder = QuboBuilder(reg)
        builder.add_form(form)
        vector = [bits[k] for k in reg.keys]
        assert builder.build().energy(vector) == pytest.approx(costs[v])


def test_the_log_encoding_offers_no_exact_cost_table():
    """x**delta has no exact quadratic form in log bits. Saying so is the point."""
    enc = BinaryEncoding()
    assert enc.supports_exact_table is False
    values = tuple(range(4))
    keys = enc.bit_keys(("x", 0, 0), values)
    assert enc.cost_form(keys, values, {v: float(v) for v in values}) is None


def test_the_log_encoding_refuses_a_non_contiguous_value_set():
    enc = BinaryEncoding()
    with pytest.raises(ValueError, match="contiguous"):
        enc.bit_keys(("x", 0, 0), (0, 3, 4))


def test_the_log_encoding_refuses_a_value_set_not_starting_at_zero():
    enc = BinaryEncoding()
    with pytest.raises(ValueError, match="contiguous"):
        enc.bit_keys(("x", 0, 0), (2, 3, 4))


def test_one_hot_validity_is_zero_exactly_on_the_valid_states():
    enc = OneHotEncoding()
    values = (0, 1, 2)
    keys = enc.bit_keys(("x", 0, 0), values)
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    reg = b.freeze()
    forms = enc.validity_forms(keys, values)
    for vector in np.ndindex(*(2,) * len(keys)):
        builder = QuboBuilder(reg)
        for f in forms:
            builder.add_square(f)
        energy = builder.build().energy(list(vector))
        valid = sum(vector) == 1
        assert (energy == 0.0) is valid


def test_domain_wall_validity_is_zero_exactly_on_the_walls():
    enc = DomainWallEncoding()
    values = (0, 1, 2, 3)
    keys = enc.bit_keys(("x", 0, 0), values)
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    reg = b.freeze()
    for vector in np.ndindex(*(2,) * len(keys)):
        builder = QuboBuilder(reg)
        for f in enc.validity_forms(keys, values):
            builder.add_square(f)
        for lhs, rhs in enc.validity_products(keys, values):
            builder.add_product(lhs, rhs)
        energy = builder.build().energy(list(vector))
        monotone = all(vector[j] >= vector[j + 1] for j in range(len(vector) - 1))
        assert (energy == 0.0) is monotone


def test_domain_wall_uses_one_fewer_bit_than_one_hot():
    values = tuple(range(6))
    assert len(DomainWallEncoding().bit_keys(("x", 0, 0), values)) == 5
    assert len(OneHotEncoding().bit_keys(("x", 0, 0), values)) == 6


@pytest.mark.parametrize(
    "encoding", [OneHotEncoding(), DomainWallEncoding()], ids=lambda e: e.name
)
def test_a_table_encoding_gives_the_positivity_indicator_for_free(encoding):
    """Cardinality needs 1[x > 0]. Under a table encoding it is already linear."""
    values = (0, 3, 4)
    keys = encoding.bit_keys(("x", 0, 0), values)
    b = RegistryBuilder()
    for k in keys:
        b.add(k)
    reg = b.freeze()
    form = encoding.indicator_form(keys, values)
    assert form is not None
    for v in values:
        bits = encoding.encode_value(keys, values, v)
        builder = QuboBuilder(reg)
        builder.add_form(form)
        vector = [bits[k] for k in reg.keys]
        assert builder.build().energy(vector) == pytest.approx(1.0 if v > 0 else 0.0)


def test_the_log_encoding_has_no_free_indicator():
    enc = BinaryEncoding()
    values = tuple(range(4))
    keys = enc.bit_keys(("x", 0, 0), values)
    assert enc.indicator_form(keys, values) is None


def test_decoding_an_invalid_one_hot_state_is_refused():
    enc = OneHotEncoding()
    values = (0, 1, 2)
    keys = enc.bit_keys(("x", 0, 0), values)
    with pytest.raises(ValueError, match="exactly one"):
        enc.decode_value(keys, values, dict.fromkeys(keys, 1))


def test_decoding_a_broken_domain_wall_is_refused():
    enc = DomainWallEncoding()
    values = (0, 1, 2, 3)
    keys = enc.bit_keys(("x", 0, 0), values)
    broken = {keys[0]: 0, keys[1]: 1, keys[2]: 0}
    with pytest.raises(ValueError, match="monotone"):
        enc.decode_value(keys, values, broken)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_integer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.encoding.integer'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/encoding/integer.py`:

```python
"""Spec section 6.1's three integer encodings, as interchangeable strategies.

The choice is a study dimension, not an implementation detail, and the three
are not equivalent:

* **One-hot** and **domain-wall** set exactly one bit per value, so *any*
  function of the cell is exactly linear in those bits. Concave impact costs
  nothing extra under either, and a value set with holes -- which is what a
  venue minimum produces -- is expressed by leaving the holes out of the
  table. Dial D2 is therefore free here.
* **Binary/log** expands the value itself, so only polynomials in the value up
  to degree two survive. ``x**delta`` does not, which is why Task 6 fits an
  explicit quadratic approximand and reports its error rather than pretending.

The log encoding uses bounded coefficients rather than powers of two.
``sum 2**b z_b`` reaches past ``max_value`` whenever ``max_value + 1`` is not a
power of two, so the bits admit states the problem forbids, and suppressing
them means a range penalty with a magnitude to tune. Coefficients
``1, 2, ..., 2**(k-2), max_value - 2**(k-1) + 1`` cover exactly ``[0,
max_value]`` and nothing outside it, so no such penalty exists. The price is
that several bitstrings share a value; ``encode_value`` picks a canonical one,
and only ``decode_value(encode_value(x)) == x`` is claimed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from quantic.encoding.qubo import LinearForm
from quantic.encoding.registry import VarKey


def bounded_coefficients(max_value: int) -> tuple[int, ...]:
    """Coefficients whose subset sums are exactly ``[0, max_value]``."""
    if max_value < 0:
        raise ValueError(f"max_value must be non-negative, got {max_value}")
    if max_value == 0:
        return ()
    k = max_value.bit_length()
    coefficients = [1 << b for b in range(k - 1)]
    coefficients.append(max_value - ((1 << (k - 1)) - 1))
    return tuple(coefficients)


def _require_contiguous_from_zero(values: tuple[int, ...]) -> int:
    if tuple(values) != tuple(range(len(values))):
        raise ValueError(
            f"the log encoding needs a contiguous value set starting at zero, got "
            f"{values}. A set with holes is the layout planner's job: it re-expands "
            "the cell as m*y + residual (spec section 6.2)"
        )
    return len(values) - 1


@runtime_checkable
class IntegerEncoding(Protocol):
    name: str
    supports_exact_table: bool

    def bit_keys(self, prefix: VarKey, values: tuple[int, ...]) -> tuple[VarKey, ...]: ...

    def value_form(
        self, keys: tuple[VarKey, ...], values: tuple[int, ...]
    ) -> LinearForm: ...

    def cost_form(
        self,
        keys: tuple[VarKey, ...],
        values: tuple[int, ...],
        costs: Mapping[int, float],
    ) -> LinearForm | None: ...

    def indicator_form(
        self, keys: tuple[VarKey, ...], values: tuple[int, ...]
    ) -> LinearForm | None: ...

    def validity_forms(
        self, keys: tuple[VarKey, ...], values: tuple[int, ...]
    ) -> tuple[LinearForm, ...]: ...

    def validity_products(
        self, keys: tuple[VarKey, ...], values: tuple[int, ...]
    ) -> tuple[tuple[LinearForm, LinearForm], ...]: ...

    def encode_value(
        self, keys: tuple[VarKey, ...], values: tuple[int, ...], x: int
    ) -> dict[VarKey, int]: ...

    def decode_value(
        self,
        keys: tuple[VarKey, ...],
        values: tuple[int, ...],
        bits: Mapping[VarKey, int],
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class BinaryEncoding:
    """``x = sum c_b z_b`` with bounded coefficients. Minimal variable count."""

    name: str = "binary"
    supports_exact_table: bool = False

    def bit_keys(self, prefix: VarKey, values: tuple[int, ...]) -> tuple[VarKey, ...]:
        maximum = _require_contiguous_from_zero(values)
        return tuple((*prefix, b) for b in range(len(bounded_coefficients(maximum))))

    def value_form(self, keys, values) -> LinearForm:
        coefficients = bounded_coefficients(_require_contiguous_from_zero(values))
        form = LinearForm()
        for key, c in zip(keys, coefficients, strict=True):
            form = form + LinearForm.of(key, float(c))
        return form

    def cost_form(self, keys, values, costs) -> LinearForm | None:
        return None

    def indicator_form(self, keys, values) -> LinearForm | None:
        return None

    def validity_forms(self, keys, values) -> tuple[LinearForm, ...]:
        _require_contiguous_from_zero(values)
        return ()

    def validity_products(self, keys, values):
        _require_contiguous_from_zero(values)
        return ()

    def encode_value(self, keys, values, x) -> dict[VarKey, int]:
        maximum = _require_contiguous_from_zero(values)
        if not 0 <= x <= maximum:
            raise ValueError(f"{x} is outside the cell's range [0, {maximum}]")
        coefficients = bounded_coefficients(maximum)
        bits = dict.fromkeys(keys, 0)
        remainder = x
        if coefficients:
            last = len(coefficients) - 1
            if coefficients[last] <= remainder:
                bits[keys[last]] = 1
                remainder -= coefficients[last]
            for b in range(last - 1, -1, -1):
                if coefficients[b] <= remainder:
                    bits[keys[b]] = 1
                    remainder -= coefficients[b]
        if remainder != 0:
            raise ValueError(f"{x} is not representable by coefficients {coefficients}")
        return bits

    def decode_value(self, keys, values, bits) -> int:
        coefficients = bounded_coefficients(_require_contiguous_from_zero(values))
        return sum(c for key, c in zip(keys, coefficients, strict=True) if bits[key])


@dataclass(frozen=True, slots=True)
class OneHotEncoding:
    """One bit per admissible value. Precision-friendly, expensive in variables."""

    name: str = "one_hot"
    supports_exact_table: bool = True

    def bit_keys(self, prefix: VarKey, values: tuple[int, ...]) -> tuple[VarKey, ...]:
        return tuple((*prefix, v) for v in values)

    def value_form(self, keys, values) -> LinearForm:
        form = LinearForm()
        for key, v in zip(keys, values, strict=True):
            form = form + LinearForm.of(key, float(v))
        return form

    def cost_form(self, keys, values, costs) -> LinearForm | None:
        form = LinearForm()
        for key, v in zip(keys, values, strict=True):
            form = form + LinearForm.of(key, float(costs[v]))
        return form

    def indicator_form(self, keys, values) -> LinearForm | None:
        return self.cost_form(keys, values, {v: (1.0 if v > 0 else 0.0) for v in values})

    def validity_forms(self, keys, values) -> tuple[LinearForm, ...]:
        form = LinearForm.constant_form(-1.0)
        for key in keys:
            form = form + LinearForm.of(key, 1.0)
        return (form,)

    def validity_products(self, keys, values):
        return ()

    def encode_value(self, keys, values, x) -> dict[VarKey, int]:
        if x not in values:
            raise ValueError(f"{x} is not one of this cell's values {values}")
        return {key: int(v == x) for key, v in zip(keys, values, strict=True)}

    def decode_value(self, keys, values, bits) -> int:
        hot = [v for key, v in zip(keys, values, strict=True) if bits[key]]
        if len(hot) != 1:
            raise ValueError(
                f"a one-hot cell needs exactly one set bit, got {len(hot)}"
            )
        return hot[0]


@dataclass(frozen=True, slots=True)
class DomainWallEncoding:
    """Chancellor's domain wall: ``w_j = 1[x >= values[j]]``.

    One fewer variable than one-hot, and the validity condition is strictly
    local -- each term couples only adjacent bits -- which is why it is
    empirically better on annealers.
    """

    name: str = "domain_wall"
    supports_exact_table: bool = True

    def bit_keys(self, prefix: VarKey, values: tuple[int, ...]) -> tuple[VarKey, ...]:
        return tuple((*prefix, v) for v in values[1:])

    def _deltas(self, values, at):
        return [at(values[j]) - at(values[j - 1]) for j in range(1, len(values))]

    def value_form(self, keys, values) -> LinearForm:
        form = LinearForm.constant_form(float(values[0]))
        for key, d in zip(keys, self._deltas(values, float), strict=True):
            form = form + LinearForm.of(key, d)
        return form

    def cost_form(self, keys, values, costs) -> LinearForm | None:
        form = LinearForm.constant_form(float(costs[values[0]]))
        deltas = self._deltas(values, lambda v: float(costs[v]))
        for key, d in zip(keys, deltas, strict=True):
            form = form + LinearForm.of(key, d)
        return form

    def indicator_form(self, keys, values) -> LinearForm | None:
        return self.cost_form(keys, values, {v: (1.0 if v > 0 else 0.0) for v in values})

    def validity_forms(self, keys, values) -> tuple[LinearForm, ...]:
        return ()

    def validity_products(self, keys, values):
        """``w_{j+1} (1 - w_j)`` -- zero exactly on the monotone states."""
        return tuple(
            (LinearForm.of(keys[j + 1], 1.0),
             LinearForm.constant_form(1.0) + LinearForm.of(keys[j], -1.0))
            for j in range(len(keys) - 1)
        )

    def encode_value(self, keys, values, x) -> dict[VarKey, int]:
        if x not in values:
            raise ValueError(f"{x} is not one of this cell's values {values}")
        return {key: int(v <= x) for key, v in zip(keys, values[1:], strict=True)}

    def decode_value(self, keys, values, bits) -> int:
        vector = [bits[key] for key in keys]
        if any(vector[j] < vector[j + 1] for j in range(len(vector) - 1)):
            raise ValueError(
                f"a domain wall must be monotone non-increasing, got {vector}"
            )
        return values[sum(vector)]


ENCODINGS: tuple[IntegerEncoding, ...] = (
    BinaryEncoding(),
    OneHotEncoding(),
    DomainWallEncoding(),
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_integer.py -q`
Expected: every test in the file passes.

- [ ] **Step 5: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 6: Commit**

```bash
git add src/quantic/encoding/integer.py tests/encoding/test_integer.py
git commit -m "feat(encoding): add the three integer encodings of spec section 6.1"
```

---

## Task 4: The layout planner — constraints decide how `x` is represented

**Files:**
- Create: `src/quantic/encoding/layout.py`
- Modify: `src/quantic/encoding/registry.py` (one docstring line)
- Test: `tests/encoding/test_layout.py`

**Interfaces:**
- Consumes: `Instance`, `Constraint` variants (`problem/`); `IntegerEncoding` (Task 3); `LinearForm` (Task 2); `RegistryBuilder`, `VarRegistry` (Task 1).
- Produces:
  - `CellPlan` — frozen, one per `(i, t)`, with `values`, `bit_keys`, `value_form`, `indicator_form`, `validity_forms`, `validity_products`, `exact_cost_form(costs) -> LinearForm | None`, `encode(x) -> dict[VarKey, int]`, `decode(bits) -> int`.
  - `Layout(cells, registry, encoding_name, n_assets, n_buckets)` with `cell(i, t) -> CellPlan`.
  - `plan_layout(instance, encoding) -> Layout`.

**Why this exists at all.** §6.2's min-participation formulation, `x = m·y + Σ 2^b z_b`, is a *different expansion of `x`*, not a penalty bolted onto a fixed one. A D4 block cell collapses to a single bit. So the constraint set has to be read before any bit is named, and the resulting asymmetry — **D2 is free under one-hot and domain-wall, and costs a penalty only under log** — is a result for the §6.1 study rather than an implementation detail.

**Per-cell bound.** A cell's maximum is `min(instance.max_lots_per_bucket, instance.initial_lots[i])`, not `max_lots_per_bucket` alone. Full liquidation caps the row total at `X_i`, so the tighter bound excludes no feasible schedule and removes bits from every tier.

**Cardinality's indicator.** The generator attaches `Cardinality` only alongside `MinParticipation` (`generator.py:235-247`), so every cell already carries an indicator whenever cardinality is present. `plan_layout` asserts that rather than assuming it.

- [ ] **Step 1: Amend the registry docstring**

In `src/quantic/encoding/registry.py`, replace the line reading
`("block", k)        the all-or-nothing bit for block k`
with
`("block", i, t)     the all-or-nothing bit for the block on cell (i, t)`.

Blocks are unique per cell — `BlockTrades.__post_init__` refuses two on one cell — so the cell names the block unambiguously and the key stays self-describing.

- [ ] **Step 2: Write the failing test**

Create `tests/encoding/test_layout.py`:

```python
import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.integer import (
    ENCODINGS,
    BinaryEncoding,
    DomainWallEncoding,
    OneHotEncoding,
)
from quantic.encoding.layout import plan_layout
from quantic.encoding.qubo import QuboBuilder
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(**kw) -> Instance:
    assets = tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B"))
    params = MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01)
            for s in ("A", "B")
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    base = dict(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(4, 4), max_lots_per_bucket=4, params=params,
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def _value_at(layout, plan, bits):
    builder = QuboBuilder(layout.registry)
    builder.add_form(plan.value_form)
    vector = [bits.get(k, 0) for k in layout.registry.keys]
    return builder.build().energy(vector)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_every_cell_round_trips_every_admissible_value(encoding):
    layout = plan_layout(_instance(), encoding)
    for plan in layout.cells:
        for v in plan.values:
            assert plan.decode(plan.encode(v)) == v


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_value_form_agrees_with_the_encoded_bits(encoding):
    layout = plan_layout(_instance(), encoding)
    for plan in layout.cells:
        for v in plan.values:
            assert _value_at(layout, plan, plan.encode(v)) == pytest.approx(float(v))


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_cell_bound_is_the_tighter_of_the_cap_and_the_position(encoding):
    """initial_lots=(1, 4) with a cap of 4: asset 0 can never trade more than 1."""
    layout = plan_layout(_instance(initial_lots=(1, 4)), encoding)
    assert max(layout.cell(0, 0).values) == 1
    assert max(layout.cell(1, 0).values) == 4


@pytest.mark.parametrize(
    "encoding", [OneHotEncoding(), DomainWallEncoding()], ids=lambda e: e.name
)
def test_d2_is_free_under_a_table_encoding(encoding):
    """The forbidden values are simply absent; no indicator and no penalty."""
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=2), MinParticipation((3, 3))),
        witness=Schedule(lots=((4, 0), (0, 4))),
        initial_lots=(4, 4),
    )
    layout = plan_layout(inst, encoding)
    plan = layout.cell(0, 0)
    assert plan.values == (0, 3, 4)
    assert plan.kind == "semi_continuous"
    assert not any(k[0] == "y" for k in layout.registry.keys)


def test_d2_under_the_log_encoding_needs_an_indicator_and_a_residual():
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=2), MinParticipation((3, 3))),
        witness=Schedule(lots=((4, 0), (0, 4))),
        initial_lots=(4, 4),
    )
    layout = plan_layout(inst, BinaryEncoding())
    plan = layout.cell(0, 0)
    assert plan.values == (0, 3, 4)
    assert plan.indicator_key == ("y", 0, 0)
    assert plan.residual_keys  # x = 3*y + residual over [0, 1]
    for v in plan.values:
        assert _value_at(layout, plan, plan.encode(v)) == pytest.approx(float(v))


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_block_cell_costs_exactly_one_bit_under_every_encoding(encoding):
    inst = _instance(
        dials=Dials(block_trades=True),
        constraints=(FullLiquidation(), BlockTrades((Block(asset=0, bucket=0, lots=2),))),
    )
    layout = plan_layout(inst, encoding)
    plan = layout.cell(0, 0)
    assert plan.kind == "block"
    assert plan.bit_keys == (("block", 0, 0),)
    assert plan.values == (0, 2)
    assert _value_at(layout, plan, plan.encode(2)) == pytest.approx(2.0)
    assert _value_at(layout, plan, plan.encode(0)) == pytest.approx(0.0)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_block_cell_refuses_a_value_it_cannot_take(encoding):
    inst = _instance(
        dials=Dials(block_trades=True),
        constraints=(FullLiquidation(), BlockTrades((Block(asset=0, bucket=0, lots=2),))),
    )
    plan = plan_layout(inst, encoding).cell(0, 0)
    with pytest.raises(ValueError, match="not one of this cell"):
        plan.encode(1)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_every_cell_has_an_indicator_when_cardinality_is_present(encoding):
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=2), MinParticipation((3, 3))),
        witness=Schedule(lots=((4, 0), (0, 4))),
    )
    layout = plan_layout(inst, encoding)
    for plan in layout.cells:
        assert plan.indicator_form is not None


def test_cardinality_without_min_participation_is_refused():
    """The generator never emits this pairing; if it ever did, silence would be worse."""
    inst = _instance(constraints=(FullLiquidation(), Cardinality(k=2)))
    with pytest.raises(ValueError, match="indicator"):
        plan_layout(inst, BinaryEncoding())


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_cell_whose_minimum_exceeds_its_cap_can_only_be_zero(encoding):
    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=2), MinParticipation((9, 1))),
        witness=Schedule(lots=((0, 0), (4, 4))),
        initial_lots=(0, 8),
        max_lots_per_bucket=4,
    )
    layout = plan_layout(inst, encoding)
    assert layout.cell(0, 0).values == (0,)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_registry_holds_every_cell_key_exactly_once(encoding):
    layout = plan_layout(_instance(), encoding)
    named = [k for plan in layout.cells for k in plan.bit_keys]
    assert sorted(named) == sorted(set(named))
    assert set(named) <= set(layout.registry.keys)


def test_an_unknown_constraint_type_is_refused_even_carrying_a_known_name():
    """Dispatch is on type. M2b's lesson, applied to constraints."""

    @dataclasses.dataclass(frozen=True)
    class _NotAConstraint:
        name: str = "min_participation"
        min_lots: tuple[int, ...] = (1, 1)

        def violation(self, schedule, initial_lots) -> float:
            return 0.0

        def is_satisfied(self, schedule, initial_lots) -> bool:
            return True

        def describe(self) -> dict:
            return {"name": self.name}

    inst = _instance(constraints=(FullLiquidation(), _NotAConstraint()))
    with pytest.raises(NotImplementedError, match="_NotAConstraint"):
        plan_layout(inst, BinaryEncoding())
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_layout.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.encoding.layout'`.

- [ ] **Step 4: Write the implementation**

Create `src/quantic/encoding/layout.py`:

```python
"""How each ``x[i,t]`` is represented, given the constraints on it.

The representation is not a property of the encoding alone. Spec section 6.2's
min-participation formulation, ``x = m*y + sum 2**b z_b``, is a *different
expansion of x*, and a block cell collapses to a single all-or-nothing bit. So
the constraint set has to be read before any bit is named.

The asymmetry that falls out is a result for the section 6.1 encoding study
rather than an implementation detail: under one-hot and domain-wall the values
a venue minimum forbids are simply absent from the table, so **D2 is free**;
under the log encoding it costs an indicator, a residual expansion and a
penalty.

A cell's bound is ``min(max_lots_per_bucket, initial_lots[i])``. Full
liquidation caps the row total at the position, so the tighter bound excludes
no feasible schedule and removes bits from every tier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from quantic.encoding.integer import IntegerEncoding
from quantic.encoding.qubo import LinearForm
from quantic.encoding.registry import RegistryBuilder, VarKey, VarRegistry
from quantic.problem.constraints.block_trades import BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.instance import Instance

PLAIN = "plain"
SEMI_CONTINUOUS = "semi_continuous"
BLOCK = "block"


@dataclass(frozen=True, slots=True)
class CellPlan:
    """How one ``(asset, bucket)`` cell becomes bits."""

    asset: int
    bucket: int
    kind: str
    values: tuple[int, ...]
    bit_keys: tuple[VarKey, ...]
    value_form: LinearForm
    indicator_form: LinearForm | None
    validity_forms: tuple[LinearForm, ...]
    validity_products: tuple[tuple[LinearForm, LinearForm], ...]
    encoding: IntegerEncoding
    table_keys: tuple[VarKey, ...]
    indicator_key: VarKey | None
    residual_keys: tuple[VarKey, ...]
    min_lots: int

    @property
    def residual_values(self) -> tuple[int, ...]:
        return tuple(range(max(self.values) - self.min_lots + 1))

    def exact_cost_form(self, costs: Mapping[int, float]) -> LinearForm | None:
        """``f(x)`` as an exact linear form, or ``None`` if this cell has no table."""
        if self.kind == BLOCK:
            biggest = max(self.values)
            return LinearForm.constant_form(float(costs[0])) + LinearForm.of(
                self.bit_keys[0], float(costs[biggest]) - float(costs[0])
            )
        if not self.table_keys:
            return None
        return self.encoding.cost_form(self.table_keys, self.values, costs)

    def encode(self, x: int) -> dict[VarKey, int]:
        if x not in self.values:
            raise ValueError(
                f"{x} is not one of this cell's values {self.values} at "
                f"(asset {self.asset}, bucket {self.bucket})"
            )
        if self.kind == BLOCK:
            return {self.bit_keys[0]: int(x != 0)}
        if self.table_keys:
            return self.encoding.encode_value(self.table_keys, self.values, x)
        if self.indicator_key is None:
            return self.encoding.encode_value(self.bit_keys, self.values, x)
        bits: dict[VarKey, int] = dict.fromkeys(self.bit_keys, 0)
        if x == 0:
            return bits
        bits[self.indicator_key] = 1
        bits.update(
            self.encoding.encode_value(
                self.residual_keys, self.residual_values, x - self.min_lots
            )
        )
        return bits

    def decode(self, bits: Mapping[VarKey, int]) -> int:
        if self.kind == BLOCK:
            return max(self.values) if bits[self.bit_keys[0]] else 0
        if self.table_keys:
            return self.encoding.decode_value(self.table_keys, self.values, bits)
        if self.indicator_key is None:
            return self.encoding.decode_value(self.bit_keys, self.values, bits)
        if not bits[self.indicator_key]:
            if any(bits[k] for k in self.residual_keys):
                raise ValueError(
                    f"cell (asset {self.asset}, bucket {self.bucket}) has residual bits "
                    "set while its indicator is clear; spec section 6.2's penalty "
                    "forbids this state"
                )
            return 0
        return self.min_lots + self.encoding.decode_value(
            self.residual_keys, self.residual_values, bits
        )


@dataclass(frozen=True, slots=True)
class Layout:
    """Every cell's plan, plus the registry every bit was named into."""

    cells: tuple[CellPlan, ...]
    registry: VarRegistry
    encoding_name: str
    n_assets: int
    n_buckets: int

    def cell(self, asset: int, bucket: int) -> CellPlan:
        return self.cells[asset * self.n_buckets + bucket]


def _cell_values(instance: Instance, i: int, minimum: int) -> tuple[int, ...]:
    cap = min(instance.max_lots_per_bucket, instance.initial_lots[i])
    if minimum <= 1:
        return tuple(range(cap + 1))
    return (0, *range(minimum, cap + 1))


def _constraint_view(instance: Instance):
    """Read the constraint set once, dispatching on type rather than on a name."""
    minimums: dict[int, int] = {}
    blocks: dict[tuple[int, int], int] = {}
    has_cardinality = False
    for c in instance.constraints:
        if isinstance(c, FullLiquidation):
            continue
        if isinstance(c, Cardinality):
            has_cardinality = True
        elif isinstance(c, MinParticipation):
            minimums = dict(enumerate(c.min_lots))
        elif isinstance(c, BlockTrades):
            blocks = {(b.asset, b.bucket): b.lots for b in c.blocks}
        else:
            raise NotImplementedError(
                f"the layout planner does not know constraint {c.name!r} "
                f"({type(c).__name__}). Dispatch is on type, never on the name, so a "
                "new constraint must be taught here rather than inherited by accident"
            )
    return minimums, blocks, has_cardinality


def plan_layout(instance: Instance, encoding: IntegerEncoding) -> Layout:
    """Name every bit and decide how each cell's value is built from them."""
    minimums, blocks, has_cardinality = _constraint_view(instance)
    builder = RegistryBuilder()
    cells: list[CellPlan] = []

    for i in range(instance.n_assets):
        for t in range(instance.n_buckets):
            minimum = int(minimums.get(i, 1))
            block_lots = blocks.get((i, t))

            if block_lots is not None:
                key: VarKey = ("block", i, t)
                builder.add(key)
                cells.append(
                    CellPlan(
                        asset=i, bucket=t, kind=BLOCK, values=(0, int(block_lots)),
                        bit_keys=(key,),
                        value_form=LinearForm.of(key, float(block_lots)),
                        indicator_form=LinearForm.of(key, 1.0),
                        validity_forms=(), validity_products=(),
                        encoding=encoding, table_keys=(), indicator_key=None,
                        residual_keys=(), min_lots=minimum,
                    )
                )
                continue

            values = _cell_values(instance, i, minimum)
            kind = SEMI_CONTINUOUS if minimum > 1 else PLAIN

            if encoding.supports_exact_table or len(values) <= 1:
                keys = encoding.bit_keys(("x", i, t), values)
                for k in keys:
                    builder.add(k)
                cells.append(
                    CellPlan(
                        asset=i, bucket=t, kind=kind, values=values, bit_keys=keys,
                        value_form=encoding.value_form(keys, values),
                        indicator_form=encoding.indicator_form(keys, values),
                        validity_forms=encoding.validity_forms(keys, values),
                        validity_products=encoding.validity_products(keys, values),
                        encoding=encoding, table_keys=keys, indicator_key=None,
                        residual_keys=(), min_lots=minimum,
                    )
                )
                continue

            if kind == PLAIN:
                keys = encoding.bit_keys(("x", i, t), values)
                for k in keys:
                    builder.add(k)
                cells.append(
                    CellPlan(
                        asset=i, bucket=t, kind=kind, values=values, bit_keys=keys,
                        value_form=encoding.value_form(keys, values),
                        indicator_form=None, validity_forms=(), validity_products=(),
                        encoding=encoding, table_keys=(), indicator_key=None,
                        residual_keys=(), min_lots=minimum,
                    )
                )
                continue

            # Spec section 6.2: x = m*y + residual, with the residual forced to
            # zero when y is clear. The penalty itself is Task 7's; the layout
            # only has to name the bits and build the value.
            y_key: VarKey = ("y", i, t)
            builder.add(y_key)
            residual_values = tuple(range(max(values) - minimum + 1))
            residual_keys = encoding.bit_keys(("x", i, t), residual_values)
            for k in residual_keys:
                builder.add(k)
            value_form = LinearForm.of(y_key, float(minimum)) + encoding.value_form(
                residual_keys, residual_values
            )
            cells.append(
                CellPlan(
                    asset=i, bucket=t, kind=SEMI_CONTINUOUS, values=values,
                    bit_keys=(y_key, *residual_keys), value_form=value_form,
                    indicator_form=LinearForm.of(y_key, 1.0),
                    validity_forms=(), validity_products=(),
                    encoding=encoding, table_keys=(), indicator_key=y_key,
                    residual_keys=residual_keys, min_lots=minimum,
                )
            )

    if has_cardinality and any(c.indicator_form is None for c in cells):
        raise ValueError(
            "cardinality needs a 1[x > 0] indicator on every cell, and at least one "
            "cell has none. The generator only ever emits Cardinality alongside "
            "MinParticipation, which supplies it; this pairing is unsupported"
        )

    return Layout(
        cells=tuple(cells),
        registry=builder.freeze(),
        encoding_name=encoding.name,
        n_assets=instance.n_assets,
        n_buckets=instance.n_buckets,
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_layout.py -q`
Expected: every test in the file passes.

- [ ] **Step 6: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 7: Commit**

```bash
git add src/quantic/encoding/layout.py src/quantic/encoding/registry.py tests/encoding/test_layout.py
git commit -m "feat(encoding): plan the bit layout from the constraint set"
```

---

## Task 5: The exactly-representable objective terms

**Files:**
- Create: `src/quantic/encoding/expand.py`
- Modify: `src/quantic/encoding/layout.py` (add `encode_schedule`)
- Test: `tests/encoding/test_expand.py`

**Interfaces:**
- Consumes: `Layout`, `CellPlan` (Task 4); `QuboBuilder`, `LinearForm`, `Qubo` (Task 2); `VarRegistry` (Task 1); `Instance`, `Schedule`, `VarianceRisk`, `CVaRRisk`, `impact_model_for`, `evaluate` (`problem/`).
- Produces:
  - `encode_schedule(layout, schedule, registry) -> np.ndarray` — in `layout.py`.
  - `cell_temporary_cost(instance, i, t, lots) -> float`
  - `QuboBreakdown(spread, temporary, permanent, risk, approximand)` with `total -> Qubo`.
  - `expand_objective(instance, layout, registry) -> QuboBreakdown`.

**The registry seam.** `plan_layout` names only the cell bits. Cardinality slack (Task 7) and the CVaR grids (Task 8) add more, and every `Qubo` in one instance must share one registry — column *k* has to mean the same bit everywhere. So expansion takes the **final** registry as a parameter rather than reading `layout.registry`. `LinearForm` is keyed by name, not index, so a form built against the cells is equally valid against the final registry. Task 10 assembles that final registry; until then, tests pass `layout.registry`.

**Why `cell_temporary_cost` exists.** It reproduces exactly what `objective.evaluate` computes for one cell, so the cost table under a table encoding is bit-identical to the reference, and Task 6's fit has a single source of truth to fit against. There is deliberately no second evaluator.

**What this task refuses.** D1 on under an encoding with no exact table — that is Task 6 — and `CVaRRisk` — that is Task 8. Both raise `NotImplementedError` naming the successor task, so this task lands on a green suite and each successor is verifiable alone. This is the staging M2b used for its `isinstance` fix.

- [ ] **Step 1: Write the failing test**

Create `tests/encoding/test_expand.py`:

```python
import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.integer import ENCODINGS, BinaryEncoding, OneHotEncoding
from quantic.encoding.layout import encode_schedule, plan_layout
from quantic.encoding.expand import cell_temporary_cost, expand_objective
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import evaluate
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
REL = 1e-9


def _instance(**kw) -> Instance:
    assets = tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B"))
    params = MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
            for j, s in enumerate(("A", "B"))
        ),
        covariance=np.array([[1.0, 0.3], [0.3, 1.0]]) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    base = dict(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(4, 4), max_lots_per_bucket=4, params=params,
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def _all_schedules(instance):
    """Every schedule on the 2x2 grid within the cell bounds. Small by design."""
    cap = instance.max_lots_per_bucket
    for a0 in range(cap + 1):
        for a1 in range(cap + 1):
            for b0 in range(cap + 1):
                for b1 in range(cap + 1):
                    yield Schedule(lots=((a0, a1), (b0, b1)))


def _energies(instance, encoding, schedule):
    layout = plan_layout(instance, encoding)
    reg = layout.registry
    breakdown = expand_objective(instance, layout, reg)
    bits = encode_schedule(layout, schedule, reg)
    return breakdown, bits


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_spread_term_matches_the_evaluator_on_every_schedule(encoding):
    inst = _instance(risk=VarianceRisk(lam=0.0))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, encoding, schedule)
        assert breakdown.spread.energy(bits) == pytest.approx(
            evaluate(inst, schedule).spread_cost, rel=REL
        )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_linear_impact_term_matches_the_evaluator_on_every_schedule(encoding):
    inst = _instance(risk=VarianceRisk(lam=0.0))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, encoding, schedule)
        assert breakdown.temporary.energy(bits) == pytest.approx(
            evaluate(inst, schedule).temporary_impact_cost, rel=REL
        )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_permanent_term_matches_the_evaluator_on_every_schedule(encoding):
    """The cross-bucket term: a path-dependent quadratic, easiest to get wrong."""
    inst = _instance(risk=VarianceRisk(lam=0.0))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, encoding, schedule)
        assert breakdown.permanent.energy(bits) == pytest.approx(
            evaluate(inst, schedule).permanent_impact_cost, rel=REL
        )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_variance_term_matches_the_evaluator_on_every_schedule(encoding):
    inst = _instance(risk=VarianceRisk(lam=2.5e-6))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, encoding, schedule)
        assert breakdown.risk.energy(bits) == pytest.approx(
            evaluate(inst, schedule).risk_cost, rel=REL
        )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_total_matches_the_evaluator_on_every_schedule(encoding):
    inst = _instance(risk=VarianceRisk(lam=2.5e-6))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, encoding, schedule)
        assert breakdown.total.energy(bits) == pytest.approx(
            evaluate(inst, schedule).total, rel=REL
        )


def test_the_linear_impact_cost_is_exactly_quadratic_in_lots():
    """The identity the log expansion relies on: cost(x) == cost(1) * x**2."""
    inst = _instance()
    one = cell_temporary_cost(inst, 0, 0, 1)
    for x in range(5):
        assert cell_temporary_cost(inst, 0, 0, x) == pytest.approx(one * x * x, rel=1e-12)


def test_concave_impact_is_exact_under_a_table_encoding():
    inst = _instance(dials=Dials(concave_impact=True), risk=VarianceRisk(lam=0.0))
    for schedule in _all_schedules(inst):
        breakdown, bits = _energies(inst, OneHotEncoding(), schedule)
        assert breakdown.approximand is None
        assert breakdown.temporary.energy(bits) == pytest.approx(
            evaluate(inst, schedule).temporary_impact_cost, rel=REL
        )


def test_concave_impact_under_the_log_encoding_defers_to_task_six():
    inst = _instance(dials=Dials(concave_impact=True))
    layout = plan_layout(inst, BinaryEncoding())
    with pytest.raises(NotImplementedError, match="approximand"):
        expand_objective(inst, layout, layout.registry)


def test_cvar_risk_defers_to_task_eight():
    import dataclasses

    from quantic.problem.risk import CVaRRisk
    from quantic.problem.scenarios import ScenarioSet

    inst = _instance()
    scenarios = ScenarioSet(
        returns=np.zeros((2, 2, 2)), weights=np.array([0.5, 0.5]), seed=0
    )
    cvar = dataclasses.replace(
        inst, risk=CVaRRisk(lam=1e-6, alpha=0.95, scenarios=scenarios)
    )
    layout = plan_layout(cvar, BinaryEncoding())
    with pytest.raises(NotImplementedError, match="CVaRRisk"):
        expand_objective(cvar, layout, layout.registry)


def test_an_unknown_risk_spec_is_refused_even_carrying_a_lam():
    """Dispatch on type. The same impostor M2b caught in the evaluator."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class _NotVariance:
        lam: float = 1.0
        name: str = "not_variance"

        def describe(self) -> dict:
            return {"name": self.name}

    inst = dataclasses.replace(_instance(), risk=_NotVariance())
    layout = plan_layout(inst, BinaryEncoding())
    with pytest.raises(NotImplementedError, match="_NotVariance"):
        expand_objective(inst, layout, layout.registry)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_encode_schedule_produces_a_binary_vector_over_the_registry(encoding):
    inst = _instance()
    layout = plan_layout(inst, encoding)
    bits = encode_schedule(layout, inst.witness, layout.registry)
    assert bits.shape == (len(layout.registry),)
    assert set(np.unique(bits)) <= {0, 1}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_expand.py -q`
Expected: FAIL — `ImportError: cannot import name 'encode_schedule'` (and the missing `expand` module).

- [ ] **Step 3: Add `encode_schedule` to `layout.py`**

Append to `src/quantic/encoding/layout.py` — at the end of the file, after `plan_layout`, and add `import numpy as np` and `from quantic.problem.schedule import Schedule` to the existing import block at the top rather than mid-file:

```python
def encode_schedule(
    layout: Layout, schedule: Schedule, registry: VarRegistry
) -> np.ndarray:
    """The canonical bit vector for ``schedule``, in ``registry`` order.

    Bits the layout does not name -- cardinality slack, the CVaR grids -- are
    left at zero here; their values depend on the penalty and gadget
    configuration and are filled in by the callers that own them.
    """
    if (schedule.n_assets, schedule.n_buckets) != (layout.n_assets, layout.n_buckets):
        raise ValueError(
            f"schedule shape ({schedule.n_assets}, {schedule.n_buckets}) does not match "
            f"the layout ({layout.n_assets}, {layout.n_buckets})"
        )
    bits: dict[VarKey, int] = {}
    for plan in layout.cells:
        bits.update(plan.encode(int(schedule.lots[plan.asset][plan.bucket])))
    return np.array([bits.get(k, 0) for k in registry.keys], dtype=np.int64)
```

- [ ] **Step 4: Write the expansion**

Create `src/quantic/encoding/expand.py`:

```python
"""Spec section 5.2's objective, as one QUBO per term.

``problem/objective.py`` returns its four terms separately, for the stated
reason that "a single scalar says nothing about which term is wrong". The
encoder mirrors that: the gate compares field by field, so a mismatch names
the term that broke rather than only the instance.

Three of the four terms are exactly representable under every encoding:
spread is linear in the lots, permanent impact is a cross-bucket quadratic,
and variance is a quadratic in the remaining holdings. The fourth -- temporary
impact -- is exact whenever the encoding offers a per-value cost table, and
under the log encoding it is exact only while dial D1 is off. D1 on under log
is Task 6's explicit approximand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.encoding.layout import BLOCK, Layout
from quantic.encoding.qubo import LinearForm, Qubo, QuboBuilder
from quantic.encoding.registry import VarRegistry
from quantic.problem.instance import Instance
from quantic.problem.objective import impact_model_for
from quantic.problem.risk import CVaRRisk, VarianceRisk


@dataclass(frozen=True, eq=False)
class QuboBreakdown:
    """One QUBO per objective term, plus whatever was approximated to get them."""

    spread: Qubo
    temporary: Qubo
    permanent: Qubo
    risk: Qubo
    approximand: object | None = None

    @property
    def total(self) -> Qubo:
        return self.spread + self.temporary + self.permanent + self.risk


def cell_temporary_cost(instance: Instance, i: int, t: int, lots: int) -> float:
    """Exactly what ``objective.evaluate`` charges cell ``(i, t)`` for ``lots``.

    The single source of truth for the cost table under a table encoding and
    for Task 6's quadratic fit under the log encoding. A second implementation
    is how the encoder and the evaluator would silently drift apart.
    """
    if lots == 0:
        return 0.0
    a = instance.params.assets[i]
    p = instance.params.impact_params(i)
    model = impact_model_for(instance, i)
    traded = float(lots) * float(instance.assets[i].lot_size)
    return float(model.temporary_cost(traded, p) * traded * a.price)


def _spread(instance: Instance, layout: Layout, registry: VarRegistry) -> Qubo:
    builder = QuboBuilder(registry)
    for plan in layout.cells:
        a = instance.params.assets[plan.asset]
        per_lot = a.half_spread * float(instance.assets[plan.asset].lot_size)
        builder.add_form(plan.value_form, per_lot)
    return builder.build()


def _temporary(instance: Instance, layout: Layout, registry: VarRegistry) -> Qubo:
    builder = QuboBuilder(registry)
    for plan in layout.cells:
        costs = {v: cell_temporary_cost(instance, plan.asset, plan.bucket, v)
                 for v in plan.values}
        exact = plan.exact_cost_form(costs)
        if exact is not None:
            builder.add_form(exact)
            continue
        if instance.dials.concave_impact:
            raise NotImplementedError(
                f"concave impact (D1) has no exact quadratic form under the "
                f"{layout.encoding_name!r} encoding, so it needs an explicit "
                "approximand and a reported error. See M3a plan Task 6"
            )
        # D1 off is Almgren-Chriss, whose cost is exactly quadratic in the lots,
        # so one coefficient carries the whole cell: cost(x) == cost(1) * x**2.
        builder.add_square(
            plan.value_form,
            cell_temporary_cost(instance, plan.asset, plan.bucket, 1),
        )
    return builder.build()


def _permanent(instance: Instance, layout: Layout, registry: VarRegistry) -> Qubo:
    builder = QuboBuilder(registry)
    for i in range(instance.n_assets):
        a = instance.params.assets[i]
        lot = float(instance.assets[i].lot_size)
        coef = a.gamma * a.sigma_bucket * a.price * lot * lot / a.bucket_volume_shares
        for t in range(instance.n_buckets):
            here = layout.cell(i, t).value_form
            # The evaluator's midpoint: everything displaced before this bucket,
            # plus half of our own, because we trade through it.
            displaced = here.scaled(0.5)
            for s in range(t):
                displaced = displaced + layout.cell(i, s).value_form
            builder.add_product(displaced, here, coef)
    return builder.build()


def holdings_form(instance: Instance, layout: Layout, i: int, t: int) -> LinearForm:
    """Shares of asset ``i`` still held after bucket ``t``, as a linear form."""
    lot = float(instance.assets[i].lot_size)
    form = LinearForm.constant_form(lot * float(instance.initial_lots[i]))
    for s in range(t + 1):
        form = form + layout.cell(i, s).value_form.scaled(-lot)
    return form


def _variance(
    instance: Instance, layout: Layout, registry: VarRegistry, lam: float
) -> Qubo:
    builder = QuboBuilder(registry)
    if lam == 0.0:
        return builder.build()
    sigma = np.asarray(instance.params.price_covariance(), dtype=float)
    for t in range(instance.n_buckets):
        forms = [holdings_form(instance, layout, i, t) for i in range(instance.n_assets)]
        for i in range(instance.n_assets):
            for j in range(instance.n_assets):
                builder.add_product(forms[i], forms[j], lam * float(sigma[i, j]))
    return builder.build()


def expand_objective(
    instance: Instance, layout: Layout, registry: VarRegistry
) -> QuboBreakdown:
    """Spec section 5.2's four terms as four QUBOs over ``registry``."""
    risk = instance.risk
    if isinstance(risk, VarianceRisk):
        risk_qubo = _variance(instance, layout, registry, risk.lam)
    elif isinstance(risk, CVaRRisk):
        raise NotImplementedError(
            "CVaRRisk needs the Rockafellar-Uryasev gadget: a binary-expanded zeta "
            "plus two auxiliaries per scenario. See M3a plan Task 8"
        )
    else:
        raise NotImplementedError(
            f"expand_objective does not know how to encode risk spec {risk.name!r} "
            f"({type(risk).__name__}). Dispatch is on type, never on a lam attribute"
        )
    return QuboBreakdown(
        spread=_spread(instance, layout, registry),
        temporary=_temporary(instance, layout, registry),
        permanent=_permanent(instance, layout, registry),
        risk=risk_qubo,
    )
```

`BLOCK` is imported for symmetry with later tasks; drop it from the import if ruff reports it unused.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_expand.py -q`
Expected: every test in the file passes.

- [ ] **Step 6: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 7: Commit**

```bash
git add src/quantic/encoding/expand.py src/quantic/encoding/layout.py tests/encoding/test_expand.py
git commit -m "feat(encoding): expand the exactly representable objective terms"
```

---

## Task 6: The D1 × log approximand, declared and measured

**Files:**
- Modify: `src/quantic/encoding/expand.py`
- Test: `tests/encoding/test_approximand.py`

**Interfaces:**
- Consumes: `cell_temporary_cost`, `QuboBreakdown` (Task 5); `CellPlan` (Task 4).
- Produces: `ApproximandReport(coefficients, max_abs_error, max_rel_error, worst_cell)` with `temporary_cost(schedule) -> float` and `describe() -> dict`. `expand_objective` now fills `QuboBreakdown.approximand` instead of raising for D1 × log.

**What is being claimed.** `f(x) = [Y σ/(δ+1)]·(q/V)^δ·q·p` is not a polynomial, so no quadratic reproduces it on every lot value. The encoder substitutes a named `q̂(x) = b·x + c·x²` and says so. The gate then makes **two exact assertions** — the QUBO equals the objective computed *with* `q̂`, and `max|f − q̂|` is what the report claims — rather than one assertion with a tolerance wide enough to hide a real bug. This is the pattern the spec's §3 decision table fixes, applied here for the first time.

**No constant term.** `f(0) = 0` exactly. A fitted constant would shift every energy, including the empty schedule's, and would survive into the penalty-derivation bound as phantom objective range.

**Why the error is computed, not bounded.** The lot domain is finite and small, so `max|f − q̂|` is a maximum over a handful of integers. An asymptotic bound would be looser and less useful, and nothing here needs one.

- [ ] **Step 1: Write the failing test**

Create `tests/encoding/test_approximand.py`:

```python
import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.expand import cell_temporary_cost, expand_objective
from quantic.encoding.integer import BinaryEncoding, DomainWallEncoding, OneHotEncoding
from quantic.encoding.layout import encode_schedule, plan_layout
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import evaluate
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
REL = 1e-9


def _instance(**kw) -> Instance:
    assets = tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B"))
    params = MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
            for j, s in enumerate(("A", "B"))
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    base = dict(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(4, 4), max_lots_per_bucket=4, params=params,
        dials=Dials(concave_impact=True), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=0.0), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def _all_schedules(cap=4):
    for a0 in range(cap + 1):
        for a1 in range(cap + 1):
            for b0 in range(cap + 1):
                for b1 in range(cap + 1):
                    yield Schedule(lots=((a0, a1), (b0, b1)))


def test_the_log_encoding_now_reports_an_approximand_instead_of_raising():
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    breakdown = expand_objective(inst, layout, layout.registry)
    assert breakdown.approximand is not None
    assert breakdown.approximand.max_abs_error > 0.0


@pytest.mark.parametrize("encoding", [OneHotEncoding(), DomainWallEncoding()])
def test_a_table_encoding_still_reports_no_approximand(encoding):
    """The absence of a report is the claim that the encoding was exact."""
    inst = _instance()
    layout = plan_layout(inst, encoding)
    assert expand_objective(inst, layout, layout.registry).approximand is None


def test_the_qubo_equals_the_objective_computed_with_the_substituted_term():
    """Assertion one of two: exact, no modelling tolerance."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    breakdown = expand_objective(inst, layout, layout.registry)
    report = breakdown.approximand
    for schedule in _all_schedules():
        bits = encode_schedule(layout, schedule, layout.registry)
        expected = dataclasses.replace(
            evaluate(inst, schedule),
            temporary_impact_cost=report.temporary_cost(schedule),
        )
        assert breakdown.total.energy(bits) == pytest.approx(expected.total, rel=REL)


def test_the_reported_error_is_the_true_maximum_over_the_lot_grid():
    """Assertion two of two: the report does not flatter itself."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    report = expand_objective(inst, layout, layout.registry).approximand
    worst = 0.0
    for plan in layout.cells:
        b, c = report.coefficients[(plan.asset, plan.bucket)]
        for v in plan.values:
            exact = cell_temporary_cost(inst, plan.asset, plan.bucket, v)
            worst = max(worst, abs(exact - (b * v + c * v * v)))
    assert report.max_abs_error == pytest.approx(worst, rel=1e-12)


def test_the_fit_is_exact_at_zero_lots():
    """f(0) == 0 exactly; a fitted constant would shift every energy."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    report = expand_objective(inst, layout, layout.registry).approximand
    empty = Schedule(lots=((0, 0), (0, 0)))
    assert report.temporary_cost(empty) == 0.0


def test_the_report_is_json_safe_for_the_results_row():
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    import json

    payload = expand_objective(inst, layout, layout.registry).approximand.describe()
    assert json.loads(json.dumps(payload))["max_abs_error"] >= 0.0


def test_no_approximand_appears_when_d1_is_off_under_the_log_encoding():
    inst = _instance(dials=Dials())
    layout = plan_layout(inst, BinaryEncoding())
    assert expand_objective(inst, layout, layout.registry).approximand is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_approximand.py -q`
Expected: FAIL — `NotImplementedError` mentioning Task 6, raised by `_temporary`.

- [ ] **Step 3: Write the implementation**

In `src/quantic/encoding/expand.py`, add after `cell_temporary_cost`:

```python
@dataclass(frozen=True)
class ApproximandReport:
    """The quadratic substituted for concave impact, and what it cost.

    Emitted only when the encoding offers no exact per-value table -- that is,
    only for the log encoding with dial D1 on. Its absence is the encoder's
    claim that the term was represented exactly.
    """

    coefficients: dict[tuple[int, int], tuple[float, float]]
    max_abs_error: float
    max_rel_error: float
    worst_cell: tuple[int, int]

    def temporary_cost(self, schedule) -> float:
        """The substituted temporary-impact term, summed over cells."""
        total = 0.0
        for (i, t), (b, c) in self.coefficients.items():
            x = float(schedule.lots[i][t])
            total += b * x + c * x * x
        return total

    def describe(self) -> dict:
        """JSON-safe, for the results row."""
        return {
            "kind": "quadratic_least_squares",
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "worst_cell": list(self.worst_cell),
        }


def _fit_quadratic(values: tuple[int, ...], costs: dict[int, float]) -> tuple[float, float]:
    """Least-squares ``b*x + c*x**2`` over the cell's admissible lot values.

    No constant term: ``f(0) == 0`` exactly, and a fitted offset would shift
    every energy including the empty schedule's.
    """
    grid = [v for v in values if v != 0]
    if not grid:
        return 0.0, 0.0
    design = np.array([[float(v), float(v) * float(v)] for v in grid], dtype=float)
    target = np.array([costs[v] for v in grid], dtype=float)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(solution[0]), float(solution[1])
```

Then replace `_temporary` with a version that returns both the QUBO and the report:

```python
def _temporary(
    instance: Instance, layout: Layout, registry: VarRegistry
) -> tuple[Qubo, ApproximandReport | None]:
    builder = QuboBuilder(registry)
    coefficients: dict[tuple[int, int], tuple[float, float]] = {}
    max_abs = 0.0
    max_rel = 0.0
    worst: tuple[int, int] = (0, 0)

    for plan in layout.cells:
        costs = {v: cell_temporary_cost(instance, plan.asset, plan.bucket, v)
                 for v in plan.values}
        exact = plan.exact_cost_form(costs)
        if exact is not None:
            builder.add_form(exact)
            continue
        if not instance.dials.concave_impact:
            # Almgren-Chriss is exactly quadratic in the lots, so one
            # coefficient carries the whole cell: cost(x) == cost(1) * x**2.
            builder.add_square(
                plan.value_form,
                cell_temporary_cost(instance, plan.asset, plan.bucket, 1),
            )
            continue
        b, c = _fit_quadratic(plan.values, costs)
        coefficients[(plan.asset, plan.bucket)] = (b, c)
        builder.add_form(plan.value_form, b)
        builder.add_square(plan.value_form, c)
        for v in plan.values:
            error = abs(costs[v] - (b * v + c * v * v))
            if error > max_abs:
                max_abs, worst = error, (plan.asset, plan.bucket)
            if costs[v] != 0.0:
                max_rel = max(max_rel, error / abs(costs[v]))

    if not coefficients:
        return builder.build(), None
    return builder.build(), ApproximandReport(
        coefficients=coefficients,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        worst_cell=worst,
    )
```

Finally, in `expand_objective`, replace the `temporary=` line so the report travels with the breakdown:

```python
    temporary, approximand = _temporary(instance, layout, registry)
    return QuboBreakdown(
        spread=_spread(instance, layout, registry),
        temporary=temporary,
        permanent=_permanent(instance, layout, registry),
        risk=risk_qubo,
        approximand=approximand,
    )
```

Narrow `QuboBreakdown.approximand`'s annotation from `object | None` to `ApproximandReport | None` now that the type exists.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_approximand.py tests/encoding/test_expand.py -q`
Expected: every test in both files passes. `test_concave_impact_under_the_log_encoding_defers_to_task_six` in `test_expand.py` now fails by design — **delete that test**, since its deferral has been implemented; the two assertions in `test_approximand.py` replace it.

- [ ] **Step 5: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 6: Commit**

```bash
git add src/quantic/encoding/expand.py tests/encoding/
git commit -m "feat(encoding): substitute a declared quadratic for concave impact under log"
```

---

## Task 7: Constraint penalties and derived bounds

**Files:**
- Create: `src/quantic/encoding/penalty.py`
- Test: `tests/encoding/test_penalty.py`

**Interfaces:**
- Consumes: `Layout`, `CellPlan` (Task 4); `bounded_coefficients` (Task 3); `QuboBuilder`, `LinearForm`, `Qubo` (Task 2); `QuboBreakdown` (Tasks 5–6).
- Produces:
  - `PenaltyStrengths(validity, full_liquidation, min_participation, cardinality, cvar, safety_factor, strategy)` with `describe() -> dict`.
  - `penalty_aux_keys(instance, layout) -> tuple[VarKey, ...]` — the cardinality slack bits.
  - `slack_bits(instance, layout, schedule) -> dict[VarKey, int]`.
  - `expand_penalties(instance, layout, registry, strengths) -> Qubo`.
  - `derive_penalties(objective: Qubo, layout, *, safety_factor=2.0) -> PenaltyStrengths`.

**Why the derivation is structural.** §6.3 is blunt about the two failure modes: too small and a constraint-violating solution is reported as a win; too large and the penalty swamps the objective's dynamic range until the annealer is solving feasibility while being reported as an optimiser. The bound used here is the largest objective change any subset of one cell's bits can produce, read directly off the expanded objective matrix. Every penalty residual in this milestone is an integer square, so a one-unit violation costs at least `1`, and the strength need only exceed that gain times a recorded safety factor.

**Cardinality's slack range.** `Σ_i y[i,t] ≤ K` becomes `Σ_i y[i,t] + s_t = K` with `s_t ∈ [0, K]` — **not** `[0, N−K]`. The slack absorbs the shortfall, not the surplus.

**What carries no penalty.** Block cells and, under a table encoding, min participation: both are enforced by the layout, so there is nothing to tune. That is the §6.1 result from Task 4 showing up as an absent term rather than a comment.

- [ ] **Step 1: Write the failing test**

Create `tests/encoding/test_penalty.py`:

```python
import dataclasses

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.expand import expand_objective
from quantic.encoding.integer import ENCODINGS, BinaryEncoding, OneHotEncoding
from quantic.encoding.layout import encode_schedule, plan_layout
from quantic.encoding.penalty import (
    derive_penalties,
    expand_penalties,
    penalty_aux_keys,
    slack_bits,
)
from quantic.encoding.registry import VarRegistry
from quantic.problem.constraints.block_trades import Block, BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import VarianceRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(**kw) -> Instance:
    assets = tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B"))
    params = MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
            for j, s in enumerate(("A", "B"))
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    base = dict(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(4, 4), max_lots_per_bucket=4, params=params,
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def _d2_instance():
    return _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=1), MinParticipation((3, 3))),
        witness=Schedule(lots=((4, 0), (0, 4))),
    )


def _d4_instance():
    return _instance(
        dials=Dials(block_trades=True),
        constraints=(FullLiquidation(), BlockTrades((Block(asset=0, bucket=0, lots=2),))),
        witness=Schedule(lots=((2, 2), (2, 2))),
    )


def _setup(instance, encoding):
    layout = plan_layout(instance, encoding)
    registry = VarRegistry(
        keys=layout.registry.keys + penalty_aux_keys(instance, layout)
    )
    objective = expand_objective(instance, layout, registry).total
    strengths = derive_penalties(objective, layout)
    penalties = expand_penalties(instance, layout, registry, strengths)
    return layout, registry, objective, strengths, penalties


def _bits(layout, registry, instance, schedule):
    vector = encode_schedule(layout, schedule, registry)
    extra = slack_bits(instance, layout, schedule)
    for key, value in extra.items():
        vector[registry.index(key)] = value
    return vector


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize(
    "factory", [_instance, _d2_instance, _d4_instance], ids=["none", "d2", "d4"]
)
def test_the_penalty_block_is_exactly_zero_at_a_feasible_witness(encoding, factory):
    """Exactly zero, not approximately: a feasible point pays nothing."""
    inst = factory()
    layout, registry, _, _, penalties = _setup(inst, encoding)
    assert penalties.energy(_bits(layout, registry, inst, inst.witness)) == 0.0


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_an_under_liquidating_schedule_pays_a_penalty(encoding):
    inst = _instance()
    layout, registry, _, _, penalties = _setup(inst, encoding)
    short = Schedule(lots=((2, 1), (2, 2)))
    assert penalties.energy(_bits(layout, registry, inst, short)) > 0.0


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_one_lot_violation_raises_the_total_energy(encoding):
    """The property derived penalties exist to guarantee."""
    inst = _instance()
    layout, registry, objective, _, penalties = _setup(inst, encoding)
    total = objective + penalties
    good = total.energy(_bits(layout, registry, inst, inst.witness))
    short = total.energy(_bits(layout, registry, inst, Schedule(lots=((2, 1), (2, 2)))))
    assert short > good


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_breaching_cardinality_pays_a_penalty(encoding):
    inst = _d2_instance()
    layout, registry, _, _, penalties = _setup(inst, encoding)
    both_active = Schedule(lots=((4, 0), (4, 0)))
    assert penalties.energy(_bits(layout, registry, inst, both_active)) > 0.0


def test_a_residual_set_without_its_indicator_pays_a_penalty():
    """Spec section 6.2's min-participation term, under the log encoding."""
    inst = _d2_instance()
    layout, registry, _, _, penalties = _setup(inst, BinaryEncoding())
    vector = np.zeros(len(registry), dtype=np.int64)
    residual = [k for k in registry.keys if k[0] == "x"]
    vector[registry.index(residual[0])] = 1
    assert penalties.energy(vector) > 0.0


def test_a_table_encoding_needs_no_min_participation_penalty():
    """D2 is free under one-hot: the forbidden values are simply not there."""
    inst = _d2_instance()
    _, _, _, strengths, _ = _setup(inst, OneHotEncoding())
    assert strengths.min_participation == 0.0


def test_an_invalid_one_hot_state_pays_a_validity_penalty():
    inst = _instance()
    layout, registry, _, _, penalties = _setup(inst, OneHotEncoding())
    vector = np.zeros(len(registry), dtype=np.int64)  # no bit set anywhere
    assert penalties.energy(vector) > 0.0


def test_cardinality_slack_spans_zero_to_k():
    """Sum(y) + slack == K. The slack absorbs the shortfall, not the surplus."""
    inst = _d2_instance()
    layout = plan_layout(inst, BinaryEncoding())
    keys = penalty_aux_keys(inst, layout)
    assert keys  # k = 1, so one slack bit per bucket
    assert all(k[0] == "card" for k in keys)


def test_derivation_is_deterministic_and_positive():
    inst = _instance()
    layout, _, objective, strengths, _ = _setup(inst, BinaryEncoding())
    again = derive_penalties(objective, layout)
    assert strengths == again
    assert strengths.full_liquidation > 0.0
    assert strengths.safety_factor == 2.0
    assert strengths.strategy == "derived_bounds"


def test_the_strengths_are_json_safe_for_the_results_row():
    import json

    inst = _instance()
    _, _, _, strengths, _ = _setup(inst, BinaryEncoding())
    assert json.loads(json.dumps(strengths.describe()))["strategy"] == "derived_bounds"


def test_an_unknown_constraint_type_is_refused_even_carrying_a_known_name():
    @dataclasses.dataclass(frozen=True)
    class _NotFullLiquidation:
        name: str = "full_liquidation"

        def violation(self, schedule, initial_lots) -> float:
            return 0.0

        def is_satisfied(self, schedule, initial_lots) -> bool:
            return True

        def describe(self) -> dict:
            return {"name": self.name}

    inst = _instance(constraints=(FullLiquidation(), _NotFullLiquidation()))
    with pytest.raises(NotImplementedError, match="_NotFullLiquidation"):
        plan_layout(inst, BinaryEncoding())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_penalty.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.encoding.penalty'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/encoding/penalty.py`:

```python
"""Hard constraints as penalties, and the magnitudes that make them bite.

Spec section 6.3 is blunt about why this is a first-class module rather than a
helper. Too small a penalty and a constraint-violating sample scores as cheap
and gets reported as a win. Too large and the penalty swamps the objective's
dynamic range: on real hardware the objective differences drop below the
analogue precision floor, and the annealer solves a feasibility problem while
being reported as an optimiser.

The derivation here is structural -- section 6.3's first strategy. The
adaptive strategy needs a reference sampler to measure a feasibility rate
against, and none exists until M4.

Two constraints carry no penalty at all. A block cell is one bit whose only
two values are the two the constraint allows, and under a table encoding the
values a venue minimum forbids are simply absent. Both are enforced by the
layout, so there is nothing to tune -- which is the section 6.1 encoding
result showing up as an absent term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.encoding.integer import bounded_coefficients
from quantic.encoding.layout import Layout
from quantic.encoding.qubo import LinearForm, Qubo, QuboBuilder
from quantic.encoding.registry import VarKey, VarRegistry
from quantic.problem.constraints.block_trades import BlockTrades
from quantic.problem.constraints.cardinality import Cardinality
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.constraints.min_participation import MinParticipation
from quantic.problem.instance import Instance
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class PenaltyStrengths:
    """One magnitude per penalty family, and how they were arrived at."""

    validity: float
    full_liquidation: float
    min_participation: float
    cardinality: float
    cvar: float
    safety_factor: float
    strategy: str = "derived_bounds"

    def describe(self) -> dict:
        """JSON-safe, for the results row -- section 6.3 requires it recorded."""
        return {
            "strategy": self.strategy,
            "safety_factor": self.safety_factor,
            "validity": self.validity,
            "full_liquidation": self.full_liquidation,
            "min_participation": self.min_participation,
            "cardinality": self.cardinality,
            "cvar": self.cvar,
        }


def _cardinality_of(instance: Instance) -> Cardinality | None:
    for c in instance.constraints:
        if isinstance(c, Cardinality):
            return c
    return None


def penalty_aux_keys(instance: Instance, layout: Layout) -> tuple[VarKey, ...]:
    """The cardinality slack bits, in registry order.

    ``sum_i y[i,t] + s_t == k`` with ``s_t`` in ``[0, k]``: the slack absorbs
    the shortfall below the cap, never a surplus above it.
    """
    cardinality = _cardinality_of(instance)
    if cardinality is None:
        return ()
    coefficients = bounded_coefficients(cardinality.k)
    return tuple(
        ("card", t, j)
        for t in range(layout.n_buckets)
        for j in range(len(coefficients))
    )


def slack_bits(
    instance: Instance, layout: Layout, schedule: Schedule
) -> dict[VarKey, int]:
    """Cardinality slack for ``schedule``, so a feasible point pays exactly zero."""
    cardinality = _cardinality_of(instance)
    if cardinality is None:
        return {}
    from quantic.encoding.integer import BinaryEncoding

    coefficients = bounded_coefficients(cardinality.k)
    encoder = BinaryEncoding()
    values = tuple(range(cardinality.k + 1))
    bits: dict[VarKey, int] = {}
    for t in range(layout.n_buckets):
        active = schedule.active_names(t)
        shortfall = cardinality.k - active
        keys = tuple(("card", t, j) for j in range(len(coefficients)))
        if shortfall < 0:
            bits.update(dict.fromkeys(keys, 0))
            continue
        bits.update(encoder.encode_value(keys, values, shortfall))
    return bits


def expand_penalties(
    instance: Instance,
    layout: Layout,
    registry: VarRegistry,
    strengths: PenaltyStrengths,
) -> Qubo:
    """Every hard constraint as a quadratic penalty over ``registry``."""
    builder = QuboBuilder(registry)

    for plan in layout.cells:
        for form in plan.validity_forms:
            builder.add_square(form, strengths.validity)
        for lhs, rhs in plan.validity_products:
            builder.add_product(lhs, rhs, strengths.validity)

    for c in instance.constraints:
        if isinstance(c, FullLiquidation):
            for i in range(layout.n_assets):
                form = LinearForm.constant_form(-float(instance.initial_lots[i]))
                for t in range(layout.n_buckets):
                    form = form + layout.cell(i, t).value_form
                builder.add_square(form, strengths.full_liquidation)

        elif isinstance(c, MinParticipation):
            for plan in layout.cells:
                if plan.indicator_key is None:
                    continue  # enforced by the layout under a table encoding
                complement = LinearForm.constant_form(1.0) + LinearForm.of(
                    plan.indicator_key, -1.0
                )
                for key in plan.residual_keys:
                    builder.add_product(
                        LinearForm.of(key, 1.0), complement, strengths.min_participation
                    )

        elif isinstance(c, Cardinality):
            coefficients = bounded_coefficients(c.k)
            for t in range(layout.n_buckets):
                form = LinearForm.constant_form(-float(c.k))
                for i in range(layout.n_assets):
                    indicator = layout.cell(i, t).indicator_form
                    if indicator is None:
                        raise ValueError(
                            f"cell (asset {i}, bucket {t}) has no indicator, so "
                            "cardinality cannot be expressed. plan_layout should have "
                            "refused this instance"
                        )
                    form = form + indicator
                for j, coef in enumerate(coefficients):
                    form = form + LinearForm.of(("card", t, j), float(coef))
                builder.add_square(form, strengths.cardinality)

        elif isinstance(c, BlockTrades):
            continue  # enforced by the layout: the cell is one bit

        else:
            raise NotImplementedError(
                f"expand_penalties does not know constraint {c.name!r} "
                f"({type(c).__name__}). Dispatch is on type, never on the name"
            )

    return builder.build()


def max_single_cell_gain(objective: Qubo, layout: Layout) -> float:
    """An upper bound on the objective change any one cell's bits can produce.

    Flipping a subset of one cell's bits moves the objective by at most the
    sum, over those bits, of the absolute coefficients in their rows and
    columns. Rigorous, cheap, and tighter than the whole objective's range --
    which is the difference between a penalty that bites and one that swamps
    the dynamic range section 6.4 measures.
    """
    q = np.abs(np.asarray(objective.Q, dtype=float))
    worst = 0.0
    for plan in layout.cells:
        total = 0.0
        for key in plan.bit_keys:
            k = objective.registry.index(key)
            total += float(q[k, k] + q[k, :].sum() + q[:, k].sum())
        worst = max(worst, total)
    return worst


def derive_penalties(
    objective: Qubo, layout: Layout, *, safety_factor: float = 2.0
) -> PenaltyStrengths:
    """Section 6.3's first strategy: a penalty larger than any gain it could buy.

    Every penalty residual in this milestone is an integer-valued square, so a
    one-unit violation costs at least ``1`` and the strength need only exceed
    the gain. The floor of ``1.0`` keeps the constraints binding when the
    objective is identically zero -- which happens whenever ``lam`` is zero and
    the dials are off.
    """
    if safety_factor <= 1.0:
        raise ValueError(
            f"safety_factor must exceed 1, got {safety_factor}: at or below 1 the "
            "penalty only ties the gain it is meant to beat"
        )
    strength = max(safety_factor * max_single_cell_gain(objective, layout), 1.0)
    needs_min_participation = any(c.indicator_key is not None for c in layout.cells)
    return PenaltyStrengths(
        validity=strength,
        full_liquidation=strength,
        min_participation=strength if needs_min_participation else 0.0,
        cardinality=strength,
        cvar=strength,
        safety_factor=safety_factor,
    )
```

Move the `BinaryEncoding` import in `slack_bits` up to the module's import block — the global constraints forbid a mid-file import, and M2b's plan corrections caught the same mistake twice.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_penalty.py -q`
Expected: every test in the file passes.

- [ ] **Step 5: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 6: Commit**

```bash
git add src/quantic/encoding/penalty.py tests/encoding/test_penalty.py
git commit -m "feat(encoding): quadratise the hard constraints and derive their strengths"
```

---

## Task 8: The CVaR gadget — a declared substitution with a reported resolution

**Files:**
- Create: `src/quantic/encoding/cvar.py`
- Modify: `src/quantic/encoding/expand.py` (accept a gadget instead of raising)
- Test: `tests/encoding/test_cvar.py`

**Interfaces:**
- Consumes: `Layout` (Task 4); `LinearForm`, `Qubo`, `QuboBuilder` (Task 2); `CVaRRisk`, `ScenarioSet`, `cvar_of`, `optimal_zeta` (`problem/`).
- Produces:
  - `Grid(lo, step, n_bits, keys)` with `hi`, `resolution`, `form()`, `snap(v)`, `encode(v)`, `decode(bits)`.
  - `CVaRGadget(zeta, u, g, alpha, lam, weights, penalty, loss_coefficients)` with `keys`, `resolution`, `losses_from(instance, layout, schedule)`, `optimal_aux(losses, zeta)`, `encoded_cvar(losses, zeta, aux)`.
  - `build_cvar_gadget(instance, layout, *, n_zeta_bits=8, penalty) -> CVaRGadget`.
  - `expand_cvar(instance, layout, registry, gadget) -> tuple[Qubo, Qubo]` — `(cost, penalty)`.
  - `expand_objective(..., gadget=None)` now encodes `CVaRRisk`.

**Why this is a substitution and not an approximation bug.** `max(L_s − ζ, 0)` needs `u_s ≥ L_s − ζ` with `u_s` minimised; staying quadratic needs a slack, so the gadget imposes `u_s − g_s = L_s − ζ`. `L_s` is linear in `x` with **real** coefficients while `ζ`, `u_s`, `g_s` live on finite binary grids, so that equality is unsatisfiable exactly. M2b §3 already named this for `ζ` itself and accepted it as the encoding's discretisation cost. The gadget therefore states precisely what it encodes, and the gate makes two exact assertions against it rather than one loose one.

**Two auxiliaries per scenario, not one.** Parent spec §5.4 says one. A one-sided constraint cannot stay quadratic without a slack. Recorded as a deviation in the design doc §11, and it is the reason §13's variable budget matters.

**Shared step.** `u` and `g` use `ζ`'s step. Then `u − g` ranges over integer multiples of one step, so the best residual is at most `step/2` and `optimal_aux` is closed-form rather than a search: when the target is non-negative take it in `u` and leave `g` at zero, otherwise the mirror.

**The penalty floor is real and is reported.** At the optimum the residual is not zero, so the QUBO's energy floor sits `penalty · S · (step/2)²` above the objective. It shrinks quadratically in the step, so `n_zeta_bits` buys accuracy at a linear cost in variables. `resolution` makes that trade visible instead of leaving it to be discovered on hardware.

- [ ] **Step 1: Write the failing test**

Create `tests/encoding/test_cvar.py`:

```python
import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.cvar import build_cvar_gadget, expand_cvar
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.encoding.layout import encode_schedule, plan_layout
from quantic.encoding.registry import VarRegistry
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import cvar_of, optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
REL = 1e-9


def _scenarios(seed=0, n=4):
    rng = np.random.default_rng(seed)
    returns = rng.normal(scale=0.01, size=(n, 2, 2))
    return ScenarioSet(returns=returns, weights=np.full(n, 1.0 / n), seed=seed)


def _instance(*, lam=1e-6, alpha=0.95, scenarios=None) -> Instance:
    assets = tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B"))
    params = MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
            for j, s in enumerate(("A", "B"))
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )
    return Instance(
        instance_id="x", tier="T0", assets=assets, n_buckets=2,
        initial_lots=(4, 4), max_lots_per_bucket=4, params=params,
        dials=Dials(cvar_risk=True), constraints=(FullLiquidation(),),
        risk=CVaRRisk(lam=lam, alpha=alpha, scenarios=scenarios or _scenarios()),
        witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )


def _setup(instance, encoding, *, n_zeta_bits=8, penalty=1.0):
    layout = plan_layout(instance, encoding)
    gadget = build_cvar_gadget(
        instance, layout, n_zeta_bits=n_zeta_bits, penalty=penalty
    )
    registry = VarRegistry(keys=layout.registry.keys + gadget.keys)
    cost, pen = expand_cvar(instance, layout, registry, gadget)
    return layout, registry, gadget, cost, pen


def _bits(layout, registry, gadget, schedule, zeta, aux):
    vector = encode_schedule(layout, schedule, registry)
    for key, value in gadget.zeta.encode(zeta).items():
        vector[registry.index(key)] = value
    for key, value in aux.items():
        vector[registry.index(key)] = value
    return vector


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_qubo_equals_the_function_the_gadget_declares_it_encodes(encoding):
    """Assertion one of two, and the one that proves the encoder correct."""
    inst = _instance()
    layout, registry, gadget, cost, pen = _setup(inst, encoding)
    schedule = inst.witness
    losses = gadget.losses_from(inst, layout, schedule)
    zeta = gadget.zeta.snap(optimal_zeta(inst, schedule))
    aux = gadget.optimal_aux(losses, zeta)
    bits = _bits(layout, registry, gadget, schedule, zeta, aux)
    assert (cost + pen).energy(bits) == pytest.approx(
        gadget.encoded_cvar(losses, zeta, aux), rel=REL
    )


def test_the_declared_function_stays_within_the_reported_resolution():
    """Assertion two of two: the discretisation is no worse than it claims."""
    inst = _instance()
    layout, _, gadget, _, _ = _setup(inst, BinaryEncoding(), n_zeta_bits=10)
    schedule = inst.witness
    losses = gadget.losses_from(inst, layout, schedule)
    zeta = gadget.zeta.snap(optimal_zeta(inst, schedule))
    aux = gadget.optimal_aux(losses, zeta)
    continuous = inst.risk.lam * cvar_of(
        inst.risk.scenarios, inst, schedule, alpha=inst.risk.alpha, zeta=zeta
    )
    assert abs(gadget.encoded_cvar(losses, zeta, aux) - continuous) <= gadget.resolution


def test_the_loss_form_reproduces_the_scenario_sets_own_losses():
    """L_s is linear in the holdings; the encoder must agree with problem/."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    gadget = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
    for schedule in (inst.witness, Schedule(lots=((4, 0), (0, 4)))):
        direct = inst.risk.scenarios.losses(inst, schedule)
        assert gadget.losses_from(inst, layout, schedule) == pytest.approx(
            direct, rel=1e-12
        )


def test_the_resolution_shrinks_quadratically_in_the_grid_step():
    """More zeta bits buy accuracy at a linear cost in variables."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    coarse = build_cvar_gadget(inst, layout, n_zeta_bits=6, penalty=1.0)
    fine = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
    assert fine.resolution < coarse.resolution
    assert fine.zeta.step < coarse.zeta.step


def test_the_true_optimal_zeta_lies_inside_the_derived_grid():
    """The bound is instance-derived, so it must hold for every schedule."""
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    gadget = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
    for schedule in (
        inst.witness,
        Schedule(lots=((4, 0), (0, 4))),
        Schedule(lots=((0, 4), (4, 0))),
        Schedule(lots=((0, 0), (0, 0))),
    ):
        z = optimal_zeta(inst, schedule)
        assert gadget.zeta.lo <= z <= gadget.zeta.hi


def test_optimal_aux_is_deterministic():
    inst = _instance()
    layout, _, gadget, _, _ = _setup(inst, BinaryEncoding())
    losses = gadget.losses_from(inst, layout, inst.witness)
    zeta = gadget.zeta.snap(optimal_zeta(inst, inst.witness))
    assert gadget.optimal_aux(losses, zeta) == gadget.optimal_aux(losses, zeta)


def test_the_gadget_costs_two_auxiliaries_per_scenario():
    """Parent spec 5.4 says one. A one-sided constraint cannot; deviation recorded."""
    inst = _instance(scenarios=_scenarios(n=3))
    layout = plan_layout(inst, BinaryEncoding())
    gadget = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
    assert len(gadget.u) == 3
    assert len(gadget.g) == 3
    assert sum(1 for k in gadget.keys if k[0] == "u") > 0
    assert sum(1 for k in gadget.keys if k[0] == "g") > 0


def test_a_zeta_grid_value_round_trips():
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    grid = build_cvar_gadget(inst, layout, n_zeta_bits=6, penalty=1.0).zeta
    for step in range(2**6):
        value = grid.lo + step * grid.step
        assert grid.decode(grid.encode(value)) == pytest.approx(value, rel=1e-12)


def test_snapping_never_leaves_the_grid():
    inst = _instance()
    layout = plan_layout(inst, BinaryEncoding())
    grid = build_cvar_gadget(inst, layout, n_zeta_bits=6, penalty=1.0).zeta
    assert grid.snap(grid.lo - 1e9) == pytest.approx(grid.lo)
    assert grid.snap(grid.hi + 1e9) == pytest.approx(grid.hi)


def test_expand_objective_no_longer_refuses_cvar():
    inst = _instance()
    from quantic.encoding.expand import expand_objective

    layout = plan_layout(inst, BinaryEncoding())
    gadget = build_cvar_gadget(inst, layout, n_zeta_bits=8, penalty=1.0)
    registry = VarRegistry(keys=layout.registry.keys + gadget.keys)
    breakdown = expand_objective(inst, layout, registry, gadget=gadget)
    assert breakdown.risk.n_vars == len(registry)


def test_expand_objective_still_refuses_cvar_without_a_gadget():
    inst = _instance()
    from quantic.encoding.expand import expand_objective

    layout = plan_layout(inst, BinaryEncoding())
    with pytest.raises(ValueError, match="gadget"):
        expand_objective(inst, layout, layout.registry)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_cvar.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.encoding.cvar'`.

- [ ] **Step 3: Write the gadget**

Create `src/quantic/encoding/cvar.py`:

```python
"""Rockafellar-Uryasev CVaR as a QUBO, and the discretisation it declares.

The risk term is

    lam * ( zeta + 1/(1 - alpha) * sum_s w_s * max(L_s(x) - zeta, 0) )

and ``max(u, 0)`` is what a QUBO cannot say directly. The standard device is
``u_s >= L_s - zeta`` with ``u_s`` minimised, which stays quadratic only with a
slack:

    cost      lam * ( zeta + 1/(1 - alpha) * sum_s w_s * u_s )
    penalty   P * sum_s ( u_s - L_s + zeta - g_s )**2

``L_s`` is linear in the holdings -- scenarios perturb prices only, which is
M2b's design decision -- but with real coefficients, while ``zeta``, ``u`` and
``g`` live on finite binary grids. So that equality cannot hold exactly, for
the same reason M2b accepted that a binary-expanded ``zeta`` cannot land on the
empirical quantile.

This module therefore states exactly what it encodes. The gate proves the QUBO
equals :meth:`CVaRGadget.encoded_cvar` exactly, and separately proves that
``encoded_cvar`` at the penalty-optimal auxiliaries sits within
:attr:`CVaRGadget.resolution` of continuous Rockafellar-Uryasev. Two exact
assertions; no tolerance on the correctness claim.

``u`` and ``g`` share ``zeta``'s step, so ``u - g`` ranges over integer
multiples of one step and the best auxiliaries are closed-form rather than a
search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.encoding.layout import Layout
from quantic.encoding.qubo import LinearForm, Qubo, QuboBuilder
from quantic.encoding.registry import VarKey, VarRegistry
from quantic.problem.instance import Instance
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, slots=True)
class Grid:
    """A uniform binary grid: ``value = lo + step * sum 2**b z_b``.

    Plain powers of two rather than the bounded coefficients the lot cells use.
    Every one of the ``2**n_bits`` states is a meaningful grid point here, so
    there is no forbidden region to suppress and the map is injective, which is
    what lets :meth:`encode` and :meth:`decode` round-trip exactly.
    """

    lo: float
    step: float
    n_bits: int
    keys: tuple[VarKey, ...]

    def __post_init__(self) -> None:
        if self.n_bits < 1:
            raise ValueError(f"a grid needs at least one bit, got {self.n_bits}")
        if self.step <= 0 or not np.isfinite(self.step):
            raise ValueError(f"step must be finite and positive, got {self.step}")
        if len(self.keys) != self.n_bits:
            raise ValueError(
                f"{len(self.keys)} keys for {self.n_bits} bits"
            )

    @property
    def hi(self) -> float:
        return self.lo + self.step * float(2**self.n_bits - 1)

    @property
    def resolution(self) -> float:
        """Worst-case distance from an arbitrary real to the nearest grid point."""
        return self.step / 2.0

    def form(self) -> LinearForm:
        out = LinearForm.constant_form(self.lo)
        for b, key in enumerate(self.keys):
            out = out + LinearForm.of(key, self.step * float(1 << b))
        return out

    def _level(self, value: float) -> int:
        raw = round((float(value) - self.lo) / self.step)
        return int(min(max(raw, 0), 2**self.n_bits - 1))

    def snap(self, value: float) -> float:
        return self.lo + self.step * float(self._level(value))

    def encode(self, value: float) -> dict[VarKey, int]:
        level = self._level(value)
        return {key: (level >> b) & 1 for b, key in enumerate(self.keys)}

    def decode(self, bits) -> float:
        level = sum((1 << b) for b, key in enumerate(self.keys) if bits[key])
        return self.lo + self.step * float(level)


def _grid_for(lo: float, step: float, span: float, prefix: VarKey) -> Grid:
    """The smallest grid at ``step`` starting at ``lo`` that reaches ``lo + span``."""
    levels = max(int(np.ceil(max(span, 0.0) / step)), 1)
    n_bits = max(int(levels).bit_length(), 1)
    return Grid(
        lo=lo, step=step, n_bits=n_bits,
        keys=tuple((*prefix, b) for b in range(n_bits)),
    )


@dataclass(frozen=True, eq=False)
class CVaRGadget:
    """The Rockafellar-Uryasev expansion, and the function it actually implements."""

    zeta: Grid
    u: tuple[Grid, ...]
    g: tuple[Grid, ...]
    alpha: float
    lam: float
    weights: np.ndarray
    penalty: float
    loss_coefficients: np.ndarray  # (S, N, T) coefficient on holdings in lots

    @property
    def n_scenarios(self) -> int:
        return len(self.u)

    @property
    def keys(self) -> tuple[VarKey, ...]:
        out: list[VarKey] = list(self.zeta.keys)
        for grid in self.u:
            out.extend(grid.keys)
        for grid in self.g:
            out.extend(grid.keys)
        return tuple(out)

    @property
    def resolution(self) -> float:
        """How far ``encoded_cvar`` can sit from continuous Rockafellar-Uryasev.

        Two sources, both at one grid step. Each ``u_s`` misses
        ``max(L_s - zeta, 0)`` by at most half a step, which the cost weight
        carries; and the residual the penalty charges is at most half a step,
        squared. The second term is why a coarse grid is expensive: it shrinks
        quadratically, so buying accuracy costs only bits.
        """
        half = self.zeta.step / 2.0
        cost_gap = self.lam * half / (1.0 - self.alpha)
        floor = self.penalty * float(self.n_scenarios) * half * half
        return float(cost_gap + floor)

    def losses_from(
        self, instance: Instance, layout: Layout, schedule: Schedule
    ) -> np.ndarray:
        """``L_s`` for every scenario, from the same coefficients the QUBO uses."""
        holdings = instance.remaining_lots(schedule).astype(float)  # (N, T)
        return np.einsum("snt,nt->s", self.loss_coefficients, holdings)

    def optimal_aux(self, losses: np.ndarray, zeta: float) -> dict[VarKey, int]:
        """The penalty-minimising auxiliaries, in closed form.

        ``u`` and ``g`` share a step, so the residual is minimised by putting
        the whole target into whichever of the two can hold it and leaving the
        other at zero. Ties never arise: the target's sign decides.
        """
        bits: dict[VarKey, int] = {}
        for s, target in enumerate(np.asarray(losses, dtype=float) - float(zeta)):
            if target >= 0.0:
                bits.update(self.u[s].encode(target))
                bits.update(self.g[s].encode(0.0))
            else:
                bits.update(self.u[s].encode(0.0))
                bits.update(self.g[s].encode(-target))
        return bits

    def encoded_cvar(self, losses: np.ndarray, zeta: float, aux) -> float:
        """The exact algebraic value the QUBO implements, penalty included."""
        losses = np.asarray(losses, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        excess = 0.0
        residual = 0.0
        for s in range(self.n_scenarios):
            u = self.u[s].decode(aux)
            g = self.g[s].decode(aux)
            excess += weights[s] * u
            gap = u - float(losses[s]) + float(zeta) - g
            residual += gap * gap
        cost = self.lam * (float(zeta) + excess / (1.0 - self.alpha))
        return float(cost + self.penalty * residual)


def build_cvar_gadget(
    instance: Instance, layout: Layout, *, n_zeta_bits: int = 8, penalty: float
) -> CVaRGadget:
    """Derive every grid from the instance alone -- never from a schedule.

    ``h[i,t]`` lies in ``[0, X_i]``, so bounding ``L_s`` cell by cell bounds it
    over every schedule at once, and the true optimal ``zeta`` -- a quantile of
    the loss distribution -- always lands inside.
    """
    risk = instance.risk
    if not isinstance(risk, CVaRRisk):
        raise TypeError(
            f"build_cvar_gadget needs a CVaRRisk, got {type(risk).__name__}"
        )
    if n_zeta_bits < 1:
        raise ValueError(f"n_zeta_bits must be at least 1, got {n_zeta_bits}")

    returns = np.asarray(risk.scenarios.returns, dtype=float)      # (S, T, N)
    per_lot = instance.lot_sizes().astype(float) * instance.params.prices()
    # L_s = -sum_{t,i} h[i,t] * lot_i * price_i * returns[s,t,i]
    coefficients = np.einsum("stn,n->snt", -returns, per_lot)      # (S, N, T)
    position = np.asarray(instance.initial_lots, dtype=float)[None, :, None]
    upper = float((np.maximum(coefficients, 0.0) * position).sum(axis=(1, 2)).max())
    lower = float((np.minimum(coefficients, 0.0) * position).sum(axis=(1, 2)).min())

    span = upper - lower
    if span <= 0.0 or not np.isfinite(span):
        # Degenerate: every scenario is flat, so every loss is zero.
        span = 1.0
        lower, upper = 0.0, 1.0
    step = span / float(2**n_zeta_bits - 1) if n_zeta_bits > 1 else span

    zeta = Grid(
        lo=lower, step=step, n_bits=n_zeta_bits,
        keys=tuple(("zeta", b) for b in range(n_zeta_bits)),
    )
    n_scenarios = int(returns.shape[0])
    u = tuple(
        _grid_for(0.0, step, upper - lower, ("u", s)) for s in range(n_scenarios)
    )
    g = tuple(
        _grid_for(0.0, step, upper - lower, ("g", s)) for s in range(n_scenarios)
    )
    return CVaRGadget(
        zeta=zeta, u=u, g=g, alpha=risk.alpha, lam=risk.lam,
        weights=np.asarray(risk.scenarios.weights, dtype=float),
        penalty=float(penalty), loss_coefficients=coefficients,
    )


def _holdings_lots_form(instance: Instance, layout: Layout, i: int, t: int) -> LinearForm:
    form = LinearForm.constant_form(float(instance.initial_lots[i]))
    for s in range(t + 1):
        form = form + layout.cell(i, s).value_form.scaled(-1.0)
    return form


def expand_cvar(
    instance: Instance, layout: Layout, registry: VarRegistry, gadget: CVaRGadget
) -> tuple[Qubo, Qubo]:
    """``(cost, penalty)`` -- kept apart so the gate can see which one broke."""
    cost = QuboBuilder(registry)
    pen = QuboBuilder(registry)
    zeta_form = gadget.zeta.form()

    cost.add_form(zeta_form, gadget.lam)
    weights = np.asarray(gadget.weights, dtype=float)
    scale = gadget.lam / (1.0 - gadget.alpha)

    holdings = [
        [_holdings_lots_form(instance, layout, i, t) for t in range(layout.n_buckets)]
        for i in range(layout.n_assets)
    ]

    for s in range(gadget.n_scenarios):
        u_form = gadget.u[s].form()
        g_form = gadget.g[s].form()
        cost.add_form(u_form, scale * float(weights[s]))

        residual = u_form + zeta_form + g_form.scaled(-1.0)
        for i in range(layout.n_assets):
            for t in range(layout.n_buckets):
                coef = float(gadget.loss_coefficients[s, i, t])
                if coef != 0.0:
                    residual = residual + holdings[i][t].scaled(-coef)
        pen.add_square(residual, gadget.penalty)

    return cost.build(), pen.build()
```

- [ ] **Step 4: Teach `expand_objective` about the gadget**

In `src/quantic/encoding/expand.py`, add `from quantic.encoding.cvar import CVaRGadget, expand_cvar` to the import block, change the signature to

```python
def expand_objective(
    instance: Instance,
    layout: Layout,
    registry: VarRegistry,
    *,
    gadget: CVaRGadget | None = None,
) -> QuboBreakdown:
```

and replace the `CVaRRisk` branch's `raise NotImplementedError(...)` with

```python
    elif isinstance(risk, CVaRRisk):
        if gadget is None:
            raise ValueError(
                "CVaRRisk needs a gadget: build_cvar_gadget derives the zeta, u and g "
                "grids from the instance, and the penalty strength comes from "
                "derive_penalties. expand_objective will not invent either"
            )
        cost, cvar_penalty = expand_cvar(instance, layout, registry, gadget)
        risk_qubo = cost + cvar_penalty
```

The CVaR penalty joins the risk term rather than the penalty block because it is not a constraint on the schedule — it is how the risk term is expressed at all, and a schedule cannot satisfy or violate it.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/encoding/test_cvar.py -q
.venv/Scripts/python.exe -m pytest tests/encoding/test_expand.py -q
```

Expected: every test in both files passes. `test_cvar_risk_defers_to_task_eight` in `test_expand.py` now fails by design — **delete it**; `test_expand_objective_still_refuses_cvar_without_a_gadget` in `test_cvar.py` is its replacement.

- [ ] **Step 6: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 7: Commit**

```bash
git add src/quantic/encoding/cvar.py src/quantic/encoding/expand.py tests/encoding/
git commit -m "feat(encoding): encode CVaR as a declared gadget with a reported resolution"
```

---

## Task 9: `encode` — the orchestrator

**Files:**
- Create: `src/quantic/encoding/encode.py`
- Test: `tests/encoding/test_encode.py`

**Interfaces:**
- Consumes: everything from Tasks 4–8.
- Produces:
  - `EncodedInstance(instance, layout, registry, breakdown, penalties, strengths, gadget)` with `qubo -> Qubo`, `n_vars -> int`, `describe() -> dict`.
  - `encode(instance, encoding, *, n_zeta_bits=8, safety_factor=2.0) -> EncodedInstance`.
  - `encode_solution(encoded, schedule, *, zeta=None) -> np.ndarray`.
  - `reference_breakdown(encoded, schedule, *, zeta=None) -> ObjectiveBreakdown` — the expected value the gate compares against, with the D1 field swapped when an approximand is present.

**The two-pass derivation, and why it is not circular.** The CVaR gadget needs a penalty strength; `derive_penalties` needs the objective QUBO; and with D3 on, the objective QUBO contains the gadget. The loop is cut by noting that the gadget's *keys* depend only on the instance and `n_zeta_bits`, never on the strength, and that only the gadget's penalty block depends on the strength at all. So: build the gadget with a strength of zero, expand, derive from that, then rebuild the gadget at the derived strength and expand once more. Deterministic, and both passes are pure.

**Why `reference_breakdown` lives here.** It is the one place that knows whether an approximand was emitted. Putting it beside `encode` keeps the gate from re-deriving that, and keeps `problem/objective.py` the only evaluator — the swap is a `dataclasses.replace` on its output, never a second implementation.

- [ ] **Step 1: Write the failing test**

Create `tests/encoding/test_encode.py`:

```python
import json

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.encode import encode, encode_solution, reference_breakdown
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import cvar_of, optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
REL = 1e-9


def _params():
    return MarketParams(
        assets=tuple(
            AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
            for j, s in enumerate(("A", "B"))
        ),
        covariance=np.eye(2) * 1e-5,
        bucket_ns=BUCKET_NS,
    )


def _instance(**kw) -> Instance:
    base = dict(
        instance_id="x", tier="T0",
        assets=tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B")),
        n_buckets=2, initial_lots=(4, 4), max_lots_per_bucket=4, params=_params(),
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def _cvar_instance():
    rng = np.random.default_rng(7)
    scenarios = ScenarioSet(
        returns=rng.normal(scale=0.01, size=(4, 2, 2)),
        weights=np.full(4, 0.25), seed=7,
    )
    return _instance(
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=1e-6, alpha=0.95, scenarios=scenarios),
    )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_energy_at_the_witness_equals_the_reference_objective(encoding):
    inst = _instance()
    enc = encode(inst, encoding)
    bits = encode_solution(enc, inst.witness)
    assert enc.qubo.energy(bits) == pytest.approx(
        reference_breakdown(enc, inst.witness).total, rel=REL
    )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_penalty_block_is_exactly_zero_at_the_witness(encoding):
    inst = _instance()
    enc = encode(inst, encoding)
    assert enc.penalties.energy(encode_solution(enc, inst.witness)) == 0.0


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_cvar_instance_encodes_and_scores_at_a_snapped_zeta(encoding):
    inst = _cvar_instance()
    enc = encode(inst, encoding)
    zeta = enc.gadget.zeta.snap(optimal_zeta(inst, inst.witness))
    bits = encode_solution(enc, inst.witness, zeta=zeta)
    losses = enc.gadget.losses_from(inst, enc.layout, inst.witness)
    aux = enc.gadget.optimal_aux(losses, zeta)
    continuous = inst.risk.lam * cvar_of(
        inst.risk.scenarios, inst, inst.witness, alpha=inst.risk.alpha, zeta=zeta
    )
    expected = (
        reference_breakdown(enc, inst.witness, zeta=zeta).total
        - continuous
        + enc.gadget.encoded_cvar(losses, zeta, aux)
    )
    assert enc.qubo.energy(bits) == pytest.approx(expected, rel=REL)


def test_encoding_a_cvar_instance_without_a_zeta_is_refused():
    enc = encode(_cvar_instance(), BinaryEncoding())
    with pytest.raises(ValueError, match="zeta"):
        encode_solution(enc, enc.instance.witness)


def test_passing_a_zeta_to_a_variance_instance_is_refused():
    enc = encode(_instance(), BinaryEncoding())
    with pytest.raises(ValueError, match="no zeta"):
        encode_solution(enc, enc.instance.witness, zeta=0.0)


def test_the_derived_strength_does_not_depend_on_its_own_penalty_block():
    """The two-pass derivation must not feed the CVaR penalty back into the bound."""
    inst = _cvar_instance()
    first = encode(inst, BinaryEncoding())
    second = encode(inst, BinaryEncoding())
    assert first.strengths == second.strengths
    assert first.gadget.penalty == first.strengths.cvar


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_encoding_is_deterministic(encoding):
    inst = _instance()
    a, b = encode(inst, encoding), encode(inst, encoding)
    assert np.array_equal(a.qubo.Q, b.qubo.Q)
    assert a.qubo.constant == b.qubo.constant
    assert a.registry.keys == b.registry.keys


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_registry_covers_every_bit_any_stage_named(encoding):
    enc = encode(_cvar_instance(), encoding)
    named = {k for plan in enc.layout.cells for k in plan.bit_keys}
    named |= set(enc.gadget.keys)
    assert named <= set(enc.registry.keys)
    assert enc.qubo.n_vars == len(enc.registry)


def test_the_description_is_json_safe_and_names_the_encoding():
    enc = encode(_instance(), BinaryEncoding())
    payload = json.loads(json.dumps(enc.describe()))
    assert payload["encoding"] == "binary"
    assert payload["n_vars"] == enc.n_vars
    assert payload["penalties"]["strategy"] == "derived_bounds"


def test_the_reference_breakdown_swaps_the_impact_term_when_approximated():
    import dataclasses

    from quantic.problem.objective import evaluate

    inst = _instance(dials=Dials(concave_impact=True))
    enc = encode(inst, BinaryEncoding())
    assert enc.breakdown.approximand is not None
    expected = dataclasses.replace(
        evaluate(inst, inst.witness),
        temporary_impact_cost=enc.breakdown.approximand.temporary_cost(inst.witness),
    )
    assert reference_breakdown(enc, inst.witness) == expected


def test_the_reference_breakdown_is_untouched_when_nothing_was_approximated():
    from quantic.problem.objective import evaluate

    inst = _instance()
    enc = encode(inst, BinaryEncoding())
    assert enc.breakdown.approximand is None
    assert reference_breakdown(enc, inst.witness) == evaluate(inst, inst.witness)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_encode.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.encoding.encode'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/encoding/encode.py`:

```python
"""One instance, one encoding, one QUBO -- and the reference it must equal.

Every stage below names its own bits: the layout names the lot cells, the
penalty module names cardinality slack, the CVaR gadget names its three grids.
They must all land in **one** registry, because column ``k`` has to mean the
same bit in the objective, the penalties and the gadget alike. This module
assembles that registry and then rebuilds every piece against it.

The strength derivation needs two passes. ``derive_penalties`` reads the
objective QUBO, and with dial D3 on the objective QUBO contains the CVaR
penalty, whose magnitude is what is being derived. The loop is cut by the
observation that the gadget's *keys* depend only on the instance and the bit
budget, never on the strength: build the gadget at strength zero, derive from
that objective, then rebuild at the derived strength. Both passes are pure, so
the result is deterministic.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantic.encoding.cvar import CVaRGadget, build_cvar_gadget
from quantic.encoding.expand import QuboBreakdown, expand_objective
from quantic.encoding.integer import IntegerEncoding
from quantic.encoding.layout import Layout, encode_schedule, plan_layout
from quantic.encoding.penalty import (
    PenaltyStrengths,
    derive_penalties,
    expand_penalties,
    penalty_aux_keys,
    slack_bits,
)
from quantic.encoding.qubo import Qubo
from quantic.encoding.registry import VarRegistry
from quantic.problem.instance import Instance
from quantic.problem.objective import ObjectiveBreakdown, evaluate
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule


@dataclass(frozen=True, eq=False)
class EncodedInstance:
    """Everything the gate, the diagnostics and M3b's solvers need."""

    instance: Instance
    layout: Layout
    registry: VarRegistry
    breakdown: QuboBreakdown
    penalties: Qubo
    strengths: PenaltyStrengths
    gadget: CVaRGadget | None

    @property
    def qubo(self) -> Qubo:
        return self.breakdown.total + self.penalties

    @property
    def n_vars(self) -> int:
        return len(self.registry)

    def describe(self) -> dict[str, Any]:
        """JSON-safe, for the results row."""
        approximand = self.breakdown.approximand
        return {
            "encoding": self.layout.encoding_name,
            "n_vars": self.n_vars,
            "penalties": self.strengths.describe(),
            "approximand": None if approximand is None else approximand.describe(),
            "cvar_resolution": None if self.gadget is None else self.gadget.resolution,
            "cvar_zeta_bits": None if self.gadget is None else self.gadget.zeta.n_bits,
        }


def _assemble(instance: Instance, layout: Layout, gadget: CVaRGadget | None) -> VarRegistry:
    keys = layout.registry.keys + penalty_aux_keys(instance, layout)
    if gadget is not None:
        keys = keys + gadget.keys
    return VarRegistry(keys=keys)


def encode(
    instance: Instance,
    encoding: IntegerEncoding,
    *,
    n_zeta_bits: int = 8,
    safety_factor: float = 2.0,
) -> EncodedInstance:
    """Turn ``instance`` into a QUBO under ``encoding``."""
    layout = plan_layout(instance, encoding)
    needs_gadget = isinstance(instance.risk, CVaRRisk)

    provisional = (
        build_cvar_gadget(instance, layout, n_zeta_bits=n_zeta_bits, penalty=0.0)
        if needs_gadget
        else None
    )
    registry = _assemble(instance, layout, provisional)
    probe = expand_objective(instance, layout, registry, gadget=provisional)
    strengths = derive_penalties(probe.total, layout, safety_factor=safety_factor)

    gadget = (
        build_cvar_gadget(
            instance, layout, n_zeta_bits=n_zeta_bits, penalty=strengths.cvar
        )
        if needs_gadget
        else None
    )
    breakdown = expand_objective(instance, layout, registry, gadget=gadget)
    penalties = expand_penalties(instance, layout, registry, strengths)
    return EncodedInstance(
        instance=instance, layout=layout, registry=registry, breakdown=breakdown,
        penalties=penalties, strengths=strengths, gadget=gadget,
    )


def encode_solution(
    encoded: EncodedInstance, schedule: Schedule, *, zeta: float | None = None
) -> np.ndarray:
    """The canonical bit vector for ``schedule``, auxiliaries included."""
    if encoded.gadget is None and zeta is not None:
        raise ValueError(
            "zeta was supplied but this instance carries no CVaR gadget, so it has "
            "no zeta to encode"
        )
    if encoded.gadget is not None and zeta is None:
        raise ValueError(
            "this instance needs an explicit zeta. Rockafellar-Uryasev treats it as a "
            "decision variable, and the encoding pins it to a grid, so the call site "
            "must say which zeta it is encoding. Snap "
            "problem.objective.optimal_zeta onto encoded.gadget.zeta"
        )

    vector = encode_schedule(encoded.layout, schedule, encoded.registry)
    for key, value in slack_bits(encoded.instance, encoded.layout, schedule).items():
        vector[encoded.registry.index(key)] = value

    if encoded.gadget is not None:
        snapped = encoded.gadget.zeta.snap(float(zeta))
        for key, value in encoded.gadget.zeta.encode(snapped).items():
            vector[encoded.registry.index(key)] = value
        losses = encoded.gadget.losses_from(
            encoded.instance, encoded.layout, schedule
        )
        for key, value in encoded.gadget.optimal_aux(losses, snapped).items():
            vector[encoded.registry.index(key)] = value
    return vector


def reference_breakdown(
    encoded: EncodedInstance, schedule: Schedule, *, zeta: float | None = None
) -> ObjectiveBreakdown:
    """What the QUBO's objective block is expected to equal.

    ``problem.objective.evaluate`` stays the only evaluator. When an
    approximand was emitted, one field of its output is replaced -- never
    recomputed by a second implementation, which is how the two would drift.
    """
    base = evaluate(encoded.instance, schedule, zeta=zeta)
    approximand = encoded.breakdown.approximand
    if approximand is None:
        return base
    return dataclasses.replace(
        base, temporary_impact_cost=approximand.temporary_cost(schedule)
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_encode.py -q`
Expected: every test in the file passes.

- [ ] **Step 5: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 6: Commit**

```bash
git add src/quantic/encoding/encode.py tests/encoding/test_encode.py
git commit -m "feat(encoding): assemble one registry and one QUBO per instance"
```

---

## Task 10: `decode` and the feasibility classification

**Files:**
- Create: `src/quantic/encoding/decode.py`
- Test: `tests/encoding/test_decode.py`

**Interfaces:**
- Consumes: `EncodedInstance` (Task 9); `classify`, `FeasibilityReport` (`problem/`).
- Produces: `DecodedSample(schedule, zeta, classification, report, reason)` and `decode(encoded, bits) -> DecodedSample`.

**The three classes, and where the line falls.** §6.5 asks for **feasible**, **repairable** and **rejected**, and the useful split is structural rather than a matter of degree:

- **rejected** — the bits do not describe a schedule at all. Two one-hot bits set in a cell; a broken domain wall; residual bits set under a clear indicator. There is no schedule to score, so there is nothing to repair.
- **repairable** — the bits decode cleanly but the schedule breaches a constraint. A repair strategy could act on it, and M4 will supply one.
- **feasible** — `classify` reports no violation.

**No repair here.** §6.5 requires repair to be a named strategy recorded on the results row, and repair only means anything against samples a solver returned. M4 owns it. Classification is pure logic and ships now, because M3b's gate needs to know that the minimising bitstring decodes to a feasible schedule.

**Scoring an infeasible sample against a feasible one is the error §6.5 names** as invalidating most published work in this area. This function is what makes that impossible to do by accident.

- [ ] **Step 1: Write the failing test**

Create `tests/encoding/test_decode.py`:

```python
import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.decode import decode
from quantic.encoding.encode import encode, encode_solution
from quantic.encoding.integer import (
    ENCODINGS,
    BinaryEncoding,
    DomainWallEncoding,
    OneHotEncoding,
)
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.objective import optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(**kw) -> Instance:
    base = dict(
        instance_id="x", tier="T0",
        assets=tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B")),
        n_buckets=2, initial_lots=(4, 4), max_lots_per_bucket=4,
        params=MarketParams(
            assets=tuple(
                AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
                for j, s in enumerate(("A", "B"))
            ),
            covariance=np.eye(2) * 1e-5, bucket_ns=BUCKET_NS,
        ),
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_feasible_schedule_round_trips(encoding):
    inst = _instance()
    enc = encode(inst, encoding)
    sample = decode(enc, encode_solution(enc, inst.witness))
    assert sample.schedule == inst.witness
    assert sample.classification == "feasible"
    assert sample.report.feasible


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_every_representable_schedule_round_trips(encoding):
    inst = _instance()
    enc = encode(inst, encoding)
    for a0 in range(5):
        for b0 in range(5):
            schedule = Schedule(lots=((a0, 4 - a0), (b0, 4 - b0)))
            assert decode(enc, encode_solution(enc, schedule)).schedule == schedule


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_a_decodable_but_infeasible_schedule_is_repairable(encoding):
    inst = _instance()
    enc = encode(inst, encoding)
    short = Schedule(lots=((2, 1), (2, 2)))
    sample = decode(enc, encode_solution(enc, short))
    assert sample.schedule == short
    assert sample.classification == "repairable"
    assert sample.report.total_violation > 0.0


def test_an_empty_one_hot_cell_is_rejected_rather_than_repaired():
    """No schedule to score, so there is nothing to repair."""
    enc = encode(_instance(), OneHotEncoding())
    sample = decode(enc, np.zeros(enc.n_vars, dtype=np.int64))
    assert sample.classification == "rejected"
    assert sample.schedule is None
    assert "one" in sample.reason


def test_a_doubly_set_one_hot_cell_is_rejected():
    enc = encode(_instance(), OneHotEncoding())
    bits = encode_solution(enc, enc.instance.witness)
    first = next(i for i, k in enumerate(enc.registry.keys) if k[0] == "x")
    bits[first] = 1
    bits[first + 1] = 1
    assert decode(enc, bits).classification == "rejected"


def test_a_broken_domain_wall_is_rejected():
    enc = encode(_instance(), DomainWallEncoding())
    bits = np.zeros(enc.n_vars, dtype=np.int64)
    walls = [i for i, k in enumerate(enc.registry.keys) if k[0] == "x"]
    bits[walls[1]] = 1  # a hole: w2 set while w1 is clear
    sample = decode(enc, bits)
    assert sample.classification == "rejected"
    assert "monotone" in sample.reason


def test_a_residual_without_its_indicator_is_rejected():
    from quantic.problem.constraints.cardinality import Cardinality
    from quantic.problem.constraints.min_participation import MinParticipation

    inst = _instance(
        dials=Dials(discrete_participation=True),
        constraints=(FullLiquidation(), Cardinality(k=1), MinParticipation((3, 3))),
        witness=Schedule(lots=((4, 0), (0, 4))),
    )
    enc = encode(inst, BinaryEncoding())
    bits = np.zeros(enc.n_vars, dtype=np.int64)
    residual = next(i for i, k in enumerate(enc.registry.keys) if k[0] == "x")
    bits[residual] = 1
    assert decode(enc, bits).classification == "rejected"


def test_a_cvar_sample_reports_the_zeta_it_decoded():
    rng = np.random.default_rng(3)
    scenarios = ScenarioSet(
        returns=rng.normal(scale=0.01, size=(4, 2, 2)),
        weights=np.full(4, 0.25), seed=3,
    )
    inst = _instance(
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=1e-6, alpha=0.95, scenarios=scenarios),
    )
    enc = encode(inst, BinaryEncoding())
    zeta = enc.gadget.zeta.snap(optimal_zeta(inst, inst.witness))
    sample = decode(enc, encode_solution(enc, inst.witness, zeta=zeta))
    assert sample.zeta == pytest.approx(zeta, rel=1e-12)
    assert sample.classification == "feasible"


def test_a_variance_sample_reports_no_zeta():
    enc = encode(_instance(), BinaryEncoding())
    assert decode(enc, encode_solution(enc, enc.instance.witness)).zeta is None


def test_decoding_a_wrong_length_vector_is_refused():
    enc = encode(_instance(), BinaryEncoding())
    with pytest.raises(ValueError, match="variables"):
        decode(enc, np.zeros(enc.n_vars - 1, dtype=np.int64))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_decode.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.encoding.decode'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/encoding/decode.py`:

```python
"""Bits back to a schedule, and the verdict spec section 6.5 demands.

Section 6.5 names scoring an infeasible quantum solution against a feasible
classical one as the error that invalidates most published comparisons in this
area. Every sample therefore arrives carrying its classification, and the line
between the three classes is structural rather than a matter of degree:

* **rejected** -- the bits do not describe a schedule at all. Two one-hot bits
  in a cell, a hole in a domain wall, residual bits under a clear indicator.
  There is nothing to score and nothing to repair.
* **repairable** -- the bits decode cleanly but the schedule breaches a
  constraint. M4's named repair strategies act here.
* **feasible** -- ``problem.feasibility.classify`` reports no violation.

Repair itself is not here. Section 6.5 requires it to be a named strategy
recorded on the results row, and it only means anything against samples a
solver returned, which is M4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantic.encoding.encode import EncodedInstance
from quantic.problem.feasibility import FeasibilityReport, classify
from quantic.problem.schedule import Schedule

FEASIBLE = "feasible"
REPAIRABLE = "repairable"
REJECTED = "rejected"


@dataclass(frozen=True)
class DecodedSample:
    """One returned bitstring, read back and judged."""

    schedule: Schedule | None
    zeta: float | None
    classification: str
    report: FeasibilityReport | None
    reason: str | None = None


def decode(encoded: EncodedInstance, bits) -> DecodedSample:
    """Read ``bits`` back into a schedule and classify it."""
    vector = np.asarray(bits)
    if vector.shape != (encoded.n_vars,):
        raise ValueError(
            f"decode needs {encoded.n_vars} variables, got a vector of shape "
            f"{vector.shape}"
        )
    by_key = {key: int(vector[i]) for i, key in enumerate(encoded.registry.keys)}

    grid = [[0] * encoded.layout.n_buckets for _ in range(encoded.layout.n_assets)]
    for plan in encoded.layout.cells:
        try:
            grid[plan.asset][plan.bucket] = plan.decode(by_key)
        except ValueError as exc:
            return DecodedSample(
                schedule=None, zeta=None, classification=REJECTED,
                report=None, reason=str(exc),
            )

    schedule = Schedule(lots=tuple(tuple(row) for row in grid))
    zeta = None
    if encoded.gadget is not None:
        zeta = encoded.gadget.zeta.decode(by_key)

    report = classify(encoded.instance, schedule)
    return DecodedSample(
        schedule=schedule,
        zeta=zeta,
        classification=FEASIBLE if report.feasible else REPAIRABLE,
        report=report,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_decode.py -q`
Expected: every test in the file passes.

- [ ] **Step 5: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 6: Commit**

```bash
git add src/quantic/encoding/decode.py tests/encoding/test_decode.py
git commit -m "feat(encoding): decode samples and classify their feasibility"
```

---

## Task 11: Diagnostics — dynamic range and the variable budget

**Files:**
- Create: `src/quantic/encoding/diagnostics.py`
- Test: `tests/encoding/test_diagnostics.py`

**Interfaces:**
- Consumes: `Qubo` (Task 2); `plan_layout` (Task 4); `penalty_aux_keys` (Task 7); `build_cvar_gadget` (Task 8); `ENCODINGS` (Task 3).
- Produces:
  - `DENSE_EMBEDDING_CEILING = 230`
  - `dynamic_range_ratio(qubo) -> float`
  - `variable_count(instance, encoding, *, n_zeta_bits=8) -> int`
  - `BudgetRow(tier, dials, encoding, n_vars, fits_dense_embedding)` with `describe()`
  - `variable_budget(instances, *, encodings=ENCODINGS, n_zeta_bits=8) -> tuple[BudgetRow, ...]`

**`variable_count` must never build a matrix.** At T3 under one-hot the layout names roughly `30 × 12 × 16 ≈ 5,760` bits, and a dense `5,760 × 5,760` float matrix is 265 MB — per term. Counting keys costs nothing, and counting is all the budget needs. This is not an optimisation; it is the difference between a table that can be produced and one that cannot.

**What the ratio means.** §6.4 defines `max|coefficient| / minimum meaningful objective gap`. The finest distinction the hardware has to resolve is the smallest non-zero coefficient in the matrix, so that is the denominator. D-Wave couplers carry roughly 4–5 bits of effective precision, so a ratio far above ~32 means the objective's fine structure falls below the analogue floor and the annealer is solving something coarser than the problem posed. §6.4: "No result is interpretable without it." M3a records it; M4's runner owns the warning threshold.

- [ ] **Step 1: Write the failing test**

Create `tests/encoding/test_diagnostics.py`:

```python
import json

import numpy as np
import pytest

from quantic.core.types import Asset
from quantic.encoding.diagnostics import (
    DENSE_EMBEDDING_CEILING,
    dynamic_range_ratio,
    variable_budget,
    variable_count,
)
from quantic.encoding.encode import encode
from quantic.encoding.integer import ENCODINGS, BinaryEncoding, OneHotEncoding
from quantic.encoding.qubo import QuboBuilder
from quantic.encoding.registry import RegistryBuilder, VarRegistry
from quantic.problem.constraints.full_liquidation import FullLiquidation
from quantic.problem.dials import Dials
from quantic.problem.instance import Instance
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk, VarianceRisk
from quantic.problem.scenarios import ScenarioSet
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000


def _instance(**kw) -> Instance:
    base = dict(
        instance_id="x", tier="T0",
        assets=tuple(Asset(s, lot_size=100, tick_size=0.01) for s in ("A", "B")),
        n_buckets=2, initial_lots=(4, 4), max_lots_per_bucket=4,
        params=MarketParams(
            assets=tuple(
                AssetParams(s, 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + j, 0.01)
                for j, s in enumerate(("A", "B"))
            ),
            covariance=np.eye(2) * 1e-5, bucket_ns=BUCKET_NS,
        ),
        dials=Dials(), constraints=(FullLiquidation(),),
        risk=VarianceRisk(lam=1e-6), witness=Schedule(lots=((2, 2), (2, 2))), seed=0,
    )
    base.update(kw)
    return Instance(**base)


def _cvar_instance(n=6):
    rng = np.random.default_rng(11)
    scenarios = ScenarioSet(
        returns=rng.normal(scale=0.01, size=(n, 2, 2)),
        weights=np.full(n, 1.0 / n), seed=11,
    )
    return _instance(
        dials=Dials(cvar_risk=True),
        risk=CVaRRisk(lam=1e-6, alpha=0.95, scenarios=scenarios),
    )


def test_the_ratio_is_the_largest_coefficient_over_the_smallest_non_zero_one():
    b = RegistryBuilder()
    b.add(("a",))
    b.add(("b",))
    builder = QuboBuilder(b.freeze())
    builder.add_linear(("a",), 8.0)
    builder.add_quadratic(("a",), ("b",), 0.5)
    assert dynamic_range_ratio(builder.build()) == pytest.approx(16.0)


def test_an_all_zero_qubo_has_no_meaningful_ratio():
    reg = VarRegistry(keys=(("a",),))
    assert dynamic_range_ratio(QuboBuilder(reg).build()) == 0.0


def test_the_constant_does_not_enter_the_ratio():
    """A constant shifts every energy equally; no coupler has to resolve it."""
    reg = VarRegistry(keys=(("a",),))
    builder = QuboBuilder(reg)
    builder.add_linear(("a",), 2.0)
    builder.add_constant(1e12)
    assert dynamic_range_ratio(builder.build()) == pytest.approx(1.0)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_the_count_agrees_with_what_encode_actually_builds(encoding):
    for inst in (_instance(), _cvar_instance()):
        assert variable_count(inst, encoding) == encode(inst, encoding).n_vars


def test_one_hot_costs_more_variables_than_the_log_encoding():
    inst = _instance()
    assert variable_count(inst, OneHotEncoding()) > variable_count(inst, BinaryEncoding())


def test_cvar_auxiliaries_dominate_the_count():
    """Why scenario reduction is mandatory rather than an optimisation."""
    plain = variable_count(_instance(), BinaryEncoding())
    cvar = variable_count(_cvar_instance(n=20), BinaryEncoding())
    assert cvar > 5 * plain


def test_the_budget_rows_flag_what_fits_a_dense_embedding():
    rows = variable_budget([_instance(), _cvar_instance(n=20)])
    assert len(rows) == 2 * len(ENCODINGS)
    small = next(r for r in rows if r.dials == "none" and r.encoding == "binary")
    assert small.fits_dense_embedding is (small.n_vars <= DENSE_EMBEDDING_CEILING)
    assert any(not r.fits_dense_embedding for r in rows)


def test_the_budget_is_json_safe_for_the_results_row():
    rows = variable_budget([_instance()])
    payload = json.loads(json.dumps([r.describe() for r in rows]))
    assert payload[0]["tier"] == "T0"


def test_counting_stays_cheap_on_a_layout_no_matrix_could_hold():
    """A dense T3-sized one-hot term is hundreds of megabytes; counting keys is free.

    The instance below names more bits than a dense matrix of that size would
    fit in memory, so the fact that this returns at all is the assertion.
    """
    wide = _instance(
        assets=tuple(
            Asset(f"S{i:02d}", lot_size=100, tick_size=0.01) for i in range(30)
        ),
        n_buckets=12,
        initial_lots=(15,) * 30,
        max_lots_per_bucket=15,
        params=MarketParams(
            assets=tuple(
                AssetParams(f"S{i:02d}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0, 0.01)
                for i in range(30)
            ),
            covariance=np.eye(30) * 1e-5,
            bucket_ns=BUCKET_NS,
        ),
        witness=Schedule(lots=tuple((15,) + (0,) * 11 for _ in range(30))),
    )
    assert variable_count(wide, OneHotEncoding()) == 30 * 12 * 16
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_diagnostics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantic.encoding.diagnostics'`.

- [ ] **Step 3: Write the implementation**

Create `src/quantic/encoding/diagnostics.py`:

```python
"""The two numbers no encoded instance may be reported without.

**Dynamic range** (spec section 6.4). D-Wave couplers carry roughly four to
five bits of effective precision. The finest distinction the hardware must
resolve is the smallest non-zero coefficient in the matrix, so the ratio of
the largest to that is what says whether the objective's structure survives
the analogue floor. Above it the annealer is solving something coarser than
the problem posed -- while being reported as an optimiser. Section 6.4: "No
result is interpretable without it."

**The variable budget** (spec section 3.2). Dense embedding runs out near 230
variables. With dial D3 on, the auxiliaries outnumber the decision variables
by an order of magnitude, so the budget is what says which cells of the study
can reach real hardware at all -- a question otherwise discovered at M5, on
the QPU, with the month's quota already spent.

The budget counts **keys, never matrices**. At T3 under one-hot the layout
names roughly 5,760 bits, and one dense term of that size is 265 MB. Counting
is free and is all the budget needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quantic.encoding.cvar import build_cvar_gadget
from quantic.encoding.integer import ENCODINGS, IntegerEncoding
from quantic.encoding.layout import plan_layout
from quantic.encoding.penalty import penalty_aux_keys
from quantic.encoding.qubo import Qubo
from quantic.problem.instance import Instance
from quantic.problem.risk import CVaRRisk

# Spec section 3.2: dense minor-embedding on Advantage2 fails above roughly
# this many logical variables.
DENSE_EMBEDDING_CEILING = 230


def dynamic_range_ratio(qubo: Qubo) -> float:
    """``max|coefficient| / min non-zero |coefficient|``, or ``0.0`` if empty.

    The constant is excluded: it shifts every energy equally, so no coupler
    has to resolve it.
    """
    magnitudes = np.abs(np.asarray(qubo.Q, dtype=float))
    non_zero = magnitudes[magnitudes != 0.0]
    if non_zero.size == 0:
        return 0.0
    return float(non_zero.max() / non_zero.min())


def variable_count(
    instance: Instance, encoding: IntegerEncoding, *, n_zeta_bits: int = 8
) -> int:
    """How many binaries ``encode`` would name, without building anything.

    The penalty strength does not change which keys exist, so the gadget is
    counted at a strength of zero.
    """
    layout = plan_layout(instance, encoding)
    total = len(layout.registry) + len(penalty_aux_keys(instance, layout))
    if isinstance(instance.risk, CVaRRisk):
        gadget = build_cvar_gadget(
            instance, layout, n_zeta_bits=n_zeta_bits, penalty=0.0
        )
        total += len(gadget.keys)
    return total


@dataclass(frozen=True, slots=True)
class BudgetRow:
    """One cell of the tier x dials x encoding table."""

    tier: str
    dials: str
    encoding: str
    n_vars: int
    fits_dense_embedding: bool

    def describe(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "dials": self.dials,
            "encoding": self.encoding,
            "n_vars": self.n_vars,
            "fits_dense_embedding": self.fits_dense_embedding,
        }


def variable_budget(
    instances,
    *,
    encodings: tuple[IntegerEncoding, ...] = ENCODINGS,
    n_zeta_bits: int = 8,
) -> tuple[BudgetRow, ...]:
    """The table spec section 13 of the M3a design promises."""
    rows: list[BudgetRow] = []
    for instance in instances:
        for encoding in encodings:
            n_vars = variable_count(instance, encoding, n_zeta_bits=n_zeta_bits)
            rows.append(
                BudgetRow(
                    tier=instance.tier,
                    dials=instance.dials.label,
                    encoding=encoding.name,
                    n_vars=n_vars,
                    fits_dense_embedding=n_vars <= DENSE_EMBEDDING_CEILING,
                )
            )
    return tuple(rows)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_diagnostics.py -q`
Expected: every test in the file passes.

- [ ] **Step 5: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 6: Commit**

```bash
git add src/quantic/encoding/diagnostics.py tests/encoding/test_diagnostics.py
git commit -m "feat(encoding): record dynamic range and the variable budget"
```

---

## Task 12: The M3a gate

**Files:**
- Create: `tests/encoding/test_m3a_gate.py`

**Interfaces:**
- Consumes: everything. Produces nothing but a verdict.

**What the gate asserts, and what it deliberately does not.** Energy equality is the whole claim: at an encoded feasible schedule the QUBO's energy equals the reference objective, term by term. No solver is involved, so nothing an optimiser does can mask an encoding bug — which is the entire reason M3 was split at this seam.

**Equality is exact up to double-precision round-off only.** `evaluate` accumulates term by term in a Python loop; `Qubo.energy` accumulates through a matrix product. The orders differ, so the results differ in the last bits and nowhere else. A relative tolerance at `1e-9` is orders of magnitude tighter than any modelling error and orders of magnitude looser than accumulation noise, so it cannot absorb a real defect. This is **not** the kind of tolerance M2b §3 refused: no modelling slack hides in it, and the two genuinely inexact pieces — the D1 approximand and the CVaR gadget — are handled by comparing against the function the encoder *declares*, with their own separately asserted bounds.

**The gate is slow, deliberately.** Each test re-encodes rather than sharing a
fixture of every encoded instance. Holding all 160 of them at once would run to
gigabytes; re-encoding keeps peak memory at one instance and costs minutes of
wall clock. Expect the file to take several minutes, and do not "optimise" it
into a module-scoped dict of `EncodedInstance`.

**Tier coverage, and why it is not all four everywhere.** A dense QUBO matrix is `n²` doubles. At T3 the one-hot layout names roughly 5,760 bits, which is 265 MB **per term**, so materialising the breakdown there would need gigabytes. Energy equality therefore runs on **T0, T1 and T2 under all three encodings** — 48 instances × 3 — and on **T3 under the log encoding**, whose 30 × 12 × 4 layout stays tractable. T3 under one-hot and domain-wall is covered by the variable budget, which is the artifact that shows those cells cannot reach the QPU in any case. The design doc's §12 records this.

- [ ] **Step 1: Write the gate**

Create `tests/encoding/test_m3a_gate.py`:

```python
"""THE M3a GATE -- the encoder is the objective, and can be shown so with no solver.

For every instance the ladder produces, under every encoding that can be
materialised at that tier, the QUBO's energy at an encoded feasible schedule
equals the reference objective term by term. Two pieces are inexact by
construction -- the concave-impact approximand under the log encoding, and the
Rockafellar-Uryasev gadget under every encoding -- and each is compared against
the function the encoder declares it implements, with its own asserted bound.
"""

import dataclasses

import numpy as np
import pytest

from quantic.encoding.decode import decode
from quantic.encoding.diagnostics import (
    DENSE_EMBEDDING_CEILING,
    dynamic_range_ratio,
    variable_budget,
)
from quantic.encoding.encode import encode, encode_solution, reference_breakdown
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.problem.generator import generate_ladder
from quantic.problem.objective import cvar_of, optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
SMALL = {"n_raw": 300, "n_target": 15}
REL = 1e-9

# T3 under a table encoding names thousands of bits, and a dense term of that
# size runs to hundreds of megabytes. Those cells are covered by the variable
# budget instead -- which is the artifact showing they cannot reach the QPU.
MATERIALISABLE = {"T0": 3, "T1": 3, "T2": 3, "T3": 1}


def _params_for(tier):
    rng = np.random.default_rng(tier.n_assets)
    a = rng.normal(size=(tier.n_assets, tier.n_assets))
    return MarketParams(
        assets=tuple(
            AssetParams(f"S{i:02d}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + i, 0.01)
            for i in range(tier.n_assets)
        ),
        covariance=(a @ a.T) / tier.n_assets * 1e-5,
        bucket_ns=BUCKET_NS,
    )


@pytest.fixture(scope="module")
def ladder():
    return generate_ladder(_params_for, seed=2026, include_cvar=True, **SMALL)


def _cases(ladder):
    for instance in ladder:
        for encoding in ENCODINGS[: MATERIALISABLE[instance.tier]]:
            yield instance, encoding


def _zeta_for(encoded, schedule):
    if encoded.gadget is None:
        return None
    return encoded.gadget.zeta.snap(optimal_zeta(encoded.instance, schedule))


def _expected_total(encoded, schedule, zeta):
    """The reference, with the gadget's declared CVaR swapped in when present."""
    reference = reference_breakdown(encoded, schedule, zeta=zeta)
    if encoded.gadget is None:
        return reference.total
    instance = encoded.instance
    continuous = instance.risk.lam * cvar_of(
        instance.risk.scenarios, instance, schedule,
        alpha=instance.risk.alpha, zeta=zeta,
    )
    losses = encoded.gadget.losses_from(instance, encoded.layout, schedule)
    aux = encoded.gadget.optimal_aux(losses, zeta)
    declared = encoded.gadget.encoded_cvar(losses, zeta, aux)
    return reference.total - continuous + declared


def test_the_ladder_is_still_sixteen_combinations_on_four_tiers(ladder):
    assert len(ladder) == 16 * 4 == 64


def test_every_instance_encodes_under_every_materialisable_encoding(ladder):
    """THE GATE, part one: nothing in the ladder defeats the encoder."""
    broken = {}
    for instance, encoding in _cases(ladder):
        try:
            encode(instance, encoding)
        except Exception as exc:  # noqa: BLE001 - the gate reports, it does not filter
            broken[(instance.instance_id, encoding.name)] = repr(exc)
    assert not broken, broken


def test_the_energy_at_the_witness_equals_the_reference_objective(ladder):
    """THE GATE, part two. The load-bearing assertion of the whole milestone."""
    wrong = {}
    for instance, encoding in _cases(ladder):
        encoded = encode(instance, encoding)
        zeta = _zeta_for(encoded, instance.witness)
        bits = encode_solution(encoded, instance.witness, zeta=zeta)
        energy = encoded.qubo.energy(bits)
        expected = _expected_total(encoded, instance.witness, zeta)
        if energy != pytest.approx(expected, rel=REL):
            wrong[(instance.instance_id, encoding.name)] = (energy, expected)
    assert not wrong, wrong


def test_each_objective_term_matches_on_its_own(ladder):
    """A single scalar says nothing about which term is wrong -- objective.py's rule."""
    wrong = {}
    for instance, encoding in _cases(ladder):
        if isinstance(instance.risk, CVaRRisk):
            continue  # the risk term is the gadget's; covered above and below
        encoded = encode(instance, encoding)
        bits = encode_solution(encoded, instance.witness)
        reference = reference_breakdown(encoded, instance.witness)
        pairs = (
            ("spread", encoded.breakdown.spread, reference.spread_cost),
            ("temporary", encoded.breakdown.temporary, reference.temporary_impact_cost),
            ("permanent", encoded.breakdown.permanent, reference.permanent_impact_cost),
            ("risk", encoded.breakdown.risk, reference.risk_cost),
        )
        for name, qubo, expected in pairs:
            if qubo.energy(bits) != pytest.approx(expected, rel=REL):
                wrong[(instance.instance_id, encoding.name, name)] = (
                    qubo.energy(bits), expected
                )
    assert not wrong, wrong


def test_the_penalty_block_is_exactly_zero_at_every_witness(ladder):
    """A feasible point pays nothing. Exactly nothing."""
    paying = {}
    for instance, encoding in _cases(ladder):
        encoded = encode(instance, encoding)
        zeta = _zeta_for(encoded, instance.witness)
        bits = encode_solution(encoded, instance.witness, zeta=zeta)
        energy = encoded.penalties.energy(bits)
        if energy != 0.0:
            paying[(instance.instance_id, encoding.name)] = energy
    assert not paying, paying


def test_a_one_lot_violation_raises_the_total_energy(ladder):
    """The property derived penalties exist to guarantee."""
    for instance, encoding in _cases(ladder):
        if instance.dials.discrete_participation or instance.dials.block_trades:
            continue  # a one-lot edit need not be representable under D2 or D4
        grid = instance.witness.as_array().copy()
        cell = np.argwhere(grid > 0)[0]
        grid[cell[0], cell[1]] -= 1
        short = Schedule.from_array(grid)
        encoded = encode(instance, encoding)
        zeta = _zeta_for(encoded, instance.witness)
        good = encoded.qubo.energy(
            encode_solution(encoded, instance.witness, zeta=zeta)
        )
        bad = encoded.qubo.energy(
            encode_solution(encoded, short, zeta=_zeta_for(encoded, short))
        )
        assert bad > good, (instance.instance_id, encoding.name)


def test_every_witness_decodes_back_to_itself(ladder):
    round_trip = {}
    for instance, encoding in _cases(ladder):
        encoded = encode(instance, encoding)
        zeta = _zeta_for(encoded, instance.witness)
        sample = decode(encoded, encode_solution(encoded, instance.witness, zeta=zeta))
        if sample.schedule != instance.witness or sample.classification != "feasible":
            round_trip[(instance.instance_id, encoding.name)] = sample.classification
    assert not round_trip, round_trip


def test_every_declared_substitution_reports_a_finite_bound(ladder):
    """The approximand and the gadget say what they cost, and mean it."""
    for instance, encoding in _cases(ladder):
        encoded = encode(instance, encoding)
        approximand = encoded.breakdown.approximand
        if approximand is not None:
            assert encoding.name == "binary"
            assert instance.dials.concave_impact
            assert np.isfinite(approximand.max_abs_error)
            assert approximand.max_abs_error >= 0.0
        elif instance.dials.concave_impact:
            assert encoding.supports_exact_table
        if encoded.gadget is not None:
            assert np.isfinite(encoded.gadget.resolution)
            assert encoded.gadget.resolution > 0.0


def test_the_declared_cvar_stays_within_its_reported_resolution(ladder):
    """The second of the gadget's two assertions, across the whole ladder."""
    outside = {}
    for instance, encoding in _cases(ladder):
        if not isinstance(instance.risk, CVaRRisk):
            continue
        encoded = encode(instance, encoding)
        zeta = _zeta_for(encoded, instance.witness)
        losses = encoded.gadget.losses_from(instance, encoded.layout, instance.witness)
        aux = encoded.gadget.optimal_aux(losses, zeta)
        declared = encoded.gadget.encoded_cvar(losses, zeta, aux)
        continuous = instance.risk.lam * cvar_of(
            instance.risk.scenarios, instance, instance.witness,
            alpha=instance.risk.alpha, zeta=zeta,
        )
        if abs(declared - continuous) > encoded.gadget.resolution:
            outside[(instance.instance_id, encoding.name)] = (
                abs(declared - continuous), encoded.gadget.resolution
            )
    assert not outside, outside


def test_every_encoded_instance_records_a_dynamic_range(ladder):
    """Section 6.4: no result is interpretable without it."""
    for instance, encoding in _cases(ladder):
        ratio = dynamic_range_ratio(encode(instance, encoding).qubo)
        assert np.isfinite(ratio)
        assert ratio >= 1.0


def test_the_variable_budget_covers_every_cell_of_the_study(ladder):
    """THE GATE, part three: which cells can reach Advantage2 at all."""
    rows = variable_budget(ladder)
    assert len(rows) == 64 * len(ENCODINGS)
    assert {r.tier for r in rows} == {"T0", "T1", "T2", "T3"}
    assert {r.encoding for r in rows} == {e.name for e in ENCODINGS}
    assert all(r.n_vars > 0 for r in rows)
    fitting = [r for r in rows if r.fits_dense_embedding]
    assert fitting, "no cell of the study fits a dense embedding"
    assert all(r.n_vars <= DENSE_EMBEDDING_CEILING for r in fitting)


def test_the_encoding_description_is_json_safe_for_every_instance(ladder):
    import json

    for instance, encoding in _cases(ladder):
        payload = json.dumps(encode(instance, encoding).describe())
        assert json.loads(payload)["encoding"] == encoding.name


def test_encoding_is_reproducible_across_calls(ladder):
    """A results row pins the QUBO; two runs must agree bit for bit."""
    for instance in ladder[:8]:
        a = encode(instance, BinaryEncoding())
        b = encode(instance, BinaryEncoding())
        assert np.array_equal(a.qubo.Q, b.qubo.Q)
        assert a.qubo.constant == b.qubo.constant
```

- [ ] **Step 2: Run the gate**

Run: `.venv/Scripts/python.exe -m pytest tests/encoding/test_m3a_gate.py -q`
Expected: every test passes. This is the first run that can fail for an interesting reason; a failure here names the instance, the encoding and the term.

- [ ] **Step 3: Record the variable budget**

Run the budget once and read it — this is the deliverable §13 of the design promises, and the answer shapes M5:

```bash
.venv/Scripts/python.exe -c "import json, numpy as np; from quantic.problem.generator import generate_ladder; from quantic.problem.params import AssetParams, MarketParams; from quantic.encoding.diagnostics import variable_budget; exec(open('tests/encoding/test_m3a_gate.py').read().split('@pytest.fixture')[0].split('def _params_for')[0]); print('see the gate module for _params_for')"
```

Simpler: add a throwaway script under the scratchpad that imports `_params_for` from the gate module, builds the ladder once and prints `[r.describe() for r in variable_budget(ladder)]` as JSON. Report in the commit message which tier × dials × encoding cells come in under `DENSE_EMBEDDING_CEILING`. Do **not** commit the script.

- [ ] **Step 4: Run the full suite, lint and the layer contract**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/lint-imports.exe
```

Expected: all green, `Contracts: 1 kept, 0 broken.`

- [ ] **Step 5: Commit**

```bash
git add tests/encoding/test_m3a_gate.py
git commit -m "feat(encoding): complete the M3a gate — the QUBO is the objective"
```

---

## M3a is complete at this point

The encoder is proven to be the objective, with no solver involved and no
tolerance on the correctness claim. Two pieces are inexact by construction, and
each is compared against the function the encoder declares it implements, with
its own separately asserted bound: the concave-impact approximand under log,
and the Rockafellar-Uryasev gadget everywhere.

## What M3b picks up

`EncodedInstance` and `decode` are the whole interface. M3b adds
`solvers/classical/`: brute force at T0, the piecewise-linear under- and
over-estimator MILPs through `scipy.optimize.milp` (HiGHS, already a
dependency), the rigorous optimality bracket of parent spec §7.3, and the
parent spec §14 gate — **decoded QUBO optimum equals exact MILP optimum on T0,
all 16 combinations**.

Two hand-offs matter. First, with D3 on, T0's QUBO carries hundreds of
variables and cannot be brute-forced; but `CVaRGadget.optimal_aux` is
closed-form for a fixed schedule and ζ, so M3b enumerates the T0 lot grid × the
ζ grid and solves the auxiliaries directly. Second, the PWL baseline must
linearise the **same** concave impact the encoder used, and under the log
encoding that is the approximand rather than the true power law — so M3b's
bracket is stated against `reference_breakdown`, exactly as this gate is.
