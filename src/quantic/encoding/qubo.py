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

    def folded(self) -> LinearForm:
        """The same form with every key named once, in first-appearance order.

        A holdings sum names a bit once per bucket it is still held through,
        so a form assembled by addition can carry each key many times over.
        Expanding a product pairwise over the repeats costs the square of
        that multiplicity and buys nothing.
        """
        summed: dict[VarKey, float] = {}
        for key, coef in self.terms:
            summed[key] = summed.get(key, 0.0) + coef
        if len(summed) == len(self.terms):
            return self
        return LinearForm(constant=self.constant, terms=tuple(summed.items()))


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
        object.__setattr__(self, "Q", q)

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
                f"energy needs {self.n_vars} variables, got a vector of shape "
                f"{z.shape}"
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
        """Accumulate ``coef * z_a * z_b``, folding ``z**2 = z`` when they
        coincide."""
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

        Both forms may repeat keys; each is folded first, so the expansion
        costs the distinct keys rather than the repeats. The ``i == j``
        case is delegated to :meth:`add_quadratic`, which folds it onto the
        diagonal -- the step a naive expansion gets wrong, because a binary's
        square is the binary itself rather than a separate quadratic term.
        """
        s = float(scale)
        a = a.folded()
        b = b.folded()
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
        return Qubo(
            Q=self._q.copy(), constant=self._constant, registry=self._registry
        )
