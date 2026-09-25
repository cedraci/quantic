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
