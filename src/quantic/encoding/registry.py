"""Names for every binary in a QUBO.

A QUBO is an opaque matrix, and the gate that proves it correct compares
energies term by term. When a term disagrees, "coefficient 47 is wrong" sends
an implementer back to the drawing board; ``('x', 1, 2, 0)`` sends them to one
line. The registry is the difference.

Key conventions, fixed here because every module below depends on them:

    ("x", i, t, b)      expansion bit b of the lots traded in cell (i, t)
    ("y", i, t)         the min-participation indicator for cell (i, t)
    ("block", i, t)     the all-or-nothing bit for the block on cell (i, t)
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
