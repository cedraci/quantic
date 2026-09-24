"""The four hardness dials of spec section 4.

Each is an independent on/off switch, and the combinations form a factor in
the experiment design (spec section 9.2): the study measures *which* hardness
features move the crossover, not merely whether one exists.

D2 governs cardinality and minimum lot **together**, per spec section 4:
cardinality without a minimum lot size is not meaningful on a real desk.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

# Order matters: it fixes the label format and the enumeration order, both of
# which appear in every results row.
_DIAL_LABELS: tuple[tuple[str, str], ...] = (
    ("concave_impact", "D1"),
    ("discrete_participation", "D2"),
    ("cvar_risk", "D3"),
    ("block_trades", "D4"),
)


@dataclass(frozen=True, slots=True)
class Dials:
    """Which sources of combinatorial hardness are switched on."""

    concave_impact: bool = False           # D1 -- square-root / concave impact law
    discrete_participation: bool = False   # D2 -- cardinality AND minimum lot
    cvar_risk: bool = False                # D3 -- CVaR instead of variance (M2b)
    block_trades: bool = False             # D4 -- all-or-nothing blocks

    @classmethod
    def combinations(cls, *, include_cvar: bool = False) -> tuple[Dials, ...]:
        """Every dial combination, all-off first.

        ``include_cvar=False`` yields M2a's 8. Once M2b lands, flipping this to
        ``True`` yields all 16 with no other change.
        """
        switchable = [name for name, _ in _DIAL_LABELS if include_cvar or name != "cvar_risk"]
        out: list[Dials] = []
        for flags in itertools.product((False, True), repeat=len(switchable)):
            out.append(cls(**dict(zip(switchable, flags, strict=True))))
        return tuple(out)

    @property
    def label(self) -> str:
        """A stable, readable identifier such as ``"D1+D4"`` or ``"none"``."""
        active = [label for name, label in _DIAL_LABELS if getattr(self, name)]
        return "+".join(active) if active else "none"
