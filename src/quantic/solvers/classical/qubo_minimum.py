"""The exact minimum of an encoded instance's energy.

The gate of spec section 14 asks where the QUBO's minimum is, and a sampler
cannot answer that: a sampler that returns a low energy has shown that the
energy is reachable, not that nothing lower exists. This module enumerates.

Two things make that affordable. The bits outside the CVaR gadget are split in
two, so that each half's own energy is computed once and only the cross term
is paid per state. And the gadget is never enumerated: once the lot bits and
``zeta`` are fixed the scenarios separate, the auxiliaries have a closed form,
and what is left is a scan over ``zeta``'s grid that a bound cuts short.

The closed form here is vectorised and so is a second implementation of
``CVaRGadget.cheapest_aux``. The winning state is rebuilt through the
production method, and callers check its energy against ``Qubo.energy``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from quantic.encoding.encode import EncodedInstance, encode_solution
from quantic.encoding.qubo import Qubo
from quantic.solvers.classical.gadget_minimum import GadgetMinimiser

EVERY_STATE = "every_state"
DECODABLE = "decodable"

# 2**28 states is a matter of minutes; each further bit doubles it.
EXHAUSTIVE_BIT_LIMIT = 28

# Energies closer than this, relative to the minimum, are the same energy.
_TIE_RTOL = 1e-9
_BLOCK = 64
_MAX_CANDIDATES = 20_000_000

# How far above the witness's energy a state may sit and still be kept. The
# split sum and the dense z'Qz accumulate in different orders, and at the
# dynamic ranges seen 1e-9 left round-off no room. M3b's final review measured
# the margin to the nearest state that matters at 5e-3 or more.
_WITNESS_RTOL = 1e-6


class EnumerationTooLarge(RuntimeError):
    """More states pass the bound than this enumeration will hold."""


@dataclass(frozen=True)
class QuboMinimum:
    """The lowest energy the QUBO reaches, and a state that reaches it."""

    bits: np.ndarray          # the full vector, auxiliaries included
    energy: float
    scope: str                # EVERY_STATE | DECODABLE
    n_states: int             # how many states of the bits outside the gadget
    n_ties: int               # of those, how many reach the minimum
    n_scanned: int            # how many went through the zeta scan
    widest_window: int        # the most zeta levels any one state needed


@dataclass(frozen=True)
class _Half:
    """Some of the bits outside the gadget, and the patterns they range over."""

    index: np.ndarray         # (k,) registry indices
    patterns: np.ndarray      # (m, k) 0/1


def _all_patterns(k: int) -> np.ndarray:
    return ((np.arange(2**k)[:, None] >> np.arange(k)[None, :]) & 1).astype(float)


def _halves_every_state(outside: np.ndarray) -> tuple[_Half, _Half]:
    cut = len(outside) // 2
    first, second = outside[:cut], outside[cut:]
    return (
        _Half(index=first, patterns=_all_patterns(len(first))),
        _Half(index=second, patterns=_all_patterns(len(second))),
    )


def _groups_decodable(enc: EncodedInstance, outside: np.ndarray) -> list[_Half]:
    """One group per cell, on the patterns its values encode to, then the slack."""
    registry = enc.registry
    groups: list[_Half] = []
    named: set[int] = set()
    for plan in enc.layout.cells:
        index = np.array([registry.index(k) for k in plan.bit_keys], dtype=np.int64)
        rows = []
        for value in plan.values:
            bits = plan.encode(value)
            rows.append([float(bits[k]) for k in plan.bit_keys])
        groups.append(_Half(index=index, patterns=np.array(rows, dtype=float)))
        named.update(int(i) for i in index)
    for i in outside:
        if int(i) not in named:
            groups.append(
                _Half(index=np.array([int(i)]), patterns=np.array([[0.0], [1.0]]))
            )
    return groups


def _merge(groups: list[_Half]) -> _Half:
    if not groups:
        return _Half(index=np.zeros(0, dtype=np.int64), patterns=np.zeros((1, 0)))
    index = np.concatenate([g.index for g in groups])
    rows = [
        np.concatenate(combo)
        for combo in itertools.product(*[list(g.patterns) for g in groups])
    ]
    return _Half(index=index, patterns=np.array(rows, dtype=float))


def _halves_decodable(enc: EncodedInstance, outside: np.ndarray) -> tuple[_Half, _Half]:
    groups = _groups_decodable(enc, outside)
    total = float(np.sum([np.log2(len(g.patterns)) for g in groups]))
    first: list[_Half] = []
    second: list[_Half] = []
    running = 0.0
    for g in groups:
        if running < total / 2.0:
            first.append(g)
            running += float(np.log2(len(g.patterns)))
        else:
            second.append(g)
    return _merge(first), _merge(second)


def _symmetric(qubo: Qubo) -> np.ndarray:
    q = np.asarray(qubo.Q, dtype=float)
    return q + q.T - np.diag(np.diag(q))


def _own_energy(sym: np.ndarray, half: _Half) -> np.ndarray:
    """``z' Q z`` over the half's own bits, for each of its patterns."""
    block = sym[np.ix_(half.index, half.index)]
    upper = np.triu(block)
    return ((half.patterns @ upper) * half.patterns).sum(axis=1)


def qubo_minimum(enc: EncodedInstance) -> QuboMinimum:
    """Minimise ``enc.qubo``'s energy, and say over which states."""
    registry = enc.registry
    inside = (
        set() if enc.gadget is None
        else {registry.index(k) for k in enc.gadget.keys}
    )
    outside = np.array(
        [i for i in range(len(registry)) if i not in inside], dtype=np.int64
    )

    if len(outside) <= EXHAUSTIVE_BIT_LIMIT:
        scope = EVERY_STATE
        first, second = _halves_every_state(outside)
    else:
        scope = DECODABLE
        first, second = _halves_decodable(enc, outside)
    n_states = len(first.patterns) * len(second.patterns)

    flat = enc.breakdown.spread + enc.breakdown.temporary + enc.breakdown.permanent
    flat = flat + enc.penalties
    if enc.gadget is None:
        flat = flat + enc.breakdown.risk
    sym = _symmetric(flat)

    own_first = _own_energy(sym, first)
    own_second = _own_energy(sym, second) + flat.constant
    cross = sym[np.ix_(first.index, second.index)]

    gadget = None if enc.gadget is None else GadgetMinimiser(enc)
    if gadget is not None:
        # The charge is at least lam * CVaR less the slack, and CVaR is at
        # least the mean loss, which is linear in the bits.
        mean = gadget.lam * (gadget.weights @ gadget.loss_matrix)
        bound_first = own_first + first.patterns @ mean[first.index]
        bound_second = (
            own_second + second.patterns @ mean[second.index]
            + gadget.lam * float(gadget.weights @ gadget.base) - gadget.slack
        )
    else:
        bound_first, bound_second = own_first, own_second

    # Somewhere to start: the witness is a state, so its energy is reachable.
    witness = enc.instance.witness
    start_bits = _rebuild(enc, gadget, _outside_of(enc, witness, outside), outside)
    incumbent = float(enc.qubo.energy(start_bits))
    threshold = incumbent + _WITNESS_RTOL * max(1.0, abs(incumbent))

    # Every state whose bound is at or below the incumbent.
    picked_first: list[np.ndarray] = []
    picked_second: list[np.ndarray] = []
    picked_bound: list[np.ndarray] = []
    n_picked = 0
    lead = first.patterns @ cross                                     # (m1, k2)
    for start in range(0, len(second.patterns), _BLOCK):
        block = second.patterns[start:start + _BLOCK]                 # (b, k2)
        values = (
            bound_first[:, None] + bound_second[None, start:start + _BLOCK]
            + lead @ block.T
        )
        a, b = np.nonzero(values <= threshold)
        if a.size:
            picked_first.append(a)
            picked_second.append(b + start)
            picked_bound.append(values[a, b])
            n_picked += a.size
            if n_picked > _MAX_CANDIDATES:
                raise EnumerationTooLarge(
                    f"more than {_MAX_CANDIDATES} states sit at or below the "
                    f"witness's energy on {enc.instance.instance_id}; the witness "
                    "is too poor a start for this enumeration"
                )

    a = np.concatenate(picked_first)
    b = np.concatenate(picked_second)
    bound = np.concatenate(picked_bound)

    if gadget is None:
        energy = bound
        level = np.zeros(a.size)
        n_scanned, widest = 0, 0
    else:
        states = np.zeros((a.size, len(registry)))
        states[:, first.index] = first.patterns[a]
        states[:, second.index] = second.patterns[b]
        losses = gadget.base[None, :] + states @ gadget.loss_matrix.T
        flat_energy = bound - (
            states @ mean + gadget.lam * float(gadget.weights @ gadget.base)
            - gadget.slack
        )
        charge, level, widest = gadget.minimise(losses)
        energy = flat_energy + charge
        n_scanned = int(a.size)

    best = int(np.argmin(energy))
    lowest = float(energy[best])
    ties = int(np.sum(energy <= lowest + _TIE_RTOL * max(1.0, abs(lowest))))

    chosen = np.zeros(len(outside))
    position = {int(i): k for k, i in enumerate(outside)}
    for half, row in ((first, a[best]), (second, b[best])):
        for k, i in enumerate(half.index):
            chosen[position[int(i)]] = half.patterns[row, k]
    bits = _rebuild(enc, gadget, chosen, outside, level=float(level[best]))
    return QuboMinimum(
        bits=bits, energy=lowest, scope=scope, n_states=n_states, n_ties=ties,
        n_scanned=n_scanned, widest_window=int(widest),
    )


def _outside_and_inside(enc: EncodedInstance, schedule) -> np.ndarray:
    """``schedule``'s canonical state. Only its bits outside the gadget are used."""
    zeta = None if enc.gadget is None else enc.gadget.zeta.lo
    return encode_solution(enc, schedule, zeta=zeta).astype(float)


def _outside_of(enc: EncodedInstance, schedule, outside: np.ndarray) -> np.ndarray:
    """The bits outside the gadget for ``schedule``'s canonical state."""
    return _outside_and_inside(enc, schedule)[outside]


def _rebuild(
    enc: EncodedInstance,
    gadget: GadgetMinimiser | None,
    chosen: np.ndarray,
    outside: np.ndarray,
    *,
    level: float | None = None,
) -> np.ndarray:
    """The full bit vector: ``chosen`` outside the gadget, its best state inside.

    The auxiliaries come from ``CVaRGadget.cheapest_aux``, the production
    method, and not from the vectorised closed form above.
    """
    registry = enc.registry
    bits = np.zeros(len(registry), dtype=np.int64)
    bits[outside] = np.asarray(chosen).astype(np.int64)
    if gadget is None:
        return bits
    losses = gadget.base + bits.astype(float) @ gadget.loss_matrix.T
    if level is None:
        _, levels, _ = gadget.minimise(losses[None, :])
        level = float(levels[0])
    zeta = gadget.lo + gadget.step * level
    for key, value in gadget.gadget.zeta.encode(zeta).items():
        bits[registry.index(key)] = value
    for key, value in gadget.gadget.cheapest_aux(losses, zeta).items():
        bits[registry.index(key)] = value
    return bits
