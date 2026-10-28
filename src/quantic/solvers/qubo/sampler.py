"""Running a sampler on a QUBO, and judging every sample it returns.

Every sampler in this package -- annealing, tabu, steepest descent, and at M5
the quantum ones -- runs on the same ``Qubo`` object and goes through this
loop, so a difference between two of them is the sampler's and not the
bookkeeping's.

Samples are drawn in batches of 1, 2, 4, ... up to ``MAX_BATCH`` reads, each
batch seeded from ``(seed, batch index)``. The stream of samples is therefore
fixed by the seed alone, and a run stopped by the clock returns a prefix of
the stream a longer run returns. A run always draws at least one batch.

Each distinct bitstring is decoded once and classified by
``encoding.decode`` -- never repaired, parent spec section 6.5. A feasible one
is scored by the evaluator at its own minimising ``zeta``, which is the true
objective; the ``zeta`` bits the sampler chose are the encoding's business,
and what they cost is the regret measured against the QUBO's minimum.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

import dimod
import numpy as np

from quantic.encoding.decode import FEASIBLE, decode
from quantic.encoding.encode import EncodedInstance, encode
from quantic.encoding.integer import ENCODINGS, IntegerEncoding
from quantic.encoding.qubo import Qubo
from quantic.problem.feasibility import classify
from quantic.problem.instance import Instance
from quantic.problem.schedule import Schedule
from quantic.solvers.base import HEURISTIC, Budget, Result
from quantic.solvers.classical.scoring import score

# The CVaR gadget's zeta width for the study (M3b gate result).
STUDY_ZETA_BITS = 8
MAX_BATCH = 64

# (bqm, reads, seed) -> (bits in variable order 0..n-1, the sampler's energies)
Draw = Callable[[dimod.BinaryQuadraticModel, int, int], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Sample:
    """One distinct bitstring a run returned, and how often."""

    digest: str
    count: int
    energy: float                 # as the sampler reported it
    classification: str           # encoding.decode's verdict
    total: float | None           # the evaluator's total, when feasible


def encoding_named(name: str) -> IntegerEncoding:
    for encoding in ENCODINGS:
        if encoding.name == name:
            return encoding
    raise KeyError(f"unknown encoding {name!r}; known: {[e.name for e in ENCODINGS]}")


def to_bqm(qubo: Qubo) -> dimod.BinaryQuadraticModel:
    """Every variable is added, including any with no coefficient at all."""
    bqm = dimod.BinaryQuadraticModel("BINARY")
    q = qubo.Q
    bqm.add_variables_from((i, float(q[i, i])) for i in range(qubo.n_vars))
    rows, cols = np.nonzero(np.triu(q, 1))
    bqm.add_interactions_from(
        (int(i), int(j), float(q[i, j])) for i, j in zip(rows, cols, strict=True)
    )
    bqm.offset = float(qubo.constant)
    return bqm


def in_variable_order(sampleset: dimod.SampleSet, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A sampleset's bits as an ``(m, n)`` 0/1 array in registry order."""
    order = [sampleset.variables.index(v) for v in range(n)]
    record = sampleset.record
    counts = record.num_occurrences
    bits = np.repeat(record.sample[:, order].astype(np.int8), counts, axis=0)
    energies = np.repeat(record.energy.astype(float), counts)
    return bits, energies


def energies(qubo: Qubo, bits: np.ndarray) -> np.ndarray:
    """``z' Q z + constant`` for each row of ``bits``."""
    z = bits.astype(float)
    return ((z @ qubo.Q) * z).sum(axis=1) + qubo.constant


def batch_seed(seed: int, batch: int) -> int:
    """A seed below 2**31: ``dwave.samplers``' annealer refuses anything larger."""
    state = np.random.SeedSequence([int(seed), int(batch)]).generate_state(1)[0]
    return int(state) & 0x7FFF_FFFF


def run_sampler(
    name: str, enc: EncodedInstance, budget: Budget, seed: int, draw: Draw,
    *, one_batch: bool = False,
) -> Result:
    """Draw batches until the budget is spent, then report every sample.

    ``one_batch`` draws the whole sample cap at once: a circuit sampled once
    at fixed angles, rather than a sampler whose stream is cut by the clock.
    """
    if one_batch and budget.max_samples is None:
        raise ValueError(f"{name}: one batch needs a sample cap to know its size")
    if budget.wall_clock_s is None and budget.max_samples is None:
        raise ValueError(
            f"{name} needs a wall clock or a sample cap: a sampler has no other "
            "way to know when to stop"
        )
    started = time.perf_counter()
    bqm = to_bqm(enc.qubo)
    seen: dict[bytes, list] = {}
    best: tuple[float, Schedule] | None = None
    best_energy = np.inf
    trace: list[tuple[float, float]] = []
    mismatch = 0.0
    drawn = 0
    batch = 0
    size = budget.max_samples if one_batch else 1

    while True:
        if budget.max_samples is not None:
            size = min(size, budget.max_samples - drawn)
        bits, reported = draw(bqm, size, batch_seed(seed, batch))
        own = energies(enc.qubo, bits)
        scale = np.maximum(1.0, np.abs(own))
        mismatch = max(mismatch, float(np.max(np.abs(reported - own) / scale)))
        now = time.perf_counter() - started
        for row, energy in zip(bits, reported, strict=True):
            key = row.tobytes()
            if key in seen:
                seen[key][1] += 1
                continue
            decoded = decode(enc, row)
            total = None
            if decoded.classification == FEASIBLE:
                total = score(enc.instance, decoded.schedule)[0].total
                if best is None or total < best[0]:
                    best = (total, decoded.schedule)
            seen[key] = [hashlib.blake2b(key, digest_size=8).hexdigest(), 1,
                         float(energy), decoded.classification, total]
            if energy < best_energy:
                best_energy = float(energy)
                trace.append((now, best_energy))
        drawn += len(bits)
        batch += 1
        size = min(2 * size, MAX_BATCH)
        if budget.max_samples is not None and drawn >= budget.max_samples:
            break
        if (
            budget.wall_clock_s is not None
            and time.perf_counter() - started >= budget.wall_clock_s
        ):
            break

    samples = tuple(Sample(*fields) for fields in seen.values())
    detail = {
        "encoding": enc.layout.encoding_name,
        "n_vars": enc.n_vars,
        "zeta_bits": None if enc.gadget is None else enc.gadget.zeta.n_bits,
        "n_samples": drawn,
        "n_batches": batch,
        "best_energy": best_energy,
        "energy_mismatch": mismatch,
        "samples": samples,
        "trace": tuple(trace),
    }
    timings = {"total_s": time.perf_counter() - started}
    if best is None:
        return Result(
            solver=name, status=HEURISTIC, schedule=None, zeta=None, objective=None,
            bracket=None, feasibility=None, timings=timings, detail=detail,
        )
    total, schedule = best
    breakdown, zeta = score(enc.instance, schedule)
    return Result(
        solver=name, status=HEURISTIC, schedule=schedule, zeta=zeta,
        objective=breakdown, bracket=None,
        feasibility=classify(enc.instance, schedule),
        timings=timings, detail=detail,
    )


class QuboSampler:
    """A sampler behind the ``Solver`` protocol, for one encoding and one seed.

    ``solve`` encodes the instance; the bench encodes once per worker and calls
    ``solve_encoded`` instead.
    """

    name: str = ""

    def __init__(
        self, encoding: str, *, seed: int = 0, zeta_bits: int = STUDY_ZETA_BITS
    ) -> None:
        self.encoding = encoding_named(encoding)
        self.seed = seed
        self.zeta_bits = zeta_bits

    def draw(self, bqm: dimod.BinaryQuadraticModel, reads: int, seed: int):
        raise NotImplementedError

    def solve(self, instance: Instance, budget: Budget) -> Result:
        return self.solve_encoded(
            encode(instance, self.encoding, n_zeta_bits=self.zeta_bits), budget
        )

    def solve_encoded(self, enc: EncodedInstance, budget: Budget) -> Result:
        if enc.layout.encoding_name != self.encoding.name:
            raise ValueError(
                f"{self.name} was built for {self.encoding.name!r} and given an "
                f"instance encoded with {enc.layout.encoding_name!r}"
            )
        return run_sampler(self.name, enc, budget, self.seed, self.draw)
