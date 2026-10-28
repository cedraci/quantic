"""QAOA behind the ``QuboSampler`` interface -- M5 design section 6.

Every shot goes through ``run_sampler`` as one batch, so decoding,
classification and scoring are the M4 samplers' own. The energy of a shot is
the QUBO's at its bits; QAOA reports no energy of its own.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from quantic.encoding.encode import EncodedInstance
from quantic.solvers.base import Budget, Result
from quantic.solvers.quantum.backends import (
    fake_backend,
    route,
    routed_stats,
    sample_ideal,
    sample_noisy,
)
from quantic.solvers.quantum.ising import to_ising
from quantic.solvers.quantum.train import Angles
from quantic.solvers.qubo.sampler import (
    STUDY_ZETA_BITS,
    QuboSampler,
    energies,
    run_sampler,
)


class QaoaSampler(QuboSampler):
    backend_name = ""

    def __init__(self, encoding: str, *, seed: int = 0, zeta_bits: int = STUDY_ZETA_BITS,
                 angles: Angles, shots: int, device: str | None = None) -> None:
        super().__init__(encoding, seed=seed, zeta_bits=zeta_bits)
        self.angles, self.shots, self.device = angles, shots, device
        self.extra: dict = {}

    def prepare(self, enc: EncodedInstance) -> None:
        """Work done once per run, before the shots: routing, for instance."""

    def sample(self, enc: EncodedInstance, shots: int, seed: int) -> np.ndarray:
        raise NotImplementedError

    def solve_encoded(self, enc: EncodedInstance, budget: Budget | None = None) -> Result:
        if enc.layout.encoding_name != self.encoding.name:
            raise ValueError(
                f"{self.name} was built for {self.encoding.name!r} and given an "
                f"instance encoded with {enc.layout.encoding_name!r}"
            )
        self.prepare(enc)

        def draw(bqm, reads, seed):
            bits = self.sample(enc, reads, seed)
            return bits, energies(enc.qubo, bits)

        result = run_sampler(self.name, enc, Budget(max_samples=self.shots), self.seed,
                             draw, one_batch=True)
        detail = {**result.detail, "backend": self.backend_name, "device": self.device,
                  "shots": self.shots, **self.extra}
        return dataclasses.replace(result, detail=detail)


class QaoaIdeal(QaoaSampler):
    name = "qaoa_ideal"
    backend_name = "ideal"

    def sample(self, enc, shots, seed):
        return sample_ideal(enc.qubo, self.angles, shots, seed)


class QaoaNoisy(QaoaSampler):
    name = "qaoa_noisy"
    backend_name = "noisy"

    def prepare(self, enc):
        if self.device is None:
            raise ValueError("qaoa_noisy needs the device whose fake twin it simulates")
        self.fake = fake_backend(self.device)
        self.routed = route(to_ising(enc.qubo), self.angles, self.fake)
        depth, two = routed_stats(self.routed)
        self.extra.update(routed_depth=depth, routed_2q=two)

    def sample(self, enc, shots, seed):
        return sample_noisy(self.routed, self.fake, shots, seed)


class QaoaCollected(QaoaSampler):
    """Bits a real device returned, judged like any other shots."""

    name = "qaoa_ibm"
    backend_name = "ibm"

    def __init__(self, encoding: str, *, bits: np.ndarray, **kw) -> None:
        super().__init__(encoding, **kw)
        self.bits = np.asarray(bits, dtype=np.int8)
        if len(self.bits) != self.shots:
            raise ValueError(f"{len(self.bits)} collected shots, {self.shots} expected")

    def sample(self, enc, shots, seed):
        return self.bits


QAOA_CLASSES = {c.name: c for c in (QaoaIdeal, QaoaNoisy, QaoaCollected)}
