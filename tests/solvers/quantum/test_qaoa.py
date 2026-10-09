import numpy as np
import pytest

from quantic.encoding.decode import FEASIBLE
from quantic.encoding.encode import encode
from quantic.solvers.base import HEURISTIC, Budget
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.scoring import score
from quantic.solvers.quantum.backends import (
    bits_from_counts,
    fake_backend,
    route,
    routed_stats,
    sample_ideal,
    sample_noisy,
)
from quantic.solvers.quantum.ising import to_ising
from quantic.solvers.quantum.qaoa import QaoaCollected, QaoaIdeal, QaoaNoisy
from quantic.solvers.quantum.train import train
from quantic.solvers.qubo.sampler import encoding_named, energies
from tests.solvers.instances import blocks, nothing_to_sell_in_one_name


def _enc(build, encoding="binary"):
    return encode(build(), encoding_named(encoding), n_zeta_bits=8)


def test_counts_are_read_little_endian():
    bits = bits_from_counts({"001": 2, "110": 1}, 3)
    assert bits.tolist() == [[1, 0, 0], [1, 0, 0], [0, 1, 1]]
    with pytest.raises(ValueError, match="3 bits"):
        bits_from_counts({"01": 1}, 3)


def test_ideal_sampling_is_seeded_and_follows_the_state():
    enc = _enc(nothing_to_sell_in_one_name)
    angles = train(enc.qubo, reps=2, alpha=0.25, maxiter=40)
    a = sample_ideal(enc.qubo, angles, 500, seed=3)
    b = sample_ideal(enc.qubo, angles, 500, seed=3)
    assert a.shape == (500, enc.n_vars) and np.array_equal(a, b)
    assert not np.array_equal(a, sample_ideal(enc.qubo, angles, 500, seed=4))


def test_an_unknown_device_names_the_known_fakes():
    with pytest.raises(KeyError, match="fake_fez"):
        fake_backend("ibm_nowhere")


def test_routing_for_a_fake_device_keeps_the_logical_measurements():
    enc = _enc(nothing_to_sell_in_one_name)
    ising = to_ising(enc.qubo)
    angles = train(enc.qubo, reps=1, alpha=0.25, maxiter=10)
    backend = fake_backend("ibm_fez")
    routed = route(ising, angles, backend)
    depth, two = routed_stats(routed)
    assert routed.num_clbits == enc.n_vars and depth > 0 and two > 0
    bits = sample_noisy(routed, backend, 64, seed=1)
    assert bits.shape == (64, enc.n_vars)


@pytest.mark.parametrize("cls", [QaoaIdeal, QaoaNoisy], ids=lambda c: c.name)
def test_every_shot_is_judged_like_a_sampler_read(cls):
    enc = _enc(blocks)
    angles = train(enc.qubo, reps=1, alpha=0.25, maxiter=20)
    sampler = cls("binary", seed=2, angles=angles, shots=64,
                  device="ibm_fez" if cls is QaoaNoisy else None)
    result = sampler.solve_encoded(enc)
    detail = result.detail
    assert result.status == HEURISTIC
    assert detail["n_samples"] == 64 and detail["n_batches"] == 1
    assert sum(s.count for s in detail["samples"]) == 64
    assert detail["energy_mismatch"] == 0.0
    assert detail["backend"] == ("noisy" if cls is QaoaNoisy else "ideal")
    for s in detail["samples"]:
        assert (s.total is not None) == (s.classification == FEASIBLE)
    if result.schedule is not None:
        assert result.objective == score(enc.instance, result.schedule)[0]
    if cls is QaoaNoisy:
        assert detail["routed_depth"] > 0 and detail["routed_2q"] > 0


def test_collected_bits_are_scored_and_nothing_is_drawn_again():
    enc = _enc(nothing_to_sell_in_one_name)
    optimum = brute_force(enc.instance).objective.total
    angles = train(enc.qubo, reps=1, alpha=0.25, maxiter=10)
    bits = sample_ideal(enc.qubo, angles, 40, seed=9)
    result = QaoaCollected("binary", angles=angles, shots=40, bits=bits,
                           device="ibm_fez").solve_encoded(enc)
    assert result.detail["n_samples"] == 40
    assert result.detail["best_energy"] == pytest.approx(energies(enc.qubo, bits).min())
    if result.objective is not None:
        assert result.objective.total >= optimum - 1e-9


def test_one_batch_needs_a_sample_cap():
    from quantic.solvers.qubo.sampler import run_sampler

    enc = _enc(nothing_to_sell_in_one_name)
    with pytest.raises(ValueError, match="one batch"):
        run_sampler("x", enc, Budget(wall_clock_s=1.0), 0,
                    lambda bqm, reads, seed: (None, None), one_batch=True)
