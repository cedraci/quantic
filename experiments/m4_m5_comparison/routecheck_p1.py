"""Physical qubits touched by the routed FakeFez circuit, reps 1 against reps 2, for the
instances whose noise-model runs were killed and a few that ran."""

import sys
from pathlib import Path

from quantic.bench.angles import AngleCache, angle_key
from quantic.bench.instances import InstanceCache
from quantic.encoding.encode import encode
from quantic.solvers.quantum.backends import fake_backend, route, routed_stats
from quantic.solvers.quantum.ising import qubo_hash, to_ising
from quantic.solvers.qubo.sampler import STUDY_ZETA_BITS, encoding_named

sys.stdout.reconfigure(encoding="utf-8")

CACHES = {2: AngleCache(Path(sys.argv[1])), 1: AngleCache(Path(sys.argv[2]))}
cache = InstanceCache()
backend = fake_backend("ibm_fez")
for dials, seed in [
    ("D1+D2", 2030),
    ("D1+D2", 2027),
    ("D1+D2", 2028),
    ("D2+D4", 2029),
    ("D2", 2030),
    ("none", 2027),
    ("D1+D2", 2026),
]:
    inst = cache.get("T0", dials, seed)
    enc = encode(inst, encoding_named("binary"), n_zeta_bits=STUDY_ZETA_BITS)
    for reps in (2, 1):
        a = CACHES[reps].get(angle_key(qubo_hash(enc.qubo), reps, 0.25, 200))
        if a is None:
            print(dials, seed, reps, "no angles")
            continue
        routed = route(to_ising(enc.qubo), a, backend)
        depth, two = routed_stats(routed)
        active = {
            q
            for op in routed.data
            if op.operation.name not in ("barrier", "measure")
            for q in op.qubits
        }
        print(
            f"{dials:9} {seed} reps {reps} logical {enc.n_vars:2d} depth {depth:5d} 2q {two:5d} "
            f"physical touched {len(active)}",
            flush=True,
        )
