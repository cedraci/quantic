"""compare_to_l2 must stay fast enough for real L3 to be worth attempting.

Review finding 4.3 / I9. The comparison filtered the symbol's whole L2 frame
inside the per-timestamp loop, and `snapshots_at` allocated a dict per message
via `iter_rows(named=True)`. Measured on 6,500 snapshots across 25 symbols:
23.4s, or 278 snapshots/s. One liquid US name is O(10^6) messages/day, so that
had to go before real L3 arrives.

Grouping the L2 frame once and streaming messages as tuples gives 13,773
snapshots/s on the same fixture -- 49x, 23.4s to 0.47s.

This asserts a throughput floor rather than a scaling ratio. A ratio between
two small timings is dominated by fixed overheads and flakes; a floor with 4x
of headroom does not, and it fails outright on either regression.
"""

import time

from quantic.data.synth import SynthConfig, generate_bundle
from quantic.micro.validation import compare_to_l2

# Measured 13,773 snapshots/s after the fix, ~1,000/s before it on this
# machine and 278/s in the review's 25-symbol measurement. 3,000 leaves 4.5x
# of headroom for a slower machine while still failing on either regression.
MIN_SNAPSHOTS_PER_SECOND = 3_000


def test_comparison_sustains_a_usable_snapshot_throughput(tmp_path):
    cfg = SynthConfig(
        symbols=tuple(f"SYN{i:02d}" for i in range(10)),
        n_days=20,
        buckets_per_day=13,
        seed=4,
        depth_levels=6,
    )
    bundle = generate_bundle(tmp_path / "perf", cfg)
    l3, l2 = bundle.l3(), bundle.l2()
    snapshots = l2.select("symbol", "ts_ns").unique().height

    # Best of three. A single timing on a contended machine measures the
    # scheduler, not the algorithm: this floor failed spuriously while the
    # repo was running concurrent agents, and passed 3/3 unloaded moments
    # later. The comparison is deterministic and read-only, so repeating it
    # is free of side effects, and taking the best run still fails outright
    # on a real regression -- the fix this guards was a 49x change, not a 2x
    # one.
    elapsed = float("inf")
    mismatches: list = []
    for _ in range(3):
        start = time.perf_counter()
        mismatches = compare_to_l2(l3, l2, levels=cfg.depth_levels)
        elapsed = min(elapsed, time.perf_counter() - start)

    assert mismatches == []
    throughput = snapshots / elapsed
    assert throughput > MIN_SNAPSHOTS_PER_SECOND, (
        f"{throughput:.0f} snapshots/s over {snapshots} snapshots "
        f"({elapsed:.2f}s, best of 3), below the {MIN_SNAPSHOTS_PER_SECOND} floor: "
        "either the "
        "per-timestamp filter over the whole L2 frame is back, or replay is "
        "allocating per message again"
    )
