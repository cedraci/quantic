"""THE M3a GATE -- the encoder is the objective, and can be shown so with no solver.

For every instance the ladder produces, under every encoding that can be
materialised at that tier, the QUBO's energy at an encoded feasible schedule
equals the reference objective term by term. Two pieces are inexact by
construction -- the concave-impact approximand under the log encoding, and the
Rockafellar-Uryasev gadget under every encoding -- and each is compared against
the function the encoder declares it implements, with its own asserted bound.
"""

import json

import numpy as np
import pytest

from quantic.encoding.decode import decode
from quantic.encoding.diagnostics import (
    DENSE_EMBEDDING_CEILING,
    dynamic_range_ratio,
    variable_budget,
)
from quantic.encoding.encode import encode, encode_solution, reference_breakdown
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.problem.generator import generate_ladder
from quantic.problem.objective import cvar_of, optimal_zeta
from quantic.problem.params import AssetParams, MarketParams
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule

BUCKET_NS = 1800 * 1_000_000_000
SMALL = {"n_raw": 300, "n_target": 15}
REL = 1e-9

# T3 under a table encoding names thousands of bits, and a dense term of that
# size runs to hundreds of megabytes. Those cells are covered by the variable
# budget instead -- which is the artifact showing they cannot reach the QPU.
MATERIALISABLE = {"T0": 3, "T1": 3, "T2": 3, "T3": 1}


def _params_for(tier):
    rng = np.random.default_rng(tier.n_assets)
    a = rng.normal(size=(tier.n_assets, tier.n_assets))
    return MarketParams(
        assets=tuple(
            AssetParams(f"S{i:02d}", 0.5, 0.8, 0.1, 0.005, 400_000.0, 100.0 + i, 0.01)
            for i in range(tier.n_assets)
        ),
        covariance=(a @ a.T) / tier.n_assets * 1e-5,
        bucket_ns=BUCKET_NS,
    )


@pytest.fixture(scope="module")
def ladder():
    return generate_ladder(_params_for, seed=2026, include_cvar=True, **SMALL)


def _cases(ladder):
    for instance in ladder:
        for encoding in ENCODINGS[: MATERIALISABLE[instance.tier]]:
            yield instance, encoding


def _zeta_for(encoded, schedule):
    if encoded.gadget is None:
        return None
    return encoded.gadget.zeta.snap(optimal_zeta(encoded.instance, schedule))


def _expected_total(encoded, schedule, zeta):
    """The reference, with the gadget's declared CVaR swapped in when present."""
    reference = reference_breakdown(encoded, schedule, zeta=zeta)
    if encoded.gadget is None:
        return reference.total
    instance = encoded.instance
    continuous = instance.risk.lam * cvar_of(
        instance.risk.scenarios, instance, schedule,
        alpha=instance.risk.alpha, zeta=zeta,
    )
    losses = encoded.gadget.losses_from(instance, encoded.layout, schedule)
    aux = encoded.gadget.optimal_aux(losses, zeta)
    declared = encoded.gadget.encoded_cvar(losses, zeta, aux)
    return reference.total - continuous + declared


@pytest.fixture(scope="module")
def gate(ladder):
    """One pass over every case, collecting each check's failures.

    Encoding is by far the expensive part -- at T2 and T3 a single instance
    carries thousands of variables and six dense terms -- so the whole gate
    encodes each case exactly once and accumulates findings per check. The
    named tests below then assert on those findings, so a failure still says
    which instance, which encoding and which check, without re-encoding
    anything. Only the small finding dicts outlive the loop, so peak memory
    stays at one instance.
    """
    findings: dict[str, dict] = {
        name: {}
        for name in (
            "encodes", "total_energy", "term_energy", "penalty_zero",
            "one_lot", "round_trip", "substitution", "cvar_resolution",
            "dynamic_range", "describe",
        )
    }

    for instance, encoding in _cases(ladder):
        tag = (instance.instance_id, encoding.name)
        try:
            encoded = encode(instance, encoding)
        except Exception as exc:  # noqa: BLE001 - the gate reports, it does not filter
            findings["encodes"][tag] = repr(exc)
            continue

        zeta = _zeta_for(encoded, instance.witness)
        bits = encode_solution(encoded, instance.witness, zeta=zeta)

        energy = encoded.qubo.energy(bits)
        expected = _expected_total(encoded, instance.witness, zeta)
        if energy != pytest.approx(expected, rel=REL):
            findings["total_energy"][tag] = (energy, expected)

        if not isinstance(instance.risk, CVaRRisk):
            reference = reference_breakdown(encoded, instance.witness)
            for name, qubo, want in (
                ("spread", encoded.breakdown.spread, reference.spread_cost),
                ("temporary", encoded.breakdown.temporary,
                 reference.temporary_impact_cost),
                ("permanent", encoded.breakdown.permanent,
                 reference.permanent_impact_cost),
                ("risk", encoded.breakdown.risk, reference.risk_cost),
            ):
                got = qubo.energy(bits)
                if got != pytest.approx(want, rel=REL):
                    findings["term_energy"][(*tag, name)] = (got, want)

        # Algebraically the penalty residual is exactly zero at a feasible
        # point. Qubo.energy evaluates it as a dense z'Qz product, so what
        # survives is double-precision cancellation: measured at ~1e-13
        # against strengths of ~24-62, i.e. ~5e-15 relative. The bound below
        # is scaled to the strength rather than being a magic absolute, and
        # it cannot hide a real breach: derive_penalties floors every
        # strength at 1.0 and every residual is an integer square, so the
        # cheapest genuine violation costs a whole strength -- nine orders
        # of magnitude above this bound, six above the noise.
        paid = encoded.penalties.energy(bits)
        floor = 1e-9 * max(encoded.strengths.full_liquidation, 1.0)
        if abs(paid) >= floor:
            findings["penalty_zero"][tag] = (paid, floor)

        if not (instance.dials.discrete_participation or instance.dials.block_trades):
            grid = instance.witness.as_array().copy()
            cell = np.argwhere(grid > 0)[0]
            grid[cell[0], cell[1]] -= 1
            short = Schedule.from_array(grid)
            bad = encoded.qubo.energy(
                encode_solution(encoded, short, zeta=_zeta_for(encoded, short))
            )
            if not bad > energy:
                findings["one_lot"][tag] = (bad, energy)

        sample = decode(encoded, bits)
        if sample.schedule != instance.witness or sample.classification != "feasible":
            findings["round_trip"][tag] = sample.classification

        approximand = encoded.breakdown.approximand
        if approximand is not None:
            ok = (
                encoding.name == "binary"
                and instance.dials.concave_impact
                and np.isfinite(approximand.max_abs_error)
                and approximand.max_abs_error >= 0.0
            )
            if not ok:
                findings["substitution"][tag] = "unexpected approximand"
        elif instance.dials.concave_impact and not encoding.supports_exact_table:
            findings["substitution"][tag] = "concave impact silently unapproximated"

        if encoded.gadget is not None:
            if not (np.isfinite(encoded.gadget.resolution)
                    and encoded.gadget.resolution > 0.0):
                findings["substitution"][tag] = "gadget reports no resolution"
            losses = encoded.gadget.losses_from(
                instance, encoded.layout, instance.witness
            )
            aux = encoded.gadget.optimal_aux(losses, zeta)
            declared = encoded.gadget.encoded_cvar(losses, zeta, aux)
            continuous = instance.risk.lam * cvar_of(
                instance.risk.scenarios, instance, instance.witness,
                alpha=instance.risk.alpha, zeta=zeta,
            )
            if abs(declared - continuous) > encoded.gadget.resolution:
                findings["cvar_resolution"][tag] = (
                    abs(declared - continuous), encoded.gadget.resolution
                )

        ratio = dynamic_range_ratio(encoded.qubo)
        if not (np.isfinite(ratio) and ratio >= 1.0):
            findings["dynamic_range"][tag] = ratio

        try:
            if json.loads(json.dumps(encoded.describe()))["encoding"] != encoding.name:
                findings["describe"][tag] = "encoding name missing"
        except (TypeError, ValueError) as exc:
            findings["describe"][tag] = repr(exc)

    return findings


def test_the_ladder_is_still_sixteen_combinations_on_four_tiers(ladder):
    assert len(ladder) == 16 * 4 == 64


def test_every_instance_encodes_under_every_materialisable_encoding(gate):
    """THE GATE, part one: nothing in the ladder defeats the encoder."""
    assert not gate["encodes"], gate["encodes"]


def test_the_energy_at_the_witness_equals_the_reference_objective(gate):
    """THE GATE, part two. The load-bearing assertion of the whole milestone."""
    assert not gate["total_energy"], gate["total_energy"]


def test_each_objective_term_matches_on_its_own(gate):
    """A single scalar says nothing about which term is wrong -- objective.py's rule."""
    assert not gate["term_energy"], gate["term_energy"]


def test_the_penalty_block_is_exactly_zero_at_every_witness(gate):
    """A feasible point pays nothing. Exactly nothing."""
    assert not gate["penalty_zero"], gate["penalty_zero"]


def test_a_one_lot_violation_raises_the_total_energy(gate):
    """The property derived penalties exist to guarantee."""
    assert not gate["one_lot"], gate["one_lot"]


def test_every_witness_decodes_back_to_itself(gate):
    assert not gate["round_trip"], gate["round_trip"]


def test_every_declared_substitution_reports_a_finite_bound(gate):
    """The approximand and the gadget say what they cost, and mean it."""
    assert not gate["substitution"], gate["substitution"]


def test_the_declared_cvar_stays_within_its_reported_resolution(gate):
    """The second of the gadget's two assertions, across the whole ladder."""
    assert not gate["cvar_resolution"], gate["cvar_resolution"]


def test_every_encoded_instance_records_a_dynamic_range(gate):
    """Section 6.4: no result is interpretable without it."""
    assert not gate["dynamic_range"], gate["dynamic_range"]


def test_the_encoding_description_is_json_safe_for_every_instance(gate):
    assert not gate["describe"], gate["describe"]


def test_the_variable_budget_covers_every_cell_of_the_study(ladder):
    """THE GATE, part three: which cells can reach Advantage2 at all.

    Counts keys rather than building matrices, so it covers all 64 instances
    under all three encodings -- including the T3 table encodings no dense
    matrix could hold.
    """
    rows = variable_budget(ladder)
    assert len(rows) == 64 * len(ENCODINGS)
    assert {r.tier for r in rows} == {"T0", "T1", "T2", "T3"}
    assert {r.encoding for r in rows} == {e.name for e in ENCODINGS}
    assert all(r.n_vars > 0 for r in rows)
    fitting = [r for r in rows if r.fits_dense_embedding]
    assert fitting, "no cell of the study fits a dense embedding"
    assert all(r.n_vars <= DENSE_EMBEDDING_CEILING for r in fitting)


def test_encoding_is_reproducible_across_calls(ladder):
    """A results row pins the QUBO; two runs must agree bit for bit."""
    for instance in ladder[:8]:
        a = encode(instance, BinaryEncoding())
        b = encode(instance, BinaryEncoding())
        assert np.array_equal(a.qubo.Q, b.qubo.Q)
        assert a.qubo.constant == b.qubo.constant
