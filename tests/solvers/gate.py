"""The M3b gate's checks, as functions the gate and its mutation tests both call.

A check that has never been seen failing is a check nobody knows works. The
gate runs these over the whole ladder; the mutation tests run the same
functions over one instance with a defect put in, and assert on which check
reports it.

Every function returns ``(findings, record)``. ``findings`` maps a check's
name to what it found wrong and is empty when nothing is. ``record`` is what
was measured, for the result document.
"""

from __future__ import annotations

import pytest

from quantic.encoding.decode import FEASIBLE, decode
from quantic.encoding.encode import encode
from quantic.problem.risk import CVaRRisk
from quantic.solvers.base import OPTIMAL
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.exact_pwl_milp import exact_pwl_milp, permanent_constant
from quantic.solvers.classical.qubo_minimum import qubo_minimum
from quantic.solvers.classical.scoring import score

# The width at which the parent gate is stated on the cells that carry CVaR,
# and the width the study runs at. Design sections 3.2 and 4.
GATE_ZETA_BITS = 20
STUDY_ZETA_BITS = 8

# Two evaluations of one schedule, or of two schedules of equal cost.
REL = 1e-9
# HiGHS works to 1e-7 in feasibility; this is that, with room.
BRACKET_RTOL = 1e-6
# A dense z'Qz at a dynamic range of 2.4e12 carries about 7e-8.
DENSE_RTOL = 1e-6

# Design section 8: how many of the sixteen combinations are covered over
# every state, under each encoding.
EVERY_STATE_CASES = {"binary": 16, "domain_wall": 10, "one_hot": 5}


def classical(instance):
    """Checks 1 to 4: the two exact methods, against each other."""
    findings: dict[str, object] = {}
    try:
        by_hand = brute_force(instance)
        by_milp = exact_pwl_milp(instance)
    except Exception as exc:  # the gate reports; it does not filter
        return {"solves": repr(exc)}, None

    for name, result in (("brute_force", by_hand), ("exact_pwl_milp", by_milp)):
        if result.status != OPTIMAL:
            findings["solves"] = (name, result.status)
        elif not result.feasibility.feasible:
            findings["solves"] = (name, dict(result.feasibility.violations))
    if findings:
        return findings, None

    if by_milp.objective.total != pytest.approx(by_hand.objective.total, rel=REL):
        findings["methods_agree"] = (by_milp.objective.total, by_hand.objective.total)

    bracket = by_milp.bracket
    if bracket.width > BRACKET_RTOL * max(1.0, abs(bracket.upper)):
        findings["bracket_closes"] = (bracket.lower, bracket.upper)

    constant = permanent_constant(instance)
    if by_milp.objective.permanent_impact_cost != pytest.approx(constant, rel=REL):
        findings["permanent_is_constant"] = (
            by_milp.objective.permanent_impact_cost, constant
        )

    record = {
        "instance": instance.instance_id,
        "optimum": by_hand.objective.total,
        "n_feasible": by_hand.detail["n_feasible"],
        "brute_force_s": by_hand.timings["total_s"],
        "milp_s": by_milp.timings["total_s"],
        "milp_rounds": by_milp.detail["rounds"],
        "milp_binaries": by_milp.detail["n_binaries"],
        "presolve_retries": by_milp.detail["presolve_retries"],
        "bracket_width": bracket.width,
    }
    return findings, record


def encoded(instance, encoding, optimum: float, *, n_zeta_bits: int):
    """Checks 5 to 8: the QUBO's minimum, against the exact optimum."""
    width = n_zeta_bits if isinstance(instance.risk, CVaRRisk) else STUDY_ZETA_BITS
    try:
        enc = encode(instance, encoding, n_zeta_bits=width)
        found = qubo_minimum(enc)
        sample = decode(enc, found.bits)
    except Exception as exc:  # the gate reports; it does not filter
        return {"minimises": repr(exc)}, None

    findings: dict[str, object] = {}
    regret = None
    if sample.classification != FEASIBLE:
        findings["minimum_is_feasible"] = (
            sample.classification,
            None if sample.report is None else dict(sample.report.violations),
        )
    else:
        total = score(instance, sample.schedule)[0].total
        regret = total - optimum
        if total != pytest.approx(optimum, rel=REL):
            findings["minimum_is_the_optimum"] = (total, optimum)

    dense = enc.qubo.energy(found.bits)
    if found.energy != pytest.approx(dense, rel=DENSE_RTOL):
        findings["energy_is_the_matrix"] = (found.energy, dense)

    record = {
        "instance": instance.instance_id,
        "encoding": encoding.name,
        "zeta_bits": width if enc.gadget is not None else None,
        "n_vars": enc.n_vars,
        "scope": found.scope,
        "n_states": found.n_states,
        "n_scanned": found.n_scanned,
        "widest_window": found.widest_window,
        "n_ties": found.n_ties,
        "regret": regret,
        "resolution": None if enc.gadget is None else enc.gadget.resolution,
        "cost_gap": None if enc.gadget is None else enc.gadget.cost_gap,
        "dense_error": abs(found.energy - dense) / max(1.0, abs(dense)),
    }
    return findings, record
