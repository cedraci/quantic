"""THE M3b GATE -- the QUBO's minimum is the exact optimum, on T0, sixteen ways.

Parent spec section 14. Two exact methods that share only the lot grid and
the evaluator agree on the optimum of every T0 instance; the minimum of every
QUBO the encoder builds for it decodes to a feasible schedule that costs
exactly that.

The cells that carry CVaR are encoded at twenty zeta bits for this, not the
study's eight. At eight the claim is false on five cells, and the last tests
here measure by how much.
"""

import collections
import dataclasses

import pytest

import quantic.encoding.encode as encode_module
from quantic.encoding.integer import ENCODINGS, BinaryEncoding
from quantic.problem.risk import CVaRRisk
from quantic.solvers.classical import exact_pwl_milp as milp_module
from quantic.solvers.classical.qubo_minimum import DECODABLE, EVERY_STATE
from tests.ladder import tier
from tests.solvers.gate import (
    EVERY_STATE_CASES,
    GATE_ZETA_BITS,
    STUDY_ZETA_BITS,
    classical,
    encoded,
)

# Design section 8: how many combinations are covered over the decodable states only.
DECODABLE_CASES = {"domain_wall": 6, "one_hot": 11}

# Checks 1 to 3: a defect in the exact program must be reported by one of these.
CLASSICAL_CHECKS = {"solves", "methods_agree", "bracket_closes"}

# Where the study's width does not find the optimum. Design section 3.2.
NOT_THE_OPTIMUM_AT_EIGHT_BITS = {
    "T0-D3-2026",
    "T0-D2+D3-2026",
    "T0-D2+D3+D4-2026",
    "T0-D1+D3-2026",
    "T0-D1+D2+D3+D4-2026",
}


@pytest.fixture(scope="module")
def ladder():
    return tier("T0")


@pytest.fixture(scope="module")
def gate(ladder):
    """One pass over every case, collecting each check's findings."""
    findings = collections.defaultdict(dict)
    records = {"classical": [], "encoded": [], "study": []}

    for instance in ladder:
        found, record = classical(instance)
        for check, what in found.items():
            findings[check][instance.instance_id] = what
        if record is None:
            continue
        records["classical"].append(record)

        for encoding in ENCODINGS:
            tag = (instance.instance_id, encoding.name)
            found, row = encoded(
                instance, encoding, record["optimum"], n_zeta_bits=GATE_ZETA_BITS
            )
            for check, what in found.items():
                findings[check][tag] = what
            if row is not None:
                records["encoded"].append(row)

            if not isinstance(instance.risk, CVaRRisk):
                continue
            found, row = encoded(
                instance, encoding, record["optimum"], n_zeta_bits=STUDY_ZETA_BITS
            )
            for check in (
                "minimises", "minimum_is_feasible", "minimum_is_the_optimum",
                "energy_is_the_matrix",
            ):
                if check in found:
                    findings[f"study_{check}"][tag] = found[check]
            if row is not None:
                records["study"].append(row)

    return {"findings": findings, "records": records}


def _none(gate, check):
    found = gate["findings"].get(check, {})
    assert not found, found


def test_the_ladder_is_still_sixteen_combinations(ladder):
    assert len(ladder) == 16
    assert sum(isinstance(i.risk, CVaRRisk) for i in ladder) == 8


def test_both_exact_methods_solve_every_instance_feasibly(gate):
    """THE GATE, check 1."""
    _none(gate, "solves")
    assert len(gate["records"]["classical"]) == 16


def test_the_two_exact_methods_agree_on_the_optimum(gate):
    """THE GATE, check 2. Enumeration and the program share no model."""
    _none(gate, "methods_agree")


def test_the_brackets_close(gate):
    """THE GATE, check 3. The lower bound is proven, the upper attained."""
    _none(gate, "bracket_closes")


def test_permanent_impact_is_the_constant_the_program_carries(gate):
    """THE GATE, check 4."""
    _none(gate, "permanent_is_constant")


def test_every_qubo_is_minimised(gate):
    _none(gate, "minimises")
    assert len(gate["records"]["encoded"]) == 16 * 3


def test_the_minimum_of_every_qubo_is_a_feasible_schedule(gate):
    """THE GATE, check 5. The penalties bind at the minimum, not only nearby."""
    _none(gate, "minimum_is_feasible")


def test_the_minimum_of_every_qubo_is_the_exact_optimum(gate):
    """THE GATE, check 6. Parent spec section 14, and the point of M3."""
    _none(gate, "minimum_is_the_optimum")


def test_the_energy_claimed_is_the_energy_of_the_state_returned(gate):
    """THE GATE, check 7. The closed form against the matrix."""
    _none(gate, "energy_is_the_matrix")


def test_the_scope_of_every_minimum_is_the_one_the_design_records(gate):
    """THE GATE, check 8. Design section 8."""
    every_state = collections.Counter(
        row["encoding"] for row in gate["records"]["encoded"]
        if row["scope"] == EVERY_STATE
    )
    assert dict(every_state) == EVERY_STATE_CASES
    covered = {
        row["instance"] for row in gate["records"]["encoded"]
        if row["scope"] == EVERY_STATE
    }
    assert len(covered) == 16, "a combination is covered over every state nowhere"
    scopes = {row["scope"] for row in gate["records"]["encoded"]}
    assert scopes <= {EVERY_STATE, DECODABLE}, scopes
    decodable = collections.Counter(
        row["encoding"] for row in gate["records"]["encoded"]
        if row["scope"] == DECODABLE
    )
    assert dict(decodable) == DECODABLE_CASES


def test_the_gate_was_stated_at_the_width_the_design_names(gate):
    widths = {row["zeta_bits"] for row in gate["records"]["encoded"]}
    assert widths == {None, GATE_ZETA_BITS}


# -- recorded: the study's width --------------------------------------------


def test_at_the_studys_width_every_minimum_is_still_feasible(gate):
    for check in ("minimises", "minimum_is_feasible", "energy_is_the_matrix"):
        _none(gate, f"study_{check}")
    assert len(gate["records"]["study"]) == 8 * 3


def test_at_the_studys_width_the_regret_is_inside_what_the_gadget_declares(gate):
    """Check 9. The bound is loose and is not the finding; the regret is."""
    outside = {
        (row["instance"], row["encoding"]): (row["regret"], row["resolution"])
        for row in gate["records"]["study"]
        if not -1e-9 <= row["regret"] <= 2.0 * row["resolution"] + row["cost_gap"]
    }
    assert not outside, outside


def test_at_the_studys_width_the_minimum_is_not_the_optimum_on_five_cells(gate):
    """What the study's QUBOs do, written down where it cannot be missed.

    This is also the gate seen failing: at eight bits check 6 reports these
    five cells under every encoding, and no others.
    """
    off = {
        row["instance"] for row in gate["records"]["study"] if row["regret"] > 1e-9
    }
    assert off == NOT_THE_OPTIMUM_AT_EIGHT_BITS
    reported = set(gate["findings"].get("study_minimum_is_the_optimum", {}))
    expected = {
        (name, encoding.name)
        for name in NOT_THE_OPTIMUM_AT_EIGHT_BITS
        for encoding in ENCODINGS
    }
    assert reported == expected, (reported ^ expected)
    for name in off:
        regrets = {
            round(row["regret"], 9) for row in gate["records"]["study"]
            if row["instance"] == name
        }
        assert len(regrets) == 1, (name, regrets)


# -- the gate, seen failing -------------------------------------------------


def _instance(ladder, name):
    return next(i for i in ladder if i.instance_id == name)


def test_the_gate_sees_penalties_too_weak_to_bind(ladder, monkeypatch):
    real = encode_module.derive_penalties

    def weak(*args, **kwargs):
        strengths = real(*args, **kwargs)
        return dataclasses.replace(
            strengths,
            validity=strengths.validity * 1e-4,
            full_liquidation=strengths.full_liquidation * 1e-4,
        )

    instance = _instance(ladder, "T0-none-2026")
    _, record = classical(instance)
    monkeypatch.setattr(encode_module, "derive_penalties", weak)
    found, _ = encoded(
        instance, BinaryEncoding(), record["optimum"], n_zeta_bits=GATE_ZETA_BITS
    )
    assert "minimum_is_feasible" in found


def test_the_gate_sees_a_program_without_cardinality(ladder, monkeypatch):
    real = milp_module.lot_grid
    monkeypatch.setattr(
        milp_module, "lot_grid",
        lambda instance: dataclasses.replace(real(instance), cardinality=None),
    )
    found, _ = classical(_instance(ladder, "T0-D2-2026"))
    # Since M4, Result refuses an OPTIMAL schedule that breaches a constraint, so
    # the sabotage is caught one step earlier: the gate records the refusal, and
    # the refusal names the solver and the constraint it breached.
    assert "solves" in found
    assert "exact_pwl_milp returned one that breaches" in found["solves"]
    assert "'cardinality': " in found["solves"]
    assert "'cardinality': 0.0" not in found["solves"]


def test_the_gate_sees_a_cut_with_the_wrong_sign(ladder, monkeypatch):
    real = milp_module.variance_gradient
    monkeypatch.setattr(
        milp_module, "variance_gradient", lambda i, s: -real(i, s)
    )
    found, _ = classical(_instance(ladder, "T0-none-2026"))
    assert CLASSICAL_CHECKS & set(found), found


def test_the_gate_sees_a_cost_the_program_got_wrong(ladder, monkeypatch):
    real = milp_module.cell_costs

    def shifted(instance, columns, **kwargs):
        costs = real(instance, columns, **kwargs)
        costs[columns.lots > 0] *= 3.0
        return costs

    monkeypatch.setattr(milp_module, "cell_costs", shifted)
    found, _ = classical(_instance(ladder, "T0-D1-2026"))
    assert CLASSICAL_CHECKS & set(found), found
