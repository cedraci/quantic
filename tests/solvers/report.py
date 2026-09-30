"""Print the tables the M3b gate result document carries.

    python -m tests.solvers.report

Runs the gate's own checks over T0, and the exact program over T1, and prints
what was measured as Markdown. Nothing is asserted here: the gate does that.
"""

from __future__ import annotations

import time

from quantic.encoding.integer import ENCODINGS
from quantic.problem.risk import CVaRRisk
from quantic.solvers.base import Budget
from quantic.solvers.classical.exact_pwl_milp import exact_pwl_milp
from tests.ladder import tier
from tests.solvers.gate import GATE_ZETA_BITS, STUDY_ZETA_BITS, classical, encoded
from tests.solvers.test_t1_bracket import BUDGET_S


def _table(header: tuple[str, ...], rows: list[tuple]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    started = time.perf_counter()
    exact, at_gate, at_study, findings = [], [], [], []

    for instance in tier("T0"):
        found, record = classical(instance)
        findings += [(instance.instance_id, "-", k, v) for k, v in found.items()]
        if record is None:
            continue
        exact.append((
            f"`{record['instance']}`", record["n_feasible"],
            f"{record['optimum']:.6f}", f"{record['brute_force_s']:.2f}",
            f"{record['milp_s']:.2f}", record["milp_rounds"],
            record["milp_binaries"], record["presolve_retries"],
            f"{record['bracket_width']:.1e}",
        ))
        for encoding in ENCODINGS:
            found, row = encoded(
                instance, encoding, record["optimum"], n_zeta_bits=GATE_ZETA_BITS
            )
            findings += [
                (instance.instance_id, encoding.name, k, v) for k, v in found.items()
            ]
            if row is not None:
                at_gate.append((
                    f"`{row['instance']}`", row["encoding"], row["n_vars"],
                    row["scope"], row["n_states"], row["n_scanned"],
                    row["widest_window"], row["n_ties"],
                    "-" if row["regret"] is None else f"{row['regret']:.1e}",
                    f"{row['dense_error']:.1e}",
                ))
            if not isinstance(instance.risk, CVaRRisk):
                continue
            _, row = encoded(
                instance, encoding, record["optimum"], n_zeta_bits=STUDY_ZETA_BITS
            )
            if row is not None and encoding is ENCODINGS[0]:
                at_study.append((
                    f"`{row['instance']}`", f"{record['optimum']:.4f}",
                    f"{row['regret']:.4f}",
                    f"{100.0 * row['regret'] / record['optimum']:.2f}%",
                    f"{row['resolution']:.2f}",
                ))
    gate_s = time.perf_counter() - started

    started = time.perf_counter()
    t1 = []
    for instance in tier("T1"):
        result = exact_pwl_milp(instance, budget=Budget(wall_clock_s=BUDGET_S))
        t1.append((
            f"`{instance.instance_id}`", result.status,
            f"{result.bracket.upper:.6f}", f"{result.bracket.width:.1e}",
            result.detail["rounds"], result.detail["n_binaries"],
            result.detail["presolve_retries"], f"{result.timings['total_s']:.1f}",
        ))
    t1_s = time.perf_counter() - started

    print(f"## Findings\n\n{len(findings)} across every check.\n")
    for row in findings:
        print(f"- {row}")
    print("\n## The exact optimum, two ways (T0)\n")
    print(_table(
        ("instance", "feasible schedules", "optimum", "brute force, s", "MILP, s",
         "rounds", "binaries", "presolve retries", "bracket width"), exact,
    ))
    print(f"\n## The QUBO's minimum at {GATE_ZETA_BITS} zeta bits (T0)\n")
    print(_table(
        ("instance", "encoding", "variables", "scope", "states", "scanned",
         "widest window", "ties", "regret", "dense error"), at_gate,
    ))
    print(f"\n## Regret at the study's {STUDY_ZETA_BITS} zeta bits (T0, CVaR cells)\n")
    print(_table(
        ("instance", "optimum", "regret", "regret / optimum", "declared resolution"),
        at_study,
    ))
    print("\n## The exact program at T1\n")
    print(_table(
        ("instance", "status", "upper", "bracket width", "rounds", "binaries",
         "presolve retries", "seconds"), t1,
    ))
    print(f"\nGate checks: {gate_s:.0f} s. T1: {t1_s:.0f} s.")


if __name__ == "__main__":
    main()
