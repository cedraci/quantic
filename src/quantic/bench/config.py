"""An experiment, read from TOML and expanded into the runs it asks for.

    [instances]
    tiers = ["T0", "T1"]
    dials = "all"                 # or a list of labels such as ["none", "D1+D3"]
    seeds = [2026, 2027]

    [references]
    qubo_minimum_tiers = ["T0"]   # where the QUBO's minimum is enumerated

    [solvers.simulated_annealing]
    encodings = ["binary", "one_hot", "domain_wall"]
    zeta_bits = 8
    seeds = "0..49"               # inclusive range, or a list
    budgets_s = [0.1, 1, 10, 60]
    max_samples = 1000            # optional, applies to every budget

A solver name is one of ``SOLVERS`` or an import path ``"package.module:Class"``
whose class takes ``seed`` (and ``encoding`` and ``zeta_bits`` when the
section names encodings). Unknown keys are refused: a misspelt key silently
ignored is a study that did not run what its author thinks it ran.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from quantic.problem.dials import Dials

SOLVERS = (
    "almgren_chriss", "cpsat", "exact_pwl_milp",
    "simulated_annealing", "tabu", "greedy",
)
QUBO_SOLVERS = ("simulated_annealing", "tabu", "greedy")

_TOP = {"instances", "references", "solvers"}
_INSTANCES = {"tiers", "dials", "seeds"}
_REFERENCES = {"qubo_minimum_tiers"}
_SOLVER = {"encodings", "zeta_bits", "seeds", "budgets_s", "max_samples"}


class ConfigError(ValueError):
    """The experiment file does not say what it means."""


@dataclass(frozen=True)
class SolverPlan:
    name: str
    encodings: tuple[str | None, ...]
    zeta_bits: int | None
    seeds: tuple[int, ...]
    budgets_s: tuple[float, ...]
    max_samples: int | None


@dataclass(frozen=True)
class Experiment:
    name: str
    tiers: tuple[str, ...]
    dials: tuple[str, ...]
    seeds: tuple[int, ...]
    qubo_minimum_tiers: tuple[str, ...]
    solvers: tuple[SolverPlan, ...]


def _unknown(where: str, got: dict, allowed: set[str]) -> None:
    extra = sorted(set(got) - allowed)
    if extra:
        raise ConfigError(f"unknown key(s) {extra} in {where}; allowed: {sorted(allowed)}")


def _seeds(value, where: str) -> tuple[int, ...]:
    if isinstance(value, str):
        lo, sep, hi = value.partition("..")
        if not sep:
            raise ConfigError(f"{where}: a seed range is written 'a..b', got {value!r}")
        return tuple(range(int(lo), int(hi) + 1))
    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return tuple(value)
    raise ConfigError(f"{where}: seeds are a list of integers or 'a..b', got {value!r}")


def _is_qubo(name: str) -> bool:
    return name in QUBO_SOLVERS


def load(path: Path) -> Experiment:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    _unknown("the file", raw, _TOP)
    inst = raw.get("instances", {})
    _unknown("[instances]", inst, _INSTANCES)

    every = tuple(d.label for d in Dials.combinations(include_cvar=True))
    dials = inst.get("dials", "all")
    if dials == "all":
        dials = every
    elif not isinstance(dials, list) or not set(dials) <= set(every):
        raise ConfigError(f"[instances] dials must be 'all' or labels from {every}")

    refs = raw.get("references", {})
    _unknown("[references]", refs, _REFERENCES)

    plans = []
    for name, body in raw.get("solvers", {}).items():
        where = f"[solvers.{name}]"
        _unknown(where, body, _SOLVER)
        if name not in SOLVERS and ":" not in name:
            raise ConfigError(f"{where}: unknown solver; known: {SOLVERS}")
        qubo = _is_qubo(name) or "encodings" in body
        if qubo and not body.get("encodings"):
            raise ConfigError(f"{where}: a QUBO sampler needs its encodings")
        if not qubo and "encodings" in body:
            raise ConfigError(f"{where}: only QUBO samplers take encodings")
        budgets = tuple(float(b) for b in body.get("budgets_s", []))
        if not budgets or any(b <= 0 for b in budgets):
            raise ConfigError(f"{where}: budgets_s must list positive seconds")
        plans.append(SolverPlan(
            name=name,
            encodings=tuple(body["encodings"]) if qubo else (None,),
            zeta_bits=int(body.get("zeta_bits", 8)) if qubo else None,
            seeds=_seeds(body.get("seeds", [0]), where),
            budgets_s=budgets,
            max_samples=body.get("max_samples"),
        ))
    if not plans:
        raise ConfigError("the experiment names no solver")

    return Experiment(
        name=Path(path).stem,
        tiers=tuple(inst.get("tiers", ["T0"])),
        dials=tuple(dials),
        seeds=_seeds(inst.get("seeds", [2026]), "[instances]"),
        qubo_minimum_tiers=tuple(refs.get("qubo_minimum_tiers", [])),
        solvers=tuple(plans),
    )
