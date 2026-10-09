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

    [quantum]                     # required by any qaoa_* solver
    reps = 2
    shots = 1000
    device = "ibm_fez"            # must have a fake twin
    quota_cap_s = 300             # never above 300

    [solvers.qaoa_ideal]          # also qaoa_noisy, qaoa_ibm; no budgets_s
    encodings = ["binary"]
    seeds = "0..4"

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
from quantic.problem.liquidation import LADDER

QUBO_SOLVERS = ("simulated_annealing", "tabu", "greedy")
QUANTUM_SOLVERS = ("qaoa_ideal", "qaoa_noisy", "qaoa_ibm")
SOLVERS = (
    "almgren_chriss", "cpsat", "exact_pwl_milp",
    "simulated_annealing", "tabu", "greedy", *QUANTUM_SOLVERS,
)

_TOP = {"instances", "references", "solvers", "quantum"}
_QUANTUM_SOLVER = {"encodings", "zeta_bits", "seeds"}
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
    quantum: QuantumSettings | None = None


@dataclass(frozen=True)
class QuantumSettings:
    reps: int = 2
    alpha: float = 0.25
    maxiter: int = 200
    shots: int = 1000
    max_qubits: int = 28
    noisy_max_qubits: int = 20
    gate_shots: int = 32
    device: str = "ibm_fez"
    quota_cap_s: float = 300.0
    per_job_cap_s: float = 60.0
    min_job_s: float = 10.0
    workers: int = 2
    train_deadline_s: float = 14400.0
    run_deadline_s: float = 7200.0


def _unknown(where: str, got: dict, allowed: set[str]) -> None:
    extra = sorted(set(got) - allowed)
    if extra:
        raise ConfigError(f"unknown key(s) {extra} in {where}; allowed: {sorted(allowed)}")


def _once(values, where: str, name: str) -> None:
    """A value listed twice is one task listed twice: the same key, written twice."""
    twice = sorted({v for v in values if list(values).count(v) > 1})
    if twice:
        raise ConfigError(f"{where}: {name} lists {twice} more than once")


def _seeds(value, where: str) -> tuple[int, ...]:
    if isinstance(value, str):
        lo, sep, hi = value.partition("..")
        try:
            first, last = int(lo), int(hi)
        except ValueError:
            first = last = None
        if not sep or first is None:
            raise ConfigError(
                f"{where}: a seed range is written 'a..b' with integers a <= b, got {value!r}"
            )
        if last < first:
            raise ConfigError(f"{where}: the seed range {value!r} is empty; write 'a..b', a <= b")
        return tuple(range(first, last + 1))
    # type(v) is int: TOML's true and false are bools, and a bool is an int to isinstance.
    if isinstance(value, list) and value and all(type(v) is int for v in value):
        _once(value, where, "seeds")
        return tuple(value)
    raise ConfigError(
        f"{where}: seeds are a non-empty list of integers or 'a..b', got {value!r}"
    )


def _positive_int(body: dict, name: str, where: str) -> int | None:
    value = body.get(name)
    if value is not None and (type(value) is not int or value <= 0):
        raise ConfigError(f"{where}: {name} must be a positive integer, got {value!r}")
    return value


def _tiers(value) -> tuple[str, ...]:
    known = [t.name for t in LADDER]
    if (not isinstance(value, list) or not value
            or not all(isinstance(t, str) and t in known for t in value)):
        raise ConfigError(
            f"[instances] tiers must be a non-empty list of tier names from {known}, "
            f"got {value!r}"
        )
    _once(value, "[instances]", "tiers")
    return tuple(value)


def _quantum(body: dict) -> QuantumSettings:
    from dataclasses import fields

    from quantic.solvers.quantum.backends import fake_backend
    from quantic.solvers.quantum.quota import QUOTA_CEILING_S

    names = {f.name: f for f in fields(QuantumSettings)}
    _unknown("[quantum]", body, set(names))
    values = {}
    for name, value in body.items():
        default = names[name].default
        if isinstance(default, str):
            if not isinstance(value, str):
                raise ConfigError(f"[quantum] {name} must be a string, got {value!r}")
        elif isinstance(default, int) and not isinstance(default, bool) \
                and not isinstance(default, float):
            if type(value) is not int or value < 1:
                raise ConfigError(f"[quantum] {name} must be a positive integer, got {value!r}")
        elif type(value) not in (int, float) or value <= 0:
            raise ConfigError(f"[quantum] {name} must be a positive number, got {value!r}")
        values[name] = float(value) if isinstance(default, float) else value
    settings = QuantumSettings(**values)
    if not 0.0 < settings.alpha <= 1.0:
        raise ConfigError(f"[quantum] alpha must be in (0, 1], got {settings.alpha}")
    if settings.quota_cap_s > QUOTA_CEILING_S:
        raise ConfigError(
            f"[quantum] quota_cap_s {settings.quota_cap_s:g} is above the "
            f"{QUOTA_CEILING_S:g} s ceiling (M5 decision D4)"
        )
    if settings.min_job_s > settings.per_job_cap_s:
        raise ConfigError("[quantum] min_job_s must not exceed per_job_cap_s")
    try:
        fake_backend(settings.device)
    except KeyError as exc:
        raise ConfigError(f"[quantum] device: {exc.args[0]}") from exc
    return settings


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
    _once(dials, "[instances]", "dials")
    tiers = _tiers(inst.get("tiers", ["T0"]))

    refs = raw.get("references", {})
    _unknown("[references]", refs, _REFERENCES)
    qubo_tiers = refs.get("qubo_minimum_tiers", [])
    if not isinstance(qubo_tiers, list) or not set(qubo_tiers) <= set(tiers):
        raise ConfigError(
            f"[references] qubo_minimum_tiers must list tiers this experiment runs, "
            f"{list(tiers)}; got {qubo_tiers!r}"
        )

    plans = []
    for name, body in raw.get("solvers", {}).items():
        where = f"[solvers.{name}]"
        if name in QUANTUM_SOLVERS:
            _unknown(where, body, _QUANTUM_SOLVER)
            if not body.get("encodings"):
                raise ConfigError(f"{where}: a QAOA solver needs its encodings")
            _once(body["encodings"], where, "encodings")
            zeta_bits = _positive_int(body, "zeta_bits", where)
            plans.append(SolverPlan(
                name=name, encodings=tuple(body["encodings"]),
                zeta_bits=8 if zeta_bits is None else zeta_bits,
                seeds=_seeds(body.get("seeds", [0]), where),
                budgets_s=(), max_samples=None,
            ))
            continue
        _unknown(where, body, _SOLVER)
        if name not in SOLVERS and ":" not in name:
            raise ConfigError(f"{where}: unknown solver; known: {SOLVERS}")
        qubo = _is_qubo(name) or "encodings" in body
        if qubo and not body.get("encodings"):
            raise ConfigError(f"{where}: a QUBO sampler needs its encodings")
        if not qubo and "encodings" in body:
            raise ConfigError(f"{where}: only QUBO samplers take encodings")
        if qubo:
            _once(body["encodings"], where, "encodings")
        raw_budgets = body.get("budgets_s", [])
        if (not isinstance(raw_budgets, list) or not raw_budgets
                or any(type(b) not in (int, float) or b <= 0 for b in raw_budgets)):
            raise ConfigError(f"{where}: budgets_s must list positive seconds")
        budgets = tuple(float(b) for b in raw_budgets)
        _once(budgets, where, "budgets_s")
        zeta_bits = _positive_int(body, "zeta_bits", where)
        plans.append(SolverPlan(
            name=name,
            encodings=tuple(body["encodings"]) if qubo else (None,),
            zeta_bits=(8 if zeta_bits is None else zeta_bits) if qubo else None,
            seeds=_seeds(body.get("seeds", [0]), where),
            budgets_s=budgets,
            max_samples=_positive_int(body, "max_samples", where),
        ))
    quantum = None
    if "quantum" in raw:
        quantum = _quantum(raw["quantum"])
    if any(p.name in QUANTUM_SOLVERS for p in plans) and quantum is None:
        raise ConfigError("a QAOA solver needs a [quantum] section")
    if not plans:
        raise ConfigError("the experiment names no solver")

    return Experiment(
        name=Path(path).stem,
        tiers=tiers,
        dials=tuple(dials),
        seeds=_seeds(inst.get("seeds", [2026]), "[instances]"),
        qubo_minimum_tiers=tuple(qubo_tiers),
        solvers=tuple(plans),
        quantum=quantum,
    )
