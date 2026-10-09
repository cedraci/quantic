"""One task, run in a worker, turned into the rows the store keeps.

A solver that raises is a row with ``store_status = "error"`` and the
exception's type and message; the study carries on. A solver that has to be
killed never reaches this module -- the pool writes that row.
"""

from __future__ import annotations

import importlib
import json
import platform
from collections import OrderedDict
from importlib import metadata
from pathlib import Path

import numpy as np

from quantic.bench.angles import AngleCache, MissingAngles, angle_key
from quantic.bench.instances import InstanceCache
from quantic.bench.keys import (
    REF_BRACKET,
    REF_BRUTE_FORCE,
    REF_QAOA_ANGLES,
    REF_QUBO_MINIMUM,
    CodeVersion,
    Task,
)
from quantic.bench.store import ERROR, OK, blank_row
from quantic.encoding.decode import FEASIBLE, decode
from quantic.encoding.diagnostics import dynamic_range_ratio
from quantic.encoding.encode import EncodedInstance, encode
from quantic.problem.instance import Instance
from quantic.solvers.base import Budget, Result
from quantic.solvers.classical.almgren_chriss import AlmgrenChrissSolver
from quantic.solvers.classical.brute_force import brute_force
from quantic.solvers.classical.cpsat import CpSat
from quantic.solvers.classical.exact_pwl_milp import ExactPwlMilp, exact_pwl_milp
from quantic.solvers.classical.qubo_minimum import qubo_minimum
from quantic.solvers.classical.scoring import score
from quantic.solvers.quantum.ising import qubo_hash, to_ising
from quantic.solvers.quantum.qaoa import QAOA_CLASSES
from quantic.solvers.quantum.train import train
from quantic.solvers.qubo.greedy import Greedy
from quantic.solvers.qubo.sampler import QuboSampler, encoding_named
from quantic.solvers.qubo.simulated_annealing import SimulatedAnnealing
from quantic.solvers.qubo.tabu import Tabu

_ENCODED_KEPT = 8
_PACKAGES = ("quantic", "numpy", "scipy", "ortools", "dwave-samplers", "dimod", "duckdb",
             "qiskit", "qiskit-aer", "qiskit-ibm-runtime")
_QUBO = {"simulated_annealing": SimulatedAnnealing, "tabu": Tabu, "greedy": Greedy}


def machine() -> str:
    return f"{platform.node()}|{platform.machine()}|{platform.processor()}"


def packages() -> dict[str, str]:
    return {name: metadata.version(name) for name in _PACKAGES}


def _solver(task: Task):
    if task.solver in _QUBO:
        return _QUBO[task.solver](task.encoding, seed=task.solver_seed,
                                  zeta_bits=task.zeta_bits)
    if task.solver == "almgren_chriss":
        return AlmgrenChrissSolver()
    if task.solver == "cpsat":
        return CpSat(seed=task.solver_seed)
    if task.solver == "exact_pwl_milp":
        return ExactPwlMilp()
    module, _, name = task.solver.partition(":")
    return getattr(importlib.import_module(module), name)(seed=task.solver_seed)


class Executor:
    """A worker's state: its caches and the pins every row carries."""

    def __init__(self, code: CodeVersion, context: dict | None = None) -> None:
        self.code = code
        self.angles = (AngleCache(Path(context["angles_dir"]))
                       if context and context.get("angles_dir") else None)
        self.where = machine()
        self.versions = json.dumps(packages(), sort_keys=True)
        self.instances = InstanceCache()
        self._encoded: OrderedDict[tuple, EncodedInstance] = OrderedDict()

    def encoded(self, task: Task, instance: Instance) -> EncodedInstance:
        key = (task.tier, task.dials, task.instance_seed, task.encoding, task.zeta_bits)
        if key in self._encoded:
            self._encoded.move_to_end(key)
            return self._encoded[key]
        enc = encode(instance, encoding_named(task.encoding), n_zeta_bits=task.zeta_bits)
        self._encoded[key] = enc
        if len(self._encoded) > _ENCODED_KEPT:
            self._encoded.popitem(last=False)
        return enc

    def run(self, key: str, task: Task) -> tuple[dict, list[tuple], list[tuple]]:
        row = blank_row(key, task, self.code, self.where, self.versions)
        try:
            instance = self.instances.get(task.tier, task.dials, task.instance_seed)
            row["instance_hash"] = instance.content_hash
            row["notional"] = instance.notional()
            return self._run(row, task, instance)
        except Exception as exc:  # the study carries on; the row says what broke
            row["store_status"] = ERROR
            row["error"] = f"{type(exc).__name__}: {exc}"
            return row, [], []

    def _run(self, row, task, instance):
        if task.solver == REF_QAOA_ANGLES:
            return self._train(row, task, instance)
        if task.solver in QAOA_CLASSES:
            return self._qaoa(row, task, instance)
        if task.solver == REF_QUBO_MINIMUM:
            enc = self.encoded(task, instance)
            found = qubo_minimum(enc)
            decoded = decode(enc, found.bits)
            row.update(
                store_status=OK, status="enumerated", qubo_energy=found.energy,
                n_vars=enc.n_vars, dynamic_range=dynamic_range_ratio(enc.qubo),
                detail={"scope": found.scope, "n_states": found.n_states},
            )
            if decoded.classification == FEASIBLE:
                row["qubo_total"] = score(instance, decoded.schedule)[0].total
            return row, [], []

        budget = Budget(wall_clock_s=task.budget_s, max_samples=task.max_samples)
        if task.solver == REF_BRACKET:
            result = exact_pwl_milp(instance)
        elif task.solver == REF_BRUTE_FORCE:
            result = brute_force(instance)
        else:
            solver = _solver(task)
            if isinstance(solver, QuboSampler):
                enc = self.encoded(task, instance)
                row["n_vars"] = enc.n_vars
                row["dynamic_range"] = dynamic_range_ratio(enc.qubo)
                result = solver.solve_encoded(enc, budget)
            else:
                result = solver.solve(instance, budget)
        return self._fill(row, result, instance)

    def _angle_cache(self) -> AngleCache:
        if self.angles is None:
            raise MissingAngles("this worker was started without an angle directory")
        return self.angles

    def _train(self, row, task, instance):
        params = json.loads(task.params)
        enc = self.encoded(task, instance)
        key = angle_key(qubo_hash(enc.qubo), params["reps"], params["alpha"],
                        params["maxiter"])
        cache = self._angle_cache()
        angles = cache.get(key)
        cached = angles is not None
        if angles is None:
            angles = train(enc.qubo, reps=params["reps"], alpha=params["alpha"],
                           maxiter=params["maxiter"])
            cache.put(key, angles, {"qubo_hash": qubo_hash(enc.qubo), **params})
        row.update(
            store_status=OK, status="trained", n_vars=enc.n_vars,
            dynamic_range=dynamic_range_ratio(enc.qubo), angle_key=key,
            reps=params["reps"], total_s=angles.train_s,
            timings={"total_s": angles.train_s},
            detail={**angles.to_json(), "cached": cached,
                    "ising_dynamic_range": to_ising(enc.qubo).dynamic_range},
        )
        return row, [], []

    def _qaoa(self, row, task, instance):
        if task.solver == "qaoa_ibm":
            raise RuntimeError("qaoa_ibm runs in the hardware lane, never in a worker")
        params = json.loads(task.params)
        enc = self.encoded(task, instance)
        key = angle_key(qubo_hash(enc.qubo), params["reps"], params["alpha"],
                        params["maxiter"])
        angles = self._angle_cache().get(key)
        if angles is None:
            raise MissingAngles(f"no trained angles {key[:12]} for {task.solver} on "
                                f"{task.tier}-{task.dials}-{task.instance_seed}")
        sampler = QAOA_CLASSES[task.solver](
            task.encoding, seed=task.solver_seed, zeta_bits=task.zeta_bits,
            angles=angles, shots=task.max_samples, device=params.get("device"),
        )
        row["n_vars"] = enc.n_vars
        row["dynamic_range"] = dynamic_range_ratio(enc.qubo)
        result = sampler.solve_encoded(enc)
        out = fill_row(row, result, instance)
        d = result.detail
        out[0].update(backend=d["backend"], device=d["device"], angle_key=key,
                      reps=params["reps"], shots=d["shots"],
                      routed_depth=d.get("routed_depth"), routed_2q=d.get("routed_2q"))
        return out

    def _fill(self, row, result: Result, instance: Instance):
        return fill_row(row, result, instance)


def fill_row(row, result: Result, instance: Instance):
    row["store_status"] = OK
    row["status"] = result.status
    detail = dict(result.detail)
    samples = detail.pop("samples", ())
    trace = detail.pop("trace", ())
    if result.schedule is not None:
        row["schedule"] = [list(r) for r in result.schedule.lots]
        row["zeta"] = result.zeta
    if result.objective is not None:
        o = result.objective
        row.update(
            spread=o.spread_cost, temporary=o.temporary_impact_cost,
            permanent=o.permanent_impact_cost, risk=o.risk_cost, total=o.total,
            total_bps=o.total * 1e4 / instance.notional(),
        )
    if result.bracket is not None:
        row["lower"] = result.bracket.lower
        row["upper"] = result.bracket.upper
    if result.feasibility is not None:
        row["feasible"] = result.feasibility.feasible
        row["violations"] = dict(result.feasibility.violations)
    if samples:
        row["n_samples"] = detail.get("n_samples")
        row["n_feasible_samples"] = sum(
            s.count for s in samples if s.classification == FEASIBLE
        )
        row["best_energy"] = detail.get("best_energy")
        row["energy_mismatch"] = detail.get("energy_mismatch")
    row["total_s"] = result.timings.get("total_s")
    row["timings"] = dict(result.timings)
    row["detail"] = json.loads(json.dumps(detail, default=_plain))
    return (
        row,
        [(s.digest, s.count, s.energy, s.classification, s.total) for s in samples],
        [tuple(point) for point in trace],
    )


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
