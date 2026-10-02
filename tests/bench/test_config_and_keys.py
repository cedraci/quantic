import multiprocessing as mp
from pathlib import Path

import pytest

from quantic.bench.config import ConfigError, load
from quantic.bench.instances import GATE_SEED, InstanceCache, tier_instances
from quantic.bench.keys import (
    REF_BRACKET,
    REF_BRUTE_FORCE,
    REF_QUBO_MINIMUM,
    REFERENCE,
    RUN,
    CodeVersion,
    Task,
    code_version,
    expand,
)
from tests.ladder import gate_ladder

ROOT = Path(__file__).resolve().parents[2]
CODE = CodeVersion(commit="c0ffee", lock_hash="10c4", dirty=False)

GOOD = """
[instances]
tiers = ["T0", "T1"]
dials = ["none", "D1+D3"]
seeds = [2026, 2027]

[references]
qubo_minimum_tiers = ["T0"]

[solvers.cpsat]
seeds = [0, 1]
budgets_s = [1, 10]

[solvers.tabu]
encodings = ["binary", "domain_wall"]
seeds = "0..2"
budgets_s = [0.5]
"""


def _load(tmp_path: Path, text: str):
    path = tmp_path / "exp.toml"
    path.write_text(text, encoding="utf-8")
    return load(path)


def test_a_config_reads_into_an_experiment(tmp_path):
    exp = _load(tmp_path, GOOD)
    assert exp.name == "exp"
    assert exp.tiers == ("T0", "T1")
    assert exp.seeds == (2026, 2027)
    tabu = next(p for p in exp.solvers if p.name == "tabu")
    assert tabu.seeds == (0, 1, 2)
    assert tabu.encodings == ("binary", "domain_wall")
    assert tabu.zeta_bits == 8
    cpsat = next(p for p in exp.solvers if p.name == "cpsat")
    assert cpsat.encodings == (None,)
    assert cpsat.zeta_bits is None


def test_all_dials_means_the_sixteen_combinations(tmp_path):
    exp = _load(tmp_path, GOOD.replace('dials = ["none", "D1+D3"]', 'dials = "all"'))
    assert len(exp.dials) == 16


@pytest.mark.parametrize(
    "edit, message",
    [
        (("seeds = [2026, 2027]", "seed = [2026]"), "unknown key"),
        (('budgets_s = [0.5]', 'budgets_s = [0]'), "positive"),
        (('encodings = ["binary", "domain_wall"]\n', ""), "encodings"),
        (("[solvers.cpsat]", "[solvers.cplex]"), "unknown solver"),
        (('dials = ["none", "D1+D3"]', 'dials = ["D5"]'), "dials"),
        (('seeds = "0..2"', 'seeds = "0-2"'), "a..b"),
        # seeds: integers only, at least one, each once, ranges that run upwards
        (("seeds = [0, 1]", "seeds = [true, 1]"), "seeds"),
        (("seeds = [2026, 2027]", "seeds = []"), "seeds"),
        (('seeds = "0..2"', 'seeds = "5..2"'), "5..2"),
        (('seeds = "0..2"', 'seeds = "a..b"'), "a..b"),
        (('seeds = "0..2"', 'seeds = "0..2.5"'), "a..b"),
        (("seeds = [0, 1]", "seeds = [0, 0]"), r"\[0\] more than once"),
        (("seeds = [2026, 2027]", "seeds = [2026, 2026]"), "more than once"),
        # budgets: each once
        (("budgets_s = [1, 10]", "budgets_s = [1, 1.0]"), "more than once"),
        (("budgets_s = [1, 10]", 'budgets_s = [1, "10"]'), "positive"),
        # tiers: a list of names the ladder knows; QUBO minima only on those tiers
        (('tiers = ["T0", "T1"]', 'tiers = "T0"'), "tiers"),
        (('tiers = ["T0", "T1"]', 'tiers = ["T0", "T9"]'), "tiers"),
        (('tiers = ["T0", "T1"]', "tiers = []"), "tiers"),
        (('tiers = ["T0", "T1"]', 'tiers = ["T0", "T0"]'), "more than once"),
        (('qubo_minimum_tiers = ["T0"]', 'qubo_minimum_tiers = ["T2"]'), "qubo_minimum_tiers"),
        (('qubo_minimum_tiers = ["T0"]', 'qubo_minimum_tiers = "T0"'), "qubo_minimum_tiers"),
        # max_samples and zeta_bits: positive integers where given
        (('budgets_s = [0.5]', 'budgets_s = [0.5]\nmax_samples = 0'), "max_samples"),
        (('budgets_s = [0.5]', 'budgets_s = [0.5]\nmax_samples = 1.5'), "max_samples"),
        (('budgets_s = [0.5]', 'budgets_s = [0.5]\nmax_samples = true'), "max_samples"),
        (('budgets_s = [0.5]', 'budgets_s = [0.5]\nzeta_bits = 0'), "zeta_bits"),
        (('budgets_s = [0.5]', 'budgets_s = [0.5]\nzeta_bits = "8"'), "zeta_bits"),
        # a coordinate listed twice is the same task twice: one key, two rows
        (('dials = ["none", "D1+D3"]', 'dials = ["none", "none"]'), "more than once"),
        (('encodings = ["binary", "domain_wall"]', 'encodings = ["binary", "binary"]'),
         "more than once"),
    ],
)
def test_a_config_that_does_not_say_what_it_means_is_refused(tmp_path, edit, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, GOOD.replace(*edit))


def test_a_config_at_the_edges_of_strictness_still_reads(tmp_path):
    exp = _load(tmp_path, GOOD.replace('seeds = "0..2"', 'seeds = "2..2"').replace(
        "budgets_s = [0.5]", "budgets_s = [0.5]\nmax_samples = 1\nzeta_bits = 1"))
    tabu = next(p for p in exp.solvers if p.name == "tabu")
    assert (tabu.seeds, tabu.max_samples, tabu.zeta_bits) == ((2,), 1, 1)


def test_a_config_with_no_solver_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no solver"):
        _load(tmp_path, "[instances]\ntiers = ['T0']\n")


def test_expansion_puts_references_first_and_covers_the_grid(tmp_path):
    tasks = expand(_load(tmp_path, GOOD))
    kinds = [t.kind for t in tasks]
    assert kinds == sorted(kinds, key=lambda k: k != REFERENCE)
    refs = [t for t in tasks if t.kind == REFERENCE]
    runs = [t for t in tasks if t.kind == RUN]
    n_instances = 2 * 2 * 2
    assert sum(t.solver == REF_BRACKET for t in refs) == n_instances
    assert sum(t.solver == REF_BRUTE_FORCE for t in refs) == n_instances // 2  # T0 only
    assert {(t.tier, t.encoding) for t in refs if t.solver == REF_QUBO_MINIMUM} == {
        ("T0", "binary"), ("T0", "domain_wall"),
    }
    assert len(runs) == n_instances * (2 * 2 + 2 * 3 * 1)
    assert len({t.key(CODE) for t in tasks}) == len(tasks)


def test_a_key_changes_with_the_code_and_with_any_coordinate():
    task = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="cpsat",
                solver_seed=0, budget_s=1.0)
    other_code = CodeVersion(commit="beef", lock_hash="10c4", dirty=False)
    assert task.key(CODE) != task.key(other_code)
    assert task.key(CODE) != Task(**{**task.__dict__, "budget_s": 2.0}).key(CODE)
    assert task.key(CODE) == Task(**task.__dict__).key(CODE)


def _key_in_child(queue):
    task = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="cpsat")
    queue.put(task.key(CODE))


def test_a_key_is_the_same_in_another_process():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=_key_in_child, args=(queue,))
    child.start()
    child.join()
    task = Task(kind=RUN, tier="T0", dials="none", instance_seed=1, solver="cpsat")
    assert queue.get(timeout=10) == task.key(CODE)


def test_the_code_version_names_this_checkout():
    code = code_version(ROOT)
    assert len(code.commit) == 40
    assert len(code.lock_hash) == 64


def test_the_gate_seed_gives_the_m3_ladder_exactly():
    for tier in ("T0", "T1"):
        mine = tier_instances(tier, GATE_SEED)
        theirs = tuple(i for i in gate_ladder() if i.tier == tier)
        assert [i.content_hash for i in mine] == [i.content_hash for i in theirs]


def test_the_cache_builds_a_tier_once_and_refuses_an_unknown_label():
    cache = InstanceCache()
    first = cache.get("T0", "none", GATE_SEED)
    assert cache.get("T0", "none", GATE_SEED) is first
    with pytest.raises(KeyError, match="D9"):
        cache.get("T0", "D9", GATE_SEED)
