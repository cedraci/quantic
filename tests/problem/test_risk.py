import numpy as np
import pytest

from quantic.problem.risk import CVaRRisk, RiskSpec, VarianceRisk
from quantic.problem.scenarios import ReductionReport, ScenarioSet


def test_variance_risk_carries_a_risk_aversion():
    assert VarianceRisk(lam=1e-6).lam == 1e-6


def test_variance_risk_names_itself():
    assert VarianceRisk(lam=1.0).name == "variance"


def test_variance_risk_satisfies_the_protocol():
    """M2b adds CVaRRisk alongside; Instance.risk is typed to the protocol."""
    assert isinstance(VarianceRisk(lam=1.0), RiskSpec)


def test_a_negative_risk_aversion_is_rejected():
    """Negative lam rewards risk, which makes the objective unbounded below."""
    with pytest.raises(ValueError, match="lam"):
        VarianceRisk(lam=-1.0)


def test_zero_risk_aversion_is_allowed():
    """Switching risk off entirely is a legitimate ablation."""
    assert VarianceRisk(lam=0.0).lam == 0.0


def test_describe_is_json_safe_for_the_instance_hash():
    """Instance.content_hash serialises the risk spec through this.

    M2b's CVaRRisk will hold numpy scenario arrays, so the protocol requires a
    describe() rather than relying on dataclasses.asdict, which would choke on
    them at the point the hash is computed.
    """
    assert VarianceRisk(lam=1e-6).describe() == {"name": "variance", "lam": 1e-6}


def _scenarios(s: int = 4, reduction: ReductionReport | None = None) -> ScenarioSet:
    return ScenarioSet(
        returns=np.zeros((s, 2, 2)),
        weights=np.full(s, 1.0 / s),
        seed=3,
        reduction=reduction,
    )


def test_cvar_risk_carries_its_tail_level_and_scenarios():
    risk = CVaRRisk(lam=1e-6, alpha=0.95, scenarios=_scenarios())
    assert risk.lam == 1e-6
    assert risk.alpha == 0.95
    assert risk.scenarios.n_scenarios == 4


def test_cvar_risk_names_itself():
    assert CVaRRisk(lam=1.0, alpha=0.95, scenarios=_scenarios()).name == "cvar"


def test_cvar_risk_satisfies_the_protocol():
    assert isinstance(CVaRRisk(lam=1.0, alpha=0.95, scenarios=_scenarios()), RiskSpec)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_a_tail_level_outside_the_open_unit_interval_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        CVaRRisk(lam=1.0, alpha=alpha, scenarios=_scenarios())


def test_a_negative_cvar_risk_aversion_is_rejected():
    """Named distinctly from VarianceRisk's equivalent test.

    `tests/problem/test_risk.py` already defines
    `test_a_negative_risk_aversion_is_rejected` for VarianceRisk. Reusing
    that name would silently redefine it -- Python keeps only the second
    definition, so the VarianceRisk check would vanish from the suite with no
    failure. ruff's F811 catches it, which is why the lint gate is a real gate.
    """
    with pytest.raises(ValueError, match="lam"):
        CVaRRisk(lam=-1.0, alpha=0.95, scenarios=_scenarios())


def test_describe_omits_the_returns_array():
    """The content hash serialises this through json.dumps.

    Embedding megabytes of scenario floats in a hash payload would be
    pathological; the seed and the scenario count identify the set.
    """
    payload = CVaRRisk(lam=1e-6, alpha=0.95, scenarios=_scenarios()).describe()
    assert payload["name"] == "cvar"
    assert payload["n_scenarios"] == 4
    assert payload["scenario_seed"] == 3
    assert payload["reduction"] is None
    flat = repr(payload)
    assert "array" not in flat and "ndarray" not in flat


def test_describe_is_json_serialisable():
    import json

    report = ReductionReport(
        n_raw=100, n_reduced=4, wasserstein=0.25,
        cvar_raw=10.0, cvar_reduced=10.5, cvar_discrepancy=0.5,
    )
    payload = CVaRRisk(lam=1e-6, alpha=0.9, scenarios=_scenarios(reduction=report)).describe()
    restored = json.loads(json.dumps(payload, sort_keys=True))
    assert restored["reduction"]["wasserstein"] == 0.25
    assert restored["reduction"]["cvar_discrepancy"] == 0.5


def test_two_cvar_risks_over_the_same_scenarios_compare_and_hash_equal():
    """ScenarioSet's explicit __eq__/__hash__ make the generated ones safe here.

    Without them, a frozen dataclass reaching an ndarray would raise
    ValueError on == and TypeError on hash().
    """
    a = CVaRRisk(lam=1e-6, alpha=0.95, scenarios=_scenarios())
    b = CVaRRisk(lam=1e-6, alpha=0.95, scenarios=_scenarios())
    assert a == b
    assert hash(a) == hash(b)


def test_cvar_risks_over_different_scenarios_compare_unequal():
    a = CVaRRisk(lam=1e-6, alpha=0.95, scenarios=_scenarios(s=4))
    b = CVaRRisk(lam=1e-6, alpha=0.95, scenarios=_scenarios(s=6))
    assert a != b
