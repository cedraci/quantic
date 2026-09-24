import pytest

from quantic.problem.risk import RiskSpec, VarianceRisk


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
