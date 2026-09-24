import numpy as np
import pytest

from quantic.micro.impact.base import ImpactParams
from quantic.problem.params import AssetParams, MarketParams

BUCKET_NS = 1800 * 1_000_000_000


def _asset(symbol="AAA", **kw) -> AssetParams:
    base = dict(
        symbol=symbol, delta=0.5, y_coef=0.8, gamma=0.1,
        sigma_bucket=0.005, bucket_volume_shares=400_000.0,
        price=100.0, half_spread=0.01,
    )
    base.update(kw)
    return AssetParams(**base)


def _params(n=2, cov=None) -> MarketParams:
    assets = tuple(_asset(f"S{i}", price=100.0 + i) for i in range(n))
    if cov is None:
        cov = np.eye(n) * 1e-5
    return MarketParams(assets=assets, covariance=cov, bucket_ns=BUCKET_NS)


def test_shape_and_symbols():
    p = _params(3)
    assert p.n_assets == 3
    assert p.symbols == ("S0", "S1", "S2")


def test_prices_are_returned_in_asset_order():
    assert _params(3).prices().tolist() == [100.0, 101.0, 102.0]


def test_impact_params_builds_micros_type_with_the_bucket_horizon_attached():
    """This is the only place micro.ImpactParams is constructed."""
    ip = _params().impact_params(0)
    assert isinstance(ip, ImpactParams)
    assert ip.symbol == "S0"
    assert ip.sigma_bucket == 0.005
    assert ip.bucket_ns == BUCKET_NS
    assert ip.bucket_volume_shares == 400_000.0
    assert ip.price == 100.0


def test_price_covariance_converts_returns_to_price_changes():
    """Sigma_price = diag(p) @ Sigma_returns @ diag(p).

    Left implicit this is a silent error of price**2 -- four orders of
    magnitude at a $100 stock -- and entirely plausible in the output.
    """
    cov = np.array([[1e-4, 2e-5], [2e-5, 9e-5]])
    p = MarketParams(
        assets=(_asset("A", price=10.0), _asset("B", price=50.0)),
        covariance=cov, bucket_ns=BUCKET_NS,
    )
    expected = np.array([[1e-4 * 100, 2e-5 * 500], [2e-5 * 500, 9e-5 * 2500]])
    assert np.allclose(p.price_covariance(), expected)


def test_price_covariance_stays_symmetric():
    assert np.allclose(_params(3).price_covariance(), _params(3).price_covariance().T)


def test_subset_keeps_the_leading_assets_and_the_matching_covariance_block():
    full = _params(4)
    sub = full.subset(2)
    assert sub.n_assets == 2
    assert sub.symbols == ("S0", "S1")
    assert sub.covariance.shape == (2, 2)
    assert np.allclose(sub.covariance, full.covariance[:2, :2])


def test_subset_rejects_asking_for_more_assets_than_exist():
    with pytest.raises(ValueError, match="4 assets"):
        _params(4).subset(9)


# --- rejections: each is a silent-wrongness source, not a nicety -----------


def test_a_non_symmetric_covariance_is_rejected():
    cov = np.array([[1e-4, 2e-5], [7e-5, 9e-5]])
    with pytest.raises(ValueError, match="symmetric"):
        MarketParams(assets=(_asset("A"), _asset("B")), covariance=cov, bucket_ns=BUCKET_NS)


def test_a_non_psd_covariance_is_rejected():
    """A non-PSD covariance makes the risk term unbounded below.

    Every optimum downstream is then meaningless, so it is refused at
    construction rather than discovered as a suspiciously good solution.
    """
    cov = np.array([[1e-4, 5e-4], [5e-4, 1e-4]])   # eigenvalue -4e-4
    with pytest.raises(ValueError, match="positive semi-definite|PSD"):
        MarketParams(assets=(_asset("A"), _asset("B")), covariance=cov, bucket_ns=BUCKET_NS)


def test_a_covariance_of_the_wrong_size_is_rejected():
    with pytest.raises(ValueError, match="shape|dimension"):
        MarketParams(assets=(_asset("A"),), covariance=np.eye(3) * 1e-5, bucket_ns=BUCKET_NS)


def _with_smallest_eigenvalue(delta: float) -> np.ndarray:
    """A symmetric 2x2 at 1e-5 scale whose smallest eigenvalue is exactly -delta."""
    cov = np.eye(2) * 1e-5
    cov[0, 1] = cov[1, 0] = 1e-5 + delta
    return cov


def test_rounding_scale_negative_eigenvalues_are_tolerated():
    """Ledoit-Wolf output is PSD only up to floating-point noise.

    -1e-16 against a 2e-5 largest eigenvalue is 5e-12 relative, well inside
    the 1e-10 relative floor.
    """
    cov = _with_smallest_eigenvalue(1e-16)
    assert np.linalg.eigvalsh(cov).min() < 0, "fixture is not actually non-PSD"
    MarketParams(assets=(_asset("A"), _asset("B")), covariance=cov, bucket_ns=BUCKET_NS)


def test_a_genuinely_negative_eigenvalue_is_rejected_even_when_small():
    """The regression that matters.

    -1e-12 against a 2e-5 largest eigenvalue is 5e-8 relative -- four orders
    of magnitude above rounding noise, and a real direction in which the risk
    term is unbounded below. An absolute -1e-10 floor would accept it.
    """
    cov = _with_smallest_eigenvalue(1e-12)
    with pytest.raises(ValueError, match="positive semi-definite|PSD"):
        MarketParams(assets=(_asset("A"), _asset("B")), covariance=cov, bucket_ns=BUCKET_NS)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("delta", 0.0, "delta"),
        ("delta", 1.5, "delta"),
        ("y_coef", -0.1, "y_coef"),
        ("gamma", -0.1, "gamma"),
        ("sigma_bucket", -0.1, "sigma_bucket"),
        ("bucket_volume_shares", 0.0, "bucket_volume_shares"),
        ("price", 0.0, "price"),
        ("half_spread", -0.01, "half_spread"),
    ],
)
def test_out_of_range_asset_params_are_rejected(field, value, match):
    with pytest.raises(ValueError, match=match):
        _asset(**{field: value})


def test_a_non_positive_bucket_ns_is_rejected():
    with pytest.raises(ValueError, match="bucket_ns"):
        MarketParams(assets=(_asset("A"),), covariance=np.eye(1) * 1e-5, bucket_ns=0)


def test_duplicate_symbols_are_rejected():
    """Two rows for one symbol means the covariance rows cannot be trusted."""
    with pytest.raises(ValueError, match="duplicate"):
        MarketParams(
            assets=(_asset("A"), _asset("A")), covariance=np.eye(2) * 1e-5,
            bucket_ns=BUCKET_NS,
        )


def test_no_assets_is_rejected():
    with pytest.raises(ValueError, match="at least one asset"):
        MarketParams(assets=(), covariance=np.zeros((0, 0)), bucket_ns=BUCKET_NS)
