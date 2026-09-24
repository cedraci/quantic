"""The synthetic generator's volatility must be the volatility it claims.

Review finding 4.1. The generator injected impact as ``Y * sigma_bucket *
sign(f) * |f|**delta`` where ``sigma_bucket`` was the *configured*
``daily_vol / sqrt(buckets_per_day)``, but scaled the diffusion term by
``noise_frac`` without compensating. Realised volatility therefore only
equalled the configured value at ``noise_frac ~ 1.0``:

    bundle    noise_frac   configured   realised    est/configured
    calib     0.05         0.005547     0.001131    0.21x
    bench25   1.0          0.005547     0.005671    0.95x

``estimate_bucket_sigma`` was correct in both cases -- it measures realised
volatility to within 2%. The fiction was ``ground_truth()["sigma_bucket"]``.

Because calibration divides by the sigma it measures, and the generator
multiplied by the sigma it configured, ``Y`` was recoverable only where the
two agreed -- at ``noise_frac ~ 1.0``, which is precisely where ``delta`` is
not recoverable. The two could not be validated in the same bundle.
"""

import numpy as np
import polars as pl
import pytest

from quantic.core.session import SYNTH_SESSION
from quantic.data.synth import SynthConfig, generate_buckets, ground_truth

BUCKET_NS = SYNTH_SESSION.length_ns // 13


def _realised_sigma(cfg: SynthConfig) -> dict[str, float]:
    buckets = generate_buckets(cfg)
    out = {}
    for symbol in cfg.symbols:
        sub = buckets.filter(pl.col("symbol") == symbol)
        r = (sub["mid_close"] / sub["mid_open"] - 1.0).to_numpy()
        out[symbol] = float(np.std(r, ddof=1))
    return out


@pytest.mark.parametrize("noise_frac", [0.05, 0.3, 1.0])
def test_realised_volatility_equals_the_configured_sigma_at_every_noise_level(noise_frac):
    cfg = SynthConfig(
        symbols=("SYNA", "SYNB", "SYNC"), n_days=120, buckets_per_day=13,
        seed=5, noise_frac=noise_frac,
    )
    target = ground_truth(cfg)["sigma_bucket"]

    for symbol, realised in _realised_sigma(cfg).items():
        assert realised == pytest.approx(target, rel=0.10), (
            f"{symbol}: realised {realised:.6f} vs configured {target:.6f} "
            f"at noise_frac={noise_frac}"
        )


def test_ground_truth_reports_the_y_that_is_actually_injected():
    """`impact_Y` must be the coefficient calibration will recover, not a wish."""
    cfg = SynthConfig(symbols=("SYNA",), n_days=20, buckets_per_day=13, noise_frac=0.05)
    gt = ground_truth(cfg)

    assert "impact_Y_configured" in gt, "the requested shape is kept for traceability"
    # Holding total variance fixed while the impact term keeps its own scale
    # means the injected Y is renormalised; ground truth must say so.
    assert gt["impact_Y"]["SYNA"] != gt["impact_Y_configured"]["SYNA"]


def test_delta_is_unaffected_by_the_renormalisation():
    """Renormalising scales impact by a constant, which moves Y but not delta."""
    for noise_frac in (0.05, 1.0):
        cfg = SynthConfig(symbols=("SYNA",), n_days=20, noise_frac=noise_frac)
        assert ground_truth(cfg)["impact_delta"]["SYNA"] == 0.45


def test_an_unphysical_configuration_is_rejected_rather_than_silently_rescaled():
    """noise_frac=0 means the mid is pure impact with no diffusion at all."""
    with pytest.raises(ValueError, match="noise_frac"):
        SynthConfig(symbols=("SYNA",), n_days=2, noise_frac=0.0)
    with pytest.raises(ValueError, match="noise_frac"):
        SynthConfig(symbols=("SYNA",), n_days=2, noise_frac=-0.1)
