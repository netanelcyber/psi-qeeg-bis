import numpy as np

from calm_helmet.calibration import (apply_calibration, fnirs_reliability, simulate_vft,
                                     vft_metrics)
from calm_helmet.config import Config


def test_delayed_response_has_later_centroid():
    typical = vft_metrics(simulate_vft(delay_s=0, seed=1))
    delayed = vft_metrics(simulate_vft(delay_s=20, persistence=0.8, seed=1))
    assert delayed["centroid_s"] > typical["centroid_s"] + 8


def test_flat_response_has_low_snr_and_zero_fnirs_weight():
    flat = vft_metrics(simulate_vft(amplitude=0.0, seed=2))
    assert fnirs_reliability(flat) == 0.0
    cfg = apply_calibration(Config(), flat)
    assert all(f.weight == 0 for f in cfg.features if f.modality == "fnirs")
    assert all(f.weight > 0 for f in cfg.features if f.modality == "eeg")


def test_good_response_keeps_full_fnirs_weight():
    good = vft_metrics(simulate_vft(amplitude=1.5, noise=0.1, seed=3))
    assert fnirs_reliability(good) == 1.0
    base = {f.name: f.weight for f in Config().features}
    cfg = apply_calibration(Config(), good)
    assert all(f.weight == base[f.name] for f in cfg.features)


def test_laterality_sign():
    left = simulate_vft(amplitude=0.4, seed=4)
    right = simulate_vft(amplitude=1.2, seed=5)
    m = vft_metrics((left + right) / 2, hbo_left=left, hbo_right=right)
    assert m["laterality"] > 0.2
    assert np.isfinite(m["integral"])
