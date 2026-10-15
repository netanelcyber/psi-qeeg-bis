"""Tests for behavioral event prediction pipeline."""

import unittest
import numpy as np
from psi_qeeg.behavior import (
    BehaviorEvent, validate_events, extract_window_features,
    build_prediction_dataset, theta_gamma_coupling,
    coherence_between, plv_between, MeltdownRiskScorer,
    _welch_bands, _risk_level,
)


def _synth_eeg(n, fs, rng, elevated_theta=False):
    """Minimal synthetic 2-channel EEG."""
    t = np.arange(n) / fs
    channels = []
    for _ in range(2):
        s = np.zeros(n)
        for f in np.arange(1.0, 4.0, 0.5):
            s += rng.uniform(10, 20) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        gain = 15 if elevated_theta else 5
        for f in np.arange(4.0, 8.0, 0.5):
            s += gain * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        for f in np.arange(8.0, 40.0, 2.0):
            s += rng.uniform(0.5, 2) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        channels.append(s)
    return np.column_stack(channels)


class TestBehaviorEvent(unittest.TestCase):
    def test_validate_events_sorted(self):
        events = [
            BehaviorEvent(20.0, 25.0, "repetitive_grooming", "test"),
            BehaviorEvent(10.0, 15.0, "repetitive_grooming", "test"),
        ]
        result = validate_events(events, 60.0)
        self.assertEqual(result[0].onset_s, 10.0)
        self.assertEqual(result[1].onset_s, 20.0)

    def test_validate_rejects_short_event(self):
        events = [BehaviorEvent(10.0, 11.0, "repetitive_grooming", "test")]
        with self.assertRaises(ValueError):
            validate_events(events, 60.0)

    def test_validate_rejects_unknown_type(self):
        events = [BehaviorEvent(10.0, 20.0, "unknown_event", "test")]
        with self.assertRaises(ValueError):
            validate_events(events, 60.0)


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.fs = 256.0
        self.n = round(4 * self.fs)
        self.eeg = _synth_eeg(self.n, self.fs, self.rng)

    def test_welch_bands_keys(self):
        bands = _welch_bands(self.eeg[:, 0], self.fs)
        self.assertIn("delta_power", bands)
        self.assertIn("theta_power", bands)
        self.assertIn("theta_relative", bands)
        self.assertIn("total_power", bands)

    def test_welch_relative_sum_to_one(self):
        bands = _welch_bands(self.eeg[:, 0], self.fs)
        rel_sum = sum(v for k, v in bands.items() if k.endswith("_relative"))
        self.assertAlmostEqual(rel_sum, 1.0, places=5)

    def test_coherence_range(self):
        c = coherence_between(self.eeg[:, 0], self.eeg[:, 1], self.fs, (4, 8))
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_plv_range(self):
        p = plv_between(self.eeg[:, 0], self.eeg[:, 1], self.fs, (4, 8))
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_theta_gamma_coupling_range(self):
        mi = theta_gamma_coupling(self.eeg[:, 0], self.fs)
        self.assertGreaterEqual(mi, 0.0)
        self.assertLessEqual(mi, 1.0)

    def test_extract_window_features(self):
        feats = extract_window_features(self.eeg, self.fs, ["ch1", "ch2"])
        self.assertIn("ch1__delta_power", feats)
        self.assertIn("pair_ch1_ch2__coh_theta", feats)
        self.assertIn("ch1__theta_gamma_mi", feats)


class TestPredictionDataset(unittest.TestCase):
    def test_build_dataset_labels(self):
        rng = np.random.default_rng(123)
        fs = 256.0
        duration = 120.0
        n = round(duration * fs)
        eeg = _synth_eeg(n, fs, rng)
        events = [BehaviorEvent(60.0, 65.0, "repetitive_grooming", "test")]
        samples = build_prediction_dataset(
            eeg, fs, ["ch1", "ch2"], events,
            animal_id="A1", day="d1",
            horizons_s=[10], window_s=4.0, stride_s=2.0,
        )
        self.assertGreater(len(samples), 0)
        labels = {s.label for s in samples}
        self.assertEqual(labels, {0, 1})

    def test_causal_constraint(self):
        rng = np.random.default_rng(456)
        fs = 256.0
        duration = 120.0
        n = round(duration * fs)
        eeg = _synth_eeg(n, fs, rng)
        events = [BehaviorEvent(60.0, 65.0, "repetitive_grooming", "test")]
        samples = build_prediction_dataset(
            eeg, fs, ["ch1", "ch2"], events,
            animal_id="A1", day="d1",
            horizons_s=[30], window_s=4.0, stride_s=2.0,
        )
        for s in samples:
            if s.label == 1:
                self.assertLessEqual(s.window_end_s, 60.0)


class TestRiskLevel(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(_risk_level(0), "low")
        self.assertEqual(_risk_level(24), "low")
        self.assertEqual(_risk_level(25), "moderate")
        self.assertEqual(_risk_level(50), "high")
        self.assertEqual(_risk_level(75), "critical")
        self.assertEqual(_risk_level(100), "critical")


if __name__ == "__main__":
    unittest.main()
