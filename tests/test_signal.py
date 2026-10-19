"""Analytical signal fixtures are synthetic numerical tests, not real EEG."""

import unittest
import numpy as np
from psi_qeeg.signal import spectral_features, phase_locking_value, squared_bicoherence, band_integral


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.fs = 128
        self.t = np.arange(512) / self.fs

    def test_known_sine_power_and_units(self):
        result = spectral_features(10 * np.sin(2*np.pi*10*self.t), self.fs)
        self.assertAlmostEqual(result["alpha_power_uv2"], 50, places=8)
        self.assertAlmostEqual(result["alpha_relative_power"], 1, places=8)

    def test_bands_partition_total(self):
        x = np.random.default_rng(3).normal(size=512)
        result = spectral_features(x, self.fs)
        powers = sum(result[f"{b}_power_uv2"] for b in ("delta", "theta", "alpha", "beta", "gamma"))
        self.assertAlmostEqual(powers, result["total_power_uv2"], places=12)
        self.assertAlmostEqual(sum(result[f"{b}_relative_power"] for b in ("delta", "theta", "alpha", "beta", "gamma")), 1, places=12)

    def test_delta_alpha_mixture(self):
        x = 2*np.sin(2*np.pi*2*self.t) + 4*np.sin(2*np.pi*10*self.t)
        self.assertAlmostEqual(spectral_features(x, self.fs)["delta_alpha_ratio"], 0.25, places=6)

    def test_gamma_stays_in_declared_band(self):
        result = spectral_features(5*np.sin(2*np.pi*35*self.t), self.fs)
        self.assertAlmostEqual(result["gamma_power_uv2"], 12.5, places=8)

    def test_zero_power_ratio_is_missing(self):
        self.assertIsNone(spectral_features(np.zeros(512), self.fs)["delta_alpha_ratio"])

    def test_low_sample_rate_rejected(self):
        with self.assertRaises(ValueError):
            spectral_features(np.ones(512), 64)

    def test_missing_samples_rejected(self):
        x = np.ones(512)
        x[20] = np.nan
        with self.assertRaises(ValueError):
            spectral_features(x, self.fs)

    def test_short_psd_rejected(self):
        with self.assertRaises(ValueError):
            spectral_features(np.ones(10), self.fs)

    def test_out_of_band_integral_rejected(self):
        with self.assertRaises(ValueError):
            band_integral(np.arange(4.), np.ones(4), 0, 4)

    def test_notch_nyquist_rejected(self):
        with self.assertRaises(ValueError):
            spectral_features(np.sin(self.t), self.fs, 64)

    def test_phase_locked_pair(self):
        a = np.sin(2*np.pi*10*self.t)
        b = np.sin(2*np.pi*10*self.t + 0.7)
        self.assertGreater(phase_locking_value(a, b, self.fs), 0.995)

    def test_plv_missing_pair_rejected(self):
        with self.assertRaises(ValueError):
            phase_locking_value(np.ones(512), np.ones(510), self.fs)

    def test_single_fft_bicoherence_not_reported(self):
        self.assertIsNone(squared_bicoherence(np.sin(2*np.pi*10*self.t), self.fs))

    def test_constant_bicoherence_missing(self):
        self.assertIsNone(squared_bicoherence(np.zeros(128*60), self.fs))

    def test_ensemble_bicoherence_bounded(self):
        x = np.random.default_rng(7).normal(size=128*60)
        value = squared_bicoherence(x, self.fs)
        self.assertIsNotNone(value)
        self.assertGreaterEqual(value, 0)
        self.assertLess(value, 0.2)



class ExactSpectrumTests(unittest.TestCase):
    def test_flat_psd_has_exact_band_widths_on_non_aligned_grid(self):
        from psi_qeeg.signal import BANDS
        f = np.arange(0, 65, 0.7)
        for low, high in BANDS.values():
            with self.subTest(band=(low, high)):
                self.assertAlmostEqual(band_integral(f, np.ones_like(f), low, high), high-low, places=12)

    def test_linear_psd_interpolation_matches_analytical_integral(self):
        f = np.arange(0, 65, 0.7)
        low, high = 0.5, 13.2
        expected = (high**2 - low**2)/2 + 2*(high-low)
        self.assertAlmostEqual(band_integral(f, f+2, low, high), expected, places=12)

    def test_boundary_tones_split_power_between_adjacent_bands(self):
        fs = 512
        t = np.arange(4*fs)/fs
        for hz, left, right in ((4,"delta","theta"), (8,"theta","alpha"),
                                 (13,"alpha","beta"), (30,"beta","gamma")):
            with self.subTest(hz=hz):
                result = spectral_features(10*np.sin(2*np.pi*hz*t), fs)
                self.assertAlmostEqual(result[f"{left}_power_uv2"], 25, places=8)
                self.assertAlmostEqual(result[f"{right}_power_uv2"], 25, places=8)

    def test_production_spectrum_integrates_half_hz_endpoint_and_excludes_42_hz(self):
        fs = 512
        t = np.arange(4*fs)/fs
        x = 10*np.sin(2*np.pi*t) + 5*np.sin(2*np.pi*35*t) + 20*np.sin(2*np.pi*42*t)
        result = spectral_features(x, fs)
        # Two-second Hann broadening puts 1/12 of the 1 Hz tone's area below
        # the 0.5 Hz integration bound. The exact endpoint retains 11/12.
        delta = 50 * 11 / 12
        self.assertAlmostEqual(result["delta_power_uv2"], delta, places=8)
        self.assertAlmostEqual(result["gamma_power_uv2"], 12.5, places=8)
        self.assertAlmostEqual(result["total_power_uv2"], delta + 12.5, places=8)

    def test_noninteger_sampling_rate_partitions_total_and_bounds_sef(self):
        fs = 127.3
        x = np.random.default_rng(17).normal(size=510)
        result = spectral_features(x, fs)
        from psi_qeeg.signal import BANDS
        self.assertAlmostEqual(sum(result[f"{b}_relative_power"] for b in BANDS), 1, places=12)
        self.assertGreaterEqual(result["sef95_hz"], 0.5)
        self.assertLessEqual(result["sef95_hz"], 40)
