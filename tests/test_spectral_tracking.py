"""Independent numerical/Monte Carlo checks, using synthetic signals only."""

from dataclasses import replace
import unittest

import numpy as np
from scipy import signal

from psi_qeeg.signal import welch_psd, band_integral
from psi_qeeg.spectral_tracking import (
    SpectralConfig, SpectralExtractor, NoiseCalibration, fit_fft_noise,
    effective_welch_averages, KalmanCUSUM,
)


def known_calibration(r, lag=None):
    p = len(r)
    cfg = SpectralConfig(128, tuple(f"C{i}" for i in range(p)))
    return NoiseCalibration(cfg, tuple(f"C{i}:log_beta_alpha" for i in range(p)),
                            np.zeros(p), np.ones(p), np.asarray(r),
                            np.zeros((p, p)) if lag is None else np.asarray(lag),
                            np.ones((p, 5)) * 20, np.zeros(p), np.ones(p), 1024, 0, 768, 10)


class SpectralMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SpectralConfig(128, ("F3", "F4"), frontal_pair=("F3", "F4"))
        self.extractor = SpectralExtractor(self.cfg)

    def test_known_sines_power_log_ratio_and_faa(self):
        t = np.arange(256) / 128
        x = np.array([10 * np.sin(2*np.pi*10*t) + 3 * np.sin(2*np.pi*20*t),
                      20 * np.sin(2*np.pi*10*t) + 3 * np.sin(2*np.pi*20*t)])
        m = self.extractor.extract(x)
        np.testing.assert_allclose(m.band_power[:, 2], [50, 200], atol=1e-10)
        np.testing.assert_allclose(m.band_power[:, 3], [4.5, 4.5], atol=1e-10)
        np.testing.assert_allclose(m.values, [np.log(4.5/50), np.log(4.5/200), np.log(4)], atol=1e-10)

    def test_one_sided_density_matches_scipy_even_and_odd_fft(self):
        x = np.random.default_rng(2).normal(size=(2, 256))
        for nfft in (128, 255, 512):
            with self.subTest(nfft=nfft):
                f, p = welch_psd(x, 128, 128, 64, nfft)
                expected_f, expected = signal.welch(x, fs=128, window=signal.windows.hann(128, sym=False),
                                                     nperseg=128, noverlap=64, nfft=nfft)
                np.testing.assert_array_equal(f, expected_f)
                np.testing.assert_allclose(p, expected, rtol=1e-13)

    def test_quadrature_partitions_and_matches_existing_band_integral(self):
        x = np.random.default_rng(3).normal(size=(2, 256))
        m = self.extractor.extract(x)
        for ch in range(2):
            expected = band_integral(self.extractor.frequencies, m.psd[ch], 0.5, 45)
            self.assertAlmostEqual(m.band_power[ch].sum(), expected, places=13)

    def test_overlapping_hann_segments_are_not_independent_averages(self):
        self.assertAlmostEqual(effective_welch_averages(self.cfg), 81/28)
        self.assertEqual(effective_welch_averages(replace(self.cfg, overlap=0)), 2)

    def test_redundant_features_and_unmeasured_faa_rejected(self):
        with self.assertRaisesRegex(ValueError, "redundant"):
            SpectralExtractor(replace(self.cfg, measurements=("F3:log_alpha", "F3:log_beta", "F3:log_beta_alpha")))
        with self.assertRaisesRegex(ValueError, "Measurement"):
            SpectralExtractor(replace(self.cfg, frontal_pair=None, measurements=("FAA",)))

    def test_bad_geometry_or_nonfinite_data_rejected(self):
        for kwargs in ({"hop_s": 3}, {"segment_s": 3}, {"overlap": 1}, {"nfft": 8}, {"hop_s": 0.123}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.cfg, **kwargs)
        x = np.ones((2, 256))
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.extractor.extract(x)
        with self.assertRaises(ValueError):
            welch_psd(np.ones(256, dtype=complex), 128, 128)


class FFTNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = SpectralConfig(128, ("C3", "C4"))
        cls.extractor = SpectralExtractor(cls.cfg)
        rng = np.random.default_rng(19)
        baseline = rng.normal(size=(2, 128*120))
        baseline[1] = 0.85*baseline[0] + np.sqrt(1-0.85**2)*baseline[1]
        cls.baseline = baseline * 8
        cls.calibration = fit_fft_noise(cls.baseline, cls.extractor, draws=2048, seed=7)

    def test_covariance_matches_independent_holdout_simulation(self):
        rng = np.random.default_rng(109)
        x = rng.normal(size=(4096, 2, 256))
        x[:, 1] = 0.85*x[:, 0] + np.sqrt(1-0.85**2)*x[:, 1]
        measured = self.extractor.extract(x*8).values
        empirical = np.cov(measured, rowvar=False)
        np.testing.assert_allclose(np.diag(self.calibration.R_fft), np.diag(empirical), rtol=0.18)
        correlation = self.calibration.R_fft[0, 1] / np.sqrt(np.prod(np.diag(self.calibration.R_fft)))
        self.assertAlmostEqual(correlation, np.corrcoef(measured, rowvar=False)[0, 1], delta=0.08)
        self.assertGreater(correlation, 0.5)

    def test_overlap_lag_and_stationary_ar_covariance(self):
        c = self.calibration
        self.assertTrue(np.all(np.diag(c.lag1_fft) / np.diag(c.R_fft) > 0.15))
        r, a, q = c.colored_noise()
        np.testing.assert_allclose(a @ r @ a.T + q, r, atol=1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(q).min(), -1e-12)
        self.assertLess(max(abs(np.linalg.eigvals(a))), 1)

    def test_colored_baseline_covariance_matches_independent_ar_noise(self):
        rng = np.random.default_rng(82)
        x = signal.lfilter([1], [1, -.9], rng.normal(size=(2, 128*120 + 512)), axis=-1)[:, 512:]*5
        fitted = fit_fft_noise(x, self.extractor, draws=1024, seed=17)
        heldout = signal.lfilter([1], [1, -.9], rng.normal(size=(4096, 2, 768)), axis=-1)[..., -256:]*5
        empirical = np.cov(self.extractor.extract(heldout).values, rowvar=False)
        np.testing.assert_allclose(np.diag(fitted.R_fft), np.diag(empirical), rtol=.2)

    def test_log_ratio_variance_exceeds_naive_independent_bin_formula(self):
        # 3 overlapping segments; 5 alpha Hz and 17 beta Hz at 1-Hz spacing.
        naive = 1/(3*5) + 1/(3*17)
        self.assertGreater(self.calibration.R_fft[0, 0], 1.3*naive)

    def test_zero_padding_does_not_create_independent_information(self):
        padded = SpectralExtractor(replace(self.cfg, nfft=512))
        c0 = fit_fft_noise(self.baseline, self.extractor, draws=512, seed=2)
        c1 = fit_fft_noise(self.baseline, padded, draws=512, seed=2)
        # Same Gaussian realizations, different integration grid. Fourfold
        # zero padding must not produce a fourfold reduction of R.
        np.testing.assert_allclose(np.diag(c1.R_fft), np.diag(c0.R_fft), rtol=0.12)

    def test_seed_and_normalization(self):
        a = fit_fft_noise(self.baseline, self.extractor, draws=64, seed=9)
        b = fit_fft_noise(self.baseline, self.extractor, draws=64, seed=9)
        np.testing.assert_array_equal(a.R_fft, b.R_fft)
        np.testing.assert_allclose(a.R_normalized, a.R_fft / np.outer(a.scale, a.scale))
        self.assertTrue(np.isfinite(a.to_dict()["gamma_log_variance_approximation"]).all())

    def test_insufficient_gapped_and_constant_baselines_rejected(self):
        bad = self.baseline.copy()
        bad[0, 10] = np.nan
        for x in (bad, self.baseline[:, :100], np.zeros_like(self.baseline)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                fit_fft_noise(x, self.extractor, draws=64)


class KalmanTests(unittest.TestCase):
    def test_matches_standard_filter_when_spectral_noise_is_white(self):
        r = np.array([[1, .4], [.4, 2.]])
        c = known_calibration(r)
        loading = np.array([1., -.3])
        tracker = KalmanCUSUM(c, loading, process_variance=.02, drift_variance=.001)
        x, p = np.zeros(2), np.diag([1., .01])
        f, q = np.array([[.98, 1], [0, 1]]), np.diag([.02, .001])
        h = np.column_stack([loading, np.zeros(2)])
        for y in ([.2, -.1], [1., -.4], [.7, .2], [.3, 0]):
            xp, pp = f @ x, f @ p @ f.T + q
            v = y - h @ xp
            s = h @ pp @ h.T + r + np.eye(2)*1e-9
            k = np.linalg.solve(s, h @ pp).T
            x, p = xp + k @ v, pp - k @ h @ pp
            result = tracker.step(y)
            np.testing.assert_allclose(tracker.x[:2], x, atol=1e-10)
            np.testing.assert_allclose(tracker.P[:2, :2], p, atol=1e-10)
            np.testing.assert_allclose(result["innovation"], v, atol=1e-10)
            expected_llr = loading @ np.linalg.solve(s, v) - .5*loading @ np.linalg.solve(s, loading)
            self.assertAlmostEqual(result["log_likelihood_increment"], expected_llr)

    def test_augmented_filter_whitens_known_ar1_simulation(self):
        r = np.array([[1., .3], [.3, .8]])
        c = known_calibration(r, .55*r)
        loading = np.array([1., .7])
        tracker = KalmanCUSUM(c, loading)
        rng = np.random.default_rng(90)
        latent, noise = np.zeros(2), rng.multivariate_normal(np.zeros(2), r)
        projected, nis = [], []
        for i in range(2200):
            latent = np.array([[.98, 1], [0, 1]]) @ latent + rng.normal(size=2)*np.sqrt([.01, 1e-5])
            noise = .55*noise + rng.multivariate_normal(np.zeros(2), (1-.55**2)*r)
            result = tracker.step(loading*latent[0] + noise)
            if i >= 200:
                projected.append(result["projected_innovation_z"])
                nis.append(result["nis"])
        self.assertAlmostEqual(np.var(projected), 1, delta=.15)
        self.assertAlmostEqual(np.mean(nis), 2, delta=.18)
        self.assertAlmostEqual(np.corrcoef(projected[1:], projected[:-1])[0, 1], 0, delta=.08)
        self.assertGreaterEqual(np.linalg.eigvalsh(tracker.P).min(), -1e-10)

    def test_cusum_accumulates_unexpected_shift_and_resets_at_gap(self):
        tracker = KalmanCUSUM(known_calibration([[1.]]), [1], phi=1,
                              process_variance=0, drift_variance=0, cusum_threshold=4)
        tracker.P[:2, :2] = 0  # Known null state for this change-detection fixture.
        for _ in range(20):
            result = tracker.step([0])
        self.assertEqual(result["cusum"], 0)
        for _ in range(4):
            result = tracker.step([2])
        self.assertTrue(result["change_flag"])
        old = tracker.P.copy()
        result = tracker.step(None)
        self.assertFalse(result["measurement_used"])
        self.assertFalse(result["change_flag"])
        self.assertEqual(result["cusum"], 0)
        self.assertIsNone(result["innovation"])
        self.assertTrue(np.isfinite(old).all())

    def test_model_parameters_and_covariance_are_checked(self):
        c = known_calibration([[1.]])
        for kw in ({"phi": 2}, {"process_variance": -1}, {"cusum_shift": 0},
                   {"residual_covariance": [[-1]]}, {"residual_covariance": [[float("nan")]]}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                KalmanCUSUM(c, [1], **kw)
        with self.assertRaises(ValueError):
            KalmanCUSUM(c, [0])
