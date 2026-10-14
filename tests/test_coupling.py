import unittest
from unittest.mock import patch

import numpy as np

from psi_qeeg.coupling import benjamini_hochberg, phase_amplitude_index, window_coupling


class CouplingTests(unittest.TestCase):
    def test_uniform_phase_amplitude_has_zero_information(self):
        phase = np.linspace(-np.pi, np.pi, 18000, endpoint=False)
        mi, distribution = phase_amplitude_index(phase, np.ones(len(phase)))
        self.assertAlmostEqual(mi, 0, places=12)
        np.testing.assert_allclose(distribution, np.full(18, 1 / 18))
        coupled, _ = phase_amplitude_index(phase, 1 + .8 * np.cos(phase))
        self.assertGreater(coupled, .05)

    def test_invalid_phase_amplitude_input(self):
        for phase, amplitude in (([0, 1], [1]), ([0, 1], [-1, 2]), ([0, np.nan], [1, 2]), ([0, 0], [1, 1])):
            with self.assertRaises(ValueError):
                phase_amplitude_index(phase, amplitude)

    def test_known_false_discovery_adjustment(self):
        np.testing.assert_allclose(benjamini_hochberg([.1, .001, .01]), [.1, .003, .015])
        with self.assertRaises(ValueError):
            benjamini_hochberg([np.nan])

    def test_identical_signals_have_unit_connectivity_and_zero_imaginary_part(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=8000)
        result = window_coupling(np.column_stack((x, 2 * x)), 500, ['a', 'b'], surrogates=19)
        pair = result['pairs'][0]
        for band in ('theta', 'gamma'):
            self.assertAlmostEqual(pair[band + '_plv'], 1, places=10)
            self.assertAlmostEqual(pair[band + '_coherence'], 1, places=10)
            self.assertLess(pair[band + '_imaginary_coherency_abs'], 1e-10)

    def test_nonstationary_modulated_positive_control_exceeds_uncoupled_control(self):
        rng = np.random.default_rng(20261013)
        t = np.arange(8000) / 500
        phase = 2 * np.pi * np.cumsum(8 + 1.5 * np.sin(2 * np.pi * .37 * t)) / 500
        theta, carrier = 20 * np.sin(phase), np.cos(2 * np.pi * 65 * t)
        coupled = theta + 8 * (1 + .85 * np.cos(phase)) * carrier + rng.normal(scale=.3, size=len(t))
        uncoupled = theta + 8 * carrier + rng.normal(scale=.3, size=len(t))
        result = window_coupling(np.column_stack((coupled, uncoupled)), 500,
                                 ['coupled', 'uncoupled'], surrogates=199)
        local = {p['theta_channel']: p for p in result['pac'] if p['relationship'] == 'within_channel'}
        self.assertGreater(local['coupled']['tort_mi'], .025)
        self.assertGreater(local['coupled']['tort_mi'], 10 * local['uncoupled']['tort_mi'])
        self.assertLessEqual(local['coupled']['surrogate_p'], .01)

    def test_common_reference_can_raise_plv_without_residual_interaction(self):
        rng = np.random.default_rng(123)
        t = np.arange(8000) / 500
        phase = 2 * np.pi * np.cumsum(8 + np.sin(2 * np.pi * .23 * t)) / 500
        shared = 20 * np.sin(phase)[:, None]
        x = shared + rng.normal(size=(len(t), 4))
        names = ['a', 'b', 'c', 'd']
        original = window_coupling(x, 500, names, surrogates=19)
        bipolar = window_coupling(x, 500, names, mode='bipolar', surrogates=19)
        self.assertGreater(np.median([p['theta_plv'] for p in original['pairs']]), .95)
        self.assertLess(bipolar['pairs'][0]['theta_plv'], .4)

    def test_flat_bipolar_and_short_windows_rejected(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=8000)
        with self.assertRaises(ValueError):
            window_coupling(np.column_stack((x, x, x, x)), 500, ['a', 'b', 'c', 'd'], mode='bipolar', surrogates=19)
        with self.assertRaises(ValueError):
            window_coupling(np.ones((100, 2)), 500, ['a', 'b'], surrogates=19)

    def test_circular_shifts_exclude_both_exact_one_second_boundaries(self):
        class BoundarySampler:
            def choice(self, candidates, size, replace):
                self.assertions(candidates, size, replace)
                return np.r_[candidates[0], candidates[-1], candidates[1:size - 1]]

            def assertions(inner, candidates, size, replace):
                # A 16s, 500Hz input becomes 7000 samples after edge trimming.
                self.assertEqual((candidates[0], candidates[-1]), (501, 6499))
                self.assertEqual(len(candidates), 5999)
                self.assertFalse(replace)
        x = np.random.default_rng(40).normal(size=(8000, 2))
        with patch('psi_qeeg.coupling.np.random.default_rng', return_value=BoundarySampler()):
            result = window_coupling(x, 500, ['a', 'b'], surrogates=19)
        shifts = np.array(result['surrogate_shifts_samples'])
        self.assertTrue((np.minimum(shifts, result['samples_after_edge_trim'] - shifts) > 500).all())
        self.assertEqual(len(np.unique(shifts)), 19)
        for row in result['pac']:
            self.assertGreaterEqual(row['surrogate_p'], 1 / 20)
            self.assertAlmostEqual(row['surrogate_p'] * 20, round(row['surrogate_p'] * 20))

    def test_seeded_null_is_reproducible_and_distinct_shift_limit_enforced(self):
        x = np.random.default_rng(81).normal(size=(8000, 2))
        first = window_coupling(x, 500, ['a', 'b'], surrogates=19, seed=7)
        second = window_coupling(x, 500, ['a', 'b'], surrogates=19, seed=7)
        self.assertEqual(first['surrogate_shifts_samples'], second['surrogate_shifts_samples'])
        self.assertEqual(first['pac'], second['pac'])
        with self.assertRaisesRegex(ValueError, 'Insufficient distinct'):
            window_coupling(x, 500, ['a', 'b'], surrogates=6000)

    def test_coherence_matches_independent_scipy_reference(self):
        from scipy.signal import coherence
        from scipy.integrate import trapezoid
        rng = np.random.default_rng(9)
        common = rng.normal(size=8000)
        x = np.column_stack((common + rng.normal(size=8000), common + rng.normal(size=8000)))
        result = window_coupling(x, 500, ['a', 'b'], surrogates=19)
        f, c = coherence(x[500:-500, 0], x[500:-500, 1], fs=500,
                         window='hann', nperseg=1000, noverlap=500, detrend='constant')
        for band, low, high in [('theta', 4, 12), ('gamma', 30, 80)]:
            mask = (f >= low) & (f <= high)
            expected = trapezoid(c[mask], f[mask]) / (high - low)
            self.assertAlmostEqual(result['pairs'][0][band + '_coherence'], expected, places=12)


if __name__ == '__main__':
    unittest.main()
