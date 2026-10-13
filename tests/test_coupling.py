import unittest

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


if __name__ == '__main__':
    unittest.main()
