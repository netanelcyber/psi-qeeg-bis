"""Artificial runner receipts; no real behavioral/clinical validation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


video = load_script('validate_public_ibl_video_lfp')
coupling = load_script('analyze_ibl_coupling')


class IblRunnerTests(unittest.TestCase):
    def test_wheel_interval_end_must_be_strictly_later_than_start(self):
        video.validate_movement_intervals(np.array([[1., 2.], [3., 4.]]))
        for value in (np.array([[1., 1.]]), np.array([[2., 1.]]), np.array([[1., np.nan]]),
                      np.array([[3., 4.], [1., 2.]])):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'movement intervals'):
                video.validate_movement_intervals(value)

    def test_nondefault_surrogate_count_matches_computation_protocol_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; source.mkdir()
            # Synthetic dimensions match the fixed excerpt solely to exercise the receipt.
            np.savez_compressed(source / 'lfp_excerpt_uv.npz', signal_uv=np.zeros((600000, 4)),
                session_time_s=80 + np.arange(600000) / 2500, sampling_rate_hz=2500,
                raw_channel_indices=[32, 96, 128, 192])
            pd.DataFrame([{'event_id': 'wheel_0', 'session_start_s': 200., 'session_end_s': 201.}]).to_csv(source / 'movement_annotations.csv', index=False)
            (source / 'video_lfp_alignment_report.json').write_text(json.dumps({
                'session_id': '274dacc9-05b2-4154-b4f7-8f6dd46efcc8', 'subject_genotype': 'synthetic_fixture'}))
            np.save(root / 'movements.npy', [[200., 201.]])
            hashes = {name: coupling.sha256(source / name) for name in coupling.HASHES}
            def estimate(block, fs, names, mode, surrogates, seed):
                pair = {band + '_' + metric: .1 for band in ('theta', 'gamma')
                        for metric in ('plv', 'coherence', 'imaginary_coherency_abs')}
                return {'pairs': [pair], 'pac': [{'theta_channel': 'a', 'gamma_channel': 'a',
                    'relationship': 'within_channel', 'tort_mi': 0., 'surrogate_p': 1.,
                    'surrogate_count': surrogates, 'phase_amplitude_distribution': [1 / 18] * 18}],
                    'max_abs_uv': 0., 'median_std_uv': 0., 'samples_after_edge_trim': 7000,
                    'surrogate_shifts_samples': list(range(501, 501 + surrogates))}
            with patch.object(coupling, 'HASHES', hashes), patch.object(coupling, 'MOVEMENT_HASH', coupling.sha256(root / 'movements.npy')), \
                    patch.object(coupling, 'window_coupling', side_effect=estimate) as mock, \
                    patch.object(coupling, 'render_report'), patch('builtins.print'):
                report = coupling.run(source, root / 'movements.npy', root / 'out', surrogates=19)
            self.assertTrue(mock.call_count > 0)
            self.assertTrue(all(call.args[4] == 19 for call in mock.call_args_list))
            self.assertEqual(report['surrogates_per_test'], 19)
            self.assertTrue(report['surrogate_control'].startswith('19 distinct'))
            self.assertNotIn('199', report['surrogate_control'])
            self.assertEqual(json.loads((root / 'out/analysis_protocol.json').read_text())['surrogates'], 19)
            self.assertTrue(pd.read_csv(root / 'out/theta_gamma_pac.csv').surrogate_count.eq(19).all())
