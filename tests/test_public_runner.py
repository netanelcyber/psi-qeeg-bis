"""Window/cache integrity checks; byte fixtures are not clinical EEG."""

import hashlib
import contextlib
import io
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

spec = importlib.util.spec_from_file_location('public_runner', Path(__file__).parents[1] / 'scripts/validate_public_schizophrenia_cohort.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class WindowTests(unittest.TestCase):
    def test_discard_export_removes_previous_waveform(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); (folder / 'recording.csv').write_text('stale export')
            frame = pd.DataFrame({'time_s': [0, 1], 'C1': [2, 3]})
            runner.export_waveform(frame, folder, True)
            self.assertFalse((folder / 'recording.csv').exists())
            runner.export_waveform(frame, folder, False)
            pd.testing.assert_frame_equal(pd.read_csv(folder / 'recording.csv'), frame)

    def test_legacy_default_and_explicit_full_window(self):
        args = runner.parse_args([])
        self.assertEqual((args.window_s, args.epoch_s), (120, 4))
        frame = pd.DataFrame({'time_s': range(1000)})
        self.assertEqual(len(runner.select_window(frame, 2, 120)), 240)
        self.assertEqual(len(runner.select_window(frame, 2, 0)), 1000)
        self.assertEqual(len(runner.select_window(frame, 2, 999)), 1000)

    def test_invalid_window_or_short_epoch_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for value in ('-1', 'nan', 'inf'):
                with self.assertRaises(SystemExit): runner.parse_args(['--window-s', value])
            with self.assertRaises(SystemExit): runner.parse_args(['--epoch-s', '2'])


class SourceCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'sample.edf'
        self.data = b'complete synthetic byte fixture'
        self.info = {'id': 1, 'filesize': len(self.data), 'md5': hashlib.md5(self.data).hexdigest()}

    def test_valid_cache_is_not_downloaded(self):
        self.path.write_bytes(self.data)
        owner, method = (runner.hashlib, 'file_digest') if hasattr(runner.hashlib, 'file_digest') else (runner, '_md5_stream')
        with patch.object(runner, 'download_file', side_effect=AssertionError('unnecessary download')), patch.object(owner, method, wraps=getattr(owner, method)) as hashed:
            digest, restored = runner.ensure_source(self.info, self.path)
        self.assertEqual(hashed.call_count, 1)
        self.assertEqual(digest, self.info['md5'])
        self.assertFalse(restored)

    def test_truncated_cache_restored_atomically(self):
        self.path.write_bytes(self.data[:4])
        with patch.object(runner, 'download_file', side_effect=lambda _, p: p.write_bytes(self.data)):
            _, restored = runner.ensure_source(self.info, self.path)
        self.assertTrue(restored)
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertFalse(self.path.with_suffix('.edf.part').exists())

    def test_bad_download_never_replaces_previous_bytes(self):
        self.path.write_bytes(b'old incomplete bytes')
        with patch.object(runner, 'download_file', side_effect=lambda _, p: p.write_bytes(b'bad')):
            with self.assertRaises(RuntimeError): runner.ensure_source(self.info, self.path)
        self.assertEqual(self.path.read_bytes(), b'old incomplete bytes')
        self.assertFalse(self.path.with_suffix('.edf.part').exists())

    def test_missing_source_checksum_rejected(self):
        with self.assertRaises(ValueError): runner.ensure_source({'id': 1}, self.path)
