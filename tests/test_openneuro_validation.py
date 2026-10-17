"""Offline analytical and provenance regressions; all generated EEG is synthetic."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from psi_qeeg import analysis, recording
from psi_qeeg.quality import QualityPolicy, assess_epoch
from research.meltdown_forecast import validate_real_eeg as validator


class OpenNeuroValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "synthetic.bdf"
        self.path.write_bytes(b"synthetic reader placeholder, not real BDF")
        self.manifest = json.loads(validator.DEFAULT_MANIFEST.read_text())
        self.manifest["source_metadata"] = {}
        self.manifest["power_line_frequency_hz"] = None
        self.manifest["expected_samples_per_channel"] = 2048
        self.manifest["download"]["size_bytes"] = self.path.stat().st_size
        self.manifest["download"]["sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.meta = self.root / "manifest.json"
        self.meta.write_text(json.dumps(self.manifest))
        t = np.arange(2048) / 512
        frame = pd.DataFrame({"time_s": t, **{c: 10000 + 10 * np.sin(2*np.pi*10*t)
                                             for c in self.manifest["eeg_columns"]}})
        self.rec = recording.Recording(frame, copy.deepcopy(self.manifest))

    def test_uses_production_analyzer_and_preserves_intentional_missing_values(self):
        with mock.patch.object(recording, "read_edf", return_value=self.rec) as reader, \
                mock.patch.object(analysis, "analyze", wraps=analysis.analyze) as analyzer:
            report = validator.validate_file(self.path, self.meta, self.root / "out")
        reader.assert_called_once_with(self.path, self.meta)
        self.assertIs(analyzer.call_args_list[0].args[0], self.rec)
        self.assertEqual(report["accepted_epochs"], 1)
        self.assertEqual(report["validation_status"], "passed")
        self.assertEqual(report["quality_warnings"], ["emg_quality_unknown"])
        self.assertIsNone(report["SOI"])
        self.assertEqual(report["classification"], "indeterminate")
        exported = pd.read_csv(self.root / "out/real_eeg_features.csv")
        self.assertTrue(exported.sqi_min.isna().all())
        self.assertTrue(exported.filter(like="bicoherence").isna().all().all())
        self.assertAlmostEqual(exported["Fp1__alpha_power_uv2"].iat[0], 50, places=8)

    def test_no_accepted_epochs_is_failure_not_vacuous_pass(self):
        self.rec.frame[self.rec.channels] = 0
        with mock.patch.object(recording, "read_edf", return_value=self.rec):
            with self.assertRaisesRegex(ValueError, "No epochs"):
                validator.validate_file(self.path, self.meta, self.root / "out")
        result = json.loads((self.root / "out/real_eeg_validation_report.json").read_text())
        self.assertEqual(result["validation_status"], "failed")

    def test_checksum_mismatch_prevents_eeg_parser(self):
        self.path.write_bytes(b"tampered")
        with mock.patch.object(recording, "read_edf") as reader:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validator.validate_file(self.path, self.meta, self.root / "out")
        reader.assert_not_called()

    def test_sidecar_checksum_mismatch_prevents_parser(self):
        self.manifest["source_metadata"] = {"channels.tsv": {"sha256": "0" * 64}}
        self.meta.write_text(json.dumps(self.manifest))
        (self.root / "channels.tsv").write_bytes(b"changed")
        with mock.patch.object(recording, "read_edf") as reader:
            with self.assertRaisesRegex(ValueError, "sidecar SHA-256 mismatch"):
                validator.validate_file(self.path, self.meta, self.root / "out")
        reader.assert_not_called()

    def test_pinned_download_url_selects_one_object_and_version(self):
        manifest = json.loads(validator.DEFAULT_MANIFEST.read_text())
        url = validator.download_url(manifest)
        self.assertIn("sub-10003_task-Restingstate_run-01_eeg.bdf?versionId=", url)
        self.assertTrue(url.endswith(manifest["download"]["version_id"]))
        self.assertNotIn("\n", url)
        self.assertEqual(manifest["dataset_version"], "1.0.0")

    def test_download_checks_bytes_before_publish_and_reuses_only_verified_file(self):
        target = self.root / "download.bdf"
        with mock.patch.object(validator, "urlopen", return_value=io.BytesIO(self.path.read_bytes())) as opener:
            validator.download_recording(self.manifest, target)
            validator.download_recording(self.manifest, target)
        opener.assert_called_once()
        target.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validator.download_recording(self.manifest, target)
        target.unlink()
        with mock.patch.object(validator, "urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validator.download_recording(self.manifest, target)
        self.assertFalse(target.exists())
        self.assertFalse(target.with_suffix(".bdf.part").exists())

    def test_missing_spectral_column_and_injected_nan_fail(self):
        features, summary = analysis.analyze(self.rec, policy=validator.POLICY)
        self.assertTrue(validator.validate_features(features.drop(columns="Fp1__alpha_power_uv2"), summary, self.rec.channels))
        features.loc[0, "Fp1__alpha_power_uv2"] = np.nan
        self.assertTrue(validator.validate_features(features, summary, self.rec.channels))

    def test_analytical_reference_checks(self):
        checks = validator.synthetic_reference_checks()
        self.assertAlmostEqual(checks["flat_psd_integral"], 39.5)
        self.assertAlmostEqual(checks["mixture_delta_alpha_ratio"], 0.25)

    def test_optional_mne_metadata_absent_does_not_break_offline_checks(self):
        original = validator.version

        def without_mne(name):
            if name == "mne":
                raise validator.PackageNotFoundError(name)
            return original(name)

        with mock.patch.object(validator, "version", side_effect=without_mne):
            self.assertIsNone(validator.environment_versions()["mne"])

    def test_wrong_epoch_accounting_fails(self):
        original = analysis.analyze

        def incorrect_accounting(rec, **kwargs):
            features, summary = original(rec, **kwargs)
            if rec is self.rec:
                summary["complete_epochs"] += 1
            return features, summary

        with mock.patch.object(recording, "read_edf", return_value=self.rec), \
                mock.patch.object(analysis, "analyze", side_effect=incorrect_accounting):
            with self.assertRaisesRegex(ValueError, "epoch counts"):
                validator.validate_file(self.path, self.meta, self.root / "out")


class OffsetQualityTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(2048) / 512
        self.x = (10000 + 10 * np.sin(2*np.pi*10*t))[:, None]
        self.frame = pd.DataFrame(index=range(len(t)))
        self.policy = QualityPolicy(amplitude_reference="epoch_mean")

    def test_default_rejects_offset_explicit_centered_amplitude_accepts(self):
        before = self.x.copy()
        self.assertFalse(assess_epoch(self.x, self.frame, QualityPolicy(), False)["accepted"])
        self.assertTrue(assess_epoch(self.x, self.frame, self.policy, False)["accepted"])
        np.testing.assert_array_equal(self.x, before)

    def test_centered_policy_still_rejects_artifact_and_missing_samples(self):
        self.x[30] += 1000
        self.assertFalse(assess_epoch(self.x, self.frame, self.policy, False)["accepted"])
        self.x[30] = np.nan
        self.assertFalse(assess_epoch(self.x, self.frame, self.policy, False)["accepted"])

    def test_invalid_amplitude_reference_rejected(self):
        with self.assertRaises(ValueError):
            QualityPolicy(amplitude_reference="unknown")
