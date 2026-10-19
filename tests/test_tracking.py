"""Tracking replay, identity, quality and CLI regression checks on synthetic EEG."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from psi_qeeg.cli import main
from psi_qeeg.recording import Recording
from psi_qeeg.tracking import track_recording


class TrackingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(128*36, 2))*8
        frame = pd.DataFrame(x, columns=["F3", "F4"])
        frame.insert(0, "time_s", np.arange(len(x))/128 + 100)
        self.recording = Recording(frame, {"sampling_rate_hz": 128, "eeg_unit": "uV",
            "eeg_columns": ["F3", "F4"], "subject_id": "synthetic-test", "reference_verified": True,
            "channel_locations": {"F3": "F3", "F4": "F4"}})
        self.model = {"subject_id": "synthetic-test", "medication_profile_id": "not_recorded",
                      "loadings": {"F3:log_beta_alpha": 1., "F4:log_beta_alpha": 1.}}

    def run_trace(self, **kwargs):
        return track_recording(self.recording, self.model, 24, draws=64, **kwargs)

    def test_timestamps_outputs_and_no_clinical_inference(self):
        trace, summary = self.run_trace()
        self.assertEqual(len(trace), 11)
        self.assertEqual(trace.window_start_s.iloc[0], 124)
        self.assertEqual(trace.window_end_s.iloc[0], 126)
        self.assertTrue(trace.measurement_used.all())
        self.assertTrue(np.isfinite(trace.latent_state).all())
        self.assertEqual(summary["classification"], "indeterminate")
        self.assertIsNone(summary["SOI"])
        self.assertIsNone(summary["RDI"])
        json.dumps(summary, allow_nan=False)

    def test_calibration_does_not_use_monitoring_samples(self):
        _, before = self.run_trace()
        self.recording.frame.loc[128*24:, ["F3", "F4"]] *= 3
        _, after = self.run_trace()
        self.assertEqual(before["noise_calibration"], after["noise_calibration"])

    def test_gaps_are_prediction_only_and_do_not_interpolate_eeg(self):
        self.recording.frame.loc[128*27, "F3"] = np.nan
        trace, summary = self.run_trace()
        bad = trace[trace.quality_reasons.str.contains("missing_eeg_samples")]
        self.assertEqual(len(bad), 2)
        self.assertFalse(bad.measurement_used.any())
        self.assertFalse(bad.change_flag.any())
        self.assertTrue((bad.cusum == 0).all())
        self.assertEqual(summary["prediction_only_windows"], 2)

    def test_high_frequency_quality_gate_is_optional_and_named_as_possible(self):
        where = np.arange(128*27, 128*30)
        self.recording.frame.loc[where, "F3"] += 60*np.sin(2*np.pi*35*where/128)
        trace, _ = self.run_trace(high_z_limit=6)
        bad = trace[trace.quality_reasons.str.contains("possible_high_frequency_contamination")]
        self.assertGreater(len(bad), 0)
        self.assertFalse(bad.measurement_used.any())

    def test_identity_and_baseline_quality_rejected(self):
        self.model["subject_id"] = "another-subject"
        with self.assertRaisesRegex(ValueError, "subject_id"):
            self.run_trace()
        self.model["subject_id"] = "synthetic-test"
        self.recording.frame.loc[20, "F3"] = np.nan
        with self.assertRaisesRegex(ValueError, "baseline is not clean"):
            self.run_trace()

    def test_bad_json_model_is_a_clear_configuration_error(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            track_recording(self.recording, [], 24, draws=64)
        self.model["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown model fields"):
            self.run_trace()

    def test_faa_requires_documented_reference_and_frontal_sites(self):
        self.model["loadings"]["FAA"] = -.2
        trace, _ = self.run_trace(frontal_pair=("F3", "F4"))
        self.assertIn("FAA", trace)
        self.recording.metadata["reference_verified"] = False
        with self.assertRaisesRegex(ValueError, "FAA requires"):
            self.run_trace(frontal_pair=("F3", "F4"))

    def test_cli_writes_strict_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            self.recording.frame.to_csv(d/"eeg.csv", index=False)
            (d/"metadata.json").write_text(json.dumps(self.recording.metadata), encoding="utf-8")
            (d/"model.json").write_text(json.dumps(self.model), encoding="utf-8")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(["track", str(d/"eeg.csv"), "--metadata", str(d/"metadata.json"),
                             "--model", str(d/"model.json"), "--baseline-sec", "24", "--bootstrap-draws", "64",
                             "--out", str(d/"output")])
            self.assertEqual(code, 0)
            summary = json.loads((d/"output/tracking.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["measurement_updates"], 11)
            self.assertEqual(len(pd.read_csv(d/"output/tracking.csv")), 11)
