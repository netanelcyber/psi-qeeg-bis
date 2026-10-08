"""Acceptance tests on the included real, attributed VitalDB BIS excerpt."""

import hashlib
import json
from pathlib import Path
import unittest
import numpy as np

from psi_qeeg.recording import read_csv
from psi_qeeg.analysis import analyze

EXAMPLE=Path(__file__).resolve().parents[1]/"examples/vitaldb_case1"


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording=read_csv(EXAMPLE/"recording.csv",EXAMPLE/"metadata.json")
        cls.features,cls.summary=analyze(cls.recording)

    def test_checksum_and_source_provenance(self):
        m=self.recording.metadata
        self.assertEqual(hashlib.sha256((EXAMPLE/"recording.csv").read_bytes()).hexdigest(),m["recording_csv_sha256"])
        self.assertEqual(len(m["source_tracks"]),8)
        self.assertEqual(m["license"],"CC-BY-4.0")

    def test_real_sampling_and_amplitudes(self):
        self.assertEqual(self.recording.fs,128)
        self.assertEqual(self.recording.eeg.shape,(15360,2))
        self.assertTrue(np.isfinite(self.recording.eeg).all())
        self.assertAlmostEqual(self.recording.time[0],332)

    def test_complete_real_epochs(self):
        self.assertEqual(self.summary["accepted_epochs"],30)
        self.assertEqual(self.summary["rejected_epochs"],0)

    def test_real_spectrum_finite(self):
        self.assertTrue(np.isfinite(self.features.BIS_EEG1__alpha_power_uv2).all())
        self.assertTrue((self.features.BIS_EEG1__alpha_power_uv2>0).all())

    def test_no_unverified_psychiatric_inference(self):
        self.assertIsNone(self.summary["SOI"])
        self.assertIsNone(self.summary["RDI"])
        self.assertEqual(self.summary["classification"],"indeterminate")
        self.assertEqual(self.summary["provenance"]["channel_locations"],{})
        self.assertFalse(any(c.startswith("pair_") for c in self.features))

    def test_bicoherence_requires_past_history(self):
        name="BIS_EEG1__auto_bicoherence_squared_60s"
        self.assertTrue(self.features[name].iloc[:14].isna().all())
        self.assertTrue(self.features[name].iloc[14:].notna().all())

