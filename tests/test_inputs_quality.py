"""Clock, scaling, gaps and artifact behavior on synthetic test inputs."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from psi_qeeg.recording import read_csv, Recording, validate
from psi_qeeg.quality import assess_epoch, QualityPolicy
from psi_qeeg.vitaldb import waveform_times, hold_numeric
from psi_qeeg.analysis import analyze


class InputTests(unittest.TestCase):
    def _input(self, unit="uV", time=None):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name)
        pd.DataFrame({"time_s": np.arange(512)/128 if time is None else time,
                      "EEG": np.sin(np.arange(512))}).to_csv(p/"raw.csv", index=False)
        (p/"meta.json").write_text(json.dumps({"sampling_rate_hz":128, "eeg_unit":unit, "eeg_columns":["EEG"]}))
        return p

    def test_volts_to_microvolts_exactly_once(self):
        p = self._input("V")
        recording = read_csv(p/"raw.csv", p/"meta.json")
        self.assertAlmostEqual(recording.eeg[1,0], np.sin(1)*1e6, places=7)
        self.assertEqual(recording.metadata["eeg_unit"], "uV")

    def test_unknown_units_rejected(self):
        p = self._input("unknown")
        with self.assertRaises(ValueError):
            read_csv(p/"raw.csv", p/"meta.json")

    def test_nonuniform_clock_rejected(self):
        t = np.arange(512)/128
        t[30] += 0.002
        p = self._input(time=t)
        with self.assertRaises(ValueError):
            read_csv(p/"raw.csv", p/"meta.json")

    def test_monitor_scalar_not_raw_eeg(self):
        with self.assertRaises(ValueError):
            validate(Recording(pd.DataFrame({"time_s":[0,1/128],"BIS":[50,51]}),{"sampling_rate_hz":128,"eeg_columns":["BIS"]}))

    def test_vitaldb_compact_clock(self):
        t = waveform_times(pd.DataFrame({"Time":[0,1/128,np.nan,np.nan],"EEG":[1,2,3,4]}))
        np.testing.assert_allclose(t,np.arange(4)/128)

    def test_numeric_causal_alignment_no_future_backfill(self):
        track = pd.DataFrame({"Time":[1.,3.],"BIS/SQI":[80.,90.]})
        values = hold_numeric(np.array([0.,1.,2.,3.,9.]), track)
        np.testing.assert_allclose(values,[np.nan,80,80,90,np.nan],equal_nan=True)

    def test_nonmonotone_numerics_rejected(self):
        track = pd.DataFrame({"Time":[2.,1.],"BIS":[50,60]})
        with self.assertRaises(ValueError):
            hold_numeric(np.array([1.]),track)

    def test_unknown_bis_mapping_suppresses_pair_features(self):
        p=self._input()
        r=read_csv(p/"raw.csv",p/"meta.json")
        r.metadata["sensor"]={"family":"BIS"}
        r.frame["SQI"]=99.
        r.frame["EMG"]=20.
        features,summary=analyze(r)
        self.assertFalse(any(c.startswith("pair_") for c in features))
        self.assertEqual(summary["classification"],"indeterminate")
        self.assertIsNone(summary["SOI"])


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.x=np.c_[np.sin(np.arange(512)),np.cos(np.arange(512))]
        self.frame=pd.DataFrame({"SQI":np.full(512,95.),"EMG":np.full(512,25.)})
        self.p=QualityPolicy()

    def test_complete_high_quality_accepted(self):
        self.assertTrue(assess_epoch(self.x,self.frame,self.p,True)["accepted"])

    def test_missing_waveform_not_interpolated(self):
        self.x[5,0]=np.nan
        qc=assess_epoch(self.x,self.frame,self.p,True)
        self.assertFalse(qc["accepted"])
        self.assertIn("missing_eeg_samples",qc["rejection_reasons"])

    def test_low_sqi_rejected(self):
        self.frame.loc[3,"SQI"]=0
        self.assertFalse(assess_epoch(self.x,self.frame,self.p,True)["accepted"])

    def test_bis_sqi_required(self):
        self.assertFalse(assess_epoch(self.x,self.frame.drop(columns="SQI"),self.p,True)["accepted"])

    def test_out_of_range_sqi_rejected(self):
        self.frame["SQI"]=101
        self.assertFalse(assess_epoch(self.x,self.frame,self.p,True)["accepted"])

    def test_high_emg_explicitly_flagged(self):
        self.frame["EMG"]=50
        self.assertIn("possible_muscle_contamination",assess_epoch(self.x,self.frame,self.p,True)["warnings"])

    def test_missing_emg_not_claimed_clean(self):
        self.assertIn("emg_quality_unknown",assess_epoch(self.x,self.frame.drop(columns="EMG"),self.p,True)["warnings"])

    def test_flat_channel_rejected(self):
        self.x[:,1]=1
        self.assertFalse(assess_epoch(self.x,self.frame,self.p,True)["accepted"])

    def test_large_amplitude_rejected(self):
        self.x[30,1]=900
        self.assertFalse(assess_epoch(self.x,self.frame,self.p,True)["accepted"])

    def test_invalid_qc_threshold_rejected(self):
        with self.assertRaises(ValueError):
            QualityPolicy(min_sqi=-1)

