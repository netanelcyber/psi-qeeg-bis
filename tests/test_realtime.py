"""Synthetic streams: they test the detector mechanics, not any prediction of real events."""

import io
from pathlib import Path
import unittest

import numpy as np
from psi_qeeg.realtime import MonitorConfig, RealtimeMonitor, evaluate_alerts, run_stream

FS = 128
CASE = Path(__file__).resolve().parents[1] / "examples" / "vitaldb_case1"


def eeg(seconds, rng, alpha=10.0, slow=0.0, n_ch=2):
    t = np.arange(round(seconds * FS)) / FS
    x = rng.normal(0, 5, (len(t), n_ch)) + alpha * np.sin(2 * np.pi * 10 * t)[:, None]
    return x + slow * np.sin(2 * np.pi * 2 * t)[:, None]


def run(signal, cfg=None):
    mon = RealtimeMonitor(FS, ["a", "b"], cfg or MonitorConfig(baseline_epochs=20))
    results = []
    for i in range(0, len(signal), 32):          # irregular chunking must not matter
        results += mon.push(signal[i:i + 32])
    return mon, results


class MonitorTests(unittest.TestCase):
    def test_no_alert_on_stationary_signal(self):
        _, res = run(eeg(200, np.random.default_rng(0)))
        self.assertFalse(any(r["state"] == "alert" for r in res))
        self.assertEqual(res[0]["state"], "warmup")

    def test_detects_drift_after_baseline_and_warns_first(self):
        rng = np.random.default_rng(1)
        sig = np.vstack([eeg(120, rng), eeg(60, rng, alpha=2, slow=60)])
        _, res = run(sig)
        alerts = [r for r in res if r["state"] == "alert"]
        self.assertTrue(alerts)
        self.assertGreater(alerts[0]["t_end_s"], 120)        # never before the change
        self.assertTrue(any(r["state"] == "rising" and r["t_end_s"] <= alerts[0]["t_end_s"] for r in res))
        self.assertTrue(alerts[0]["top_features"])

    def test_artifact_epochs_rejected_and_signal_lost(self):
        rng = np.random.default_rng(2)
        bad = eeg(40, rng) * 1000                             # amplitude outliers
        mon, res = run(np.vstack([eeg(100, rng), bad]))
        self.assertTrue(any(not r["accepted"] for r in res))
        self.assertEqual(mon.state, "signal_lost")
        self.assertFalse(any(r["state"] == "alert" for r in res))

    def test_chunking_invariant(self):
        sig = eeg(100, np.random.default_rng(3))
        a = [r["t_end_s"] for r in run(sig)[1]]
        mon = RealtimeMonitor(FS, ["a", "b"], MonitorConfig(baseline_epochs=20))
        b = [r["t_end_s"] for r in mon.push(sig)]
        self.assertEqual(a, b)

    def test_config_and_inputs_validated(self):
        with self.assertRaises(ValueError):
            MonitorConfig(hop_s=10)
        with self.assertRaises(ValueError):
            RealtimeMonitor(64, ["a"])
        with self.assertRaises(ValueError):
            RealtimeMonitor(FS, ["a", "b"]).push(np.zeros((10, 3)))

    def test_real_excerpt_runs(self):
        from psi_qeeg.recording import read_recording
        rec = read_recording(CASE / "recording.csv", CASE / "metadata.json")
        mon = RealtimeMonitor(rec.fs, rec.channels, MonitorConfig(baseline_epochs=15))
        run_stream(mon, [rec.eeg[i:i + 128] for i in range(0, len(rec.eeg), 128)], out=io.StringIO())
        self.assertIn(mon.state, {"normal", "rising", "alert", "warmup", "signal_lost"})


class EvaluationTests(unittest.TestCase):
    def test_lead_time_and_false_alerts(self):
        r = evaluate_alerts([50, 300], [100, 500], duration_s=3600, horizon_s=60)
        self.assertEqual((r["detected"], r["false_alerts"]), (1, 1))
        self.assertEqual(r["median_lead_s"], 50)
        self.assertEqual(r["false_alerts_per_hour"], 1)

    def test_alert_after_event_is_not_a_warning(self):
        r = evaluate_alerts([120], [100], duration_s=600)
        self.assertEqual((r["detected"], r["false_alerts"]), (0, 1))


if __name__ == "__main__":
    unittest.main()
