"""bis_proxy is checked on the shipped real VitalDB excerpt; no BIS equivalence is claimed."""

from pathlib import Path
import unittest

import numpy as np
from psi_qeeg.analysis import analyze
from psi_qeeg.convert import bis_proxy
from psi_qeeg.recording import read_recording

CASE = Path(__file__).resolve().parents[1] / "examples" / "vitaldb_case1"


class ProxyTests(unittest.TestCase):
    def test_proxy_range_and_rejected_epochs(self):
        rec = read_recording(CASE / "recording.csv", CASE / "metadata.json")
        features, _ = analyze(rec)
        proxy = bis_proxy(features, rec.channels[0])
        ok = proxy[features.accepted]
        self.assertTrue(((ok >= 0) & (ok <= 100)).all())
        self.assertTrue(proxy[~features.accepted].isna().all())
        self.assertTrue(np.isfinite(ok).all() and len(ok) > 0)

    def test_missing_features_give_nan(self):
        rec = read_recording(CASE / "recording.csv", CASE / "metadata.json")
        features, _ = analyze(rec)
        self.assertTrue(bis_proxy(features, "nope").isna().all())


if __name__ == "__main__":
    unittest.main()


class CohortTests(unittest.TestCase):
    def test_manifest_validation(self):
        import tempfile
        from psi_qeeg.convert import convert_cohort
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "m.csv"
            m.write_text("file,subject_id,group\na.edf,s1,XYZ\n")
            with self.assertRaises(ValueError):
                convert_cohort(m, Path(d) / "out")
            m.write_text("file,subject_id\na.edf,s1\n")
            with self.assertRaises(ValueError):
                convert_cohort(m, Path(d) / "out")


class ScaleTests(unittest.TestCase):
    def test_resume_failure_isolation_and_union_columns(self):
        import tempfile
        from unittest import mock
        import pandas as pd
        from psi_qeeg import convert as cv

        def fake_load(path, channels=None):
            if "bad" in str(path):
                raise ValueError("corrupt scan")
            frame = pd.read_csv(CASE / "recording.csv")[["time_s", "BIS_EEG1", "BIS_EEG2"]]
            return frame, 128.0, ["BIS_EEG1", "BIS_EEG2"]

        with tempfile.TemporaryDirectory() as d, mock.patch.object(cv, "load_raw", fake_load):
            root = Path(d)
            for g, name in (("ASD", "a.edf"), ("HC", "h.edf"), ("PSY", "bad.edf")):
                (root / g).mkdir()
                (root / g / name).write_bytes(b"x")
            self.assertEqual(cv.build_manifest(root, root / "m.csv"), 3)
            r = cv.convert_cohort(root / "m.csv", root / "out")
            self.assertEqual((r["converted"], r["failed"], r["total"]), (2, 1, 3))
            self.assertEqual(r["subjects_per_group"], {"autism_spectrum": 1, "healthy_control": 1})
            self.assertIn("corrupt scan", (root / "out/failures.csv").read_text())
            self.assertFalse((root / "out/subjects/ASD_a/recording.csv").exists())
            pooled = pd.read_csv(root / "out/cohort_features.csv")
            self.assertEqual(set(pooled.subject_id), {"ASD_a", "HC_h"})
            # resume: finished subjects are not recomputed
            with mock.patch.object(cv, "convert", side_effect=AssertionError("recomputed")):
                r2 = cv.convert_cohort(root / "m.csv", root / "out")
            self.assertEqual(r2["converted"], 2)


if __name__ == "__main__":
    unittest.main()
