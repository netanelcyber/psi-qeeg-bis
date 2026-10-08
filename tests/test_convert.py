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
