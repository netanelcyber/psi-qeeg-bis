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


class BisMontageTests(unittest.TestCase):
    def frame(self, cols):
        import pandas as pd
        n = 8
        return pd.DataFrame({"time_s": np.arange(n) / 128, **{c: np.full(n, float(i + 1)) for i, c in enumerate(cols)}})

    def test_left_montage_algebra_and_names(self):
        from psi_qeeg.convert import bis_montage
        f = self.frame(["EEG Fp1-LE", "EEG Fp2-LE", "EEG F7-LE", "EEG Fpz-LE"])
        out, loc, ref = bis_montage(f, "left")
        self.assertEqual(ref, "Fpz")
        self.assertEqual(loc, {"BISlike_Fp1_minus_ref": "Fp1", "BISlike_F7_minus_ref": "F7"})
        self.assertTrue((out["BISlike_Fp1_minus_ref"] == 1 - 4).all())
        self.assertTrue((out["BISlike_F7_minus_ref"] == 3 - 4).all())

    def test_fpz_approximated_and_flagged(self):
        from psi_qeeg.convert import bis_montage
        _, _, ref = bis_montage(self.frame(["Fp1", "Fp2", "F7"]), "left")
        self.assertIn("approximate", ref)

    def test_missing_electrodes_rejected(self):
        from psi_qeeg.convert import bis_montage
        with self.assertRaises(ValueError):
            bis_montage(self.frame(["Fp1", "Fp2", "Cz"]), "left")
        with self.assertRaises(ValueError):
            bis_montage(self.frame(["F7", "Cz"]), "left")


class ReviewFixTests(unittest.TestCase):
    def test_bipolar_rejected_and_reference_normalised(self):
        import pandas as pd
        from psi_qeeg.convert import bis_montage
        n = 8
        base = {"time_s": np.arange(n) / 128}
        bip = pd.DataFrame({**base, "EEG Fp1-F7": np.ones(n), "EEG F7-T3": np.ones(n), "EEG Fp2-F8": np.ones(n)})
        with self.assertRaises(ValueError):
            bis_montage(bip, "left")
        f = pd.DataFrame({**base, "EEG Fp1-REF": np.ones(n), "EEG F7-REF": 2 * np.ones(n), "EEG Cz-REF": 5 * np.ones(n)})
        out, _, ref = bis_montage(f, "left", reference="EEG Cz-REF")
        self.assertEqual(ref, "cz_scan_channel")
        self.assertTrue((out["BISlike_F7_minus_ref"] == -3).all())

    def test_manifest_ids_unique_and_absolute(self):
        import tempfile
        import pandas as pd
        from psi_qeeg.convert import build_manifest
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "ASD" / "a").mkdir(parents=True)
            (root / "ASD" / "a_b.edf").write_bytes(b"x")
            (root / "ASD" / "a" / "b.edf").write_bytes(b"x")
            build_manifest(root, root / "elsewhere.csv")
            m = pd.read_csv(root / "elsewhere.csv")
            self.assertFalse(m.subject_id.duplicated().any())
            self.assertTrue(all(Path(p).is_absolute() for p in m.file))

    def test_context_kept_resume_invalidates_and_failures_persist(self):
        import tempfile
        from unittest import mock
        import pandas as pd
        from psi_qeeg import convert as cv

        def fake_load(path, channels=None):
            if "bad" in str(path):
                raise ValueError("corrupt")
            return pd.read_csv(CASE / "recording.csv")[["time_s", "BIS_EEG1"]], 128.0, ["BIS_EEG1"]

        with tempfile.TemporaryDirectory() as d, mock.patch.object(cv, "load_raw", fake_load):
            root = Path(d)
            (root / "a.edf").write_bytes(b"x")
            (root / "bad.edf").write_bytes(b"x")
            m = root / "m.csv"
            m.write_text("file,subject_id,group,context\na.edf,s1,HC,awake_observational_research\n")
            cv.convert_cohort(m, root / "out")
            self.assertIn("recording_context", pd.read_csv(root / "out/cohort_features.csv"))
            m.write_text("file,subject_id,group,context\na.edf,s1,PSY,awake_observational_research\n")
            cv.convert_cohort(m, root / "out")  # group changed: stale features must not be reused
            self.assertEqual(set(pd.read_csv(root / "out/cohort_features.csv").group), {"psychosis_spectrum"})
            m.write_text("file,subject_id,group\nbad.edf,s2,HC\n")
            with self.assertRaises(ValueError):
                cv.convert_cohort(m, root / "out2")
            self.assertIn("corrupt", (root / "out2/failures.csv").read_text())

    def test_default_reference_not_reported(self):
        import inspect
        from psi_qeeg.convert import convert
        self.assertEqual(inspect.signature(convert).parameters["reference"].default, "not_reported")


try:
    import sklearn  # noqa: F401
    _SK = True
except ImportError:
    _SK = False


@unittest.skipUnless(_SK, "scikit-learn not installed")
class MajorityTests(unittest.TestCase):
    def test_tie_is_not_resolved_alphabetically(self):
        import pandas as pd
        from psi_qeeg.ml import _majority
        self.assertEqual(_majority(pd.Series(["autism_spectrum", "healthy_control"])), "no_majority")
        self.assertEqual(_majority(pd.Series(["healthy_control", "healthy_control", "autism_spectrum"])), "healthy_control")
