"""Article arithmetic and calibration missingness; no patient data in fixtures."""

import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from psi_qeeg.indices import score_indices, fit_calibration, SOI_DIRECTIONS, RDI_COMPONENTS

ROOT=Path(__file__).resolve().parents[1]


def mathematical_fixture():
    calibration={"context":"illustrative_paper_example","duration_s":86400}
    values={}
    for index,names in (("SOI",SOI_DIRECTIONS),("RDI",RDI_COMPONENTS)):
        calibration[index]={"components":{n:{"mean":1,"sd":0.1} for n in names}}
        values.update({n:1. for n in names})
    return values,calibration


class IndexTests(unittest.TestCase):
    def test_paper_soi_is_100_not_18(self):
        d=ROOT/"examples/paper_calculation"
        f=json.loads((d/"features.json").read_text())
        c=json.loads((d/"calibration.json").read_text())
        scores=score_indices(f["values"],c)["indices"]
        self.assertAlmostEqual(scores["SOI"]["score"],100)
        self.assertIsNone(scores["RDI"]["score"])

    def test_baseline_zero_soi_and_100_rdi(self):
        v,c=mathematical_fixture()
        r=score_indices(v,c)["indices"]
        self.assertAlmostEqual(r["SOI"]["score"],0)
        self.assertAlmostEqual(r["RDI"]["score"],100)

    def test_decreased_gating_and_plv_raise_soi(self):
        v,c=mathematical_fixture()
        v["gating_efficiency"]=0.7
        v["frontoparietal_plv"]=0.7
        self.assertAlmostEqual(score_indices(v,c)["indices"]["SOI"]["score"],40)

    def test_symmetric_rdi_deviation(self):
        v,c=mathematical_fixture()
        values=[]
        for x in (0.85,1.15):
            v.update({n:x for n in RDI_COMPONENTS})
            values.append(score_indices(v,c)["indices"]["RDI"]["score"])
        self.assertAlmostEqual(values[0],50)
        self.assertAlmostEqual(values[1],50)

    def test_missing_component_no_silent_reweighting(self):
        v,c=mathematical_fixture()
        del v["posterior_gamma"]
        r=score_indices(v,c)["indices"]["SOI"]
        self.assertIsNone(r["score"])
        self.assertEqual(r["component_coverage"],0.8)
        self.assertIn("posterior_gamma",r["missing"])

    def test_nonfinite_measurement_is_missing(self):
        v,c=mathematical_fixture()
        v["alpha_variance"]=np.nan
        self.assertIsNone(score_indices(v,c)["indices"]["SOI"]["score"])

    def test_zero_calibration_sd_rejected(self):
        v,c=mathematical_fixture()
        c["SOI"]["components"]["alpha_variance"]["sd"]=0
        with self.assertRaises(ValueError):
            score_indices(v,c)

    def test_nonfinite_calibration_sd_rejected(self):
        v,c=mathematical_fixture()
        c["RDI"]["components"]["temporal_csd"]["sd"]=float("inf")
        with self.assertRaises(ValueError):
            score_indices(v,c)

    def test_weights_must_sum_to_one(self):
        v,c=mathematical_fixture()
        c["SOI"]["weights"]={n:1 for n in SOI_DIRECTIONS}
        with self.assertRaises(ValueError):
            score_indices(v,c)

    def test_short_calibration_blocks_composites(self):
        v,c=mathematical_fixture()
        c["duration_s"]=3600
        self.assertIsNone(score_indices(v,c)["indices"]["RDI"]["score"])

    def test_unknown_context_blocks_composites(self):
        v,c=mathematical_fixture()
        c["context"]="anesthesia"
        self.assertIsNone(score_indices(v,c)["indices"]["SOI"]["score"])

    def test_nan_calibration_duration_cannot_bypass_24h(self):
        v,c=mathematical_fixture()
        c["duration_s"]=float("nan")
        with self.assertRaises(ValueError):
            score_indices(v,c)

    def test_calibration_excludes_other_subject_and_episode(self):
        frame=pd.DataFrame({"time_s":[0,1,2,3,4],"epoch_end_s":[1,2,3,4,5],
                            "subject_id":["A","A","A","B","A"],
                            "recording_context":["awake_personal_baseline"]*5,
                            "research_label":["baseline","baseline","baseline","baseline","pre_meltdown"],
                            "posterior_gamma":[1,2,3,1000,1000]})
        result=fit_calibration(frame,"A")
        self.assertEqual(result["SOI"]["components"]["posterior_gamma"]["mean"],2)
        self.assertEqual(result["duration_s"],3)

    def test_overlapping_epochs_do_not_manufacture_hours(self):
        frame=pd.DataFrame({"time_s":[0,1,2],"epoch_end_s":[10,11,12],"subject_id":["A"]*3,
                            "recording_context":["awake_personal_baseline"]*3,
                            "research_label":["baseline"]*3,"posterior_gamma":[1,2,3]})
        self.assertEqual(fit_calibration(frame,"A")["duration_s"],12)

    def test_constant_baseline_rejected(self):
        frame=pd.DataFrame({"time_s":[0,1,2],"epoch_end_s":[1,2,3],"subject_id":["A"]*3,
                            "recording_context":["awake_personal_baseline"]*3,
                            "research_label":["baseline"]*3,"posterior_gamma":[1,1,1]})
        with self.assertRaises(ValueError):
            fit_calibration(frame,"A")

    def test_surgical_baseline_cannot_be_called_awake_calibration(self):
        frame=pd.DataFrame({"time_s":[0,1,2],"epoch_end_s":[1,2,3],"subject_id":["A"]*3,
                            "recording_context":["perioperative_anesthesia"]*3,
                            "research_label":["baseline"]*3,"posterior_gamma":[1,2,3]})
        with self.assertRaises(ValueError):
            fit_calibration(frame,"A")

    def test_calibration_does_not_mix_medication_profiles(self):
        frame=pd.DataFrame({"time_s":[0,1,2],"epoch_end_s":[1,2,3],"subject_id":["A"]*3,
                            "recording_context":["awake_personal_baseline"]*3,
                            "medication_profile_id":["A","A","B"],
                            "research_label":["baseline"]*3,"posterior_gamma":[1,2,3]})
        with self.assertRaises(ValueError):
            fit_calibration(frame,"A")

