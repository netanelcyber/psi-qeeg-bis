"""Synthetic electrode-selection fixtures, never patient or clinical evidence."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from psi_qeeg.cohort import SPECTRAL_FEATURES, compare_electrode_coverage, electrode_feature_sets


class ElectrodeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.columns = [f"{channel}__{feature}" for channel in ("Fp1", "Fp2", "F7", "F8", "P3")
                        for feature in SPECTRAL_FEATURES]

    def test_regional_sets_exclude_posterior_and_other_hemisphere(self):
        sets = electrode_feature_sets(self.columns + ["P3__bis_proxy_uncalibrated", "group"])
        self.assertEqual(len(sets["full_scalp"]), 20)
        self.assertEqual(len(sets["left_frontal_temporal"]), 8)
        self.assertEqual({c.split("__")[0] for c in sets["left_frontal_temporal"]}, {"Fp1", "F7"})
        self.assertEqual({c.split("__")[0] for c in sets["right_frontal_temporal"]}, {"Fp2", "F8"})

    def test_prefixed_edf_channel_names(self):
        columns = [c.replace("Fp1__", "EEG FP1-REF__") for c in self.columns]
        self.assertTrue(any(c.startswith("EEG FP1-REF__") for c in
                            electrode_feature_sets(columns)["left_frontal_temporal"]))

    def test_missing_channel_or_feature_rejected(self):
        for columns in ([c for c in self.columns if not c.startswith("F7__")], self.columns[1:]):
            with self.assertRaises(ValueError):
                electrode_feature_sets(columns)

    def test_ambiguous_site_rejected(self):
        with self.assertRaises(ValueError):
            electrode_feature_sets(self.columns + ["EEG Fp1-REF__alpha_power_uv2"])

    def test_identical_input_table_and_full_scalp_output_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            table = root / "features.csv"
            pd.DataFrame([{c: 1.0 for c in self.columns}]).to_csv(table, index=False)
            fake = {"subjects_per_group": {"healthy_control": 2, "psychosis_spectrum": 2},
                    "epochs": 4, "epoch_balanced_accuracy": .5, "subject_balanced_accuracy": .5,
                    "subject_confusion_matrix": {"labels": [], "rows_true_cols_pred": []}}
            def train(path, out, columns):
                self.assertEqual(path, table)
                return dict(fake, features=columns)
            with patch("psi_qeeg.cohort.train_group_model", side_effect=train) as mock:
                report = compare_electrode_coverage(table, root / "out")
            self.assertEqual(mock.call_count, 3)
            self.assertEqual(mock.call_args_list[0].args[1], root / "out")
            self.assertEqual(len(report["cohort_features_sha256"]), 64)
            self.assertTrue((root / "out/electrode_comparison.json").exists())
