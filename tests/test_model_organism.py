"""Synthetic repeated-measure schema fixtures; not clinical or mouse evidence."""

import unittest

import pandas as pd

from psi_qeeg.model_organism import EXPECTED_ISI_MS, summarize_mouse_gating


class MouseGatingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"ID": animal, "ISI": isi, "genotype": "wt" if animal < 3 else "hem", "ratio": animal / 10}
            for animal in range(1, 5) for isi in sorted(EXPECTED_ISI_MS)])

    def test_animals_are_units_not_repeated_rows(self):
        result = summarize_mouse_gating(self.frame)
        self.assertEqual(result["rows"], 28)
        self.assertEqual(result["animals_per_genotype"], {"hem": 2, "wt": 2})
        self.assertAlmostEqual(result["animal_mean_across_all_isis"]["wt"]["mean_ratio"], .15)
        self.assertEqual(result["meltdown_labels"], "not_available")
        self.assertIsNone(result["SOI"])

    def test_unmatched_conditions_or_duplicate_measurement_rejected(self):
        for frame in (self.frame.iloc[1:], pd.concat([self.frame, self.frame.iloc[:1]])):
            with self.assertRaises(ValueError):
                summarize_mouse_gating(frame)

    def test_inconsistent_genotype_or_nonfinite_ratio_rejected(self):
        changed = self.frame.copy()
        changed.loc[0, "genotype"] = "hem"
        invalid = self.frame.copy()
        invalid.loc[0, "ratio"] = float("inf")
        for frame in (changed, invalid):
            with self.assertRaises(ValueError):
                summarize_mouse_gating(frame)
