"""Artificial tables check animal-level sampling, not Fmr1 discrimination."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('fmr1_runner', Path(__file__).parents[1] / 'scripts/validate_public_fmr1_qeeg.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class SourceVectorTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([{'AnID': f'fixture-{a}', 'GT': 'WT' if a == 0 else 'KO', 'Loc': loc,
                                   'Freq': frequency, 'Pow': 0.1 + frequency}
                                  for a in range(2) for loc in ('Front', 'Temp') for frequency in range(1, 100)])

    def test_frequency_rows_are_not_independent_animals(self):
        f, names, excluded = runner.animal_vectors(self.frame.sample(frac=1, random_state=1), 'fixture')
        self.assertEqual((len(f), len(names), len(excluded)), (2, 198, 0))
        self.assertNotIn('genotype', names)
        self.assertAlmostEqual(f[names[0]].iloc[0], np.log10(1.1))

    def test_missing_power_excludes_whole_animal_without_imputation(self):
        self.frame.loc[0, 'Pow'] = np.nan
        f, _, excluded = runner.animal_vectors(self.frame, 'fixture')
        self.assertEqual(list(f.animal_id), ['fixture-1'])
        self.assertEqual(excluded[0]['animal_id'], 'fixture-0')

    def test_duplicate_condition_and_inconsistent_genotype_rejected(self):
        with self.assertRaises(ValueError): runner.animal_vectors(pd.concat([self.frame, self.frame.iloc[:1]]), 'fixture')
        self.frame.loc[0, 'GT'] = 'KO'
        with self.assertRaises(ValueError): runner.animal_vectors(self.frame, 'fixture')
