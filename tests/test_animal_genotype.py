"""Synthetic split/linear-rule checks, never evidence of a biological predictor."""

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from psi_qeeg.animal_genotype import SOURCE_FEATURES, _validate, linear_rule, train_source_genotype, verify_source_snapshot

HAS_ML = importlib.util.find_spec('sklearn') is not None and importlib.util.find_spec('joblib') is not None


class GenotypeContractTests(unittest.TestCase):
    def test_explicit_retained_source_snapshot_checks_all_bytes_and_pins(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); data = b'complete synthetic source bytes'
            (root / 'fixture.csv').write_bytes(data)
            expected = {'fixture.csv': hashlib.sha256(data).hexdigest()}
            record = {'filename': 'fixture.csv', 'sha256': expected['fixture.csv'], 'bytes': len(data)}
            path = root / 'provenance.json'; path.write_text(json.dumps({'files': [record]}))
            self.assertEqual(verify_source_snapshot(root, path, expected), [record])
            (root / 'fixture.csv').write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError, 'mismatch'): verify_source_snapshot(root, path, expected)

    def test_retained_source_snapshot_cannot_change_pins_or_omit_sources(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root / 'fixture.csv').write_bytes(b'abc')
            path = root / 'provenance.json'; expected = {'fixture.csv': hashlib.sha256(b'abc').hexdigest()}
            path.write_text(json.dumps({'files': []}))
            with self.assertRaisesRegex(ValueError, 'exactly'): verify_source_snapshot(root, path, expected)
            path.write_text(json.dumps({'files': [{'filename': 'fixture.csv', 'sha256': '0' * 64, 'bytes': 3}]}))
            with self.assertRaisesRegex(ValueError, 'mismatch'): verify_source_snapshot(root, path, expected)

    def setUp(self):
        self.names = SOURCE_FEATURES[:8]
        rng = np.random.default_rng(3)
        self.frame = pd.DataFrame([{'animal_id': f'fixture-{i:02d}', 'experiment': f'fixture-experiment-{i // 8}',
                                   'genotype': 'KO' if i % 8 < 4 else 'WT',
                                   **dict(zip(self.names, rng.normal(size=8) + (i % 8 >= 4)))} for i in range(16)])

    def test_duplicate_animals_outcome_predictors_and_real_partial_vectors_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one source-condition'):
            _validate(pd.concat([self.frame, self.frame.iloc[:1]]), self.names, 'synthetic_fixture')
        with self.assertRaisesRegex(ValueError, 'cannot be predictors'):
            _validate(self.frame, ['genotype'], 'synthetic_fixture')
        with self.assertRaisesRegex(ValueError, 'complete two-site'):
            _validate(self.frame, self.names, 'real_published_processed_animal_qeeg_features')
        broken = self.frame.copy(); broken.loc[0, self.names[0]] = np.nan
        with self.assertRaisesRegex(ValueError, 'No missing'):
            _validate(broken, self.names, 'synthetic_fixture')

    def fit(self, frame, out):
        return train_source_genotype(frame, self.names, out, data_scope='synthetic_fixture',
                                     c_values=(0.1, 1.0), pca_components=(None, 2), inner_folds=2)

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_nested_animal_splits_final_refit_and_saved_model_rule(self):
        import joblib
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); report = self.fit(self.frame, root)
            self.assertEqual(report['data_scope'], 'synthetic_fixture')
            for fold in report['nested_folds']:
                self.assertNotIn(fold['held_out_animal'], fold['training_animals'])
                for inner in fold['inner_splits']:
                    self.assertFalse(set(inner['training_animals']) & set(inner['validation_animals']))
                    self.assertNotIn(fold['held_out_animal'], inner['training_animals'] + inner['validation_animals'])
                    self.assertEqual(set(inner['training_animals'] + inner['validation_animals']), set(fold['training_animals']))
            for fold in report['experiment_holdouts'].values():
                self.assertFalse(set(fold['training_animals']) & set(fold['test_animals']))
            bundle = joblib.load(root / report['final_fit']['artifact'])
            self.assertEqual(bundle['target'], 'source_WT_KO_genotype_not_behavioral_onset')
            self.assertEqual(set(bundle['training_ids']), set(self.frame.animal_id))
            fixed = joblib.load(root / report['fixed_C1_final_fit']['artifact'])
            self.assertEqual(fixed['model'].named_steps['svm'].C, 1.0)
            self.assertEqual(fixed['model'].named_steps['pca'], 'passthrough')
            self.assertEqual(set(fixed['training_ids']), set(self.frame.animal_id))
            np.testing.assert_allclose(bundle['model'].named_steps['scaler'].mean_, self.frame[self.names].mean().to_numpy())
            coefficients = pd.read_csv(root / 'final_linear_svm_coefficients.csv')
            rule = json.loads((root / 'final_linear_decision_rule.json').read_text())
            manual = self.frame[self.names].to_numpy() @ coefficients.native_log10_power_weight.to_numpy() + rule['native_log10_intercept']
            np.testing.assert_allclose(manual, bundle['model'].decision_function(self.frame[self.names].to_numpy()), atol=1e-9)
            self.assertIn('not an independent test', report['final_fit']['evaluation_status'])

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_outer_test_features_and_label_cannot_change_its_inner_selection(self):
        with tempfile.TemporaryDirectory() as name:
            first = self.fit(self.frame, Path(name) / 'first')
            changed = self.frame.copy(); changed.loc[0, self.names] = 100000; changed.loc[0, 'genotype'] = 'WT'
            second = self.fit(changed, Path(name) / 'second')
            a, b = first['nested_folds'][0], second['nested_folds'][0]
            self.assertEqual(a['held_out_animal'], 'fixture-00')
            self.assertEqual(a['selected_parameters'], b['selected_parameters'])
            self.assertEqual(a['candidate_scores'], b['candidate_scores'])
            self.assertEqual(a['inner_splits'], b['inner_splits'])

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_manual_linear_decision_rule_with_and_without_pca(self):
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        x = self.frame[self.names].to_numpy()
        for pca in ('passthrough', PCA(n_components=2, svd_solver='full')):
            model = Pipeline([('scaler', StandardScaler()), ('pca', pca), ('svm', SVC(kernel='linear'))]).fit(x, self.frame.genotype)
            coefficients, rule = linear_rule(model, self.names)
            test = np.random.default_rng(8).normal(size=(3, len(self.names)))
            np.testing.assert_allclose(test @ coefficients.native_log10_power_weight.to_numpy() + rule['native_log10_intercept'],
                                       model.decision_function(test), atol=1e-9)

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_impossible_inner_selection_is_rejected(self):
        too_small = self.frame[self.frame.animal_id.isin([f'fixture-{i:02d}' for i in (0, 1, 4, 5, 8, 9, 12, 13)])]
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, 'at least two animals'):
                self.fit(too_small, name)
