"""Artificial event timing tests, never empirical episode-prediction evidence."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from psi_qeeg.events import history_windows, prepare_event_dataset, train_event_models, predict_event_windows
from psi_qeeg.analysis import analyze, human_acquisition_contract
from psi_qeeg.quality import QualityPolicy
from psi_qeeg.recording import Recording

HAS_ML = importlib.util.find_spec('sklearn') is not None and importlib.util.find_spec('joblib') is not None


class HumanAcquisitionTests(unittest.TestCase):
    def test_known_waveform_epochs_retain_actual_preprocessing_and_qc(self):
        t = np.arange(2000) / 250
        recording = Recording(pd.DataFrame({'time_s': t, 'Fp1': 10 * np.sin(2 * np.pi * 10 * t)}), {
            'sampling_rate_hz': 250, 'eeg_unit': 'uV', 'eeg_columns': ['Fp1'],
            'channel_locations': {'Fp1': 'synthetic_Fp1'}, 'reference': 'synthetic_reference',
            'reference_verified': True, 'recording_context': 'synthetic_unit_test'})
        table, summary = analyze(recording, line_hz=50, policy=QualityPolicy(max_abs_uv=15), compute_bicoherence=False)
        self.assertTrue(table.accepted.all())
        contract = json.loads(table.acquisition_contract.iat[0])
        self.assertEqual(contract, summary['acquisition_contract'])
        self.assertEqual(contract['preprocessing']['notch_hz'], 50)
        self.assertEqual(contract['quality_policy']['max_abs_uv'], 15)
        self.assertEqual(contract['epoch_s'], 4)
        rejected, strict = analyze(recording, policy=QualityPolicy(max_abs_uv=5), compute_bicoherence=False)
        self.assertFalse(rejected.accepted.any())
        self.assertNotEqual(contract, strict['acquisition_contract'])


class EventTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frame = pd.DataFrame([{'subject_id': f'fixture-{s}', 'session_id': 'artificial-session',
                                    'time_s': float(t), 'epoch_end_s': float(t + 4), 'accepted': True,
                                    'recording_context': 'synthetic_unit_test', 'alpha': float(t + s)}
                                   for s in range(4) for t in range(0, 120, 4)])
        self.contract = human_acquisition_contract(Recording(pd.DataFrame(), {
            'sampling_rate_hz': 250, 'eeg_unit': 'uV', 'eeg_columns': ['C1'],
            'channel_locations': {'C1': 'synthetic_scalp_site'}, 'reference': 'synthetic_reference',
            'reference_verified': True, 'recording_context': 'synthetic_unit_test'}), QualityPolicy(), 4, None, False)
        self.frame['acquisition_contract'] = json.dumps(self.contract)
        self.events = pd.DataFrame([{'subject_id': f'fixture-{s}', 'session_id': 'artificial-session',
                                     'event_id': kind, 'event_type': kind, 'onset_s': 40 if kind == 'meltdown' else 80,
                                     'offset_s': 44 if kind == 'meltdown' else 84, 'label_source': 'consensus',
                                     'source': 'synthetic_unit_test', 'review_status': 'confirmed'}
                                    for s in range(4) for kind in ('meltdown', 'psychosis_onset')])
        self.coverage = pd.DataFrame([{'subject_id': f'fixture-{s}', 'session_id': 'artificial-session',
                                       'start_s': 0, 'end_s': 120, 'label_source': 'consensus',
                                       'source': 'synthetic_unit_test', 'review_status': 'complete',
                                       'event_types_reviewed': 'meltdown;psychosis_onset'} for s in range(4)])

    def prepare(self):
        for name, frame in [('features', self.frame), ('events', self.events), ('coverage', self.coverage)]:
            frame.to_csv(self.root / f'{name}.csv', index=False)
        return prepare_event_dataset(self.root / 'features.csv', self.root / 'events.csv', self.root / 'coverage.csv',
                                     self.root / 'dataset', ['alpha'], history_s=8, horizon_s=12, min_lead_s=4)

    def test_history_never_reads_future_rows_or_crosses_sessions(self):
        first, names, _ = history_windows(self.frame, ['alpha'], history_s=8)
        self.frame.loc[self.frame.time_s >= 40, 'alpha'] = 100000
        changed, _, _ = history_windows(self.frame, ['alpha'], history_s=8)
        pd.testing.assert_frame_equal(first[first.prediction_cutoff_s <= 40][names], changed[changed.prediction_cutoff_s <= 40][names])
        self.frame['session_id'] = np.where(self.frame.time_s < 40, 'session-a', 'session-b')
        windows, _, _ = history_windows(self.frame, ['alpha'], history_s=8)
        self.assertFalse(((windows.history_start_s < 40) & (windows.prediction_cutoff_s > 40)).any())

    def test_rejected_and_missing_history_skipped(self):
        self.frame.loc[self.frame.time_s == 20, 'accepted'] = False
        windows, _, report = history_windows(self.frame, ['alpha'], history_s=8)
        self.assertGreater(report['qc_or_gap_windows_skipped'], 0)
        self.assertFalse(windows.prediction_cutoff_s.isin([24, 28]).any())

    def test_small_real_time_gap_is_not_hidden_by_large_clock(self):
        self.frame.loc[self.frame.time_s >= 20, ['time_s', 'epoch_end_s']] += 0.01
        self.frame[['time_s', 'epoch_end_s']] += 1000000
        windows, _, report = history_windows(self.frame, ['alpha'], history_s=8)
        self.assertEqual(report['qc_or_gap_windows_skipped'], 4)
        self.assertFalse(np.isclose(windows.prediction_cutoff_s, 1000024.01, atol=1e-6, rtol=0).any())

    def test_exact_future_horizon_and_current_event_exclusion(self):
        self.prepare()
        windows = pd.read_csv(self.root / 'dataset/windows.csv')
        self.assertTrue(windows[windows.prediction_cutoff_s == 28].target_meltdown.eq(1).all())
        self.assertTrue(windows[windows.prediction_cutoff_s == 24].target_meltdown.eq(0).all())
        self.assertFalse(windows.prediction_cutoff_s.isin([40, 44, 48]).any())
        self.assertTrue((windows.prediction_cutoff_s + 12 <= 120).all())

    def test_absent_followup_not_assumed_negative(self):
        self.coverage = self.coverage.iloc[:0]
        with self.assertRaisesRegex(ValueError, 'follow-up'): self.prepare()

    def test_unreviewed_coverage_gap_censors_windows(self):
        self.coverage['end_s'] = 60
        self.prepare()
        windows = pd.read_csv(self.root / 'dataset/windows.csv')
        self.assertTrue((windows.prediction_cutoff_s <= 48).all())

    def test_cooccurring_events_are_two_targets(self):
        self.events.loc[self.events.event_type == 'psychosis_onset', ['onset_s', 'offset_s']] = [40, 44]
        self.prepare()
        windows = pd.read_csv(self.root / 'dataset/windows.csv')
        rows = windows[windows.prediction_cutoff_s == 28]
        self.assertTrue(rows.target_meltdown.eq(1).all() and rows.target_psychosis_onset.eq(1).all())

    def test_eeg_guesses_and_unadjudicated_psychosis_rejected(self):
        self.events['label_source'] = 'EEG_guess'
        with self.assertRaises(ValueError): self.prepare()
        self.events['label_source'] = 'participant'
        with self.assertRaisesRegex(ValueError, 'adjudication'): self.prepare()

    def test_outcomes_and_identifiers_never_predictors(self):
        self.frame['target_meltdown'] = 1
        for feature in ('subject_id', 'target_meltdown', 'onset_s', 'offset_s', 'start_s', 'end_s',
                        'time_to_event_s', 'review_status', 'event_types_reviewed', 'ONSET_S'):
            self.frame[feature] = 1
            with self.assertRaises(ValueError): history_windows(self.frame, [feature], history_s=8)

    def test_mixed_synthetic_and_real_preparation_rejected(self):
        self.frame.loc[self.frame.subject_id == 'fixture-0', 'recording_context'] = 'awake_observational_research'
        with self.assertRaisesRegex(ValueError, 'kept separate'): self.prepare()

    def test_missing_or_incompatible_acquisition_metadata_rejected(self):
        self.frame = self.frame.drop(columns='acquisition_contract')
        with self.assertRaisesRegex(ValueError, 'acquisition contract'): self.prepare()
        self.frame['acquisition_contract'] = json.dumps(self.contract)
        changed = dict(self.contract, reference='other_reference')
        self.frame.loc[self.frame.subject_id == 'fixture-0', 'acquisition_contract'] = json.dumps(changed)
        with self.assertRaisesRegex(ValueError, 'one matching'): self.prepare()

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_human_inference_requires_matching_full_acquisition_contract(self):
        import copy
        self.prepare()
        train_event_models(self.root / 'dataset/windows.csv', self.root / 'dataset/dataset_manifest.json', self.root / 'model')
        for key, value in [('reference', 'other_reference'), ('channel_locations', {'C1': 'other_scalp_site'}),
                           ('eeg_unit', 'V'), ('sampling_rate_hz', 500), ('signal_scope', 'raw_continuous_lfp'),
                           ('acquisition_filters', 'other_hardware_filter'), ('reference_verified', False),
                           ('require_sqi', True), ('preprocessing.notch_hz', 50), ('quality_policy.max_abs_uv', 600)]:
            with self.subTest(key=key):
                contract = copy.deepcopy(self.contract)
                parent, _, child = key.partition('.')
                if child: contract[parent][child] = value
                else: contract[key] = value
                changed = self.frame.copy(); changed['acquisition_contract'] = json.dumps(contract)
                changed.to_csv(self.root / 'changed.csv', index=False)
                with self.assertRaisesRegex(ValueError, 'acquisition contract'):
                    predict_event_windows(self.root / 'model/event_models.joblib', self.root / 'changed.csv', self.root / 'bad')
                self.assertFalse((self.root / 'bad/event_window_predictions.csv').exists())

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_human_contract_is_retained_and_legacy_model_rejected(self):
        import joblib
        manifest = self.prepare()
        self.assertEqual(manifest['acquisition_contract'], self.contract)
        windows = pd.read_csv(self.root / 'dataset/windows.csv')
        self.assertEqual(json.loads(windows.acquisition_contract.iat[0]), self.contract)
        train_event_models(self.root / 'dataset/windows.csv', self.root / 'dataset/dataset_manifest.json', self.root / 'model')
        bundle = joblib.load(self.root / 'model/event_models.joblib')
        bundle['manifest'].pop('acquisition_contract')
        joblib.dump(bundle, self.root / 'legacy.joblib')
        with self.assertRaisesRegex(ValueError, 'acquisition contract'):
            predict_event_windows(self.root / 'legacy.joblib', self.root / 'features.csv', self.root / 'bad')

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_subject_splits_model_reload_and_causal_inference(self):
        self.prepare()
        report = train_event_models(self.root / 'dataset/windows.csv', self.root / 'dataset/dataset_manifest.json', self.root / 'model')
        self.assertEqual(report['data_scope'], 'synthetic_fixture')
        for fold in report['folds']:
            self.assertNotIn(fold['held_out_subject'], fold['training_subjects'])
        status = predict_event_windows(self.root / 'model/event_models.joblib', self.root / 'features.csv', self.root / 'prediction')
        self.assertEqual(status['clinical_validation'], 'not_established')
        self.assertEqual(status['probability_calibration'], 'not_established')
        self.assertTrue((self.root / 'prediction/event_window_predictions.csv').exists())

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_cli_requires_annotations_and_keeps_synthetic_scope(self):
        from psi_qeeg.cli import main
        self.prepare()
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(['train-events', str(self.root / 'dataset/windows.csv'), '--manifest',
                         str(self.root / 'dataset/dataset_manifest.json'), '--out', str(self.root / 'cli-model')])
        self.assertEqual(code, 0)
        report = json.loads((self.root / 'cli-model/event_evaluation.json').read_text())
        self.assertEqual(report['data_scope'], 'synthetic_fixture')

    def test_shorter_than_required_lead_is_not_a_negative_example(self):
        self.events.loc[self.events.event_type == 'meltdown', 'onset_s'] = 39
        self.prepare()
        windows = pd.read_csv(self.root / 'dataset/windows.csv')
        self.assertFalse(windows.prediction_cutoff_s.eq(36).any())

    def test_gap_in_reviewed_followup_is_not_filled(self):
        second = self.coverage.copy()
        self.coverage['end_s'] = 24
        second['start_s'] = 36
        self.coverage = pd.concat([self.coverage, second], ignore_index=True)
        self.prepare()
        windows = pd.read_csv(self.root / 'dataset/windows.csv')
        self.assertFalse(windows.prediction_cutoff_s.isin([28, 32, 36]).any())

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_changed_window_table_rejected_and_missing_classes_not_trained(self):
        self.events = self.events[self.events.event_type == 'meltdown']
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'positive and negative'):
            train_event_models(self.root / 'dataset/windows.csv', self.root / 'dataset/dataset_manifest.json', self.root / 'model')
        with (self.root / 'dataset/windows.csv').open('a') as stream: stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            train_event_models(self.root / 'dataset/windows.csv', self.root / 'dataset/dataset_manifest.json', self.root / 'model')

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_diagnosis_model_cannot_be_used_for_event_inference(self):
        import joblib
        joblib.dump({'models': {'diagnosis': object()}}, self.root / 'wrong.joblib')
        with self.assertRaisesRegex(ValueError, 'diagnosis-group'):
            predict_event_windows(self.root / 'wrong.joblib', self.root / 'features.csv', self.root / 'prediction')

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_mixed_training_scope_rejected_even_with_matching_table_hash(self):
        self.prepare(); path = self.root / 'dataset/windows.csv'
        table = pd.read_csv(path); table.loc[0, 'recording_context'] = 'awake_observational_research'; table.to_csv(path, index=False)
        manifest_path = self.root / 'dataset/dataset_manifest.json'
        manifest = json.loads(manifest_path.read_text()); manifest['windows_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'kept separate'):
            train_event_models(path, manifest_path, self.root / 'bad')

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_legacy_outcome_descriptors_cannot_be_trained_with_a_matching_manifest(self):
        self.prepare(); path = self.root / 'dataset/windows.csv'
        table = pd.read_csv(path); table.columns = [name.replace('__alpha', '__onset_s') for name in table.columns]
        table.to_csv(path, index=False)
        manifest_path = self.root / 'dataset/dataset_manifest.json'
        manifest = json.loads(manifest_path.read_text()); manifest['input_features'] = ['onset_s']
        manifest['model_features'] = [name.replace('__alpha', '__onset_s') for name in manifest['model_features']]
        manifest['windows_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest(); manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'future outcomes'):
            train_event_models(path, manifest_path, self.root / 'bad')

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_inference_rejects_changed_epoch_duration_and_real_use_of_fixture(self):
        self.prepare(); train_event_models(self.root / 'dataset/windows.csv', self.root / 'dataset/dataset_manifest.json', self.root / 'model')
        changed = self.frame.copy(); changed['time_s'] /= 2; changed['epoch_end_s'] /= 2
        changed.to_csv(self.root / 'changed.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'epoch duration'):
            predict_event_windows(self.root / 'model/event_models.joblib', self.root / 'changed.csv', self.root / 'bad')
        changed = self.frame.copy(); changed['recording_context'] = 'awake_observational_research'
        changed.to_csv(self.root / 'changed.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'Synthetic fixture'):
            predict_event_windows(self.root / 'model/event_models.joblib', self.root / 'changed.csv', self.root / 'bad')
