"""Synthetic engineering fixtures, never animal onset-prediction evidence."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from psi_qeeg.animal_qeeg import acquisition_contract, analyze_animal_qeeg, band_powers
from psi_qeeg.animal_events import prepare_animal_events, train_animal_events, predict_animal_events, TARGET
from psi_qeeg.recording import Recording, read_recording
from psi_qeeg.events import history_windows, predict_event_windows

HAS_ML = importlib.util.find_spec('sklearn') is not None and importlib.util.find_spec('joblib') is not None


def metadata():
    return {'sampling_rate_hz': 1000, 'eeg_unit': 'uV', 'eeg_columns': ['C1'],
            'animal_id': 'artificial-animal', 'session_id': 'artificial-session', 'species': 'Mus musculus',
            'model_family': 'Shank3', 'genotype': 'synthetic-KO', 'reference': 'synthetic_reference',
            'reference_verified': True, 'channel_locations': {'C1': 'synthetic_cortical_location'},
            'recording_context': 'synthetic_animal_unit_test', 'signal_scope': 'synthetic_signal_fixture',
            'source': 'synthetic_unit_test', 'license': 'synthetic_fixture',
            'animal_quality_policy': {'max_amplitude_uv': 500, 'min_std_uv': 0.01}}


class AnimalQeegTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        time = np.arange(12000) / 1000
        self.recording = Recording(pd.DataFrame({'time_s': time, 'C1': 4 * np.sin(2 * np.pi * 70 * time) + 3 * np.sin(2 * np.pi * 90 * time)}), metadata())

    def test_known_powers_and_explicit_denominator(self):
        r = band_powers(self.recording.eeg[:, 0], 1000, {'low': (30, 80), 'high': (80, 100)}, (1, 100))
        self.assertAlmostEqual(r['band_power']['low'], 8)
        self.assertAlmostEqual(r['band_power']['high'], 4.5)
        self.assertAlmostEqual(r['relative_power']['low'], 0.64)
        with self.assertRaises(ValueError): band_powers(np.zeros(2000), 160, {'gamma': (30, 80)})

    def test_future_signal_does_not_change_past_epoch_powers(self):
        analyze_animal_qeeg(self.recording, self.root / 'before')
        before = pd.read_csv(self.root / 'before/animal_epoch_features.csv')
        self.recording.frame.loc[self.recording.time >= 8, 'C1'] = 10000
        after = analyze_animal_qeeg(self.recording, self.root / 'after')
        changed = pd.read_csv(self.root / 'after/animal_epoch_features.csv')
        pd.testing.assert_frame_equal(before[before.epoch_end_s <= 8], changed[changed.epoch_end_s <= 8], check_dtype=False)
        self.assertEqual(after['rejected_epochs'], 1)

    def test_missing_and_flat_epochs_are_not_interpolated(self):
        self.recording.frame.loc[self.recording.time < 4, 'C1'] = np.nan
        self.recording.frame.loc[(self.recording.time >= 4) & (self.recording.time < 8), 'C1'] = 0
        r = analyze_animal_qeeg(self.recording, self.root / 'out')
        self.assertEqual((r['accepted_epochs'], r['rejected_epochs']), (1, 2))
        f = pd.read_csv(self.root / 'out/animal_epoch_features.csv')
        self.assertTrue(f.loc[~f.accepted, 'C1__gamma_low_power_uv2'].isna().all())

    def test_metadata_rejects_humans_unverified_reference_and_processed_erp(self):
        for key, value in [('species', 'Homo sapiens'), ('reference_verified', False), ('signal_scope', 'processed_erp')]:
            meta = metadata(); meta[key] = value
            with self.assertRaises(ValueError): analyze_animal_qeeg(Recording(self.recording.frame, meta), self.root / key)

    def test_volts_convert_once_and_cli_retains_synthetic_scope(self):
        frame = self.recording.frame.copy(); frame.C1 /= 1e6
        frame.to_csv(self.root / 'raw.csv', index=False)
        meta = metadata(); meta['eeg_unit'] = 'V'
        (self.root / 'metadata.json').write_text(json.dumps(meta))
        converted = read_recording(self.root / 'raw.csv', self.root / 'metadata.json')
        np.testing.assert_allclose(converted.eeg, self.recording.eeg, atol=1e-10)
        from psi_qeeg.cli import main
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(['analyze-animal-qeeg', str(self.root / 'raw.csv'), '--metadata', str(self.root / 'metadata.json'), '--out', str(self.root / 'cli')])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((self.root / 'cli/animal_qeeg_analysis.json').read_text())['data_scope'], 'synthetic_fixture')


class AnimalEventTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.features = pd.DataFrame([{'animal_id': f'artificial-{s}', 'session_id': 'fixture-session',
            'species': 'Mus musculus', 'model_family': 'Shank3', 'genotype': 'synthetic-KO',
            'recording_context': 'synthetic_animal_unit_test', 'signal_scope': 'synthetic_signal_fixture',
            'qeeg_profile': 'autism_research_1_100', 'time_s': float(t), 'epoch_end_s': float(t + 4),
            'accepted': True, 'C1__gamma_low_power_uv2': float(t + s),
            'acquisition_contract': json.dumps(acquisition_contract(metadata()), sort_keys=True, separators=(',', ':'))}
            for s in range(4) for t in range(0, 120, 4)])
        self.events = pd.DataFrame([{'animal_id': f'artificial-{s}', 'session_id': 'fixture-session',
            'event_id': f'fixture-event-{t}', 'event_type': TARGET, 'behavior_name': 'fixture_grooming',
            'onset_s': t, 'offset_s': t + 4, 'label_source': 'consensus_video_review',
            'source': 'synthetic_unit_test', 'review_status': 'confirmed'} for s in range(4) for t in (40, 80)])
        self.coverage = pd.DataFrame([{'animal_id': f'artificial-{s}', 'session_id': 'fixture-session',
            'start_s': 0, 'end_s': 120, 'behavior_name': 'fixture_grooming', 'event_types_reviewed': TARGET,
            'label_source': 'consensus_video_review', 'source': 'synthetic_unit_test', 'review_status': 'complete'} for s in range(4)])

    def prepare(self):
        for name, frame in [('features', self.features), ('events', self.events), ('coverage', self.coverage)]:
            frame.to_csv(self.root / f'{name}.csv', index=False)
        return prepare_animal_events(self.root / 'features.csv', self.root / 'events.csv', self.root / 'coverage.csv',
            self.root / 'dataset', ['C1__gamma_low_power_uv2'], 'fixture_grooming', 'Synthetic artificial event for engineering tests only', 8, 12, 4)

    def test_causal_targets_and_current_event_censoring(self):
        m = self.prepare(); f = pd.read_csv(self.root / 'dataset/animal_windows.csv')
        self.assertEqual(m['data_scope'], 'synthetic_fixture')
        self.assertTrue(f[f.prediction_cutoff_s == 28].target_animal_behavioral_event_onset.eq(1).all())
        self.assertTrue(f[f.prediction_cutoff_s == 24].target_animal_behavioral_event_onset.eq(0).all())
        self.assertFalse(f.prediction_cutoff_s.isin([40, 44, 48]).any())

    def test_missing_followup_and_wrong_behavior_cannot_be_negatives(self):
        original = self.coverage.copy(); self.coverage = original.iloc[:0]
        with self.assertRaisesRegex(ValueError, 'follow-up'): self.prepare()
        self.coverage = original; self.coverage['behavior_name'] = 'different_behavior'
        with self.assertRaisesRegex(ValueError, 'same declared behavior'): self.prepare()

    def test_eeg_guesses_human_labels_and_genotype_features_rejected(self):
        self.events['label_source'] = 'EEG_guess'
        with self.assertRaises(ValueError): self.prepare()
        self.events['label_source'] = 'consensus_video_review'; self.events['event_type'] = 'meltdown'
        with self.assertRaises(ValueError): self.prepare()
        self.events['event_type'] = TARGET; self.prepare()
        with self.assertRaisesRegex(ValueError, 'predictors'):
            prepare_animal_events(self.root / 'features.csv', self.root / 'events.csv', self.root / 'coverage.csv', self.root / 'bad', ['genotype'], 'fixture_grooming', 'Synthetic artificial behavior criterion')

    def test_averaged_erp_and_mixed_model_families_rejected(self):
        self.features['signal_scope'] = 'averaged_erp'
        with self.assertRaisesRegex(ValueError, 'continuous'): self.prepare()
        self.features['signal_scope'] = 'synthetic_signal_fixture'
        self.features.loc[self.features.animal_id == 'artificial-0', 'model_family'] = 'Fmr1'
        with self.assertRaisesRegex(ValueError, 'one documented'): self.prepare()

    def test_human_default_history_contract_does_not_accept_animals(self):
        frame = self.features.copy(); frame['subject_id'] = frame.animal_id
        with self.assertRaises(ValueError): history_windows(frame, ['C1__gamma_low_power_uv2'], 8)

    def test_incomplete_history_and_followup_gap_are_censored(self):
        self.features.loc[self.features.time_s == 20, 'accepted'] = False
        second = self.coverage.copy(); self.coverage['end_s'] = 60; second['start_s'] = 76
        self.coverage = pd.concat([self.coverage, second], ignore_index=True)
        self.prepare(); f = pd.read_csv(self.root / 'dataset/animal_windows.csv')
        self.assertFalse(f.prediction_cutoff_s.isin([24, 28, 52, 56, 60, 64, 68, 72, 76, 80]).any())

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_held_out_animals_reload_and_inference(self):
        self.prepare(); r = train_animal_events(self.root / 'dataset/animal_windows.csv', self.root / 'dataset/animal_dataset_manifest.json', self.root / 'model')
        self.assertEqual(r['data_scope'], 'synthetic_fixture')
        for fold in r['folds']: self.assertNotIn(fold['held_out_animal'], fold['training_animals'])
        status = predict_animal_events(self.root / 'model/animal_event_model.joblib', self.root / 'features.csv', self.root / 'pred')
        self.assertEqual(status['human_transfer_validation'], 'not_established')
        self.assertEqual(status['probability_calibration'], 'not_established')

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_synthetic_animal_model_cannot_be_used_on_real_data(self):
        self.prepare(); train_animal_events(self.root / 'dataset/animal_windows.csv', self.root / 'dataset/animal_dataset_manifest.json', self.root / 'model')
        self.features['recording_context'] = 'awake_model_animal_research'; self.features['signal_scope'] = 'raw_continuous_lfp'
        self.features.to_csv(self.root / 'real_declared.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'Synthetic fixture'):
            predict_animal_events(self.root / 'model/animal_event_model.joblib', self.root / 'real_declared.csv', self.root / 'pred')

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_human_and_animal_model_bundles_cannot_be_interchanged(self):
        import joblib
        joblib.dump({'models': {'meltdown': object(), 'psychosis_onset': object()}}, self.root / 'human.joblib')
        with self.assertRaisesRegex(ValueError, 'animal behavioral model'): predict_animal_events(self.root / 'human.joblib', self.root / 'features.csv', self.root / 'wrong')
        self.prepare(); train_animal_events(self.root / 'dataset/animal_windows.csv', self.root / 'dataset/animal_dataset_manifest.json', self.root / 'model')
        with self.assertRaisesRegex(ValueError, 'diagnosis-group'): predict_event_windows(self.root / 'model/animal_event_model.joblib', self.root / 'features.csv', self.root / 'wrong2')

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_tampered_table_and_absent_outcome_classes_not_trained(self):
        self.events = self.events.iloc[:0]; self.prepare()
        with self.assertRaisesRegex(ValueError, 'positive and negative'): train_animal_events(self.root / 'dataset/animal_windows.csv', self.root / 'dataset/animal_dataset_manifest.json', self.root / 'model')
        with (self.root / 'dataset/animal_windows.csv').open('a') as stream: stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'matching'): train_animal_events(self.root / 'dataset/animal_windows.csv', self.root / 'dataset/animal_dataset_manifest.json', self.root / 'model')

    def test_mixed_electrode_references_rejected_before_preparation(self):
        contract = acquisition_contract(metadata()); contract['reference'] = 'other_reference'
        self.features.loc[self.features.animal_id == 'artificial-0', 'acquisition_contract'] = json.dumps(contract)
        with self.assertRaisesRegex(ValueError, 'acquisition contract'): self.prepare()

    @unittest.skipUnless(HAS_ML, 'optional ML dependencies')
    def test_inference_requires_fitted_electrodes_reference_and_epoch_duration(self):
        self.prepare(); train_animal_events(self.root / 'dataset/animal_windows.csv', self.root / 'dataset/animal_dataset_manifest.json', self.root / 'model')
        original = self.features.copy()
        for change in ('reference', 'location'):
            contract = acquisition_contract(metadata())
            if change == 'reference': contract['reference'] = 'other_reference'
            else: contract['channel_locations']['C1'] = 'other_anatomical_site'
            self.features = original.copy(); self.features['acquisition_contract'] = json.dumps(contract)
            self.features.to_csv(self.root / 'changed.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'acquisition contract'):
                predict_animal_events(self.root / 'model/animal_event_model.joblib', self.root / 'changed.csv', self.root / 'bad')
        self.features = original.copy()
        self.features['time_s'] /= 2; self.features['epoch_end_s'] /= 2
        self.features.to_csv(self.root / 'two_seconds.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'epoch duration'):
            predict_animal_events(self.root / 'model/animal_event_model.joblib', self.root / 'two_seconds.csv', self.root / 'bad')
