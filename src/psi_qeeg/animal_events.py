"""Animal-specific, annotation-dependent behavioral onset SVM research.

Mouse genotype/diagnosis labels never become behavioral-event targets.
Synthetic fixture models are rejected for inference on real animal tables.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .animal_qeeg import ANIMAL_CONTEXTS, MODEL_FAMILIES, PROFILES
from .events import _covered, _predictor_contract, _required, history_windows

TARGET = 'animal_behavioral_event_onset'
MODEL_KIND = 'psi_autism_animal_behavior_svm_v1'
METADATA = ('animal_id', 'session_id', 'species', 'model_family', 'genotype',
            'recording_context', 'signal_scope', 'qeeg_profile', 'acquisition_contract')
REFERENCES = {'expert_video_review', 'consensus_video_review', 'synchronized_behavioral_measurement'}


def _read(path):
    return pd.read_csv(path, dtype={'animal_id': str, 'session_id': str})


def _validate_metadata(frame):
    _required(frame, set(METADATA), 'Animal metadata table')
    if set(frame.species) != {'Mus musculus'} or len(set(frame.model_family)) != 1 or not set(frame.model_family) <= MODEL_FAMILIES:
        raise ValueError('Use one documented Fmr1/Shank3/Cntnap2 mouse model family per behavioral model')
    if not set(frame.recording_context) <= ANIMAL_CONTEXTS or len(set(frame.recording_context)) != 1:
        raise ValueError('Keep real awake-animal data and synthetic fixtures separate')
    expected = {'synthetic_signal_fixture'} if set(frame.recording_context) == {'synthetic_animal_unit_test'} else {'raw_continuous_eeg', 'raw_continuous_lfp'}
    if not set(frame.signal_scope) <= expected:
        raise ValueError('Need continuous pre-event animal EEG/LFP, not averaged ERPs or diagnosis/genotype tables')
    if len(set(frame.qeeg_profile)) != 1 or not set(frame.qeeg_profile) <= set(PROFILES):
        raise ValueError('Use one documented qEEG profile per behavioral model')
    contracts = []
    for value in frame.acquisition_contract.unique():
        try:
            contract = json.loads(value)
            valid = (isinstance(contract, dict) and contract.get('reference_verified') is True and
                     isinstance(contract.get('reference'), str) and bool(contract['reference'].strip()) and
                     contract.get('eeg_unit') == 'uV' and np.isfinite(contract.get('sampling_rate_hz', np.nan)) and
                     contract['sampling_rate_hz'] > 2 * PROFILES[frame.qeeg_profile.iat[0]][1][1] and
                     isinstance(contract.get('channel_locations'), dict) and bool(contract['channel_locations']) and
                     all(isinstance(k, str) and isinstance(v, str) and bool(v.strip()) for k, v in contract['channel_locations'].items()))
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            raise ValueError('Epochs require a valid acquisition contract with channel locations and verified reference')
        contracts.append(contract)
    if len(contracts) != 1:
        raise ValueError('Keep one matching electrode/reference acquisition contract per animal model')
    for _, part in frame.groupby('animal_id'):
        if any(part[key].nunique() != 1 for key in ('species', 'model_family', 'genotype')):
            raise ValueError('An animal cannot change species, model family or genotype')
    for _, part in frame.groupby(['animal_id', 'session_id']):
        if any(part[key].nunique() != 1 for key in METADATA):
            raise ValueError('Animal/session metadata must be consistent')
    return contracts[0]


def _validate_epochs(frame):
    _required(frame, {*METADATA, 'time_s', 'epoch_end_s', 'accepted'}, 'Animal epoch table')
    return _validate_metadata(frame)


def _power_features(contract, profile):
    return {f'{channel}__{band}_{suffix}' for channel in contract['channel_locations']
            for band in PROFILES[profile][0] for suffix in ('power_uv2', 'relative_power')}


def _windows(frame, features, history_s, stride_s):
    contract = _validate_epochs(frame)
    allowed = _power_features(contract, frame.qeeg_profile.iat[0])
    if not set(features) <= allowed:
        raise ValueError('Animal predictors must be selected channel qEEG powers, not genotype, behavior or identifiers')
    mapped = frame.copy()
    mapped['subject_id'] = mapped.animal_id
    windows, names, protocol = history_windows(mapped, features, history_s, stride_s,
                                               allowed_contexts=ANIMAL_CONTEXTS)
    windows = windows.rename(columns={'subject_id': 'animal_id'})
    metadata = frame[list(METADATA)].drop_duplicates()
    windows = windows.merge(metadata, on=['animal_id', 'session_id', 'recording_context'], validate='many_to_one')
    return windows, names, protocol


def prepare_animal_events(features_path, events_path, coverage_path, out, features,
                          behavior_name, behavior_definition, history_s=60, horizon_s=30, min_lead_s=4, stride_s=None):
    if not isinstance(behavior_name, str) or not behavior_name.strip() or behavior_name in {'meltdown', 'psychosis_onset', 'genotype'}:
        raise ValueError('Specify an animal behavioral outcome, not a human psychiatric or genotype label')
    if not isinstance(behavior_definition, str) or len(behavior_definition.strip()) < 10:
        raise ValueError('Specify an observable operational behavioral event definition')
    if not np.isfinite([horizon_s, min_lead_s]).all() or not 0 < min_lead_s <= horizon_s:
        raise ValueError('Need a finite positive horizon and minimum lead')
    frame, events, coverage = (_read(p) for p in (features_path, events_path, coverage_path))
    windows, names, protocol = _windows(frame, features, history_s, stride_s)
    _required(events, {'animal_id', 'session_id', 'event_id', 'event_type', 'behavior_name',
                       'onset_s', 'offset_s', 'label_source', 'source', 'review_status'}, 'Animal event annotations')
    _required(coverage, {'animal_id', 'session_id', 'start_s', 'end_s', 'behavior_name', 'event_types_reviewed',
                         'label_source', 'source', 'review_status'}, 'Animal observation coverage')
    if coverage.empty or not coverage.review_status.eq('complete').all() or not coverage.event_types_reviewed.eq(TARGET).all():
        raise ValueError('Complete independently reviewed animal follow-up is required; missing labels are not negatives')
    if not events.event_type.eq(TARGET).all() or not events.review_status.eq('confirmed').all():
        raise ValueError('Only confirmed animal behavioral onsets are accepted')
    if not set(events.label_source) <= REFERENCES or not set(coverage.label_source) <= REFERENCES:
        raise ValueError('Animal targets need independent video/behavior references; EEG guesses and genotype are not outcomes')
    if not events.behavior_name.eq(behavior_name).all() or not coverage.behavior_name.eq(behavior_name).all():
        raise ValueError('Events and follow-up must review the same declared behavior')
    keys = set(zip(frame.animal_id, frame.session_id))
    for table, left, right in ((events, 'onset_s', 'offset_s'), (coverage, 'start_s', 'end_s')):
        if not set(zip(table.animal_id, table.session_id)) <= keys:
            raise ValueError('Annotation refers to an unknown animal/session')
        bounds = table[[left, right]].to_numpy(float)
        if not np.isfinite(bounds).all() or (bounds[:, 1] <= bounds[:, 0]).any():
            raise ValueError('Annotations require finite increasing bounds on the recording clock')
    if events.duplicated(['animal_id', 'session_id', 'event_id']).any():
        raise ValueError('Animal event identifiers must be unique per session')
    selected, excluded = [], {'unreviewed_follow_up': 0, 'current_or_recent_event': 0, 'insufficient_lead': 0}
    for row in windows.to_dict('records'):
        matched = events[(events.animal_id == row['animal_id']) & (events.session_id == row['session_id'])]
        observed = coverage[(coverage.animal_id == row['animal_id']) & (coverage.session_id == row['session_id'])]
        cutoff, beginning = row['prediction_cutoff_s'], row['history_start_s']
        if not _covered(observed[['start_s', 'end_s']].to_numpy(float), beginning, cutoff + horizon_s):
            excluded['unreviewed_follow_up'] += 1; continue
        if ((matched.onset_s <= cutoff) & (matched.offset_s > beginning)).any():
            excluded['current_or_recent_event'] += 1; continue
        if ((matched.onset_s > cutoff) & (matched.onset_s < cutoff + min_lead_s)).any():
            excluded['insufficient_lead'] += 1; continue
        future = matched[(matched.onset_s >= cutoff + min_lead_s) & (matched.onset_s <= cutoff + horizon_s)]
        row['target_animal_behavioral_event_onset'] = int(not future.empty)
        row['event_ids'] = ';'.join(future.event_id.astype(str))
        selected.append(row)
    if not selected:
        raise ValueError('No past animal windows with reviewed future behavioral outcomes')
    protocol.update(horizon_s=float(horizon_s), min_lead_s=float(min_lead_s))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    path = out / 'animal_windows.csv'; pd.DataFrame(selected).to_csv(path, index=False)
    manifest = {'model_kind': MODEL_KIND, 'target': TARGET, 'use': 'animal_research_only',
                'schema_version': 2, 'acquisition_contract': _validate_epochs(frame),
                'behavior_name': behavior_name, 'behavior_definition': behavior_definition,
                'target_origin': 'independently_confirmed_behavior_and_complete_follow_up',
                'species': 'Mus musculus', 'model_family': frame.model_family.iat[0], 'qeeg_profile': frame.qeeg_profile.iat[0],
                'data_scope': 'synthetic_fixture' if frame.recording_context.iat[0] == 'synthetic_animal_unit_test' else 'supplied_annotated_animal_data',
                'input_features': list(features), 'model_features': names, 'protocol': protocol,
                'rows': len(selected), 'animals': len({r['animal_id'] for r in selected}), 'excluded_windows': excluded,
                'windows_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'source_sha256': {key: hashlib.sha256(Path(p).read_bytes()).hexdigest() for key, p in
                                  (('features', features_path), ('events', events_path), ('coverage', coverage_path))},
                'human_transfer_validation': 'not_established'}
    (out / 'animal_dataset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def train_animal_events(windows_path, manifest_path, out):
    try:
        import joblib
        from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise ValueError("Animal fitting requires: pip install 'psi-qeeg-bis[ml]'") from exc
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('model_kind') != MODEL_KIND or manifest.get('target_origin') != 'independently_confirmed_behavior_and_complete_follow_up' or manifest.get('windows_sha256') != hashlib.sha256(Path(windows_path).read_bytes()).hexdigest():
        raise ValueError('Use the matching annotation-derived animal dataset manifest')
    frame = _read(windows_path); names = manifest['model_features']
    _required(frame, {*METADATA, 'prediction_cutoff_s', 'target_animal_behavioral_event_onset', *names}, 'Prepared animal windows')
    contract = _validate_metadata(frame)
    scope = 'synthetic_fixture' if frame.recording_context.iat[0] == 'synthetic_animal_unit_test' else 'supplied_annotated_animal_data'
    if (manifest.get('schema_version') != 2 or contract != manifest.get('acquisition_contract') or
            scope != manifest.get('data_scope') or frame.model_family.iat[0] != manifest.get('model_family') or
            frame.qeeg_profile.iat[0] != manifest.get('qeeg_profile') or
            not set(manifest['input_features']) <= _power_features(contract, manifest['qeeg_profile']) or
            names != _predictor_contract(manifest['input_features'])):
        raise ValueError('Prepared animal data must match the acquisition, scope and predictor contract')
    if frame.animal_id.nunique() < 3 or frame.duplicated(['animal_id', 'session_id', 'prediction_cutoff_s']).any():
        raise ValueError('At least three animals and unique windows are required for fitting, not sufficient validation')
    if not names or any(not n.startswith(('history_mean__', 'history_sd__', 'history_slope_per_s__')) for n in names):
        raise ValueError('Use prepared causal qEEG history descriptors')
    x, y, groups = frame[names].to_numpy(float), frame.target_animal_behavioral_event_onset.to_numpy(), frame.animal_id.to_numpy()
    if not np.isfinite(x).all() or set(y) != {0, 1}:
        raise ValueError('Need finite predictors and independently observed positive and negative outcomes')
    factory = lambda: make_pipeline(StandardScaler(), SVC(kernel='linear', C=1, class_weight='balanced'))
    prediction, margins, folds = np.empty(len(frame), int), np.empty(len(frame)), []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        if set(y[train]) != {0, 1}: raise ValueError('Every animal-held-out training fold must contain both outcome classes')
        model = factory().fit(x[train], y[train])
        prediction[test], margins[test] = model.predict(x[test]), model.decision_function(x[test])
        folds.append({'held_out_animal': str(groups[test][0]), 'training_animals': sorted(set(groups[train])),
                      'n_train': len(train), 'n_test': len(test)})
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average='binary', zero_division=0)
    frame['held_out_prediction'], frame['held_out_margin'] = prediction, margins
    report = {'model_kind': MODEL_KIND, 'target': TARGET, 'behavior_name': manifest['behavior_name'],
              'use': 'animal_research_only', 'data_scope': manifest['data_scope'], 'evaluation': 'leave_one_animal_out',
              'animals': int(frame.animal_id.nunique()), 'windows': len(frame), 'folds': folds,
              'metrics': {'window_balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
                          'window_precision': float(precision), 'window_recall': float(recall), 'window_f1': float(f1)},
              'human_transfer_validation': 'not_established', 'probability_calibration': 'not_established',
              'limitations': 'Correlated window metrics only; event sensitivity, false alerts/hour, lead time and external/prospective animal validation remain necessary'}
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'animal_held_out_predictions.csv', index=False)
    (out / 'animal_event_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    joblib.dump({'model_kind': MODEL_KIND, 'model': factory().fit(x, y), 'manifest': manifest}, out / 'animal_event_model.joblib')
    return report


def predict_animal_events(model_path, features_path, out):
    try:
        import joblib
    except ImportError as exc:
        raise ValueError("Animal inference requires: pip install 'psi-qeeg-bis[ml]'") from exc
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or bundle.get('model_kind') != MODEL_KIND:
        raise ValueError('Provide an animal behavioral model, not a human/group classifier')
    manifest, epochs = bundle['manifest'], _read(features_path)
    contract = _validate_epochs(epochs)
    if contract != manifest.get('acquisition_contract'):
        raise ValueError('Inference electrode/reference acquisition contract must match the fitted animal model')
    if epochs.model_family.iat[0] != manifest['model_family'] or epochs.qeeg_profile.iat[0] != manifest['qeeg_profile']:
        raise ValueError('Inference model family/profile must match the fitted contract')
    if manifest['data_scope'] == 'synthetic_fixture' and epochs.recording_context.iat[0] != 'synthetic_animal_unit_test':
        raise ValueError('Synthetic fixture models cannot be used on real animal recordings')
    p = manifest['protocol']
    frame, names, inferred = _windows(epochs, manifest['input_features'], p['history_s'], p['stride_s'])
    if not np.isclose(inferred['epoch_s'], p['epoch_s'], atol=1e-6, rtol=0):
        raise ValueError('Inference epoch duration must match the fitted animal model')
    if frame.empty or names != manifest['model_features']:
        raise ValueError('No complete past windows matching the fitted animal model')
    x = frame[names].to_numpy(float)
    frame['research_behavior_prediction'], frame['research_behavior_margin'] = bundle['model'].predict(x), bundle['model'].decision_function(x)
    status = {'use': 'animal_research_only', 'target': TARGET, 'behavior_name': manifest['behavior_name'],
              'data_scope': manifest['data_scope'], 'windows': len(frame), 'protocol': p,
              'probability_calibration': 'not_established', 'human_transfer_validation': 'not_established',
              'interpretation': 'Research future-window decisions, not a validated behavioral warning or human psychiatric prediction'}
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'animal_event_predictions.csv', index=False)
    (out / 'animal_prediction_status.json').write_text(json.dumps(status, indent=2) + '\n')
    return status
