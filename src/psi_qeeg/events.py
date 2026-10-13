"""Causal research windows with independently annotated future events.

Diagnosis groups and EEG deviations are never used to manufacture event labels.
No clinical risk probability or automatic medical alert is produced.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .quality import QualityPolicy
from .signal import BANDS

EVENT_TYPES = ('meltdown', 'psychosis_onset')
CONTEXTS = {'awake_observational_research', 'awake_longitudinal_research', 'synthetic_unit_test'}
IDS = {'subject_id', 'session_id', 'epoch_id', 'time_s', 'epoch_end_s', 'accepted',
       'recording_context', 'group', 'site', 'classification', 'research_label', 'label_source',
       'animal_id', 'species', 'model_family', 'genotype', 'signal_scope', 'qeeg_profile', 'source',
       'onset_s', 'offset_s', 'start_s', 'end_s', 'event_type', 'event_id', 'event_ids',
       'behavior_name', 'behavior_definition', 'review_status', 'event_types_reviewed',
       'target_origin', 'time_to_event_s', 'time_to_onset_s', 'prediction_cutoff_s',
       'history_start_s', 'acquisition_contract'}


def _unmixed_scope(frame):
    synthetic = frame.recording_context.astype(str).str.startswith('synthetic')
    if synthetic.any() and not synthetic.all():
        raise ValueError('Synthetic fixtures and real observations must be kept separate')
    return 'synthetic_fixture' if synthetic.all() else 'supplied_annotated_research_data'


def _read(path):
    return pd.read_csv(path, dtype={'subject_id': str, 'session_id': str})


def _required(frame, names, description):
    if not set(names) <= set(frame) or frame[list(names)].isna().any().any():
        raise ValueError(f'{description} requires complete columns {sorted(names)}')
    for column in names:
        if frame[column].dtype == object and frame[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'{description}: {column} cannot be empty')


def _predictor_contract(features):
    if not features or len(set(features)) != len(features) or any(str(f).strip().casefold() in IDS or str(f).strip().casefold().startswith(('target_', 'event_', 'future_', 'label_')) for f in features):
        raise ValueError('Identifiers, labels and future outcomes cannot be predictors')
    return [f'history_{kind}__{feature}' for kind in ('mean', 'sd', 'slope_per_s') for feature in features]


def _human_acquisition(frame, epoch_s):
    """One canonical EEG derivation, processing and QC policy per model."""
    _required(frame, {'acquisition_contract', 'recording_context'}, 'Human acquisition contract')
    expected_scope = 'synthetic_signal_fixture' if _unmixed_scope(frame) == 'synthetic_fixture' else 'raw_continuous_eeg'
    contracts = []
    for value in frame.acquisition_contract.unique():
        try:
            contract = json.loads(value)
            locations, preprocessing = contract['channel_locations'], contract['preprocessing']
            policy = contract['quality_policy']
            valid = (contract['schema_version'] == 1 and contract['eeg_unit'] == 'uV' and
                     contract['signal_scope'] == expected_scope and contract['reference_verified'] is True and
                     isinstance(contract['reference'], str) and bool(contract['reference'].strip()) and
                     np.isfinite(contract['sampling_rate_hz']) and contract['sampling_rate_hz'] > 80 and
                     np.isclose(contract['epoch_s'], epoch_s, atol=1e-6, rtol=0) and
                     isinstance(locations, dict) and bool(locations) and
                     all(isinstance(k, str) and bool(k.strip()) and isinstance(v, str) and bool(v.strip()) for k, v in locations.items()) and
                     preprocessing['method'] == 'psi_human_qeeg_v1' and
                     preprocessing['bands_hz'] == {k: list(v) for k, v in BANDS.items()} and
                     isinstance(preprocessing['compute_bicoherence'], bool) and
                     (preprocessing['notch_hz'] is None or 0 < preprocessing['notch_hz'] < contract['sampling_rate_hz'] / 2) and
                     contract['quality_method'] == 'psi_assess_epoch_v1' and isinstance(contract['require_sqi'], bool) and
                     isinstance(policy, dict) and set(policy) == {'min_sqi', 'max_abs_uv', 'flat_std_uv', 'emg_warning_db'} and
                     all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in policy.values()) and
                     'acquisition_filters' in contract)
            if valid:
                QualityPolicy(**policy)
        except (ValueError, TypeError, KeyError, AttributeError):
            valid = False
        if not valid:
            raise ValueError('Human acquisition contract requires EEG scope, verified electrodes/reference, uV, epoch duration, preprocessing and QC policy; regenerate legacy tables')
        if contract not in contracts:
            contracts.append(contract)
    if len(contracts) != 1:
        raise ValueError('Use one matching human electrode/reference, preprocessing and QC acquisition contract')
    return contracts[0]


def history_windows(frame, features, history_s=60, stride_s=None, *, allowed_contexts=None):
    """Mean, SD and slope from complete accepted past epochs only; no future labels."""
    required = {'subject_id', 'session_id', 'time_s', 'epoch_end_s', 'accepted', 'recording_context'}
    _required(frame, required, 'Epoch table')
    if not features or len(set(features)) != len(features) or any(f not in frame for f in features):
        raise ValueError('Select unique existing numeric waveform features')
    names = _predictor_contract(features)
    contexts = CONTEXTS if allowed_contexts is None else set(allowed_contexts)
    if not set(frame.recording_context) <= contexts:
        raise ValueError('Event research requires an awake observational context or explicit synthetic fixture')
    _unmixed_scope(frame)
    if frame.duplicated(['subject_id', 'session_id', 'time_s']).any():
        raise ValueError('Duplicate subject/session/epoch times')
    durations = frame.epoch_end_s.to_numpy(float) - frame.time_s.to_numpy(float)
    if not len(durations) or not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError('Epoch times must be finite with positive durations')
    step = float(durations[0])
    if not np.allclose(durations, step, atol=1e-6, rtol=0):
        raise ValueError('Use one documented non-overlapping epoch duration')
    stride_s = step if stride_s is None else float(stride_s)
    if not np.isfinite([history_s, stride_s]).all() or history_s < 2 * step or stride_s < step:
        raise ValueError('History needs at least two epochs; stride must be at least one epoch')
    if not np.isclose(history_s / step, round(history_s / step)) or not np.isclose(stride_s / step, round(stride_s / step)):
        raise ValueError('History and stride must align with the epoch grid')
    quality = frame.accepted.astype(str).str.lower()
    if not set(quality) <= {'true', 'false'}:
        raise ValueError('accepted must explicitly be true or false')
    count, advance = round(history_s / step), round(stride_s / step)
    rows, skipped = [], 0
    for (subject, session), part in frame.groupby(['subject_id', 'session_id'], sort=True):
        part = part.sort_values('time_s').reset_index(drop=True)
        if part.recording_context.nunique() != 1:
            raise ValueError('A session cannot mix recording contexts')
        starts, ends = part.time_s.to_numpy(float), part.epoch_end_s.to_numpy(float)
        if not np.isfinite(np.r_[starts, ends]).all() or (np.diff(starts) < step - 1e-6).any():
            raise ValueError('Epochs must have finite non-overlapping sample times')
        values = part[features].to_numpy(float)
        for last in range(count - 1, len(part), advance):
            first = last + 1 - count
            chunk = values[first:last + 1]
            good = part.accepted.iloc[first:last + 1].astype(str).str.lower().eq('true').all()
            if not good or not np.isfinite(chunk).all() or not np.allclose(starts[first + 1:last + 1], ends[first:last], atol=1e-6, rtol=0):
                skipped += 1
                continue
            times = ends[first:last + 1]
            centered = times - times.mean()
            slope = np.sum(centered[:, None] * (chunk - chunk.mean(axis=0)), axis=0) / np.sum(centered ** 2)
            row = {'subject_id': str(subject), 'session_id': str(session), 'history_start_s': float(starts[first]),
                   'prediction_cutoff_s': float(ends[last]), 'recording_context': part.recording_context.iat[0]}
            row.update(zip(names, np.r_[chunk.mean(axis=0), chunk.std(axis=0, ddof=1), slope]))
            rows.append(row)
    table = pd.DataFrame(rows, columns=['subject_id', 'session_id', 'history_start_s', 'prediction_cutoff_s', 'recording_context'] + names)
    return table, names, {'history_s': float(history_s), 'stride_s': stride_s, 'epoch_s': step, 'qc_or_gap_windows_skipped': skipped}


def _covered(intervals, start, end):
    cursor = start
    for left, right in sorted(intervals, key=lambda pair: (float(pair[0]), float(pair[1]))):
        if left > cursor + 1e-6:
            break
        cursor = max(cursor, right)
        if cursor >= end - 1e-6:
            return True
    return False


def prepare_event_dataset(features_path, events_path, coverage_path, out, features,
                          history_s=60, horizon_s=300, min_lead_s=4, stride_s=None):
    if not np.isfinite([horizon_s, min_lead_s]).all() or not 0 < min_lead_s <= horizon_s:
        raise ValueError('Specify a finite positive minimum lead and prediction horizon')
    frame, events, coverage = _read(features_path), _read(events_path), _read(coverage_path)
    _required(events, {'subject_id', 'session_id', 'event_id', 'event_type', 'onset_s', 'offset_s',
                       'label_source', 'source', 'review_status'}, 'Event annotations')
    _required(coverage, {'subject_id', 'session_id', 'start_s', 'end_s', 'label_source', 'source',
                         'review_status', 'event_types_reviewed'}, 'Observation coverage')
    if coverage.empty or not coverage.review_status.eq('complete').all():
        raise ValueError('Complete independently reviewed follow-up coverage is required; absence of labels is not a negative outcome')
    if not set(coverage.label_source) <= {'clinician', 'consensus'} or any(set(str(v).split(';')) != set(EVENT_TYPES) for v in coverage.event_types_reviewed):
        raise ValueError('Coverage must independently review both event types with clinician/consensus provenance')
    if not set(events.event_type) <= set(EVENT_TYPES) or not events.review_status.eq('confirmed').all():
        raise ValueError('Only confirmed independently annotated meltdown/psychosis_onset events are accepted')
    if not set(events.label_source) <= {'participant', 'clinician', 'consensus'}:
        raise ValueError('Event labels require independent participant/clinician/consensus provenance')
    if events[events.event_type == 'psychosis_onset'].label_source.isin(['participant']).any():
        raise ValueError('Psychosis-onset labels require clinician or consensus adjudication')
    if events.duplicated(['subject_id', 'session_id', 'event_id']).any():
        raise ValueError('Event identifiers must be unique within subject/session')
    keys = set(zip(frame.subject_id.astype(str), frame.session_id.astype(str)))
    for table, start, end in ((events, 'onset_s', 'offset_s'), (coverage, 'start_s', 'end_s')):
        if not set(zip(table.subject_id, table.session_id)) <= keys:
            raise ValueError('Annotations reference an unknown subject/session')
        bounds = table[[start, end]].to_numpy(float)
        if not np.isfinite(bounds).all() or (bounds[:, 1] <= bounds[:, 0]).any():
            raise ValueError('Annotations need finite increasing interval endpoints on the same session clock')
    windows, names, protocol = history_windows(frame, features, history_s, stride_s)
    contract = _human_acquisition(frame, protocol['epoch_s'])
    windows['acquisition_contract'] = json.dumps(contract, sort_keys=True, separators=(',', ':'))
    protocol.update(horizon_s=float(horizon_s), min_lead_s=float(min_lead_s))
    selected, excluded = [], {'unreviewed_follow_up': 0, 'current_or_recent_event': 0, 'insufficient_lead': 0}
    for row in windows.to_dict('records'):
        matched = events[(events.subject_id == row['subject_id']) & (events.session_id == row['session_id'])]
        observed = coverage[(coverage.subject_id == row['subject_id']) & (coverage.session_id == row['session_id'])]
        cutoff, beginning = row['prediction_cutoff_s'], row['history_start_s']
        if not _covered(observed[['start_s', 'end_s']].to_numpy(float), beginning, cutoff + horizon_s):
            excluded['unreviewed_follow_up'] += 1
            continue
        if ((matched.onset_s <= cutoff) & (matched.offset_s > beginning)).any():
            excluded['current_or_recent_event'] += 1
            continue
        if ((matched.onset_s > cutoff) & (matched.onset_s < cutoff + min_lead_s)).any():
            excluded['insufficient_lead'] += 1
            continue
        for kind in EVENT_TYPES:
            future = matched[(matched.event_type == kind) & (matched.onset_s >= cutoff + min_lead_s) & (matched.onset_s <= cutoff + horizon_s)]
            row[f'target_{kind}'] = int(not future.empty)
            row[f'event_ids_{kind}'] = ';'.join(future.event_id.astype(str))
        selected.append(row)
    if not selected:
        raise ValueError('No scoreable past windows with reviewed future outcomes; supply genuine event/follow-up annotations')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    windows_path = out / 'windows.csv'
    pd.DataFrame(selected).to_csv(windows_path, index=False)
    manifest = {'use': 'research_only', 'clinical_validation': 'not_established', 'schema_version': 2,
                'acquisition_contract': contract,
                'input_features': list(features), 'model_features': names, 'protocol': protocol,
                'rows': len(selected), 'subjects': len({r['subject_id'] for r in selected}), 'excluded_windows': excluded,
                'target_origin': 'confirmed_event_onsets_with_independently_reviewed_follow_up',
                'windows_sha256': hashlib.sha256(windows_path.read_bytes()).hexdigest(),
                'source_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in
                                  (('features', features_path), ('events', events_path), ('coverage', coverage_path))}}
    (out / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def train_event_models(windows_path, manifest_path, out):
    try:
        from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        import joblib
    except ImportError as exc:
        raise ValueError("Event training requires: pip install 'psi-qeeg-bis[ml]'") from exc

    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('target_origin') != 'confirmed_event_onsets_with_independently_reviewed_follow_up' or manifest.get('windows_sha256') != hashlib.sha256(Path(windows_path).read_bytes()).hexdigest():
        raise ValueError('Training requires the matching prepared event dataset manifest')
    frame = _read(windows_path)
    names = manifest['model_features']
    if names != _predictor_contract(manifest.get('input_features', [])):
        raise ValueError('Prepared descriptors must match the permitted waveform-feature contract')
    _required(frame, {'subject_id', 'session_id', 'prediction_cutoff_s', 'recording_context', *names,
                       *(f'target_{kind}' for kind in EVENT_TYPES)}, 'Prepared event windows')
    if frame.subject_id.nunique() < 3 or frame.duplicated(['subject_id', 'session_id', 'prediction_cutoff_s']).any():
        raise ValueError('Need at least three subjects and unique subject/session/cutoff windows')
    if not set(frame.recording_context) <= CONTEXTS or not names or any(not n.startswith(('history_mean__', 'history_sd__', 'history_slope_per_s__')) for n in names):
        raise ValueError('Only prepared past-history descriptors can be model features')
    scope = _unmixed_scope(frame)
    contract = _human_acquisition(frame, manifest['protocol']['epoch_s'])
    if manifest.get('schema_version') != 2 or contract != manifest.get('acquisition_contract'):
        raise ValueError('Prepared human acquisition contract must match the event manifest; regenerate legacy datasets')
    x = frame[names].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Event descriptors must be finite; no imputation is performed')
    factory = lambda: make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
    groups = frame.subject_id.to_numpy()
    results, models, folds = {}, {}, []
    for kind in EVENT_TYPES:
        y = frame[f'target_{kind}'].to_numpy()
        if set(y) != {0, 1}:
            raise ValueError(f'{kind}: independently observed positive and negative future outcomes are required')
        prediction = np.empty(len(frame), dtype=int)
        margin = np.empty(len(frame))
        for train, test in LeaveOneGroupOut().split(x, y, groups):
            if set(y[train]) != {0, 1}:
                raise ValueError(f'{kind}: a held-out subject leaves no training class; collect events from more subjects')
            model = factory().fit(x[train], y[train])
            prediction[test], margin[test] = model.predict(x[test]), model.decision_function(x[test])
            folds.append({'event_type': kind, 'held_out_subject': str(groups[test][0]),
                          'training_subjects': sorted(set(groups[train])), 'n_train': len(train), 'n_test': len(test)})
        precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average='binary', zero_division=0)
        results[kind] = {'window_balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
                         'window_precision': float(precision), 'window_recall': float(recall), 'window_f1': float(f1)}
        frame[f'held_out_{kind}_prediction'], frame[f'held_out_{kind}_margin'] = prediction, margin
        models[kind] = factory().fit(x, y)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {'use': 'research_only', 'clinical_validation': 'not_established', 'data_scope': scope,
              'evaluation': 'leave_one_subject_out', 'metrics': results, 'folds': folds,
              'subjects': frame.subject_id.nunique(), 'windows': len(frame), 'protocol': manifest['protocol'],
              'limitations': 'Window metrics only; no calibrated risk probability, event-level alert validation, external/prospective evidence or clinical decision rule'}
    frame.to_csv(out / 'held_out_event_predictions.csv', index=False)
    (out / 'event_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    joblib.dump({'models': models, 'manifest': manifest, 'data_scope': scope}, out / 'event_models.joblib')
    return report


def predict_event_windows(model_path, features_path, out):
    try:
        import joblib
    except ImportError as exc:
        raise ValueError("Event inference requires: pip install 'psi-qeeg-bis[ml]'") from exc
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or set(bundle.get('models', {})) != set(EVENT_TYPES):
        raise ValueError('Provide an explicitly trained event-model bundle, not a diagnosis-group classifier')
    manifest = bundle['manifest']
    protocol = manifest['protocol']
    epochs = _read(features_path)
    frame, names, inferred = history_windows(epochs, manifest['input_features'], protocol['history_s'], protocol['stride_s'])
    if not np.isclose(inferred['epoch_s'], protocol['epoch_s'], atol=1e-6, rtol=0):
        raise ValueError('Inference epoch duration must match the fitted event model')
    if bundle['data_scope'] == 'synthetic_fixture' and _unmixed_scope(epochs) != 'synthetic_fixture':
        raise ValueError('Synthetic fixture models cannot be used on real observations')
    contract = _human_acquisition(epochs, inferred['epoch_s'])
    if manifest.get('schema_version') != 2 or contract != manifest.get('acquisition_contract'):
        raise ValueError('Inference human electrode/reference, units, preprocessing and QC acquisition contract must match the fitted model')
    if names != manifest['model_features'] or frame.empty:
        raise ValueError('No complete past windows matching the trained event feature contract')
    x = frame[names].to_numpy(float)
    for kind in EVENT_TYPES:
        model = bundle['models'][kind]
        frame[f'research_{kind}_prediction'] = model.predict(x)
        frame[f'research_{kind}_margin'] = model.decision_function(x)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'event_window_predictions.csv', index=False)
    status = {'use': 'research_only', 'clinical_validation': 'not_established', 'data_scope': bundle['data_scope'],
              'protocol': protocol, 'windows': len(frame), 'probability_calibration': 'not_established',
              'interpretation': 'Model predictions for the prespecified future interval; not a diagnosis or impending-attack determination'}
    (out / 'prediction_status.json').write_text(json.dumps(status, indent=2) + '\n')
    return status
