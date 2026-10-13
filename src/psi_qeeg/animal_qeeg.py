"""Quantitative animal EEG arrays/epochs; no behavioral or human diagnosis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .signal import band_integral
from .recording import validate

def band_powers(x_uv, sampling_rate_hz, bands, relative_range=None):
    """Welch PSD and exact-edge integrals for explicitly supplied frequency bands.

    This uses PSI's two-second Hann/50%-overlap convention, rather than claiming
    to reproduce every estimator or normalization in the cited animal studies.
    Relative power is optional and always documents its denominator range.
    """
    x = np.asarray(x_uv, dtype=float)
    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0 or x.ndim != 1 or len(x) < round(2 * fs) or not np.isfinite(x).all():
        raise ValueError('Supply at least two seconds of finite, calibrated, one-dimensional EEG/LFP')
    if not bands:
        raise ValueError('Specify at least one named band')
    intervals = [*bands.values()] + ([] if relative_range is None else [relative_range])
    for interval in intervals:
        if len(interval) != 2 or not np.isfinite(interval).all() or not 0 < interval[0] < interval[1] < fs / 2:
            raise ValueError('Band edges must be finite, positive and strictly below Nyquist')
    if relative_range is not None and any(low < relative_range[0] or high > relative_range[1] for low, high in bands.values()):
        raise ValueError('The relative-power denominator must contain every requested band')
    segment = round(2 * fs)
    frequencies, density = signal.welch(x, fs=fs, window='hann', nperseg=segment,
                                       noverlap=segment // 2, detrend='constant', scaling='density')
    powers = {name: band_integral(frequencies, density, *edges) for name, edges in bands.items()}
    denominator = None if relative_range is None else band_integral(frequencies, density, *relative_range)
    return {
        'estimator': 'Welch, two-second Hann, 50% overlap, constant detrending',
        'sampling_rate_hz': fs,
        'frequency_grid_hz': float(frequencies[1] - frequencies[0]),
        'bands_hz': bands,
        'power_unit': 'uV^2',
        'band_power': powers,
        'relative_power_denominator_hz': relative_range,
        'relative_power': None if relative_range is None else
            {name: value / denominator if denominator > 0 else None for name, value in powers.items()},
        'event_prediction': None,
    }


MODEL_FAMILIES = {'Fmr1', 'Shank3', 'Cntnap2'}
ANIMAL_CONTEXTS = {'awake_model_animal_research', 'synthetic_animal_unit_test'}
PROFILES = {
    'shank3_2017_gamma': ({'gamma': (30, 80)}, (1, 80)),
    'autism_research_1_100': ({'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
                            'beta': (13, 30), 'gamma_low': (30, 80), 'gamma_high': (80, 100)}, (1, 100)),
}


def validate_animal_metadata(metadata):
    required = {'animal_id', 'session_id', 'species', 'model_family', 'genotype', 'recording_context',
                'signal_scope', 'reference', 'source', 'license'}
    if any(not isinstance(metadata.get(k), str) or not metadata[k].strip() for k in required):
        raise ValueError('Animal recordings require explicit animal/session, species, model, genotype, context, reference and source metadata')
    if metadata['species'] != 'Mus musculus' or metadata['model_family'] not in MODEL_FAMILIES:
        raise ValueError('This autism-model workflow supports documented Fmr1/Shank3/Cntnap2 mouse cohorts')
    if metadata['recording_context'] not in ANIMAL_CONTEXTS:
        raise ValueError('Use an awake model-animal research context or explicit synthetic animal fixture')
    expected = {'synthetic_signal_fixture'} if metadata['recording_context'] == 'synthetic_animal_unit_test' else {'raw_continuous_eeg', 'raw_continuous_lfp'}
    if metadata['signal_scope'] not in expected:
        raise ValueError('Continuous EEG/LFP is required; processed or averaged ERPs cannot become pre-event qEEG')
    if metadata.get('reference_verified') is not True or metadata.get('eeg_unit') != 'uV':
        raise ValueError('Require a verified reference and input converted once to uV by read_recording')
    locations = metadata.get('channel_locations', {})
    if any(not isinstance(locations.get(ch), str) or not locations[ch].strip() for ch in metadata.get('eeg_columns', [])):
        raise ValueError('Document the recording location of each animal EEG/LFP channel')
    policy = metadata.get('animal_quality_policy', {})
    values = [policy.get('max_amplitude_uv'), policy.get('min_std_uv')]
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v) for v in values) or not values[0] > 0 or values[1] < 0:
        raise ValueError('Prespecify finite animal max_amplitude_uv > 0 and min_std_uv >= 0')
    return policy


def acquisition_contract(metadata):
    """Canonical physical derivation, retained with epochs and fitted models."""
    policy = validate_animal_metadata(metadata)
    return {'schema_version': 2, 'signal_scope': metadata['signal_scope'],
            'sampling_rate_hz': float(metadata['sampling_rate_hz']), 'eeg_unit': 'uV',
            'reference': metadata['reference'], 'reference_verified': True,
            'channel_locations': {ch: metadata['channel_locations'][ch] for ch in metadata['eeg_columns']},
            'acquisition_filters': metadata.get('acquisition_filters', 'not_specified'),
            'quality_method': 'finite_max_amplitude_and_min_std_v1',
            'quality_policy': {key: float(policy[key]) for key in ('max_amplitude_uv', 'min_std_uv')}}


def analyze_animal_qeeg(recording, out, epoch_s=4, profile='autism_research_1_100'):
    """Complete past-contained epoch powers; numerical QC is not artifact cleanup."""
    validate(recording)
    policy = validate_animal_metadata(recording.metadata)
    if profile not in PROFILES or not np.isfinite(epoch_s) or epoch_s < 2:
        raise ValueError('Choose a documented animal profile and finite epochs of at least two seconds')
    bands, denominator = PROFILES[profile]
    if recording.fs / 2 <= denominator[1]:
        raise ValueError('Acquisition sampling rate must preserve the entire animal profile below Nyquist')
    count = round(epoch_s * recording.fs)
    if not np.isclose(count, epoch_s * recording.fs, atol=1e-6, rtol=0):
        raise ValueError('Animal epoch duration must align with the sample clock')
    if len(recording.frame) < count:
        raise ValueError('No complete animal epoch')
    meta, rows = recording.metadata, []
    contract = acquisition_contract(meta)
    values, times = recording.eeg, recording.time
    for first in range(0, len(recording.frame) - count + 1, count):
        x = values[first:first + count]
        reasons = []
        if not np.isfinite(x).all():
            reasons.append('missing_samples')
        else:
            if (np.abs(x) > policy['max_amplitude_uv']).any(): reasons.append('amplitude_limit')
            if (x.std(axis=0) <= policy['min_std_uv']).any(): reasons.append('flat_signal')
        row = {k: meta[k] for k in ('animal_id', 'session_id', 'species', 'model_family', 'genotype', 'recording_context', 'signal_scope', 'source')}
        row.update(epoch_id=first // count, time_s=float(times[first]),
                   epoch_end_s=float(times[first] + count / recording.fs),
                   accepted=not reasons, quality_reasons=';'.join(reasons), qeeg_profile=profile,
                   acquisition_contract=json.dumps(contract, sort_keys=True, separators=(',', ':')))
        if not reasons:
            for column, channel in enumerate(recording.channels):
                result = band_powers(x[:, column], recording.fs, bands, denominator)
                row.update({f'{channel}__{name}_power_uv2': v for name, v in result['band_power'].items()})
                row.update({f'{channel}__{name}_relative_power': v for name, v in result['relative_power'].items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    for channel in recording.channels:
        for name in bands:
            for suffix in ('power_uv2', 'relative_power'):
                key = f'{channel}__{name}_{suffix}'
                if key not in frame: frame[key] = np.nan
    summary = {'use': 'animal_research_only', 'species': meta['species'], 'model_family': meta['model_family'],
               'animal_id': meta['animal_id'], 'session_id': meta['session_id'],
               'data_scope': 'synthetic_fixture' if meta['recording_context'] == 'synthetic_animal_unit_test' else 'supplied_continuous_animal_signal',
               'profile': profile, 'bands_hz': bands, 'relative_denominator_hz': denominator,
               'complete_epochs': len(frame), 'accepted_epochs': int(frame.accepted.sum()),
               'rejected_epochs': int((~frame.accepted).sum()), 'unused_tail_samples': len(recording.frame) % count,
               'quality_policy': policy, 'animal_metadata': meta, 'acquisition_contract': contract,
               'quality_limit': 'Numerical missing/flat/amplitude checks only; movement, muscle and line-noise artifacts require independent review',
               'event_prediction': None, 'SOI': None, 'RDI': None, 'human_transfer_validation': 'not_established'}
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'animal_epoch_features.csv', index=False)
    (out / 'animal_qeeg_analysis.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


