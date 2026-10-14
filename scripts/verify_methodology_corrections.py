"""Verify retained real IBL coupling and subject-balanced RepOD run artifacts.

This validates numerical/provenance receipts, not psychiatric event prediction.
Run after the public-data runners; requires the optional ML dependencies.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.integrate import trapezoid
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(coupling, human, lfp_source, out):
    root = Path(__file__).resolve().parents[1]
    checks = {}
    def check(name, condition):
        if not condition:
            raise ValueError('Verification failed: ' + name)
        checks[name] = True

    c = json.loads((coupling / 'coupling_report.json').read_text())
    pac = pd.read_csv(coupling / 'theta_gamma_pac.csv')
    requests = pd.read_csv(coupling / 'window_requests.csv')
    shifts = json.loads((coupling / 'surrogate_shifts.json').read_text())
    pair = pd.read_csv(coupling / 'coupling_pairs.csv')
    check('coupling_runner_and_estimator_fingerprints',
          c['runner_sha256'] == sha256(root / 'scripts/analyze_ibl_coupling.py') and
          c['estimator_sha256'] == sha256(root / 'src/psi_qeeg/coupling.py'))
    check('coupling_input_fingerprints', all(sha256(lfp_source / row['filename']) == row['sha256']
          for row in c['source_provenance'][:3]))
    check('pac_count_and_unique_family', len(pac) == c['pac_test_family_size'] and
          requests.analysis_id.nunique() == c['unique_windows'] and
          not pac.duplicated(['analysis_id', 'mode', 'theta_channel', 'gamma_channel']).any())
    check('configured_surrogate_count_consistent', pac.surrogate_count.eq(c['surrogates_per_test']).all())
    check('strict_signed_one_second_exclusion', all(
        len(row['surrogate_shifts_samples']) == c['surrogates_per_test'] and
        len(set(row['surrogate_shifts_samples'])) == c['surrogates_per_test'] and
        np.all(np.minimum(row['surrogate_shifts_samples'],
                         row['samples_after_edge_trim'] - np.asarray(row['surrogate_shifts_samples'])) > 500)
        for row in shifts))
    check('all_signal_samples_precede_cutoff', (requests.last_signal_sample_s < requests.window_end_s).all())
    pre = requests[requests.role == 'pre_onset']
    check('pre_onset_samples_precede_target', (pre.last_signal_sample_s < pre.target_onset_s).all())
    distribution = np.array([json.loads(p) for p in pac.phase_amplitude_distribution])
    check('pac_matches_independent_entropy', np.allclose(
        stats.entropy(distribution, np.full(distribution.shape, 1 / 18), axis=1) / np.log(18),
        pac.tort_mi, atol=1e-12, rtol=0))
    check('bh_matches_independent_scipy', np.allclose(
        stats.false_discovery_control(pac.surrogate_p, method='bh'), pac.surrogate_q_bh, atol=1e-12, rtol=0))
    check('empirical_p_grid', np.allclose(pac.surrogate_p * (c['surrogates_per_test'] + 1),
          np.round(pac.surrogate_p * (c['surrogates_per_test'] + 1))) and (pac.surrogate_p > 0).all())

    # Recompute one real pre-bout relationship using independent SciPy
    # binned means / relative entropy, and its saved offsets rather than RNG.
    spec = requests[requests.window_id == 'wheel_0__pre_onset'].iloc[0]
    lfp = np.load(lfp_source / 'lfp_excerpt_uv.npz', allow_pickle=False)
    block = lfp['signal_uv'][spec.start_sample:spec.end_sample_exclusive]
    block = signal.resample_poly(block, 1, 5, axis=0)
    theta = signal.sosfiltfilt(signal.butter(4, [4, 12], 'bandpass', fs=500, output='sos'), block[:, 0])
    gamma = signal.sosfiltfilt(signal.butter(4, [30, 80], 'bandpass', fs=500, output='sos'), block[:, 0])
    phase, amplitude = np.angle(signal.hilbert(theta))[500:-500], np.abs(signal.hilbert(gamma))[500:-500]
    def mi(envelope):
        means = stats.binned_statistic(phase, envelope, statistic='mean', bins=np.linspace(-np.pi, np.pi, 19)).statistic
        return float(stats.entropy(means / means.sum(), np.full(18, 1 / 18)) / np.log(18))
    offsets = next(r['surrogate_shifts_samples'] for r in shifts if r['analysis_id'] == spec.analysis_id and r['mode'] == 'original')
    null = np.array([mi(np.roll(amplitude, offset)) for offset in offsets])
    observed = mi(amplitude)
    saved = pac[(pac.analysis_id == spec.analysis_id) & (pac['mode'] == 'original') &
                (pac.theta_channel == 'raw_32') & (pac.gamma_channel == 'raw_32')].iloc[0]
    check('real_pac_and_null_recomputed_independently',
          np.isclose(observed, saved.tort_mi, atol=1e-12) and
          np.isclose(null.mean(), saved.surrogate_mean_mi, atol=1e-12) and
          np.isclose(null.std(ddof=1), saved.surrogate_sd_mi, atol=1e-12) and
          np.isclose((1 + (null >= observed).sum()) / (len(null) + 1), saved.surrogate_p, atol=1e-12))
    f, coherence = signal.coherence(block[500:-500, 0], block[500:-500, 1], fs=500,
        window='hann', nperseg=1000, noverlap=500, detrend='constant')
    saved_pair = pair[(pair.analysis_id == spec.analysis_id) & (pair['mode'] == 'original') &
                      (pair.channel_a == 'raw_32') & (pair.channel_b == 'raw_96')].iloc[0]
    for band, low, high in [('theta', 4, 12), ('gamma', 30, 80)]:
        mask = (f >= low) & (f <= high)
        check('real_' + band + '_coherence_matches_scipy', np.isclose(
            trapezoid(coherence[mask], f[mask]) / (high - low), saved_pair[band + '_coherence'], atol=1e-12))
    counts = {mode: int(((pac['mode'] == mode) & (pac.surrogate_q_bh < .05)).sum()) for mode in c['modes']}
    check('reported_pac_counts_match_saved_tests', counts == c['pac_q05_by_mode'])

    h = json.loads((human / 'public_cohort_validation.json').read_text())
    table = pd.read_csv(human / 'cohort_features.csv', dtype={'subject_id': str})
    accepted = table[table.accepted.astype(str).str.lower().eq('true')]
    comparison = json.loads((human / 'electrode_comparison.json').read_text())
    check('human_table_fingerprint', sha256(human / 'cohort_features.csv') == h['cohort_features_sha256'])
    check('same_human_epoch_membership_as_original_run', len(table) == 7205 and len(accepted) == 6870 and accepted.subject_id.nunique() == 28)
    for name, result in comparison['comparisons'].items():
        folder = human if name == 'full_scalp' else human / 'electrode_comparison' / name
        weights = pd.read_csv(folder / 'training_subject_weights.csv')
        votes = pd.read_csv(folder / 'held_out_subject_predictions.csv', dtype={'subject_id': str})
        evaluation = json.loads((folder / 'group_evaluation.json').read_text())
        model = joblib.load(folder / 'group_model.joblib')
        check(name + '_one_scaler_unit_per_person', len(weights) == 28 and np.allclose(weights.scaler_mass, 1))
        check(name + '_equal_svm_class_mass', np.allclose(weights.groupby('group').svm_mass.sum(), 14))
        check(name + '_held_out_subject_separation', all(
            fold['held_out_subject'] not in fold['training_subjects'] and len(fold['training_subjects']) == 27
            for fold in evaluation['folds']) and len(evaluation['folds']) == 28)
        features = result['features']
        means = accepted.groupby('subject_id')[features].mean().mean().to_numpy()
        check(name + '_saved_scaler_is_subject_weighted', np.allclose(model.named_steps['standardscaler'].mean_, means, atol=1e-10))
        check(name + '_subject_metrics_recomputed', np.isclose(balanced_accuracy_score(votes.truth, votes.pred), result['subject_balanced_accuracy']) and
              confusion_matrix(votes.truth, votes.pred, labels=result['subject_confusion_matrix']['labels']).tolist() == result['subject_confusion_matrix']['rows_true_cols_pred'])
        check(name + '_model_reload', len(model.predict(accepted[features].to_numpy())) == len(accepted))
    check('no_clinical_event_validation_claim', not c['behavioral_model_trained'] and not c['one_minute_prediction_validated'] and
          h['event_prediction']['status'] == 'not_trained_missing_event_annotations')
    receipt = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Real coupling numerical controls and human diagnosis-group evaluation',
               'checks': checks, 'all_passed': True, 'pac_q05_by_mode': counts,
               'subject_balanced_accuracy': {k: r['subject_balanced_accuracy'] for k, r in comparison['comparisons'].items()},
               'clinical_event_prediction_validated': False,
               'verifier_sha256': sha256(__file__)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coupling', type=Path, default=Path('outputs/ibl_coupling_corrected'))
    parser.add_argument('--human', type=Path, default=Path('outputs/repod_subject_balanced'))
    parser.add_argument('--lfp-source', type=Path, default=Path('outputs/ibl_video_lfp'))
    parser.add_argument('--out', type=Path, default=Path('outputs/methodology_verification.json'))
    args = parser.parse_args()
    verify(args.coupling, args.human, args.lfp_source, args.out)
