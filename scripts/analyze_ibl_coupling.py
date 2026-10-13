"""Retained SH023 video/LFP coupling study; exploratory task movement only."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from psi_qeeg.coupling import GAMMA, THETA, benjamini_hochberg, window_coupling

HASHES = {'lfp_excerpt_uv.npz': 'bdaedac3674eb2791acd61f4b842a9e5d902c1a578edde731c5b7bb50533990f',
          'movement_annotations.csv': 'fe49ebacb747b95b66e97da39220efdb56af211552904623e94b6b5342bbc3f8',
          'video_lfp_alignment_report.json': '085251b9d3a1f9ee9c9cb0079bfa4972a9101f291da103b2a217e17ac635a75e'}
# Reproduced identical signal/annotations; receipt differs only in the hash of
# the corrected positive-duration interval validator. Both versions are pinned.
CORRECTED_ALIGNMENT_HASH = 'f5c290876610192b5f108816d74fd1066223a48896a376cb072ddb44448f280e'
MOVEMENT_HASH = '49e6ddd78e692b1aeeee023be22973fcef2e19bba284eaa94717ff0bfa38c96e'
MODES = ['original', 'notch_50_60', 'bipolar']
METRICS = ['theta_plv_median', 'theta_coherence_median', 'theta_imaginary_coherency_abs_median',
           'gamma_plv_median', 'gamma_coherence_median', 'within_pac_mi_median']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quiet_window(movements, start, end, guard=2):
    return not np.any((movements[:, 0] < end + guard) & (movements[:, 1] > start - guard))


def quiet_controls(movements, start, end, duration=16, guard=2):
    """Greedy disjoint windows in the complement of guarded encoder bouts."""
    intervals = []
    for a, b in movements:
        a, b = max(start, a - guard), min(end, b + guard)
        if b <= a:
            continue
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], b)
        else:
            intervals.append([a, b])
    gaps, cursor = [], start
    for a, b in intervals:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        gaps.append((cursor, end))
    result = []
    for a, b in gaps:
        # Move one sample-length safety margin away from exact guard edges.
        a += .001
        while a + duration < b - .001:
            result.append((float(a), float(a + duration)))
            a += duration
    return result


def merge_episodes(movements, gap=16):
    episodes = []
    for start, end in movements:
        if episodes and start - episodes[-1]['end_s'] < gap:
            episodes[-1]['end_s'] = max(episodes[-1]['end_s'], float(end))
            episodes[-1]['source_bouts'] += 1
        else:
            episodes.append({'episode_id': 'episode_' + str(len(episodes)),
                             'start_s': float(start), 'end_s': float(end), 'source_bouts': 1})
    return episodes


def run(source, movements_path, out, surrogates=199):
    out.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name, expected in HASHES.items():
        path = source / name
        actual = sha256(path)
        if actual != expected and not (name == 'video_lfp_alignment_report.json' and actual == CORRECTED_ALIGNMENT_HASH):
            raise ValueError('Retained excerpt source changed; review before reuse: ' + name)
        provenance.append({'filename': name, 'sha256': actual, 'bytes': path.stat().st_size})
    if sha256(movements_path) != MOVEMENT_HASH:
        raise ValueError('Published whole-session wheel intervals changed')
    provenance.append({'filename': movements_path.name, 'sha256': MOVEMENT_HASH,
                       'bytes': movements_path.stat().st_size})
    data = np.load(source / 'lfp_excerpt_uv.npz', allow_pickle=False)
    recording = json.loads((source / 'video_lfp_alignment_report.json').read_text())
    x, times = data['signal_uv'], data['session_time_s']
    fs = float(data['sampling_rate_hz'])
    movements = np.load(movements_path, allow_pickle=False)
    annotations = pd.read_csv(source / 'movement_annotations.csv')
    if x.shape != (600000, 4) or fs != 2500 or not np.isfinite(x).all() or not np.all(np.diff(times) > 0):
        raise ValueError('Incomplete or unrecognized retained signal')
    if not np.array_equal(data['raw_channel_indices'], [32, 96, 128, 192]):
        raise ValueError('Original channel contract mismatch')
    if (movements.ndim != 2 or movements.shape[1] != 2 or not np.isfinite(movements).all()
            or np.any(movements[:, 1] <= movements[:, 0]) or np.any(np.diff(movements[:, 0]) < 0)):
        raise ValueError('Invalid wheel measurement intervals')
    # CSV float parsing can move an onset by ~1e-14 s. Restore the exact
    # deposited encoder timestamp, so the target cannot count as its own
    # preceding bout because of a numerical boundary discrepancy.
    for index, row in annotations.iterrows():
        match = np.flatnonzero(np.isclose(movements[:, 0], row.session_start_s, atol=1e-6, rtol=0)
                               & np.isclose(movements[:, 1], row.session_end_s, atol=1e-6, rtol=0))
        if len(match) != 1:
            raise ValueError('Video annotation must match one deposited encoder interval')
        annotations.loc[index, ['session_start_s', 'session_end_s']] = movements[match[0]]
    if recording['session_id'] != '274dacc9-05b2-4154-b4f7-8f6dd46efcc8':
        raise ValueError('Wrong session')
    names = ['raw_' + str(c) for c in data['raw_channel_indices']]
    dt = float(np.median(np.diff(times)))
    start, end = float(times[0]), float(times[-1] + dt)
    requests = []
    def add(identifier, role, a, b, target=None):
        if a < start or b > end or b <= a:
            return
        i, j = int(np.searchsorted(times, a)), int(np.searchsorted(times, b))
        if j - i < 8 * fs or not np.all(times[i:j] < b):
            raise ValueError('Window must contain past-only samples before cutoff')
        overlap = movements[(movements[:, 0] < b) & (movements[:, 1] > a)]
        future = movements[(movements[:, 0] >= b) & (movements[:, 0] < target)] if target is not None else []
        requests.append({'window_id': identifier, 'role': role, 'analysis_id': f'{i}:{j}',
                         'start_sample': i, 'end_sample_exclusive': j,
                         'window_start_s': float(a), 'window_end_s': float(b),
                         'last_signal_sample_s': float(times[j - 1]),
                         'target_onset_s': target, 'lead_to_target_s': target - b if target is not None else None,
                         'encoder_bouts_overlapping': len(overlap),
                         'wheel_quiet_with_2s_guard': quiet_window(movements, a, b),
                         'wheel_quiet_with_past_2s_guard': not np.any((movements[:, 0] < b) & (movements[:, 1] > a - 2)),
                         'intervening_onsets_before_target': len(future),
                         'abnormal_behavior_verified': False})
    window = round(16 * fs)
    for i in range(0, len(x) - window + 1, round(2 * fs)):
        j = i + window
        a, b = float(times[i]), float(times[j - 1] + dt)
        # Use exact source indices for the regular grid; floating-boundary
        # roundoff is avoided by shifting the end less than half one sample.
        add(f'grid_{i // round(2 * fs)}', 'sliding_grid', a, b - dt * .01)
    controls = quiet_controls(movements, start, end)
    for index, (a, b) in enumerate(controls):
        add('quiet_' + str(index), 'quiet_control', a, b)
    episodes = [e for e in merge_episodes(movements) if e['start_s'] < end and e['end_s'] > start]
    for event_id, onset in [(row.event_id, float(row.session_start_s)) for row in annotations.itertuples()] + [(e['episode_id'], e['start_s']) for e in episodes]:
        for role, a, b in [('pre_onset', onset - 16, onset),
                           ('after_onset', onset, onset + 16),
                           ('sixty_seconds_before', onset - 76, onset - 60),
                           ('earlier_reference', onset - 48, onset - 32)]:
            add(event_id + '__' + role, role, a, b, onset)
    (out / 'analysis_protocol.json').write_text(json.dumps({
        'source': provenance, 'animal_id': 'SH023', 'duration_s': 16, 'grid_step_s': 2,
        'theta_hz': THETA, 'gamma_hz': GAMMA, 'modes': MODES, 'surrogates': surrogates,
        'seed': 20261013, 'edge_trim_s': 1, 'analysis_sampling_rate_hz': 500,
        'surrogate_exclusion_s': 1, 'surrogate_boundary_excluded': True,
        'surrogate_p_resolution': 1 / (surrogates + 1),
        'control_selection': 'Greedy disjoint 16s windows with no encoder bout within 2s on either side',
        'episode_grouping': 'Merge published bouts separated by <16s, using the whole session table',
        'multiplicity_family': 'All unique window/mode/local-and-first-last-directed PAC relationships; duplicated requested windows reuse one estimate',
        'measurement_target': 'published task wheel movement, not an abnormal event',
        'operational_choices': 'Exploratory declared analysis choices, not a preregistered clinical protocol'
    }, indent=2) + '\n')
    unique = {r['analysis_id']: r for r in requests}
    pairs, pac, summaries, shift_receipts = [], [], [], []
    for number, (analysis_id, spec) in enumerate(unique.items()):
        block = x[spec['start_sample']:spec['end_sample_exclusive']]
        for mode in MODES:
            seed = 20261013 + int(hashlib.sha256((analysis_id + mode).encode()).hexdigest()[:8], 16)
            estimates = window_coupling(block, fs, names, mode, surrogates, seed)
            shift_receipts.append({'analysis_id': analysis_id, 'mode': mode, 'seed': seed,
                                   'samples_after_edge_trim': estimates['samples_after_edge_trim'],
                                   'surrogate_shifts_samples': estimates['surrogate_shifts_samples']})
            for row in estimates['pairs']:
                pairs.append({'analysis_id': analysis_id, 'mode': mode, **row})
            for row in estimates['pac']:
                pac.append({'analysis_id': analysis_id, 'mode': mode, **row})
            summary = {'analysis_id': analysis_id, 'mode': mode, 'window_start_s': spec['window_start_s'],
                       'window_end_s': spec['window_end_s'], 'max_abs_uv': estimates['max_abs_uv'],
                       'median_std_uv': estimates['median_std_uv']}
            for metric in ('theta_plv', 'theta_coherence', 'theta_imaginary_coherency_abs',
                           'gamma_plv', 'gamma_coherence', 'gamma_imaginary_coherency_abs'):
                summary[metric + '_median'] = float(np.median([r[metric] for r in estimates['pairs']]))
            local = [p for p in estimates['pac'] if p['relationship'] == 'within_channel']
            summary['within_pac_mi_median'] = float(np.median([p['tort_mi'] for p in local]))
            summaries.append(summary)
        if number % 30 == 0:
            print(f'Calculated {number + 1}/{len(unique)} unique windows', flush=True)
    pac_frame = pd.DataFrame(pac)
    pac_frame['surrogate_q_bh'] = benjamini_hochberg(pac_frame.surrogate_p.to_numpy())
    pac_frame['phase_amplitude_distribution'] = pac_frame.phase_amplitude_distribution.map(json.dumps)
    pair_frame, summary_frame = pd.DataFrame(pairs), pd.DataFrame(summaries)
    local = pac_frame[pac_frame.relationship == 'within_channel'].groupby(['analysis_id', 'mode'])
    q = local.surrogate_q_bh.agg(['min', 'max']).rename(columns={'min': 'within_pac_q_min', 'max': 'within_pac_q_max'})
    summary_frame = summary_frame.merge(q, on=['analysis_id', 'mode'], validate='one_to_one')
    req = pd.DataFrame(requests)
    comparisons = req.merge(summary_frame.drop(columns=['window_start_s', 'window_end_s']), on='analysis_id', validate='many_to_many')
    grid = comparisons[comparisons.role == 'sliding_grid']
    quiet = comparisons[comparisons.role == 'quiet_control']
    overview = {}
    for mode in MODES:
        part = grid[grid['mode'] == mode]
        reference = quiet[quiet['mode'] == mode]
        overview[mode] = {'grid_windows': len(part), 'quiet_control_windows': len(reference),
                          'grid_medians': {k: float(part[k].median()) for k in METRICS},
                          'quiet_medians': {k: float(reference[k].median()) for k in METRICS}}
    changes = []
    for mode in MODES:
        reference = quiet[quiet['mode'] == mode]
        for spec in comparisons[(comparisons['mode'] == mode) & comparisons.role.isin(['pre_onset', 'after_onset', 'sixty_seconds_before'])].to_dict('records'):
            row = {k: spec[k] for k in ('window_id', 'role', 'mode', 'window_start_s', 'window_end_s', 'target_onset_s', 'encoder_bouts_overlapping', 'wheel_quiet_with_2s_guard', 'wheel_quiet_with_past_2s_guard', 'intervening_onsets_before_target')}
            for metric in METRICS:
                value = spec[metric]
                median = float(reference[metric].median())
                row[metric] = value
                row[metric + '__delta_quiet_median'] = value - median
            changes.append(row)
    pd.DataFrame(changes).to_csv(out / 'event_changes_vs_quiet.csv', index=False)
    pair_frame.to_csv(out / 'coupling_pairs.csv', index=False)
    pac_frame.to_csv(out / 'theta_gamma_pac.csv', index=False)
    summary_frame.to_csv(out / 'unique_window_summary.csv', index=False)
    comparisons.to_csv(out / 'annotated_window_summary.csv', index=False)
    req.to_csv(out / 'window_requests.csv', index=False)
    (out / 'surrogate_shifts.json').write_text(json.dumps(shift_receipts, indent=2) + '\n')
    report = {'animal_id': 'SH023', 'session_id': recording['session_id'], 'independent_animals': 1,
              'source_provenance': provenance, 'theta_hz': THETA, 'gamma_hz': GAMMA,
              'analysis_window_s': 16, 'edge_trim_s': 1, 'grid_step_s': 2,
              'unique_windows': len(unique), 'requested_windows': len(req), 'quiet_control_windows': len(controls),
              'video_source_bouts': len(annotations), 'wheel_bouts_overlapping_lfp': int(np.sum((movements[:, 0] < end) & (movements[:, 1] > start))),
              'episodes_overlapping_lfp': episodes, 'modes': overview,
              'pac_test_family_size': len(pac_frame), 'surrogates_per_test': surrogates,
              'surrogate_exclusion_s': 1, 'surrogate_boundary_excluded': True,
              'surrogate_p_resolution': 1 / (surrogates + 1),
              'pac_q05_by_mode': {mode: int(((pac_frame['mode'] == mode) & (pac_frame.surrogate_q_bh < .05)).sum()) for mode in MODES},
              'phase_amplitude_estimator': 'Tort KL modulation index, 18 phase bins; normalized MVL supplementary',
              'surrogate_control': f'{surrogates} distinct circular amplitude shifts, excluding distances <=1s including both boundaries; plus-one p, BH over all unique window/mode/PAC relationships',
              'coherence_estimator': 'Magnitude-squared Welch coherency; 2s Hann, 50% overlap, constant detrend; integrated band mean',
              'imaginary_control': 'Mean absolute imaginary coherency over each band; not a guarantee of neural origin',
              'bipolar_derivations': ['raw_32-raw_96', 'raw_128-raw_192'],
              'filtering': 'Each supplied window alone: antialias 2500->500Hz; optional notch50/60 Q30; fourth-order bandpass/Hilbert; discard 1s each edge',
              'past_only_boundary': 'No input sample at or after each requested window end; all preprocessing is window-contained',
              'subject_genotype': recording['subject_genotype'], 'reference_location_verified': False,
              'artifact_expert_review_complete': False, 'behavioral_model_trained': False,
              'abnormal_event_onsets_verified': False, 'one_minute_prediction_validated': False,
              'human_transfer_validated': False,
              'limitations': ['One mouse; channels, bouts and overlapping windows are not independent subjects',
                              'Quiet controls mean no measured wheel bout, not no motion/muscle/reference artifact',
                              'High PLV/coherence may reflect shared reference or common sources',
                              'PAC surrogate rejection does not rule out transient or nonsinusoidal waveform effects',
                              'A fixed periodic oscillation may retain PAC after a circular shift; non-rejection is not proof of absent coupling',
                              'Overlapping tests are dependent; BH-adjusted exploratory p values do not establish event-level significance or FDR under arbitrary dependence',
                              'Declared 16s windows dilute subsecond movement; no adjudicated abnormal event or forecasting evaluation'],
              'method_sources': ['https://doi.org/10.1152/jn.00106.2010', 'https://doi.org/10.1016/j.jneumeth.2014.01.002',
                                 'https://doi.org/10.1016/j.clinph.2004.04.029', 'https://doi.org/10.1016/j.conb.2014.08.002',
                                 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html'],
              'software': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__},
              'runner_sha256': sha256(__file__), 'estimator_sha256': sha256(Path(__file__).resolve().parents[1] / 'src/psi_qeeg/coupling.py')}
    (out / 'coupling_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    render_report(out, report, comparisons, pac_frame, movements)
    print(json.dumps({'quiet_control_windows': len(controls), 'modes': overview,
                      'pac_q05_by_mode': report['pac_q05_by_mode']}, indent=2), flush=True)
    return report


def render_report(out, report, comparisons, pac, movements):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    colors = {'original': '#2166ac', 'notch_50_60': '#737373', 'bipolar': '#d6604d'}
    titles = [('theta_plv_median', 'Theta PLV\n4-12 Hz'),
              ('theta_coherence_median', 'Theta coherence\n(MSC)'),
              ('gamma_coherence_median', 'Gamma coherence\n30-80 Hz (MSC)'),
              ('within_pac_mi_median', 'Theta-gamma PAC\nlocal median MI')]
    grid = comparisons[comparisons.role == 'sliding_grid']
    for ax, (metric, label) in zip(axes, titles):
        for mode in MODES:
            part = grid[grid['mode'] == mode].sort_values('window_end_s')
            ax.plot(part.window_end_s, part[metric], color=colors[mode], label=mode.replace('_', ' '),
                    linewidth=1.2, linestyle='--' if mode == 'notch_50_60' else '-')
        for a, b in movements:
            if a < 321 and b > 79:
                ax.axvspan(max(a, 80), min(b, 320), color='#d7b365', alpha=.17, linewidth=0)
        for onset in (200.3823164, 217.4893164, 218.0643164, 222.2743164, 223.2473164):
            ax.axvline(onset, color='#8c510a', alpha=.45, linewidth=.7)
        ax.set(ylabel=label, xlim=(95, 320))
        if metric != 'within_pac_mi_median':
            ax.set_ylim(0, 1.02)
        ax.grid(alpha=.2)
    part = grid[grid['mode'] == 'original'].sort_values('window_end_s')
    axes[1].plot(part.window_end_s, part.theta_imaginary_coherency_abs_median,
                 color='black', linestyle='--', linewidth=1, label='original: |imaginary coherency|')
    axes[0].legend(loc='lower left', ncol=3)
    axes[1].legend(loc='lower left', fontsize=8)
    axes[-1].set_xlabel('Experiment clock: end of trailing 16-second window (s)')
    fig.suptitle('SH023: real LFP connectivity and theta-gamma coupling\nOrange bands = measured wheel bouts; one mouse, no abnormal-event prediction', fontsize=12)
    fig.savefig(out / 'coupling_timecourse.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    centers = (np.arange(18) + .5) * 360 / 18 - 180
    for r, (mode, channel) in enumerate([('original', 'raw_32'), ('bipolar', 'raw_32-raw_96')]):
        for c, role in enumerate(['pre_onset', 'after_onset']):
            w = comparisons[(comparisons.window_id == 'wheel_0__' + role) & (comparisons['mode'] == mode)].iloc[0]
            row = pac[(pac.analysis_id == w.analysis_id) & (pac['mode'] == mode)
                      & (pac.theta_channel == channel) & (pac.gamma_channel == channel)].iloc[0]
            distribution = np.asarray(json.loads(row.phase_amplitude_distribution))
            axes[r, c].bar(centers, distribution, width=18, color=colors[mode])
            axes[r, c].axhline(1 / 18, color='black', linestyle='--', linewidth=1)
            axes[r, c].set_title(f'{channel}, {role.replace("_", " ")}\nMI={row.tort_mi:.5f}; surrogate q={row.surrogate_q_bh:.3f}', fontsize=10)
            axes[r, c].set_ylabel('Normalized mean gamma amplitude')
            axes[r, c].set_xlabel('Theta phase (degrees)')
            axes[r, c].set_ylim(0, .07)
    fig.suptitle('First video movement onset: 200.382316 s\n16 seconds before vs 16 seconds after; dashed line = uniform phase distribution', fontsize=12)
    fig.savefig(out / 'theta_gamma_phase_distributions.png', dpi=160)
    plt.close(fig)
    lines = ['SH023: exploratory PLV / coherence / theta-gamma coupling',
             'Executed on retained public IBL LFP; DOI 10.6084/m9.figshare.30024880; CC-BY-4.0.',
             'Credit IBL and Noel et al., Nature Neuroscience 2025, DOI 10.1038/s41593-025-01965-8.',
             '', 'Data: one mouse; 240 seconds; four original channels, 2500 samples/s.',
             'Target annotations: measured wheel movement, not abnormal behavior, meltdown or psychosis.',
             f'Sliding windows: 16s, step 2s. Disjoint guarded wheel-quiet controls: {report["quiet_control_windows"]}.',
             'Four of the five video bouts form one cluster under the declared 16s-gap grouping.',
             '', 'Methods:',
             'PLV = |mean(exp(i*(phase_x-phase_y)))| after bandpass and Hilbert phase.',
             'Magnitude-squared coherence = |Pxy|^2/(Pxx*Pyy); Welch 2s Hann, 50% overlap; band mean.',
             'Tort MI = sum(P_j*log(P_j/(1/18)))/log(18); P_j from mean gamma amplitude in theta phase bins.',
             'Theta 4-12Hz; gamma 30-80Hz. All preprocessing uses the supplied 16s window only.',
             'Antialias to 500Hz; fourth-order filters; trim 1s at both ends.',
             'Sensitivity: original reference, notch50/60 Q30, disjoint bipolar differences (32-96,128-192).',
             f'PAC: {report["surrogates_per_test"]} circular shifts per test; +/-1s excluded; plus-one p; BH family {report["pac_test_family_size"]}.',
             'Local and ordered first/last phase-amplitude relationships are tested; order is not causal direction.',
             '', 'First video onset: 200.382316s. Values are descriptive, not event-level significance:',
             'mode | period | theta PLV | theta coherence | gamma coherence | local PAC MI']
    for mode in MODES:
        values = report['modes'][mode]['quiet_medians']
        lines.append(f'{mode} | quiet median | {values["theta_plv_median"]:.4f} | {values["theta_coherence_median"]:.4f} | {values["gamma_coherence_median"]:.4f} | {values["within_pac_mi_median"]:.6f}')
        for role in ['pre_onset', 'after_onset', 'sixty_seconds_before']:
            w = comparisons[(comparisons.window_id == 'wheel_0__' + role) & (comparisons['mode'] == mode)].iloc[0]
            lines.append(f'{mode} | {role} | {w.theta_plv_median:.4f} | {w.theta_coherence_median:.4f} | {w.gamma_coherence_median:.4f} | {w.within_pac_mi_median:.6f}')
    lines.extend(['', 'Unique-window PAC q<.05 counts by mode: ' + json.dumps(report['pac_q05_by_mode']),
                  'These are coupling-timing tests; they do not test behavioral prediction.', '', 'Limits:'])
    lines.extend('- ' + value for value in report['limitations'])
    lines.extend(['- Genotype and anatomical reference location remain unverified; no expert artifact adjudication.',
                  '- There is no trained event model, one-minute prediction validation or human transfer validation.',
                  '', 'The CSV tables retain every window, channel pair, PAC relationship and comparison.',
                  'Repository methods/report: https://github.com/netanelcyber/psi-qeeg-bis/pull/3'])
    (out / 'Coupling_Report.txt').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/ibl_video_lfp'))
    parser.add_argument('--movement-intervals', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/ibl_coupling'))
    parser.add_argument('--surrogates', type=int, default=199, help='Distinct circular shifts per PAC test (at least 19)')
    args = parser.parse_args()
    run(args.source, args.movement_intervals, args.out, args.surrogates)


if __name__ == '__main__':
    main()
