"""Download and align a public IBL video/LFP excerpt; no attack classifier.

Extra dependencies: ibl-neuropixel (spikeglx/mtscomp), ffmpeg and ffprobe.
The source is the IBL autism collection, not a psychiatric-event dataset.
Subject genotype and anatomical reference location are not inferred from names.
"""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from psi_qeeg.animal_qeeg import band_powers

BUCKET = 'https://ibl-brain-wide-map-public.s3.amazonaws.com/'
SESSION = '274dacc9-05b2-4154-b4f7-8f6dd46efcc8'
PREFIX = 'data/angelakilab/Subjects/SH023/2021-10-06/001/'
CHANNELS = [32, 96, 128, 192]
BANDS = {'delta_1_4': (1, 4), 'theta_4_12': (4, 12),
         'beta_12_30': (12, 30), 'gamma_30_80': (30, 80), 'high_80_100': (80, 100)}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, size, headers=None):
    """Bounded request with retries; a ranged request must return the exact range."""
    if size > 128 * 1024 * 1024:
        raise ValueError('Excerpt exceeds the 128 MiB request limit')
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers or {}), timeout=40) as response:
                if headers and 'Range' in headers:
                    expected = headers['Range'].replace('bytes=', 'bytes ') + '/'
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                        raise ValueError('Server did not honor the requested byte range')
                data = response.read(size + 1)
                if len(data) != size:
                    raise ValueError('Source byte count mismatch')
                return data
        except (OSError, ValueError):
            if attempt == 2:
                raise


def inventory():
    query = urlencode({'list-type': '2', 'prefix': PREFIX, 'max-keys': 1000})
    with urlopen(BUCKET + '?' + query, timeout=30) as response:
        root = ET.fromstring(response.read(2_000_000))
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    if root.find('s:IsTruncated', ns).text != 'false':
        raise ValueError('Complete session inventory required')
    return [{'key': item.find('s:Key', ns).text,
             'size': int(item.find('s:Size', ns).text),
             'etag': item.find('s:ETag', ns).text.strip('"')}
            for item in root.findall('s:Contents', ns)]


def choose(files, token, suffix):
    matches = [f for f in files if token in f['key'] and f['key'].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {token}{suffix}, found {len(matches)}')
    return matches[0]


def download_small(item, cache, receipt):
    path = cache / item['key'].split('/001/')[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    if item['size'] > 8_000_000 or '-' in item['etag']:
        raise ValueError('Small source needs a whole-object MD5 ETag')
    data = path.read_bytes() if path.exists() else b''
    if len(data) != item['size'] or hashlib.md5(data).hexdigest() != item['etag']:
        data = fetch(BUCKET + item['key'], item['size'])
        if hashlib.md5(data).hexdigest() != item['etag']:
            raise ValueError('Source ETag MD5 mismatch')
        path.write_bytes(data)
    receipt.append({**item, 'sha256': digest(data), 'whole_object_md5_verified': True})
    return path


def align_probe_times(times, sync):
    if sync.ndim != 2 or sync.shape[1] != 2 or not np.isfinite(sync).all():
        raise ValueError('Require finite probe/session clock pairs')
    if not np.all(np.diff(sync, axis=0) > 0):
        raise ValueError('Clock pairs must increase strictly')
    if times[0] < sync[0, 0] or times[-1] > sync[-1, 0]:
        raise ValueError('Refuse clock extrapolation')
    return np.interp(times, sync[:, 0], sync[:, 1])


def validate_movement_intervals(movement):
    """Published bouts must have positive duration and ordered start times."""
    if (movement.ndim != 2 or movement.shape[1] != 2 or not np.isfinite(movement).all()
            or np.any(movement[:, 1] <= movement[:, 0]) or np.any(np.diff(movement[:, 0]) < 0)):
        raise ValueError('Invalid independent movement intervals: require strictly positive duration and ordered starts')


def render_annotations(out, report):
    """Render measured wheel intervals, without labeling attacks or video review."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import signal

    fps = report['source_video_fps']
    times = np.load(out / 'video_session_timestamps.npy')
    events = pd.read_csv(out / 'movement_annotations.csv')
    def ass_time(value):
        centiseconds = round(value * 100)
        hours, rem = divmod(centiseconds, 360000)
        minutes, rem = divmod(rem, 6000)
        seconds, fraction = divmod(rem, 100)
        return f'{hours}:{minutes:02d}:{seconds:02d}.{fraction:02d}'
    lines = ['[Script Info]', 'ScriptType: v4.00+', 'PlayResX: 640', 'PlayResY: 512',
             '[V4+ Styles]',
             'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
             'Style: Title,DejaVu Sans,17,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,3,2,0,8,10,10,8,1',
             'Style: Clock,DejaVu Sans,18,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,3,2,0,2,10,10,10,1',
             'Style: Event,DejaVu Sans,23,&H0000CCFF,&H0000CCFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,2,0,2,10,10,55,1',
             '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
    duration = len(times) / fps
    lines.append(f'Dialogue: 0,0:00:00.00,{ass_time(duration)},Title,,0,0,0,,SH023 | measured task movement\\NNo psychiatric attack labels')
    # Time labels follow the released frame clock, not nominal camera FPS.
    step = 1
    for frame in range(0, len(times), step):
        end = min(frame + step, len(times))
        lines.append(f'Dialogue: 0,{ass_time(frame / fps)},{ass_time(end / fps)},Clock,,0,0,0,,Session {times[frame]:.3f} s | source frame {report["source_frame_interval"][0] + frame}')
    for row in events.itertuples():
        start = max(0, row.clip_start_frame) / fps
        end = min(len(times), row.clip_end_frame_exclusive) / fps
        if end > start:
            lines.append(f'Dialogue: 1,{ass_time(start)},{ass_time(end)},Event,,0,0,0,,WHEEL MOVEMENT | {row.session_start_s:.3f} s')
    subtitle = out / 'movement_overlay.ass'
    subtitle.write_text('\n'.join(lines) + '\n')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(out / 'source_video_excerpt.mp4'),
                    '-vf', f'subtitles={subtitle.resolve()}', '-an', '-c:v', 'libx264',
                    '-preset', 'fast', '-crf', '23', str(out / 'annotated_video_lfp.mp4')], check=True, timeout=90)
    data = np.load(out / 'lfp_excerpt_uv.npz')
    x, clock = data['signal_uv'], data['session_time_s']
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), constrained_layout=True)
    display = signal.resample_poly(x, up=1, down=5, axis=0)
    display_clock = clock[::5]
    shown = (display_clock >= 180) & (display_clock <= 240)
    for col, channel in enumerate(CHANNELS):
        axes[0].plot(display_clock[shown], display[shown, col] + col * 1400, linewidth=.35, label=f'raw channel {channel}')
    axes[0].set(ylabel='LFP display <250 Hz\n(uV; vertical offsets)', xlim=(180, 240))
    axes[0].legend(loc='upper right', ncol=2, fontsize=8)
    f, t, psd = signal.spectrogram(x[:, 0], fs=report['sampling_rate_hz'], window='hann',
                                  nperseg=5000, noverlap=2500, detrend='constant', scaling='density', mode='psd')
    chosen = (f >= 1) & (f <= 100)
    st = np.interp(t + report['probe_clock_window_s'][0], data['probe_time_s'], clock)
    mesh = axes[1].pcolormesh(st, f[chosen], 10 * np.log10(np.maximum(psd[chosen], 1e-12)), shading='auto', cmap='viridis')
    axes[1].set(ylabel='Frequency (Hz)', xlim=(180, 240), ylim=(1, 100))
    fig.colorbar(mesh, ax=axes[1], label='dB re 1 uV^2/Hz')
    for row in events.itertuples():
        for ax in axes:
            ax.axvline(row.session_start_s, color='#c15c00', alpha=.75, linewidth=.8)
            ax.axvspan(row.session_start_s, row.session_end_s, color='#e7a329', alpha=.13)
        axes[2].broken_barh([(row.session_start_s, row.session_end_s - row.session_start_s)], (.1, .6), facecolors='#c15c00')
    axes[2].set(xlim=(180, 240), ylim=(0, 1), yticks=[], xlabel='Experiment clock (s)', ylabel='Measured\nwheel bouts')
    fig.suptitle('Public IBL SH023: synchronized video / raw LFP / measured task movement\nExploratory signal analysis; no artifact adjudication or attack prediction', fontsize=11)
    fig.savefig(out / 'synchronized_lfp_movement.png', dpi=160)
    plt.close(fig)


def prepare(cache, out, lfp_window=(80, 320), video_window=(180, 240)):
    import mtscomp
    import spikeglx

    out.mkdir(parents=True, exist_ok=True)
    files = inventory()
    (out / 'session_source_inventory.json').write_text(json.dumps(files, indent=2) + '\n')
    sources = []
    get = lambda token, suffix: download_small(choose(files, token, suffix), cache, sources)
    camera = np.load(get('_ibl_bodyCamera.times.', '.npy'), allow_pickle=False)
    sync = np.load(get('/probe00/_spikeglx_ephysData_g0_t0.imec0.sync.', '.npy'), allow_pickle=False)
    header = json.loads(get('/probe00/_spikeglx_ephysData_g0_t0.imec0.lf.', '.ch').read_text())
    meta_path = get('/probe00/_spikeglx_ephysData_g0_t0.imec0.lf.', '.meta')
    metadata = spikeglx.read_meta_data(meta_path)
    movement = np.load(get('_ibl_wheelMoves.intervals.', '.npy'), allow_pickle=False)
    trials = pd.read_parquet(get('_ibl_trials.table.', '.pqt'))
    atlas_ids = np.load(get('/probe00/electrodeSites.brainLocationIds_ccf_2017.', '.npy'), allow_pickle=False)
    coordinates = np.load(get('/probe00/electrodeSites.mlapdv.', '.npy'), allow_pickle=False)
    if camera.ndim != 1 or not np.isfinite(camera).all() or not np.all(np.diff(camera) > 0):
        raise ValueError('Require finite, ordered frame timestamps')
    validate_movement_intervals(movement)
    fs = float(header['sample_rate'])
    if fs != metadata['imSampRate'] or header['n_channels'] != metadata['nSavedChans']:
        raise ValueError('Compression/acquisition metadata mismatch')
    counts = np.asarray(header['chunk_bounds'], dtype=np.int64)
    offsets = np.asarray(header['chunk_offsets'], dtype=np.int64)
    first_sample, last_sample = (round(x * fs) for x in lfp_window)
    if not 0 <= first_sample < last_sample <= counts[-1]:
        raise ValueError('LFP excerpt outside source sample bounds')
    first = np.searchsorted(counts, first_sample, side='right') - 1
    last = np.searchsorted(counts, last_sample, side='left')
    lo, hi = int(offsets[first]), int(offsets[last])
    raw_source = choose(files, '/probe00/_spikeglx_ephysData_g0_t0.imec0.lf.', '.cbin')
    range_cache = cache / f'lfp_range_{lo}_{hi}.cbin'
    range_receipt = range_cache.with_suffix('.json')
    prior = json.loads(range_receipt.read_text()) if range_receipt.exists() else {}
    compressed = range_cache.read_bytes() if range_cache.exists() else b''
    if len(compressed) != hi - lo or prior.get('source_etag') != raw_source['etag'] or prior.get('sha256') != digest(compressed):
        compressed = fetch(BUCKET + raw_source['key'], hi - lo,
                           {'Range': f'bytes={lo}-{hi - 1}', 'If-Match': '"' + raw_source['etag'] + '"'})
        range_cache.parent.mkdir(parents=True, exist_ok=True)
        range_cache.write_bytes(compressed)
        range_receipt.write_text(json.dumps({'source_etag': raw_source['etag'], 'sha256': digest(compressed)}))
    partial_header = {k: v for k, v in header.items() if not k.startswith('sha1_')}
    partial_header['chunk_bounds'] = (counts[first:last + 1] - counts[first]).tolist()
    partial_header['chunk_offsets'] = (offsets[first:last + 1] - lo).tolist()
    reader = mtscomp.Reader(n_threads=2)
    reader.open(range_cache, partial_header)
    int_data = reader[first_sample - counts[first]:last_sample - counts[first], CHANNELS]
    reader.close()
    gains = spikeglx._conversion_sample2v_from_meta(metadata)['lf'][CHANNELS] * 1e6
    if not np.isfinite(gains).all() or not np.all(gains > 0):
        raise ValueError('Invalid source voltage conversion')
    uv = int_data.astype(np.float32) * gains.astype(np.float32)
    if uv.shape != (last_sample - first_sample, len(CHANNELS)) or not np.isfinite(uv).all():
        raise ValueError('Incomplete/nonfinite calibrated signal')
    probe_times = np.arange(first_sample, last_sample, dtype=float) / fs
    session_times = align_probe_times(probe_times, sync)
    np.savez_compressed(out / 'lfp_excerpt_uv.npz', signal_uv=uv, probe_time_s=probe_times,
                        session_time_s=session_times, raw_channel_indices=CHANNELS,
                        sampling_rate_hz=fs, atlas_ids=atlas_ids[CHANNELS], mlapdv_um=coordinates[CHANNELS])
    rows = []
    for start in range(0, len(uv) - round(4 * fs) + 1, round(4 * fs)):
        for column, channel in enumerate(CHANNELS):
            x = uv[start:start + round(4 * fs), column]
            power = band_powers(x, fs, BANDS, (1, 100))
            rows.append({'epoch_id': start // round(4 * fs), 'session_start_s': session_times[start],
                         'session_end_s': align_probe_times(np.array([probe_times[start] + 4]), sync)[0],
                         'raw_channel_index': channel, 'std_uv': float(x.std()), 'peak_abs_uv': float(np.abs(x).max()),
                         **{k + '_power_uv2': v for k, v in power['band_power'].items()},
                         **{k + '_relative_power': v for k, v in power['relative_power'].items()}})
    pd.DataFrame(rows).to_csv(out / 'lfp_epoch_powers.csv', index=False)
    video = choose(files, '_iblrig_bodyCamera.raw.', '.mp4')
    url = BUCKET + video['key']
    command = ['ffprobe', '-v', 'error', '-rw_timeout', '20000000', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate,avg_frame_rate,nb_frames,duration,start_time,width,height', '-of', 'json', url]
    info = json.loads(subprocess.run(command, check=True, capture_output=True, text=True, timeout=50).stdout)['streams'][0]
    fps = float(Fraction(info['avg_frame_rate']))
    if int(info['nb_frames']) != len(camera) or info['r_frame_rate'] != info['avg_frame_rate'] or float(info['start_time']) != 0:
        raise ValueError('Cannot establish source frame-index mapping')
    first_frame, last_frame = np.searchsorted(camera, video_window)
    if not 0 <= first_frame < last_frame <= len(camera):
        raise ValueError('Video excerpt outside aligned frame bounds')
    clip_times = camera[first_frame:last_frame]
    np.save(out / 'video_session_timestamps.npy', clip_times)
    annotated = movement[(movement[:, 1] >= clip_times[0]) & (movement[:, 0] <= clip_times[-1])]
    pd.DataFrame([{'event_id': f'wheel_{i}', 'animal_id': 'SH023', 'session_id': SESSION,
                  'event_type': 'task_wheel_movement', 'session_start_s': float(a), 'session_end_s': float(b),
                  'clip_start_frame': int(np.searchsorted(clip_times, a)),
                  'clip_end_frame_exclusive': int(np.searchsorted(clip_times, b)),
                  'label_source': 'published_wheel_encoder_intervals',
                  'video_expert_review': False, 'abnormal_behavior': 'unassessed'}
                 for i, (a, b) in enumerate(annotated)]).to_csv(out / 'movement_annotations.csv', index=False)
    selected_trials = trials[(trials.intervals_1 >= clip_times[0]) & (trials.intervals_0 <= clip_times[-1])]
    selected_trials.to_csv(out / 'source_task_trials_in_clip.csv', index=True)
    clip_path = out / 'source_video_excerpt.mp4'
    command = ['ffmpeg', '-v', 'error', '-y', '-rw_timeout', '20000000', '-ss', f'{first_frame / fps:.9f}',
               '-i', url, '-frames:v', str(last_frame - first_frame), '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '24', '-fps_mode', 'passthrough', str(clip_path)]
    subprocess.run(command, check=True, timeout=180)
    output_probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                  '-show_entries', 'stream=nb_frames', '-of', 'json', str(clip_path)],
                                 check=True, capture_output=True, text=True)
    if int(json.loads(output_probe.stdout)['streams'][0]['nb_frames']) != len(clip_times):
        raise ValueError('Clip and timestamp counts disagree')
    report = {'dataset': 'IBL Autism 2025', 'dataset_doi': '10.6084/m9.figshare.30024880',
              'source_docs': 'https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_autism_noel.html',
              'license': 'CC-BY-4.0', 'session_id': SESSION, 'animal_id': 'SH023',
              'subject_genotype': 'unverified; do not infer KO from nickname',
              'signal_scope': 'real_raw_intracranial_LFP_excerpt', 'sampling_rate_hz': fs,
              'raw_channel_indices': CHANNELS, 'uv_per_adc_count': gains.tolist(),
              'reference_location_verified': False, 'reference_note': 'imroTbl retained; anatomical reference location not established',
              'probe_clock_window_s': lfp_window, 'session_clock_window_s': [float(session_times[0]), float(session_times[-1])],
              'complete_4s_epochs': len(rows) // len(CHANNELS), 'band_calculation': 'PSI band_powers: 2s Hann Welch, 50% overlap, constant detrending, exact-edge band integration',
              'bands_hz': BANDS, 'relative_power_denominator_hz': [1, 100],
              'signal_artifact_review': 'not_completed; powers are exploratory, not accepted diagnostic epochs',
              'video_source': video, 'video_source_frames': int(info['nb_frames']),
              'source_video_fps': fps, 'source_frame_interval': [int(first_frame), int(last_frame)],
              'clip_session_window_s': [float(clip_times[0]), float(clip_times[-1])],
              'clip_frames': len(clip_times), 'wheel_events_in_clip': len(annotated),
              'frame_time_mapping': 'Exact released camera timestamp array indexed by original video frame; matched full and excerpt frame counts',
              'probe_time_mapping': 'Piecewise linear interpolation of released probe/session sync pairs; no extrapolation',
              'sources': sources, 'lfp_range': {**raw_source, 'byte_interval_exclusive_end': [lo, hi],
                                               'excerpt_sha256': digest(compressed), 'whole_object_checksum_verified': False,
                                               'integrity': 'Exact HTTP 206 Content-Range and If-Match source ETag; local excerpt SHA256'},
              'attack_onsets_labeled': False, 'behavioral_onset_model_trained': False,
              'one_minute_forecasting_validation': False, 'human_transfer_validation': False,
              'runner_sha256': digest(Path(__file__).read_bytes())}
    (out / 'video_lfp_alignment_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    render_annotations(out, report)
    print(json.dumps({k: report[k] for k in ('animal_id', 'sampling_rate_hz', 'complete_4s_epochs', 'clip_frames', 'wheel_events_in_clip', 'attack_onsets_labeled')}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/model_organism/ibl_sh023'))
    parser.add_argument('--out', type=Path, default=Path('outputs/ibl_video_lfp'))
    args = parser.parse_args()
    prepare(args.cache, args.out)


if __name__ == '__main__':
    main()
