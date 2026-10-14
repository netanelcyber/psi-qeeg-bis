"""Exploratory window-contained PLV, coherence and phase-amplitude coupling.

These numerical estimators do not identify abnormal behavior or train PSI.
All filtering/resampling uses only samples supplied in the analysis window.
"""

from itertools import combinations

import numpy as np
from scipy import signal

from .signal import band_integral

THETA = (4.0, 12.0)
GAMMA = (30.0, 80.0)


def phase_amplitude_index(phase, amplitude, bins=18):
    """Tort MI: KL divergence of mean amplitude by phase bin / log(bins)."""
    phase, amplitude = np.asarray(phase, float), np.asarray(amplitude, float)
    if (phase.ndim != 1 or phase.shape != amplitude.shape or not len(phase)
            or not np.isfinite(phase).all() or not np.isfinite(amplitude).all()
            or np.any(amplitude < 0) or not isinstance(bins, int) or bins < 3):
        raise ValueError('Require aligned finite phase and nonnegative amplitude')
    labels = np.floor(((phase + np.pi) % (2 * np.pi)) / (2 * np.pi) * bins).astype(int)
    counts = np.bincount(labels, minlength=bins)
    if not np.all(counts > 0):
        raise ValueError('All phase bins must have observations')
    means = np.bincount(labels, weights=amplitude, minlength=bins) / counts
    if not means.sum() > 0:
        raise ValueError('Gamma amplitude must have positive mass')
    distribution = means / means.sum()
    positive = distribution > 0
    mi = 1 + np.sum(distribution[positive] * np.log(distribution[positive])) / np.log(bins)
    return float(np.clip(mi, 0, 1)), distribution


def benjamini_hochberg(p_values):
    """BH adjusted p values; the analysis defines and records the test family."""
    p = np.asarray(p_values, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Finite p values in [0,1] required')
    if not len(p):
        return p.copy()
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def window_coupling(x_uv, fs, names, mode='original', surrogates=199, seed=20261013):
    """Estimate declared bands on a >=8s block; primary study uses 16s blocks.

    `notch_50_60` is an alternative to original, not a preprocessing default.
    `bipolar` uses two disjoint differences and requires exactly four inputs.
    PAC includes each local channel and both first/last cross-channel directions.
    Circular surrogates test timing alignment, not whether its origin is neural.
    """
    x = np.asarray(x_uv, dtype=float)
    fs = float(fs)
    if (not np.isfinite(fs) or fs < 500 or x.ndim != 2 or x.shape[1] < 2
            or len(names) != x.shape[1] or len(set(names)) != len(names)
            or not np.isfinite(x).all() or len(x) < round(8 * fs)
            or np.any(x.std(axis=0) <= 1e-8)):
        raise ValueError('Require >=8s of finite nonflat multichannel signal, fs>=500 Hz')
    if mode not in ('original', 'notch_50_60', 'bipolar'):
        raise ValueError('Unknown reference/noise sensitivity mode')
    ratio = round(fs / 500)
    if not np.isclose(fs / 500, ratio, atol=1e-9, rtol=0):
        raise ValueError('Analysis resampling requires an integer ratio to 500 Hz')
    if isinstance(surrogates, bool) or not isinstance(surrogates, int) or surrogates < 19:
        raise ValueError('At least 19 surrogate shifts required')
    names = list(names)
    if mode == 'bipolar':
        if x.shape[1] != 4:
            raise ValueError('Bipolar sensitivity requires four original channels')
        x = np.column_stack((x[:, 0] - x[:, 1], x[:, 2] - x[:, 3]))
        names = [names[0] + '-' + names[1], names[2] + '-' + names[3]]
        if np.any(x.std(axis=0) <= 1e-8):
            raise ValueError('Flat bipolar difference cannot define connectivity')
    if ratio > 1:
        x = signal.resample_poly(x, 1, ratio, axis=0)
    rate = 500.0
    if mode == 'notch_50_60':
        for frequency in (50, 60):
            b, a = signal.iirnotch(frequency, 30, fs=rate)
            x = signal.filtfilt(b, a, x, axis=0)
    margin = round(rate)  # One second at each edge for all estimators.
    raw = x[margin:-margin]
    analytic = {}
    for band, edges in (('theta', THETA), ('gamma', GAMMA)):
        sos = signal.butter(4, edges, btype='bandpass', fs=rate, output='sos')
        analytic[band] = signal.hilbert(signal.sosfiltfilt(sos, x, axis=0), axis=0)[margin:-margin]
    phase_theta = np.angle(analytic['theta'])
    phase_gamma = np.angle(analytic['gamma'])
    amplitude = np.abs(analytic['gamma'])
    parameters = dict(fs=rate, window='hann', nperseg=1000, noverlap=500,
                      detrend='constant', scaling='density')
    frequencies, psd = signal.welch(raw, axis=0, **parameters)
    pairs = []
    for a, b in combinations(range(raw.shape[1]), 2):
        _, cross = signal.csd(raw[:, a], raw[:, b], **parameters)
        denominator = np.sqrt(psd[:, a] * psd[:, b])
        if not np.all(denominator > 0):
            raise ValueError('Zero spectral density prevents coherency')
        coherency = cross / denominator
        row = {'channel_a': names[a], 'channel_b': names[b]}
        for band, edges, phases in (('theta', THETA, phase_theta), ('gamma', GAMMA, phase_gamma)):
            row[band + '_plv'] = float(abs(np.mean(np.exp(1j * (phases[:, a] - phases[:, b])))))
            row[band + '_coherence'] = band_integral(frequencies, np.clip(abs(coherency) ** 2, 0, 1), *edges) / (edges[1] - edges[0])
            row[band + '_imaginary_coherency_abs'] = band_integral(frequencies, np.abs(coherency.imag), *edges) / (edges[1] - edges[0])
        pairs.append(row)
    rng = np.random.default_rng(seed)
    # np.roll(N - margin) is the -1s boundary. Exclude both signed
    # boundaries as well as every smaller circular distance from zero.
    available = np.arange(margin + 1, len(raw) - margin)
    if surrogates > len(available):
        raise ValueError('Insufficient distinct shifts for requested surrogate count')
    shifts = rng.choice(available, size=surrogates, replace=False)
    relationships = [(i, i) for i in range(raw.shape[1])] + [(0, raw.shape[1] - 1), (raw.shape[1] - 1, 0)]
    pac = []
    for phase_channel, amplitude_channel in relationships:
        phase = phase_theta[:, phase_channel]
        amp = amplitude[:, amplitude_channel]
        mi, distribution = phase_amplitude_index(phase, amp)
        null = np.array([phase_amplitude_index(phase, np.roll(amp, int(shift)))[0] for shift in shifts])
        sd = float(null.std(ddof=1))
        pac.append({'theta_channel': names[phase_channel], 'gamma_channel': names[amplitude_channel],
                    'relationship': 'within_channel' if phase_channel == amplitude_channel else 'cross_channel',
                    'tort_mi': mi, 'normalized_mvl': float(abs(np.sum(amp * np.exp(1j * phase))) / amp.sum()),
                    'surrogate_mean_mi': float(null.mean()), 'surrogate_sd_mi': sd,
                    'surrogate_p': float((1 + np.sum(null >= mi)) / (surrogates + 1)),
                    'surrogate_z': float((mi - null.mean()) / sd) if sd > 0 else None,
                    'surrogate_count': surrogates, 'phase_amplitude_distribution': distribution.tolist()})
    return {'pairs': pairs, 'pac': pac, 'analysis_sampling_rate_hz': rate,
            'samples_after_edge_trim': len(raw), 'edge_trim_s': 1,
            'surrogate_shifts_samples': shifts.tolist(),
            'surrogate_exclusion_s': 1, 'surrogate_boundary_excluded': True,
            'max_abs_uv': float(np.max(abs(x))), 'median_std_uv': float(np.median(raw.std(axis=0)))}
