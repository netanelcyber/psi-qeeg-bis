"""Standalone animal qEEG arithmetic example, not an onset predictor.

Input must be calibrated, complete samples in microvolts. Running this file
uses a clearly synthetic signal only; no real model-animal data are supplied.
The animal analysis/annotation/QC workflow is not implemented by this example.
"""

import json

import numpy as np
from scipy import signal

from psi_qeeg.signal import band_integral


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


def synthetic_demo():
    fs = 1000
    time = np.arange(8 * fs) / fs
    # Known sine amplitudes: A^2/2 gives 8 and 4.5 uV^2 of signal power.
    x = 4 * np.sin(2 * np.pi * 70 * time) + 3 * np.sin(2 * np.pi * 175 * time)
    shank3 = band_powers(x, fs, {'gamma': (30, 80)}, relative_range=(1, 80))
    mk801 = band_powers(x, fs, {'delta': (1, 4), 'theta': (6, 8), 'alpha': (8, 12),
                              'high_gamma': (60, 100), 'hfo': (150, 200)})
    return {'data_scope': 'synthetic_arithmetic_example_not_real_animal_data',
            'expected_power_uv2': {'70_hz_sine': 8.0, '175_hz_sine': 4.5},
            'shank3_study_band_example': shank3, 'mk801_study_band_example': mk801,
            'interpretation': 'Band power arithmetic only; no genotype classification, behavioral forecast, psychiatric label or clinical threshold'}


if __name__ == '__main__':
    print(json.dumps(synthetic_demo(), indent=2, allow_nan=False))
