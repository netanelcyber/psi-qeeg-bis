"""Standalone animal qEEG arithmetic example, not an onset predictor.

Input must be calibrated, complete samples in microvolts. Running this file
uses a clearly synthetic signal only; no real model-animal data are supplied.
The animal analysis/annotation/QC workflow is not implemented by this example.
"""

import json

import numpy as np
from psi_qeeg.animal_qeeg import band_powers


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
