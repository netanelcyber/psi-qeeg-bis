"""Conservative, configurable research QC; these cutoffs are not clinical rules."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityPolicy:
    min_sqi: float = 80
    max_abs_uv: float = 300
    flat_std_uv: float = 0.05
    emg_warning_db: float = 40

    def __post_init__(self):
        if not np.isfinite([self.min_sqi, self.max_abs_uv, self.flat_std_uv, self.emg_warning_db]).all():
            raise ValueError("QC thresholds must be finite")
        if not 0 <= self.min_sqi <= 100 or self.max_abs_uv <= 0 or self.flat_std_uv < 0:
            raise ValueError("Invalid SQI, amplitude or flat-signal threshold")


def assess_epoch(eeg: np.ndarray, frame: pd.DataFrame, policy: QualityPolicy, require_sqi: bool) -> dict:
    rejected, warnings = [], []
    if not np.isfinite(eeg).all():
        rejected.append("missing_eeg_samples")
    else:
        if np.max(abs(eeg)) > policy.max_abs_uv:
            rejected.append("amplitude_outlier")
        if np.any(np.std(eeg, axis=0) < policy.flat_std_uv):
            rejected.append("flat_signal")
    sqi_min = None
    if "SQI" in frame:
        q = frame.SQI.to_numpy(dtype=float)
        if not np.isfinite(q).all():
            rejected.append("missing_sqi")
        elif np.any((q < 0) | (q > 100)):
            rejected.append("invalid_sqi")
        else:
            sqi_min = float(q.min())
            if sqi_min < policy.min_sqi:
                rejected.append("low_sqi")
    elif require_sqi:
        rejected.append("missing_sqi")
    emg_max = None
    if "EMG" in frame:
        e = frame.EMG.to_numpy(dtype=float)
        if not np.isfinite(e).all():
            warnings.append("emg_quality_unknown")
        else:
            emg_max = float(e.max())
            if emg_max >= policy.emg_warning_db:
                warnings.append("possible_muscle_contamination")
    else:
        warnings.append("emg_quality_unknown")
    return {"accepted": not rejected, "rejection_reasons": rejected, "warnings": warnings,
            "sqi_min": sqi_min, "emg_max_db": emg_max}

