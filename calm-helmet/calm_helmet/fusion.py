"""Personal-baseline fusion of all helmet modalities into one arousal index.

Steps per second:
  1. robust z-score of every feature against the person's own calm baseline
     (median / MAD from the calibration period, slowly adapted while calm);
  2. sign so that "up" always means more arousal;
  3. motion gating: autonomic features (HR, HRV, EDA) rise with plain exercise, so
     while the head IMU shows clear physical activity (and for a recovery period
     after it) they are left out and the cortical/muscle channels carry the index;
  4. weighted mean over the features that are valid at that second. Weights are
     renormalised, so losing a modality (e.g. fNIRS optode contact) degrades the
     index gracefully instead of breaking it.
"""
import numpy as np

from .config import Config


def fuse(streams, cfg=None, use_modalities=None):
    """streams: dict feature name -> 1D array (1 Hz). Returns (index, n_modalities, z)."""
    cfg = cfg or Config()
    feats = [f for f in cfg.features if f.name in streams
             and (use_modalities is None or f.modality in use_modalities)]
    if not feats:
        raise ValueError("no usable features")
    T = len(next(iter(streams.values())))
    b = int(cfg.baseline_sec)

    med, scale = {}, {}
    for f in feats:
        x = np.asarray(streams[f.name], float)[:b]
        x = x[np.isfinite(x)]
        if len(x) < 30:
            raise ValueError(f"not enough clean baseline samples for {f.name}")
        med[f.name] = float(np.median(x))
        mad = float(np.median(np.abs(x - med[f.name])))
        scale[f.name] = 1.4826 * mad if mad > 0 else float(np.std(x)) or 1.0

    has_imu = any(f.modality == "imu" for f in feats)
    index = np.full(T, np.nan)
    n_mod = np.zeros(T, int)
    z = {f.name: np.full(T, np.nan) for f in feats}

    gate_until = -1
    for i in range(T):
        zi = {}
        for f in feats:
            v = streams[f.name][i]
            if np.isfinite(v):
                zi[f.name] = np.clip(f.sign * (v - med[f.name]) / scale[f.name], -cfg.z_clip, cfg.z_clip)
        if has_imu and zi.get("imu_activity", 0.0) >= cfg.motion_gate_z:
            gate_until = i + cfg.motion_hold_sec
        if i <= gate_until:
            for f in feats:
                if f.motion_corrected:
                    zi.pop(f.name, None)
        if not zi:
            continue
        used = [f for f in feats if f.name in zi and f.weight > 0]
        if not used:
            continue
        w = np.array([f.weight for f in used])
        v = np.array([zi[f.name] for f in used])
        index[i] = float(w @ v / w.sum())
        n_mod[i] = len({f.modality for f in used})
        for k, val in zi.items():
            z[k][i] = val

        # adapt the baseline slowly, only while calm and after calibration
        if i >= b and index[i] < cfg.calm_level_z:
            for f in feats:
                if f.name in zi:
                    med[f.name] += cfg.baseline_adapt_alpha * (streams[f.name][i] - med[f.name])
    return index, n_mod, z
