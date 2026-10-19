"""EEG feature extraction: band powers, ratios, asymmetry and artifact checks."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from .config import Config


def _bandpass(x: np.ndarray, fs: float, lo: float = 1.0, hi: float = 45.0) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _notch(x: np.ndarray, fs: float, f0: float) -> np.ndarray:
    if f0 >= fs / 2:
        return x
    sos = butter(2, [f0 - 2, f0 + 2], btype="bandstop", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def quality_ok(window: np.ndarray, cfg: Config) -> bool:
    """Return True if the window (channels x samples, microvolts) is usable."""
    ptp = np.ptp(window, axis=-1)
    std = np.std(window, axis=-1)
    if np.any(ptp > cfg.max_peak_to_peak_uv):
        return False
    if np.any(std < cfg.min_std_uv):
        return False
    return bool(np.all(np.isfinite(window)))


def band_powers(window: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    """Absolute band power per channel (uV^2) using Welch's method."""
    x = _bandpass(window, cfg.fs)
    x = _notch(x, cfg.fs, 50.0)
    x = _notch(x, cfg.fs, 60.0)
    nper = int(min(x.shape[-1], cfg.fs))
    f, pxx = welch(x, fs=cfg.fs, nperseg=nper, axis=-1)
    df = f[1] - f[0]
    out = {}
    for name, (lo, hi) in cfg.bands.items():
        m = (f >= lo) & (f < hi)
        out[name] = pxx[..., m].sum(axis=-1) * df
    return out


def extract(window: np.ndarray, cfg: Config) -> Optional[Dict[str, float]]:
    """Compute arousal-related features for one window, or None if rejected."""
    return extract_all(window, cfg)[0]


def extract_all(window: np.ndarray, cfg: Config):
    """Return (combined features, per-channel features) for one window, or (None, None)."""
    if not quality_ok(window, cfg):
        return None, None
    bp = band_powers(window, cfg)
    eps = 1e-12
    total = sum(bp[b] for b in ("delta", "theta", "alpha", "beta")) + eps

    alpha = bp["alpha"] + eps
    feats = {
        "alpha_rel": float(np.mean(bp["alpha"] / total)),
        "theta_rel": float(np.mean(bp["theta"] / total)),
        "beta_alpha_ratio": float(np.mean(np.log(bp["beta"] + eps) - np.log(alpha))),
        "muscle_tension": float(np.mean(np.log(bp["high"] + eps))),
    }

    if cfg.frontal_pair and all(c in cfg.channels for c in cfg.frontal_pair):
        li = cfg.channels.index(cfg.frontal_pair[0])
        ri = cfg.channels.index(cfg.frontal_pair[1])
        # Classic frontal alpha asymmetry: ln(right alpha) - ln(left alpha).
        # Lower values = relatively more right-frontal activity (withdrawal / negative affect).
        feats["frontal_asymmetry"] = float(np.log(alpha[ri]) - np.log(alpha[li]))

    per_channel = {}
    for i, ch in enumerate(cfg.channels[: alpha.shape[0]]):
        per_channel[ch] = {
            "alpha_rel": float(bp["alpha"][i] / total[i]),
            "beta_alpha_ratio": float(np.log(bp["beta"][i] + eps) - np.log(alpha[i])),
            "muscle_tension": float(np.log(bp["high"][i] + eps)),
        }
    return feats, per_channel
