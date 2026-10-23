"""Two-level alert logic on the fused arousal index (same design as psi-qeeg-bis/meltdown-alert).

EARLY WARNING: one-sided CUSUM on the index, gated by a still-rising 3-minute trend.
IMMINENT:      smoothed index above `imminent_z` for `imminent_hold_sec`.
Hysteresis: the imminent level re-arms once the smoothed index falls below
`rearm_below_z`; the early level re-arms only after the index has been calm
(below `calm_level_z`) for `early_rearm_calm_sec`, so one slow build-up produces
one early warning. Plus a refractory period per level. No early warnings during an
active imminent episode. No alerts when fewer than `min_modalities` are valid.
"""
from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Alert:
    t: float
    level: str        # "early" or "imminent"
    index: float
    trend_z_per_min: float


def detect(index, n_mod, cfg=None):
    cfg = cfg or Config()
    T = len(index)
    a = 1.0 / cfg.smooth_sec
    smooth = np.full(T, np.nan)
    s = 0.0
    S = 0.0
    above = 0
    armed = {"early": True, "imminent": True}
    last = {"early": -1e9, "imminent": -1e9}
    episode = False
    calm_run = 0
    alerts = []
    tw = int(cfg.trend_window_sec)
    tt = np.arange(tw, dtype=float)

    for i in range(T):
        x = index[i]
        if not np.isfinite(x):
            smooth[i] = s
            continue
        s = x if i == 0 else s + a * (x - s)
        smooth[i] = s
        if i < cfg.baseline_sec:
            continue
        ok = n_mod[i] >= cfg.min_modalities

        if s < cfg.rearm_below_z:
            armed["imminent"] = True
            episode = False
        calm_run = calm_run + 1 if s < cfg.calm_level_z else 0
        if calm_run >= cfg.early_rearm_calm_sec:
            armed["early"] = True

        if episode:
            S = 0.0          # an ongoing episode and its recovery must not feed the next early warning
        elif ok:
            S = max(0.0, S + x - cfg.cusum_k)

        # trend over the last few minutes of the smoothed index, z per minute
        trend = 0.0
        if i >= tw:
            w = smooth[i - tw + 1:i + 1]
            if np.all(np.isfinite(w)):
                trend = float(np.polyfit(tt, w, 1)[0] * 60)

        # imminent
        above = above + 1 if s >= cfg.imminent_z else 0
        if (ok and above >= cfg.imminent_hold_sec and armed["imminent"]
                and i - last["imminent"] >= cfg.refractory_sec):
            alerts.append(Alert(float(i), "imminent", float(s), trend))
            armed["imminent"] = False
            last["imminent"] = i
            episode = True

        # early warning
        if (ok and not episode and S >= cfg.cusum_h and trend >= cfg.trend_min_z_per_min
                and armed["early"] and i - last["early"] >= cfg.refractory_sec):
            alerts.append(Alert(float(i), "early", float(s), trend))
            armed["early"] = False
            last["early"] = i
            S = 0.0
    return alerts, smooth
