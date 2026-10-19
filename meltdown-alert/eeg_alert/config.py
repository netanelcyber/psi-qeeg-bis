"""Configuration for the EEG early-warning pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Config:
    model: str = "index"                   # "index" (weighted z-sum) or "kalman" (state-space, FFT-derived noise)
    profile: str = "meltdown"               # see profiles.py: "meltdown" or "psychosis"
    early_msg: str = ("Early warning: sustained rise in arousal; {when}. Consider a calm break, "
                      "lowering sensory load, or the person's own regulation strategy now.")
    imminent_msg: str = ("Imminent: arousal far above personal baseline. Reduce demands and sensory "
                         "input, offer a quiet space, keep language minimal and calm.")

    # --- signal ---------------------------------------------------------
    fs: float = 256.0                      # sampling rate (Hz); Muse = 256, OpenBCI Cyton = 250
    channels: List[str] = field(default_factory=lambda: ["TP9", "AF7", "AF8", "TP10"])
    frontal_pair: Optional[Tuple[str, str]] = ("AF7", "AF8")  # (left, right) for alpha asymmetry

    # --- windowing ------------------------------------------------------
    window_sec: float = 2.0                # analysis window length
    step_sec: float = 1.0                  # hop between windows

    # --- frequency bands (Hz) --------------------------------------------
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "high": (30.0, 45.0),              # mostly scalp EMG on dry wearables -> "muscle tension"
    })

    # --- artifact rejection ---------------------------------------------
    max_peak_to_peak_uv: float = 250.0     # windows above this are rejected
    min_std_uv: float = 0.5                # flat-line detection

    # --- personal baseline ------------------------------------------------
    baseline_sec: float = 180.0            # calm calibration period at the start of a session
    adapt_baseline: bool = True            # slowly update baseline while state is calm
    adapt_rate: float = 0.002              # per-window EWMA rate for baseline adaptation

    # --- arousal index weights (applied to robust z-scores) -------------
    weights: Dict[str, float] = field(default_factory=lambda: {
        "beta_alpha_ratio": 1.0,           # cortical activation
        "alpha_rel": -1.0,                 # alpha suppression
        "theta_rel": -0.3,
        "frontal_asymmetry": -0.5,         # relatively lower right-frontal alpha (withdrawal)
        "muscle_tension": 0.7,             # high-frequency (EMG) power - tension / agitation
    })

    # --- detection --------------------------------------------------------
    smooth_tau_sec: float = 20.0           # EWMA time constant for the level
    trend_window_sec: float = 180.0        # window for slope estimation (3 min)
    horizon_min: float = 10.0              # early-warning horizon (minutes)

    # Early warning uses a one-sided CUSUM on the raw index: it accumulates small, persistent
    # rises above `cusum_k` and fires when the sum exceeds `cusum_h` while the trend is rising.
    # This catches slow build-ups minutes earlier than a level threshold would.
    cusum_k: float = 0.3                   # reference drift (z) - rises smaller than this are ignored
    cusum_h: float = 80.0                  # decision threshold (z * windows)
    early_min_duration_sec: float = 30.0   # condition must hold this long
    early_min_slope: float = 0.05          # trend must be rising at least this fast (z / minute)

    alert_level: float = 2.5               # z-level for an "imminent" alert
    alert_min_duration_sec: float = 10.0
    release_level: float = 1.0             # hysteresis: state resets below this level
    refractory_sec: float = 300.0          # do not repeat the same alert level within this time

    min_quality: float = 0.6               # fraction of clean windows required to issue alerts

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
