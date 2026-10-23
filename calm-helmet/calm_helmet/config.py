"""All tunable parameters in one place.

The pipeline works on *feature streams* sampled at 1 Hz (one value per second per
feature). Raw-signal processing (EEG band powers, fNIRS optical density -> HbO/HbR,
PPG -> HR/HRV, EMG RMS, IMU activity counts) happens upstream in the device firmware
or in per-modality adapters.
"""
from dataclasses import dataclass, field

FS = 1.0  # feature rate, Hz


@dataclass(frozen=True)
class Feature:
    name: str
    modality: str
    sign: int  # +1 if the feature rises with arousal, -1 if it falls
    weight: float
    motion_corrected: bool = False  # autonomic features are discounted by movement


FEATURES = (
    # cortical, electrical
    Feature("eeg_beta_alpha", "eeg", +1, 0.15),
    Feature("eeg_alpha_rel", "eeg", -1, 0.05),
    # cortical, haemodynamic (prefrontal fNIRS / HD-DOT, after short-separation cleaning)
    Feature("fnirs_pfc_hbo", "fnirs", +1, 0.15),
    Feature("fnirs_pfc_hbr", "fnirs", -1, 0.05),
    # muscle (temporalis EMG)
    Feature("emg_temporalis_rms", "emg", +1, 0.15),
    # autonomic (forehead PPG + wrist EDA)
    Feature("hr", "autonomic", +1, 0.10, motion_corrected=True),
    Feature("hrv_rmssd", "autonomic", -1, 0.10, motion_corrected=True),
    Feature("eda", "autonomic", +1, 0.15, motion_corrected=True),
    # behaviour (head IMU) - context only: gates the autonomic features during exercise
    Feature("imu_activity", "imu", +1, 0.0),
)


@dataclass
class Config:
    baseline_sec: int = 180          # calm calibration period at the start of a session
    baseline_adapt_alpha: float = 1 / 1800  # slow baseline update, only while calm
    calm_level_z: float = 0.5        # index must be below this to adapt the baseline
    z_clip: float = 6.0
    motion_gate_z: float = 3.0       # head movement above this z = "physically active"
    motion_hold_sec: int = 120       # keep autonomic features gated this long after movement stops
    min_modalities: int = 2          # no alerts with fewer valid modalities than this

    # fNIRS short-separation regression
    ss_fit_sec: int = 180            # fit scalp coupling on the calibration period

    # early warning (one-sided CUSUM + rising trend)
    cusum_k: float = 0.35
    cusum_h: float = 120.0
    trend_window_sec: int = 180
    trend_min_z_per_min: float = 0.05

    # imminent alert
    smooth_sec: int = 30
    imminent_z: float = 2.5
    imminent_hold_sec: int = 10

    # hysteresis and refractory
    rearm_below_z: float = 1.0       # imminent re-arms once the smoothed index drops below this
    early_rearm_calm_sec: int = 300  # early warning re-arms after this long below calm_level_z
    refractory_sec: int = 300

    horizon_min: float = 10.0
    features: tuple = field(default=FEATURES)
