"""Session figure: per-modality z, fused index, alerts, event onsets."""
import numpy as np

from .config import Config
from .detector import detect
from .fusion import fuse
from .pipeline import prepare_streams


def plot_session(session, cfg=None, path="demo.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or Config()
    streams = prepare_streams(session, cfg)
    index, n_mod, z = fuse(streams, cfg)
    alerts, smooth = detect(index, n_mod, cfg)
    tm = session.t / 60

    groups = {
        "EEG β/α": ["eeg_beta_alpha"],
        "fNIRS PFC HbO (cleaned)": ["fnirs_pfc_hbo"],
        "Temporalis EMG": ["emg_temporalis_rms"],
        "Autonomic (HR, HRV, EDA)": ["hr", "hrv_rmssd", "eda"],
        "Head movement": ["imu_activity"],
    }
    fig, axes = plt.subplots(len(groups) + 1, 1, figsize=(11, 10), sharex=True)
    k = 30
    ker = np.ones(k) / k
    for ax, (title, names) in zip(axes, groups.items()):
        stack = np.array([z[n] for n in names])
        valid = np.isfinite(stack).any(axis=0)
        v = np.where(valid, np.nanmean(np.where(np.isfinite(stack), stack, 0), axis=0)
                     * stack.shape[0] / np.maximum(np.isfinite(stack).sum(axis=0), 1), 0.0)
        sm = np.convolve(v, ker, mode="same") / np.maximum(np.convolve(valid, ker, mode="same"), 1e-9)
        sm[~valid] = np.nan  # gaps = gated (exercise) or lost contact
        ax.plot(tm, sm, lw=1)
        ax.set_ylabel("z", fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
    ax = axes[-1]
    ax.plot(tm, smooth, color="black", lw=1.2, label="fused arousal index (smoothed)")
    ax.axhline(cfg.imminent_z, color="red", ls=":", lw=0.8)
    for a in alerts:
        ax.axvline(a.t / 60, color="orange" if a.level == "early" else "red", lw=1.2)
    ax.set_ylabel("z")
    ax.set_xlabel("minutes")
    ax.legend(fontsize=8, loc="upper left")
    for ax in axes:
        for e in session.events:
            ax.axvspan(e / 60, e / 60 + 1, color="purple", alpha=0.2)
        for s, e in session.exercise:
            ax.axvspan(s / 60, e / 60, color="green", alpha=0.08)
    fig.suptitle("Synthetic session: purple = event, green = exercise, orange = early warning, red = imminent",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
