"""Session plot: arousal index, thresholds, trend, alerts and labelled events."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .config import Config
from .detector import EARLY_WARNING, IMMINENT, Step


def plot_session(steps: Sequence[Step], cfg: Config, path: str, events_sec: Sequence[float] = (),
                 truth: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([s.t for s in steps]) / 60.0
    idx = np.array([np.nan if s.index is None else s.index for s in steps])
    cus = np.array([np.nan if s.state == "CALIBRATING" else s.cusum for s in steps])

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3]})
    ax = axes[0]
    ax.axvspan(0, cfg.baseline_sec / 60, color="#9aa5b1", alpha=0.18, lw=0)
    ax.text(cfg.baseline_sec / 120, 0.03, "baseline\ncalibration", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color="#52606d")
    ax.plot(t, idx, color="#1f5aa6", lw=1.6, label="EEG arousal index (personal z)")
    ax.axhline(cfg.alert_level, color="#c0392b", ls="--", lw=1, label="imminent level (base)")
    for e in events_sec:
        ax.axvline(e / 60, color="#111", lw=1.2)
        ax.axvspan(e / 60 - cfg.horizon_min, e / 60, color="#d68910", alpha=0.06, lw=0)
        ax.text(e / 60, 0.99, " event", transform=ax.get_xaxis_transform(), va="top", fontsize=8)
    has_clinical = False
    for s in steps:
        if s.alert is None:
            continue
        if s.alert.level == "CLINICAL_RISK":
            has_clinical = True
            ax.plot(s.t / 60, 0.06, marker="D", color="#7d3c98", ms=8, mec="white", mew=0.8,
                    transform=ax.get_xaxis_transform())
            continue
        c = "#d68910" if s.alert.level == EARLY_WARNING else "#c0392b"
        m = "v" if s.alert.level == EARLY_WARNING else "X"
        ax.plot(s.t / 60, s.alert.index, marker=m, color=c, ms=10, mec="white", mew=0.8)
    ax.plot([], [], "v", color="#d68910", label=f"early warning (≤{cfg.horizon_min:.0f} min)")
    ax.plot([], [], "X", color="#c0392b", label="imminent alert")
    if has_clinical:
        ax.plot([], [], "D", color="#7d3c98", label="BVC clinical notice (thresholds lowered)")
    if truth is not None:
        ax2 = ax.twinx()
        ax2.plot(t, truth(t * 60.0), color="#7b8794", lw=0.8, alpha=0.6)
        ax2.set_ylabel("simulated true arousal", color="#7b8794", fontsize=8)
        ax2.tick_params(axis="y", colors="#7b8794", labelsize=7)
    ax.set_ylabel("arousal index (z)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("EEG early-warning prototype - synthetic session (pipeline demo, not clinical evidence)",
                 fontsize=10)

    ax = axes[1]
    ax.plot(t, cus, color="#d68910", lw=1.2)
    ax.axhline(cfg.cusum_h, color="#d68910", ls=":", lw=1)
    ax.text(t[-1], cfg.cusum_h, " threshold (base)", va="bottom", ha="right", fontsize=7, color="#d68910")
    ax.set_ylim(0, max(cfg.cusum_h * 2.5, 1))
    ax.set_ylabel("build-up\n(CUSUM)", fontsize=8)
    ax.set_xlabel("session time (min)")
    for e in events_sec:
        ax.axvline(e / 60, color="#111", lw=1.2)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
