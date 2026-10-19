"""Personalised arousal index and two-level early-warning logic.

Levels
------
* EARLY_WARNING  - a slow, persistent rise in arousal has accumulated (one-sided
                   CUSUM above ``cusum_h``) while the trend is still rising. Aimed at
                   a ~``horizon_min`` (default 10) minute warning; when the trend allows,
                   it includes a projected time until the imminent level.
* IMMINENT       - arousal is above ``alert_level`` for ``alert_min_duration_sec``.

The index is a weighted sum of robust z-scores of EEG features relative to the
person's own calm baseline, so it adapts to each individual rather than using
population thresholds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import Config

CALIBRATING = "CALIBRATING"
CALM = "CALM"
EARLY_WARNING = "EARLY_WARNING"
IMMINENT = "IMMINENT"
LOW_QUALITY = "LOW_QUALITY"
CLINICAL_RISK = "CLINICAL_RISK"


@dataclass
class Alert:
    t: float                      # session time (s)
    level: str                    # EARLY_WARNING or IMMINENT
    index: float                  # smoothed arousal index (z)
    slope_per_min: float          # trend (z / minute)
    eta_min: Optional[float]      # projected minutes until IMMINENT level (early warnings)
    message: str
    onset: Optional[dict] = None  # sensor-level onset map (see onset.py)


@dataclass
class Step:
    t: float
    state: str
    raw_index: Optional[float]
    index: Optional[float]
    slope_per_min: Optional[float]
    eta_min: Optional[float]
    quality: float
    alert: Optional[Alert] = None
    cusum: float = 0.0
    wearable: Optional[dict] = None  # optional QC/provenance; absent on the legacy EEG path


class Baseline:
    """Robust (median / MAD) per-feature baseline with optional slow adaptation."""

    def __init__(self) -> None:
        self.samples: List[Dict[str, float]] = []
        self.median: Dict[str, float] = {}
        self.scale: Dict[str, float] = {}
        self.ready = False

    def add(self, feats: Dict[str, float]) -> None:
        self.samples.append(feats)

    def finalize(self) -> None:
        keys = self.samples[0].keys()
        for k in keys:
            v = np.array([s[k] for s in self.samples if k in s])
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med))) * 1.4826
            self.median[k] = med
            self.scale[k] = max(mad, 1e-6, 0.05 * abs(med))
        self.ready = True

    def z(self, feats: Dict[str, float]) -> Dict[str, float]:
        return {k: (v - self.median[k]) / self.scale[k] for k, v in feats.items() if k in self.median}

    def adapt(self, feats: Dict[str, float], rate: float) -> None:
        for k, v in feats.items():
            if k in self.median:
                self.median[k] += rate * (v - self.median[k])


class Detector:
    def __init__(self, cfg: Config, context=None) -> None:
        self.cfg = cfg
        self.context = context                 # optional clinical.ClinicalContext (BVC)
        self._notified_obs = set()
        self.baseline = Baseline()
        self.level: Optional[float] = None
        self.hist: Deque[Tuple[float, float]] = deque()      # (t, smoothed level) for slope
        self.quality_hist: Deque[bool] = deque(maxlen=60)
        self.early_since: Optional[float] = None
        self.alert_since: Optional[float] = None
        self.last_alert_t: Dict[str, float] = {}
        self.armed: Dict[str, bool] = {EARLY_WARNING: True, IMMINENT: True}
        self.cusum = 0.0
        from .onset import OnsetTracker
        self.onset = OnsetTracker(cfg.channels)
        self.kf = None
        if cfg.model == "kalman":
            from .kalman import KalmanArousal
            self.kf = KalmanArousal(cfg)

    # ------------------------------------------------------------------
    def _index(self, feats: Dict[str, float]) -> float:
        z = self.baseline.z(feats)
        w = self.cfg.weights
        num = sum(w[k] * z[k] for k in w if k in z)
        den = sum(abs(w[k]) for k in w if k in z) or 1.0
        # Normalise so a 1-SD shift in every feature in the "aroused" direction gives ~ +1 * sqrt(n).
        return float(num / np.sqrt(den))

    def _slope(self, t: float) -> Optional[float]:
        win = self.cfg.trend_window_sec
        while self.hist and self.hist[0][0] < t - win:
            self.hist.popleft()
        if len(self.hist) < 10 or (self.hist[-1][0] - self.hist[0][0]) < min(60.0, win / 2):
            return None
        ts = np.array([h[0] for h in self.hist])
        ys = np.array([h[1] for h in self.hist])
        slope_per_sec = np.polyfit(ts - ts[0], ys, 1)[0]
        return float(slope_per_sec * 60.0)

    def _can_fire(self, level: str, t: float) -> bool:
        last = self.last_alert_t.get(level)
        return self.armed[level] and (last is None or t - last >= self.cfg.refractory_sec)

    def reset_evidence(self) -> None:
        """Break persistence after a time gap or a change in fused feature availability.

        Preserve personal calibration, refractory times, and alert arming. A data
        discontinuity must not supply evidence of a sustained physiological rise.
        """
        self.level = None
        self.hist.clear()
        self.quality_hist.clear()
        self.early_since = self.alert_since = None
        self.cusum = 0.0
        self.onset.reset()
        if self.kf is not None:
            self.kf.reset_state()

    # ------------------------------------------------------------------
    def update(self, t: float, feats: Optional[Dict[str, float]],
               per_channel: Optional[Dict[str, Dict[str, float]]] = None, *,
               auxiliary_index: Optional[float] = None, auxiliary_weight: float = 0.0) -> Step:
        cfg = self.cfg
        self.quality_hist.append(feats is not None)
        quality = float(np.mean(self.quality_hist))

        # --- calibration ----------------------------------------------
        if not self.baseline.ready:
            if feats is not None:
                self.baseline.add(feats)
                if self.kf is not None:
                    self.kf.add_baseline(feats)
                if per_channel is not None:
                    self.onset.add_baseline(per_channel)
            if t >= cfg.baseline_sec and len(self.baseline.samples) >= 20:
                self.baseline.finalize()
                self.onset.finalize()
                if self.kf is not None:
                    self.kf.finalize()
            return Step(t, CALIBRATING, None, None, None, None, quality)

        if feats is None:
            return Step(t, LOW_QUALITY, None, self.level, None, None, quality, None, self.cusum)

        # --- clinical context (BVC) modulates sensitivity ---------------
        h_mult, lvl_off, band, bvc_score = (1.0, 0.0, "none", 0)
        if self.context is not None:
            h_mult, lvl_off, band, bvc_score = self.context.modifiers(t)
        cusum_h = cfg.cusum_h * h_mult
        alert_level = cfg.alert_level + lvl_off

        raw = self.kf.update(feats) if self.kf is not None else self._index(feats)
        if auxiliary_index is not None and auxiliary_weight:
            if not np.isfinite(auxiliary_index) or not 0 <= auxiliary_weight <= 1:
                raise ValueError("Auxiliary index must be finite and its weight in [0, 1]")
            raw = (1.0 - auxiliary_weight) * raw + auxiliary_weight * auxiliary_index
        a = 1.0 - np.exp(-cfg.step_sec / cfg.smooth_tau_sec)
        self.level = raw if self.level is None else self.level + a * (raw - self.level)
        self.hist.append((t, self.level))
        slope = self._slope(t)
        self.cusum = max(0.0, self.cusum + raw - cfg.cusum_k)
        if self.level < 0.3:                      # clearly calm: partially forget old evidence
            self.cusum = min(self.cusum, 0.5 * cusum_h)
            self.onset.reset()
        else:
            self.onset.update(t, per_channel)

        # --- hysteresis re-arm ---------------------------------------
        if self.level < cfg.release_level:
            # Re-arm alerts only; the persistence timers reset on their own conditions below.
            self.armed = {EARLY_WARNING: True, IMMINENT: True}
            if cfg.adapt_baseline:
                self.baseline.adapt(feats, cfg.adapt_rate)
                if self.kf is not None:
                    self.kf.adapt(feats, cfg.adapt_rate)

        # --- projected time to the imminent level --------------------
        eta = None
        if slope is not None and slope > 0 and self.level < alert_level:
            eta = (alert_level - self.level) / slope

        state = CALM
        alert: Optional[Alert] = None
        enough_quality = quality >= cfg.min_quality

        # --- IMMINENT ---------------------------------------------------
        if self.level >= alert_level:
            self.alert_since = self.alert_since if self.alert_since is not None else t
            state = IMMINENT
            if (t - self.alert_since >= cfg.alert_min_duration_sec and enough_quality
                    and self._can_fire(IMMINENT, t)):
                alert = Alert(t, IMMINENT, self.level, slope or 0.0, 0.0, cfg.imminent_msg)
                self.last_alert_t[IMMINENT] = t
                self.armed[IMMINENT] = False
        else:
            self.alert_since = None

        # --- EARLY WARNING (aimed at ~horizon_min ahead) ------------
        early_cond = (
            self.cusum > cusum_h
            and slope is not None and slope >= cfg.early_min_slope
            and self.level < alert_level
            and self.armed[IMMINENT]          # no early warnings during/after an imminent episode
        )
        if early_cond and state != IMMINENT:
            self.early_since = self.early_since if self.early_since is not None else t
            state = EARLY_WARNING
            if (alert is None and t - self.early_since >= cfg.early_min_duration_sec
                    and enough_quality and self._can_fire(EARLY_WARNING, t)):
                when = (("projected to reach the imminent level within a minute" if eta < 1 else
                         f"projected to reach the imminent level in ~{eta:.0f} min")
                        if eta is not None and eta <= cfg.horizon_min else
                        f"escalation may follow within ~{cfg.horizon_min:.0f} min")
                if band in ("moderate", "very_high"):
                    when += f" (sensitivity raised: BVC {bvc_score} = {band.replace('_', ' ')} risk)"
                alert = Alert(t, EARLY_WARNING, self.level, slope, eta, cfg.early_msg.format(when=when))
                self.last_alert_t[EARLY_WARNING] = t
                self.armed[EARLY_WARNING] = False
        elif state != IMMINENT:
            self.early_since = None

        if alert is not None and alert.level in (EARLY_WARNING, IMMINENT):
            from .onset import describe
            alert.onset = self.onset.summary(t)
            alert.message = f"{alert.message}\n  {describe(alert.onset)}"

        if alert is None and self.context is not None and band == "very_high":
            o = self.context.current(t)
            if o is not None and id(o) not in self._notified_obs:
                self._notified_obs.add(id(o))
                alert = Alert(t, CLINICAL_RISK, self.level, slope or 0.0, None,
                              f"Clinical notice: BVC score {bvc_score} (>2) = very high risk of violence "
                              f"within 24 h. Agree a preventive, least-restrictive plan; monitoring "
                              f"sensitivity has been raised.")

        return Step(t, state, raw, self.level, slope, eta, quality, alert, self.cusum)
