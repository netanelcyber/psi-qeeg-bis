"""Streaming deviation monitor: flags EEG drifting away from this recording's own baseline, epoch by epoch.

This is an anomaly detector, not a validated predictor. It cannot know that a behavioural event
will follow; whether alerts precede real events (lead time, false alerts per hour) must be measured
against independently annotated events with :func:`evaluate_alerts` before any use. Research only.
"""

from collections import deque
from dataclasses import dataclass
import json
import sys

import numpy as np
import pandas as pd

from .quality import QualityPolicy, assess_epoch
from .signal import spectral_features

FEATURES = ("delta_relative_power", "theta_relative_power", "alpha_relative_power", "beta_relative_power",
            "gamma_relative_power", "sef95_hz", "spectral_entropy", "log_total_power")


@dataclass(frozen=True)
class MonitorConfig:
    epoch_s: float = 4.0
    hop_s: float = 2.0
    baseline_epochs: int = 30       # accepted epochs needed before any alert; baseline is then frozen
    ewma_lambda: float = 0.3
    warn_level: float = 2.0         # smoothed deviation (robust-z RMS) for the early "rising" state
    alert_level: float = 3.0
    clear_level: float = 2.0
    persist: int = 3                # consecutive epochs above alert_level before alerting
    max_rejected_run: int = 5       # consecutive rejected epochs -> signal_lost
    z_clip: float = 8.0
    line_hz: float | None = None

    def __post_init__(self):
        if self.epoch_s < 2 or not 0 < self.hop_s <= self.epoch_s:
            raise ValueError("epoch_s must be >= 2 s and 0 < hop_s <= epoch_s")
        if not 0 < self.ewma_lambda <= 1 or self.baseline_epochs < 10 or self.persist < 1:
            raise ValueError("Invalid smoothing, baseline length or persistence")
        if not self.clear_level <= self.alert_level or not self.warn_level <= self.alert_level:
            raise ValueError("Need warn_level <= alert_level and clear_level <= alert_level")


class RealtimeMonitor:
    """Feed sample chunks with :meth:`push`; it returns one result dict per completed epoch."""

    def __init__(self, fs: float, channels: list[str], config: MonitorConfig | None = None,
                 policy: QualityPolicy | None = None):
        self.cfg, self.policy = config or MonitorConfig(), policy or QualityPolicy()
        if fs / 2 <= 40:
            raise ValueError("Needs a sampling rate above 80 Hz for the 0.5-40 Hz features")
        self.fs, self.channels = float(fs), list(channels)
        self.n_epoch, self.n_hop = round(self.cfg.epoch_s * fs), round(self.cfg.hop_s * fs)
        self._buf = np.empty((0, len(channels)))
        self._aux = {}
        self._t0 = 0                       # index of first buffered sample
        self._baseline: list[np.ndarray] = []
        self._median = self._scale = None
        self._ewma = 0.0
        self._above = 0
        self._rejected_run = 0
        self._state = "warmup"
        self._recent = deque(maxlen=4)

    @property
    def state(self) -> str:
        return self._state

    def _vector(self, chunk: np.ndarray) -> np.ndarray:
        values = []
        for j in range(len(self.channels)):
            f = spectral_features(chunk[:, j], self.fs, self.cfg.line_hz)
            f["log_total_power"] = float(np.log10(max(f["total_power_uv2"], 1e-12)))
            values += [f[k] for k in FEATURES]
        return np.asarray(values, dtype=float)

    def _names(self):
        return [f"{c}__{k}" for c in self.channels for k in FEATURES]

    def push(self, samples, sqi=None, emg=None) -> list[dict]:
        """``samples``: (n, channels) in uV. Optional per-sample ``sqi``/``emg`` arrays from a monitor."""
        x = np.atleast_2d(np.asarray(samples, dtype=float))
        if x.shape[1] != len(self.channels):
            raise ValueError("samples must have one column per channel")
        self._buf = np.vstack([self._buf, x])
        for name, v in (("SQI", sqi), ("EMG", emg)):
            if v is not None:
                self._aux[name] = np.r_[self._aux.get(name, np.empty(0)), np.asarray(v, dtype=float)]
        out = []
        while len(self._buf) >= self.n_epoch:
            out.append(self._epoch(self._buf[:self.n_epoch], {k: v[:self.n_epoch] for k, v in self._aux.items()}))
            self._buf = self._buf[self.n_hop:]
            self._aux = {k: v[self.n_hop:] for k, v in self._aux.items()}
            self._t0 += self.n_hop
        return out

    def _epoch(self, chunk, aux) -> dict:
        t_end = (self._t0 + self.n_epoch) / self.fs
        qc = assess_epoch(chunk, pd.DataFrame(aux), self.policy, require_sqi=False)
        res = {"t_end_s": round(t_end, 3), "accepted": qc["accepted"], "reasons": qc["rejection_reasons"],
               "warnings": qc["warnings"], "score": None, "ewma": None, "state": self._state, "top_features": []}
        if not qc["accepted"]:
            self._rejected_run += 1
            self._above = 0                      # artifacts neither build nor clear evidence
            if self._rejected_run >= self.cfg.max_rejected_run and self._state != "signal_lost":
                self._state = "signal_lost"
            res["state"] = self._state
            return res
        self._rejected_run = 0
        if self._state == "signal_lost":
            self._state = "normal" if self._median is not None else "warmup"
        vec = self._vector(chunk)
        if self._median is None:
            # Warm-up uses hop-spaced epochs; overlap makes them correlated, so require the configured count.
            self._baseline.append(vec)
            if len(self._baseline) >= self.cfg.baseline_epochs:
                b = np.vstack(self._baseline)
                self._median = np.median(b, axis=0)
                mad = 1.4826 * np.median(abs(b - self._median), axis=0)
                self._scale = np.maximum(mad, 0.05 * np.maximum(abs(self._median), 1e-3))  # floor: avoid /0
                self._state = "normal"
            res["state"] = self._state
            return res
        z = np.clip((vec - self._median) / self._scale, -self.cfg.z_clip, self.cfg.z_clip)
        score = float(np.sqrt(np.mean(z ** 2)))
        self._ewma = self.cfg.ewma_lambda * score + (1 - self.cfg.ewma_lambda) * self._ewma
        self._recent.append(self._ewma)
        cfg = self.cfg
        self._above = self._above + 1 if self._ewma >= cfg.alert_level else 0
        if self._state == "alert":
            if self._ewma < cfg.clear_level:
                self._state = "normal"
        elif self._above >= cfg.persist:
            self._state = "alert"
        else:
            rising = len(self._recent) == self._recent.maxlen and all(
                b > a for a, b in zip(list(self._recent)[:-1], list(self._recent)[1:]))
            if self._ewma < cfg.warn_level:
                self._state = "normal"
            elif rising or self._state == "rising":   # sticky while still above the warning level
                self._state = "rising"
        order = np.argsort(-abs(z))[:3]
        names = self._names()
        res.update(score=round(score, 3), ewma=round(self._ewma, 3), state=self._state,
                   top_features=[(names[i], round(float(z[i]), 2)) for i in order])
        return res


def evaluate_alerts(alert_times, event_times, duration_s, horizon_s=60.0):
    """Lead time and false-alert rate against independently annotated events.

    An alert is a true warning for an event if it starts within ``horizon_s`` before it. Alerts that
    precede no event, or that arrive after one, count as false. Never tune thresholds on the events you
    evaluate with; use held-out subjects or sessions.
    """
    alerts, events = sorted(alert_times), sorted(event_times)
    leads, used = [], set()
    for e in events:
        candidates = [a for a in alerts if e - horizon_s <= a <= e]
        if candidates:
            leads.append(e - candidates[0])
            used.update(candidates)
    false = [a for a in alerts if a not in used]
    return {"events": len(events), "detected": len(leads), "sensitivity": len(leads) / len(events) if events else None,
            "median_lead_s": float(np.median(leads)) if leads else None, "false_alerts": len(false),
            "false_alerts_per_hour": len(false) / (duration_s / 3600) if duration_s > 0 else None,
            "horizon_s": horizon_s}


def run_stream(monitor: RealtimeMonitor, rows, out=sys.stdout, verbose=False) -> list[dict]:
    """Consume an iterable of (n, channels[+aux]) arrays; print state changes as JSON lines."""
    last, transitions = None, []
    for chunk in rows:
        for r in monitor.push(chunk):
            if r["state"] != last or verbose:
                print(json.dumps(r), file=out, flush=True)
            if r["state"] != last:
                transitions.append(r)
                last = r["state"]
    return transitions
