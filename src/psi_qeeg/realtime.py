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
from .signal import RODENT_BANDS, spectral_features

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
    species: str = "human"           # "rodent" switches to rodent band edges

    def __post_init__(self):
        if self.species not in ("human", "rodent"):
            raise ValueError("species must be human or rodent")
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
            f = spectral_features(chunk[:, j], self.fs, self.cfg.line_hz, RODENT_BANDS if self.cfg.species == "rodent" else None)
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


@dataclass(frozen=True)
class LongRunConfig:
    """Multi-day monitoring: per-vigilance-state rolling baselines, hourly summaries, checkpoints."""
    slow_delta_fraction: float = 0.5      # relative delta >= this -> "slow" (sleep-like) state, else "fast"
    emg_active: float | None = None       # if EMG is supplied and above this, state is "active"
    min_state_epochs: int = 60            # accepted epochs a state needs before it can raise alerts
    window_epochs: int = 2160             # rolling baseline memory per state (6 h at 10 s epochs)
    refit_every: int = 60                 # recompute median/MAD this often
    min_hours: float = 24.0


class LongRunMonitor(RealtimeMonitor):
    """RealtimeMonitor for >= 24 h runs, e.g. a rodent model against its own circadian structure.

    A single frozen baseline fails over a day: sleep and wake differ more than most abnormalities.
    Here every vigilance state (slow / fast / active) keeps its own rolling robust baseline. Only
    non-alert epochs update it, so a sustained abnormality cannot quietly become the new normal
    until it has ended. State is crudely inferred from relative delta (+ EMG when supplied): use it
    to stratify, not as a validated sleep scorer. Altered state proportions are reported hourly.
    """

    def __init__(self, fs, channels, config=None, long=None, policy=None):
        super().__init__(fs, channels, config or MonitorConfig(epoch_s=10, hop_s=10, species="rodent"), policy)
        self.long = long or LongRunConfig()
        self._pool = {}                # state -> deque of feature vectors (accepted, non-alert)
        self._fit = {}                 # state -> (median, scale, epochs_since_refit)
        self._hours = {}               # hour index -> counters
        self._epoch_index = 0

    # -- state inference ------------------------------------------------------------------
    def _vigilance(self, vec, emg):
        if emg is not None and self.long.emg_active is not None and emg > self.long.emg_active:
            return "active"
        delta = float(np.mean(vec[0::len(FEATURES)]))      # relative delta, mean over channels
        return "slow" if delta >= self.long.slow_delta_fraction else "fast"

    def _refit(self, state):
        b = np.vstack(self._pool[state])
        med = np.median(b, axis=0)
        mad = 1.4826 * np.median(abs(b - med), axis=0)
        self._fit[state] = (med, np.maximum(mad, 0.05 * np.maximum(abs(med), 1e-3)), 0)

    def _hour(self, t_end):
        return self._hours.setdefault(int(t_end // 3600), {"epochs": 0, "accepted": 0, "alerts": 0, "score_sum": 0.0,
                                                           "scored": 0, "slow": 0, "fast": 0, "active": 0})

    def _epoch(self, chunk, aux) -> dict:
        t_end = (self._t0 + self.n_epoch) / self.fs
        h = self._hour(t_end - self.cfg.epoch_s)  # bucket by epoch start
        h["epochs"] += 1
        qc = assess_epoch(chunk, pd.DataFrame(aux), self.policy, require_sqi=False)
        res = {"t_end_s": round(t_end, 3), "accepted": qc["accepted"], "reasons": qc["rejection_reasons"],
               "warnings": qc["warnings"], "score": None, "ewma": None, "state": self._state,
               "vigilance": None, "top_features": []}
        if not qc["accepted"]:
            self._rejected_run += 1
            self._above = 0
            if self._rejected_run >= self.cfg.max_rejected_run:
                self._state = "signal_lost"
            res["state"] = self._state
            return res
        h["accepted"] += 1
        self._rejected_run = 0
        if self._state == "signal_lost":
            self._state = "warmup" if not self._fit else "normal"
        vec = self._vector(chunk)
        emg = float(np.mean(aux["EMG"])) if "EMG" in aux else None
        vig = self._vigilance(vec, emg)
        res["vigilance"] = vig
        h[vig] += 1
        pool = self._pool.setdefault(vig, deque(maxlen=self.long.window_epochs))
        if vig not in self._fit:
            pool.append(vec)
            if len(pool) >= self.long.min_state_epochs:
                self._refit(vig)
            if not self._fit:
                res["state"] = self._state = "warmup"
                return res
            res["state"] = self._state = "normal" if self._state == "warmup" else self._state
            return res
        med, scale, since = self._fit[vig]
        z = np.clip((vec - med) / scale, -self.cfg.z_clip, self.cfg.z_clip)
        score = float(np.sqrt(np.mean(z ** 2)))
        self._ewma = self.cfg.ewma_lambda * score + (1 - self.cfg.ewma_lambda) * self._ewma
        cfg, was_alert = self.cfg, self._state == "alert"
        self._above = self._above + 1 if self._ewma >= cfg.alert_level else 0
        if was_alert:
            if self._ewma < cfg.clear_level:
                self._state = "normal"
        elif self._above >= cfg.persist:
            self._state = "alert"
        else:
            self._recent.append(self._ewma)
            rising = len(self._recent) == self._recent.maxlen and all(
                b > a for a, b in zip(list(self._recent)[:-1], list(self._recent)[1:]))
            self._state = "normal" if self._ewma < cfg.warn_level else ("rising" if (rising or self._state == "rising") else self._state)
        if self._state == "alert" and not was_alert:
            h["alerts"] += 1
        if self._state not in ("alert", "rising") and self._ewma < cfg.warn_level:
            pool.append(vec)                       # only quiet epochs refresh the baseline
            since += 1
            self._fit[vig] = (med, scale, since)
            if since >= self.long.refit_every:
                self._refit(vig)
        h["score_sum"] += score
        h["scored"] += 1
        names = self._names()
        res.update(score=round(score, 3), ewma=round(self._ewma, 3), state=self._state,
                   top_features=[(names[i], round(float(z[i]), 2)) for i in np.argsort(-abs(z))[:3]])
        return res

    # -- reporting and persistence --------------------------------------------------------
    def hourly_summary(self) -> list[dict]:
        rows = []
        for hour, h in sorted(self._hours.items()):
            n = max(h["accepted"], 1)
            rows.append({"hour": hour, "epochs": h["epochs"], "accepted_fraction": round(h["accepted"] / max(h["epochs"], 1), 3),
                         "mean_score": round(h["score_sum"] / h["scored"], 3) if h["scored"] else None,
                         "alerts": h["alerts"], **{f"{k}_fraction": round(h[k] / n, 3) for k in ("slow", "fast", "active")}})
        return rows

    def duration_h(self) -> float:
        return (self._t0 + len(self._buf)) / self.fs / 3600

    def status(self) -> dict:
        d = self.duration_h()
        return {"duration_h": round(d, 3), "min_hours": self.long.min_hours, "meets_min_duration": d >= self.long.min_hours,
                "alerts": sum(h["alerts"] for h in self._hours.values()), "state": self._state,
                "baselines": {k: len(v) for k, v in self._pool.items()}}

    def save(self, path):
        meta = {"t0": self._t0, "ewma": self._ewma, "state": self._state, "hours": self._hours,
                "fits": {k: v[2] for k, v in self._fit.items()}, "above": self._above}
        arrays = {f"pool_{k}": np.vstack(v) for k, v in self._pool.items() if len(v)}
        for k, (m, sc, _) in self._fit.items():
            arrays[f"median_{k}"], arrays[f"scale_{k}"] = m, sc
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)

    def load(self, path):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        self._t0, self._ewma, self._state, self._above = meta["t0"], meta["ewma"], meta["state"], meta["above"]
        self._hours = {int(k): v for k, v in meta["hours"].items()}
        for k in [n[5:] for n in d.files if n.startswith("pool_")]:
            self._pool[k] = deque(d[f"pool_{k}"], maxlen=self.long.window_epochs)
        for k, since in meta["fits"].items():
            self._fit[k] = (d[f"median_{k}"], d[f"scale_{k}"], since)
        return self
