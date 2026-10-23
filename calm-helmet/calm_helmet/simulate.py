"""Synthetic multimodal sessions for software testing.

Everything here is an ASSUMPTION about how signals might behave before a meltdown.
It exists only to check that the software does what it is designed to do.
It says nothing about real people.

Model (1 Hz):
  r(t)  cortical/behavioural build-up: rises over `buildup_min` minutes before each
        event (quadratic ramp), stays at 1 for the event, then decays.
  a(t)  autonomic arousal = r(t) + harmless stress spikes + physical exercise.
  m(t)  head movement = exercise + late agitation (r^3).

The fNIRS channel is deliberately contaminated by scalp blood flow, which follows
autonomic arousal and slow Mayer waves. A short-separation channel sees the scalp
part only, so `fnirs.short_separation_clean` can remove it.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Session:
    t: np.ndarray                 # seconds
    raw: dict                     # feature name -> 1D array (fnirs HbO/HbR still contaminated)
    short_sep_hbo: np.ndarray
    short_sep_hbr: np.ndarray
    events: list                  # onset times in seconds
    exercise: list                # (start, end) tuples


def _ramp(t, events, buildup_s, dur_s=60, decay_s=120):
    r = np.zeros_like(t, dtype=float)
    for e in events:
        pre = (t >= e - buildup_s) & (t < e)
        r[pre] = np.maximum(r[pre], ((t[pre] - (e - buildup_s)) / buildup_s) ** 2)
        during = (t >= e) & (t < e + dur_s)
        r[during] = 1.0
        post = t >= e + dur_s
        r[post] = np.maximum(r[post], np.exp(-(t[post] - e - dur_s) / decay_s))
    return r


def simulate_session(
    duration_min=60,
    n_events=2,
    n_spikes=3,
    n_exercise=1,
    buildup_min=15,
    fnirs_dropout=0.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    T = int(duration_min * 60)
    t = np.arange(T, dtype=float)

    # event onsets: after the calibration period plus a full build-up, well separated
    events = []
    if n_events:
        lo, hi = 3 * 60 + buildup_min * 60 + 60, T - 120
        slots = np.linspace(lo, hi, n_events + 1)
        for i in range(n_events):
            events.append(float(rng.uniform(slots[i], slots[i + 1] - 300)))
    r = _ramp(t, events, buildup_min * 60)

    # harmless autonomic stress spikes (startle, a loud noise) - not meltdowns
    spikes = np.zeros(T)
    for _ in range(n_spikes):
        c = rng.uniform(240, T - 120)
        spikes += 1.2 * np.exp(-0.5 * ((t - c) / 25.0) ** 2)

    # physical exercise: HR/EDA up together with head movement
    exercise = np.zeros(T)
    ex_windows = []
    for _ in range(n_exercise):
        s = rng.uniform(240, T - 600)
        e = s + rng.uniform(240, 420)
        ex_windows.append((s, e))
        exercise[(t >= s) & (t < e)] = 1.0
    exercise = np.convolve(exercise, np.ones(30) / 30, mode="same")

    a = r + spikes + 1.4 * exercise
    m = 1.3 * exercise + 0.8 * r ** 3

    def noise(sd):
        return rng.normal(0, sd, T)

    # slow scalp haemodynamics (Mayer waves ~0.1 Hz -> aliased slow drift at 1 Hz) + random walk
    mayer = 0.6 * np.sin(2 * np.pi * t / rng.uniform(40, 80)) + np.cumsum(noise(0.02))
    mayer -= mayer.mean()
    scalp = 0.8 * a + mayer

    raw = {
        "eeg_beta_alpha": 2.0 * r + noise(0.5),
        "eeg_alpha_rel": 0.5 - 0.15 * r + noise(0.05),
        # brain part is modest; scalp contamination is large on purpose
        "fnirs_pfc_hbo": 1.2 * r + 1.1 * scalp + noise(0.25),
        "fnirs_pfc_hbr": -0.4 * r - 0.3 * scalp + noise(0.12),
        "emg_temporalis_rms": 1.0 + 1.6 * r ** 1.5 + 0.3 * m + noise(0.3),
        "hr": 72 + 14 * a + noise(2.5),
        "hrv_rmssd": 42 - 12 * a + noise(3.0),
        "eda": 2.0 + 1.3 * a + noise(0.25),
        "imu_activity": 0.2 + m + np.abs(noise(0.08)),
    }
    ss_hbo = scalp + noise(0.1)
    ss_hbr = -0.27 * scalp + noise(0.05)

    if fnirs_dropout > 0:  # lost optode contact (hair, movement)
        n_gaps = max(1, int(fnirs_dropout * T / 120))
        for _ in range(n_gaps):
            s = int(rng.uniform(240, T - 120))
            sl = slice(s, s + 120)
            for k in ("fnirs_pfc_hbo", "fnirs_pfc_hbr"):
                raw[k][sl] = np.nan
            ss_hbo[sl] = np.nan
            ss_hbr[sl] = np.nan

    return Session(t, raw, ss_hbo, ss_hbr, events, ex_windows)
