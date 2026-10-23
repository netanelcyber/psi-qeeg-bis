"""Trait calibration with a short verbal fluency task (VFT) on prefrontal fNIRS.

Protocol (single block, as in the clinical optical-topography protocol used in Japan):
  30 s pre-task rest -> 60 s verbal fluency (say as many words as possible starting
  with a given letter/syllable, switched every 20 s) -> 70 s post-task rest.

From the cleaned prefrontal HbO response we compute:
  integral       mean HbO change during the task (after linear baseline correction)
  centroid_s     centre of gravity of the positive response over task + post-task,
                 in seconds from task onset (later = slower / more delayed response)
  persistence    mean HbO in the first 30 s after the task, relative to the task mean
  laterality     (R - L) / (|R| + |L|) of the task integral, if both sides are given
  snr            task mean / SD of the pre-task rest

WHAT THIS IS FOR
  * Checking that the fNIRS channel works for this person on this day. A weak or noisy
    response lowers the fNIRS weight in the fused index instead of letting a bad
    channel drive alerts (`apply_calibration`).
  * Recording a personal haemodynamic profile (amplitude, timing) for research, so
    that later state changes can be compared with the person's own trait response.

WHAT THIS IS NOT
  Group studies report reduced frontal VFT responses in autism and reduced *and
  delayed* responses in schizophrenia, but these are group-level findings with large
  overlap. The metrics here must NOT be used to diagnose, or to choose between the
  `meltdown` and `psychosis` profiles. That choice is a clinical decision.
"""
import dataclasses

import numpy as np

PRE, TASK, POST = 30, 60, 70


def vft_metrics(hbo, hbo_left=None, hbo_right=None, fs=1.0, pre=PRE, task=TASK, post=POST):
    """hbo: 1D cleaned prefrontal HbO covering pre + task + post (sampled at fs)."""
    y = _baseline_correct(np.asarray(hbo, float), fs, pre, task, post)
    p, t_end = int(pre * fs), int((pre + task) * fs)
    end = int((pre + task + post) * fs)
    task_seg = y[p:t_end]
    win = y[p:end]
    pos = np.clip(win, 0, None)
    tt = np.arange(len(win)) / fs
    centroid = float((tt * pos).sum() / pos.sum()) if pos.sum() > 0 else float("nan")
    integral = float(np.nanmean(task_seg))
    post30 = y[t_end:t_end + int(30 * fs)]
    pre_sd = float(np.nanstd(y[:p])) or 1e-9
    out = {
        "integral": integral,
        "centroid_s": centroid,
        "persistence": float(np.nanmean(post30) / integral) if integral > 0 else float("nan"),
        "snr": integral / pre_sd,
        "laterality": float("nan"),
    }
    if hbo_left is not None and hbo_right is not None:
        L = np.nanmean(_baseline_correct(np.asarray(hbo_left, float), fs, pre, task, post)[p:t_end])
        R = np.nanmean(_baseline_correct(np.asarray(hbo_right, float), fs, pre, task, post)[p:t_end])
        den = abs(L) + abs(R)
        out["laterality"] = float((R - L) / den) if den > 0 else 0.0
    return out


def _baseline_correct(y, fs, pre, task, post):
    """Linear baseline through the last 10 s of pre-task rest and the last 5 s of post-task rest."""
    p, end = int(pre * fs), int((pre + task + post) * fs)
    y = y[:end].copy()
    a_idx = np.arange(max(0, p - int(10 * fs)), p)
    b_idx = np.arange(end - int(5 * fs), end)
    xa, xb = a_idx.mean(), b_idx.mean()
    ya, yb = np.nanmean(y[a_idx]), np.nanmean(y[b_idx])
    slope = (yb - ya) / (xb - xa)
    return y - (ya + slope * (np.arange(len(y)) - xa))


def fnirs_reliability(metrics, snr_good=3.0, snr_min=1.0):
    """0..1: how much to trust this person's fNIRS today, from the VFT signal-to-noise ratio."""
    snr = metrics.get("snr", float("nan"))
    if not np.isfinite(snr) or snr <= snr_min:
        return 0.0
    return float(min(1.0, (snr - snr_min) / (snr_good - snr_min)))


def apply_calibration(cfg, metrics, floor=0.0):
    """Return a copy of cfg with the fNIRS feature weights scaled by the VFT reliability."""
    r = max(floor, fnirs_reliability(metrics))
    feats = tuple(dataclasses.replace(f, weight=f.weight * r) if f.modality == "fnirs" else f
                  for f in cfg.features)
    return dataclasses.replace(cfg, features=feats)


def simulate_vft(amplitude=1.0, delay_s=0.0, noise=0.15, persistence=0.3, seed=0):
    """Synthetic VFT HbO trace (pre + task + post). For tests and demos only."""
    rng = np.random.default_rng(seed)
    n = PRE + TASK + POST
    t = np.arange(n, dtype=float)
    on = PRE + delay_s
    rise = 1 - np.exp(-np.clip(t - on, 0, None) / 8.0)
    off = PRE + TASK + delay_s
    decay_tau = 10 + 60 * persistence
    resp = np.where(t < off, rise, rise[min(int(off), n - 1)] * np.exp(-(t - off) / decay_tau))
    resp[t < on] = 0
    return amplitude * resp + rng.normal(0, noise, n) + np.cumsum(rng.normal(0, 0.005, n))
