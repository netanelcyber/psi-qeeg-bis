"""Sensor-level onset mapping: which scalp region changed first before an alert.

What this is
------------
For every channel we keep a personal baseline and a one-sided CUSUM of two per-channel
indices:
  * neural   - cortical activation (beta/alpha up, relative alpha down)
  * muscle   - 30-45 Hz power, which on dry headbands is mostly scalp EMG
The first channel whose CUSUM crosses its threshold since the last calm period is the
"earliest-changing region". When an alert fires, we report the order of onsets, a
left/right lateralisation index, and whether the earliest change looks neural or muscular.

What this is NOT
----------------
* Not source localisation. With 2-8 dry electrodes the inverse problem is hopelessly
  under-determined; sensor-level changes also reflect the reference and volume conduction.
* Not evidence of a brain "focus". Meltdowns and psychotic agitation are not known to start
  from a focal generator the way epileptic seizures can.
For dense EEG (>= 32 channels, digitised positions) a source-space analysis (e.g. MNE-Python
with eLORETA on band-power changes) would be the next step; it is out of scope here.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# 10-20 / 10-10 label -> human-readable region and hemisphere
REGIONS = {
    "Fp1": ("left frontopolar", "L"), "Fp2": ("right frontopolar", "R"), "Fpz": ("midline frontopolar", "M"),
    "AF7": ("left frontal", "L"), "AF8": ("right frontal", "R"), "AF3": ("left frontal", "L"), "AF4": ("right frontal", "R"),
    "F3": ("left frontal", "L"), "F4": ("right frontal", "R"), "F7": ("left inferior frontal", "L"),
    "F8": ("right inferior frontal", "R"), "Fz": ("midline frontal", "M"),
    "C3": ("left central", "L"), "C4": ("right central", "R"), "Cz": ("midline central", "M"),
    "T7": ("left temporal", "L"), "T8": ("right temporal", "R"), "T3": ("left temporal", "L"), "T4": ("right temporal", "R"),
    "TP9": ("left temporoparietal", "L"), "TP10": ("right temporoparietal", "R"),
    "P3": ("left parietal", "L"), "P4": ("right parietal", "R"), "Pz": ("midline parietal", "M"),
    "O1": ("left occipital", "L"), "O2": ("right occipital", "R"), "Oz": ("midline occipital", "M"),
}


def region_of(ch: str):
    return REGIONS.get(ch, (ch, "?"))


class _RobustBaseline:
    def __init__(self) -> None:
        self.vals: List[float] = []
        self.med = 0.0
        self.scale = 1.0

    def finalize(self) -> None:
        v = np.asarray(self.vals)
        self.med = float(np.median(v))
        self.scale = max(float(np.median(np.abs(v - self.med))) * 1.4826, 1e-6, 0.05 * abs(self.med))

    def z(self, x: float) -> float:
        return (x - self.med) / self.scale


class OnsetTracker:
    def __init__(self, channels: List[str], k: float = 0.5, h: float = 40.0) -> None:
        self.channels = list(channels)
        self.k, self.h = k, h
        kinds = ("bar", "alpha", "muscle")
        self.base = {c: {kd: _RobustBaseline() for kd in kinds} for c in self.channels}
        self.cusum = {c: {"neural": 0.0, "muscle": 0.0} for c in self.channels}
        self.onset: Dict[str, Dict[str, Optional[float]]] = {c: {"neural": None, "muscle": None} for c in self.channels}
        self.ready = False

    # -- calibration ------------------------------------------------------
    def add_baseline(self, pc: Dict[str, Dict[str, float]]) -> None:
        for c in self.channels:
            if c in pc:
                self.base[c]["bar"].vals.append(pc[c]["beta_alpha_ratio"])
                self.base[c]["alpha"].vals.append(pc[c]["alpha_rel"])
                self.base[c]["muscle"].vals.append(pc[c]["muscle_tension"])

    def finalize(self) -> None:
        for c in self.channels:
            for b in self.base[c].values():
                if b.vals:
                    b.finalize()
        self.ready = True

    # -- monitoring -------------------------------------------------------
    def update(self, t: float, pc: Optional[Dict[str, Dict[str, float]]]) -> None:
        if not self.ready or pc is None:
            return
        for c in self.channels:
            if c not in pc:
                continue
            b = self.base[c]
            neural = (b["bar"].z(pc[c]["beta_alpha_ratio"]) - b["alpha"].z(pc[c]["alpha_rel"])) / np.sqrt(2)
            muscle = b["muscle"].z(pc[c]["muscle_tension"])
            for kind, val in (("neural", neural), ("muscle", muscle)):
                s = max(0.0, self.cusum[c][kind] + val - self.k)
                self.cusum[c][kind] = s
                if s > self.h and self.onset[c][kind] is None:
                    self.onset[c][kind] = t

    def reset(self) -> None:
        """Called when the person is calm again: start a fresh onset episode."""
        for c in self.channels:
            self.cusum[c] = {"neural": 0.0, "muscle": 0.0}
            self.onset[c] = {"neural": None, "muscle": None}

    def summary(self, t: float) -> Optional[dict]:
        events = []
        for c in self.channels:
            for kind in ("neural", "muscle"):
                ot = self.onset[c][kind]
                if ot is not None:
                    events.append((ot, c, kind))
        if not events:
            return None
        events.sort()
        first_t, first_ch, first_kind = events[0]
        region, hemi = region_of(first_ch)
        # lateralisation of the neural change: (R - L) / (R + L) on current CUSUM magnitude
        left = sum(self.cusum[c]["neural"] for c in self.channels if region_of(c)[1] == "L")
        right = sum(self.cusum[c]["neural"] for c in self.channels if region_of(c)[1] == "R")
        lat = (right - left) / (right + left) if (right + left) > 0 else 0.0
        order = [{"channel": c, "region": region_of(c)[0], "kind": k, "min_before_alert": round((t - ot) / 60, 1)}
                 for ot, c, k in events]
        return {
            "first_channel": first_ch,
            "first_region": region,
            "first_kind": first_kind,
            "first_min_before_alert": round((t - first_t) / 60, 1),
            "lateralisation_R_minus_L": round(float(lat), 2),
            "order": order,
        }


def describe(summary: Optional[dict]) -> str:
    if not summary:
        return "Onset map: no channel-level onset detected yet."
    kind = ("cortical (beta/alpha) change" if summary["first_kind"] == "neural"
            else "30-45 Hz change, most likely muscle tension rather than brain activity")
    lat = summary["lateralisation_R_minus_L"]
    side = "right-dominant" if lat > 0.2 else "left-dominant" if lat < -0.2 else "bilateral"
    nxt = [o for o in summary["order"][1:4]]
    tail = "; then " + ", ".join(f"{o['region']} ({o['kind']})" for o in nxt) if nxt else ""
    return (f"Onset map (sensor level, not a brain focus): earliest change at {summary['first_region']} "
            f"({summary['first_channel']}), {summary['first_min_before_alert']} min before this alert - {kind}; "
            f"cortical change is {side} (R-L index {lat:+.2f}){tail}.")
