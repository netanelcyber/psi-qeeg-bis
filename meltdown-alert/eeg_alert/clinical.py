"""Structured clinical observations that modulate the physiological alert.

Brøset Violence Checklist (BVC)
-------------------------------
Six behaviours, each scored 1 if present *relative to the patient's usual behaviour*:
confused, irritable, boisterous, verbally threatening, physically threatening, attacking objects.
Total 0 = low, 1-2 = moderate, >2 = very high risk of violence within the next 24 hours.

How it is used here: a recent BVC score does not raise an alert by itself (except a one-time
notice for very high risk). Instead it makes the physiological detector more sensitive for
24 hours, so a smaller rise in arousal triggers the early warning. This combines a validated
clinical tool with continuous monitoring; the combination itself is not validated.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

BVC_ITEMS = ("confused", "irritable", "boisterous", "verbal_threats", "physical_threats", "attacking_objects")
BVC_VALID_SEC = 24 * 3600.0

# (cusum_h multiplier, alert_level offset) for each BVC risk band
BVC_MODIFIERS: Dict[str, Tuple[float, float]] = {
    "low": (1.0, 0.0),
    "moderate": (0.8, -0.25),
    "very_high": (0.65, -0.5),
}


def bvc_band(score: int) -> str:
    if score <= 0:
        return "low"
    return "moderate" if score <= 2 else "very_high"


@dataclass
class BVCObservation:
    t: float                      # session time (s)
    items: Dict[str, bool]

    @property
    def score(self) -> int:
        return sum(1 for k in BVC_ITEMS if self.items.get(k))


class ClinicalContext:
    def __init__(self, observations: Optional[List[BVCObservation]] = None) -> None:
        self.obs = sorted(observations or [], key=lambda o: o.t)

    def current(self, t: float) -> Optional[BVCObservation]:
        latest = None
        for o in self.obs:
            if o.t <= t:
                latest = o
            else:
                break
        if latest is None or t - latest.t > BVC_VALID_SEC:
            return None
        return latest

    def modifiers(self, t: float) -> Tuple[float, float, str, int]:
        o = self.current(t)
        if o is None:
            return 1.0, 0.0, "none", 0
        band = bvc_band(o.score)
        h_mult, lvl_off = BVC_MODIFIERS[band]
        return h_mult, lvl_off, band, o.score


def load_bvc_csv(path: str) -> ClinicalContext:
    """CSV columns: time_min, confused, irritable, boisterous, verbal_threats,
    physical_threats, attacking_objects  (values 0/1)."""
    obs: List[BVCObservation] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_min"]) * 60.0
            items = {k: str(row.get(k, "0")).strip() in ("1", "true", "True", "yes") for k in BVC_ITEMS}
            obs.append(BVCObservation(t, items))
    return ClinicalContext(obs)
