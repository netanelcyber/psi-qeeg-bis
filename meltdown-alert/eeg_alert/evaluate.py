"""Lead-time / false-alarm evaluation against labelled event times."""
from __future__ import annotations

from typing import Dict, List, Sequence

from .detector import EARLY_WARNING, IMMINENT, Step


def evaluate(steps: Sequence[Step], events_sec: Sequence[float], horizon_min: float,
             lookback_min: float = 15.0) -> Dict[str, object]:
    alerts = [s.alert for s in steps if s.alert is not None and s.alert.level in (EARLY_WARNING, IMMINENT)]
    per_event: List[Dict[str, object]] = []
    used = set()
    for e in events_sec:
        row: Dict[str, object] = {"event_min": round(e / 60, 2)}
        for lvl in (EARLY_WARNING, IMMINENT):
            cands = [a for a in alerts if a.level == lvl and e - lookback_min * 60 <= a.t <= e]
            if cands:
                first = min(cands, key=lambda a: a.t)
                row[f"{lvl.lower()}_lead_min"] = round((e - first.t) / 60, 2)
                used.update(id(a) for a in cands)
            else:
                row[f"{lvl.lower()}_lead_min"] = None
        per_event.append(row)
    # an alert is a false alarm if no event follows within horizon + 2 minutes
    false_alarms = [a for a in alerts
                    if not any(0 <= e - a.t <= (horizon_min + 2) * 60 for e in events_sec)]
    return {
        "events": per_event,
        "n_alerts": len(alerts),
        "false_alarms": len(false_alarms),
        "detected_early": sum(1 for r in per_event if r["early_warning_lead_min"] is not None),
        "detected_imminent": sum(1 for r in per_event if r["imminent_lead_min"] is not None),
    }
