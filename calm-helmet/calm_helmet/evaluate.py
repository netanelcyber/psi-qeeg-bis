"""Lead time and false alarms against labelled event onsets."""
import numpy as np


def score(alerts, events, horizon_min=10.0, post_event_sec=300, early_window_min=None):
    """An alert counts for an event if it falls in [onset - window, onset).
    Alerts after an onset (until post_event_sec) are ignored; everything else is a false alarm.
    Early warnings are scored against a window 1.5 x the target horizon, so that an
    early warning a little ahead of the target still counts.
    """
    ew = (early_window_min or 1.5 * horizon_min) * 60
    res = {"early_lead_min": [], "imminent_lead_min": [], "missed": 0, "false_alarms": 0}
    used = set()
    for e in events:
        hit = False
        for lvl, win in (("early", ew), ("imminent", horizon_min * 60)):
            cands = [a for a in alerts if a.level == lvl and e - win <= a.t < e]
            if cands:
                first = min(cands, key=lambda a: a.t)
                res[f"{lvl}_lead_min"].append((e - first.t) / 60)
                used.update(id(a) for a in cands)
                hit = True
        if not hit:
            res["missed"] += 1
    for a in alerts:
        if id(a) in used:
            continue
        if any(e <= a.t < e + post_event_sec for e in events):
            continue
        res["false_alarms"] += 1
    return res


def summarise(results):
    early = np.concatenate([r["early_lead_min"] for r in results] or [[]])
    imm = np.concatenate([r["imminent_lead_min"] for r in results] or [[]])

    def stats(x):
        if len(x) == 0:
            return None
        return {"median": round(float(np.median(x)), 1),
                "min": round(float(np.min(x)), 1), "max": round(float(np.max(x)), 1), "n": int(len(x))}

    return {
        "early_warning_lead_min": stats(early),
        "imminent_lead_min": stats(imm),
        "missed": int(sum(r["missed"] for r in results)),
        "false_alarms": int(sum(r["false_alarms"] for r in results)),
    }
