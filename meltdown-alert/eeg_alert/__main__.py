"""Command-line entry point.

Examples
--------
Synthetic demo (no hardware), with plot and evaluation:
    python -m eeg_alert --source synthetic --minutes 40 --plot docs/demo.png

Muse headband (run `muselsl stream` in another terminal first):
    python -m eeg_alert --source lsl --fs 256 --channels TP9 AF7 AF8 TP10

OpenBCI Cyton (OpenBCI GUI -> Networking -> LSL, type EEG):
    python -m eeg_alert --source lsl --fs 250 --channels Fp1 Fp2 F3 F4 --frontal F3 F4

Send alerts to WhatsApp (Twilio env vars set) and a webhook:
    python -m eeg_alert --source lsl --notify console whatsapp webhook --webhook-url https://...
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math

from .config import Config
from .evaluate import evaluate
from .notify import build
from .pipeline import run


def main() -> None:
    p = argparse.ArgumentParser(prog="eeg_alert", description="EEG and optional wearable arousal tracking (research prototype, not clinically validated)")
    p.add_argument("--source", choices=["synthetic", "lsl", "csv"], default="synthetic")
    p.add_argument("--file", help="CSV file for --source csv")
    p.add_argument("--fs", type=float, default=256.0)
    p.add_argument("--channels", nargs="+", default=["TP9", "AF7", "AF8", "TP10"])
    p.add_argument("--frontal", nargs=2, default=["AF7", "AF8"], metavar=("LEFT", "RIGHT"))
    p.add_argument("--baseline-sec", type=float, default=180.0)
    p.add_argument("--profile", choices=["meltdown", "psychosis"], default="meltdown",
                   help="meltdown = autistic meltdown (10 min); psychosis = agitation/aggression risk in acute psychosis (20 min)")
    p.add_argument("--model", choices=["index", "kalman"], default="index",
                   help="index = weighted z-sum; kalman = state-space model with FFT-derived noise")
    p.add_argument("--bvc-file", help="CSV of Broset Violence Checklist observations (raises sensitivity for 24 h)")
    p.add_argument("--horizon-min", type=float, default=None, help="override the profile's early-warning horizon (min)")
    p.add_argument("--cusum-k", type=float, default=0.3, help="early-warning drift reference (z)")
    p.add_argument("--cusum-h", type=float, default=80.0,
                   help="early-warning threshold; lower = earlier but more false alarms")
    p.add_argument("--minutes", type=float, default=50.0, help="synthetic session length")
    p.add_argument("--events", type=float, nargs="*", default=None,
                   help="event times (min); synthetic defaults to 22,42; CSV has no labels unless supplied")
    p.add_argument("--wearable-file", action="append", default=[], metavar="CSV",
                   help="optional canonical wearable CSV; repeat for distinct sensor metrics")
    p.add_argument("--session-start", help="shared ISO-8601 origin with timezone for absolute timestamps")
    p.add_argument("--wearable-offset-sec", type=float, default=0.0,
                   help="known clock correction added to all wearable timestamps (seconds)")
    p.add_argument("--wearable-fusion-weight", type=float, default=0.25,
                   help="experimental wearable fraction of fused index, in [0,1]")
    p.add_argument("--wearable-weight", action="append", default=[], metavar="METRIC=WEIGHT",
                   help="override a signed wearable feature weight; temperature/motion default to context only")
    p.add_argument("--wearable-max-age", action="append", default=[], metavar="METRIC=SECONDS",
                   help="override a metric's maximum sample age before it becomes stale")
    p.add_argument("--wearable-min-quality", type=float, default=0.6,
                   help="reject supplied wearable quality below this value, in [0,1]")
    p.add_argument("--steps-jsonl", help="save per-window results, wearable QC and provenance as strict JSONL")
    p.add_argument("--notify", nargs="+", default=["console"], choices=["console", "webhook", "whatsapp"])
    p.add_argument("--webhook-url")
    p.add_argument("--plot", help="save a PNG of the session")
    p.add_argument("--report", help="save evaluation JSON (synthetic or when --events given)")
    a = p.parse_args()

    wearable = None
    try:
        if a.session_start:
            from .timing import _absolute_time
            _absolute_time(a.session_start)
        if a.wearable_file:
            if a.source == "lsl":
                p.error("wearable file fusion supports synthetic/CSV replay only; live LSL clock bridging is not implemented")
            from .wearable import LIMITS, WearableConfig, WearableProcessor, WearableReplay, load_measurements
            defaults = WearableConfig()
            weights, ages = dict(defaults.weights), dict(defaults.max_age_sec)
            for entries, target in ((a.wearable_weight, weights), (a.wearable_max_age, ages)):
                for entry in entries:
                    key, val = entry.split("=", 1)
                    if key not in LIMITS:
                        raise ValueError(f"Unknown wearable metric: {key!r}")
                    target[key] = float(val)
            wcfg = WearableConfig(min_quality=a.wearable_min_quality,
                                  fusion_weight=a.wearable_fusion_weight,
                                  max_age_sec=ages, weights=weights)
            measurements = []
            for path in a.wearable_file:
                measurements.extend(load_measurements(path, session_start=a.session_start,
                                                       offset_sec=a.wearable_offset_sec))
            wearable = WearableProcessor(WearableReplay(measurements), a.baseline_sec, wcfg)
        if not math.isfinite(a.fs) or a.fs <= 90:
            raise ValueError("--fs must be finite and greater than 90 Hz for the configured EEG bands")
        if not math.isfinite(a.baseline_sec) or a.baseline_sec <= 0:
            raise ValueError("--baseline-sec must be finite and positive")
    except (ValueError, OSError) as exc:
        p.error(str(exc))

    event_minutes = a.events if a.events is not None else ([22.0, 42.0] if a.source == "synthetic" else [])

    from .profiles import apply_profile
    cfg = apply_profile(Config(fs=a.fs, channels=a.channels, frontal_pair=tuple(a.frontal),
                               baseline_sec=a.baseline_sec, cusum_k=a.cusum_k, cusum_h=a.cusum_h),
                        a.profile)
    cfg.model = a.model
    if a.horizon_min is not None:
        cfg.horizon_min = a.horizon_min
    context = None
    if a.bvc_file:
        from .clinical import load_bvc_csv
        context = load_bvc_csv(a.bvc_file)

    if a.source == "synthetic":
        from .sources import SyntheticEEG
        src = SyntheticEEG(cfg, minutes=a.minutes, events_min=event_minutes)
    elif a.source == "lsl":
        from .sources import LSLSource
        src = LSLSource(cfg)
    else:
        from .sources import CSVSource
        if not a.file:
            p.error("--file is required for --source csv")
        try:
            src = CSVSource(cfg, a.file, timestamped=wearable is not None, session_start=a.session_start)
        except (ValueError, OSError) as exc:
            p.error(str(exc))

    notifiers = build(a.notify, a.webhook_url)
    print(f"Calibrating personal baseline for {cfg.baseline_sec:.0f} s - please stay calm and still...")
    print(f"Profile: {a.profile} | early-warning horizon {cfg.horizon_min:.0f} min")
    if wearable is not None:
        print("Experimental wearable file fusion; engineering demonstration only, no clinical validation.")
    chunks = src.timestamped_stream() if a.source == "csv" and wearable is not None else src.stream()
    steps = []
    with open(a.steps_jsonl, "w", encoding="utf-8") if a.steps_jsonl else nullcontext() as output:
        for step in run(chunks, cfg, notifiers, context, wearable=wearable):
            steps.append(step)
            if output is not None:
                row = asdict(step)
                row["provenance"] = {"eeg_source": a.source, "eeg_model": cfg.model,
                                     "wearable_source": "csv" if wearable else None,
                                     "event_labels": ("synthetic" if a.source == "synthetic" else
                                                      "user_supplied" if event_minutes else "none"),
                                     "clinical_validation": False}
                output.write(json.dumps(row, allow_nan=False) + "\n")

    if wearable is not None and not steps:
        p.error("No complete EEG windows; check recording length, sample rate and timestamp continuity")

    events_sec = [e * 60 for e in event_minutes]
    res = None
    if events_sec:
        res = evaluate(steps, events_sec, cfg.horizon_min)
        print(json.dumps(res, indent=2))
    if a.report:
        if res is None:
            res = {"n_windows": len(steps), "evaluation": None,
                   "reason": "No event labels supplied; no prediction performance estimated"}
        if wearable is not None:
            res = dict(res, provenance={"eeg_source": a.source, "eeg_model": cfg.model,
                                       "wearable_source": "csv",
                                       "clinical_validation": False})
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, allow_nan=False)
    if a.plot:
        from .plot import plot_session
        truth = src.arousal if a.source == "synthetic" else None
        plot_session(steps, cfg, a.plot, events_sec, truth)
        print(f"Saved plot to {a.plot}")


if __name__ == "__main__":
    main()
