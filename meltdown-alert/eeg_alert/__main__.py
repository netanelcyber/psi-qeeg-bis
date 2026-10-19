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
import json

from .config import Config
from .evaluate import evaluate
from .notify import build
from .pipeline import run


def main() -> None:
    p = argparse.ArgumentParser(prog="eeg_alert", description="EEG early warning for autistic meltdowns (research prototype)")
    p.add_argument("--source", choices=["synthetic", "lsl", "csv"], default="synthetic")
    p.add_argument("--file", help="CSV file for --source csv")
    p.add_argument("--fs", type=float, default=256.0)
    p.add_argument("--channels", nargs="+", default=["TP9", "AF7", "AF8", "TP10"])
    p.add_argument("--frontal", nargs=2, default=["AF7", "AF8"], metavar=("LEFT", "RIGHT"))
    p.add_argument("--baseline-sec", type=float, default=180.0)
    p.add_argument("--profile", choices=["meltdown", "psychosis"], default="meltdown",
                   help="meltdown = autistic meltdown (10 min); psychosis = agitation/aggression risk in acute psychosis (20 min)")
    p.add_argument("--bvc-file", help="CSV of Broset Violence Checklist observations (raises sensitivity for 24 h)")
    p.add_argument("--horizon-min", type=float, default=None, help="override the profile's early-warning horizon (min)")
    p.add_argument("--cusum-k", type=float, default=0.3, help="early-warning drift reference (z)")
    p.add_argument("--cusum-h", type=float, default=80.0,
                   help="early-warning threshold; lower = earlier but more false alarms")
    p.add_argument("--minutes", type=float, default=50.0, help="synthetic session length")
    p.add_argument("--events", type=float, nargs="*", default=[22.0, 42.0], help="synthetic event times (min)")
    p.add_argument("--notify", nargs="+", default=["console"], choices=["console", "webhook", "whatsapp"])
    p.add_argument("--webhook-url")
    p.add_argument("--plot", help="save a PNG of the session")
    p.add_argument("--report", help="save evaluation JSON (synthetic or when --events given)")
    a = p.parse_args()

    from .profiles import apply_profile
    cfg = apply_profile(Config(fs=a.fs, channels=a.channels, frontal_pair=tuple(a.frontal),
                               baseline_sec=a.baseline_sec, cusum_k=a.cusum_k, cusum_h=a.cusum_h),
                        a.profile)
    if a.horizon_min is not None:
        cfg.horizon_min = a.horizon_min
    context = None
    if a.bvc_file:
        from .clinical import load_bvc_csv
        context = load_bvc_csv(a.bvc_file)

    if a.source == "synthetic":
        from .sources import SyntheticEEG
        src = SyntheticEEG(cfg, minutes=a.minutes, events_min=a.events)
    elif a.source == "lsl":
        from .sources import LSLSource
        src = LSLSource(cfg)
    else:
        from .sources import CSVSource
        if not a.file:
            p.error("--file is required for --source csv")
        src = CSVSource(cfg, a.file)

    notifiers = build(a.notify, a.webhook_url)
    print(f"Calibrating personal baseline for {cfg.baseline_sec:.0f} s - please stay calm and still...")
    print(f"Profile: {a.profile} | early-warning horizon {cfg.horizon_min:.0f} min")
    steps = list(run(src.stream(), cfg, notifiers, context))

    events_sec = [e * 60 for e in a.events] if a.source == "synthetic" or a.report else []
    if events_sec:
        res = evaluate(steps, events_sec, cfg.horizon_min)
        print(json.dumps(res, indent=2))
        if a.report:
            with open(a.report, "w") as f:
                json.dump(res, f, indent=2)
    if a.plot:
        from .plot import plot_session
        truth = src.arousal if a.source == "synthetic" else None
        plot_session(steps, cfg, a.plot, events_sec, truth)
        print(f"Saved plot to {a.plot}")


if __name__ == "__main__":
    main()
