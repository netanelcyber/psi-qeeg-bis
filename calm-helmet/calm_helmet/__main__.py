"""Synthetic benchmark: python -m calm_helmet [--out docs/synthetic_validation.json] [--plot docs/demo.png]"""
import argparse
import dataclasses
import json

from .config import Config
from .evaluate import score, summarise
from .pipeline import run
from .simulate import simulate_session

ABLATIONS = {
    "all (helmet + wrist)": None,
    "eeg only": {"eeg"},
    "fnirs only": {"fnirs"},
    "autonomic only (wrist-style, no movement gating)": {"autonomic"},
    "eeg + fnirs": {"eeg", "fnirs"},
    "eeg + fnirs + emg": {"eeg", "fnirs", "emg"},
}


def benchmark(cfg, n_event_sessions=12, n_quiet_sessions=4, fnirs_dropout=0.0,
              use_modalities=None, clean_fnirs=True):
    results = []
    for k in range(n_event_sessions):
        s = simulate_session(n_events=2, fnirs_dropout=fnirs_dropout, seed=1000 + k)
        alerts, *_ = run(s, cfg, use_modalities, clean_fnirs)
        results.append(score(alerts, s.events, cfg.horizon_min))
    quiet_fa = 0
    for k in range(n_quiet_sessions):
        s = simulate_session(n_events=0, n_spikes=5, n_exercise=2,
                             fnirs_dropout=fnirs_dropout, seed=5000 + k)
        alerts, *_ = run(s, cfg, use_modalities, clean_fnirs)
        quiet_fa += len(alerts)
    out = summarise(results)
    out["false_alarms_in_quiet_hours"] = quiet_fa
    out["quiet_hours"] = n_quiet_sessions
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="calm_helmet")
    p.add_argument("--out", help="write benchmark JSON here")
    p.add_argument("--plot", help="save a demo session figure here")
    args = p.parse_args(argv)

    cfg = Config()
    one_mod = dataclasses.replace(cfg, min_modalities=1)
    report = {
        "note": "Synthetic data only. Results show the software behaves as designed, not that it works on people.",
        "main": benchmark(cfg),
        "fnirs_without_short_separation_cleaning": benchmark(cfg, clean_fnirs=False),
        "with_fnirs_contact_loss": benchmark(cfg, fnirs_dropout=0.15),
        "ablation": {name: benchmark(one_mod if mods and len({m for m in mods} - {"imu"}) == 1 else cfg,
                                     use_modalities=mods)
                     for name, mods in ABLATIONS.items()},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    if args.plot:
        from .plot import plot_session
        plot_session(simulate_session(n_events=2, seed=1000), cfg, args.plot)


if __name__ == "__main__":
    main()
