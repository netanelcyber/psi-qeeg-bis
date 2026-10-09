"""Small explicit CLI for EEG analysis, paper arithmetic and research training."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__


def _write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PSI qEEG/BIS research workbench; no clinical diagnosis")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("analyze", help="Raw CSV/EDF/BDF to epoch features and an offline report")
    p.add_argument("input", type=Path)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epoch-s", type=float, default=4)
    p.add_argument("--line-hz", type=float, choices=(50, 60))
    p.add_argument("--min-sqi", type=float, default=80)
    p = sub.add_parser("convert", help="Convert EDF/BDF/EEGLAB/FIF to project CSV + metadata; optional BIS-proxy analysis")
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--subject-id", required=True)
    p.add_argument("--channels", nargs="+")
    p.add_argument("--context", default="awake_observational_research")
    p.add_argument("--license", default="unspecified")
    p.add_argument("--bis-montage", choices=("left", "right"), help="Re-derive BIS-like Fp/F7 (or Fp2/F8) minus Fpz from a standard scan")
    p.add_argument("--bis-reference", help="Scan channel to use as reference when Fpz is absent (default: mean of Fp1,Fp2)")
    p.add_argument("--proxy", action="store_true", help="Also run analyze and add an uncalibrated bis_proxy column")
    p = sub.add_parser("convert-cohort", help="Batch-convert ASD / PSY / HC recordings from a manifest CSV into pooled, group-labelled features")
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epoch-s", type=float, default=4)
    p.add_argument("--workers", type=int, default=1, help="Parallel processes for thousands of scans")
    p.add_argument("--keep-raw", action="store_true", help="Keep per-subject recording.csv (large)")
    p.add_argument("--max-failures", type=int, help="Abort after more than this many failed scans")
    p = sub.add_parser("make-manifest", help="Scan ROOT/{ASD,PSY,HC}/ for EDF/BDF/SET/FIF files into a manifest CSV")
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("monitor", help="Replay a recording (or read stdin) through the streaming deviation monitor")
    p.add_argument("input", type=Path, nargs="?", help="Raw CSV; omit with --stdin")
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--stdin", action="store_true", help="Read comma-separated samples (one line per sample) from stdin")
    p.add_argument("--speed", type=float, default=0, help="1 = real time replay, 0 = as fast as possible")
    p.add_argument("--baseline-epochs", type=int, default=30)
    p.add_argument("--alert-level", type=float, default=3.0)
    p.add_argument("--verbose", action="store_true", help="Print every epoch, not only state changes")
    p = sub.add_parser("monitor-long", help="Multi-day (>= 24 h) streaming monitor with per-state baselines, hourly summary and checkpoints")
    p.add_argument("input", type=Path, help="Raw CSV, read in bounded-memory chunks")
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--species", choices=("rodent", "human"), default="rodent")
    p.add_argument("--min-hours", type=float, default=24)
    p.add_argument("--epoch-s", type=float, default=10)
    p.add_argument("--emg-active", type=float, help="EMG column threshold marking an 'active' state")
    p.add_argument("--resume", action="store_true", help="Continue from out/checkpoint.npz")
    p = sub.add_parser("fetch-vitaldb", help="Download real public BIS EEG and monitor numerics")
    p.add_argument("--caseid", type=int, default=1)
    p.add_argument("--start-s", type=float, default=332)
    p.add_argument("--duration-s", type=float, default=120)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p = sub.add_parser("score", help="Reproduce section 6.5 from ten supplied feature values and a baseline")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("calibrate", help="Fit personal statistics from explicitly labelled baseline epochs")
    p.add_argument("input", type=Path)
    p.add_argument("--subject-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("train", help="Optional labelled-cohort SVM with leave-one-subject-out validation")
    p.add_argument("input", type=Path)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("train-groups", help="Leave-one-subject-out ASD / PSY / HC classifier on cohort_features.csv")
    p.add_argument("input", type=Path)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            from .recording import read_recording
            from .analysis import analyze
            from .quality import QualityPolicy
            from .report import write_report
            recording = read_recording(args.input, args.metadata)
            features, summary = analyze(recording, args.epoch_s, args.line_hz, QualityPolicy(min_sqi=args.min_sqi))
            write_report(recording, features, summary, args.out)
            print(json.dumps({k: summary[k] for k in ("complete_epochs", "accepted_epochs", "rejected_epochs", "SOI", "RDI", "classification")}))
        elif args.command == "convert":
            from .convert import convert, bis_proxy
            meta = convert(args.input, args.out, args.subject_id, args.channels, args.context, license_id=args.license,
                           bis_side=args.bis_montage, bis_reference=args.bis_reference)
            result = {"channels": meta["eeg_columns"], "sampling_rate_hz": meta["sampling_rate_hz"]}
            if args.proxy:
                from .recording import read_recording
                from .analysis import analyze
                from .report import write_report
                recording = read_recording(args.out / "recording.csv", args.out / "metadata.json")
                features, summary = analyze(recording)
                for ch in recording.channels:
                    features[f"{ch}__bis_proxy_uncalibrated"] = bis_proxy(features, ch)
                write_report(recording, features, summary, args.out / "analysis")
                result["accepted_epochs"] = summary["accepted_epochs"]
            print(json.dumps(result))
        elif args.command == "convert-cohort":
            from .convert import convert_cohort
            print(json.dumps(convert_cohort(args.manifest, args.out, args.epoch_s, args.workers, args.keep_raw, args.max_failures)))
        elif args.command == "make-manifest":
            from .convert import build_manifest
            print(json.dumps({"files": build_manifest(args.root, args.out)}))
        elif args.command == "monitor":
            import time
            import numpy as np
            from .realtime import MonitorConfig, RealtimeMonitor, run_stream
            meta = json.loads(args.metadata.read_text(encoding="utf-8"))
            cfg = MonitorConfig(baseline_epochs=args.baseline_epochs, alert_level=args.alert_level)
            monitor = RealtimeMonitor(meta["sampling_rate_hz"], meta["eeg_columns"], cfg)
            step = max(1, round(0.25 * monitor.fs))

            def chunks():
                if args.stdin:
                    buf = []
                    for line in sys.stdin:
                        buf.append([float(v) for v in line.split(",")])
                        if len(buf) >= step:
                            yield np.array(buf)
                            buf = []
                else:
                    from .recording import read_recording
                    rec = read_recording(args.input, args.metadata)
                    for i in range(0, len(rec.eeg), step):
                        yield rec.eeg[i:i + step]
                        if args.speed > 0:
                            time.sleep(step / rec.fs / args.speed)
            transitions = run_stream(monitor, chunks(), verbose=args.verbose)
            print(json.dumps({"state_changes": len(transitions), "final_state": monitor.state,
                              "alerts": sum(t["state"] == "alert" for t in transitions)}), file=sys.stderr)
        elif args.command == "monitor-long":
            import pandas as pd
            from .realtime import LongRunConfig, LongRunMonitor, MonitorConfig
            meta = json.loads(args.metadata.read_text(encoding="utf-8"))
            fs, cols = float(meta["sampling_rate_hz"]), meta["eeg_columns"]
            monitor = LongRunMonitor(fs, cols, MonitorConfig(epoch_s=args.epoch_s, hop_s=args.epoch_s, species=args.species),
                                     LongRunConfig(min_hours=args.min_hours, emg_active=args.emg_active))
            args.out.mkdir(parents=True, exist_ok=True)
            checkpoint, skip = args.out / "checkpoint.npz", 0
            if args.resume and checkpoint.exists():
                monitor.load(checkpoint)
                skip = monitor._t0
            usecols = ["time_s", *cols] + (["EMG"] if args.emg_active is not None else [])
            seen, last_state, events_log = 0, None, open(args.out / "events.jsonl", "a", encoding="utf-8")
            for chunk in pd.read_csv(args.input, usecols=usecols, chunksize=round(fs * 60)):
                if seen + len(chunk) <= skip:
                    seen += len(chunk)
                    continue
                chunk = chunk.iloc[max(0, skip - seen):]
                seen += len(chunk) + max(0, skip - seen)
                for r in monitor.push(chunk[cols].to_numpy(float), emg=chunk["EMG"].to_numpy(float) if "EMG" in chunk else None):
                    if r["state"] != last_state:
                        events_log.write(json.dumps(r) + "\n")
                        events_log.flush()
                        last_state = r["state"]
                monitor.save(checkpoint)
            _write_json({"status": monitor.status(), "hourly": monitor.hourly_summary(),
                         "note": "Deviation from this animal's own per-state baseline; research only, not a validated predictor"},
                        args.out / "summary.json")
            st = monitor.status()
            print(json.dumps({k: st[k] for k in ("duration_h", "meets_min_duration", "alerts", "state")}))
        elif args.command == "fetch-vitaldb":
            from .vitaldb import download
            metadata = download(args.caseid, args.out, args.start_s, args.duration_s, args.cache)
            print(json.dumps({"caseid": args.caseid, "excerpt": metadata["excerpt"], "sha256": metadata["recording_csv_sha256"]}))
        elif args.command == "score":
            from .indices import score_indices
            supplied = json.loads(args.features.read_text(encoding="utf-8"))
            calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
            if not supplied.get("subject_id") or not calibration.get("subject_id"):
                raise ValueError("Current features and baseline must explicitly identify the subject")
            if not supplied.get("medication_profile_id") or not calibration.get("medication_profile_id"):
                raise ValueError("Declare medication_profile_id explicitly, using not_recorded if unknown")
            if supplied.get("subject_id") != calibration.get("subject_id"):
                raise ValueError("Feature subject_id must match the personal calibration subject_id")
            if supplied.get("medication_profile_id") != calibration.get("medication_profile_id"):
                raise ValueError("Medication context changed; review and update personal calibration")
            result = score_indices(supplied["values"], calibration)
            _write_json(result, args.out)
            print(json.dumps({name: value["score"] for name, value in result["indices"].items()}))
        elif args.command == "calibrate":
            import pandas as pd
            from .indices import fit_calibration
            _write_json(fit_calibration(pd.read_csv(args.input, dtype={"subject_id": str}), args.subject_id), args.out)
            print("Personal baseline statistics saved; incomplete coverage remains unavailable.")
        elif args.command == "train":
            from .ml import train_research_model
            result = train_research_model(args.input, args.out, args.features)
            print(json.dumps({k: result[k] for k in ("subjects", "rows", "balanced_accuracy", "macro_f1")}))
        elif args.command == "train-groups":
            from .ml import train_group_model
            result = train_group_model(args.input, args.out, args.features)
            print(json.dumps({k: result[k] for k in ("subjects_per_group", "epochs", "subject_balanced_accuracy", "epoch_balanced_accuracy")}))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0

