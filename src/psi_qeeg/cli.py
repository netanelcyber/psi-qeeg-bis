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
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0

