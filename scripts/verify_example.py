"""Offline acceptance test on real VitalDB samples, not a clinical benchmark."""

import hashlib
import json
from pathlib import Path
import numpy as np

from psi_qeeg.recording import read_csv
from psi_qeeg.analysis import analyze

root = Path(__file__).resolve().parents[1]
example = root / "examples" / "vitaldb_case1"
metadata = json.loads((example / "metadata.json").read_text(encoding="utf-8"))
digest = hashlib.sha256((example / "recording.csv").read_bytes()).hexdigest()
if digest != metadata["recording_csv_sha256"]:
    raise SystemExit("Real EEG excerpt checksum mismatch")
recording = read_csv(example / "recording.csv", example / "metadata.json")
if recording.fs != 128 or len(recording.eeg) != 15360 or not np.isfinite(recording.eeg).all():
    raise SystemExit("Real EEG dimensions/clock/completeness mismatch")
features, summary = analyze(recording)
if summary["complete_epochs"] != 30 or summary["accepted_epochs"] != 30:
    raise SystemExit("Unexpected real-data epoch QC result")
if summary["SOI"] is not None or summary["RDI"] is not None or summary["classification"] != "indeterminate":
    raise SystemExit("Unsupported psychiatric result was emitted")
if any(c.startswith("pair_") for c in features):
    raise SystemExit("Unverified BIS channels were misrepresented as scalp connectivity")
if not np.isfinite(features.BIS_EEG1__alpha_power_uv2).all():
    raise SystemExit("Real-data spectral features are missing")
print(json.dumps({"real_data": "VitalDB case 1", "sha256": digest,
                  "samples_per_channel": len(recording.eeg), "sample_rate_hz": recording.fs,
                  "accepted_epochs": summary["accepted_epochs"], "classification": "indeterminate"}))

