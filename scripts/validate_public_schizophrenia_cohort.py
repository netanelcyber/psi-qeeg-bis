"""Download and evaluate the public RepOD schizophrenia/control EEG cohort.

This is an exploratory diagnosis-group benchmark, not an episode-state or clinical validation.
The source dataset is CC0: https://doi.org/10.18150/repod.0107441
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path("data/public_schizophrenia")
MANIFEST = ROOT / "manifest.csv"
OUT = Path("outputs/repod_schizophrenia")
PERSISTENT_ID = "doi:10.18150/repod.0107441"
API = "https://repod.icm.edu.pl/api/datasets/:persistentId/?" + urlencode({"persistentId": PERSISTENT_ID})
EXPECTED = {*(f"h{i:02d}.edf" for i in range(1, 15)), *(f"s{i:02d}.edf" for i in range(1, 15))}


def get_json(url):
    req = Request(url, headers={"User-Agent": "psi-qeeg-bis-research-validation/1.0"})
    with urlopen(req, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(file_id, target):
    req = Request(f"https://repod.icm.edu.pl/api/access/datafile/{file_id}",
                  headers={"User-Agent": "psi-qeeg-bis-research-validation/1.0"})
    with urlopen(req, timeout=180) as response, target.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def main():
    payload = get_json(API)
    if payload.get("status") != "OK":
        raise RuntimeError(f"RepOD API did not return OK: {payload.get('status')}")
    version = payload["data"]["latestVersion"]
    files = version["files"]
    by_name = {str(item.get("label", item.get("dataFile", {}).get("filename", ""))).lower(): item
               for item in files}
    missing = sorted(EXPECTED - set(by_name))
    if missing:
        raise RuntimeError(f"Expected 28 EDFs missing from RepOD API: {missing}; got {sorted(by_name)}")

    rows = []
    downloaded = []
    for filename in sorted(EXPECTED):
        item = by_name[filename]
        info = item.get("dataFile", {})
        file_id = info.get("id")
        if file_id is None:
            raise RuntimeError(f"No dataFile.id for {filename}: {item}")
        group = "HC" if filename.startswith("h") else "PSY"
        subject = f"{group}_{Path(filename).stem.upper()}"
        target = ROOT / group / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size == 0:
            download_file(file_id, target)
        digest = hashlib.md5(target.read_bytes()).hexdigest()
        expected_md5 = str(info.get("md5") or "").lower()
        if expected_md5 and digest != expected_md5:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"MD5 mismatch for {filename}: {digest} != {expected_md5}")
        rows.append({
            "file": f"{group}/{filename}", "subject_id": subject, "group": group,
            "site": "repod_eeg_schizophrenia_cohort",
            "context": "awake_observational_research", "license": "CC0-1.0",
            "source": f"https://doi.org/10.18150/repod.0107441; file={filename}",
        })
        downloaded.append({"file": filename, "bytes": target.stat().st_size, "md5": digest,
                           "source_file_id": file_id, "group": group, "subject_id": subject})
        print(f"Verified {filename}: {target.stat().st_size:,} bytes, MD5 {digest}", flush=True)

    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "provenance.json").write_text(json.dumps({
        "dataset": "EEG in schizophrenia", "persistent_id": PERSISTENT_ID,
        "citation": "Olejarczyk E, Jernajczyk W (2017), RepOD v1",
        "license": "CC0 1.0", "api_url": API, "files": downloaded,
        "groups": {"HC": sum(x["group"] == "HC" for x in downloaded),
                   "PSY": sum(x["group"] == "PSY" for x in downloaded)},
        "note": "Diagnostic-group labels from dataset documentation; not episode/state labels.",
    }, indent=2) + "\n", encoding="utf-8")

    from psi_qeeg.convert import load_raw
    from psi_qeeg.recording import read_recording
    from psi_qeeg.analysis import analyze

    # Keep a fixed 120-second window per participant and skip the optional, expensive
    # 60-second bicoherence estimator for this first reproducibility benchmark.
    # Spectral features and epoch QC are the tested project pipeline.
    subject_dirs = []
    for row in rows:
        sid = row["subject_id"]
        target_dir = OUT / "subjects" / sid
        target_dir.mkdir(parents=True, exist_ok=True)
        frame, fs, channel_names = load_raw(ROOT / row["file"])
        frame = frame.iloc[:min(len(frame), round(fs * 120))].copy()
        if len(frame) < round(fs * 4):
            raise RuntimeError(f"Recording too short for four-second epochs: {sid}")
        frame.to_csv(target_dir / "recording.csv", index=False)
        metadata = {
            "sampling_rate_hz": fs, "eeg_unit": "uV", "eeg_columns": channel_names,
            "channel_locations": {}, "reference": "documented_reference_between_Fz_and_Cz",
            "reference_verified": False, "sensor": {"family": "research_eeg", "model": "not_reported"},
            "recording_context": "awake_observational_research", "subject_id": sid,
            "psychiatric_labels": None, "source": row["source"], "license": "CC0-1.0",
            "changes": "First 120 seconds only; converted to uV; no filtering, resampling or interpolation",
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        recording = read_recording(target_dir / "recording.csv", target_dir / "metadata.json")
        part, summary = analyze(recording, compute_bicoherence=False)
        part.insert(0, "site", row["site"])
        part.insert(0, "group", "psychosis_spectrum" if row["group"] == "PSY" else "healthy_control")
        part.insert(0, "subject_id", sid)
        part.to_csv(target_dir / "features.csv", index=False)
        subject_dirs.append(target_dir / "features.csv")
        print(f"Processed {sid}: {summary['accepted_epochs']}/{summary['complete_epochs']} accepted epochs", flush=True)

    if len(subject_dirs) != 28:
        raise RuntimeError(f"Expected 28 converted subjects, got {len(subject_dirs)}")
    parts = [pd.read_csv(path) for path in subject_dirs]
    features = pd.concat(parts, ignore_index=True, sort=False)
    feature_file = OUT / "cohort_features.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_file, index=False)

    common = None
    for path in subject_dirs:
        cols = set(pd.read_csv(path, nrows=0).columns)
        common = cols if common is None else common & cols
    feature_columns = sorted(c for c in (common or set())
                             if c.endswith(("__alpha_power_uv2", "__delta_alpha_ratio",
                                            "__sef95_hz", "__spectral_entropy")))
    if len(feature_columns) < 4:
        raise RuntimeError(f"Too few common, comparable EEG features for LOSO evaluation: {feature_columns}")
    from psi_qeeg.ml import train_group_model
    result = train_group_model(feature_file, OUT, feature_columns)
    accepted = features[features["accepted"].astype(str).str.lower() == "true"]
    report = {
        "status": "completed",
        "dataset": "EEG in schizophrenia (RepOD, CC0)",
        "source": "https://doi.org/10.18150/repod.0107441",
        "downloaded_files": len(downloaded),
        "downloaded_bytes": sum(x["bytes"] for x in downloaded),
        "subjects_per_group": result["subjects_per_group"],
        "converted_subjects": len(subject_dirs),
        "failed_subjects": 0,
        "feature_rows": int(len(features)),
        "accepted_epochs": int(len(accepted)),
        "features_used": feature_columns,
        "evaluation": "leave-one-subject-out",
        "epoch_balanced_accuracy": result["epoch_balanced_accuracy"],
        "subject_balanced_accuracy": result["subject_balanced_accuracy"],
        "subject_confusion_matrix": result["subject_confusion_matrix"],
        "site_checked": result["site_checked"],
        "clinical_validation": "not_established",
        "interpretation": "Exploratory schizophrenia-vs-healthy-control diagnosis-group benchmark only. It does not test autistic meltdown, psychosis onset, episode-state prediction, or clinical utility.",
        "limitations": [
            "Single public cohort/acquisition context; external-site generalization is not tested.",
            "Diagnosis labels are not momentary episode labels.",
            "No clinical or prospective validation.",
            "Epoch metrics are secondary; subject-level metrics are the more meaningful result.",
        ],
    }
    (OUT / "public_cohort_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
