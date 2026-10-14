"""Download and evaluate the public RepOD schizophrenia/control EEG cohort.

This is an exploratory diagnosis-group benchmark, not an episode-state or clinical validation.
The source dataset is CC0: https://doi.org/10.18150/repod.0107441
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from importlib.metadata import version as package_version
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-s', type=float, default=120,
                        help='Seconds from recording start; 0 analyzes the entire recording (default: 120)')
    parser.add_argument('--epoch-s', type=float, default=4, help='Non-overlapping epoch duration, at least four seconds')
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--source-metadata', type=Path, help='Use an explicitly retained RepOD API JSON snapshot')
    parser.add_argument('--discard-waveform-csv', action='store_true',
                        help='Analyze calibrated in-memory waveforms without writing large intermediate CSVs')
    args = parser.parse_args(argv)
    if not math.isfinite(args.window_s) or args.window_s < 0:
        parser.error('--window-s must be finite and nonnegative; 0 means full recording')
    if not math.isfinite(args.epoch_s) or args.epoch_s < 4:
        parser.error('--epoch-s must be finite and at least four seconds')
    return args


def select_window(frame, fs, window_s):
    if not math.isfinite(window_s) or window_s < 0:
        raise ValueError('Window must be finite and nonnegative')
    return frame.copy() if window_s == 0 else frame.iloc[:min(len(frame), round(fs * window_s))].copy()


def ensure_source(info, target):
    """Verify cached bytes or atomically replace a corrupt/incomplete source download."""
    checksum = info.get('checksum', {})
    expected = str(info.get('md5') or (checksum.get('value') if checksum.get('type') == 'MD5' else '')).lower()
    if len(expected) != 32 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('RepOD source must supply an MD5 checksum')
    size = info.get('filesize')

    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'md5').hexdigest() if hasattr(hashlib, 'file_digest') else _md5_stream(stream)

    def valid(path):
        return path.is_file() and (size is None or path.stat().st_size == size) and digest(path) == expected

    cached_valid = valid(target)
    restored = target.exists() and not cached_valid
    if not cached_valid:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.part')
        try:
            download_file(info['id'], temporary)
            if not valid(temporary):
                raise RuntimeError(f'Source checksum/size mismatch for {target.name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return expected, restored


def _md5_stream(stream):
    checksum = hashlib.md5()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        checksum.update(chunk)
    return checksum.hexdigest()


def export_waveform(frame, target_dir, discard):
    path = Path(target_dir) / 'recording.csv'
    if discard:
        path.unlink(missing_ok=True)
    else:
        frame.to_csv(path, index=False)


def main(argv=None):
    args = parse_args(argv)
    root, out = args.data_root, args.out
    payload = json.loads(args.source_metadata.read_text()) if args.source_metadata else get_json(API)
    if payload.get("status") != "OK":
        raise RuntimeError(f"RepOD API did not return OK: {payload.get('status')}")
    dataset_version = payload["data"]["latestVersion"]
    files = dataset_version["files"]
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
        target = root / group / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        digest, restored = ensure_source(info, target)
        rows.append({
            "file": f"{group}/{filename}", "subject_id": subject, "group": group,
            "site": "repod_eeg_schizophrenia_cohort",
            "context": "awake_observational_research", "license": "CC0-1.0",
            "source": f"https://doi.org/10.18150/repod.0107441; file={filename}",
        })
        downloaded.append({"file": filename, "bytes": target.stat().st_size, "md5": digest,
                           "source_file_id": file_id, "group": group, "subject_id": subject,
                           "restored_corrupt_cached_file": restored})
        print(f"Verified {filename}: {target.stat().st_size:,} bytes, MD5 {digest}", flush=True)

    pd.DataFrame(rows).to_csv(root / 'manifest.csv', index=False)
    root.mkdir(parents=True, exist_ok=True)
    (root / "provenance.json").write_text(json.dumps({
        "dataset": "EEG in schizophrenia", "persistent_id": PERSISTENT_ID,
        "citation": "Olejarczyk E, Jernajczyk W (2017), RepOD v1",
        "license": "CC0 1.0", "api_url": API, "files": downloaded,
        "dataset_version": {k: dataset_version.get(k) for k in ("versionNumber", "versionMinorNumber", "releaseTime")},
        "groups": {"HC": sum(x["group"] == "HC" for x in downloaded),
                   "PSY": sum(x["group"] == "PSY" for x in downloaded)},
        "note": "Diagnostic-group labels from dataset documentation; not episode/state labels.",
    }, indent=2) + "\n", encoding="utf-8")

    from psi_qeeg.convert import load_raw
    from psi_qeeg.recording import read_recording, Recording, validate
    from psi_qeeg.analysis import analyze

    # Default 120 seconds preserves the original experiment; full duration is an
    # explicit option. More epochs do not increase the number of independent people.
    subject_dirs = []
    subject_extent = []
    for row in rows:
        sid = row["subject_id"]
        target_dir = out / "subjects" / sid
        target_dir.mkdir(parents=True, exist_ok=True)
        frame, fs, channel_names = load_raw(root / row["file"])
        original_samples = len(frame)
        frame = select_window(frame, fs, args.window_s)
        if len(frame) < round(fs * args.epoch_s):
            raise RuntimeError(f"Recording too short for the requested epoch duration: {sid}")
        export_waveform(frame, target_dir, args.discard_waveform_csv)
        metadata = {
            "sampling_rate_hz": fs, "eeg_unit": "uV", "eeg_columns": channel_names,
            "channel_locations": {}, "reference": "documented_reference_between_Fz_and_Cz",
            "reference_verified": False, "sensor": {"family": "research_eeg", "model": "not_reported"},
            "recording_context": "awake_observational_research", "subject_id": sid,
            "psychiatric_labels": None, "source": row["source"], "license": "CC0-1.0",
            "changes": f"Window seconds={args.window_s} (0=full); converted to uV; no filtering, resampling or interpolation",
            "selection": {"window_s": args.window_s, "original_samples": original_samples,
                          "analyzed_samples": len(frame), "waveform_csv_exported": not args.discard_waveform_csv},
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        recording = (validate(Recording(frame, metadata)) if args.discard_waveform_csv else
                     read_recording(target_dir / "recording.csv", target_dir / "metadata.json"))
        part, summary = analyze(recording, epoch_s=args.epoch_s, compute_bicoherence=False)
        part.insert(0, 'epoch_id', np.arange(len(part)))
        part.insert(0, 'session_id', 'repod_resting_recording')
        part.insert(0, "recording_context", row["context"])
        part.insert(0, "site", row["site"])
        part.insert(0, "group", "psychosis_spectrum" if row["group"] == "PSY" else "healthy_control")
        part.insert(0, "subject_id", sid)
        part.to_csv(target_dir / "features.csv", index=False)
        subject_dirs.append(target_dir / "features.csv")
        subject_extent.append({"subject_id": sid, "recording_seconds": original_samples / fs,
                               "analyzed_seconds": len(frame) / fs, "complete_epochs": summary['complete_epochs'],
                               "accepted_epochs": summary['accepted_epochs'], "unused_tail_samples": summary['unused_tail_samples']})
        print(f"Processed {sid}: {summary['accepted_epochs']}/{summary['complete_epochs']} accepted epochs", flush=True)

    if len(subject_dirs) != 28:
        raise RuntimeError(f"Expected 28 converted subjects, got {len(subject_dirs)}")
    parts = [pd.read_csv(path) for path in subject_dirs]
    features = pd.concat(parts, ignore_index=True, sort=False)
    feature_file = out / "cohort_features.csv"
    out.mkdir(parents=True, exist_ok=True)
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
    from psi_qeeg.cohort import compare_electrode_coverage
    comparison = compare_electrode_coverage(feature_file, out, progress=lambda message: print(message, flush=True))
    result = comparison["comparisons"]["full_scalp"]
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
        "selection_protocol": {"window_s": args.window_s, "full_recording": args.window_s == 0,
                               "epoch_s": args.epoch_s, "overlap_s": 0,
                               "waveform_csv_exported": not args.discard_waveform_csv,
                               "subjects": subject_extent},
        "features_used": feature_columns,
        "evaluation": "leave-one-subject-out",
        "training_balance": comparison['protocol']['model'],
        "epoch_balanced_accuracy": result["epoch_balanced_accuracy"],
        "subject_balanced_accuracy": result["subject_balanced_accuracy"],
        "subject_confusion_matrix": result["subject_confusion_matrix"],
        "site_checked": True,
        "electrode_comparison": comparison["comparisons"],
        "cohort_features_sha256": comparison["cohort_features_sha256"],
        "software_versions": {name: package_version(name) for name in
                              ("psi-qeeg-bis", "numpy", "scipy", "pandas", "mne", "scikit-learn")},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "clinical_validation": "not_established",
        "event_prediction": {"status": "not_trained_missing_event_annotations",
                             "meltdown": None, "psychosis_onset": None,
                             "reason": "RepOD supplies diagnosis groups, not longitudinal event onsets or reviewed event-free follow-up"},
        "interpretation": "Exploratory schizophrenia-vs-healthy-control diagnosis-group benchmark only. It does not test autistic meltdown, psychosis onset, episode-state prediction, or clinical utility.",
        "limitations": [
            "Single public cohort/acquisition context; external-site generalization is not tested.",
            "Diagnosis labels are not momentary episode labels.",
            "No clinical or prospective validation.",
            "Epoch metrics are secondary; subject-level metrics are the more meaningful result.",
        ],
    }
    (out / "public_cohort_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
