#!/usr/bin/env python3
"""Reproducible engineering validation through the project's EEG pipeline.

No psychiatric model is trained or evaluated. Missing event-onset annotations
make meltdown/psychosis forecasting validation impossible on this recording.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from psi_qeeg import analysis, recording
from psi_qeeg.quality import QualityPolicy
from psi_qeeg.signal import BANDS, SPECTRAL_RANGE_HZ, band_integral

DEFAULT_MANIFEST = Path(__file__).with_name("ds006780_manifest.json")
EPOCH_S = 4.0
POLICY = QualityPolicy(amplitude_reference="epoch_mean")


def verify_file(path, expected):
    """Check both content identity and completeness before parsing EEG."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"Size/SHA-256 mismatch: {path.name}")
    return digest.hexdigest()


def download_url(manifest):
    source = manifest["download"]
    # One explicit object and immutable S3 version: no paginated key selection.
    return (f"https://s3.amazonaws.com/{quote(source['bucket'], safe='')}/"
            f"{quote(source['key'], safe='/')}?{urlencode({'versionId': source['version_id']})}")


def download_recording(manifest, path):
    path = Path(path)
    if path.exists():
        verify_file(path, manifest["download"])
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(download_url(manifest), timeout=60) as response, temporary.open("wb") as handle:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(block)
        verify_file(temporary, manifest["download"])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def synthetic_reference_checks():
    """Compare the production analyzer to analytical, explicitly synthetic inputs."""
    fs = 512
    t = np.arange(int(EPOCH_S * fs)) / fs
    signals = {f"tone_{band}": 10 * np.sin(2 * np.pi * hz * t)
               for band, hz in zip(BANDS, (2, 6, 10, 20, 35))}
    signals["mixture"] = 2 * np.sin(2 * np.pi * 2 * t) + 4 * np.sin(2 * np.pi * 10 * t)
    frame = pd.DataFrame({"time_s": t, **signals})
    rec = recording.validate(recording.Recording(frame, {
        "sampling_rate_hz": fs, "eeg_columns": list(signals),
        "eeg_unit": "uV", "recording_context": "synthetic_numerical_fixture",
    }))
    features, _ = analysis.analyze(rec, epoch_s=EPOCH_S)
    values = {}
    for band in BANDS:
        measured = float(features[f"tone_{band}__{band}_power_uv2"].iat[0])
        if not np.isclose(measured, 50.0, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Synthetic {band} tone power differs from A^2/2")
        values[band] = {"expected_power_uv2": 50.0, "measured_power_uv2": measured}
    ratio = float(features["mixture__delta_alpha_ratio"].iat[0])
    if not np.isclose(ratio, 0.25, atol=1e-10):
        raise ValueError("Synthetic mixture delta/alpha ratio mismatch")
    # Non-grid-aligned boundaries have an analytical integral for a flat PSD.
    f = np.arange(0, 65, 0.7)
    flat_power = sum(band_integral(f, np.ones_like(f), *bounds) for bounds in BANDS.values())
    if not np.isclose(flat_power, SPECTRAL_RANGE_HZ[1] - SPECTRAL_RANGE_HZ[0], atol=1e-12):
        raise ValueError("Exact band endpoints do not partition a flat PSD")
    return {"scope": "synthetic numerical fixtures; not patient evidence", "tones": values,
            "mixture_delta_alpha_ratio": ratio, "flat_psd_integral": flat_power}


def validate_features(features, summary, channels):
    """Check the required spectrum; preserve intentional missing/QC outputs."""
    errors = []
    accepted = features.loc[features.accepted]
    if accepted.empty:
        return ["No epochs passed production QC; spectral extraction was not exercised"]
    for channel in channels:
        required = [f"{channel}__{band}_{suffix}"
                    for band in BANDS for suffix in ("power_uv2", "relative_power")]
        required += [f"{channel}__{name}" for name in ("total_power_uv2", "sef95_hz", "spectral_entropy")]
        if any(name not in features for name in required):
            errors.append(f"{channel}: missing required production spectral columns")
            continue
        if not np.isfinite(accepted[required].to_numpy(float)).all():
            errors.append(f"{channel}: non-finite required spectrum")
        powers = accepted[[f"{channel}__{band}_power_uv2" for band in BANDS]].to_numpy(float)
        relative = accepted[[f"{channel}__{band}_relative_power" for band in BANDS]].to_numpy(float)
        total = accepted[f"{channel}__total_power_uv2"].to_numpy(float)
        if not ((powers >= 0).all() and (total > 0).all()
                and np.allclose(powers.sum(axis=1), total, rtol=1e-10, atol=1e-10)):
            errors.append(f"{channel}: absolute powers do not partition the total")
        if not ((relative >= 0).all() and (relative <= 1 + 1e-12).all()
                and np.allclose(relative.sum(axis=1), 1, rtol=1e-10, atol=1e-10)):
            errors.append(f"{channel}: relative powers do not partition unity")
        if not accepted[f"{channel}__sef95_hz"].between(*SPECTRAL_RANGE_HZ).all():
            errors.append(f"{channel}: SEF95 outside spectrum")
        if not accepted[f"{channel}__spectral_entropy"].between(0, 1 + 1e-12).all():
            errors.append(f"{channel}: entropy outside [0, 1]")
    if (summary["SOI"] is not None or summary["RDI"] is not None
            or summary["classification"] != "indeterminate"
            or features[["soi", "rdi"]].notna().any().any()
            or not features.classification.eq("indeterminate").all()):
        errors.append("Unsupported psychiatric inference")
    if any(name.startswith("pair_") for name in features):
        errors.append("Connectivity emitted despite unverified acquisition reference")
    return errors


def validate_file(eeg_path, manifest_path, out):
    manifest_path, out = Path(manifest_path), Path(out)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = verify_file(eeg_path, manifest["download"])
    for name, source in manifest.get("source_metadata", {}).items():
        sidecar = manifest_path.with_name(name)
        if hashlib.sha256(sidecar.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Source sidecar SHA-256 mismatch: {name}")
    # This is exactly the public ingestion/analysis path, not a parallel extractor.
    rec = recording.read_edf(eeg_path, manifest_path)
    if (rec.fs != manifest["sampling_rate_hz"]
            or len(rec.frame) != manifest["expected_samples_per_channel"]
            or rec.channels != manifest["eeg_columns"]):
        raise ValueError("Recording dimensions/clock/channels differ from the pinned fixture")
    features, summary = analysis.analyze(rec, epoch_s=EPOCH_S,
                                        line_hz=manifest["power_line_frequency_hz"], policy=POLICY)
    errors = validate_features(features, summary, rec.channels)
    samples_per_epoch = round(EPOCH_S * rec.fs)
    expected_epochs, tail = divmod(len(rec.frame), samples_per_epoch)
    if (len(features) != expected_epochs or summary["complete_epochs"] != expected_epochs
            or summary["unused_tail_samples"] != tail
            or summary["accepted_epochs"] != int(features.accepted.sum())
            or summary["rejected_epochs"] != int((~features.accepted).sum())):
        errors.append("Production epoch counts/tail accounting differ from the input sample clock")
    elif (not np.allclose(features.time_s, rec.time[::samples_per_epoch][:expected_epochs])
          or not np.allclose(features.epoch_end_s, features.time_s + EPOCH_S)):
        errors.append("Production epoch timestamps differ from the input sample clock")
    synthetic = synthetic_reference_checks()
    report = {
        "validation_status": "failed" if errors else "passed", "errors": errors,
        "scope": "Engineering validation of real EEG ingestion, QC and spectral extraction only",
        "dataset": manifest["source"], "dataset_version": manifest["dataset_version"],
        "dataset_git_commit": manifest["dataset_git_commit"], "input_sha256": digest,
        "input_size_bytes": Path(eeg_path).stat().st_size, "download": manifest["download"],
        "selected_channels": rec.channels, "sampling_rate_hz": rec.fs,
        "samples_per_channel": len(rec.frame), "duration_seconds": len(rec.frame) / rec.fs,
        "window_seconds": EPOCH_S, "n_feature_windows": len(features),
        "accepted_epochs": summary["accepted_epochs"], "rejected_epochs": summary["rejected_epochs"],
        "unused_tail_samples": summary["unused_tail_samples"],
        "quality_policy": summary["quality_policy"], "spectrum": summary["spectrum"],
        "notch_hz": summary["notch_hz"],
        "quality_warnings": sorted({warning for text in features.quality_warnings
                                    for warning in text.split(";") if warning}),
        "required_spectral_checks_passed": not errors,
        "missing_values": "Retained for SQI/EMG, unavailable history, undefined ratios and rejected epochs",
        "synthetic_comparisons": synthetic,
        "has_meltdown_event_labels": False, "has_psychosis_onset_labels": False,
        "forecast_validation": "NOT PERFORMED: no annotated behavioral/psychiatric event onsets",
        "model_training": "NOT PERFORMED", "clinical_validation": "not_established",
        "classification": summary["classification"], "SOI": summary["SOI"], "RDI": summary["RDI"],
        "limitations": ["One recording and six selected EEG channels; not a cohort evaluation",
                        "Acquisition reference unverified; no connectivity validation",
                        "EMG quality unknown; gamma power is not established as neural activity",
                        "No BIS-index equivalence, diagnostic accuracy or prediction lead time measured"],
        "environment": {"python": platform.python_version(),
                        **{name: version(name) for name in ("numpy", "scipy", "pandas", "mne", "psi-qeeg-bis")}},
        "implementation_sha256": {
            **{f"psi_qeeg/{name}.py": hashlib.sha256(Path(recording.__file__).with_name(name + ".py").read_bytes()).hexdigest()
               for name in ("recording", "analysis", "signal", "quality")},
            "validate_real_eeg.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "real_eeg_features.csv", index=False)
    for name, value in (("real_eeg_validation_report.json", report), ("pipeline_summary.json", summary)):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("; ".join(errors))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eeg-file", type=Path, default=Path(os.environ.get("EEG_FILE", "sample_eeg.bdf")))
    parser.add_argument("--out", type=Path, default=Path("real_eeg_validation"))
    parser.add_argument("--download", action="store_true", help="Fetch the pinned S3 version and verify SHA-256")
    args = parser.parse_args(argv)
    if args.download:
        download_recording(json.loads(args.manifest.read_text(encoding="utf-8")), args.eeg_file)
    report = validate_file(args.eeg_file, args.manifest, args.out)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
