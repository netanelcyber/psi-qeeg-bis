#!/usr/bin/env python3
"""Validate six-channel EEG feature extraction on a real public BIDS EEG recording.
This is signal-pipeline validation only: it does NOT validate Meltdown prediction.
"""
import json, os, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

OUT = Path("real_eeg_validation")
OUT.mkdir(exist_ok=True)

def main():
    eeg_path = os.environ.get("EEG_FILE")
    if not eeg_path or not Path(eeg_path).exists():
        raise SystemExit("EEG_FILE must point to a downloaded real EEG file")
    raw = mne.io.read_raw_bdf(eeg_path, preload=True, verbose="ERROR") if eeg_path.lower().endswith(".bdf") else mne.io.read_raw_edf(eeg_path, preload=True, verbose="ERROR")
    # Keep EEG channels only; exclude trigger, EOG, ECG, EMG and other auxiliaries.
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False, exclude=[])
    if len(eeg_picks) < 6:
        # Some BioSemi BDF files label EEG channels as misc; fall back to non-trigger data channels.
        bad_names = re.compile(r"(status|trigger|stim|eog|ecg|emg|resp|photo|sync)", re.I)
        candidates = [i for i, n in enumerate(raw.ch_names) if not bad_names.search(n)]
        eeg_picks = np.array(candidates, dtype=int)
    if len(eeg_picks) < 6:
        raise SystemExit(f"Only {len(eeg_picks)} possible EEG channels found: {raw.ch_names}")
    picks = eeg_picks[:6]
    data = raw.get_data(picks=picks) * 1e6  # V -> microvolts
    sfreq = float(raw.info["sfreq"])
    # Apply a conservative filter for feature extraction, not for clinical interpretation.
    filtered = mne.filter.filter_data(data, sfreq=sfreq, l_freq=1.0, h_freq=min(45.0, sfreq/2-1), verbose="ERROR")
    window = max(1, int(round(4 * sfreq)))
    rows = []
    bands = {"delta": (1,4), "theta": (4,8), "alpha": (8,13), "beta": (13,30), "gamma": (30,min(45, sfreq/2-1))}
    for wi, start in enumerate(range(0, filtered.shape[1]-window+1, window)):
        seg = filtered[:, start:start+window]
        row = {"window_index": wi, "window_start_seconds": start/sfreq, "window_end_seconds": (start+window)/sfreq}
        for ci, ch in enumerate([raw.ch_names[i] for i in picks]):
            x = seg[ci]
            row[f"eeg_{ch}_rms_uv"] = float(np.sqrt(np.mean(x*x)))
            row[f"eeg_{ch}_std_uv"] = float(np.std(x))
            row[f"eeg_{ch}_line_length"] = float(np.mean(np.abs(np.diff(x))))
            freqs, psd = welch(x, fs=sfreq, nperseg=min(len(x), int(2*sfreq)))
            total = np.trapezoid(psd[(freqs>=1)&(freqs<=min(45,sfreq/2-1))], freqs[(freqs>=1)&(freqs<=min(45,sfreq/2-1))])
            for band, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs < hi)
                power = np.trapezoid(psd[mask], freqs[mask]) if mask.sum() >= 2 else 0.0
                row[f"eeg_{ch}_{band}_relpower"] = float(power / total) if total > 0 else 0.0
        rows.append(row)
    features = pd.DataFrame(rows)
    features.to_csv(OUT / "real_eeg_features.csv", index=False)
    report = {
        "dataset": "OpenNeuro ds006780 (SFARI_EEG multi-paradigm)",
        "input_file": Path(eeg_path).name,
        "recording_format": Path(eeg_path).suffix.lower(),
        "n_channels_in_file": len(raw.ch_names),
        "selected_channels": [raw.ch_names[i] for i in picks],
        "sampling_rate_hz": sfreq,
        "duration_seconds": float(raw.n_times/sfreq),
        "window_seconds": 4,
        "n_feature_windows": len(features),
        "n_feature_columns": int(features.shape[1]-3),
        "all_features_finite": bool(np.isfinite(features.select_dtypes(include=[np.number]).to_numpy()).all()),
        "has_meltdown_event_labels": False,
        "forecast_validation": "NOT PERFORMED: this dataset has stimulus/task events, not annotated Meltdown onset timestamps",
        "scope": "Real EEG signal and feature-extraction validation only; not a Meltdown forecasting evaluation."
    }
    (OUT / "real_eeg_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if len(features) == 0 or not report["all_features_finite"]:
        raise SystemExit("Feature validation failed")

if __name__ == "__main__":
    main()
