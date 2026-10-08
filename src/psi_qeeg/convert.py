"""Convert an arbitrary EEG file to this project's CSV + sidecar, plus an uncalibrated depth proxy.

The commercial BIS algorithm is proprietary and is not reproduced. ``bis_proxy`` is a
transparent 0-100 descriptor of spectral slowing, for ranking epochs within one study only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROXY_NOTE = ("Uncalibrated research descriptor: 100*mean(clip(SEF95/30), spectral entropy, "
              "1-relative delta). Not the proprietary BIS index; not comparable across devices or subjects.")


def load_raw(path, channels=None):
    """Return (frame with time_s + EEG in uV, fs, channel names). EDF/BDF/EEGLAB .set/FIF via MNE (CSV already matches the project schema)."""
    path = Path(path)
    try:
        import mne
    except ImportError as exc:
        raise ValueError("Non-CSV input requires: pip install 'psi-qeeg-bis[edf]'") from exc
    readers = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
               ".set": mne.io.read_raw_eeglab, ".fif": mne.io.read_raw_fif}
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported EEG format: {path.suffix}")
    raw = reader(str(path), preload=True, verbose="ERROR")
    picks = channels or [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    missing = [c for c in picks if c not in raw.ch_names]
    if missing or not picks:
        raise ValueError(f"EEG channels not found in file: {missing or 'none typed eeg'}")
    frame = pd.DataFrame(raw.get_data(picks=picks).T * 1e6, columns=picks)  # MNE volts -> uV, once
    frame.insert(0, "time_s", raw.times)
    return frame, float(raw.info["sfreq"]), list(picks)


def bis_proxy(features: pd.DataFrame, channel: str) -> pd.Series:
    """Per-epoch proxy for accepted epochs; NaN elsewhere."""
    need = [f"{channel}__{k}" for k in ("sef95_hz", "spectral_entropy", "delta_relative_power")]
    if any(c not in features for c in need):
        return pd.Series(np.nan, index=features.index)
    sef, ent, delta = (features[c].astype(float) for c in need)
    parts = pd.concat([(sef / 30).clip(0, 1), ent.clip(0, 1), (1 - delta).clip(0, 1)], axis=1)
    return 100 * parts.mean(axis=1, skipna=False)


def convert(path, out_dir, subject_id, channels=None, context="awake_observational_research",
            reference="documented_common_reference", reference_verified=False, license_id="unspecified", source=""):
    frame, fs, names = load_raw(path, channels)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "recording.csv", index=False)
    meta = {"sampling_rate_hz": fs, "eeg_unit": "uV", "eeg_columns": names, "channel_locations": {},
            "reference": reference, "reference_verified": bool(reference_verified),
            "sensor": {"family": "research_eeg", "model": "not_reported"},
            "recording_context": context, "subject_id": subject_id, "psychiatric_labels": None,
            "source": source or str(path), "license": license_id,
            "changes": "Converted to uV CSV; no filtering, resampling or interpolation",
            "bis_proxy": PROXY_NOTE}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


GROUPS = {"ASD": "autism_spectrum", "PSY": "psychosis_spectrum", "HC": "healthy_control"}


def convert_cohort(manifest, out_dir, epoch_s=4.0):
    """Batch-convert a multi-group cohort and pool epoch features with a ``group`` column.

    ``manifest`` CSV columns: ``file, subject_id, group`` (ASD, PSY or HC), optional
    ``channels`` (space separated), ``context``, ``license``, ``source``. ``group`` is a
    diagnostic-group label from the dataset's own documentation. It is deliberately NOT written to
    ``research_label``: the trainer's state labels (baseline, pre_psychotic, ...) describe episodes,
    not diagnosis, and must come from independent annotation.
    """
    from .analysis import analyze
    from .recording import read_recording

    table = pd.read_csv(manifest, dtype=str).fillna("")
    for col in ("file", "subject_id", "group"):
        if col not in table:
            raise ValueError(f"Manifest needs column: {col}")
    bad = sorted(set(table.group) - set(GROUPS))
    if bad:
        raise ValueError(f"group must be one of {sorted(GROUPS)}; got {bad}")
    if table.subject_id.duplicated().any():
        raise ValueError("subject_id must be unique per manifest row")
    out = Path(out_dir)
    pooled = []
    for row in table.itertuples():
        sub = out / "subjects" / row.subject_id
        channels = row.channels.split() if getattr(row, "channels", "") else None
        convert(Path(manifest).parent / row.file, sub, row.subject_id, channels,
                getattr(row, "context", "") or "awake_observational_research",
                license_id=getattr(row, "license", "") or "unspecified", source=getattr(row, "source", ""))
        rec = read_recording(sub / "recording.csv", sub / "metadata.json")
        features, _ = analyze(rec, epoch_s)
        for ch in rec.channels:
            features[f"{ch}__bis_proxy_uncalibrated"] = bis_proxy(features, ch)
        features.insert(0, "group", GROUPS[row.group])
        features.insert(0, "subject_id", row.subject_id)
        pooled.append(features)
    result = pd.concat(pooled, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "cohort_features.csv", index=False)
    return result.groupby("group").subject_id.nunique().to_dict()
