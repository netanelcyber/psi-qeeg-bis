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


def _process_one(job):
    """Worker: convert + analyze one recording. Returns (subject_id, error or None). Resumable."""
    from .analysis import analyze
    from .recording import read_recording
    manifest_dir, sub, row, epoch_s, keep_raw = job
    try:
        done = sub / "features.csv"
        if done.exists():
            return row["subject_id"], None
        channels = row["channels"].split() if row.get("channels") else None
        convert(manifest_dir / row["file"], sub, row["subject_id"], channels,
                row.get("context") or "awake_observational_research",
                license_id=row.get("license") or "unspecified", source=row.get("source", ""))
        rec = read_recording(sub / "recording.csv", sub / "metadata.json")
        features, _ = analyze(rec, epoch_s)
        for ch in rec.channels:
            features[f"{ch}__bis_proxy_uncalibrated"] = bis_proxy(features, ch)
        if row.get("site"):
            features.insert(0, "site", row["site"])
        features.insert(0, "group", GROUPS[row["group"]])
        features.insert(0, "subject_id", row["subject_id"])
        features.to_csv(sub / "features.tmp", index=False)
        (sub / "features.tmp").replace(done)  # atomic marker: only finished subjects are skipped on resume
        if not keep_raw:
            (sub / "recording.csv").unlink(missing_ok=True)
        return row["subject_id"], None
    except Exception as exc:  # one bad scan must not abort thousands
        return row["subject_id"], f"{type(exc).__name__}: {exc}"


def build_manifest(root, out_csv):
    """Scan ``root/{ASD,PSY,HC}/**/*.{edf,bdf,set,fif}`` into a manifest; subject_id = '<group>_<relative path stem>'."""
    root = Path(root)
    rows = []
    for group in GROUPS:
        for f in sorted((root / group).rglob("*")):
            if f.suffix.lower() in {".edf", ".bdf", ".set", ".fif"}:
                rel = f.relative_to(root)
                rows.append({"file": str(rel), "subject_id": f"{group}_" + "_".join(rel.with_suffix("").parts[1:]), "group": group})
    if not rows:
        raise ValueError(f"No EEG files found under {root}/ASD, PSY or HC")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return len(rows)


def convert_cohort(manifest, out_dir, epoch_s=4.0, workers=1, keep_raw=False, max_failures=None):
    """Batch-convert thousands of recordings into pooled, group-labelled features.

    Parallel (``workers``), resumable (finished subjects are skipped), memory-bounded (features are
    streamed to disk, never concatenated in RAM) and fault tolerant (failures go to ``failures.csv``).
    Raw per-subject CSVs are deleted after analysis unless ``keep_raw``.

    ``manifest`` CSV columns: ``file, subject_id, group`` (ASD, PSY or HC), optional ``site``
    (device/site, used to detect group-site confounding), ``channels`` (space separated), ``context``,
    ``license``, ``source``. ``group`` is NOT written to ``research_label``: the trainer's state labels
    describe episodes, not diagnosis, and must come from independent annotation.
    """
    from concurrent.futures import ProcessPoolExecutor

    table = pd.read_csv(manifest, dtype=str).fillna("")
    for col in ("file", "subject_id", "group"):
        if col not in table:
            raise ValueError(f"Manifest needs column: {col}")
    bad = sorted(set(table.group) - set(GROUPS))
    if bad:
        raise ValueError(f"group must be one of {sorted(GROUPS)}; got {bad}")
    if table.subject_id.duplicated().any():
        raise ValueError("subject_id must be unique per manifest row")
    if table.subject_id.str.contains(r"[\\/]|^\.").any():
        raise ValueError("subject_id must be a plain name (no path separators)")
    out = Path(out_dir)
    manifest_dir = Path(manifest).resolve().parent
    jobs = []
    for row in table.to_dict("records"):
        sub = out / "subjects" / row["subject_id"]
        sub.mkdir(parents=True, exist_ok=True)
        jobs.append((manifest_dir, sub, row, epoch_s, keep_raw))
    failures = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = pool.map(_process_one, jobs, chunksize=4)
            for n, (sid, err) in enumerate(results, 1):
                if err:
                    failures.append({"subject_id": sid, "error": err})
                if max_failures is not None and len(failures) > max_failures:
                    raise ValueError(f"Aborted after {len(failures)} failures; last: {err}")
                if n % 100 == 0:
                    print(f"{n}/{len(jobs)} processed", flush=True)
    else:
        for n, job in enumerate(jobs, 1):
            sid, err = _process_one(job)
            if err:
                failures.append({"subject_id": sid, "error": err})
            if max_failures is not None and len(failures) > max_failures:
                raise ValueError(f"Aborted after {len(failures)} failures; last: {err}")
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} processed", flush=True)
    done = [j[1] / "features.csv" for j in jobs if (j[1] / "features.csv").exists()]
    if not done:
        raise ValueError("No recording converted successfully; see failures.csv")
    columns = []  # union header from files' first rows only
    for f in done:
        columns += [c for c in pd.read_csv(f, nrows=0).columns if c not in columns]
    counts = {}
    with open(out / "cohort_features.csv", "w", newline="", encoding="utf-8") as handle:
        first = True
        for f in done:
            part = pd.read_csv(f).reindex(columns=columns)
            part.to_csv(handle, header=first, index=False)
            first = False
            group = part.group.iat[0]
            counts[group] = counts.get(group, 0) + 1
    pd.DataFrame(failures, columns=["subject_id", "error"]).to_csv(out / "failures.csv", index=False)
    return {"subjects_per_group": counts, "converted": len(done), "failed": len(failures), "total": len(jobs)}
