"""Reproducible public BIS EEG excerpts with hashes and documented time alignment."""

from datetime import datetime, timezone
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://api.vitaldb.net/"
TRACKS = ("BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/BIS", "BIS/SQI", "BIS/EMG", "BIS/SEF", "BIS/SR", "BIS/TOTPOW")


def _fetch_csv(url: str, cache_path: Path) -> bytes:
    if cache_path.exists():
        content = cache_path.read_bytes()
    else:
        request = Request(url, headers={"User-Agent": "psi-qeeg-bis/0.1.0 research"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    return content


def waveform_times(frame: pd.DataFrame) -> np.ndarray:
    """VitalDB's compact waveform clock is metadata, not a sparse time series."""
    if len(frame) < 2 or "Time" not in frame:
        raise ValueError("Malformed VitalDB waveform")
    start, interval = float(frame.Time.iloc[0]), float(frame.Time.iloc[1])
    if not np.isfinite([start, interval]).all() or interval <= 0:
        raise ValueError("Invalid waveform start or sample interval")
    return start + np.arange(len(frame)) * interval


def hold_numeric(times: np.ndarray, track: pd.DataFrame, max_age_s: float = 5) -> np.ndarray:
    """Causal last observation, no backfill; stale monitor values become NaN."""
    if track.shape[1] != 2:
        raise ValueError("VitalDB track needs Time and exactly one value column")
    if len(track) == 0:
        return np.full(len(times), np.nan)
    tt = track.Time.to_numpy(dtype=float)
    if not np.isfinite(tt).all() or np.any(np.diff(tt) <= 0):
        raise ValueError("Numeric monitor timestamps must be finite and strictly increasing")
    positions = np.searchsorted(tt, times, side="right") - 1
    clipped = np.maximum(positions, 0)
    valid = (positions >= 0) & (times - tt[clipped] <= max_age_s)
    values = track.iloc[clipped, 1].to_numpy(dtype=float)
    return np.where(valid, values, np.nan)


def write_excerpt(caseid: int, tracks: dict, provenance: list, out: Path,
                  start_s: float, duration_s: float) -> dict:
    if not np.isfinite([start_s, duration_s]).all() or start_s < 0 or duration_s <= 0:
        raise ValueError("Excerpt start must be nonnegative; duration must be positive")
    for name in TRACKS[:2]:
        if name not in tracks:
            raise ValueError(f"Raw waveform required: {name}")
    t = waveform_times(tracks[TRACKS[0]])
    other = waveform_times(tracks[TRACKS[1]])
    if len(t) != len(other) or not np.allclose(t, other):
        raise ValueError("BIS waveform tracks must have identical sampling clocks")
    fs = 1 / (t[1] - t[0])
    if not np.isclose(fs, 128):
        raise ValueError("Unexpected VitalDB BIS sample rate; inspect track metadata")
    first, count = round(start_s * fs), round(duration_s * fs)
    if not np.isclose(first / fs, start_s) or not np.isclose(count / fs, duration_s):
        raise ValueError("Excerpt bounds must align to the raw sample grid")
    if first + count > len(t) or count < 2:
        raise ValueError("Requested excerpt lies outside the recorded track")
    excerpt_t = t[first:first + count]
    data = {"time_s": excerpt_t,
            "BIS_EEG1": tracks[TRACKS[0]].iloc[first:first + count, 1].to_numpy(),
            "BIS_EEG2": tracks[TRACKS[1]].iloc[first:first + count, 1].to_numpy()}
    for name in TRACKS[2:]:
        if name in tracks:
            data[name.split("/")[-1]] = hold_numeric(excerpt_t, tracks[name])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(out / "recording.csv", index=False, float_format="%.12g")
    digest = hashlib.sha256((out / "recording.csv").read_bytes()).hexdigest()
    metadata = {
        "sampling_rate_hz": fs, "eeg_unit": "uV", "eeg_columns": ["BIS_EEG1", "BIS_EEG2"],
        "channel_locations": {}, "reference": "not_reported", "reference_verified": False,
        "sensor": {"family": "BIS", "monitor": "BIS Vista", "model": "not_reported",
                   "side": "not_reported", "physical_channel_pairs": "not_reported"},
        "recording_context": "perioperative_anesthesia", "subject_id": f"vitaldb-case-{caseid}",
        "psychiatric_labels": None, "source": "VitalDB Open Dataset",
        "source_url": "https://vitaldb.net/dataset/", "caseid": caseid,
        "excerpt": {"start_s": start_s, "duration_s": duration_s, "samples": count},
        "numeric_alignment": {"method": "causal_last_observation", "max_age_s": 5,
                              "backfill": False, "waveform_interpolation": False},
        "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "citation": "Lee HC et al. VitalDB. Scientific Data 9, 279 (2022). doi:10.1038/s41597-022-01411-5",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "changes": "Excerpt selected; waveform clock reconstructed; numerics causally aligned; raw EEG amplitudes retained",
        "recording_csv_sha256": digest, "source_tracks": provenance,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def download(caseid: int, out: Path, start_s=332.0, duration_s=120.0, cache: Path | None = None) -> dict:
    if not 1 <= caseid <= 6388:
        raise ValueError("VitalDB caseid must lie in 1..6388")
    cache = cache or out / "source_cache"
    listing = pd.read_csv(BytesIO(_fetch_csv(BASE_URL + "trks", cache / "trks.csv.gz")))
    listing = listing[listing.caseid == caseid]
    tracks, provenance = {}, []
    for name in TRACKS:
        entries = listing[listing.tname == name]
        if len(entries) == 0:
            continue
        if len(entries) != 1:
            raise ValueError(f"Ambiguous track listing for {name}")
        tid = str(entries.iloc[0].tid)
        if not re.fullmatch(r"[0-9a-f]{40}", tid):
            raise ValueError("Unexpected track identifier")
        url = BASE_URL + tid
        content = _fetch_csv(url, cache / f"{tid}.csv.gz")
        tracks[name] = pd.read_csv(BytesIO(content))
        provenance.append({"tname": name, "tid": tid, "url": url,
                           "decompressed_csv_sha256": hashlib.sha256(content).hexdigest()})
    return write_excerpt(caseid, tracks, provenance, out, start_s, duration_s)

