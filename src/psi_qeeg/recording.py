"""Raw EEG input with mandatory units, sample clock and channel provenance."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Recording:
    frame: pd.DataFrame
    metadata: dict

    @property
    def fs(self) -> float:
        return float(self.metadata["sampling_rate_hz"])

    @property
    def channels(self) -> list[str]:
        return list(self.metadata["eeg_columns"])

    @property
    def eeg(self) -> np.ndarray:
        return self.frame[self.channels].to_numpy(dtype=float)

    @property
    def time(self) -> np.ndarray:
        return self.frame["time_s"].to_numpy(dtype=float)


def validate(recording: Recording) -> Recording:
    fs = recording.fs
    if not np.isfinite(fs) or fs < 2:
        raise ValueError("sampling_rate_hz must be a finite number >= 2")
    if not recording.channels or len(set(recording.channels)) != len(recording.channels):
        raise ValueError("eeg_columns must name unique raw EEG channels")
    if "time_s" not in recording.frame or len(recording.frame) < 2:
        raise ValueError("CSV needs a time_s column and at least two samples")
    if any(c not in recording.frame for c in recording.channels):
        raise ValueError("An EEG channel named in metadata is absent from the CSV")
    monitor_names = {"BIS", "SQI", "EMG", "SEF", "SR", "TOTPOW", "time_s"}
    monitor_names |= {f"BIS/{name}" for name in monitor_names}
    if set(recording.channels) & monitor_names:
        raise ValueError("Monitor numeric values and timestamps are not raw EEG channels")
    t = recording.time
    if not np.isfinite(t).all() or not np.allclose(np.diff(t), 1 / fs, rtol=1e-4, atol=1e-7):
        raise ValueError("EEG timestamps must be finite, increasing and uniformly sampled; preserve gaps as NaN")
    # Validate numeric representation without replacing or interpolating missing data.
    x = recording.eeg
    if np.isinf(x).any():
        raise ValueError("Infinite EEG amplitudes are invalid; use NaN for missing samples")
    return recording


def read_csv(path: str | Path, metadata_path: str | Path) -> Recording:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    frame = pd.read_csv(path)
    units = {"uV": 1.0, "µV": 1.0, "mV": 1000.0, "V": 1e6}
    if metadata.get("eeg_unit") not in units:
        raise ValueError("Metadata must explicitly specify eeg_unit as uV, mV or V")
    for c in metadata.get("eeg_columns", []):
        if c not in frame:
            raise ValueError(f"Missing raw EEG column: {c}")
        frame[c] = pd.to_numeric(frame[c], errors="raise") * units[metadata["eeg_unit"]]
    metadata["original_eeg_unit"] = metadata["eeg_unit"]
    metadata["eeg_unit"] = "uV"
    return validate(Recording(frame, metadata))


def read_edf(path: str | Path, metadata_path: str | Path) -> Recording:
    """MNE converts EDF physical units to volts; convert exactly once to uV."""
    try:
        import mne
    except ImportError as exc:
        raise ValueError("EDF/BDF input requires: pip install 'psi-qeeg-bis[edf]'") from exc
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    reader = mne.io.read_raw_bdf if Path(path).suffix.lower() == ".bdf" else mne.io.read_raw_edf
    raw = reader(str(path), preload=True, verbose="ERROR")
    names = metadata.get("eeg_columns")
    if not names or any(c not in raw.ch_names for c in names):
        raise ValueError("Sidecar eeg_columns must select actual EEG channels from the EDF/BDF")
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    if any(types[c] != "eeg" for c in names):
        raise ValueError("Selected EDF/BDF channels must be typed EEG, not ECG, EOG or monitor numerics")
    frame = pd.DataFrame(raw.get_data(picks=names).T * 1e6, columns=names)
    frame.insert(0, "time_s", raw.times)
    metadata.update(sampling_rate_hz=float(raw.info["sfreq"]), eeg_unit="uV", original_eeg_unit="MNE volts")
    # An EDF label alone never supplies a verified physical electrode pair.
    return validate(Recording(frame, metadata))


def read_recording(path: str | Path, metadata_path: str | Path) -> Recording:
    if Path(path).suffix.lower() in {".edf", ".bdf"}:
        return read_edf(path, metadata_path)
    return read_csv(path, metadata_path)

