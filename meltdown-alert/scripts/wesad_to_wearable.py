"""Convert one trusted WESAD subject pickle into PR #6 canonical wearable CSV.

Offline research/QC adapter. All computations use past and present samples only;
no EEG, events, notifications, diagnostic inference or clinical alerts are run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import pickle

import numpy as np
from scipy.signal import butter, find_peaks, lfilter, sosfilt

FS_CHEST = 700
FS_WRIST_TEMP = 4
FS_WRIST_ACC = 32
FS_WRIST_BVP = 64
METRICS = ("heart_rate_bpm", "hrv_rmssd_ms", "respiration_bpm",
           "skin_temperature_c", "movement_g")
SENSORS = {"heart_rate_bpm": "wesad.chest.ECG", "hrv_rmssd_ms": "wesad.chest.ECG",
           "respiration_bpm": "wesad.chest.Resp", "skin_temperature_c": "wesad.wrist.TEMP",
           "movement_g": "wesad.wrist.ACC"}
SIGNAL_KEYS = {"ECG": ("chest", "ECG", FS_CHEST),
               "Resp": ("chest", "Resp", FS_CHEST),
               "TEMP": ("wrist", "TEMP", FS_WRIST_TEMP),
               "ACC": ("wrist", "ACC", FS_WRIST_ACC)}


def load_subject(path: str | Path) -> dict:
    """Unpickle ONLY files from the original trusted WESAD download.

    Pickle is executable, so do not pass user-supplied or unknown pickle files.
    """
    with Path(path).open("rb") as fh:
        obj = pickle.load(fh, encoding="latin1")
    if not isinstance(obj, dict) or not isinstance(obj.get("signal"), dict):
        raise ValueError("Expected WESAD synchronized pickle with signal dictionary")
    return obj


def _array(subject: dict, name: str, duration: float) -> np.ndarray | None:
    device, key, rate = SIGNAL_KEYS[name]
    raw = subject["signal"].get(device, {}).get(key)
    if raw is None:
        return None
    data = np.asarray(raw, dtype=float)
    if name == "ACC":
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError("WESAD wrist ACC must have three axes")
    elif data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    elif data.ndim != 1:
        raise ValueError(f"WESAD {name} must be one-dimensional")
    if abs(len(data) / rate - duration) > 1.0:
        raise ValueError(f"WESAD {name} duration differs from synchronized labels by >1 second")
    return data


def _window(samples: np.ndarray | None, rate: int, end: float, seconds: float):
    if samples is None or end < seconds:
        return None
    start_idx, end_idx = int(round((end - seconds) * rate)), int(round(end * rate))
    if start_idx < 0 or end_idx > len(samples):
        return None
    return samples[start_idx:end_idx]


def _clean(window: np.ndarray | None, rate: int, seconds: float) -> bool:
    return (window is not None and len(window) >= round(rate * seconds) - 1
            and np.all(np.isfinite(window)))


def _ecg_features(window: np.ndarray | None, motion: float | None):
    """Causal ECG filter and past-window QRS-energy beat detection.

    RMSSD is calculated from *adjacent accepted* R-R intervals, never from HR,
    interpolated intervals or non-adjacent intervals across rejected beats.
    """
    if not _clean(window, FS_CHEST, 60):
        return {"heart_rate_bpm": (None, 0., "missing_or_nonfinite_ecg"),
                "hrv_rmssd_ms": (None, 0., "missing_or_nonfinite_ecg")}
    sos = butter(2, (5.0, 20.0), btype="bandpass", fs=FS_CHEST, output="sos")
    filtered = sosfilt(sos, window)
    derivative = np.diff(filtered, prepend=filtered[0])
    energy = lfilter(np.ones(round(0.12 * FS_CHEST)) / round(0.12 * FS_CHEST),
                     [1.], derivative * derivative)
    # Startup of the causal filter does not count as a candidate peak.
    first = 2 * FS_CHEST
    e = energy[first:]
    if not np.isfinite(e).all() or np.percentile(e, 95) < 1e-12:
        return {"heart_rate_bpm": (None, 0., "flat_ecg"),
                "hrv_rmssd_ms": (None, 0., "flat_ecg")}
    floor = float(np.median(e))
    threshold = floor + 0.30 * (float(np.percentile(e, 95)) - floor)
    peaks, _ = find_peaks(e, height=threshold,
                          distance=int(0.30 * FS_CHEST), prominence=max(threshold * .25, 1e-12))
    rr = np.diff(peaks) / FS_CHEST
    if len(rr) < 8:
        return {"heart_rate_bpm": (None, 0., "insufficient_r_peaks"),
                "hrv_rmssd_ms": (None, 0., "insufficient_r_peaks")}
    plausible = (rr >= 0.33) & (rr <= 2.0)
    median_rr = float(np.median(rr[plausible])) if plausible.any() else math.nan
    if not np.isfinite(median_rr):
        return {"heart_rate_bpm": (None, 0., "implausible_r_intervals"),
                "hrv_rmssd_ms": (None, 0., "implausible_r_intervals")}
    # Reject suspicious/ectopic intervals conservatively; no interpolation.
    good = plausible & (np.abs(rr - median_rr) <= 0.25 * median_rr)
    fraction = float(np.mean(good))
    motion_penalty = 0.35 if motion is not None and motion > 0.30 else 1.0
    result = {}
    if good.sum() >= 8:
        bpm = 60 / float(np.median(rr[good]))
        q = min(1.0, good.sum() / 25.0) * fraction * motion_penalty
        result["heart_rate_bpm"] = (bpm, q, "ok" if q >= .6 else "low_signal_quality")
    else:
        result["heart_rate_bpm"] = (None, 0., "insufficient_clean_r_intervals")
    # A rejected interval breaks adjacency; adjacent-valid mask retains only
    # consecutive RR differences. 60 s support has no future R-peaks.
    pair_mask = good[:-1] & good[1:]
    diffs = np.diff(rr)[pair_mask]
    if diffs.size >= 15:
        rmssd = float(np.sqrt(np.mean(diffs * diffs)) * 1000)
        q = min(1.0, diffs.size / 25.0) * fraction * motion_penalty
        result["hrv_rmssd_ms"] = (rmssd, q, "ok" if q >= .6 else "low_signal_quality")
    else:
        result["hrv_rmssd_ms"] = (None, 0., "insufficient_adjacent_r_intervals")
    return result


def _respiration(window: np.ndarray | None):
    if not _clean(window, FS_CHEST, 60):
        return None, 0., "missing_or_nonfinite_resp"
    # Low order, forward-only bandpass: no filtfilt, centered window or lookahead.
    sos = butter(2, (0.07, 0.70), btype="bandpass", fs=FS_CHEST, output="sos")
    filtered = sosfilt(sos, window)[10 * FS_CHEST:]
    amplitude = float(np.std(filtered))
    if amplitude < 1e-8:
        return None, 0., "flat_resp"
    peaks, _ = find_peaks(filtered, prominence=amplitude * 0.45,
                          distance=FS_CHEST)
    breaths = np.diff(peaks) / FS_CHEST
    valid = (breaths >= 1.0) & (breaths <= 12.0)
    if valid.sum() < 3:
        return None, 0., "insufficient_breath_cycles"
    quality = min(1.0, valid.sum() / 8.0) * float(np.mean(valid))
    rate = 60 / float(np.median(breaths[valid]))
    return rate, quality, "ok" if quality >= .6 else "low_signal_quality"


def _motion(window: np.ndarray | None):
    if not _clean(window, FS_WRIST_ACC, 5):
        return None, 0., "missing_or_nonfinite_acc"
    # Empatica E4 raw accelerometer counts: 64 counts = 1 g.
    g = window / 64.0
    if np.any(np.abs(g) >= 2):
        return None, 0., "saturated_acc"
    # Orientation/gravity removed using only preceding 5 s of observations.
    dynamic = g - np.median(g, axis=0)
    rms = float(np.sqrt(np.mean(np.sum(dynamic ** 2, axis=1))))
    return rms, 1., "ok"


def _temperature(window: np.ndarray | None):
    if not _clean(window, FS_WRIST_TEMP, 5):
        return None, 0., "missing_or_nonfinite_temp"
    temp = float(np.median(window))
    if not 10 <= temp <= 45:
        return None, 0., "invalid_temperature"
    return temp, 1., "ok"


def convert(subject: dict, output_csv: str | Path, output_report: str | Path,
            *, step_sec: int = 10, max_seconds: int | None = None,
            subject_id: str = "unknown") -> dict:
    """Write window-ending observations and an auditable per-metric QC report.

    Sample clock: original WESAD synchronized recording start = t=0 s.
    Non-finite/failed estimates become blank values (not zeros or carried data).
    """
    if step_sec < 1 or max_seconds is not None and max_seconds < 60:
        raise ValueError("step_sec must be >=1; max_seconds must be >=60")
    labels = np.asarray(subject.get("label", []))
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Expected nonempty 700 Hz WESAD synchronized labels")
    duration = len(labels) / FS_CHEST
    arrays = {key: _array(subject, key, duration) for key in SIGNAL_KEYS}
    end_at = min(duration, float(max_seconds)) if max_seconds is not None else duration
    ends = list(range(60, math.floor(end_at), step_sec))
    if ends and ends[-1] + step_sec <= end_at + 1e-9:
        ends.append(ends[-1] + step_sec)
    elif not ends and end_at >= 60:
        ends = [60]
    # No partial 60s support and no fractions of a window.
    counts = {metric: Counter() for metric in METRICS}
    accepted = {metric: 0 for metric in METRICS}
    written = {metric: 0 for metric in METRICS}
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "metric", "value", "quality", "sensor"])
        for t in ends:
            motion = _motion(_window(arrays["ACC"], FS_WRIST_ACC, t, 5))
            measured = {
                **_ecg_features(_window(arrays["ECG"], FS_CHEST, t, 60), motion[0]),
                "respiration_bpm": _respiration(_window(arrays["Resp"], FS_CHEST, t, 60)),
                "skin_temperature_c": _temperature(_window(arrays["TEMP"], FS_WRIST_TEMP, t, 5)),
                "movement_g": motion,
            }
            for metric in METRICS:
                if arrays[{"heart_rate_bpm": "ECG", "hrv_rmssd_ms": "ECG",
                           "respiration_bpm": "Resp", "skin_temperature_c": "TEMP",
                           "movement_g": "ACC"}[metric]] is None:
                    counts[metric]["missing_sensor"] += 1
                    continue
                value, quality, reason = measured[metric]
                # Gate estimates at the same 0.6 default QC cutoff as PR #6.
                valid = value is not None and np.isfinite(value) and quality >= .6
                if valid:
                    if metric == "hrv_rmssd_ms" and not 0 <= value <= 300:
                        valid, reason = False, "out_of_range_rmssd"
                    elif metric == "heart_rate_bpm" and not 20 <= value <= 250:
                        valid, reason = False, "out_of_range_hr"
                    elif metric == "respiration_bpm" and not 1 <= value <= 80:
                        valid, reason = False, "out_of_range_resp"
                    elif metric == "movement_g" and not 0 <= value <= 16:
                        valid, reason = False, "out_of_range_motion"
                writer.writerow([t, metric, f"{value:.8g}" if valid else "",
                                 f"{quality:.4f}", SENSORS[metric]])
                written[metric] += 1
                accepted[metric] += int(valid)
                counts[metric]["ok" if valid else reason] += 1
    report = {
        "dataset": "WESAD", "subject": subject_id,
        "recording_duration_sec": duration, "processed_through_sec": end_at,
        "clock": "synchronized WESAD recording start, t=0; no actual wall-clock timestamp",
        "sample_rates_hz": {"chest_ECG": 700, "chest_Resp": 700, "wrist_TEMP": 4,
                            "wrist_ACC": 32, "wrist_BVP_unused": 64},
        "method": {"window_sec": 60, "stride_sec": step_sec,
                   "ecg": "forward-only Butterworth and QRS-energy peaks; accepted adjacent RR for RMSSD",
                   "resp": "forward-only bandpass and breath peak intervals",
                   "movement": "5s wrist ACC dynamic RMS, counts/64 g, gravity/orientation median removed",
                   "timestamp_semantics": "right endpoint of past-only windows"},
        "total_windows": len(ends),
        "signals_present": {k: arrays[k] is not None for k in SIGNAL_KEYS},
        "metrics": {k: {"valid_windows": accepted[k], "emitted_windows": written[k],
                         "valid_fraction_of_expected": round(accepted[k] / len(ends), 5) if ends else 0.0,
                         "reject_reasons": dict(counts[k])} for k in METRICS},
        "labels": "WESAD protocol conditions, not ASD meltdown or psychosis events; not exported as events",
        "prediction_validation": False, "alerts_emitted": False,
        "data_origin": "Source WESAD subject (not included in this repository)",
        "license": "Scientific, non-commercial purposes only; attribution to Schmidt et al. (2018)",
    }
    Path(output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(output_report).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="WESAD raw sensor -> PSI canonical CSV + QC only (no alerts)")
    parser.add_argument("--subject-pkl", type=Path, required=True,
                        help="Original trusted WESAD S2/S2.pkl etc; never unpickle unknown files")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--qc-json", type=Path, required=True)
    parser.add_argument("--step-sec", type=int, default=10)
    parser.add_argument("--max-seconds", type=int, default=None)
    a = parser.parse_args()
    try:
        subject = load_subject(a.subject_pkl)
        report = convert(subject, a.output_csv, a.qc_json, step_sec=a.step_sec,
                         max_seconds=a.max_seconds, subject_id=a.subject_pkl.stem)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"subject": report["subject"], "windows": report["total_windows"],
                      "qc_json": str(a.qc_json), "csv": str(a.output_csv),
                      "alerts_emitted": False}, indent=2))


if __name__ == "__main__":
    main()
