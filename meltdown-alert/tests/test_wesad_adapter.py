"""WESAD-shaped SYNTHETIC fixtures, not a substitute for real-recording QA."""
from __future__ import annotations

import csv
import json
import pickle

import numpy as np
import pytest

from scripts.wesad_to_wearable import convert, load_subject


def fake_subject(seconds=90, *, missing=(), motion=0.0):
    t = np.arange(700 * seconds) / 700
    # Gaussian QRS peaks, with physiologic alternating R-R intervals.
    intervals = np.tile([.82, .86, .84, .88], seconds)
    peak_times = np.cumsum(intervals)
    peak_times = peak_times[peak_times < seconds]
    ecg = np.zeros_like(t)
    for center in peak_times:
        i = int(center * 700)
        idx = np.arange(max(i - 12, 0), min(i + 13, len(ecg)))
        ecg[idx] += np.exp(-0.5 * ((idx - i) / 3.5)**2)
    resp = np.sin(2 * np.pi * .25 * t)
    wrist_t = np.arange(32 * seconds) / 32
    acc = np.column_stack([np.full(wrist_t.size, 0.), np.full(wrist_t.size, 0.),
                           np.full(wrist_t.size, 64.) + motion * 64 * np.sin(2*np.pi * wrist_t)])
    subject = {"label": np.ones(700 * seconds, dtype=int), "signal": {
        "chest": {"ECG": ecg.reshape(-1, 1), "Resp": resp.reshape(-1, 1)},
        "wrist": {"ACC": acc, "TEMP": np.full((seconds*4, 1), 32.5)},
    }}
    for key in missing:
        for device in ("chest", "wrist"):
            subject["signal"][device].pop(key, None)
    return subject


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_export_quality_and_pr6_schema(tmp_path):
    source = tmp_path / "S2.pkl"
    with source.open("wb") as f:
        pickle.dump(fake_subject(), f)
    data = load_subject(source)
    csv_path, json_path = tmp_path / "measurements.csv", tmp_path / "qc.json"
    report = convert(data, csv_path, json_path, subject_id="S2")
    rows = read_csv(csv_path)
    assert report["total_windows"] == 4
    assert len(rows) == 20
    assert set(rows[0]) == {"timestamp", "metric", "value", "quality", "sensor"}
    assert set(x["timestamp"] for x in rows) == {"60", "70", "80", "90"}
    assert all(float(x["timestamp"]) >= 60 for x in rows)
    assert all(0 <= float(x["quality"]) <= 1 for x in rows)
    # Integration contract: output must be accepted by the actual PR #6 parser.
    from eeg_alert.wearable import load_measurements, WearableReplay, WearableProcessor
    observations = load_measurements(csv_path)
    assert len(observations) == len(rows)
    qc_frame = WearableProcessor(WearableReplay(observations), baseline_sec=80).update(90)
    assert qc_frame.metrics["heart_rate_bpm"]["status"] == "uncalibrated"
    hr = [r for r in rows if r["metric"] == "heart_rate_bpm"]
    assert any(50 <= float(r["value"]) <= 100 for r in hr if r["value"])
    hrv = [r for r in rows if r["metric"] == "hrv_rmssd_ms"]
    assert any(float(r["value"]) > 0 for r in hrv if r["value"])
    assert report["prediction_validation"] is False
    assert report["alerts_emitted"] is False
    assert json.loads(json_path.read_text())["total_windows"] == 4


def test_future_only_changes_do_not_change_earlier_export(tmp_path):
    a = fake_subject()
    b = fake_subject()
    b["signal"]["chest"]["ECG"][700*71:] *= 9
    b["signal"]["chest"]["Resp"][700*71:] = 0
    paths = []
    for name, subject in (("a", a), ("b", b)):
        out, qc = tmp_path / (name + ".csv"), tmp_path / (name + ".json")
        convert(subject, out, qc)
        paths.append(read_csv(out))
    assert [r for r in paths[0] if float(r["timestamp"]) <= 70] == [
        r for r in paths[1] if float(r["timestamp"]) <= 70]


def test_missing_sensor_and_ecg_artifact_explicit(tmp_path):
    subject = fake_subject(missing=("TEMP",))
    subject["signal"]["chest"]["ECG"][700*75] = np.nan
    out, qc = tmp_path / "a.csv", tmp_path / "a.json"
    report = convert(subject, out, qc)
    assert report["metrics"]["skin_temperature_c"]["emitted_windows"] == 0
    assert report["metrics"]["skin_temperature_c"]["reject_reasons"]["missing_sensor"] == 4
    assert report["metrics"]["heart_rate_bpm"]["reject_reasons"].get("missing_or_nonfinite_ecg",0) > 0
    bad = [r for r in read_csv(out) if r["metric"] == "heart_rate_bpm" and r["timestamp"] in ("80","90")]
    assert bad and all(r["value"] == "" for r in bad)


def test_duration_skew_rejected(tmp_path):
    subject = fake_subject()
    subject["signal"]["wrist"]["ACC"] = subject["signal"]["wrist"]["ACC"][:64]
    with pytest.raises(ValueError, match="duration differs"):
        convert(subject, tmp_path / "x.csv", tmp_path / "x.json")


def test_short_recording_reports_zero_coverage(tmp_path):
    subject = fake_subject(50)
    report = convert(subject, tmp_path / "x.csv", tmp_path / "x.json")
    assert report["total_windows"] == 0
    assert all(v["valid_fraction_of_expected"] == 0 for v in report["metrics"].values())


def test_invalid_parameters_rejected(tmp_path):
    subject = fake_subject(60)
    with pytest.raises(ValueError, match="step_sec"):
        convert(subject, tmp_path / "x.csv", tmp_path / "x.json", step_sec=0)
