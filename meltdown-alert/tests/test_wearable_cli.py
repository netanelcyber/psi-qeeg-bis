"""File/CLI integration coverage, using exclusively generated measurements."""
import csv
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from eeg_alert.__main__ import main


@pytest.fixture
def files(tmp_path):
    generator = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "generate_wearable_demo.py"))
    generator["generate"](tmp_path, seconds=45)
    return tmp_path


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["eeg_alert", *map(str, args)])
    main()


@pytest.mark.parametrize("model", ["index", "kalman"])
def test_file_replay_exports_strict_jsonl_with_qc_and_no_invented_event_labels(files, monkeypatch, model):
    output, report = files / "steps.jsonl", files / "report.json"
    invoke(monkeypatch, "--source", "csv", "--file", files / "eeg.csv", "--fs", 128,
           "--model", model,
           "--baseline-sec", 25, "--wearable-file", files / "cardiorespiratory.csv",
           "--wearable-file", files / "temperature_motion.csv",
           "--steps-jsonl", output, "--report", report)
    rows = [json.loads(line, parse_constant=lambda v: pytest.fail(f"non-finite JSON: {v}"))
            for line in output.read_text().splitlines()]
    assert len(rows) == 44
    assert set(rows[-1]["wearable"]["metrics"]) == {
        "heart_rate_bpm", "hrv_rmssd_ms", "respiration_bpm", "skin_temperature_c", "movement_g"}
    assert rows[-1]["wearable"]["metrics"]["skin_temperature_c"]["status"] == "uncalibrated"
    assert rows[-1]["provenance"]["clinical_validation"] is False
    assert rows[-1]["provenance"]["eeg_model"] == model
    assert rows[-1]["provenance"]["event_labels"] == "none"
    assert rows[-1]["wearable"]["fusion_weight"] == 0.25
    assert np.isfinite(rows[-1]["raw_index"])
    assert json.loads(report.read_text())["evaluation"] is None


def test_absolute_clocks_are_not_rebased_independently(tmp_path, monkeypatch):
    # EEG begins 10 s after the shared origin. Wearable begins at 0, so it must
    # already be stale at the first EEG window, rather than shifted to 10 s.
    fs = 128
    times = np.arange(3 * fs) / fs
    x = 10 * np.sin(2 * np.pi * 10 * times)
    eeg_path = tmp_path / "eeg.csv"
    with eeg_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "TP9", "AF7", "AF8", "TP10"])
        for t, value in zip(times, x):
            writer.writerow([f"2026-10-19T09:00:{10 + t:09.6f}Z", value, value, value, value])
    wearable = tmp_path / "wearable.csv"
    wearable.write_text("timestamp,heart_rate_bpm\n2026-10-19T09:00:00Z,70\n")
    output = tmp_path / "steps.jsonl"
    invoke(monkeypatch, "--source", "csv", "--file", eeg_path, "--fs", fs,
           "--wearable-file", wearable, "--session-start", "2026-10-19T12:00:00+03:00",
           "--wearable-max-age", "heart_rate_bpm=5", "--steps-jsonl", output)
    first = json.loads(output.read_text().splitlines()[0])
    assert first["t"] == pytest.approx(12)
    assert first["wearable"]["metrics"]["heart_rate_bpm"]["status"] == "stale"


@pytest.mark.parametrize("args,expected", [
    (["--source", "lsl"], "live LSL clock bridging is not implemented"),
    (["--wearable-fusion-weight", "nan"], "fusion_weight"),
    (["--wearable-max-age", "heart_rate_bpm=0"], "maximum ages"),
    (["--wearable-weight", "unknown=1"], "Unknown wearable metric"),
    (["--wearable-weight", "heart_rate_bpm=inf"], "weights"),
    (["--wearable-offset-sec", "inf"], "offset"),
    (["--session-start", "2026-10-19T09:00:00"], "timezone"),
])
def test_invalid_fusion_cli_fails_before_hardware_or_notifications(tmp_path, monkeypatch, capsys, args, expected):
    path = tmp_path / "wearable.csv"
    path.write_text("timestamp,heart_rate_bpm\n0,70\n")
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, "--wearable-file", path, *args)
    assert exc.value.code == 2
    assert expected in capsys.readouterr().err


def test_eeg_only_csv_without_timestamps_remains_supported(files, monkeypatch):
    source = files / "eeg.csv"
    lines = source.read_text().splitlines()
    source.write_text("\n".join(line.split(",", 1)[1] for line in lines) + "\n")
    output = files / "eeg-only.jsonl"
    invoke(monkeypatch, "--source", "csv", "--file", source, "--fs", 128, "--steps-jsonl", output)
    assert all(json.loads(line)["wearable"] is None for line in output.read_text().splitlines())


def test_fused_csv_without_eeg_timestamps_is_rejected(files, monkeypatch, capsys):
    source = files / "eeg.csv"
    lines = source.read_text().splitlines()
    source.write_text("\n".join(line.split(",", 1)[1] for line in lines) + "\n")
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, "--source", "csv", "--file", source, "--fs", 128,
               "--wearable-file", files / "cardiorespiratory.csv")
    assert exc.value.code == 2
    assert "EEG timestamp" in capsys.readouterr().err
