"""Generate SYNTHETIC EEG and asynchronous, imperfect wearable CSVs.

Run from meltdown-alert: python examples/generate_wearable_demo.py /tmp/wearable-demo
This exercises software only; no human data or fitted physiological model is used.
"""
from __future__ import annotations

import argparse
import csv
from datetime import timedelta
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_alert.config import Config  # noqa: E402
from eeg_alert.sources import SyntheticEEG  # noqa: E402
from eeg_alert.timing import _absolute_time  # noqa: E402


def generate(directory: Path, seconds: float = 360.0, fs: float = 128.0,
             seed: int = 7, session_start: str | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    origin = _absolute_time(session_start) if session_start else None

    def stamp(t):
        return (origin + timedelta(seconds=float(t))).isoformat() if origin else f"{t:.8f}"

    cfg = Config(fs=fs)
    src = SyntheticEEG(cfg, minutes=seconds / 60, events_min=(5.0,), ramp_min=1.5,
                       distractors_min=(), seed=seed)
    times, eeg = src.generate()
    with (directory / "eeg.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", *cfg.channels])
        writer.writerows([stamp(t), *[f"{v:.6f}" for v in row]] for t, row in zip(times, eeg.T))

    rng = np.random.default_rng(seed)
    specs = {
        "cardiorespiratory.csv": [("heart_rate_bpm", 1, 70, 25, 0.8),
                                   ("hrv_rmssd_ms", 5, 45, -18, 1.0),
                                   ("respiration_bpm", 5, 14, 8, 0.3)],
        "temperature_motion.csv": [("skin_temperature_c", 15, 33, 0.3, 0.03),
                                    ("movement_g", 0.5, 0.05, 0.2, 0.005)],
    }
    for filename, sensors in specs.items():
        rows = []
        for metric, period, base, change, noise in sensors:
            offset = 0.25 if metric == "movement_g" else 0.0
            for t in np.arange(offset, seconds, period):
                if metric == "heart_rate_bpm" and 250 <= t < 280:
                    continue  # real elapsed time is retained through this dropout
                arousal = float(np.clip((t - 180) / 90, 0, 1))
                value = base + change * arousal + rng.normal(0, noise)
                quality = 0.1 if metric == "respiration_bpm" and 210 <= t < 230 else 0.95
                rows.append((t, metric, value, quality))
        with (directory / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "metric", "value", "quality", "sensor"])
            for t, metric, value, quality in sorted(rows):
                writer.writerow([stamp(t), metric, f"{value:.6f}", quality, filename.removesuffix(".csv")])
    print(f"Synthetic files saved to {directory}. Engineering fixtures, not clinical evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seconds", type=float, default=360.0)
    parser.add_argument("--fs", type=float, default=128.0)
    parser.add_argument("--session-start", help="optional timezone-aware ISO origin")
    args = parser.parse_args()
    generate(args.directory, args.seconds, args.fs, session_start=args.session_start)
