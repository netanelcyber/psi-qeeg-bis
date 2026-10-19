"""Four-channel FFT/Kalman demonstration: entirely synthetic, no clinical labels.

Run after ``pip install -e .``. Generates CSV/metadata/model files accepted by
the ordinary ``psi-qeeg track`` command, a trace, JSON and a diagnostic figure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from psi_qeeg.recording import read_recording
from psi_qeeg.tracking import track_recording


def create_fixture(out):
    """Write the reusable synthetic raw EEG/metadata/model fixture."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fs, seconds = 256, 180
    t = np.arange(fs * seconds) / fs
    rng = np.random.default_rng(42)
    modulation = np.clip((t - 100) / 50, 0, 1.5)
    channels = ["F3", "F4", "C3", "C4"]
    eeg = []
    for i, _ in enumerate(channels):
        components = []
        for lo, hi, amplitude in ((4, 8, 3), (8, 13, 9), (13, 30, 4)):
            raw = rng.normal(size=len(t) + fs * 5)
            filtered = signal.sosfilt(signal.butter(3, [lo, hi], fs=fs, btype="bandpass", output="sos"), raw)[fs * 5:]
            scale = amplitude / np.std(filtered)
            envelope = np.ones_like(t)
            if lo == 8:
                envelope = np.exp(-(.20 + .03*(i == 1)) * modulation)
            elif lo == 13:
                envelope = np.exp(.4 * modulation)
            components.append(filtered * scale * envelope)
        eeg.append(sum(components) + rng.normal(size=len(t))*2)
    frame = pd.DataFrame(np.array(eeg).T, columns=channels)
    frame.insert(0, "time_s", t)
    frame.to_csv(out/"recording.csv", index=False)
    metadata = {"subject_id": "synthetic-fft-demo", "medication_profile_id": "not_applicable",
                "sampling_rate_hz": fs, "eeg_unit": "uV", "eeg_columns": channels,
                "channel_locations": {ch: ch for ch in channels}, "reference_verified": True,
                "reference": "synthetic common reference; no physical recording",
                "source": "synthetic band-limited Gaussian components; seed 42",
                "recording_context": "synthetic_numerical_demonstration", "psychiatric_labels": None}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    model_path = Path(__file__).resolve().parents[1]/"examples/fft_tracking/model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    (out/"model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    return frame, metadata, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/fft_tracking_demo"))
    parser.add_argument("--bootstrap-draws", type=int, default=1024)
    args = parser.parse_args()
    frame, metadata, model = create_fixture(args.out)
    t = frame.time_s.to_numpy()
    modulation = np.clip((t - 100) / 50, 0, 1.5)
    recording = read_recording(args.out/"recording.csv", args.out/"metadata.json")
    trace, summary = track_recording(recording, model, 60, frontal_pair=("F3", "F4"),
                                     draws=args.bootstrap_draws, seed=5, high_z_limit=6)
    trace.to_csv(args.out/"tracking.csv", index=False)
    (args.out/"tracking.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(t, modulation, color="#6b7280")
    axes[0].axvspan(0, 60, alpha=.15, color="#15803d", label="Declared baseline")
    axes[0].set(ylabel="Input modulation", title="Synthetic FFT / Kalman demonstration — no clinical labels")
    axes[0].legend(loc="upper left")
    time = trace.window_end_s.to_numpy()
    mean, sd = trace.latent_state.to_numpy(), trace.state_sd.to_numpy()
    axes[1].plot(time, mean, label="Latent research state", color="#2563eb")
    axes[1].fill_between(time, mean - 1.96*sd, mean + 1.96*sd, alpha=.2, color="#2563eb",
                         label="Gaussian model interval (not clinical coverage)")
    axes[1].set(ylabel="State (model units)")
    axes[1].legend(loc="upper left")
    axes[2].plot(time, trace.cusum, color="#7c3aed", label="Innovation LLR CUSUM")
    axes[2].axhline(model["cusum_threshold"], linestyle="--", color="#9ca3af", label="Illustrative threshold")
    axes[2].set(ylabel="CUSUM")
    axes[2].legend(loc="upper left")
    r = np.array(summary["noise_calibration"]["R_fft_normalized"])
    color = axes[3].imshow(r, cmap="RdBu_r", vmin=-abs(r).max(), vmax=abs(r).max(), aspect="auto")
    labels = list(model["loadings"])
    axes[3].set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axes[3].set_yticks(range(len(labels)), labels)
    axes[3].set_title("Full FFT-derived measurement covariance R (normalized units)")
    for ax in axes[:3]:
        ax.set_xlabel("Session time (s); estimates available at window end")
    fig.colorbar(color, ax=axes[3], label="Covariance")
    fig.savefig(args.out/"tracking.png", dpi=140)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("use", "monitoring_windows", "measurement_updates", "change_flag_windows")}))


if __name__ == "__main__":
    main()
