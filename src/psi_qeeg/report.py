"""Offline report with measured EEG/DSA; index availability remains explicit."""

import base64
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def write_report(recording, features, summary: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "features.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
        x, t, fs = recording.eeg, recording.time, recording.fs
        short = t < t[0] + 12
        for i, name in enumerate(recording.channels):
            # Offset traces only for readability; label the offset explicitly.
            axes[0].plot(t[short] - t[0], x[short, i] + i * 100, lw=0.8, label=f"{name} (+{i * 100} uV display offset)")
        axes[0].set(xlabel="Seconds from excerpt start", ylabel="EEG (uV; offset traces)", title="Measured raw EEG: first 12 seconds")
        axes[0].legend(loc="upper right", fontsize=8)
        clean_epochs = features[features.accepted].index.to_numpy()
        if len(clean_epochs):
            n = round(summary["epoch_s"] * fs)
            clean = np.concatenate([x[i*n:(i+1)*n, 0] for i in clean_epochs])
            # Compute each accepted epoch separately: never create false transitions at rejected gaps.
            spectra = [signal.welch(x[i*n:(i+1)*n, 0], fs, nperseg=round(2*fs), scaling="density") for i in clean_epochs]
            f = spectra[0][0]
            mean_psd = np.mean([p for _, p in spectra], axis=0)
            band = (f >= 0.5) & (f <= 40)
            axes[1].semilogy(f[band], mean_psd[band], color="#2563eb")
            del clean
        axes[1].set(xlabel="Frequency (Hz)", ylabel="Power density (uV²/Hz)", title=f"{recording.channels[0]}: mean accepted-epoch Welch PSD (unnotched)")
        n = round(summary["epoch_s"] * fs)
        dsa, freq = [], None
        for i, row in features.iterrows():
            if row.accepted:
                freq, p = signal.welch(x[i*n:(i+1)*n, 0], fs, nperseg=round(2*fs), scaling="density")
                dsa.append(10 * np.log10(np.maximum(p, 1e-12)))
            else:
                freq = np.fft.rfftfreq(round(2*fs), 1/fs)
                dsa.append(np.full(len(freq), np.nan))
        band = (freq >= 0.5) & (freq <= 40)
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad("#d1d5db")
        epoch_edges = np.r_[features.time_s.to_numpy(), features.epoch_end_s.iloc[-1]] - t[0]
        ff = freq[band]
        freq_edges = np.r_[ff - (ff[1]-ff[0])/2, ff[-1]+(ff[1]-ff[0])/2]
        im = axes[2].pcolormesh(epoch_edges, freq_edges, np.asarray(dsa)[:, band].T, cmap=cmap, shading="flat")
        axes[2].set(xlabel="Seconds from excerpt start", ylabel="Frequency (Hz)", title="Density spectral array; rejected epochs are gray (unnotched)")
        fig.colorbar(im, ax=axes[2], label="10 log₁₀ PSD [uV²/Hz]")
        for column, label in (("BIS", "Recorded device BIS"), ("SQI", "Recorded signal quality index")):
            if column in recording.frame:
                axes[3].plot(t - t[0], recording.frame[column], label=label)
        axes[3].set(xlabel="Seconds from excerpt start", ylabel="Recorded monitor value", ylim=(0, 105), title="Monitor numerics: separate from EEG features and PSI")
        if axes[3].lines:
            axes[3].legend(fontsize=8)
        fig.suptitle("PSI qEEG/BIS · real-data engineering report · research only", fontsize=15)
        fig.savefig(out / "qeeg.png", dpi=150)
        plt.close(fig)
    image_data = base64.b64encode((out / "qeeg.png").read_bytes()).decode("ascii")
    source = escape(str(recording.metadata.get("source", "User-supplied recording")))
    context = escape(summary["recording_context"])
    sensor = escape(str(recording.metadata.get("sensor", {}).get("model", "not_reported")))
    document = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSI qEEG/BIS research report</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#182238;background:#f4f6fb;margin:0}}main{{max-width:1100px;margin:auto;padding:40px 24px}}h1{{line-height:1.15}}.card{{background:white;border:1px solid #dce2ed;border-radius:14px;padding:24px;margin:20px 0}}.tag{{color:#394fb5;font-weight:700}}.stats{{display:flex;gap:38px;flex-wrap:wrap}}.stats strong{{display:block;font-size:32px}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}summary{{cursor:pointer;font-weight:700}}a{{color:#2648af}}</style>
<main><div class="tag">PSI • qEEG / BIS WORKBENCH</div><h1>Measured EEG.<br>Documented electrode coverage.</h1>
<p>Source: {source} · Context: {context} · {recording.fs:g} Hz · {len(recording.channels)} EEG channels</p>
<div class="card"><div class="stats"><div><strong>{summary["accepted_epochs"]}/{summary["complete_epochs"]}</strong>accepted epochs</div><div><strong>Unavailable</strong>SOI / RDI composites</div><div><strong>Indeterminate</strong>psychiatric state</div></div>
<p>{escape(summary["index_reason"])}</p><p>This report describes signal processing. It does not establish psychiatric diagnosis or treatment decisions.</p></div>
<div class="card"><h2>Signal and spectral features</h2><img alt="Raw EEG, power spectrum, density spectral array and recorded monitor values" src="data:image/png;base64,{image_data}"></div>
<div class="card"><h2>Sensor evidence</h2><p>Sensor model: {sensor}. Physical channel pairs and sensor side are retained from the source metadata; missing information remains unknown. Two BIS waveform exports do not establish two hemispheres or posterior coverage.</p>
<p>EMG can contaminate frontal high-frequency EEG. BIS EMG numerics are recorded device values; the 128 Hz export cannot independently measure the monitor's higher-frequency EMG band.</p></div>
<div class="card"><details><summary>Analysis settings and source provenance</summary><pre>{escape(json.dumps(summary, indent=2, allow_nan=False))}</pre></details></div></main></html>'''
    (out / "report.html").write_text(document, encoding="utf-8")

