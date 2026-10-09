"""Epoch features, montage awareness and explicit indeterminate PSI output."""

from dataclasses import asdict
import numpy as np
import pandas as pd

from .quality import QualityPolicy, assess_epoch
from .signal import spectral_features, squared_bicoherence, phase_locking_value
from .indices import SOI_DIRECTIONS, RDI_COMPONENTS


def analyze(recording, epoch_s=4.0, line_hz=None, policy: QualityPolicy | None = None, compute_bicoherence=True):
    policy = policy or QualityPolicy()
    if recording.fs / 2 <= 40:
        raise ValueError("The full 0.5-40 Hz feature set needs a sampling rate above 80 Hz")
    if not np.isfinite(epoch_s) or epoch_s < 4:
        raise ValueError("Epochs must be at least four seconds")
    n = round(epoch_s * recording.fs)
    if not np.isclose(n / recording.fs, epoch_s):
        raise ValueError("Epoch duration must align to the sample grid")
    if len(recording.frame) < n:
        raise ValueError("Recording is shorter than one complete epoch")
    is_bis = recording.metadata.get("sensor", {}).get("family") == "BIS"
    positions = recording.metadata.get("channel_locations", {})
    site_to_channel = {site: name for name, site in positions.items() if name in recording.channels}
    reference_verified = recording.metadata.get("reference_verified") is True
    rows, quality = [], []
    x = recording.eeg
    for first in range(0, len(x) - n + 1, n):
        chunk = x[first:first + n]
        frame = recording.frame.iloc[first:first + n]
        qc = assess_epoch(chunk, frame, policy, require_sqi=is_bis)
        quality.append(qc)
        row = {"time_s": float(recording.time[first]), "epoch_end_s": float(recording.time[first] + epoch_s),
               "accepted": qc["accepted"], "quality_reasons": ";".join(qc["rejection_reasons"]),
               "quality_warnings": ";".join(qc["warnings"]), "sqi_min": qc["sqi_min"],
               "emg_max_db": qc["emg_max_db"], "soi": None, "rdi": None, "classification": "indeterminate"}
        for name in ("BIS", "SQI", "EMG", "SEF", "SR", "TOTPOW"):
            if name in frame:
                finite = frame[name].to_numpy(float)
                finite = finite[np.isfinite(finite)]
                row[f"recorded_{name.lower()}"] = float(np.median(finite)) if len(finite) else None
        if qc["accepted"]:
            for j, name in enumerate(recording.channels):
                for key, value in spectral_features(chunk[:, j], recording.fs, line_hz).items():
                    row[f"{name}__{key}"] = value
                # Only a complete, accepted past 60 seconds can support this estimator.
                history_n = round(60 * recording.fs)
                begin = first + n - history_n
                bic = None
                if compute_bicoherence and begin >= 0:
                    history = x[begin:first + n]
                    past_qc = assess_epoch(history, recording.frame.iloc[begin:first + n], policy, require_sqi=is_bis)
                    if past_qc["accepted"] and not past_qc["warnings"]:
                        bic = squared_bicoherence(history[:, j], recording.fs)
                row[f"{name}__auto_bicoherence_squared_60s"] = bic
            # Keep pair features named by actual scalp sites, never by channel number.
            # No connectivity is emitted for an unverified/common reference.
            if reference_verified:
                for left, right in (("Fp1", "P3"), ("Fp2", "P4"), ("Fp1", "T7"), ("Fp2", "T8")):
                    if left in site_to_channel and right in site_to_channel:
                        a = recording.channels.index(site_to_channel[left])
                        b = recording.channels.index(site_to_channel[right])
                        row[f"pair_{left}_{right}__alpha_plv"] = phase_locking_value(chunk[:, a], chunk[:, b], recording.fs)
        rows.append(row)
    feature_frame = pd.DataFrame(rows)
    accepted = int(feature_frame.accepted.sum())
    summary = {
        "use": "research_only", "clinical_validation": "not_established", "classification": "indeterminate",
        "SOI": None, "RDI": None,
        "index_reason": "The ten article components and a documented 24-hour awake personal baseline are not supplied to this raw-EEG analysis",
        "recording_context": recording.metadata.get("recording_context", "unspecified"),
        "sample_rate_hz": recording.fs, "samples": len(x), "channels": recording.channels,
        "epoch_s": epoch_s, "complete_epochs": len(rows), "accepted_epochs": accepted,
        "rejected_epochs": len(rows) - accepted, "unused_tail_samples": len(x) % n,
        "quality_policy": asdict(policy), "notch_hz": line_hz,
        "bicoherence": {"estimator": "mean_squared_auto_bicoherence", "history_s": 60,
                        "subsegment_s": 2, "overlap": 0.5, "f1_f2_hz": [5, 15],
                        "claim": "experimental waveform descriptor; not the proprietary BIS algorithm or the article's calibrated bifrontal component"},
        "montage": {"sensor": recording.metadata.get("sensor", {}), "scalp_locations": positions,
                    "reference_verified": reference_verified,
                    "bilateral_measurement": "not_established" if is_bis else "depends_on_verified_channel_metadata"},
        "article_components_required": {"SOI": list(SOI_DIRECTIONS), "RDI": list(RDI_COMPONENTS)},
        "provenance": recording.metadata,
    }
    return feature_frame, summary

