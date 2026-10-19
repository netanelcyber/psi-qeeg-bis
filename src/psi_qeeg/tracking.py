"""Causal offline replay of the spectral tracker using project input and QC."""

from dataclasses import asdict
import numpy as np
import pandas as pd

from .quality import QualityPolicy, assess_epoch
from .recording import validate
from .spectral_tracking import SpectralConfig, SpectralExtractor, fit_fft_noise, KalmanCUSUM, sample_count


MODEL_PARAMETERS = ("phi", "process_variance", "drift_variance", "residual_covariance",
                    "cusum_shift", "cusum_threshold")


def track_recording(recording, model: dict, baseline_s: float, *, window_s=2.0, hop_s=1.0,
                    segment_s=1.0, nfft=None, frontal_pair=None, draws=1024, seed=0,
                    high_z_limit=None, policy: QualityPolicy | None = None):
    """Fit only the declared baseline prefix; replay subsequent windows causally.

    This is a research-state/change trace. It does not operate the separate
    meltdown-alert notification system or compute psychiatric classifications.
    """
    validate(recording)
    if not isinstance(model, dict):
        raise ValueError("Model must be a JSON object")
    if recording.metadata.get("eeg_unit") != "uV":
        raise ValueError("Tracking expects project Recording amplitudes in uV")
    subject = recording.metadata.get("subject_id")
    if not subject or model.get("subject_id") != subject:
        raise ValueError("Model subject_id must match the recording's declared subject_id")
    medication = recording.metadata.get("medication_profile_id", "not_recorded")
    if model.get("medication_profile_id") != medication:
        raise ValueError("Model medication_profile_id must match the recording (or explicitly not_recorded)")
    loadings = model.get("loadings")
    if not isinstance(loadings, dict) or not loadings:
        raise ValueError("Model must supply named measurement loadings")
    unknown = set(model) - set(MODEL_PARAMETERS) - {"subject_id", "medication_profile_id", "loadings", "description"}
    if unknown:
        raise ValueError(f"Unknown model fields: {', '.join(sorted(unknown))}")
    if frontal_pair:
        positions = recording.metadata.get("channel_locations", {})
        sites = tuple(positions.get(c) for c in frontal_pair)
        permitted = {("Fp1", "Fp2"), ("AF3", "AF4"), ("AF7", "AF8"),
                     ("F1", "F2"), ("F3", "F4"), ("F7", "F8")}
        if recording.metadata.get("reference_verified") is not True or sites not in permitted:
            raise ValueError("FAA requires a verified reference and documented left/right frontal scalp sites")
    if high_z_limit is not None and (not np.isfinite(high_z_limit) or high_z_limit <= 0):
        raise ValueError("high_z_limit must be finite and positive")
    cfg = SpectralConfig(recording.fs, tuple(recording.channels), window_s, hop_s, segment_s,
                         nfft=nfft, frontal_pair=frontal_pair, measurements=tuple(loadings))
    extractor = SpectralExtractor(cfg)
    baseline_n = sample_count(baseline_s, recording.fs)
    eeg = recording.eeg
    if len(eeg) < baseline_n + cfg.window_n:
        raise ValueError("Need a baseline prefix followed by at least one complete monitoring window")
    policy = policy or QualityPolicy()
    is_bis = recording.metadata.get("sensor", {}).get("family") == "BIS"
    qc = assess_epoch(eeg[:baseline_n], recording.frame.iloc[:baseline_n], policy, is_bis)
    if not qc["accepted"] or "possible_muscle_contamination" in qc["warnings"]:
        raise ValueError("Declared baseline is not clean: " + ";".join(qc["rejection_reasons"] + qc["warnings"]))
    calibration = fit_fft_noise(eeg[:baseline_n].T, extractor, draws, seed)
    tracker = KalmanCUSUM(calibration, list(loadings.values()),
                          **{name: model[name] for name in MODEL_PARAMETERS if name in model})
    rows = []
    for start in range(baseline_n, len(eeg) - cfg.window_n + 1, cfg.hop_n):
        frame = recording.frame.iloc[start:start + cfg.window_n]
        window = eeg[start:start + cfg.window_n]
        qc = assess_epoch(window, frame, policy, is_bis)
        reasons = list(qc["rejection_reasons"])
        if "possible_muscle_contamination" in qc["warnings"]:
            reasons.append("possible_muscle_contamination")
        row = {"window_start_s": float(recording.time[start]),
               "window_end_s": float(recording.time[start] + window_s),
               "quality_warnings": ";".join(qc["warnings"])}
        normalized = None
        if not reasons:
            measured = extractor.extract(window.T)
            for i, ch in enumerate(cfg.channels):
                for j, band in enumerate(extractor.bands):
                    row[f"{ch}:log_{band}"] = float(measured.log_power[i, j])
                high_z = (measured.log_power[i, -1] - calibration.high_center[i]) / calibration.high_scale[i]
                row[f"{ch}:high_z"] = float(high_z)
                if high_z_limit is not None and high_z > high_z_limit:
                    reasons.append(f"possible_high_frequency_contamination:{ch}")
            for name, value in zip(extractor.names, measured.values):
                row[name] = float(value)
            if not reasons:
                normalized = calibration.normalize(measured.values)
        result = tracker.step(normalized)
        innovation = result.pop("innovation")
        s = result.pop("innovation_covariance")
        if innovation is not None:
            for i, name in enumerate(extractor.names):
                row[f"innovation:{name}"] = innovation[i]
                row[f"innovation_variance:{name}"] = s[i][i]
        rows.append({**row, **result, "quality_reasons": ";".join(reasons)})
    trace = pd.DataFrame(rows)
    summary = {
        "use": "research_only", "clinical_validation": "not_established", "classification": "indeterminate",
        "SOI": None, "RDI": None, "subject_id": subject, "medication_profile_id": medication,
        "baseline_s": baseline_s, "baseline_samples": baseline_n,
        "baseline_is_calm": "declared_by_caller_not_verified_by_this_algorithm",
        "monitoring_windows": len(rows), "measurement_updates": int(trace.measurement_used.sum()),
        "prediction_only_windows": int((~trace.measurement_used).sum()),
        "change_flag_windows": int(trace.change_flag.sum()), "high_z_limit": high_z_limit,
        "model": model, "effective_model": tracker.parameters, "quality_policy": asdict(policy),
        "unused_tail_samples": (len(eeg) - baseline_n - cfg.window_n) % cfg.hop_n,
        "noise_calibration": calibration.to_dict(), "provenance": recording.metadata,
    }
    return trace, summary
