"""Exact section 6.5 arithmetic, with missingness and calibration made explicit."""

import math

SOI_DIRECTIONS = {"alpha_variance": 1, "gating_efficiency": -1, "posterior_gamma": 1,
                  "frontoparietal_plv": -1, "autonomic_surge": 1}
RDI_COMPONENTS = ("temporal_csd", "delta_alpha_ratio", "bifrontal_bicoherence",
                  "hemispheric_asymmetry", "frontotemporal_coherence")


def _clip(value: float) -> float:
    return min(100.0, max(0.0, value))


def score_indices(values: dict, calibration: dict) -> dict:
    """No imputation or reweighting. All five components required per index.

    A finite score is an experimental mathematical composite, not a probability,
    a measure of consciousness, or a clinical classification.
    """
    result = {"use": "research_only", "classification": "indeterminate", "indices": {}}
    calibration_status = calibration.get("context", "unspecified")
    result["calibration_context"] = calibration_status
    context_issues = []
    duration = float(calibration.get("duration_s", 0))
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Calibration duration must be finite and nonnegative")
    if duration < 86400:
        context_issues.append("calibration_shorter_than_article_24h")
    if calibration_status not in {"awake_personal_baseline", "illustrative_paper_example"}:
        context_issues.append("calibration_not_documented_awake_baseline")
    for index, components in (("SOI", SOI_DIRECTIONS), ("RDI", RDI_COMPONENTS)):
        names = list(components)
        settings = calibration.get(index, {})
        weights = settings.get("weights", {n: 0.2 for n in names})
        if set(weights) != set(names) or any(not math.isfinite(float(v)) or float(v) < 0 for v in weights.values()):
            raise ValueError(f"{index} needs five nonnegative finite component weights")
        if not math.isclose(sum(map(float, weights.values())), 1, abs_tol=1e-9):
            raise ValueError(f"{index} weights must sum to 1; partial inputs cannot be renormalized")
        sub, missing = {}, []
        for name in names:
            current = values.get(name)
            baseline = settings.get("components", {}).get(name)
            if current is None or not math.isfinite(float(current)) or baseline is None:
                sub[name] = None
                missing.append(name)
                continue
            mean, sd = float(baseline["mean"]), float(baseline["sd"])
            if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
                raise ValueError(f"{name}: calibration SD must be positive and finite")
            if index == "SOI":
                sub[name] = _clip(100 * SOI_DIRECTIONS[name] * (float(current) - mean) / (3 * sd))
            else:
                sub[name] = _clip(100 * (1 - abs(float(current) - mean) / (3 * sd)))
        score = None if missing or context_issues else sum(float(weights[n]) * sub[n] for n in names)
        result["indices"][index] = {"score": score, "components": sub, "missing": missing,
                                      "component_coverage": (len(names) - len(missing)) / len(names),
                                      "calibration_issues": list(context_issues)}
    return result


def fit_calibration(frame, subject_id: str, context: str = "awake_personal_baseline") -> dict:
    """Fit only explicitly labelled calm epochs for one subject.

    Input rows need time_s, epoch_end_s, subject_id, research_label and ten
    section-6.5 feature columns. No diagnosis labels are inferred from EEG.
    """
    import numpy as np
    required = {"time_s", "epoch_end_s", "subject_id", "research_label", "recording_context"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Calibration table needs {sorted(required)}")
    selected = frame[(frame.subject_id.astype(str) == str(subject_id)) & (frame.research_label == "baseline")].copy()
    if len(selected) < 3:
        raise ValueError("Need at least three explicitly labelled baseline epochs for this subject")
    if not set(selected.recording_context).issubset({"awake_personal_baseline", "awake_observational_research"}):
        raise ValueError("Personal calibration requires an independently documented awake baseline context")
    for column in ("session_id", "medication_profile_id"):
        if column in selected and (selected[column].isna().any() or selected[column].nunique() != 1):
            raise ValueError(f"Calibration {column} must be consistent; do not mix sessions or medication profiles")
    selected = selected.sort_values("time_s")
    starts, ends = selected.time_s.to_numpy(float), selected.epoch_end_s.to_numpy(float)
    if not np.isfinite(np.r_[starts, ends]).all() or np.any(ends <= starts):
        raise ValueError("Calibration epoch times must be finite with positive duration")
    # Union duration prevents overlapping/sliding epochs from counting as extra hours.
    duration, end = 0.0, -float("inf")
    for a, b in zip(starts, ends):
        duration += max(0.0, b - max(a, end))
        end = max(end, b)
    result = {"subject_id": str(subject_id), "context": context, "duration_s": duration}
    result["medication_profile_id"] = str(selected.medication_profile_id.iloc[0]) if "medication_profile_id" in selected else "not_recorded"
    if "session_id" in selected:
        result["session_id"] = str(selected.session_id.iloc[0])
    for index, names in (("SOI", SOI_DIRECTIONS), ("RDI", RDI_COMPONENTS)):
        stats = {}
        for name in names:
            if name not in selected:
                continue
            values = selected[name].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Baseline {name} contains missing values; define and review the calibration cohort")
            sd = float(np.std(values, ddof=1))
            if sd <= 0:
                raise ValueError(f"Baseline {name} has zero variance")
            stats[name] = {"mean": float(values.mean()), "sd": sd, "n": len(values)}
        result[index] = {"components": stats}
    return result

