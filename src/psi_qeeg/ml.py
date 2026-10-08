"""Optional subject-held-out research training. No psychiatric labels are invented."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED_LABELS = {"baseline", "sensory_loading", "pre_meltdown", "pre_psychotic", "indeterminate"}


def train_research_model(table_path, out: Path, feature_columns: list[str]) -> dict:
    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        import joblib
    except ImportError as exc:
        raise ValueError("Training requires: pip install 'psi-qeeg-bis[ml]'") from exc
    frame = pd.read_csv(table_path, dtype={"subject_id": str})
    required = {"subject_id", "session_id", "epoch_id", "research_label", "label_source", "recording_context"}
    if not required.issubset(frame):
        raise ValueError(f"Label table needs {sorted(required)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Label provenance and identifiers cannot be missing")
    if not set(frame.research_label).issubset(ALLOWED_LABELS) or frame.research_label.nunique() < 2:
        raise ValueError("Need at least two explicitly labelled research states")
    if not set(frame.label_source).issubset({"participant", "clinician", "consensus"}):
        raise ValueError("Labels require participant, clinician or consensus provenance")
    if any("anesthesia" in str(c).lower() or "perioperative" in str(c).lower() for c in frame.recording_context):
        raise ValueError("Surgical BIS recordings cannot supply psychiatric-state training labels")
    if frame.subject_id.nunique() < 3:
        raise ValueError("Subject-held-out validation requires at least three distinct subjects")
    if frame.duplicated(["subject_id", "session_id", "epoch_id"]).any():
        raise ValueError("Duplicate subject/session/epoch records can leak into evaluation")
    if not feature_columns or set(feature_columns) & required or any(c not in frame for c in feature_columns):
        raise ValueError("Select explicit numeric features; identifiers, provenance and labels cannot be features")
    x = frame[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Missing features require a reviewed protocol; this trainer does not impute them")
    y, groups = frame.research_label.to_numpy(), frame.subject_id.to_numpy()
    model_factory = lambda: make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced"))
    predictions, folds = np.empty(len(frame), dtype=object), []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        if set(y[train]) != set(y):
            raise ValueError("A held-out fold lacks a training class; collect labels across more subjects")
        model = model_factory().fit(x[train], y[train])
        predicted = model.predict(x[test])
        predictions[test] = predicted
        folds.append({"held_out_subject": str(groups[test][0]), "n_train": len(train), "n_test": len(test),
                      "macro_f1": float(f1_score(y[test], predicted, labels=sorted(set(y)), average="macro", zero_division=0))})
    result = {"use": "research_only", "clinical_validation": "not_established", "evaluation": "leave_one_subject_out",
              "subjects": len(set(groups)), "rows": len(frame), "features": feature_columns,
              "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
              "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)), "folds": folds,
              "limitations": "Metrics apply only to this supplied labelled cohort; no prospective validation or diagnostic claim"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    frame.assign(held_out_prediction=predictions).to_csv(out / "held_out_predictions.csv", index=False)
    joblib.dump(model_factory().fit(x, y), out / "research_model.joblib")
    return result

