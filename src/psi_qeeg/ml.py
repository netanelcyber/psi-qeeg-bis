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



GROUP_LABELS = {"autism_spectrum", "psychosis_spectrum", "healthy_control"}


def subject_training_weights(subjects, labels):
    """One unit of scaler mass per person; balance SVM classes by people.

    Compute on the training fold only. Exact replication of one person's
    epochs preserves their total contribution and the SVM regularization.
    """
    table = pd.DataFrame({'subject': subjects, 'label': labels})
    if table.empty or table.isna().any().any() or (table.groupby('subject').label.nunique() != 1).any():
        raise ValueError('Weights need nonempty subjects with one diagnosis per subject')
    counts = table.groupby('subject').subject.transform('size').to_numpy(float)
    scaler = 1 / counts
    people = table.drop_duplicates('subject')
    class_counts = people.groupby('label').size()
    class_weights = len(people) / (len(class_counts) * class_counts)
    svm = scaler * table.label.map(class_weights).to_numpy(float)
    return scaler, svm


def train_group_model(table_path, out: Path, feature_columns: list[str], progress=None) -> dict:
    """Leave-one-subject-out classifier of diagnostic group (ASD / PSY / HC) from pooled epoch features.

    Input is ``cohort_features.csv`` from ``convert-cohort``. Only accepted epochs are used and missing
    features are not imputed. Epochs of one person are not independent, so subject-level majority-vote
    metrics are reported next to epoch-level ones. Group is a dataset-documented diagnosis, never a
    state label, and a ``site`` column (if present) is checked for total confounding with group.
    Every training participant has equal scaler mass; SVM weights also balance
    classes by participant count, so recording length does not define influence.
    """
    try:
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        import joblib
    except ImportError as exc:
        raise ValueError("Training requires: pip install 'psi-qeeg-bis[ml]'") from exc
    frame = pd.read_csv(table_path, dtype={"subject_id": str})
    if not {"subject_id", "group"}.issubset(frame):
        raise ValueError("Group table needs subject_id and group columns (see convert-cohort)")
    if "accepted" in frame:
        frame = frame[frame.accepted.astype(str).str.lower() == "true"].reset_index(drop=True)
    if frame[["subject_id", "group"]].isna().any().any():
        raise ValueError("subject_id and group cannot be missing")
    if not set(frame.group) <= GROUP_LABELS or frame.group.nunique() < 2:
        raise ValueError(f"Need at least two groups from {sorted(GROUP_LABELS)}")
    if frame.groupby("subject_id").group.nunique().max() > 1:
        raise ValueError("A subject_id belongs to more than one group")
    per_group = frame.groupby("group").subject_id.nunique()
    if per_group.min() < 2:
        raise ValueError(f"Each group needs at least two subjects for held-out validation; got {per_group.to_dict()}")
    if "recording_context" in frame and frame.recording_context.astype(str).str.contains("anesthesia|perioperative", case=False).any():
        raise ValueError("Surgical BIS recordings cannot supply psychiatric-group training data")
    if "site" in frame and (frame.groupby("site").group.nunique() == 1).all():
        raise ValueError("Every site contains a single group; group is fully confounded with site/device")
    id_cols = {"subject_id", "group", "site", "accepted"}
    if not feature_columns or set(feature_columns) & id_cols or any(c not in frame for c in feature_columns):
        raise ValueError("Select explicit numeric features; identifiers, group and site cannot be features")
    x = frame[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Missing features require a reviewed protocol; this trainer does not impute them")
    y, subjects = frame.group.to_numpy(), frame.subject_id.to_numpy()
    classes = sorted(set(y))
    factory = lambda: make_pipeline(StandardScaler(), SVC(kernel="linear", C=1))

    def fit(indices):
        scaler_weights, svm_weights = subject_training_weights(subjects[indices], y[indices])
        return factory().fit(x[indices], y[indices], standardscaler__sample_weight=scaler_weights,
                             svc__sample_weight=svm_weights)

    predicted = np.empty(len(frame), dtype=object)
    folds = []
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(x, y, subjects), 1):
        if progress:
            progress(f'Fold {fold}/{len(set(subjects))}: hold out {subjects[test][0]}, train={len(train)}, test={len(test)}')
        predicted[test] = fit(train).predict(x[test])
        folds.append({'held_out_subject': str(subjects[test][0]),
                      'training_subjects': sorted(set(subjects[train])),
                      'training_subjects_per_group': frame.iloc[train].groupby('group').subject_id.nunique().to_dict(),
                      'train_epochs': len(train), 'test_epochs': len(test)})
    votes = pd.DataFrame({"subject_id": subjects, "truth": y, "pred": predicted})
    per_subject = votes.groupby("subject_id").agg(truth=("truth", "first"), pred=("pred", lambda s: s.mode().iloc[0]),
                                                  epochs=("pred", "size")).reset_index()
    result = {"use": "research_only", "clinical_validation": "not_established", "evaluation": "leave_one_subject_out",
              "classes": classes, "subjects_per_group": per_group.to_dict(), "epochs": len(frame), "features": feature_columns,
              "epoch_balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
              "epoch_macro_f1": float(f1_score(y, predicted, labels=classes, average="macro", zero_division=0)),
              "subject_balanced_accuracy": float(balanced_accuracy_score(per_subject.truth, per_subject.pred)),
              "subject_confusion_matrix": {"labels": classes, "rows_true_cols_pred":
                  confusion_matrix(per_subject.truth, per_subject.pred, labels=classes).tolist()},
              "site_checked": "site" in frame,
              "training_balance": {"scaler": "1 / accepted training epochs per subject; total mass 1 per person",
                                   "svm": "scaler weight * training_subjects / (classes * training_subjects_in_class)",
                                   "regularization": "C=1 with one unit per person before class balancing; independent of recording duration",
                                   "fold_scope": "Weights and StandardScaler estimated from training subjects only"},
              "folds": folds,
              "limitations": "Group = documented diagnosis in the supplied cohort. Without a site column, group may be confounded "
                             "with device, site or age. Subject-level metrics are the meaningful ones. No diagnostic claim."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "group_evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    per_subject.to_csv(out / "held_out_subject_predictions.csv", index=False)
    scaler_weights, svm_weights = subject_training_weights(subjects, y)
    frame[['subject_id', 'group']].assign(scaler_weight=scaler_weights, svm_weight=svm_weights).groupby(
        ['subject_id', 'group']).agg(epochs=('scaler_weight', 'size'), scaler_mass=('scaler_weight', 'sum'),
                                   svm_mass=('svm_weight', 'sum')).reset_index().to_csv(out / 'training_subject_weights.csv', index=False)
    joblib.dump(fit(np.arange(len(frame))), out / "group_model.joblib")
    return result
