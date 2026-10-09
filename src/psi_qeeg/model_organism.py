"""Descriptive re-analysis of published mouse auditory-gating measurements.

Genotype and stimulus conditions are preserved as experimental labels. They
never become human psychiatric-state labels. Repeated ISIs are not independent
animals, and already averaged ERPs are not treated as raw continuous EEG.
"""

import numpy as np
import pandas as pd


EXPECTED_ISI_MS = {100, 200, 300, 400, 500, 750, 1000}


def source_erp_gating(erp: np.ndarray, channels=None) -> np.ndarray:
    """Reproduce the author's peak windows on processed 7 x 1343 x 5 ERPs.

    With channels supplied, average their waveforms *before* peak extraction,
    as the source MATLAB analysis does. Indices are translated from MATLAB's
    inclusive one-based windows; no resampling or new raw-EEG PSD is performed.
    """
    erp = np.asarray(erp, dtype=float)
    if erp.shape != (7, 1343, 5) or not np.isfinite(erp).all():
        raise ValueError("Expected finite processed ERP array of shape (7, 1343, 5)")
    if channels is not None:
        if not channels or any(c not in range(5) for c in channels) or len(set(channels)) != len(channels):
            raise ValueError("Select unique zero-based ERP channel indices 0..4")
        erp = erp[:, :, channels].mean(axis=2, keepdims=True)
    result = []
    for index, shift in enumerate((78, 156, 234, 312, 390, 585, 780)):
        first = np.ptp(erp[index, 79:320, :], axis=0)
        second = np.ptp(erp[index, shift + 79:shift + 320, :], axis=0)
        if (first <= np.finfo(float).eps).any():
            raise ValueError("Cannot calculate a gating ratio from a zero first-stimulus response")
        result.append(second / first)
    return np.asarray(result)


def compare_mouse_channel_coverage(animals: list[dict]) -> dict:
    """Fixed five-channel vs original two-channel genotype benchmark, held out by animal.

    The restricted set averages source channels 2 and 3 (zero-based 1 and 2),
    matching the published gating analysis. No channel/ISI is selected by outcome.
    """
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    identifiers = [a["animal_id"] for a in animals]
    y = np.asarray([a["genotype"] for a in animals])
    if len(identifiers) != len(set(identifiers)) or set(y) != {"wt", "hem"}:
        raise ValueError("Need unique animal IDs and documented wt/hem labels")
    if min(np.count_nonzero(y == label) for label in set(y)) < 2:
        raise ValueError("Need two or more independent animals per genotype")
    matrices = {
        "five_recorded_channels": np.asarray([source_erp_gating(a["erp"]).ravel() for a in animals]),
        "original_two_channel_average": np.asarray([source_erp_gating(a["erp"], [1, 2]).ravel() for a in animals]),
    }
    labels = ["hem", "wt"]
    results = {}
    for name, x in matrices.items():
        predictions = np.empty(len(y), dtype=object)
        for train, test in LeaveOneOut().split(x):
            model = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced"))
            predictions[test] = model.fit(x[train], y[train]).predict(x[test])
        results[name] = {
            "animals": len(y), "features": int(x.shape[1]),
            "animal_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "confusion_matrix": {"labels": labels, "rows_true_cols_pred": confusion_matrix(y, predictions, labels=labels).tolist()},
            "held_out_predictions": [{"animal_id": sid, "truth": str(truth), "prediction": str(pred)}
                                     for sid, truth, pred in zip(identifiers, y, predictions)],
        }
    return {
        "evaluation": "leave_one_animal_out", "target": "documented genotype, not episode/state",
        "data": "processed averaged ERPs; seven ISIs per animal",
        "comparisons": results,
        "protocol": "All 22 animals and seven ISIs in both sets; training-fold-only scaling; fixed linear SVC with balanced class weights; no tuning or winning-channel selection",
        "coverage_limit": "Five recorded ERP channels versus original two-channel mean only; not synchronized video, EMG, autonomic monitoring or BIS-equivalent electrodes",
        "interpretation": "Exploratory monitoring-coverage comparison for genotype information; no event detection, significance claim or clinical transfer",
    }


def summarize_mouse_gating(frame: pd.DataFrame) -> dict:
    required = {"ID", "ISI", "ratio", "genotype"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Mouse gating table requires complete ID, ISI, ratio and genotype columns")
    if set(frame.genotype) != {"wt", "hem"}:
        raise ValueError("Preserve documented wt and hem genotype labels")
    if not np.isfinite(frame[["ISI", "ratio"]].to_numpy(float)).all() or (frame.ratio < 0).any():
        raise ValueError("ISI and gating ratio must be finite; ratios cannot be negative")
    if frame.groupby("ID").genotype.nunique().max() != 1:
        raise ValueError("An animal cannot have multiple genotypes")
    if frame.duplicated(["ID", "ISI"]).any():
        raise ValueError("Duplicate animal/ISI measurements would overweight an animal")
    for _, rows in frame.groupby("ID"):
        if set(rows.ISI) != EXPECTED_ISI_MS:
            raise ValueError("Each animal requires the same seven documented ISIs")
    counts = frame.groupby("genotype").ID.nunique()
    if counts.min() < 2:
        raise ValueError("Need at least two animals per genotype for descriptive SD")
    conditions = []
    for isi, rows in frame.groupby("ISI", sort=True):
        groups = {label: {"animals": int(len(part)), "mean_ratio": float(part.ratio.mean()),
                          "sd_ratio": float(part.ratio.std(ddof=1))}
                  for label, part in rows.groupby("genotype")}
        conditions.append({"isi_ms": int(isi), "groups": groups,
                           "mean_difference_hem_minus_wt": groups["hem"]["mean_ratio"] - groups["wt"]["mean_ratio"]})
    animal_means = frame.groupby(["genotype", "ID"]).ratio.mean().reset_index()
    overall = {label: {"animals": int(len(part)), "mean_ratio": float(part.ratio.mean()),
                       "sd_between_animal_means": float(part.ratio.std(ddof=1))}
               for label, part in animal_means.groupby("genotype")}
    return {
        "species": "Mus musculus", "model": "Fmr1 KO2; source labels wt / hem",
        "experimental_state": "freely behaving, paired auditory stimulation",
        "data_type": "Published processed per-animal ERP gating ratios, not raw EEG",
        "measure": "S2/S1 peak-to-peak P1-N1 response ratio; larger values indicate less suppression",
        "source_channel_aggregation": "Source MATLAB script averages ERP channels 2 and 3 before extracting peaks",
        "animals_per_genotype": {k: int(v) for k, v in counts.items()},
        "rows": int(len(frame)), "conditions": conditions, "animal_mean_across_all_isis": overall,
        "inference": "Descriptive only; no hypothesis test, event detector or clinical interpretation",
        "meltdown_labels": "not_available", "psychosis_onset_labels": "not_available",
        "SOI": None, "RDI": None,
    }
