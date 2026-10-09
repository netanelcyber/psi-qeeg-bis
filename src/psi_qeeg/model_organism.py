"""Descriptive re-analysis of published mouse auditory-gating measurements.

Genotype and stimulus conditions are preserved as experimental labels. They
never become human psychiatric-state labels. Repeated ISIs are not independent
animals, and already averaged ERPs are not treated as raw continuous EEG.
"""

import numpy as np
import pandas as pd


EXPECTED_ISI_MS = {100, 200, 300, 400, 500, 750, 1000}


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
