"""Fixed electrode-coverage comparisons on one shared, QC-filtered cohort.

Frontal/temporal scalp subsets retain the source reference. They are not actual
BIS sensor derivations, and no commercial BIS values or state labels are inferred.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .convert import site_of
from .ml import train_group_model


SPECTRAL_FEATURES = ("alpha_power_uv2", "delta_alpha_ratio", "sef95_hz", "spectral_entropy")
REGIONAL_SITES = {"left_frontal_temporal": ("fp1", "f7"),
                  "right_frontal_temporal": ("fp2", "f8")}


def electrode_feature_sets(columns) -> dict[str, list[str]]:
    """Select a fixed four-feature family for full scalp and two regional subsets.

    Selection uses column names alone, before evaluation. Missing/ambiguous sites
    fail rather than silently changing the electrode comparison.
    """
    full, by_site = [], {}
    for column in columns:
        channel, separator, feature = str(column).rpartition("__")
        if separator and feature in SPECTRAL_FEATURES:
            full.append(column)
            site = site_of(channel)
            by_site.setdefault(site, {}).setdefault(channel, {})[feature] = column
    if not full:
        raise ValueError("No spectral EEG features available for electrode comparison")
    result = {"full_scalp": sorted(full)}
    for name, sites in REGIONAL_SITES.items():
        selected = []
        for site in sites:
            candidates = by_site.get(site, {})
            if len(candidates) != 1:
                raise ValueError(f"Need exactly one channel for {site}; got {sorted(candidates)}")
            features = next(iter(candidates.values()))
            missing = set(SPECTRAL_FEATURES) - set(features)
            if missing:
                raise ValueError(f"Missing {site} spectral features: {sorted(missing)}")
            selected.extend(features[feature] for feature in SPECTRAL_FEATURES)
        result[name] = sorted(selected)
    return result


def compare_electrode_coverage(table_path, out: Path) -> dict:
    """LOSO fits with identical rows, subjects, QC and model settings for each set.

    Full-scalp outputs stay at ``out`` for compatibility. Restricted outputs are
    stored below ``out/electrode_comparison/<set>``. No best-side selection is made.
    """
    table_path, out = Path(table_path), Path(out)
    frame = pd.read_csv(table_path, dtype={"subject_id": str})
    sets = electrode_feature_sets(frame.columns)
    results = {}
    for name, columns in sets.items():
        destination = out if name == "full_scalp" else out / "electrode_comparison" / name
        result = train_group_model(table_path, destination, columns)
        results[name] = {key: result[key] for key in (
            "features", "subjects_per_group", "epochs", "epoch_balanced_accuracy",
            "subject_balanced_accuracy", "subject_confusion_matrix")}
    reference = results["full_scalp"]
    for result in results.values():
        if (result["epochs"] != reference["epochs"] or
                result["subjects_per_group"] != reference["subjects_per_group"]):
            raise RuntimeError("Electrode comparisons must use identical epochs and subjects")
    report = {
        "use": "research_only", "clinical_validation": "not_established",
        "evaluation": "leave_one_subject_out",
        "cohort_features_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "protocol": {
            "feature_selection": "Fixed channel names and four spectral families; no outcome-based selection",
            "quality_control": "Identical accepted full-scalp epochs used in all comparisons",
            "model": "StandardScaler fitted within training fold; linear SVC, balanced class weights",
            "reference": "Source common reference retained; no Fpz estimation or re-referencing",
            "selection": "All three comparisons reported; no best-side/model selection",
        },
        "comparisons": results,
        "limitations": [
            "Fp1/F7 and Fp2/F8 are scalp coverage approximations, not BIS Quatro derivations or BIS indices.",
            "Full-scalp QC can reject epochs for artifacts outside the regional subsets; this is a matched coverage comparison, not standalone sensor performance.",
            "Small single cohort; point estimates have no external or prospective validation.",
            "Schizophrenia/control diagnosis groups do not label autistic meltdown or psychosis onset.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "electrode_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
