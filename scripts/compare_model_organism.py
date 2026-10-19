"""Verify and summarize real public mouse data alongside the human cohort's scope.

Reuses the author's processed data. It does not fabricate raw recordings,
behavioral episode labels, shared electrode locations or cross-species scores.
"""

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np
from scipy.io import loadmat

from psi_qeeg.model_organism import summarize_mouse_gating, source_erp_gating, compare_mouse_channel_coverage


FILES = {
    "paired_tone_gating.csv": "611cd921d2109700616b1857",
    "paired_tone_analysis.m": "611cd9222dab24005c25bbc5",
    "description.txt": "611cd8a12dab24005925ba64",
}
PINNED_GATING_SHA256 = "04b3aaa6e7b9c0104850774af54950723fd16994cd612b04a32e94e3f845fb07"


def fetch(url):
    request = Request(url, headers={"User-Agent": "psi-qeeg-bis-model-organism-comparison/1.0"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def load_mouse_erps(cache, gating):
    """Download the 22 source-verified processed ERPs and check the published ratios."""
    analysis = (cache / "paired_tone_analysis.m").read_text(encoding="utf-8")
    groups = {}
    ordered = []
    for label in ("wt", "hem"):
        match = re.search(rf"ERP{label}\s*=\s*cat\(4,(.*?)\);", analysis, re.S)
        if not match:
            raise ValueError(f"Cannot verify {label} animal assignments in the source MATLAB script")
        members = re.findall(r"PT_FE\d+", match.group(1))
        ordered.extend(members)
        groups.update({member: label for member in members})
    if len(ordered) != 22 or len(groups) != 22:
        raise ValueError("Expected 22 unique source-assigned animals")
    listing_url = "https://api.osf.io/v2/nodes/cvefk/files/osfstorage/611cd91387c8f10069b11318/?page[size]=100"
    listing = json.loads(fetch(listing_url))
    if listing.get("links", {}).get("next"):
        raise ValueError("Source directory needs pagination; review before continuing")
    by_name = {item["attributes"]["name"]: item for item in listing["data"]}

    def load_one(sid):
        item = by_name[sid + ".mat"]
        path = cache / (sid + ".mat")
        data = path.read_bytes() if path.exists() else fetch(item["links"]["download"])
        expected = item["attributes"]["extra"]["hashes"]["sha256"]
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"Source ERP SHA-256 mismatch for {sid}")
        path.write_bytes(data)
        erp = loadmat(path)[sid]
        return {"animal_id": sid, "genotype": groups[sid], "erp": erp}, {
            "source_filename": sid + ".mat", "source_file_id": item["id"], "sha256": actual,
            "bytes": len(data), "source_url": item["links"]["download"],
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        pairs = list(pool.map(load_one, ordered))
    animals, provenance = [x[0] for x in pairs], [x[1] for x in pairs]
    differences = []
    for source_id, animal in enumerate(animals, 1):
        rows = gating[gating.ID == source_id].sort_values("ISI")
        if len(rows) != 7 or set(rows.genotype) != {animal["genotype"]}:
            raise ValueError("Source CSV animal ordering/genotypes do not match the MATLAB assignments")
        reproduced = source_erp_gating(animal["erp"], [1, 2]).ravel()
        differences.extend(abs(reproduced - rows.ratio.to_numpy()))
    error = float(np.max(differences))
    if error > 5e-9:
        raise ValueError(f"Processed ERPs do not reproduce the published paired-tone ratios: max error {error}")
    return animals, provenance, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/model_organism/fmr1_ko2"))
    parser.add_argument("--human-report", type=Path,
                        default=Path("outputs/repod_schizophrenia/public_cohort_validation.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/model_organism/comparison.json"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    provenance = []
    for filename, file_id in FILES.items():
        api = f"https://api.osf.io/v2/files/{file_id}/"
        info = json.loads(fetch(api))["data"]
        attributes = info["attributes"]
        expected = attributes["extra"]["hashes"]["sha256"]
        path = args.cache / filename
        if not path.exists():
            data = fetch(info["links"]["download"])
        else:
            data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected or (filename == "paired_tone_gating.csv" and actual != PINNED_GATING_SHA256):
            raise ValueError(f"Source SHA-256 mismatch for {filename}; review the source version before continuing")
        path.write_bytes(data)
        provenance.append({"local_filename": filename, "source_filename": attributes["name"],
                           "file_id": file_id, "source_api": api, "source_url": info["links"]["download"],
                           "sha256": actual, "bytes": len(data), "source_modified": attributes["date_modified"]})
    gating = pd.read_csv(args.cache / "paired_tone_gating.csv")
    mouse = summarize_mouse_gating(gating)
    if mouse["animals_per_genotype"] != {"hem": 12, "wt": 10} or mouse["rows"] != 154:
        raise ValueError("Source data no longer match the documented 22-animal, seven-ISI cohort")
    human = json.loads(args.human_report.read_text(encoding="utf-8"))
    animals, erp_provenance, reproduction_error = load_mouse_erps(args.cache, gating)
    monitoring = compare_mouse_channel_coverage(animals)
    monitoring["published_ratio_reproduction_max_absolute_error"] = reproduction_error
    report = {
        "status": "descriptive_model_organism_comparison_completed",
        "event_validation_status": "not_possible_with_these_labels",
        "mouse_source": "https://doi.org/10.17605/OSF.IO/CVEFK",
        "source_readme": "Raw EDFs available on author request; processed ERPs and gating CSV are public",
        "mouse_provenance": provenance, "mouse": mouse,
        "processed_erp_provenance": erp_provenance,
        "expanded_channel_monitoring": monitoring,
        "human": {key: human[key] for key in (
            "dataset", "source", "subjects_per_group", "accepted_epochs", "subject_balanced_accuracy")},
        "comparison": {
            "human_sensory_gating": None,
            "reason": "Human resting-state cohort has no paired-stimulus timestamps or S1/S2 ERPs; the same gating ratio cannot be calculated",
            "animal_spectral_classification": None,
            "spectral_reason": "Published averaged ERPs and ratios cannot substitute for continuous single-trial EEG when estimating the human spectral benchmark",
            "human_vs_mouse_episode_accuracy": None,
            "episode_reason": "Neither dataset provides independently adjudicated autistic-meltdown or psychosis-onset events",
            "cross_species_model_transfer": "not_run; incompatible labels, acquisition and task",
        },
        "psychosis_model_evidence": {
            "source": "https://doi.org/10.3389/fnins.2022.1001869",
            "species": "Mus musculus", "signal": "mPFC / CA1 intracranial LFP",
            "experimental_label": "NMDAR-antagonist exposure; a schizophrenia-related pharmacological model",
            "data_access": "Article and supplementary results; no public raw LFP repository identified in its data-availability statement",
            "measured_here": False,
            "band_comparison": "Source high gamma 60-100 Hz and HFO 150-200 Hz are outside the project's 0.5-40 Hz features; 128 Hz BIS EEG cannot span those bands",
            "onset_interpretation": "Experimental intervention time is not documented spontaneous human psychosis onset",
        },
        "clinical_validation": "not_established",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.out), "mouse_animals": mouse["animals_per_genotype"],
                      "gating_mean_by_genotype": mouse["animal_mean_across_all_isis"],
                      "expanded_channel_monitoring": monitoring,
                      "event_validation_status": report["event_validation_status"]}, indent=2))


if __name__ == "__main__":
    main()
