# Public cohort results, 9 October 2026

The RepOD schizophrenia/control dataset contains 14 diagnosed participants and 14 healthy controls, recorded at 250 Hz with 19 scalp channels and a reference between Fz and Cz ([dataset, V1](https://doi.org/10.18150/repod.0107441)). This is a diagnosis-group benchmark, not a study of episode onset.

## Reproduced experiment

- All 28 EDFs matched the source MD5 checksums; total source size 274,341,860 bytes.
- Fixed first 120 seconds per person; four-second non-overlapping epochs.
- 840 complete epochs, 821 accepted under the existing full-scalp QC policy; all 28 subjects retained.
- No outcome-based feature selection, parameter tuning, imputation or model/side selection.
- Four features per channel: alpha power, delta/alpha ratio, SEF95 and spectral entropy.
- Leave-one-subject-out linear SVC with balanced class weights. StandardScaler is fitted only on each training fold. Subject predictions use the existing majority vote; ties use the first sorted modal class.
- Identical accepted epochs, participants, folds and model settings for all coverage comparisons.

| Coverage | Channels | Features | Subject balanced accuracy | Confusion matrix (HC, PSY; rows true, columns predicted) |
| --- | --- | --- | --- | --- |
| Full scalp | 19 | 76 | 64.29% | `[[9, 5], [5, 9]]` |
| Left frontal/temporal | Fp1, F7 | 8 | 46.43% | `[[6, 8], [7, 7]]` |
| Right frontal/temporal | Fp2, F8 | 8 | 60.71% | `[[8, 6], [5, 9]]` |

The full-scalp result reproduces the earlier successful [GitHub run](https://github.com/netanelcyber/psi-qeeg-bis/actions/runs/37894100018). The two regional results are new locally executed comparisons. Reproduction commands are in the README; complete numeric results and dependency versions are in [public_cohort_validation.json](public_cohort_validation.json) and [electrode_comparison.json](electrode_comparison.json). These describe the executed code and processed feature file via SHA-256 fingerprints.

## Interpretation limits

These are point estimates from 28 people in one acquisition context, without confidence intervals, a permutation test, or independent external validation. A few subject decisions change the apparent performance substantially. No significance or population-level superiority is claimed for any feature set, and the lower left-side score is not evidence for a biological hemispheric effect.

Fp1/F7 and Fp2/F8 retain the dataset's original common reference. Fpz is absent and is not estimated. Their locations do not establish Quatro contact equivalence or its physical reference, ground, amplifier, SQI or commercial BIS algorithm. Full-scalp QC also uses channels outside each regional subset; the comparison isolates coverage on matched usable data rather than standalone sensor performance.

There are no ASD participants, meltdown annotations, synchronized onset events or prospective psychosis-onset outcomes in this cohort. Medication, age, movement and other residual group differences were not controlled by this experiment. The site check verifies both diagnoses occur in the same dataset; it does not establish absence of all confounding. SOI/RDI and diagnostic/clinical performance remain unvalidated.
