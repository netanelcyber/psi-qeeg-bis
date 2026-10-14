# RepOD: full-recording human EEG experiment

**Corrected and rerun 13 October 2026 with participant-balanced training.** Source: [EEG in schizophrenia, RepOD v1.0](https://doi.org/10.18150/repod.0107441), CC0, 28 recordings/participants (14 schizophrenia, 14 controls), 19 scalp channels at 250 Hz. All source EDFs were downloaded and verified against the publisher metadata's MD5 and byte count. The original 12 October epoch-pooled benchmark is retained as [historical unweighted evidence](expanded_human_cohort_unweighted_history.json), separate from the current run.

## Segments and results

The full recordings contain **28,863 seconds** (8.02 hours), **7,205 complete non-overlapping four-second epochs** and **6,870 accepted epochs**. The previous 120-second selection yielded 840 complete and 821 accepted epochs. This is 8.58 times as many complete epochs and 8.37 times as many accepted epochs, still only **28 independent people**. The 335 rejected epochs follow the existing quality rules; one recording, PSY_S14, contributes 227 of them. Quality rejection is not a psychiatric outcome.

| Feature coverage | Corrected entire-recording subject balanced accuracy | Corrected epoch balanced accuracy | Subject confusion: HC/PSY, rows true / columns predicted |
| --- | ---: | ---: | --- |
| Full scalp, 19 channels / 76 features | 78.57% | 68.99% | `[[13,1],[5,9]]` |
| Left regional, Fp1/F7 / 8 features | 50.00% | 48.01% | `[[9,5],[9,5]]` |
| Right regional, Fp2/F8 / 8 features | 25.00% | 41.63% | `[[1,13],[8,6]]` |

All comparisons use the same accepted epochs/subjects and full-scalp quality decision. Regional columns retain the dataset's original reference; they do not recreate commercial BIS derivations. Features remain per-channel alpha power, delta/alpha ratio, SEF95 and entropy. Leave-one-subject-out evaluation fits a weighted StandardScaler plus linear SVC in each training fold. For participant `s` with `n_s` accepted training epochs, scaler row weight is `1/n_s`, giving every participant total mass 1. SVC row weight is `(1/n_s)*S/(K*S_class)`, balancing classes by training-participant counts, with `C=1`. Weights are recomputed from training participants only. Their normalization uses people as the objective's units and changes regularization relative to the former epoch-pooled model; exact replication of one person's epochs preserves the objective. Subject predictions use epoch majority voting, with each person counted once. Epoch metrics remain secondary and can reflect differing durations. Final all-data model refits are distinct from held-out evaluation.

The full-scalp balanced accuracy coincides with the former 78.57%, but its confusion matrix and participant predictions changed. Former full-recording regional scores were 67.86%/39.29%; corrected scores are 50%/25%. Historical first-120-second scores (64.29%/46.43%/60.71%) also used the former epoch-pooled objective and are not a matched duration comparison with the current model. No improvement is claimed. No significance test, participant-level confidence interval, hyperparameter search, independent site or prospective evaluation was performed. These point estimates support an exploratory diagnosis-group comparison only.

## Reproduction and checks

```bash
python -m pip install -e '.[edf,ml]'
python scripts/validate_public_schizophrenia_cohort.py \
  --window-s 0 --epoch-s 4 --discard-waveform-csv --out outputs/repod_subject_balanced
python scripts/verify_methodology_corrections.py
```

Omit `--window-s 0` to retain the historical 120-second default. A pinned API snapshot can be supplied with `--source-metadata`. `--data-root` selects a source cache. The runner checks source MD5/size, atomically replaces corrupt downloads, exports session/epoch IDs and records actual extent per subject. Discarding the intermediate waveform CSV does not discard derived feature tables.

The delivered receipt retains arguments, timestamps, source metadata, dependency versions and current SHA-256 fingerprints. Recomputed subject metrics/confusion matrices, matching epoch counts, all three model reloads, equal subject/class weight totals and the actual weighted scaler means are checked independently. Source total remains 274,341,860 bytes. Each fit exports `training_subject_weights.csv`, and LOSO receipts list held-out and training participants. See [expanded_human_cohort.json](expanded_human_cohort.json) for current aggregate evidence and [methodology_corrections_verification.json](methodology_corrections_verification.json) for the independent checks. Prior feature/runner fingerprints belong to the historical unweighted receipt.

The cohort runner disables auto-bicoherence, has no verified-site PLV/personal psychiatric baseline, and does not construct all ten SOI/RDI components. SOI/RDI remain unavailable and raw-EEG classification remains `indeterminate`.

## Event prediction remains unevaluated on this dataset

RepOD supplies diagnosis groups, **no ASD/meltdown onsets, psychosis-onset times or independently reviewed event-free follow-up**. Therefore no real event model was trained or evaluated, and no epoch was assigned an impending-attack label. A separate [annotation-dependent SVM workflow](EVENT_PREDICTION.md) has engineering tests using explicit synthetic fixtures. Its passing tests do not establish clinical prediction. Literature candidates and estimator gaps are [marked separately](PRE_EVENT_SIGNAL_EVIDENCE.md). LSTM/Transformer remains unimplemented.
