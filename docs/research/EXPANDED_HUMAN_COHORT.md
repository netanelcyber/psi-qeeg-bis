# RepOD: full-recording human EEG experiment

Executed **12 October 2026, 21:19–21:22 UTC** (13 October in Israel), extending the previous first-120-second benchmark. Source: [EEG in schizophrenia, RepOD v1.0](https://doi.org/10.18150/repod.0107441), CC0, 28 recordings/participants (14 schizophrenia, 14 controls), 19 scalp channels at 250 Hz. The same source EDFs were verified against the pinned publisher metadata's MD5 and byte count before analysis.

## Segments and results

The full recordings contain **28,863 seconds** (8.02 hours), **7,205 complete non-overlapping four-second epochs** and **6,870 accepted epochs**. The previous 120-second selection yielded 840 complete and 821 accepted epochs. This is 8.58 times as many complete epochs and 8.37 times as many accepted epochs, still only **28 independent people**. The 335 rejected epochs follow the existing quality rules; one recording, PSY_S14, contributes 227 of them. Quality rejection is not a psychiatric outcome.

| Feature coverage | First 120 s: subject balanced accuracy | Entire recording: subject balanced accuracy | Entire recording: epoch balanced accuracy |
| --- | ---: | ---: | ---: |
| Full scalp, 19 channels / 76 features | 64.29% | 78.57% | 67.87% |
| Left regional, Fp1/F7 / 8 features | 46.43% | 67.86% | 56.99% |
| Right regional, Fp2/F8 / 8 features | 60.71% | 39.29% | 39.29% |

All comparisons use the same accepted epochs/subjects and full-scalp quality decision. Regional columns retain the dataset's original reference; they do not recreate commercial BIS derivations. Features remain per-channel alpha power, delta/alpha ratio, SEF95 and entropy. The fixed model is training-fold StandardScaler plus linear SVC (`C=1`, balanced class weights), evaluated with leave-one-subject-out folds. Subject predictions use epoch majority voting. Final all-data model refits are distinct from the held-out evaluation.

Confusion matrices, rows true and columns predicted in order `[healthy_control, psychosis_spectrum]`: full scalp `[[12,2],[4,10]]`; left `[[10,4],[5,9]]`; right `[[4,10],[7,7]]`. The right regional result worsened. No significance test, participant-level confidence interval, hyperparameter search, independent site or prospective evaluation was performed. These point estimates support an exploratory diagnosis-group comparison only.

## Reproduction and checks

```bash
python -m pip install -e '.[edf,ml]'
python scripts/validate_public_schizophrenia_cohort.py \
  --window-s 0 --epoch-s 4 --discard-waveform-csv --out outputs/repod_extended
```

Omit `--window-s 0` to retain the historical 120-second default. A pinned API snapshot can be supplied with `--source-metadata`. `--data-root` selects a source cache. The runner checks source MD5/size, atomically replaces corrupt downloads, exports session/epoch IDs and records actual extent per subject. Discarding the intermediate waveform CSV does not discard derived feature tables.

The delivered run receipt retains arguments, timestamps, source metadata, dependency versions and SHA-256 fingerprints. Recomputed subject metrics/confusion matrices, common epoch membership and all three final-model reloads are checked separately. Source total is 274,341,860 bytes; pooled feature SHA-256 is `8d464a557a7130ba75dd9de688f9fce2dba82251bb8f756cbd0dd3ad6b1b9158`; executed runner SHA-256 is `ec06858ef1d21c9ea203c06ee158f1f3a7d5445cc91fdf048286129a24a4f0cd`. See [expanded_human_cohort.json](expanded_human_cohort.json) for machine-readable aggregate evidence.

The cohort runner disables auto-bicoherence, has no verified-site PLV/personal psychiatric baseline, and does not construct all ten SOI/RDI components. SOI/RDI remain unavailable and raw-EEG classification remains `indeterminate`.

## Event prediction remains unevaluated on this dataset

RepOD supplies diagnosis groups, **no ASD/meltdown onsets, psychosis-onset times or independently reviewed event-free follow-up**. Therefore no real event model was trained or evaluated, and no epoch was assigned an impending-attack label. A separate [annotation-dependent SVM workflow](EVENT_PREDICTION.md) has engineering tests using explicit synthetic fixtures. Its passing tests do not establish clinical prediction. Literature candidates and estimator gaps are [marked separately](PRE_EVENT_SIGNAL_EVIDENCE.md). LSTM/Transformer remains unimplemented.
