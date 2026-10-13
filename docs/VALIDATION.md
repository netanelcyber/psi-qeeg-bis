# Validation status and next experiment

## Methodology corrections, 13 October 2026

The [PR #3 corrections](research/METHODOLOGY_CORRECTIONS.md) pass **159 local tests without failures, errors or skips**. The real SH023 data were reproduced byte-for-byte and all 2,288 PAC tests rerun with strict one-second surrogate exclusion, including both boundaries. An independent SciPy/entropy audit verifies saved PAC, a real null distribution, coherence and BH values. The real full RepOD run retains 7,205/6,870 complete/accepted epochs and 28 people, now with equal participant mass in training-fold scaling/SVM fitting. Corrected subject accuracies are 78.57% full scalp, 50% Fp1/F7 and 25% Fp2/F8. Training/inference acquisition and QC contracts, modality separation, positive-duration wheel annotations and configured surrogate receipts have regressions. [Current verification](research/methodology_corrections_verification.json) supersedes the earlier claims below where the protocol changed. No real abnormal-event prediction or clinical validation is established.

## Coupling analysis, 13 October 2026

The [IBL coupling study](research/IBL_COUPLING.md) executes PLV, Welch magnitude-squared/imaginary coherency and Tort theta–gamma modulation index on 143 unique real LFP windows, with three wheel-quiet controls, 199 PAC surrogates per relationship, multiplicity adjustment and notch/bipolar sensitivity checks. Its initial seven numerical regressions brought the then-local suite to **144 passing tests**; the corrected run is documented above. Tests distinguish identity/common-reference coherence from residual bipolar interactions and known amplitude modulation from an uncoupled control. Strong original synchrony is derivation-sensitive; selected raw pre-bout PAC increases already overlap previous movements. These are exploratory measurements in one mouse, not a trained event predictor or a validated SOI/RDI component, and no LSTM or clinical inference is introduced.

## Public synchronized video/LFP, 13 October 2026

The [IBL run](research/OPEN_VIDEO_LFP.md) executes a real continuous intracranial signal excerpt, calibrated FFT/Welch powers in 60 four-second epochs per channel, probe/session clock mapping and frame-count-checked video annotations of five deposited wheel movements. Runtime checks verify small-object hashes, exact compressed ranges, source gain/sample counts, ordered clocks and matching full/excerpt frame counts. These are acquisition/alignment checks. Individual genotype/reference metadata and artifact adjudication are incomplete, no abnormal-event labels are established and no behavioral model or one-minute forecast is trained. Strict event-training contracts remain in force.

## Continued training, 13 October 2026

The [continued Fmr1 study](research/CONTINUED_FMR1_TRAINING.md) fits nested animal-separated SVM selection, two source-experiment holdouts and selected/fixed-C=1 final all-animal refits. It reuses the same 43 mice and does not establish improved prediction. Its then-local suite passed **137 tests without skips**. Regressions cover annotation/outcome leakage, synthetic/real separation, fitted epoch duration, animal acquisition contracts, nested selection isolation and coefficient/model reload checks. [fmr1_continued_training_verification.json](research/fmr1_continued_training_verification.json) records those dated checks; no real event-trained model or LSTM is added.

## Animal extension, 13 October 2026 in Israel

The initial separate [autism-model mouse workflow](research/AUTISM_ANIMAL_MODEL.md) implements continuous-qEEG epochs and annotation-dependent behavioral SVM preparation, fitting and inference. Its initial local suite passed **123 tests without skips**, including 15 animal-workflow tests and three deposited-vector tests. Animal behavioral fitting evidence uses synthetic fixtures only. The real Fmr1 processed-qEEG benchmark fits genotype classifiers across two experiments with 43 unique animals; no independently timed behavioral outcomes or raw EEG were supplied by those source CSVs. Primary/reverse animal balanced accuracies are 57.95%/71.72%. [autism_animal_verification.json](research/autism_animal_verification.json) records that initial extension, while older sections below retain their dated scope.

## Extension, 12 October 2026 UTC (13 October in Israel)

Full-recording analysis and the separate annotation-dependent event workflow are documented in [EXPANDED_HUMAN_COHORT.md](research/EXPANDED_HUMAN_COHORT.md), [EVENT_PREDICTION.md](research/EVENT_PREDICTION.md) and [PRE_EVENT_SIGNAL_EVIDENCE.md](research/PRE_EVENT_SIGNAL_EVIDENCE.md). The event module has engineering checks for past-only windows, reviewed follow-up, timing boundaries, co-occurring targets, subject separation and model reload. These are synthetic fixtures; RepOD still supplies no empirical event-prediction evaluation. The 12 October audit below is retained as a dated record of its then-current code.

## Implementation audit, 12 October 2026

Code reviewed: `research/run-public-psychiatric-cohort` at `23936944186ed92bf7f123ada3b6fb50036585df`, package version 0.1.0. The local run passed **84 tests, with no failures, errors or skips**, followed by `scripts/verify_example.py`, the real-EEG `analyze` CLI and the paper `score` CLI. This audit changes documentation only. [VERIFICATION.json](VERIFICATION.json) records commands, results, environment and evidence scope; the [technical brief](TECHNICAL_BRIEF.md) provides a short shareable account.

| Component | Implemented and runnable | Automated evidence and limits |
| --- | --- | --- |
| CSV import, unit conversion and quality checks | `recording.py`, `quality.py`; explicit clock/units, missing-data handling, SQI and EMG flags | `test_inputs_quality.py`; EMG flags do not clean or necessarily reject contaminated spectral epochs |
| FFT-based qEEG | `signal.py`, `analysis.py`; Welch band power, relative power, delta/alpha ratio, SEF95, entropy | `test_signal.py`: analytical sine power/units, band partition, ratios and invalid-input checks; `test_real_data.py`: actual VitalDB samples |
| Experimental auto-bicoherence and alpha PLV | Multi-segment auto-bicoherence; montage-gated pair PLV | Bounds/insufficient history, locked phases, missing pair and unverified mapping tests; no calibrated bifrontal index or general connectivity validation |
| SOI/RDI arithmetic and personal baseline | `indices.py`; strict components, duration and context; CLI checks identity | `test_indices.py`: formula direction, missingness, SD, duration union and context; no complete acquired 24-hour psychiatric calibration fixture |
| State-label SVM (`train`) | `train_research_model`; linear SVC with subject-held-out predictions | `test_ml.py` uses explicitly synthetic labels and checks schemas/splits; no clinical episode model or longitudinal dataset is shipped |
| Diagnosis-group SVM (`train-groups`) | `train_group_model`; group prediction and subject voting | `test_ml.py` synthetic checks plus historical public-cohort CI; group discrimination does not establish episode prediction |
| EDF conversion and electrode comparisons | Optional MNE import, cohort conversion, regional feature subsets | `test_convert.py` covers algebra and a mocked raw-file reader; `test_cohort.py` mocks training for comparison routing; actual EDF conversion and real SVM fitting were exercised by the public-cohort CI job |
| Mouse processed ERP/gating analysis | `model_organism.py` and comparison runner | `test_model_organism.py` plus historical source-verified CI; stimulus/genotype outcomes are not meltdown or onset labels |
| Reports | CLI produces HTML, PNG, CSV and JSON | Real-data CLI smoke run succeeded; no browser/render automation or measured real-time device latency |
| LSTM/Transformer, microstates, CSD, event gating from raw EEG, autonomic fusion, learned medication correction, clinical alerts | Not implemented as PSI estimators/predictors | No automated validation of these proposed components; an input field or roadmap entry is not an estimator |

The [offline CI run](https://github.com/netanelcyber/psi-qeeg-bis/actions/runs/37900411420) at the audited commit completed successfully on Ubuntu 24.04 and Windows with Python 3.10, 3.12 and 3.13. Its [public-cohort job](https://github.com/netanelcyber/psi-qeeg-bis/actions/runs/37900411420/job/113721377990) also completed the source-verified 28-person EDF benchmark and the processed mouse comparison. Job logs were inspected during this audit; they report 821 accepted human epochs and subject balanced accuracies of 64.29%, 46.43% and 60.71% for full scalp, left and right regional sets, respectively. A [separate cohort workflow](https://github.com/netanelcyber/psi-qeeg-bis/actions/runs/37900411457) also succeeded at that commit.

Those are **previous completed runs**, not fresh cohort downloads or training in this audit. The local audit did not install MNE or download the full source cohorts. Unit tests exercise SVM fitting with synthetic tables; source-verified cohort evidence comes from the inspected CI logs and retained aggregate results. Passing either kind of check does not establish clinical validity.

## What the real-data check establishes

The real-data check uses an attributed, public VitalDB case-1 BIS recording. It verifies the waveform clock, microvolt representation, excerpt checksum, SQI/EMG alignment, complete epochs, deterministic spectral feature extraction, report generation, and explicit absence of unsupported PSI composites. This is **engineering validation on real EEG**, not validation of the paper's psychiatric hypotheses.

The excerpt was selected for a complete 120-second window with recorded SQI ≥80 and EMG <40 dB. That selection is disclosed in its data card. It is not an estimate of whole-recording signal quality or evidence of diagnostic accuracy. Sensitivity to low quality, missing samples and muscle activity is tested separately with clearly synthetic fixtures.

## Data needed for psychiatric evaluation

1. Prospective consented observational acquisition with recorded sensor model, scalp sites, reference, sampling rate, impedances, medication context and synchronized events/autonomic signals.
2. Participant-directed annotation for overload/distress, and independent clinician-adjudicated psychosis-related assessments. Permit overlap and indeterminate states. Do not provoke meltdowns or alter treatment to obtain labels.
3. Personal awake calm calibration with a prespecified quality/duration protocol; no use of held-out subjects' future episode measurements when fitting calibration.
4. A reproducible specification of gating, alpha bursts, posterior aggregation, CSD, bifrontal bicoherence, asymmetry and GSR/PPG fusion before looking at final outcomes.
5. Subject-separated, session-aware training/evaluation with a locked external or prospective cohort; preserve medication and device stratification and account for temporal overlap.
6. Report event-level sensitivity, false alerts per hour, precision, calibration, lead time and confidence intervals. Compare BIS-only ablations against the expanded montage and clinical baseline models.

## Scope of the optional trainer

Both trainers use `StandardScaler` and `SVC(kernel="linear", class_weight="balanced")`, with default `C=1.0` and no probability calibration. Scaling is fitted within each leave-one-subject-out training fold. They write held-out predictions, evaluation JSON and a final `joblib` model fitted to the supplied table; the final refit is not a separate evaluation.

`train` accepts independently annotated research states and verifies label provenance, subject/session/epoch identifiers and class coverage. Its schema is not evidence that those clinical labels have been acquired. `train-groups` predicts dataset-documented ASD/PSY/HC groups, uses accepted epochs when that column exists, aggregates by subject majority vote and checks complete site/group confounding when site is supplied. These are different research targets and input contracts. The available public human benchmark contains PSY/HC only, not ASD or pre-episode labels.

The trainers cannot establish that labels are truthful or that upstream calibration, temporal overlap, feature selection and medication/device effects were handled without leakage. Those remain protocol/review requirements. `analyze` never loads either exported model; no pretrained psychiatric-state classifier is shipped with the surgical EEG example. Neither accuracy nor an SVC decision identifies a calibrated risk probability.

## Requirements before developing LSTM/Transformer

1. Define the prospective task and reference labels: participant-directed overload annotations, independently adjudicated psychosis-related outcomes, overlap/indeterminate states, onset timestamps and censoring. Diagnosis group, surgical depth and mouse genotype cannot substitute for these outcomes.
2. Collect ethically approved, synchronized longitudinal EEG and relevant auxiliary/context data across subjects and sessions. Specify channel/reference/device provenance, medication context, artefact/QC rules, missingness masks and awake calibration available before the prediction cutoff.
3. Prespecify the input sequence, observation window, sampling/epoch stride, prediction horizon and loss/targets. The article's suggested 5–30-minute history does not demonstrate prediction over hours or days. Quantify cohort size and event coverage for the task rather than choosing a model from an unsupported sample-size rule.
4. Implement and version an actual model, training loop, reproducible seeds/configuration, saved weights and inference interface. Keep feature extraction, scaling, calibration, tuning and overlapping windows inside subject/session/time-respecting development splits. Use participant-level aggregation or weighting; do not treat correlated epochs as independent people.
5. Evaluate against simple temporal and SVM baselines on a locked external/prospective cohort. Report discrimination and calibration, event-level sensitivity, false alerts per hour, confidence intervals and warning lead time; assess medication/device strata, missing-data behaviour, abstention and deployment latency.

None of these steps supplies an implemented sequence model in the audited checkout. Automated tests for tensor shapes, timestamp cutoffs, sequence split leakage, training/inference consistency and saved-model reload would be necessary engineering checks after implementation; they would still be separate from prospective clinical validation.

## Deliberately deferred

Commercial-device serial streaming requires an actual documented export interface; no undocumented pinout or homemade amplifier is included. Clinical alert thresholds, CSD estimation, event gating, automatic baseline medication corrections and a longitudinal LSTM/Transformer require further specifications and data. A finite research score never authorizes diagnosis, restraint, sedation or medication changes.

