# Validation status and next experiment

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

The SVM implementation demonstrates a reproducible participant-held-out workflow for a supplied cohort. It checks basic schema/provenance and prevents subject-level train/test overlap. It cannot verify that human labels are truthful or that upstream calibration and feature selection avoided leakage. Those remain protocol/review requirements. No psychiatric model or diagnostic accuracy claim is shipped with the surgical EEG example.

## Deliberately deferred

Commercial-device serial streaming requires an actual documented export interface; no undocumented pinout or homemade amplifier is included. Clinical alert thresholds, CSD estimation, event gating, automatic baseline medication corrections and a longitudinal LSTM/Transformer require further specifications and data. A finite research score never authorizes diagnosis, restraint, sedation or medication changes.

