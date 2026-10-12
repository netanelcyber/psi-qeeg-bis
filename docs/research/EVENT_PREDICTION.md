# Annotated event prediction: research workflow

This module's current targets and provenance contract concern human research outcomes. The separately implemented [animal qEEG/behavioral workflow](AUTISM_ANIMAL_MODEL.md) has its own outcome namespace and adapter, and still requires independently annotated real continuous data for empirical behavioral fitting. Animals must not be assigned human meltdown/psychosis labels to make these commands run.

The event module implements causal feature preparation, two binary research SVMs and past-only inference. **No event model has been trained on real data in the RepOD experiment.** More resting-state epochs do not create meltdown or psychosis-onset outcomes. The existing diagnosis-group classifier is not accepted by `predict-events`.

## Input and independent reference outcomes

Epoch CSV requires `subject_id, session_id, time_s, epoch_end_s, accepted, recording_context`, plus selected numeric EEG features. Epochs must have one documented duration, non-overlapping times and known awake context. Only complete finite accepted past windows contribute. Sessions are never joined across gaps. The extended RepOD runner now supplies session and epoch identifiers, but no event annotations.

`events.csv` requires the columns in [the empty schema](../../examples/event_schema/events.csv): event type `meltdown` or `psychosis_onset`, onset/offset seconds on the recording session's clock, an independently assigned event ID, source, label provenance and `review_status=confirmed`. Meltdown annotation can be participant/clinician/consensus based. Psychosis-onset annotations require clinician/consensus adjudication. These provenance fields are protocol declarations; software cannot establish that human assessments are correct.

`coverage.csv` requires [the separate empty schema](../../examples/event_schema/coverage.csv). Each interval must explicitly record independent clinician/consensus review of **both** event types (`event_types_reviewed=meltdown;psychosis_onset`) with `review_status=complete`. An empty event file without complete follow-up is not a set of negative examples. Resting-state diagnosis, an outlying band power, a smartwatch's generic relapse flag, distress or a stimulus-response label cannot substitute for the requested outcomes. Define meltdown and psychosis onset prospectively with participants and clinicians before annotation.

## Causal dataset and modelling contract

At cutoff `t`, descriptors use only accepted epochs in `[t-history_s,t]`: per-feature mean, sample SD and linear time slope. Minimum two epochs are required. No future signal, interpolation or subject/session mixing is used. A positive target requires a confirmed onset in `[t+min_lead_s,t+horizon_s]`. Observation coverage must span the entire history and future horizon. Current/recent events intersecting the history and onsets closer than the minimum lead are excluded. Simultaneous future meltdown and psychosis outcomes are retained as two positive targets.

The defaults (60-second history, 300-second horizon and four-second minimum lead) are **illustrative experimental settings**, not demonstrated clinical warning times. A longer recording is not longitudinal clinical follow-up. Choose and lock the task, window, horizon, event definitions and missing-data rules before final evaluation.

Preparation writes a fingerprinted window table and manifest. Training requires that matching manifest, both positive and negative outcomes for each target, at least three subjects and class coverage in every subject-held-out training fold. Each target uses training-fold StandardScaler and a fixed linear SVC with balanced class weights. The final all-data refit is saved separately from held-out evaluation. No LSTM or probability calibration is introduced.

Metrics currently describe **windows**, not clinical events or alert performance. Overlapping histories are correlated. The pipeline requires external/prospective validation, participant-level uncertainty, event sensitivity, false alerts per observed hour, refractory/abstention rules, lead-time evaluation and medication/device/context analyses before clinical interpretation. Inference cannot establish that no event is already active; it returns research predictions and margins with explicit unvalidated status, not a medical alert.

## Commands after genuine annotated data are supplied

```bash
python -m psi_qeeg prepare-events data/epochs.csv \
  --events data/events.csv --coverage data/coverage.csv \
  --features Fp1__alpha_power_uv2 Fp1__delta_alpha_ratio \
  --history-s 60 --horizon-s 300 --min-lead-s 4 --out outputs/event_dataset
python -m psi_qeeg train-events outputs/event_dataset/windows.csv \
  --manifest outputs/event_dataset/dataset_manifest.json --out outputs/event_model
python -m psi_qeeg predict-events data/new_epochs.csv \
  --model outputs/event_model/event_models.joblib --out outputs/event_inference
```

The empty schemas are not data. RepOD cannot run these training steps with its native labels. Automated tests use clearly named synthetic fixtures solely to verify timing, censoring, co-occurring targets, leakage prevention, model reload and CLI contracts. Their fitting metrics are not evidence of clinical prediction.

## Public-source review

Checked 12 October 2026 UTC (13 October in Israel): [RepOD](https://doi.org/10.18150/repod.0107441) documents diagnosis groups; [Mendeley's ASD/M-CHAT dataset](https://doi.org/10.17632/4zdf2h8rzw.1) documents diagnostic/screening features; [CALMED](https://arxiv.org/abs/2307.13706) documents emotion-labelled audio/video features; [e-Prevention research](https://doi.org/10.1038/s41598-025-03856-1) uses wearable activity/HRV and daily relapse labels, with a broader relapse definition and source-request access. These inspected sources do not supply the combined EEG and independently timed meltdown/psychosis-onset annotations required here. This bounded review is not a claim that no suitable dataset exists anywhere.
