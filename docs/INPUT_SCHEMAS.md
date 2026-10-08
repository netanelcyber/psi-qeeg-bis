# Input schemas

## Raw EEG CSV

One row per evenly spaced sample, `time_s` in seconds, with missing waveform samples represented by blank/NaN values. Do not drop missing rows. A simultaneous monitor value such as BIS is a numeric auxiliary measurement, not an EEG amplitude.

```json
{
  "sampling_rate_hz": 256,
  "eeg_unit": "uV",
  "eeg_columns": ["Fp1", "Fp2", "T7", "T8", "P3", "P4", "O1", "O2"],
  "channel_locations": {"Fp1":"Fp1", "Fp2":"Fp2", "T7":"T7", "T8":"T8", "P3":"P3", "P4":"P4", "O1":"O1", "O2":"O2"},
  "reference": "documented_common_reference",
  "reference_verified": true,
  "sensor": {"family": "research_eeg", "model": "record_actual_model"},
  "recording_context": "awake_observational_research",
  "subject_id": "participant-001"
}
```

The example above is a schema illustration, not evidence that this montage has been acquired. Allowed EEG input units are `V`, `mV`, `uV`/`µV`; all features are computed after a single conversion to microvolts. Changing a label does not establish electrode identity. `reference_verified` may be set only from actual acquisition records. The importer requires uniformly spaced times matching the declared rate.

Optional aligned auxiliary columns: `BIS`, `SQI`, `EMG`, `SEF`, `SR`, `TOTPOW`. For BIS family recordings, missing SQI blocks an epoch. EMG dB ≥40 is a configurable research warning, not a universal clinical threshold.

EDF/BDF uses an explicit sidecar selecting EEG channels. MNE supplies calibrated physical amplitudes in volts; the importer converts them to microvolts. Sensor/reference details still need independent documentation. EDF support is optional; the shipped real-data acceptance test uses CSV.

## Complete article features

`score` expects JSON with `subject_id`, `medication_profile_id` and a `values` object:

| Key | Meaning | Deviation used |
| --- | --- | --- |
| `alpha_variance` | Defined posterior alpha burst-duration variance | Increase |
| `gating_efficiency` | Protocol-defined stimulus/spontaneous gating ratio | Decrease |
| `posterior_gamma` | Defined posterior gamma power | Increase |
| `frontoparietal_plv` | Defined frontal/parietal PLV aggregation | Decrease |
| `autonomic_surge` | Defined GSR/PPG-derived measure | Increase |
| `temporal_csd` | Defined temporal beta/gamma CSD measure | Absolute deviation |
| `delta_alpha_ratio` | Defined global delta/alpha ratio | Absolute deviation |
| `bifrontal_bicoherence` | Defined bifrontal bicoherence estimator | Absolute deviation |
| `hemispheric_asymmetry` | Defined bilateral power-asymmetry measure | Absolute deviation |
| `frontotemporal_coherence` | Defined frontal/temporal connectivity | Absolute deviation |

All measures must use the same physiological units and operational definitions as their baseline. A calibration file stores `subject_id`, `medication_profile_id`, `duration_s`, `context`, and `SOI`/`RDI` objects containing `components[key].mean` and `.sd`. `context` must identify `awake_personal_baseline`; the supplied paper fixture is separately labelled `illustrative_paper_example`. Finite but insufficient calibration prevents a composite score.

## Labelled feature table

Required columns for `train`:

`subject_id, session_id, epoch_id, research_label, label_source, recording_context` plus explicitly selected numeric feature columns.

Labels: `baseline`, `sensory_loading`, `pre_meltdown`, `pre_psychotic`, `indeterminate`. Label sources: participant, clinician or consensus. Collect appropriate clinician adjudication for psychosis-related labels. Labels describe a study's independently collected reference annotations; EEG values do not create labels. At least three unique subjects and two classes are required. Every training fold must contain all classes.

The trainer refuses perioperative/anesthesia rows. It rejects duplicate subject/session/epoch records, missing identifiers, unknown provenance and missing feature values. The scaler is fitted separately inside each training fold. An epoch random split is not used. Final training metrics do not establish prospective clinical performance.

For `calibrate`, rows additionally need `time_s`, `epoch_end_s` and the component columns. Only that subject's rows explicitly labelled `baseline` contribute. `recording_context` must document `awake_personal_baseline` or `awake_observational_research`; surgical context is rejected. Use a documented common time coordinate; overlapping intervals count once. If supplied, session and medication-profile identifiers must remain constant. The output preserves the medication identifier, or explicitly marks `not_recorded` when it was not supplied.

