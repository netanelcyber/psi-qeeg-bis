# Real public EEG signal-pipeline validation

## Scope and interpretation

This experiment exercises `psi_qeeg.recording.read_edf` → `psi_qeeg.analysis.analyze`
→ `psi_qeeg.signal.spectral_features` on a real BioSemi BDF recording. Synthetic
analytical comparisons check numerical estimators. Neither experiment trains a
psychiatric model or measures diagnostic accuracy, behavioral sensitivity, false
alerts or prediction lead time.

The earlier PR #4 validator used a separate 1–45 Hz MNE/SciPy extractor. Its green
CI runs are **not evidence that the repository pipeline was validated**. Those
features used different bands, preprocessing and integration boundaries, so they
must not be compared as the same estimators or pooled with the corrected output.

## Immutable input and provenance

- OpenNeuro **ds006780, version 1.0.0**, DOI
  [10.18112/openneuro.ds006780.v1.0.0](https://doi.org/10.18112/openneuro.ds006780.v1.0.0).
- Dataset Git commit: `799d1502296ba5f74033734e149160c3d333e470`.
- Object: `ds006780/sub-10003/eeg/sub-10003_task-Restingstate_run-01_eeg.bdf`.
- S3 version ID: `m3WdGUp1FxE71ZuQgwWDztDdgUfMK2VI`.
- Size: **6,970,880 bytes**.
- SHA-256: `8b582451f21969960652931bcb880d55ff0da456b23fc5f6c872d0247ca8cc57`.

The hash and size come from the dataset snapshot's
[git-annex pointer](https://github.com/OpenNeuroDatasets/ds006780/blob/799d1502296ba5f74033734e149160c3d333e470/sub-10003/eeg/sub-10003_task-Restingstate_run-01_eeg.bdf),
and were verified against the downloaded S3 object. The downloader requests exactly
one object version, with no AWS listing/query or first-match fallback. A missing
version, byte-size discrepancy or checksum mismatch fails validation; a changed
recording is never silently substituted. Existing cached files are also checked.

The manifest, original BIDS EEG/channels/events sidecars and dataset description
are retained in `research/meltdown_forecast/`. Sidecar hashes are verified before
EEG parsing. Dataset authors: Sophie Molholm, Theo Vanneau and John J. Foxe;
dataset license: CC0. This is a raw resting-state recording, not a BIS recording.
The supplied events identify recording/task starts, not meltdown or psychosis
onsets. Diagnosis/cohort membership is not an event annotation.

## Processing specification

The first six documented EEG channels are fixed explicitly: **Fp1, AF7, AF3, F1,
F3 and F5**. No auxiliary channel is relabelled as EEG. The public reader checks
channel names/types and converts MNE volts to microvolts once. Acquisition
reference remains unverified, so no PLV/connectivity result is emitted.

The analyzer uses non-overlapping **4 s** epochs, the production QC policy, a
60 Hz notch, and two-second Hann Welch segments with 50% overlap, constant
detrending and density scaling. No separate 1–45 Hz bandpass is applied. Band
limits come exclusively from `psi_qeeg.signal.BANDS`:

| Band | Hz |
| --- | --- |
| Delta | 0.5–4 |
| Theta | 4–8 |
| Alpha | 8–13 |
| Beta | 13–30 |
| Gamma | 30–40 |

All powers integrate to exact, interpolated endpoints using
`scipy.integrate.trapezoid`, including on non-aligned frequency grids. Adjacent
bands partition the full **0.5–40 Hz** integral. The same exact endpoints also
bound SEF95. Integration is compatible with NumPy 1.24 and later.

BioSemi electrode DC offsets in this file exceed the zero-referenced amplitude
threshold. This experiment explicitly selects
`QualityPolicy(amplitude_reference="epoch_mean")`: the existing 300 µV limit is
applied to deviations from each channel's epoch mean. This only changes the QC
amplitude reference; raw samples, reference derivations and spectral estimators
are retained. The default policy continues to use zero. Artifact, flat-signal
and missing-sample rejection still apply. This selectable engineering policy is
not a clinically validated artifact-removal method.

No SQI or calibrated EMG monitor values are supplied to this six-channel run.
Every epoch retains `emg_quality_unknown`; an accepted epoch is **not** certified
free of movement or muscle artifact. Gamma power is a waveform measurement,
not proof of cortical gamma activity. Bicoherence remains missing under the
existing history/quality requirements. Undefined ratios and rejected epochs
retain missing values; the validator requires finite core spectral outputs
only for accepted epochs and fails if none are accepted.

## Recorded results

Reference runs and environment versions are retained in
[`real_eeg_validation/`](real_eeg_validation/). Results below are engineering
observations from one selected recording, not population-level findings.

- 512 Hz; 31,744 samples/channel; **62.0 s** of stored waveform.
- 15 complete epochs (60 s analyzed); 1,024 tail samples (2 s) unused.
- 15 accepted, 0 rejected under the explicitly documented centered-amplitude policy.
- Required per-channel spectral features finite; absolute bands partition total
  power and relative bands sum to one within the numerical tolerance.
- Synthetic 10 µV peak sine waves at 2, 6, 10, 20 and 35 Hz each yield the
  analytical 50 µV² in the corresponding band. The 2/10 Hz amplitude mixture
  yields delta/alpha = 0.25. A flat PSD integrates to 39.5 over 0.5–40 Hz.
- Additional regressions test shared band boundaries, non-grid-aligned endpoints,
  the exclusion of 42 Hz, DC offsets, retained missing values, zero accepted
  epochs, checksum failures and corrupted/missing production spectral outputs.
- **SOI and RDI unavailable; psychiatric classification indeterminate.**

Cross-version comparisons and test counts are recorded in
`docs/VERIFICATION.json`. No outcome-labelled psychiatric model was evaluated.
Clinical forecasting validation requires independently annotated event onsets,
participant/session-separated evaluation, false-alert and event-sensitivity
metrics, and prospective confirmation. The present recording cannot supply
those conclusions.

## Reproduce

Use a clean environment and install the project itself:

```bash
# Python 3.10/3.11, NumPy 1.24 family:
python -m pip install -c research/meltdown_forecast/constraints-min.txt '.[edf,ml]'
# Or Python 3.12, NumPy 2.x reference environment:
python -m pip install -c research/meltdown_forecast/constraints-current.txt '.[edf,ml]'

python -m unittest discover -s tests -v
python scripts/verify_example.py
python research/meltdown_forecast/validate_real_eeg.py --download \
  --eeg-file sample_eeg.bdf --out real_eeg_validation
```

Outputs are the actual production feature table, full pipeline summary and a
strict JSON validation report including data hashes, parameters, quality
warnings, analytical comparisons and environment versions. To generate the
normal project report from the same data/parameters:

```bash
python -m psi_qeeg analyze sample_eeg.bdf \
  --metadata research/meltdown_forecast/ds006780_manifest.json \
  --line-hz 60 --amplitude-reference epoch_mean --out outputs/openneuro
```

GitHub Actions runs both dependency environments on relevant pull requests and
pushes to `main` and the PR branch. It also runs the offline analytical suite and
existing VitalDB real-data acceptance check. Only derived reports/features are
uploaded as CI artifacts, with distinct names per dependency environment.
