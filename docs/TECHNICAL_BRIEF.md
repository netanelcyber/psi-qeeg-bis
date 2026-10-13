# PSI: technical implementation brief

**Netanel Stern | 12 October 2026 | Research software 0.1.0**

This is the initial audit snapshot at the pinned code below. See the [later technical update](research/TECHNICAL_UPDATE.md) for the fresh full-recording run and annotation-dependent event extension.

Scope: `netanelcyber/psi-qeeg-bis`, branch `research/run-public-psychiatric-cohort`, code commit [`23936944186ed92bf7f123ada3b6fb50036585df`](https://github.com/netanelcyber/psi-qeeg-bis/tree/23936944186ed92bf7f123ada3b6fb50036585df).

**Current status.** PSI is an offline research workbench with FFT-based qEEG, experimental waveform descriptors, mathematical SOI/RDI scoring and optional linear SVM training/evaluation. There is no implemented or trained LSTM/Transformer, sequence-model inference or clinically validated PSI predictor in this checkout. The current article's section 7.2 proposes future state detection and trajectory prediction; it does not report trained PSI models.

## FFT and qEEG processing

CSV input requires an explicit uniform sample clock, amplitudes and channel metadata. Optional MNE readers support EDF/BDF; conversion also supports EEGLAB/FIF. Data are converted once to microvolts. The default analysis uses non-overlapping four-second epochs. Missing samples, extreme amplitude and flat channels cause rejection; BIS-family input additionally requires SQI. EMG contamination is flagged, not automatically removed from spectral estimates. These are configurable engineering quality rules.

Welch PSD uses two-second Hann segments, 50% overlap and constant detrending, giving a nominal 0.5 Hz frequency grid. An optional 50/60 Hz notch operates before spectral estimation when below Nyquist. Exact-edge trapezoidal integration yields band powers in microvolt-squared for delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30 and gamma 30–40 Hz. Outputs include relative powers, delta/alpha ratio, normalized spectral entropy and SEF95 over 0.5–40 Hz. This full band set requires a sampling rate above 80 Hz. FFT estimates waveform properties; it does not identify a psychiatric state.

Experimental squared auto-bicoherence uses real FFTs from an ensemble of overlapping two-second Hann segments, testing coupling between 5–15 Hz frequencies whose sum is at most 30 Hz. The analysis screens the complete past 60-second history as one window and requires that window to pass QC without warnings. It does not require every constituent four-second epoch to pass separately. Insufficient history returns missing. This single-channel descriptor is neither proprietary BIS nor the article's calibrated bifrontal component. Alpha PLV uses a Butterworth bandpass and Hilbert phase for documented frontal/parietal or frontal/temporal pairs only with verified scalp/reference metadata.

Quatro has four physical contacts but two EEG derivations. The supplied VitalDB excerpt has two channels with unknown physical mappings; channel numbers cannot establish left/right scalp sites or posterior coverage. Regional Fp1/F7 and Fp2/F8 comparisons retain the source reference and approximate coverage only. An optional `bis_proxy` is an uncalibrated spectral descriptor, not a commercial BIS measurement.

## What the SVM actually does

Both workflows use training-fold `StandardScaler` followed by a linear `SVC`, balanced class weights and default `C=1.0`. Leave-one-subject-out folds keep each person's epochs together. Outputs include held-out predictions, evaluation JSON and a final refit exported to `joblib`; no probability calibration is implemented.

| Workflow | Reference target | Evidence boundary |
| --- | --- | --- |
| `train` | Independently supplied baseline, sensory loading, pre-meltdown, pre-psychotic or indeterminate labels | Synthetic schema/fitting tests; no acquired episode-labelled training cohort or pretrained episode model is shipped |
| `train-groups` | Dataset-documented ASD/PSY/HC group, with subject majority voting | Executed public benchmark includes PSY/HC only; diagnosis discrimination is not onset prediction |

The source-verified RepOD benchmark uses 28 subjects, first 120 seconds each, 821 accepted four-second epochs and fixed spectral features. Inspected historical CI logs reproduce subject balanced accuracies of **64.29%** (19 channels), **46.43%** (Fp1/F7) and **60.71%** (Fp2/F8). These small-cohort point estimates lack confidence intervals or external validation. There are no ASD, meltdown or prospective onset outcomes in that cohort. `analyze` does not load either SVM and always returns `classification = indeterminate`.

## Scores, tests and limits

SOI/RDI implement the article's clipped, personal mean/SD normalization with five components per index and default equal weights. RDI decreases for deviations in either direction. Missing components or failure of the proposed 24-hour awake calibration rule prevent a composite; no components are silently reweighted. Baseline fitting counts overlapping intervals once and checks context. Weight learning, complete raw-EEG component extraction and medication-effect correction are not implemented. The real EEG demo therefore returns unavailable SOI/RDI; the illustrative paper inputs separately yield SOI 100.

On 12 October, **84 automated tests passed without skips**, the VitalDB checksum/dimension check passed, and the analysis/report and scoring CLIs ran. Tests cover signal arithmetic, quality/missingness, calibration, SVM contracts, montage selection and processed mouse analysis. Historical CI also ran actual public EDF conversion/SVM fitting. Local tests use real VitalDB EEG and explicitly synthetic ML fixtures; the full human/mouse cohorts were not newly downloaded in this audit. This establishes engineering checks, not clinical validity. See [validation evidence](VALIDATION.md).

## Requirements for future LSTM research

1. Acquire consented longitudinal, synchronized EEG/context data with independently annotated events, overlap/indeterminate states and adjudicated psychosis-related outcomes.
2. Define the sequence features, timestamps/missingness masks, observation window, stride, prediction horizon and targets. A proposed 5–30-minute history is not evidence for predictions over days.
3. Implement and version the architecture, training loop, configuration, learned weights and inference path. Keep preprocessing/calibration and overlapping windows within subject/session/time-respecting splits.
4. Compare with simple temporal/SVM baselines on locked external/prospective data; assess calibration, event sensitivity, false alerts/hour, lead time, uncertainty, medication/device effects and abstention.

Microstates, raw-event gating, CSD, autonomic fusion, on-chip streaming and clinical alert rules also require specifications and validation. There is no demonstrated individual diagnostic or treatment capability.

**Primary evidence:** [signal estimators](https://github.com/netanelcyber/psi-qeeg-bis/blob/23936944186ed92bf7f123ada3b6fb50036585df/src/psi_qeeg/signal.py), [analysis](https://github.com/netanelcyber/psi-qeeg-bis/blob/23936944186ed92bf7f123ada3b6fb50036585df/src/psi_qeeg/analysis.py), [SVM trainers](https://github.com/netanelcyber/psi-qeeg-bis/blob/23936944186ed92bf7f123ada3b6fb50036585df/src/psi_qeeg/ml.py), [cohort CI job](https://github.com/netanelcyber/psi-qeeg-bis/actions/runs/37900411420/job/113721377990), [current article](https://docs.google.com/document/d/1sH4rkEbuDHridgQbBHvTpPzudxVAtggyEUS-7IO91ao/edit). Historical PDF/current-article distinctions are in [article alignment](ARTICLE_ALIGNMENT.md).
