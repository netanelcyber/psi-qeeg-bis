# PSI qEEG / BIS

A working Python research project based on Netanel Stern's **Toward a Psychiatric State Index: Differentiating Autistic Meltdown from Psychotic Episode Onset Using Real-Time EEG and AI** (2026), with particular attention to **BIS electrodes, raw EEG and montage limitations**.

The repository contains a real, attributed VitalDB BIS EEG excerpt, a reproducible qEEG pipeline, electrode specifications, the article's SOI/RDI arithmetic, an offline report, and optional subject-held-out SVM training for a separately supplied labelled cohort.

**Research status:** signal processing and mathematical implementation. Psychiatric classification has not been clinically validated. The included surgical recording has no meltdown or psychosis labels; its SOI/RDI scores remain unavailable and its psychiatric state is `indeterminate`.

## Run the real-data example

Python 3.10 or later:

```bash
python -m pip install -e .
psi-qeeg analyze examples/vitaldb_case1/recording.csv \
  --metadata examples/vitaldb_case1/metadata.json \
  --out outputs/vitaldb_case1
```

Open `outputs/vitaldb_case1/report.html`. It is self-contained and works offline. Outputs also include `features.csv`, `summary.json` and `qeeg.png`.

The included excerpt is **VitalDB case 1, seconds 332–452**, with 15,360 samples per EEG channel at **128 Hz**. It contains raw `BIS/EEG1_WAV` and `BIS/EEG2_WAV`, plus recorded BIS, SQI, EMG, SEF, SR and total-power numerics. Sampling and amplitude provenance are documented in its metadata and [data card](examples/vitaldb_case1/DATA_CARD.md).

To reproduce the download directly from the source:

```bash
psi-qeeg fetch-vitaldb --caseid 1 --start-s 332 --duration-s 120 \
  --out data/vitaldb_case1 --cache data/vitaldb_cache
```

This downloads complete source tracks, then extracts the requested window. Source-track and excerpt SHA-256 hashes are retained. Numeric values are aligned causally with a five-second expiry; raw waveform gaps are preserved as `NaN` and are never interpolated.

## What BIS electrodes actually provide

The conventional **Quatro sensor has four physical contacts and two measured EEG derivations**, including a shared reference and a ground. It is a unilateral forehead arrangement. A bilateral BIS sensor is a different arrangement.

| Quatro contact | Conventional role | Placement documented in manufacturer material |
| --- | --- | --- |
| 1 | Shared reference | Midline forehead, approximately 5 cm above the nose |
| 2 | Ground | Integral contact on the forehead strip; record placement from the actual sensor IFU |
| 3 | Signal contact | Temple between the outer eye corner and hairline |
| 4 | Signal contact | Directly above and adjacent to the eyebrow |

These product facts do not establish which sensor was used in a particular recording. VitalDB exports two BIS channels but does not identify the sensor model, side or physical derivations in the track listing. The example therefore preserves **unknown** electrode mappings. It does not rename those channels `Fp1` and `Fp2` or compute hemispheric asymmetry from channel numbers.

Read [BIS electrodes and montage coverage](docs/BIS_ELECTRODES.md) for sources, a schematic, reference handling and the expanded PSI montage.

## Implemented features

- Four-second non-overlapping epochs; configurable SQI, amplitude and flat-signal checks; explicit EMG contamination flags.
- Welch absolute band power in µV², relative power, delta/alpha ratio, normalized spectral entropy and SEF95 over 0.5–40 Hz.
- Explicit bands: delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30 and gamma 30–40 Hz. Exact integration boundaries partition total power.
- Experimental squared auto-bicoherence over **60 seconds**, using many two-second segments. Insufficient history returns missing, avoiding the trivial single-FFT result of one.
- Alpha PLV for documented frontal–parietal/temporal pairs only when the scalp sites and reference are verified.
- Explicit-unit CSV input; optional EDF/BDF through MNE. Standard scalp labels are recorded separately from physical channel derivations.
- Personal baseline fitting from explicitly labelled calm epochs and exact section 6.5 normalization with five components per index. Missing components are never silently reweighted.
- Optional research SVM with leave-one-subject-out evaluation and training-fold-only scaling. No model is pretrained on invented psychiatric labels.

This is not an implementation of the proprietary commercial BIS algorithm. Exported EEG cannot reproduce the device's higher-frequency EMG measurement above its 64 Hz Nyquist frequency.

## Reproduce the article's arithmetic

```bash
psi-qeeg score \
  --features examples/paper_calculation/features.json \
  --calibration examples/paper_calculation/calibration.json \
  --out outputs/paper_calculation.json
```

The five supplied SOI values yield **SOI = 100**. The paper's separate trajectory starting at SOI 18 is illustrative; it is not derived from those same five measurements. The example inputs are expressly labelled **illustrative, not real patient data**. See [article alignment](docs/ARTICLE_ALIGNMENT.md).

For your own feature tables:

```bash
psi-qeeg calibrate data/calm_features.csv --subject-id participant-001 \
  --out data/participant-001-baseline.json
```

The article calls for 24 hours of awake personal calibration. Shorter calibration is recorded and blocks composite output. Baseline duration counts the union of epoch intervals, so overlapping windows cannot manufacture 24 hours of coverage. Physiological units and operational definitions must match between baseline and current measurements. The score command checks subject and medication-profile identity.

## EDF/BDF and labelled research

```bash
python -m pip install -e '.[edf,ml]'
psi-qeeg analyze data/recording.edf --metadata data/recording-metadata.json \
  --out outputs/recording
psi-qeeg train data/labelled_epochs.csv \
  --features alpha_power delta_alpha_ratio \
  --out outputs/research_model
```

See [input schemas](docs/INPUT_SCHEMAS.md) and [validation protocol](docs/VALIDATION.md). Raw clinical data and trained models are ignored by Git. The trainer refuses perioperative/anesthesia rows as psychiatric-state training data.

## Verify

```bash
python -m unittest discover -s tests -v
python scripts/verify_example.py
```

Analytical waveforms used by unit tests are explicitly synthetic test fixtures. They are not presented as real EEG or clinical evidence. CI runs the offline test suite; no subject data or network download is needed.

## Create the GitHub repository

With GitHub CLI (`gh`) authenticated and your Git author identity configured:

```bash
python scripts/create_github_repo.py
```

The helper creates a new **private** `psi-qeeg-bis` repository and pushes the reviewed local commit. Choose `--public` explicitly if public visibility is intended. It refuses a dirty checkout and does not overwrite an existing repository. GitHub creation and remote CI are separate from the locally verified build.

## Sources and licensing

- Article derivation: [source fingerprint and implementation notes](docs/ARTICLE_ALIGNMENT.md).
- [Medtronic Quatro sensor](https://www.medtronic.com/en-us/healthcare-professionals/products/patient-monitoring/brain-monitoring/brain-sensors/bis-quatro-sensor.html) and [BIS product support](https://www.medtronic.com/covidien/pt-pt/support/products/brain-monitoring/bis-complete-2-channel-monitor.html).
- [VitalDB overview, track definitions and data-use agreement](https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md), [API](https://vitaldb.net/dataset/?query=api), [PhysioNet dataset](https://physionet.org/content/vitaldb/1.0.0/).
- Lee HC et al. *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients*. Scientific Data 9, 279 (2022). [doi:10.1038/s41597-022-01411-5](https://doi.org/10.1038/s41597-022-01411-5).

Code: MIT. Included VitalDB data: CC BY 4.0 with separate attribution and modification notices. BIS is a Medtronic trademark; this independent project is not endorsed by the manufacturer.

