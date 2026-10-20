# WESAD -> PSI wearable CSV (offline research-only adapter)

This adapter converts **one synchronized WESAD subject** into the canonical long
CSV accepted by the optional wearable ingestion in PR #6 and a companion JSON
quality report. It does **not** run EEG analysis, train a classifier, create
meltdown/psychosis event labels, emit notifications or make clinical predictions.

## Dataset, permission, and provenance

Download WESAD from the [University of Siegen original dataset page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html).
The original archive is approximately 2.5 GB compressed. The owners permit
**scientific, non-commercial** use with attribution. Their policy is not a general
open-source/commercial license. No WESAD recordings, derived participant measurements
or trained models are redistributed by this repository. Review license suitability
before incorporating the dataset into any commercial workflow.

**Cite:** Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven,
K. (2018). *Introducing WESAD, a multimodal dataset for Wearable Stress and
Affect Detection*, Proceedings of ICMI 2018, DOI: 10.1145/3242969.3242985.

The dataset provides 15 laboratory subjects with labels for baseline (1), stress
(2), and amusement (3), plus other protocol conditions. **None** is an annotated
autistic meltdown or psychotic episode. These labels are **not** copied into the
wearable CSV or treated as clinical warning events.

## Run

Extract the authentic archive manually. From `meltdown-alert/`:

```bash
python -m pip install -r requirements.txt pytest
python scripts/wesad_to_wearable.py \
  --subject-pkl /path/to/WESAD/S2/S2.pkl \
  --output-csv /path/to/outputs/S2-wearable.csv \
  --qc-json /path/to/outputs/S2-qc.json \
  --step-sec 10 --max-seconds 240
```

Remove `--max-seconds 240` to process the entire subject. Repeat separately for
other subjects; never merge their time series into one replay. Original `.pkl`
files use Python pickle, which **can execute arbitrary code**: only unpickle
trusted files from the original archive. Keep WESAD data out of Git.

`timestamp` is a **session-relative number of seconds** since WESAD's common
synchronized start (`t=0`), not a real UTC timestamp. Every derived measurement
is stamped at the **right end of its completed support window**. No signal is
re-based to its first nonmissing measurement. Duration agreement is checked
against WESAD's 700-Hz protocol-label clock (1 s tolerance for rounding);
this coarse check does **not** prove pointwise synchronization or correct for
drift, undocumented sensor gaps or collection delays.

## Sensor mapping

| WESAD source | Rate | Canonical output | Derivation |
|---|---:|---|---|
| Chest `ECG` | 700 Hz | `heart_rate_bpm` | Plausible R-R intervals from causal QRS-energy detection, 60 s |
| Chest `ECG` | 700 Hz | `hrv_rmssd_ms` | RMSSD from adjacent accepted R-R pairs, 60 s; never from HR |
| Chest `Resp` | 700 Hz | `respiration_bpm` | Causal bandpass and breath-peak spacing, 60 s |
| Wrist `TEMP` | 4 Hz | `skin_temperature_c` | Median over previous 5 s |
| Wrist `ACC` | 32 Hz | `movement_g` | Dynamic acceleration RMS over previous 5 s, 64 raw counts/g |
| Wrist `BVP` | 64 Hz | Not used | No PPG pulse/IBI validation; avoids guessing HRV from BVP |

ECG and respiratory filters are *forward-only* (`sosfilt`, `lfilter`), with
no `filtfilt`, centered moving windows, interpolation, or observations after
`timestamp`. ECG uses a 5–20 Hz QRS-energy detector, a 0.30 s refractory
interval, plausible R-R range 0.33–2.0 s, and rejection of RR intervals over
25% away from the window median. RMSSD needs at least 15 consecutive usable
RR differences. Respiratory peaks come from a 0.07–0.70 Hz forward bandpass,
with at least 3 plausible cycles. These are **heuristics**, not benchmarked
medical-grade beat or breath annotations. Inspect raw signals and compare
against expert-reviewed beat annotations before calling them physiologically
accurate. Motion >0.30 g RMS lowers the ECG estimate's QC score; it does not
silently correct it. Stationary accelerometer gravity/orientation is removed
using samples from the previous 5 s only. Other motion artifacts may remain.

Each CSV row is `timestamp,metric,value,quality,sensor`. The numeric quality
score is an **adapter-derived engineering heuristic**, *not* a vendor confidence
or clinical probability. Scores below 0.6 are emitted as **blank values**, with
the quality and rejection reason in the JSON report. The existing PR #6 parser
interprets a blank as an explicit invalid reading; it cannot backfill the last
good one. Missing source channels are not synthesized or zero-filled and appear
as `missing_sensor` in the report. ECG/respiratory channels containing any
non-finite sample in the supporting window are rejected rather than interpolated.

The report includes the number of expected windows, emitted/accepted windows,
**per-metric usable coverage** (`valid_windows / total_windows`), reasons for
rejection, sensor availability, rates and derivation parameters. Coverage does
not prove peak annotations correct. No true event labels, lead-time scores,
false-alarm rates or alert counts are estimated. This script exports only
measurement and QC records even if the repository's EEG alert command exists.

## Verification and constraints

```bash
python -m pytest tests/test_wesad_adapter.py -q
```

The checked-in tests generate **synthetic WESAD-shaped pickles** and verify
schema, causal-window isolation, missing channels, nonfinite artifacts and
clock/duration guardrails. They **do not** establish real WESAD data quality.
A real-data validation run requires downloading one authentic subject (e.g.
S2.pkl), then preserving its `S2-qc.json` separately with the dataset version,
exact source path/checksum, execution command, and observed coverage. Do not
present synthetic test coverage as observed WESAD coverage.

**Integration boundary:** PR #6's combined detection pipeline still requires
valid EEG. Do not pass WESAD data through a fabricated/synthetic EEG stream and
report resulting alerts as real-subject outcomes. This adapter intentionally
has no EEG-free alert path. An actual wearable-only model needs independently
labeled human behavioral events and prospective, subject-held-out validation.
