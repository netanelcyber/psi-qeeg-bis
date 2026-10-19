# Optional wearable file processing

This is a **software integration experiment**. It ingests and aligns physiological
measurements with EEG, reports data quality, and optionally changes the research
arousal index. Successful tests establish engineering behavior only. There are no
human prediction results, validated lead times, or validated clinical thresholds
for this multimodal path. Neither profile diagnoses or distinguishes psychiatric
states. Existing synthetic EEG-only results do not validate wearable fusion.

## Supported inputs

Use one participant and one session per run. Inputs are vendor-neutral UTF-8 CSV
files, with either long or wide rows. Repeat `--wearable-file` for different sensor
metrics. Each metric must have one sensor owner; duplicate/reversed timestamps
within a metric and conflicting owners fail explicitly. An entire metric can be
absent. No missing value becomes a measured zero.

| Canonical metric | Required unit/meaning | Default max age (s) | Default index weight |
|---|---|---:|---:|
| `heart_rate_bpm` | beats/min | 15 | +1 |
| `hrv_rmssd_ms` | precomputed RMSSD, milliseconds | 30 | -1 |
| `respiration_bpm` | breaths/min | 15 | +1 |
| `skin_temperature_c` | skin temperature, degrees Celsius | 30 | 0 (context) |
| `movement_g` | dynamic acceleration RMS in g, gravity removed externally | 2 | 0 (context) |

The signs and fusion weight are **unfitted research assumptions**. Temperature has
no universally appropriate arousal direction, and exercise confounds movement and
heart rate, so temperature/motion are context-only by default. Both are calibrated,
quality-checked and exported, and can be explicitly weighted for an experiment.
There is no exercise correction in this prototype.

HRV must be computed upstream from quality-controlled beat/R-R data. This code
does **not** compute HRV from sampled heart rates, detect R peaks, or infer which
vendor field means RMSSD. Document the upstream window length, minimum beat count,
artifact/ectopic correction, units and algorithm with the dataset. Timestamp a
derived measurement at the **end** of its causal support window, when it becomes
available. A centered/future-looking RMSSD window would leak future information
despite correct replay timestamps. Raw ECG, R-R intervals, respiratory waveforms
and accelerometer axes need a verified preprocessing adapter before this input.

Long format:

```csv
timestamp,metric,value,quality,sensor
0,heart_rate_bpm,70,0.95,chest-A
0,hrv_rmssd_ms,45,0.95,chest-A
0,respiration_bpm,14,0.95,chest-A
0.25,skin_temperature_c,33.1,0.9,temperature-B
0.25,movement_g,0.05,0.9,accelerometer-C
```

Wide format:

```csv
timestamp,heart_rate_bpm,hrv_rmssd_ms,quality,hrv_rmssd_ms_quality,sensor
0,70,45,0.95,0.8,chest-A
5,71,,0.95,0.1,chest-A
```

`quality` and `sensor` are optional. Quality is a normalized number in [0,1],
where higher is better; a `<metric>_quality` column overrides shared quality in
wide input. Quality below 0.6 rejects that sample by default. Missing quality is
exported as `null` with `quality_supplied=false`, and is accepted subject to
finite-value/range checks; it does **not** imply verified sensor confidence.
Translate vendor-specific confidence scales deliberately. Blank/NaN/Inf values
are invalid samples, so a new bad reading immediately replaces a prior good one.
Malformed rows, unmapped column names and invalid supplied quality fail clearly.

## Clocks and synchronization

- Numeric `timestamp` values mean seconds on a **shared session clock**. EEG and
  wearables must use the same origin. Numeric Unix epochs are not auto-detected;
  convert them to session seconds or timezone-aware ISO-8601 first.
- ISO-8601 timestamps must include `Z` or a UTC offset. Supply one common
  `--session-start 2026-10-19T12:00:00+03:00` for all absolute files. Different
  timezone representations of the same instant align identically. Files are
  never aligned by subtracting each file's first sample independently.
- `--wearable-offset-sec -2` corrects a known wearable clock that is 2 s ahead.
  The offset is added after conversion and applies to every wearable file in
  that CLI invocation. For different per-sensor corrections, use
  `load_measurements(path, session_start=..., offset_sec=...)` separately and
  combine the results in `WearableReplay`, or normalize files upstream.
- At each EEG window end, only observations timestamped **at or before** that
  time can be used. The latest sample is held only up to its per-metric maximum
  age. There is no future backfill or interpolation through missing intervals.
- Fused CSV replay requires an EEG timestamp column (`timestamp`, `time`, or `t`)
  and every configured EEG channel in microvolts. EEG gaps do not compress time.
  Windows crossing sample intervals outside +/-20% of `1/fs` are discarded;
  the first complete post-gap window resets detector evidence. Regular sampling
  with the configured nominal rate is still required for Welch analysis. There
  is no automatic clock drift estimation or resampling.
- Synthetic EEG uses its existing sample-derived session clock. EEG-only CSV
  and LSL behavior remains available without any wearable option. Live LSL plus
  wearable files is rejected: a verified common-clock live bridge is not
  implemented. This release does not pair with or stream from an Equivital device.

## Quality, calibration and fusion

Each metric has a separate median/MAD baseline using **distinct observations** in
`[0, baseline_sec)`, with at least 5 QC-passing samples. Reusing a held reading at
many EEG steps does not manufacture calibration samples. A stream that arrives
after calibration remains `uncalibrated`; restart with an adequate calm recording
to calibrate it. Wearable baselines are fixed during the session. Existing EEG
calibration and optional EEG baseline adaptation are unchanged.

The wearable scale is `max(1.4826*MAD, 0.05*abs(median), 1e-6)`, and current
per-metric robust z-scores are clipped to [-5,5]. For calibrated metric set `C`
and currently usable, nonzero-weight metric set `A`:

```text
W = sum(abs(weight[k]) for k in C)
wearable_index = sum(weight[k] * z[k] for k in A) / sqrt(W)
coverage = sum(abs(weight[k]) for k in A) / W
fused_index = (1 - lambda) * eeg_index + lambda * wearable_index
```

`lambda` defaults to 0.25 when a wearable file is supplied. If no weighted metric
is usable, the detector receives the original EEG index. The denominator does
not shrink when a calibrated metric drops out, so remaining sensors are not
silently amplified. Missingness still changes the score distribution: changing
the active metric set resets CUSUM, trend, level and persistence timers before
processing the next score. Refractory times and alert arming are preserved.
Rejected EEG also breaks persistence on the optional fusion path. Wearable data
alone cannot bypass EEG quality or generate an alert. This is not a standalone
wearable detector.

Per-metric QC states are `missing`, `calibrating`, `uncalibrated`, `ok`, `stale`,
`invalid_value`, `out_of_range`, and `low_quality`. Broad acquisition sanity
limits are HR 20–250 bpm, RMSSD 0–300 ms, respiration 1–80 breaths/min, skin
temperature 10–45 C and movement 0–16 g. These are engineering input filters,
**not** clinical normal ranges. A stale reading remains excluded even if its
original quality score was high.

`--steps-jsonl` exports every window, metric value/time/age, supplied quality,
calibration count, z-score, active metric set, coverage and actual fusion weight.
It also records `clinical_validation=false`. EEG's existing `quality` field is
still EEG window quality; wearable QC is separate. No event performance is
estimated for an unlabeled CSV session; synthetic default event times are never
silently assigned to it.

## Reproducible synthetic example

From `meltdown-alert/`:

```bash
python -m pip install -r requirements.txt
python examples/generate_wearable_demo.py /tmp/wearable-demo
python -m eeg_alert --source csv --file /tmp/wearable-demo/eeg.csv --fs 128 --wearable-file /tmp/wearable-demo/cardiorespiratory.csv --wearable-file /tmp/wearable-demo/temperature_motion.csv --steps-jsonl /tmp/wearable-demo/steps.jsonl --report /tmp/wearable-demo/report.json
```

Use a local directory such as `demo-data` instead of `/tmp/wearable-demo` on
Windows. The generator creates 6 minutes of **synthetic** EEG plus HR at 1 Hz,
rolling RMSSD and respiration at 0.2 Hz, skin temperature every 15 s, and movement
at 2 Hz with a 0.25 s phase offset. HR drops out at 250–280 s and respiration
quality is low at 210–230 s. The simulated changes are scripted assumptions.
The demo intentionally supplies no event labels to the detector. Add `--events 5`
only to exercise scoring against the generator's scripted event; this remains
synthetic engineering output and cannot estimate human prediction accuracy.

Optional experiments:

```bash
# Process/report all wearable measurements while retaining EEG-only scoring:
python -m eeg_alert --source synthetic --minutes 6 --events --wearable-file /tmp/wearable-demo/cardiorespiratory.csv --wearable-fusion-weight 0 --steps-jsonl /tmp/wearable-demo/context.jsonl

# Explicitly include contextual temperature/movement in an experimental index:
python -m eeg_alert --source csv --file /tmp/wearable-demo/eeg.csv --fs 128 --wearable-file /tmp/wearable-demo/temperature_motion.csv --wearable-weight skin_temperature_c=0.2 --wearable-weight movement_g=0.3 --wearable-max-age movement_g=3
```

For timezone-aware synthetic files, pass `--session-start` to the generator and
the replay command with the same value. The small checked-in fixtures in
`tests/fixtures/` are also entirely synthetic, not authentic vendor exports.

## Equivital: documented capability versus implemented support

Checked on **2026-10-19**, using vendor-hosted sources:

| Interface | What official documentation supports | Status in this prototype |
|---|---|---|
| eq02+ LifeMonitor measurements | ECG, HR, R-R intervals, respiration, skin temperature, accelerometer/activity | Relevant candidate; no hardware recording tested |
| Equivital Manager | Download/splice SEM files and extract CSV/Excel; configure a small number of SEMs over USB | Canonical CSV import implemented; actual headers/units/quality/timestamps require a verified mapping |
| eqView Professional | Live ECG/respiratory viewing, multiple-sensor synchronization and event markers | Not integrated |
| SDKs | Vendor offers Android, .NET and Java SDKs for embedding LifeMonitor data | No SDK obtained or executed; no Python/Bluetooth/API protocol assumed |
| ADInstruments / VivoSense | Vendor documents software partnerships and export/analysis integration | Not integrated |

Sources:

- [Equivital Training and Research](https://equivital.com/products/research):
  CSV export and software/product descriptions.
- [Training and Research brochure, EQ-001272-BR-1](https://equivital.com/wp-content/uploads/2024/02/Training-and-Research-Brochure-EQ-001272-BR-1.pdf):
  p.2 data/export workflow; p.4 Manager CSV/Excel; p.5 eqView synchronization
  and Android/.NET/Java SDKs; p.6 partnerships.
- [eq02+ LifeMonitor](https://equivital.com/products/eq02-lifemonitor):
  measurement families including HR, R-R interval, respiration and movement.

Before claiming Equivital compatibility, obtain a consented/deidentified export
and its data dictionary: exact product/firmware/software version, headers,
units, missing-value sentinels, timestamp timezone/origin, sample/beat timing,
quality/confidence definitions and channel identity. For HRV, confirm whether
RMSSD is actually exported or must be derived from validated R-R intervals.
For a live bridge, obtain the applicable SDK, license/platform requirements,
documented events and synchronization behavior, then test against actual hardware.
No proprietary SEM decoder or guessed vendor field names are included here.

## Engineering checks and scientific next step

```bash
python -m pip install -r requirements.txt pytest
python -m pytest tests -q
```

Regression coverage includes unchanged EEG-only output, optional fusion,
different sensor rates, missing/stale/low-quality/non-finite channels, missing
calibration, duplicate/reversed times, timezone alignment, clock corrections,
future-data exclusion, real-time gaps and CLI output/provenance. CI runs on
Linux/Windows with Python 3.10, 3.12 and 3.13.

Clinical prediction needs a separate prospective study with synchronized human
recordings, independent behavioral labels, participant-held-out evaluation,
confounder analysis, calibration and false-alarm burden. None of those are
established by this PR or by the scripted synthetic demo.
