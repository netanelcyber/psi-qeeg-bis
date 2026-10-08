# Real BIS EEG example: VitalDB case 1

| Field | Value |
| --- | --- |
| Source | VitalDB Open Dataset, Seoul National University Hospital |
| Public case identifier | 1 |
| Context | Surgical/perioperative anesthesia recording |
| Window | Seconds 332 through 452; end is exclusive |
| Raw EEG | `BIS/EEG1_WAV`, `BIS/EEG2_WAV` |
| Sampling rate | 128 Hz |
| Samples | 15,360 per EEG channel |
| EEG units | Microvolts |
| Monitor | BIS Vista, per the dataset device documentation |
| Individual sensor model, side, electrode pairs | Not reported; no Quatro/bilateral identity inferred |
| Monitor numerics | BIS, SQI, EMG, SEF, SR, TOTPOW |
| Psychiatric labels | None |
| Personal awake calibration | None |
| Intended example | Real-data pipeline engineering and qEEG visualization |
| Clinical validation | Not established |
| License | CC BY 4.0; see root DATA_LICENSE.md |

The excerpt was chosen as the first complete 120-second interval, searching every four seconds starting at 160 seconds, with finite EEG samples, recorded SQI ≥80, and recorded EMG <40 dB. This deliberate quality selection must not be interpreted as whole-dataset quality or clinical effectiveness. Both waveform channels were subsequently checked for complete finite data.

Download locations and SHA-256 hashes of each decompressed original source track are in `metadata.json`. The excerpt CSV has its own SHA-256. Numeric tracks were aligned using the last observation at or before each EEG timestamp, with a five-second expiry. Values before the first numeric measurement remain missing. No raw EEG interpolation, filtering or psychiatric relabelling occurred when preparing this excerpt.

The project computes quantitative EEG descriptors from these real waveform values. It does not claim the excerpt is an autism or psychosis dataset. Generic frontal spectral features are not substituted for the paper's posterior, temporal-CSD, stimulus-gating or autonomic components.

Citation: Lee HC et al. *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients*. Scientific Data 9, 279 (2022). https://doi.org/10.1038/s41597-022-01411-5

Dataset documentation: https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md

