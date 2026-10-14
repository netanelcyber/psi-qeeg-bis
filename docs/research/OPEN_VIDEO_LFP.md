# Public synchronized video and LFP: executed IBL example

**Executed 13 October 2026.** Public paired recordings exist. The [IBL Autism 2025 release](https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_autism_noel.html) covers Fmr1, Cntnap2, Shank3 and wildtype mice performing the IBL decision task, with confidently aligned video. The data DOI is [10.6084/m9.figshare.30024880](https://doi.org/10.6084/m9.figshare.30024880); the public data license is CC-BY-4.0. Cite IBL and Noel et al., [Nature Neuroscience 2025](https://doi.org/10.1038/s41593-025-01965-8), when reusing these excerpts.

## Executed evidence

| Item | Downloaded and executed |
| --- | --- |
| Session | `274dacc9-05b2-4154-b4f7-8f6dd46efcc8`, SH023, 6 October 2021, session 001 |
| Signal | Real continuous intracranial LFP, probe00, 2,500 samples/s, source probe time 80–320 s; not sorted spikes or synthetic EEG |
| Channels | Original raw indices 32, 96, 128 and 192; released electrode-site atlas IDs and coordinates retained |
| Calibration | 0.6 V input range / 512 ADC scale / LF gain 250 = **4.6875 uV/count**; sync channel excluded |
| Analysis | 60 nonoverlapping four-second epochs per channel; 240 channel/epoch rows |
| Video | Body camera, experiment clock 180.002998–239.984932 s; 1,805 excerpt frames at nominal encoded 30 fps |
| Alignment | Full video: 170,645 frames and exactly 170,645 released frame timestamps. Released probe/session sync pairs are interpolated without extrapolation |
| Labels | Five independently measured wheel-movement intervals supplied by the dataset; not EEG-derived labels or expert-reviewed video attack annotations |

The Alyx subject metadata does not give SH023's genotype; the inspected OSF `behavior_list_SH.xlsx` is a session list, not a verified KO/WT mapping. Individual genotype remains **unverified**, despite collection membership. The anatomical reference electrode location also remains unverified. The strict animal training metadata contract was not bypassed to fit this example.

The current public S3 inventory identifies the downloaded files. Whole small metadata objects are verified against their MD5 ETags. Only the required compressed LFP byte range is downloaded, with HTTP 206, exact Content-Range and source If-Match checks; its local SHA-256 is retained. The entire multigigabyte LFP/video objects were not downloaded or fully checksummed. A stale cached file-size assumption is not used as current source integrity evidence.

## Calculation and annotation

For each four-second channel epoch, PSI's existing `animal_qeeg.band_powers` uses Welch PSD: two-second Hann windows, 50% overlap, constant detrending and density scaling. At 2,500 Hz, each FFT has 5,000 samples and a 0.5 Hz grid. Exact-edge integration gives band power in uV^2; relative power divides by integrated 1–100 Hz power. The declared exploratory bands are 1–4, **4–12**, 12–30, 30–80 and 80–100 Hz. These choices are not universal abnormal-wave cutoffs or a reproduction of the article's estimator.

The separate first-channel spectrogram uses two-second windows with one-second steps. Only the waveform display is resampled with an antialias filter to 500 Hz; saved epoch powers use original samples. Numerical amplitude/flatness statistics are recorded. Movement, muscle contamination, line interference and reference artifacts still require review; deflections around movement are not established neural warning signatures.

Movement onsets displayed in the clip are **200.382316, 217.489316, 218.064316, 222.274316 and 223.247316 s** in the experiment clock. Start/offset intervals, original frame indices, annotated MP4, spectrum/movement figure and calibrated LFP arrays are produced. Text uses the released timestamp of each original frame, not a constant nominal-FPS offset. Video expert-review status remains false.

## Forecasting boundary

These are ordinary movements in a head-fixed task. No meltdown, psychotic attack or adjudicated abnormal behavioral event is established. No behavioral classifier, LSTM or one-minute warning model was trained. Completed Fmr1 SVM models remain genotype classifiers.

A true minimum lead of 60 s requires a separately defined onset in **[t+60, t+H], H > 60**, using samples at or before t, independently reviewed event-free coverage, artifact handling and animal/session-separated evaluation. A 60-second clip or horizon does not prove this lead. Animal task movement would require separate validation before inference about human crises.

## Reproduce

```bash
python -m pip install -e . ibl-neuropixel
python scripts/validate_public_ibl_video_lfp.py --cache data/model_organism/ibl_sh023 --out outputs/ibl_video_lfp
```

`ffmpeg` and `ffprobe` must be available. The runner limits each LFP range request to 128 MiB and verifies calibration, sample/frame counts, source checksums, ordered clocks and clip length. Outputs are research artifacts, not accepted diagnostic epochs or event-training input.
