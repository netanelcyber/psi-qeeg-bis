# Synchronized video + EEG demonstration

The `video-eeg` command replays a local MP4/WebM beside raw EEG, Welch spectra,
the existing FFT-calibrated Kalman/CUSUM trace, a separate frame-difference
descriptor and supplied event annotations. It calls the project's ordinary
`track_recording`, recording reader, quality checks and spectral extractor.

## Run the complete demo

Install the project and [FFmpeg](https://ffmpeg.org/download.html); both
`ffmpeg` and `ffprobe` must be on PATH. No additional Python dependency is
introduced. Pillow, used only by the avatar generator, is already a Matplotlib
dependency.

```bash
python -m pip install -e .
python scripts/demo_video_eeg.py --out outputs/video_eeg_demo
```

Open `outputs/video_eeg_demo/report.html` in a browser. It is self-contained,
including the generated video, and works offline. Use Play/Pause, playback
speed, the time slider, timeline clicks or event buttons to navigate. The
current-state cards use the latest completed EEG window, not an interpolated
future estimate. The overview plots intentionally show the recorded session.

The demo is **synthetic**: four EEG channels at 256 Hz and an animated avatar
are generated from a shared scripted timetable. It reuses the FFT demo's raw
EEG generator. A 100-second native EEG clock offset and 100-ppm video/EEG clock
rate difference exercise synchronization. The first 60 seconds are declared
calibration; later changes include spectral modulation, a movement/high-band
artifact interval and a gap in one EEG channel. These are numerical test
fixtures, not observed symptoms, patient video or clinical labels.

Outputs:

| File | Contents |
| --- | --- |
| `report.html` | Interactive offline replay, EEG/spectra/state/motion and provenance |
| `recording.csv`, `metadata.json`, `model.json` | Ordinary project inputs; synthetic demo only |
| `avatar.mp4`, `sync.json`, `events.csv` | Scripted source video, anchors and event sources |
| `tracking.csv`, `tracking.json` | Actual project model trace, QC, covariance and synchronization |
| `video_motion.csv` | Video/EEG timestamps and independent frame differences |

## Analyze your paired recording

```bash
psi-qeeg video-eeg recording.csv --metadata metadata.json \
  --video participant.mp4 --sync sync.json \
  --model model.json --baseline-sec 60 --events events.csv \
  --out outputs/paired_session
```

CSV/EDF/BDF support and physical units follow the existing recording reader.
The EEG file supplies the raw numeric channels; an EEG screen-capture movie
cannot reconstruct those amplitudes or their calibration. Use a browser-
compatible MP4/WebM codec; H.264 MP4 is exercised by the demo. Other containers
should be converted before recording synchronization anchors.

Without `--embed-video`, the report uses a copied adjacent media file. Pass
`--embed-video` to include videos of up to 25 MiB in one HTML file. A shared
report contains its EEG samples and, when embedded, the source video. No
external service/CDN or network request is used by the viewer.

`model.json` follows [FFT_KALMAN.md](FFT_KALMAN.md): named explicit loadings,
subject/medication identity and research model parameters. `--frontal LEFT
RIGHT` requires the same verified reference/scalp metadata as ordinary tracking.
`--high-z-limit` is an optional spectral-quality heuristic, not a clinical rule.

## Explicit synchronization

Example `sync.json` for a private, caller-supplied recording:

```json
{
  "subject_id": "participant-001",
  "video_kind": "participant",
  "source": "Local consented research recording, session 001",
  "license": "private_research_recording",
  "evidence": "Two acquisition sync pulses visible in video and logged by EEG",
  "anchors": [
    {"video_s": 0.0, "eeg_s": 100.0},
    {"video_s": 170.0, "eeg_s": 270.017}
  ],
  "uncertainty_ms": 20.0,
  "motion_roi": [0.2, 0.1, 0.6, 0.8]
}
```

Times are seconds on the browser's video playback clock and the recording's
native `time_s` EEG clock. At least two increasing anchors are required. An
affine map `EEG = offset + rate × video` is fitted by least squares. Multiple
anchors can test whether one offset/rate is adequate; a fit residual exceeding
the declared uncertainty fails. Video anchors outside the inspected video and
video/EEG pairs with no common interval also fail.

`uncertainty_ms` is **caller-declared timing uncertainty**, not measured model
accuracy. A perfect two-anchor fit is not independent validation of the pairing.
Real-data reports retain `caller_declared_not_independently_verified`. Video
frame intervals and motion resampling have their own finite time resolution;
the anchor uncertainty is not a claim of sub-frame behavioral timing accuracy.
Edited clips, dropped/duplicated frames and nonlinear clock drift may require
piecewise synchronization outside this simple affine implementation.

`video_kind` must be `participant`, `stimulus`, `eeg_screen_capture` or
`synthetic`. A stimulus movie is explicitly labeled as a stimulus and is never
represented as a recording of a participant's behavior. The subject identifier,
video SHA-256, source, license, anchors, fit residual and alignment status are
saved in the report JSON.

## Motion and annotations

`--motion-fps 4` decodes grayscale frames at 4 Hz, optionally inside normalized
`[x,y,width,height]` ROI coordinates. Consecutive 96×72 frame mean absolute
differences are divided by 255. The timestamp belongs to the later frame;
the first observation is unavailable. `--motion-fps 0` disables extraction.
This simple scene-change descriptor responds to camera motion, lighting and
compression as well as participant movement. It is not a validated measure of
motor activity, agitation, emotion, seizure severity or psychiatric state.

Per-window `video_motion_mean` uses only descriptor observations whose mapped
EEG time lies inside that completed EEG window. It never contributes a future
observation or enters the Kalman measurement vector. Visual co-occurrence in
the replay does not establish a neural/behavioral correlation or predictive
ability. That requires independent annotations, artifact control and held-out
participants/sessions.

Optional event CSV:

```csv
start_s,end_s,clock,label,source
120,124,video,Visible movement,manual_research_annotation
245,245.125,eeg,EEG gap,acquisition_log
```

The clock (`video` or `eeg`), label and source are mandatory. Times are mapped
to EEG seconds and checked against coverage. Supplied annotations are not
automatically verified. The demo's sources are always `synthetic_script`.

## Verification and data availability

Offline tests cover known clock offset/drift, inverse maps, conflicting anchors,
partial/no overlap, event provenance, peak/gap-preserving waveform display,
known video frame transitions, causal motion pairing, media copying/embedding,
HTML data escaping, CLI output and indeterminate psychiatric output. FFmpeg
integration tests skip when FFmpeg is unavailable. The demo is also checked
in Chromium for actual decoding, seeking, playback, clock mapping, artifact/gap
status, calibration boundaries and a device-pixel-ratio-2 display.

Primary sources inspected for real paired EEG/video examples:

- [EAV data descriptor](https://doi.org/10.1038/s41597-024-03838-4) and
  [dataset access description](https://pmc.ncbi.nlm.nih.gov/articles/PMC11413008/):
  actual EEG/audio/video; access requires an identified research request.
- [EEGMeditation data descriptor](https://doi.org/10.1038/s41597-026-07209-z)
  and [repository](https://huggingface.co/datasets/alexeykashevnik/EEGMeditation):
  EEG and frontal video when available; the inspected repository requires
  accepting access conditions. The authors describe alignment using eye
  transitions and EEG event markers, rather than a universally supplied exact
  shared timestamp for every clip.
- [Siena Scalp EEG file description](https://physionet.org/content/siena-scalp-eeg/1.0.0/):
  acquisition used video-EEG, but the public file description lists EDF EEG and
  event metadata. It does not establish an accessible synchronized patient
  video file for this demonstration.

No restricted patient footage was downloaded or republished for the demo.
The included runnable demonstration is explicitly synthetic; clinical
video-EEG and psychiatric prediction have not been validated by these checks.
