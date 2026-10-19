"""Synchronized local video/EEG replay; video motion is an independent descriptor.

Clock anchors and provenance are explicit. No EEG is reconstructed from video,
and no behavior/diagnosis is inferred from faces or the frame-difference metric.
"""

import base64
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

from .quality import QualityPolicy, assess_epoch
from .spectral_tracking import SpectralConfig, SpectralExtractor
from .tracking import track_recording


@dataclass(frozen=True)
class VideoClock:
    rate: float
    offset_s: float
    max_anchor_error_ms: float
    uncertainty_ms: float

    @classmethod
    def fit(cls, anchors, uncertainty_ms):
        """Fit EEG clock seconds = offset + rate * player video seconds."""
        if not isinstance(anchors, list) or len(anchors) < 2:
            raise ValueError("Supply at least two video_s/eeg_s synchronization anchors")
        try:
            x = np.array([a["video_s"] for a in anchors], dtype=float)
            y = np.array([a["eeg_s"] for a in anchors], dtype=float)
            uncertainty_ms = float(uncertainty_ms)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("Anchors must declare numeric video_s/eeg_s and uncertainty_ms") from exc
        if not np.isfinite(np.r_[x, y, uncertainty_ms]).all() or uncertainty_ms < 0:
            raise ValueError("Clock anchors and nonnegative uncertainty_ms must be finite")
        if np.any(x < 0) or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
            raise ValueError("Anchors must have increasing video/EEG times, with video_s >= 0")
        xx, yy = x - x.mean(), y - y.mean()
        rate = float(xx @ yy / (xx @ xx))
        offset = float(y.mean() - rate * x.mean())
        error = float(np.max(abs(y - (offset + rate*x))) * 1000)
        if error > uncertainty_ms + 1e-5:
            raise ValueError("Clock anchor residual exceeds declared uncertainty_ms; inspect synchronization")
        return cls(rate, offset, error, float(uncertainty_ms))

    def to_eeg(self, video_s):
        return self.offset_s + self.rate * np.asarray(video_s)

    def to_video(self, eeg_s):
        return (np.asarray(eeg_s) - self.offset_s) / self.rate

    def overlap(self, eeg_start, eeg_end, video_duration):
        start = max(0.0, float(self.to_video(eeg_start)))
        end = min(float(video_duration), float(self.to_video(eeg_end)))
        if end <= start:
            raise ValueError("Video and EEG have no overlapping interval under the declared clock mapping")
        return start, end


def _program(name):
    executable = shutil.which(name)
    if executable is None:
        raise ValueError(f"Video replay requires {name}; install FFmpeg (ffmpeg and ffprobe)")
    return executable


def probe_video(path):
    path = Path(path).resolve()
    if path.suffix.lower() not in {".mp4", ".webm"} or not path.is_file():
        raise ValueError("Use an existing local MP4 or WebM video")
    command = [_program("ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe",
               "-select_streams", "v:0", "-show_entries", "stream=width,height,duration:format=duration",
               "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError("ffprobe could not read a video stream")
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError("Input contains no video stream")
    stream = data["streams"][0]
    duration = float(stream.get("duration", data.get("format", {}).get("duration", "nan")))
    if not np.isfinite(duration) or duration <= 0 or min(stream["width"], stream["height"]) <= 0:
        raise ValueError("Video duration and dimensions must be known and positive")
    return {"duration_s": duration, "width": stream["width"], "height": stream["height"]}


def frame_difference(path, clock, fps=4.0, roi=None):
    """Mean absolute grayscale frame difference / 255 at a fixed replay rate.

    Returns the time of the later frame. Lighting/camera motion also change
    this descriptor. It is not a pose, agitation, emotion or seizure score.
    """
    if not np.isfinite(fps) or not 0 < fps <= 30:
        raise ValueError("Motion sampling fps must be finite and in (0, 30]")
    filters = ["setpts=PTS-STARTPTS"]
    if roi is not None:
        roi = np.asarray(roi, float)
        if (roi.shape != (4,) or not np.isfinite(roi).all() or np.any(roi < 0)
                or np.any(roi[2:] <= 0) or np.any(roi[:2] + roi[2:] > 1)):
            raise ValueError("motion_roi is normalized [x, y, width, height] within the frame")
        x, y, w, h = roi
        filters.append(f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}")
    filters.extend([f"fps={fps}", "scale=96:72", "format=gray"])
    command = [_program("ffmpeg"), "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
               "-i", str(Path(path).resolve()), "-an", "-vf", ",".join(filters),
               "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    rows, previous = [], None
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            i = 0
            while True:
                raw = process.stdout.read(96*72)
                if not raw:
                    break
                if len(raw) != 96*72:
                    raise ValueError("Video decoder returned a truncated frame")
                current = np.frombuffer(raw, np.uint8).astype(np.int16)
                value = None if previous is None else float(np.mean(abs(current - previous)) / 255)
                rows.append({"video_s": i/fps, "eeg_s": float(clock.to_eeg(i/fps)), "frame_difference": value})
                previous, i = current, i + 1
            if process.wait():
                raise ValueError("FFmpeg could not decode the video for frame differences")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
    if len(rows) < 2:
        raise ValueError("Video must contain at least two motion-sampling frames")
    return pd.DataFrame(rows)


def read_annotations(path, clock, eeg_start, eeg_end):
    if path is None:
        return []
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"start_s", "end_s", "clock", "label", "source"}
    if not required <= set(frame):
        raise ValueError("Events CSV needs start_s,end_s,clock,label,source")
    events = []
    for row in frame.to_dict("records"):
        a, b = float(row["start_s"]), float(row["end_s"])
        if row["clock"] not in {"video", "eeg"}:
            raise ValueError("Event clock must be explicitly video or eeg")
        if row["clock"] == "video":
            a, b = float(clock.to_eeg(a)), float(clock.to_eeg(b))
        if (not np.isfinite([a, b]).all() or b < a or a < eeg_start or b > eeg_end
                or not str(row["label"]).strip() or not str(row["source"]).strip()):
            raise ValueError("Events need finite ordered times inside EEG coverage, labels and sources")
        events.append({"start_s": a, "end_s": b, "label": str(row["label"]), "source": str(row["source"])})
    return sorted(events, key=lambda e: e["start_s"])


def _raw_display(recording, max_blocks=30000):
    """Min/max display blocks retain peaks and break at any missing sample."""
    x = recording.eeg
    stride = max(1, int(np.ceil(len(x) / max_blocks)))
    count = int(np.ceil(len(x) / stride))
    padded = np.full((count*stride, x.shape[1]), np.nan)
    padded[:len(x)] = x
    blocks = padded.reshape(count, stride, x.shape[1])
    finite = np.isfinite(blocks)
    valid = finite.all(axis=1)
    # Padding of the final partial block is not a gap in the original EEG.
    if len(x) % stride:
        valid[-1] = np.isfinite(blocks[-1, :len(x) % stride]).all(axis=0)
    low = np.where(valid, np.where(finite, blocks, np.inf).min(axis=1), np.nan)
    high = np.where(valid, np.where(finite, blocks, -np.inf).max(axis=1), np.nan)
    times = recording.time[0] + np.minimum((np.arange(count)+1)*stride, len(x)) / recording.fs
    def safe(values):
        return [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in values]
    return {"end_s": times.tolist(), "min_uv": safe(low), "max_uv": safe(high),
            "block_samples": stride, "original_fs": recording.fs}


def _spectrogram(recording, trace, summary):
    config = summary["noise_calibration"]["config"]
    extractor = SpectralExtractor(SpectralConfig(**config))
    frequencies = extractor.frequencies
    mask = (frequencies >= .5) & (frequencies <= 45)
    cfg, x = extractor.config, recording.eeg
    starts = [(s, True) for s in range(0, summary["baseline_samples"] - cfg.window_n + 1, cfg.hop_n)]
    starts += [(round((row.window_start_s-recording.time[0])*recording.fs), row.measurement_used)
               for row in trace.itertuples()]
    values, times = [], []
    policy = QualityPolicy(**summary["quality_policy"])
    bis = recording.metadata.get("sensor", {}).get("family") == "BIS"
    for start, usable in starts:
        times.append(float(recording.time[start] + cfg.window_s))
        qc = assess_epoch(x[start:start+cfg.window_n], recording.frame.iloc[start:start+cfg.window_n], policy, bis)
        if not usable or not qc["accepted"]:
            values.append(None)
            continue
        spectrum = extractor.extract(x[start:start+cfg.window_n].T).psd.mean(axis=0)[mask]
        values.append(np.round(10*np.log10(np.maximum(spectrum, 1e-12)), 3).tolist())
    return {"end_s": times, "frequencies_hz": frequencies[mask].tolist(), "db_uv2_hz": values}


def write_video_eeg_report(recording, model, baseline_s, video, sync, out, *, events=None,
                           embed_video=False, motion_fps=4.0, **tracking_options):
    """Run the real project tracker and write an offline synchronized report."""
    if not isinstance(sync, dict) or sync.get("subject_id") != recording.metadata.get("subject_id"):
        raise ValueError("Video synchronization subject_id must match the EEG recording")
    for key in ("source", "license", "evidence"):
        if not isinstance(sync.get(key), str) or not sync[key].strip():
            raise ValueError(f"Video synchronization manifest must declare {key}")
    if sync.get("video_kind") not in {"synthetic", "participant", "stimulus", "eeg_screen_capture"}:
        raise ValueError("Declare video_kind as synthetic, participant, stimulus or eeg_screen_capture")
    clock = VideoClock.fit(sync.get("anchors"), sync.get("uncertainty_ms", float("nan")))
    info = probe_video(video)
    if max(float(a["video_s"]) for a in sync["anchors"]) > info["duration_s"] + 1e-6:
        raise ValueError("Synchronization anchor is outside the video duration")
    eeg_start, eeg_end = recording.time[0], recording.time[-1] + 1/recording.fs
    overlap = clock.overlap(eeg_start, eeg_end, info["duration_s"])
    annotations = read_annotations(events, clock, eeg_start, eeg_end)
    if not np.isfinite(motion_fps) or not 0 <= motion_fps <= 30:
        raise ValueError("motion_fps must be finite and in [0, 30]")
    path = Path(video).resolve()
    if embed_video and path.stat().st_size > 25*1024**2:
        raise ValueError("Embedded video is limited to 25 MiB; omit --embed-video for an adjacent media file")
    trace, summary = track_recording(recording, model, baseline_s, **tracking_options)
    motion = frame_difference(video, clock, motion_fps, sync.get("motion_roi")) if motion_fps else None
    if motion is not None:
        means, counts = [], []
        for row in trace.itertuples():
            selected = motion.loc[(motion.eeg_s >= row.window_start_s) & (motion.eeg_s <= row.window_end_s), "frame_difference"].dropna()
            means.append(float(selected.mean()) if len(selected) else None)
            counts.append(len(selected))
        trace["video_motion_mean"] = means
        trace["video_motion_samples"] = counts
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    summary["video_eeg"] = {"video": {**info, "basename": path.name, "sha256": digest},
                             "synchronization": {**asdict(clock), "manifest": sync},
                             "common_video_interval_s": list(overlap),
                             "alignment_status": "synthetic_shared_clock" if sync["video_kind"] == "synthetic" else "caller_declared_not_independently_verified",
                             "motion": {"method": "mean_absolute_grayscale_frame_difference_divided_by_255",
                                        "fps": motion_fps, "roi": sync.get("motion_roi"),
                                        "used_in_kalman_measurements": False},
                             "events": annotations, "behavior_inference": "not_performed"}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mime = "video/mp4" if path.suffix.lower() == ".mp4" else "video/webm"
    if embed_video:
        media = {"mime": mime, "base64": base64.b64encode(path.read_bytes()).decode("ascii")}
    else:
        target = out / ("source_video" + path.suffix.lower())
        if target.resolve() != path:
            shutil.copyfile(path, target)
        media = {"mime": mime, "relative_path": target.name}
    payload = {"summary": summary, "channels": recording.channels, "eeg_start_s": float(eeg_start),
               "eeg_end_s": float(eeg_end), "raw": _raw_display(recording),
               "spectrogram": _spectrogram(recording, trace, summary),
               "trace": json.loads(trace.to_json(orient="records")),
               "motion": [] if motion is None else json.loads(motion.to_json(orient="records")), "media": media}
    template = Path(__file__).with_name("video_eeg_viewer.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    (out/"report.html").write_text(template.replace("__VIDEO_EEG_PAYLOAD__", data), encoding="utf-8")
    trace.to_csv(out/"tracking.csv", index=False)
    (out/"tracking.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if motion is not None:
        motion.to_csv(out/"video_motion.csv", index=False)
    return summary
