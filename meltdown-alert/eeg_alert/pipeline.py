"""Glue: buffer incoming chunks into sliding windows, extract features, run the detector."""
from __future__ import annotations

from typing import Iterable, Iterator, List, Optional

import numpy as np

from .config import Config
from .detector import Detector, Step
from .features import extract_all
from .notify import Notifier
from .timing import EEGChunk
from .wearable import WearableProcessor


def run(chunks: Iterable[np.ndarray | EEGChunk], cfg: Config, notifiers: Optional[List[Notifier]] = None,
        context=None, *, wearable: Optional[WearableProcessor] = None) -> Iterator[Step]:
    det = Detector(cfg, context)
    previous_metrics = ()
    for t, window, discontinuity in _windows(chunks, cfg):
        feats, per_channel = extract_all(window, cfg)
        frame = wearable.update(t) if wearable is not None else None
        enabled = wearable is not None and wearable.cfg.fusion_weight > 0
        if discontinuity or (enabled and (frame.active_metrics != previous_metrics or feats is None)):
            det.reset_evidence()
        if frame is not None:
            previous_metrics = frame.active_metrics
        step = det.update(t, feats, per_channel,
                          auxiliary_index=frame.index if frame else None,
                          auxiliary_weight=frame.fusion_weight if frame else 0.0)
        if frame is not None:
            step.wearable = {"metrics": frame.metrics, "index": frame.index,
                             "active_metrics": list(frame.active_metrics),
                             "fusion_weight": frame.fusion_weight if step.raw_index is not None else 0.0,
                             "coverage": frame.coverage,
                             "eeg_required": True, "clinical_validation": False}
        if step.alert and notifiers:
            for n in notifiers:
                n.send(step.alert)
        yield step


def _windows(chunks: Iterable[np.ndarray | EEGChunk], cfg: Config):
    """Yield legacy sample-clock windows or timestamped windows, never mixing clocks."""
    from itertools import chain

    if cfg.fs <= 0 or cfg.window_sec <= 0 or cfg.step_sec <= 0:
        raise ValueError("EEG sampling rate, window and hop must be positive")
    win = int(cfg.window_sec * cfg.fs)
    hop = int(cfg.step_sec * cfg.fs)
    if not 0 < hop <= win:
        raise ValueError("EEG hop must be between one sample and the window length")
    iterator = iter(chunks)
    first = next(iterator, None)
    if first is None:
        return
    chunks = chain([first], iterator)
    if isinstance(first, EEGChunk):
        yield from _timestamped_windows(chunks, cfg, win, hop)
        return
    buf = np.zeros((len(cfg.channels), 0))
    consumed = 0  # samples dropped from the front of buf so far
    next_start = 0
    for chunk in chunks:
        if isinstance(chunk, EEGChunk):
            raise ValueError("Do not mix sample-clock and timestamped EEG chunks")
        buf = np.concatenate([buf, chunk], axis=1)
        while next_start + win <= consumed + buf.shape[1]:
            s = next_start - consumed
            window = buf[:, s:s + win]
            t = (next_start + win) / cfg.fs
            yield t, window, False
            next_start += hop
        drop = max(0, next_start - consumed)
        if drop:
            buf = buf[:, drop:]
            consumed += drop


def _timestamped_windows(chunks, cfg: Config, win: int, hop: int):
    buf = np.empty((len(cfg.channels), 0))
    times = np.empty(0)
    previous_t = None
    discontinuity = False
    for chunk in chunks:
        if not isinstance(chunk, EEGChunk):
            raise ValueError("Do not mix sample-clock and timestamped EEG chunks")
        x = np.asarray(chunk.samples)
        ts = np.asarray(chunk.timestamps, dtype=float)
        if (x.ndim != 2 or x.shape[0] != len(cfg.channels) or ts.ndim != 1
                or x.shape[1] != ts.size):
            raise ValueError("EEGChunk needs configured channels and one timestamp per sample")
        if not ts.size:
            continue
        if not np.all(np.isfinite(ts)) or ts[0] < 0 or np.any(np.diff(ts) <= 0):
            raise ValueError("EEG timestamps must be finite, nonnegative and strictly increasing")
        if previous_t is not None and ts[0] <= previous_t:
            raise ValueError("EEG timestamps must increase across chunks")
        delta = np.diff(np.concatenate(([previous_t], ts)) if previous_t is not None else ts)
        # ±20% sample-interval jitter is accepted. Larger gaps/rate mismatch
        # break windows instead of feeding irregular samples into Welch's FFT.
        bad = np.flatnonzero(np.abs(delta - 1 / cfg.fs) > 0.2 / cfg.fs)
        cuts = bad if previous_t is not None else bad + 1
        start = 0
        for end in [*cuts, ts.size]:
            buf = np.concatenate((buf, x[:, start:end]), axis=1)
            times = np.concatenate((times, ts[start:end]))
            while times.size >= win:
                yield float(times[win - 1] + 1 / cfg.fs), buf[:, :win], discontinuity
                discontinuity = False
                buf, times = buf[:, hop:], times[hop:]
            if end != ts.size:
                buf, times = buf[:, :0], times[:0]
                discontinuity = True
            start = end
        previous_t = float(ts[-1])
