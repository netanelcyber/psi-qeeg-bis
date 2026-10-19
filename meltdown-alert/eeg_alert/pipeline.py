"""Glue: buffer incoming chunks into sliding windows, extract features, run the detector."""
from __future__ import annotations

from typing import Iterable, Iterator, List, Optional

import numpy as np

from .config import Config
from .detector import Detector, Step
from .features import extract_all
from .notify import Notifier


def run(chunks: Iterable[np.ndarray], cfg: Config, notifiers: Optional[List[Notifier]] = None,
        context=None) -> Iterator[Step]:
    det = Detector(cfg, context)
    win = int(cfg.window_sec * cfg.fs)
    hop = int(cfg.step_sec * cfg.fs)
    buf = np.zeros((len(cfg.channels), 0))
    consumed = 0  # samples dropped from the front of buf so far
    next_start = 0
    for chunk in chunks:
        buf = np.concatenate([buf, chunk], axis=1)
        while next_start + win <= consumed + buf.shape[1]:
            s = next_start - consumed
            window = buf[:, s:s + win]
            t = (next_start + win) / cfg.fs
            feats, per_channel = extract_all(window, cfg)
            step = det.update(t, feats, per_channel)
            if step.alert and notifiers:
                for n in notifiers:
                    n.send(step.alert)
            yield step
            next_start += hop
        drop = max(0, next_start - consumed)
        if drop:
            buf = buf[:, drop:]
            consumed += drop
