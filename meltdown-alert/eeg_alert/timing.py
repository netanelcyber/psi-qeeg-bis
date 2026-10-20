"""Explicit session-clock conversion shared by EEG and wearable file replay."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np


def _absolute_time(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Absolute timestamps must include a timezone (Z or +/-HH:MM)")
    return dt


def session_seconds(value: str, session_start: str | None = None) -> float:
    """Numbers are session seconds; ISO timestamps require an explicit shared origin.

    Never align sensors by independently subtracting each file's first timestamp.
    Numeric Unix timestamps must be converted by the caller, not guessed here.
    """
    try:
        seconds = float(value)
    except ValueError:
        if session_start is None:
            raise ValueError("ISO timestamps require --session-start with a timezone") from None
        seconds = (_absolute_time(value) - _absolute_time(session_start)).total_seconds()
    if not math.isfinite(seconds):
        raise ValueError("Timestamps must be finite")
    return seconds


@dataclass
class EEGChunk:
    """EEG in microvolts plus one session-relative timestamp per sample."""

    samples: np.ndarray
    timestamps: np.ndarray
