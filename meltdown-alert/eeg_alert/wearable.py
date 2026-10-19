"""Vendor-neutral wearable file input, causal alignment, QC and experimental fusion.

No device driver or Equivital SDK is used. Units are encoded in metric names.
HRV is supplied as precomputed RMSSD, never inferred from sampled heart rate.
"""
from __future__ import annotations

from bisect import bisect_right
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from .timing import session_seconds

# Acquisition sanity limits, NOT clinical normal ranges or alarm thresholds.
LIMITS = {
    "heart_rate_bpm": (20.0, 250.0),
    "hrv_rmssd_ms": (0.0, 300.0),
    "respiration_bpm": (1.0, 80.0),
    "skin_temperature_c": (10.0, 45.0),
    "movement_g": (0.0, 16.0),  # externally computed dynamic acceleration RMS; gravity removed
}


@dataclass
class WearableConfig:
    min_quality: float = 0.6
    min_baseline_samples: int = 5
    fusion_weight: float = 0.25
    max_age_sec: dict[str, float] = field(default_factory=lambda: {
        "heart_rate_bpm": 15.0, "hrv_rmssd_ms": 30.0, "respiration_bpm": 15.0,
        "skin_temperature_c": 30.0, "movement_g": 2.0,
    })
    # Research assumptions, not fitted predictors. Temperature and motion are
    # context-only by default because direction/confounding is task-dependent.
    weights: dict[str, float] = field(default_factory=lambda: {
        "heart_rate_bpm": 1.0, "hrv_rmssd_ms": -1.0, "respiration_bpm": 1.0,
        "skin_temperature_c": 0.0, "movement_g": 0.0,
    })

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_quality) or not 0 <= self.min_quality <= 1:
            raise ValueError("Wearable min_quality must be in [0, 1]")
        if not math.isfinite(self.fusion_weight) or not 0 <= self.fusion_weight <= 1:
            raise ValueError("Wearable fusion_weight must be in [0, 1]")
        if not isinstance(self.min_baseline_samples, int) or self.min_baseline_samples < 2:
            raise ValueError("Wearable min_baseline_samples must be an integer >= 2")
        if set(self.max_age_sec) != set(LIMITS) or set(self.weights) != set(LIMITS):
            raise ValueError("Wearable max_age_sec and weights must define every canonical metric")
        if any(not math.isfinite(v) or v <= 0 for v in self.max_age_sec.values()):
            raise ValueError("Wearable maximum ages must be finite and positive")
        if any(not math.isfinite(v) for v in self.weights.values()):
            raise ValueError("Wearable feature weights must be finite")


@dataclass(frozen=True)
class Measurement:
    t: float
    metric: str
    value: float
    quality: float | None = None  # None = not supplied, not device-verified quality
    sensor: str = "unspecified"

    def __post_init__(self) -> None:
        if self.metric not in LIMITS:
            raise ValueError(f"Unknown wearable metric: {self.metric!r}")
        if not math.isfinite(self.t):
            raise ValueError("Wearable timestamps must be finite")
        if self.quality is not None and (not math.isfinite(self.quality) or not 0 <= self.quality <= 1):
            raise ValueError("Wearable quality must be in [0, 1], or blank when unknown")


class WearableSource(Protocol):
    """A future verified bridge may implement this session-clock interface."""

    metrics: Sequence[str]

    def through(self, t: float) -> Mapping[str, Sequence[Measurement]]:
        """Return new observations at or before t, once each, in time order."""
        ...


class WearableReplay:
    """Replay distinct metric streams. Each metric must have exactly one owner.

    Samples are not sorted on input: duplicates, reversals and multiple sensors
    supplying the same metric are rejected instead of silently mixing subjects.
    """

    def __init__(self, measurements: Iterable[Measurement]) -> None:
        self._series: dict[str, list[Measurement]] = {}
        for m in measurements:
            seq = self._series.setdefault(m.metric, [])
            if seq and m.sensor != seq[0].sensor:
                raise ValueError(f"Multiple sensors supply {m.metric}; select one before replay")
            if seq and m.t <= seq[-1].t:
                raise ValueError(f"Timestamps for {m.metric} must be strictly increasing; duplicate/out-of-order sample")
            seq.append(m)
        self.metrics = tuple(self._series)
        self._times = {k: [m.t for m in v] for k, v in self._series.items()}
        self._pos = {k: 0 for k in self.metrics}
        self._last_t = -math.inf

    def through(self, t: float) -> dict[str, Sequence[Measurement]]:
        if not math.isfinite(t) or t < self._last_t:
            raise ValueError("Wearable replay requires finite, nondecreasing session time")
        self._last_t = t
        out = {}
        for metric in self.metrics:
            end = bisect_right(self._times[metric], t)
            out[metric] = self._series[metric][self._pos[metric]:end]
            self._pos[metric] = end
        return out


def load_measurements(path: str | Path, *, session_start: str | None = None,
                      offset_sec: float = 0.0) -> list[Measurement]:
    """Load canonical long or wide CSV. A blank value is an explicit invalid sample.

    Long: timestamp,metric,value[,quality,sensor]
    Wide: timestamp,heart_rate_bpm,...[,quality,<metric>_quality,sensor]
    Positive offset moves wearable observations later on the EEG session clock.
    """
    if not math.isfinite(offset_sec):
        raise ValueError("Wearable clock offset must be finite")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV column")
        if "timestamp" not in fields:
            raise ValueError(f"{path}: a timestamp column is required")
        long = "metric" in fields
        metrics = [k for k in LIMITS if k in fields]
        allowed = {"timestamp", "quality", "sensor"}
        allowed |= {"metric", "value"} if long else set(LIMITS) | {f"{k}_quality" for k in LIMITS}
        if set(fields) - allowed or (long and "value" not in fields) or (not long and not metrics):
            raise ValueError(f"{path}: expected canonical long/wide wearable CSV; map vendor columns first")
        result = []
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("row has the wrong number of fields")
                t = session_seconds(row["timestamp"], session_start) + offset_sec
                for metric in ([row["metric"]] if long else metrics):
                    val = row["value"] if long else row[metric]
                    quality = row.get(f"{metric}_quality", row.get("quality", ""))
                    result.append(Measurement(t, metric, float(val) if val.strip() else math.nan,
                                              float(quality) if quality.strip() else None,
                                              row.get("sensor", "unspecified") or "unspecified"))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
    if not result:
        raise ValueError(f"{path}: no wearable measurements")
    # Validate each file before combining it with other sensor files.
    WearableReplay(result)
    return result


@dataclass
class WearableFrame:
    metrics: dict[str, dict]
    index: float | None
    active_metrics: tuple[str, ...]
    fusion_weight: float
    coverage: float


class WearableProcessor:
    """Causal sample-and-hold with expiry and per-metric calm calibration.

    No interpolation, future samples, zero-filling, or last-good-value fallback
    after a bad sample. Only distinct, QC-passing observations in [0, baseline_sec)
    enter the baseline. Missing calibration never becomes a zero z-score.
    """

    def __init__(self, source: WearableSource, baseline_sec: float,
                 cfg: WearableConfig | None = None) -> None:
        self.source = source
        self.cfg = cfg or WearableConfig()
        if not math.isfinite(baseline_sec) or baseline_sec <= 0:
            raise ValueError("Wearable baseline_sec must be finite and positive")
        self.baseline_sec = baseline_sec
        self._latest: dict[str, Measurement] = {}
        self._base: dict[str, list[float]] = {k: [] for k in LIMITS}
        self._scales: dict[str, tuple[float, float]] = {}
        self._calibrated = False
        self._last_t = -math.inf

    def _status(self, m: Measurement) -> str:
        if not math.isfinite(m.value):
            return "invalid_value"
        lo, hi = LIMITS[m.metric]
        if not lo <= m.value <= hi:
            return "out_of_range"
        if m.quality is not None and m.quality < self.cfg.min_quality:
            return "low_quality"
        return "ok"

    def update(self, t: float) -> WearableFrame:
        if not math.isfinite(t) or t < self._last_t:
            raise ValueError("Wearable processing requires finite, nondecreasing session time")
        self._last_t = t
        for metric, seq in self.source.through(t).items():
            for m in seq:
                if metric != m.metric or m.t > t:
                    raise ValueError("Wearable source returned a mismatched metric or future observation")
                previous = self._latest.get(metric)
                if previous is not None and m.sensor != previous.sensor:
                    raise ValueError(f"Multiple sensors supply {metric}; select one before processing")
                if previous is not None and m.t <= previous.t:
                    raise ValueError("Wearable source must return each measurement once, in time order")
                self._latest[metric] = m
                if not self._calibrated and 0 <= m.t < self.baseline_sec and self._status(m) == "ok":
                    self._base[metric].append(m.value)
        if not self._calibrated and t >= self.baseline_sec:
            for k, vals in self._base.items():
                if len(vals) >= self.cfg.min_baseline_samples:
                    med = float(np.median(vals))
                    scale = max(float(np.median(np.abs(np.array(vals) - med))) * 1.4826,
                                0.05 * abs(med), 1e-6)
                    self._scales[k] = (med, scale)
            self._calibrated = True
        metrics = {}
        usable = {}
        for k in LIMITS:
            m = self._latest.get(k)
            status = "missing"
            z = None
            if m is not None:
                status = self._status(m)
                if t - m.t > self.cfg.max_age_sec[k]:
                    status = "stale"
                if status == "ok":
                    if k in self._scales:
                        med, scale = self._scales[k]
                        z = float(np.clip((m.value - med) / scale, -5.0, 5.0))
                        usable[k] = z
                    else:
                        status = "uncalibrated" if self._calibrated else "calibrating"
            metrics[k] = {
                "status": status, "value": m.value if m and math.isfinite(m.value) else None,
                "sample_t": m.t if m else None, "age_sec": t - m.t if m else None,
                "quality": m.quality if m else None,
                "quality_supplied": m is not None and m.quality is not None,
                "sensor": m.sensor if m else None,
                "baseline_samples": len(self._base[k]), "z": z,
            }
        active = tuple(k for k in LIMITS if k in usable and self.cfg.weights[k] != 0)
        # Fixed denominator across configured, calibrated features: losing a
        # channel cannot silently amplify the remaining channels' contribution.
        total = sum(abs(self.cfg.weights[k]) for k in self._scales)
        present = sum(abs(self.cfg.weights[k]) for k in active)
        index = (sum(self.cfg.weights[k] * usable[k] for k in active) / math.sqrt(total)
                 if active and total else None)
        coverage = present / total if total else 0.0
        return WearableFrame(metrics, index, active,
                             self.cfg.fusion_weight if index is not None else 0.0, coverage)
