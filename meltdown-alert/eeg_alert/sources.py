"""EEG data sources: synthetic simulator, Lab Streaming Layer (Muse / OpenBCI), CSV replay.

Every source yields chunks shaped (n_channels, n_samples) in microvolts.
"""
from __future__ import annotations

import time
from typing import Iterator, List, Optional, Sequence

import numpy as np

from .config import Config


# ---------------------------------------------------------------------------
class SyntheticEEG:
    """Simulated 4-channel wearable EEG with scripted arousal build-ups.

    Before each event the simulator ramps arousal over ``ramp_min`` minutes:
    alpha is suppressed, beta rises, right-frontal alpha drops relative to
    left, and high-frequency muscle activity increases. This exists only to
    exercise the pipeline end to end; it is NOT evidence that real meltdowns
    look like this.
    """

    def __init__(self, cfg: Config, minutes: float = 50.0, events_min: Sequence[float] = (22.0, 42.0),
                 ramp_min: float = 15.0, distractors_min: Sequence[float] = (30.0,),
                 effect: float = 0.25, seed: int = 7, chunk_sec: float = 1.0,
                 lead_channel: Optional[str] = None, lead_min: float = 0.0) -> None:
        self.cfg = cfg
        self.n = int(minutes * 60 * cfg.fs)
        self.events = [e * 60.0 for e in events_min]
        self.distractors = [d * 60.0 for d in distractors_min]
        self.ramp = ramp_min * 60.0
        self.effect = effect          # how strongly arousal modulates the EEG
        self.rng = np.random.default_rng(seed)
        self.chunk = int(chunk_sec * cfg.fs)
        # optional: one channel starts changing `lead_min` minutes before the others
        self.lead_channel = lead_channel
        self.lead_sec = lead_min * 60.0

    def arousal(self, t: np.ndarray) -> np.ndarray:
        """Ground-truth arousal in [0, 1]: slow build-up before events, short distractor bumps."""
        t = np.asarray(t, dtype=float)
        a = np.zeros_like(t)
        for e in self.events:
            pre = (t >= e - self.ramp) & (t < e)
            a[pre] = np.maximum(a[pre], (t[pre] - (e - self.ramp)) / self.ramp)
            post = (t >= e) & (t < e + 150)
            a[post] = np.maximum(a[post], 1.0 - (t[post] - e) / 150.0)
        for d in self.distractors:   # brief stress spike that does NOT lead to a meltdown
            a = np.maximum(a, 0.35 * np.exp(-0.5 * ((t - d) / 35.0) ** 2))
        return a

    def _pink(self, n: int) -> np.ndarray:
        white = self.rng.standard_normal(n)
        f = np.fft.rfftfreq(n, 1 / self.cfg.fs)
        f[0] = 1.0
        spec = np.fft.rfft(white) / np.sqrt(f)
        x = np.fft.irfft(spec, n)
        return x / np.std(x)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        fs = self.cfg.fs
        t = np.arange(self.n) / fs
        a = self.arousal(t)
        # slow, random, person-specific fluctuation (state changes unrelated to meltdowns)
        drift = _band_noise(self.rng.standard_normal(self.n // int(fs) + 1), 1.0, 0.003, 0.03)
        a = a + 0.06 * np.repeat(drift, int(fs))[: self.n]
        a = np.clip(a, 0, None)
        k = self.effect
        nch = len(self.cfg.channels)
        data = np.zeros((nch, self.n))
        phase = self.rng.uniform(0, 2 * np.pi, size=nch)
        a_lead = None
        if self.lead_channel is not None and self.lead_sec > 0:
            a_lead = np.clip(self.arousal(t + self.lead_sec) + (a - self.arousal(t)), 0, None)
        for c, name in enumerate(self.cfg.channels):
            ac = a_lead if (a_lead is not None and name == self.lead_channel) else a
            right_frontal = name in ("AF8", "F4", "Fp2")
            alpha_amp = 9.0 * (1.0 - 0.6 * k * ac * (1.2 if right_frontal else 1.0))
            beta_amp = 2.0 * (1.0 + 1.5 * k * ac)
            emg_amp = 0.6 * (1.0 + 3.0 * k * ac)
            alpha = alpha_amp * np.sin(2 * np.pi * 10.0 * t + phase[c] + 0.3 * self.rng.standard_normal() )
            beta = beta_amp * self.rng.standard_normal(self.n)
            beta = _band_noise(beta, fs, 14, 28)
            emg = emg_amp * _band_noise(self.rng.standard_normal(self.n), fs, 32, 44)
            data[c] = 8.0 * self._pink(self.n) + alpha + beta + emg
        # sprinkle occasional blink / movement artifacts
        for _ in range(int(self.n / fs / 90)):
            i = self.rng.integers(0, self.n - int(fs))
            data[:, i:i + int(0.3 * fs)] += 400.0
        return t, data

    def stream(self) -> Iterator[np.ndarray]:
        _, data = self.generate()
        for i in range(0, data.shape[1], self.chunk):
            yield data[:, i:i + self.chunk]


def _band_noise(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    return y / (np.std(y) + 1e-12)


# ---------------------------------------------------------------------------
class LSLSource:
    """Read EEG from a Lab Streaming Layer stream.

    Works with Muse (``muselsl stream``), OpenBCI (OpenBCI GUI -> Networking -> LSL,
    or BrainFlow), and most research amplifiers.
    """

    def __init__(self, cfg: Config, stream_type: str = "EEG", chunk_sec: float = 1.0,
                 channel_indices: Optional[List[int]] = None) -> None:
        try:
            from pylsl import StreamInlet, resolve_byprop  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise SystemExit("pylsl is required for --source lsl  (pip install pylsl)") from e
        streams = resolve_byprop("type", stream_type, timeout=15)
        if not streams:
            raise SystemExit(f"No LSL stream of type {stream_type!r} found.")
        self.inlet = StreamInlet(streams[0], max_chunklen=int(cfg.fs * chunk_sec))
        self.cfg = cfg
        self.idx = channel_indices
        self.chunk = int(cfg.fs * chunk_sec)

    def stream(self) -> Iterator[np.ndarray]:  # pragma: no cover - needs hardware
        buf: List[List[float]] = []
        while True:
            samples, _ = self.inlet.pull_chunk(timeout=1.0)
            buf.extend(samples)
            while len(buf) >= self.chunk:
                block = np.asarray(buf[: self.chunk]).T
                buf = buf[self.chunk:]
                if self.idx is not None:
                    block = block[self.idx]
                yield block[: len(self.cfg.channels)]


# ---------------------------------------------------------------------------
class CSVSource:
    """Replay a CSV file: one column per channel (header = channel names), microvolts.

    An optional first column named ``timestamp`` / ``time`` is ignored.
    """

    def __init__(self, cfg: Config, path: str, chunk_sec: float = 1.0, realtime: bool = False) -> None:
        import csv
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [r for r in reader if r]
        cols = [i for i, h in enumerate(header) if h.strip().lower() not in ("timestamp", "time", "t")]
        names = [header[i].strip() for i in cols]
        arr = np.array([[float(r[i]) for i in cols] for r in rows]).T
        wanted = [names.index(c) for c in cfg.channels if c in names]
        self.data = arr[wanted] if wanted else arr[: len(cfg.channels)]
        self.chunk = int(cfg.fs * chunk_sec)
        self.realtime = realtime
        self.chunk_sec = chunk_sec

    def stream(self) -> Iterator[np.ndarray]:
        for i in range(0, self.data.shape[1], self.chunk):
            if self.realtime:
                time.sleep(self.chunk_sec)
            yield self.data[:, i:i + self.chunk]
