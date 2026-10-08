"""Explicit qEEG estimators; no reconstruction of the proprietary BIS index."""

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


def band_integral(f: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    if low < f[0] or high > f[-1] or high <= low:
        raise ValueError("Requested band lies outside measured frequencies")
    # Insert exact endpoints so adjacent bands partition the total integral.
    inside = (f > low) & (f < high)
    ff = np.r_[low, f[inside], high]
    pp = np.r_[np.interp(low, f, psd), psd[inside], np.interp(high, f, psd)]
    return float(trapezoid(pp, ff))


def spectral_features(x: np.ndarray, fs: float, line_hz: float | None = None) -> dict:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < round(2 * fs) or not np.isfinite(x).all():
        raise ValueError("PSD needs at least two seconds of complete one-dimensional EEG")
    if fs / 2 <= 40:
        raise ValueError("Full 0.5-40 Hz qEEG requires a sampling rate above 80 Hz")
    if line_hz is not None:
        if not 0 < line_hz < fs / 2:
            raise ValueError("Notch frequency must be positive and below Nyquist")
        b, a = signal.iirnotch(line_hz, 30, fs=fs)
        x = signal.filtfilt(b, a, x)
    nperseg = round(2 * fs)
    f, p = signal.welch(x, fs=fs, window="hann", nperseg=nperseg,
                        noverlap=nperseg // 2, detrend="constant", scaling="density")
    powers = {name: band_integral(f, p, *limits) for name, limits in BANDS.items()}
    total = band_integral(f, p, 0.5, 40)
    out = {f"{name}_power_uv2": value for name, value in powers.items()}
    out.update({f"{name}_relative_power": value / total if total > 0 else None
                for name, value in powers.items()})
    out["total_power_uv2"] = total
    out["delta_alpha_ratio"] = powers["delta"] / powers["alpha"] if powers["alpha"] > 1e-12 else None
    mask = (f >= 0.5) & (f <= 40)
    ff, pp = f[mask], p[mask]
    mass = (pp[:-1] + pp[1:]) / 2 * np.diff(ff)
    cumulative = np.r_[0.0, np.cumsum(mass)]
    out["sef95_hz"] = float(np.interp(0.95 * total, cumulative, ff)) if total > 0 else None
    prob = pp / pp.sum() if pp.sum() > 0 else np.zeros_like(pp)
    positive = prob > 0
    out["spectral_entropy"] = float(-np.sum(prob[positive] * np.log(prob[positive])) / np.log(len(pp)))
    return out


def phase_locking_value(x: np.ndarray, y: np.ndarray, fs: float, band=(8, 13)) -> float:
    if len(x) != len(y) or len(x) < 4 * fs or not np.isfinite(np.r_[x, y]).all():
        raise ValueError("PLV needs aligned, finite signals of at least four seconds")
    if not 0 < band[0] < band[1] < fs / 2:
        raise ValueError("PLV band must lie below Nyquist")
    sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    a = signal.hilbert(signal.sosfiltfilt(sos, x))
    b = signal.hilbert(signal.sosfiltfilt(sos, y))
    trim = round(fs / 2)
    delta = np.angle(a[trim:-trim]) - np.angle(b[trim:-trim])
    return float(abs(np.mean(np.exp(1j * delta))))


def squared_bicoherence(x: np.ndarray, fs: float, min_segments=20) -> float | None:
    """Mean normalized auto-bicoherence squared, f1/f2=5..15 Hz, f1+f2<=30.

    Estimate over many two-second Hann segments, 50% overlap. A single FFT
    trivially gives one, so an insufficient ensemble returns None.
    """
    if not np.isfinite(x).all():
        return None
    n = round(2 * fs)
    if fs / 2 <= 30 or len(x) < n:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(x), n)[::n // 2]
    if len(windows) < min_segments:
        return None
    fft = np.fft.rfft((windows - windows.mean(axis=1, keepdims=True)) * signal.windows.hann(n, sym=False), axis=1)
    frequencies = np.fft.rfftfreq(n, 1 / fs)
    bins = np.flatnonzero((frequencies >= 5) & (frequencies <= 15))
    i, j = np.meshgrid(bins, bins, indexing="ij")
    keep = (i <= j) & (frequencies[i] + frequencies[j] <= 30)
    i, j = i[keep], j[keep]
    product = fft[:, i] * fft[:, j]
    summed = fft[:, i + j]
    denominator = np.sum(abs(product) ** 2, axis=0) * np.sum(abs(summed) ** 2, axis=0)
    valid = denominator > np.finfo(float).tiny
    if not valid.any():
        return None
    numerator = abs(np.sum(product * np.conj(summed), axis=0)) ** 2
    values = np.clip(numerator[valid] / denominator[valid], 0, 1)
    return float(values.mean())

