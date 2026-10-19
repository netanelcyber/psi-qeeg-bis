"""FFT-derived measurement noise and experimental Kalman/CUSUM tracking.

R_fft is a conditional stationary-Gaussian *estimator* covariance, not all
physiological/model error. See docs/FFT_KALMAN.md for the assumptions, the
colored-noise approximation and the distinction from SOI/RDI or diagnosis.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy import signal, special
from scipy.fft import next_fast_len
from scipy.linalg import cho_factor, cho_solve

from .signal import BANDS, band_integral, welch_psd


TRACK_BANDS = {**{k: v for k, v in BANDS.items() if k != "gamma"}, "high": (30, 45)}


def sample_count(seconds: float, fs: float) -> int:
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Durations must be finite and positive")
    n = round(seconds * fs)
    if n < 1 or not np.isclose(n, seconds * fs, rtol=0, atol=1e-8):
        raise ValueError("Durations must align to the sample grid")
    return n


@dataclass(frozen=True)
class SpectralConfig:
    fs: float
    channels: tuple[str, ...]
    window_s: float = 2.0
    hop_s: float = 1.0
    segment_s: float = 1.0
    overlap: float = 0.5
    nfft: int | None = None
    frontal_pair: tuple[str, str] | None = None
    measurements: tuple[str, ...] | None = None

    def __post_init__(self):
        if not np.isfinite(self.fs) or self.fs <= 90:
            raise ValueError("Tracking through 45 Hz requires fs > 90 Hz")
        if (not self.channels or not all(isinstance(c, str) and c for c in self.channels)
                or len(set(self.channels)) != len(self.channels)):
            raise ValueError("Tracking requires unique channel names")
        if self.window_n < 2 or self.segment_n < 2 or self.segment_n > self.window_n:
            raise ValueError("Welch segments must fit the analysis window")
        if self.hop_n > self.window_n:
            raise ValueError("hop_s must not exceed window_s")
        if not np.isfinite(self.overlap) or not 0 <= self.overlap < 1:
            raise ValueError("Welch overlap must be in [0, 1)")
        if not 0 <= self.overlap_n < self.segment_n:
            raise ValueError("Rounded overlap must be smaller than a segment")
        if self.nfft is not None and (not isinstance(self.nfft, int) or self.nfft < self.segment_n):
            raise ValueError("nfft must be an integer at least the segment length")
        if self.frontal_pair is not None:
            if (len(self.frontal_pair) != 2 or len(set(self.frontal_pair)) != 2
                    or not set(self.frontal_pair) <= set(self.channels)):
                raise ValueError("frontal_pair must name distinct measured left/right channels")

    @property
    def window_n(self):
        return sample_count(self.window_s, self.fs)

    @property
    def hop_n(self):
        return sample_count(self.hop_s, self.fs)

    @property
    def segment_n(self):
        return sample_count(self.segment_s, self.fs)

    @property
    def overlap_n(self):
        return round(self.segment_n * self.overlap)


@dataclass
class SpectralWindow:
    psd: np.ndarray
    band_power: np.ndarray
    log_power: np.ndarray
    values: np.ndarray


class SpectralExtractor:
    """Use the project Welch estimator and exact-endpoint band integrals.

    Input is (..., channels, window_samples), in microvolts. ``high`` is an
    observed spectral band; it is not a specific EMG or psychiatric marker.
    """

    def __init__(self, config: SpectralConfig):
        self.config = config
        self.bands = tuple(TRACK_BANDS)
        self.frequencies = np.fft.rfftfreq(config.nfft or config.segment_n, 1 / config.fs)
        # Integrating identity spectra constructs the exact linear quadrature
        # used by band_integral, including fractional band endpoints.
        identity = np.eye(len(self.frequencies))
        self.integration = np.array([
            [band_integral(self.frequencies, unit, *TRACK_BANDS[b]) for unit in identity]
            for b in self.bands
        ])
        raw_names = tuple(f"{ch}:log_{b}" for ch in config.channels for b in self.bands)
        vectors = {name: unit for name, unit in zip(raw_names, np.eye(len(raw_names)))}
        ratios = []
        for ch in config.channels:
            name = f"{ch}:log_beta_alpha"
            vectors[name] = vectors[f"{ch}:log_beta"] - vectors[f"{ch}:log_alpha"]
            ratios.append(name)
        if config.frontal_pair:
            left, right = config.frontal_pair
            vectors["FAA"] = vectors[f"{right}:log_alpha"] - vectors[f"{left}:log_alpha"]
        self.names = (config.measurements if config.measurements is not None else
                      tuple(ratios + (["FAA"] if config.frontal_pair else [])))
        if not self.names or len(set(self.names)) != len(self.names) or any(n not in vectors for n in self.names):
            raise ValueError("Measurement names must be unique available log powers, log ratios or FAA")
        self.design = np.array([vectors[n] for n in self.names])
        if np.linalg.matrix_rank(self.design) != len(self.names):
            raise ValueError("Measurements are linearly redundant; select an independent feature set")

    def extract(self, x: np.ndarray) -> SpectralWindow:
        cfg = self.config
        x = np.asarray(x, dtype=float)
        if x.ndim < 2 or x.shape[-2:] != (len(cfg.channels), cfg.window_n):
            raise ValueError("Expected (..., channels, window_samples) EEG")
        _, psd = welch_psd(x, cfg.fs, cfg.segment_n, cfg.overlap_n, cfg.nfft)
        power = np.einsum("...cf,bf->...cb", psd, self.integration)
        logs = np.log(np.maximum(power, 1e-12))
        flat = logs.reshape(*logs.shape[:-2], -1)
        return SpectralWindow(psd, power, logs, flat @ self.design.T)


def effective_welch_averages(config: SpectralConfig) -> float:
    """Overlap-only white-noise approximation; not a complete band-power DOF."""
    w = signal.windows.hann(config.segment_n, sym=False)
    step = config.segment_n - config.overlap_n
    m = 1 + (config.window_n - config.segment_n) // step
    correction = 1.0
    for lag in range(1, m):
        offset = lag * step
        if offset >= len(w):
            break
        rho = w[:-offset] @ w[offset:] / (w @ w)
        correction += 2 * (1 - lag / m) * rho**2
    return float(m / correction)


def _psd_matrix(matrix, floor=0.0):
    matrix = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, floor)) @ vectors.T


@dataclass
class NoiseCalibration:
    config: SpectralConfig
    names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    R_fft: np.ndarray
    lag1_fft: np.ndarray
    effective_dof: np.ndarray
    high_center: np.ndarray
    high_scale: np.ndarray
    draws: int
    seed: int
    synthesis_n: int
    spectral_segments: int

    @property
    def R_normalized(self):
        return self.R_fft / np.outer(self.scale, self.scale)

    def normalize(self, values):
        return (np.asarray(values) - self.center) / self.scale

    def colored_noise(self):
        """Stable AR(1) approximation: e_t = A e_(t-1) + u_t.

        Whitened singular values are capped below one so Q = R - A R A'
        is PSD despite finite bootstrap error. This is an approximation to
        the entire overlap autocovariance, not an exact whitening claim.
        """
        r = _psd_matrix(self.R_normalized, 1e-8)
        values, vectors = np.linalg.eigh(r)
        root = (vectors * np.sqrt(values)) @ vectors.T
        inverse_root = (vectors / np.sqrt(values)) @ vectors.T
        lag = self.lag1_fft / np.outer(self.scale, self.scale)
        u, singular, vh = np.linalg.svd(inverse_root @ lag @ inverse_root)
        contraction = (u * np.minimum(singular, 0.98)) @ vh
        a = root @ contraction @ inverse_root
        q = _psd_matrix(r - a @ r @ a.T)
        return r, a, q

    def to_dict(self):
        r, a, q = self.colored_noise()
        return {
            "use": "research_only", "method": "multivariate_gaussian_fft_parametric_bootstrap",
            "config": asdict(self.config), "bands_hz": TRACK_BANDS, "measurements": self.names,
            "center": self.center.tolist(), "scale": self.scale.tolist(),
            "R_fft_log_units": self.R_fft.tolist(), "R_fft_normalized": self.R_normalized.tolist(),
            "lag1_fft_log_units": self.lag1_fft.tolist(), "noise_stationary_covariance": r.tolist(),
            "noise_AR1": a.tolist(), "noise_process_covariance": q.tolist(),
            "effective_band_dof": self.effective_dof.tolist(),
            "gamma_log_variance_approximation": special.polygamma(1, self.effective_dof / 2).tolist(),
            "effective_welch_averages_overlap_only": effective_welch_averages(self.config),
            "high_center": self.high_center.tolist(), "high_scale": self.high_scale.tolist(),
            "bootstrap_draws": self.draws, "seed": self.seed, "synthesis_samples": self.synthesis_n,
            "spectral_segments": self.spectral_segments,
            "limitations": ["conditional on the estimated stationary Gaussian baseline spectrum",
                            "R_fft excludes additional model error and baseline-estimation uncertainty",
                            "AR1 approximates, rather than removes exactly, temporal dependence",
                            "CUSUM thresholds require held-out session calibration"],
        }


def fit_fft_noise(baseline: np.ndarray, extractor: SpectralExtractor, draws=1024, seed=0) -> NoiseCalibration:
    """Fit a multichannel FFT spectrum, synthesize independent *pairs* of windows,
    and estimate full log-feature covariance and adjacent-window covariance.

    Baseline must be a clean, contiguous (channels, samples) personal reference.
    Shared complex Gaussian Fourier draws preserve the estimated cross-spectrum.
    The identical Welch/taper/integration/log/feature transform is then applied.
    """
    cfg = extractor.config
    x = np.asarray(baseline, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(cfg.channels) or not np.isfinite(x).all():
        raise ValueError("Baseline must contain finite contiguous EEG for every configured channel")
    if not isinstance(draws, int) or draws < max(64, 4 * len(extractor.names)):
        raise ValueError("Use at least max(64, four times the measurement count) bootstrap draws")
    pair_n = cfg.window_n + cfg.hop_n
    synthesis_n = next_fast_len(2 * pair_n)
    if synthesis_n % 2:
        synthesis_n += 1
    if x.shape[-1] < synthesis_n:
        raise ValueError("Baseline is too short for the FFT noise model")
    segments = np.lib.stride_tricks.sliding_window_view(x, synthesis_n, axis=-1)[:, ::synthesis_n // 2]
    if segments.shape[1] < max(4, len(cfg.channels) + 1):
        raise ValueError("Need at least max(4, channels+1) long FFT segments in the clean baseline")
    w = signal.windows.hann(synthesis_n, sym=False)
    fft = np.fft.rfft((segments - segments.mean(axis=-1, keepdims=True)) * w, axis=-1)
    csd = np.einsum("cmf,dmf->fcd", fft, fft.conj()) / (segments.shape[1] * cfg.fs * (w @ w))
    # These are TWO-sided spectral-density values on the positive grid.
    # Complex interior coefficients generate both signs through irfft.
    eigenvalues, eigenvectors = np.linalg.eigh(csd)
    roots = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
    rng = np.random.default_rng(seed)
    first, second, powers = [], [], []
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        shape = (size, len(csd), len(cfg.channels))
        u = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
        # DC and Nyquist are real, with variance one, not one half.
        u[:, (0, -1)] = rng.normal(size=(size, 2, len(cfg.channels)))
        coefficients = np.einsum("fcd,bfd->bcf", roots, u) * np.sqrt(synthesis_n * cfg.fs)
        simulated = np.fft.irfft(coefficients, n=synthesis_n, axis=-1)
        left = extractor.extract(simulated[..., :cfg.window_n])
        right = extractor.extract(simulated[..., cfg.hop_n:pair_n])
        first.append(left.values)
        second.append(right.values)
        powers.append(left.band_power)
    first, second, powers = np.concatenate(first), np.concatenate(second), np.concatenate(powers)
    v0, v1 = first - first.mean(axis=0), second - second.mean(axis=0)
    r = (v0.T @ v0 + v1.T @ v1) / (2 * (draws - 1))
    lag = v1.T @ v0 / (draws - 1)
    dof = 2 * powers.mean(axis=0)**2 / np.maximum(powers.var(axis=0, ddof=1), 1e-30)

    baseline_values, high = [], []
    for start in range(0, x.shape[-1] - cfg.window_n + 1, cfg.hop_n):
        measured = extractor.extract(x[:, start:start + cfg.window_n])
        baseline_values.append(measured.values)
        high.append(measured.log_power[:, extractor.bands.index("high")])
    baseline_values, high = np.array(baseline_values), np.array(high)
    center, high_center = np.median(baseline_values, axis=0), np.median(high, axis=0)
    scale = 1.4826 * np.median(abs(baseline_values - center), axis=0)
    if np.any(scale < 1e-8):
        raise ValueError("Baseline has a constant measurement; remove it or supply a variable baseline")
    high_scale = np.maximum(1.4826 * np.median(abs(high - high_center), axis=0), 1e-6)
    return NoiseCalibration(cfg, extractor.names, center, scale, r, lag, dof, high_center,
                            high_scale, draws, seed, synthesis_n, segments.shape[1])


class KalmanCUSUM:
    """Augmented state [a, d, spectral_noise...] and innovation LLR CUSUM.

    Loadings relate normalized measurements to a user-defined research state.
    phi, Q, residual error and the change threshold are model parameters that
    cannot be inferred from FFT settings alone. One step is exactly one hop.
    """

    def __init__(self, calibration: NoiseCalibration, loadings, phi=0.98,
                 process_variance=0.01, drift_variance=1e-5, residual_covariance=None,
                 cusum_shift=1.0, cusum_threshold=8.0):
        self.calibration = calibration
        self.loadings = np.asarray(loadings, dtype=float)
        p = len(calibration.names)
        if self.loadings.shape != (p,) or not np.isfinite(self.loadings).all() or not np.any(self.loadings):
            raise ValueError("Supply one finite loading per measurement, with at least one nonzero loading")
        try:
            phi, process_variance, drift_variance, cusum_shift, cusum_threshold = np.asarray(
                [phi, process_variance, drift_variance, cusum_shift, cusum_threshold], dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Kalman and CUSUM parameters must be numeric") from exc
        if not np.isfinite([phi, process_variance, drift_variance, cusum_shift, cusum_threshold]).all():
            raise ValueError("Kalman and CUSUM parameters must be finite")
        if not 0 < phi <= 1 or min(process_variance, drift_variance) < 0 or min(cusum_shift, cusum_threshold) <= 0:
            raise ValueError("Require 0 < phi <= 1, nonnegative process variances and positive CUSUM parameters")
        r, a, q = calibration.colored_noise()
        residual = np.zeros((p, p)) if residual_covariance is None else np.asarray(residual_covariance, float)
        if (residual.shape != (p, p) or not np.isfinite(residual).all()
                or not np.allclose(residual, residual.T) or np.linalg.eigvalsh(residual).min() < -1e-10):
            raise ValueError("residual_covariance must be a finite symmetric PSD matrix in normalized units")
        self.parameters = {"loadings": dict(zip(calibration.names, self.loadings.tolist())),
                           "phi": float(phi), "process_variance": float(process_variance),
                           "drift_variance": float(drift_variance),
                           "residual_covariance": residual.tolist(), "cusum_shift": float(cusum_shift),
                           "cusum_threshold": float(cusum_threshold),
                           "initial_latent_mean": [0.0, 0.0], "initial_latent_covariance": [[1.0, 0.0], [0.0, 0.01]]}
        self.F = np.zeros((p + 2, p + 2))
        self.F[:2, :2] = [[phi, 1], [0, 1]]
        self.F[2:, 2:] = a
        self.Q = np.zeros_like(self.F)
        self.Q[:2, :2] = np.diag([process_variance, drift_variance])
        self.Q[2:, 2:] = q
        self.H = np.column_stack([self.loadings, np.zeros(p), np.eye(p)])
        self.R = residual + np.eye(p) * 1e-9  # Numerical floor, not physiological noise.
        self.x = np.zeros(p + 2)
        self.P = np.zeros_like(self.F)
        self.P[:2, :2] = np.diag([1.0, 0.01])
        self.P[2:, 2:] = r
        self.cusum = 0.0
        self.shift, self.threshold = cusum_shift, cusum_threshold

    def step(self, measurement=None):
        predicted = self.F @ self.x
        covariance = self.F @ self.P @ self.F.T + self.Q
        diagnostic = {"innovation": None, "innovation_covariance": None,
                      "projected_innovation_z": None, "nis": None, "log_likelihood_increment": None}
        if measurement is None:
            self.x, self.P = predicted, covariance
            self.cusum = 0.0  # Do not accumulate evidence across a quality gap.
        else:
            y = np.asarray(measurement, float)
            if y.shape != self.loadings.shape or not np.isfinite(y).all():
                raise ValueError("Measurement must be a complete finite normalized feature vector")
            innovation = y - self.H @ predicted
            s = self.H @ covariance @ self.H.T + self.R
            factor = cho_factor(s, lower=True)
            weighted = cho_solve(factor, innovation)
            precision_loading = cho_solve(factor, self.loadings)
            information = self.loadings @ precision_loading
            llr = self.shift * (self.loadings @ weighted) - 0.5 * self.shift**2 * information
            self.cusum = max(0.0, self.cusum + float(llr))
            gain = cho_solve(factor, self.H @ covariance).T
            self.x = predicted + gain @ innovation
            remainder = np.eye(len(predicted)) - gain @ self.H
            # Joseph form retains PSD with correlated measurements.
            self.P = remainder @ covariance @ remainder.T + gain @ self.R @ gain.T
            self.P = (self.P + self.P.T) / 2
            diagnostic = {"innovation": innovation.tolist(), "innovation_covariance": s.tolist(),
                          "projected_innovation_z": float(self.loadings @ weighted / np.sqrt(information)),
                          "nis": float(innovation @ weighted), "log_likelihood_increment": float(llr)}
        return {"latent_state": float(self.x[0]), "state_sd": float(np.sqrt(max(0, self.P[0, 0]))),
                "drift_per_min": float(self.x[1] * 60 / self.calibration.config.hop_s),
                "cusum": self.cusum, "change_flag": bool(measurement is not None and self.cusum >= self.threshold),
                "measurement_used": measurement is not None, **diagnostic}
