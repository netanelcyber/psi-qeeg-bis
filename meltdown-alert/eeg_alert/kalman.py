"""State-space arousal model with FFT-derived measurement noise.

Model (one latent arousal state a_t, per-window log-spectral measurements y_t):

    a_t   = phi * a_{t-1} + eta_t,          eta_t ~ N(0, q),   q = 1 - phi^2  (unit stationary variance)
    y_f,t = mu_f + lambda_f * a_t + eps_f,t, eps_f,t ~ N(0, R_f)

Measurement noise is split into two parts:

    R_f = sigma_spec,f^2 + sigma_phys,f^2

* sigma_spec^2 is the variance of a log band power estimated by Welch/FFT. A Welch
  periodogram bin is ~ P * chi2_{2M} / (2M), so ln(sum over K_b bins) has variance
  ~ 1 / (K_b * M). It is averaged over C channels (divide by C) and summed for
  differences of logs (e.g. ln beta - ln alpha).
* sigma_phys^2 = max(baseline variance - sigma_spec^2, 0.1 * baseline variance) is the
  remaining physiological variability, estimated during calibration.

lambda_f = w_f * sd_f maps one unit of latent arousal to a shift of w_f baseline SDs in
feature f (w_f = profile weight). The posterior mean a_t|t replaces the hand-weighted index.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import Config

# features used by the state-space model, all in log space
LOG_FEATURES = ("beta_alpha_ratio", "log_alpha_rel", "log_theta_rel", "frontal_asymmetry", "muscle_tension")
WEIGHT_KEY = {"log_alpha_rel": "alpha_rel", "log_theta_rel": "theta_rel"}


def welch_dof(cfg: Config) -> tuple[int, Dict[str, int]]:
    """Number of averaged Welch segments M and FFT bins K_b per band for the window config."""
    n = int(cfg.window_sec * cfg.fs)
    seg = int(min(n, cfg.fs))
    step = seg // 2                               # scipy default 50 % overlap
    m = 1 + (n - seg) // step
    df = cfg.fs / seg
    freqs = np.arange(seg // 2 + 1) * df
    k = {b: int(np.sum((freqs >= lo) & (freqs < hi))) for b, (lo, hi) in cfg.bands.items()}
    return m, k


def spectral_variance(cfg: Config) -> Dict[str, float]:
    """FFT-theoretical variance of each log feature (estimation noise only)."""
    m, k = welch_dof(cfg)
    c = max(1, len(cfg.channels))
    v = {b: 1.0 / (max(k[b], 1) * m) for b in k}
    total = 1.0 / (max(sum(k[b] for b in ("delta", "theta", "alpha", "beta")), 1) * m)
    return {
        "beta_alpha_ratio": (v["beta"] + v["alpha"]) / c,
        "log_alpha_rel": (v["alpha"] + total) / c,
        "log_theta_rel": (v["theta"] + total) / c,
        "frontal_asymmetry": 2 * v["alpha"],      # one channel each side, no averaging
        "muscle_tension": v["high"] / c,
    }


def apply_kalman_defaults(cfg: Config, keep_k: bool = False, keep_h: bool = False) -> Config:
    """Thresholds tuned for the Kalman output (synthetic tuning seeds 1-6; checked on held-out 50-61).

    The filtered output is smooth and autocorrelated, so the same z level carries more evidence
    than for the raw index; it therefore needs a higher imminent level and CUSUM reference.
    """
    cfg.model = "kalman"
    cfg.alert_level = 6.0
    if not keep_k:
        cfg.cusum_k = 2.0
    if not keep_h:
        cfg.cusum_h = 150.0
    return cfg


class KalmanArousal:
    def __init__(self, cfg: Config, phi: float = 0.98) -> None:
        self.cfg = cfg
        self.phi = phi
        self.q = 1.0 - phi ** 2
        self.spec = spectral_variance(cfg)
        self.samples: List[Dict[str, float]] = []
        self.mu: Dict[str, float] = {}
        self.lam: Dict[str, float] = {}
        self.R: Dict[str, float] = {}
        self.a = 0.0
        self.P = 1.0
        self.z_mu, self.z_sd = 0.0, 1.0
        self.ready = False

    @staticmethod
    def to_log(feats: Dict[str, float]) -> Dict[str, float]:
        out = {k: feats[k] for k in ("beta_alpha_ratio", "muscle_tension", "frontal_asymmetry") if k in feats}
        if "alpha_rel" in feats:
            out["log_alpha_rel"] = float(np.log(max(feats["alpha_rel"], 1e-9)))
        if "theta_rel" in feats:
            out["log_theta_rel"] = float(np.log(max(feats["theta_rel"], 1e-9)))
        return out

    def add_baseline(self, feats: Dict[str, float]) -> None:
        self.samples.append(self.to_log(feats))

    def finalize(self) -> None:
        for f in LOG_FEATURES:
            v = np.array([s[f] for s in self.samples if f in s])
            if v.size < 5:
                continue
            med = float(np.median(v))
            sd = max(float(np.median(np.abs(v - med))) * 1.4826, 1e-6)
            w = self.cfg.weights.get(WEIGHT_KEY.get(f, f), 0.0)
            if w == 0.0:
                continue
            var_total = sd ** 2
            spec = self.spec[f]
            phys = max(var_total - spec, 0.1 * var_total)
            self.mu[f] = med
            self.lam[f] = w * sd
            self.R[f] = spec + phys
        self.ready = True
        # Normalise the posterior to z units: run the filter over the calm baseline and use the
        # robust spread of the posterior mean, so thresholds share the same scale as the index model.
        post = [self._step(s) for s in self.samples]
        burn = post[min(len(post) // 4, 30):] or post
        self.z_mu = float(np.median(burn))
        self.z_sd = max(float(np.median(np.abs(np.asarray(burn) - self.z_mu))) * 1.4826, 1e-3)
        self.a, self.P = 0.0, 1.0

    def update(self, feats: Dict[str, float]) -> float:
        """One Kalman step; returns the posterior mean in baseline z units."""
        return (self._step(self.to_log(feats)) - self.z_mu) / self.z_sd

    def _step(self, y: Dict[str, float]) -> float:
        a_pred = self.phi * self.a
        p_pred = self.phi ** 2 * self.P + self.q
        info = 1.0 / p_pred
        num = a_pred / p_pred
        for f, lam in self.lam.items():
            if f in y:
                info += lam ** 2 / self.R[f]
                num += lam * (y[f] - self.mu[f]) / self.R[f]
        self.P = 1.0 / info
        self.a = self.P * num
        return self.a

    def adapt(self, feats: Dict[str, float], rate: float) -> None:
        y = self.to_log(feats)
        for f in self.mu:
            if f in y:
                self.mu[f] += rate * (y[f] - self.mu[f])

    def snr_weights(self) -> Dict[str, float]:
        """Effective weight of each feature in the posterior (lambda / R), normalised."""
        w = {f: abs(self.lam[f]) / self.R[f] for f in self.lam}
        s = sum(w.values()) or 1.0
        return {f: round(v / s, 3) for f, v in w.items()}
