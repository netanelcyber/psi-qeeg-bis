"""Behavioral event prediction from continuous EEG.

Defines measurable behavioral events (repetitive grooming bouts in BTBR mice),
extracts neural features with strict temporal causality, and evaluates prediction
at multiple horizons with held-out animal/day cross-validation.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sig

from .signal import BANDS, band_integral


# ---------------------------------------------------------------------------
# 1. Behavioral event definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BehaviorEvent:
    """A single scored behavioral event with video-aligned onset."""
    onset_s: float
    offset_s: float
    event_type: str
    scorer: str
    confidence: float = 1.0
    video_frame: int | None = None


DEFINED_EVENTS = {
    "repetitive_grooming": {
        "description": "Self-directed grooming bout lasting >= 3 s with stereotyped rostro-caudal sequence",
        "minimum_duration_s": 3.0,
        "minimum_inter_event_s": 5.0,
        "scoring_source": "manual_video_annotation",
        "btbr_relevance": "Elevated repetitive grooming is a core BTBR behavioral phenotype (Silverman et al. 2010)",
    },
}


def validate_events(events: list[BehaviorEvent], recording_duration_s: float) -> list[BehaviorEvent]:
    """Validate and sort behavioral events chronologically."""
    if not events:
        raise ValueError("At least one behavioral event is required")
    for ev in events:
        if ev.onset_s < 0 or ev.onset_s >= recording_duration_s:
            raise ValueError(f"Event onset {ev.onset_s}s outside recording [0, {recording_duration_s})")
        if ev.offset_s <= ev.onset_s:
            raise ValueError(f"Event offset must follow onset: {ev.onset_s} -> {ev.offset_s}")
        if ev.event_type not in DEFINED_EVENTS:
            raise ValueError(f"Unknown event type: {ev.event_type}")
        defn = DEFINED_EVENTS[ev.event_type]
        if ev.offset_s - ev.onset_s < defn["minimum_duration_s"]:
            raise ValueError(f"Event shorter than minimum {defn['minimum_duration_s']}s")
    return sorted(events, key=lambda e: e.onset_s)


# ---------------------------------------------------------------------------
# 2. Neural feature extraction with temporal causality
# ---------------------------------------------------------------------------

def _welch_bands(x: np.ndarray, fs: float) -> dict[str, float]:
    """Band powers from a 1-D signal segment."""
    nperseg = min(round(2 * fs), len(x))
    f, p = sig.welch(x, fs=fs, window="hann", nperseg=nperseg,
                     noverlap=nperseg // 2, detrend="constant", scaling="density")
    if f[-1] < 40:
        usable = {n: (lo, min(hi, f[-1])) for n, (lo, hi) in BANDS.items() if lo < f[-1]}
    else:
        usable = BANDS
    powers = {}
    for name, (lo, hi) in usable.items():
        if lo < f[-1]:
            powers[name] = band_integral(f, p, lo, min(hi, f[-1]))
    total = sum(powers.values())
    out = {f"{n}_power": v for n, v in powers.items()}
    out.update({f"{n}_relative": v / total if total > 0 else 0.0 for n, v in powers.items()})
    out["total_power"] = total
    return out


def coherence_between(x: np.ndarray, y: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """Mean magnitude-squared coherence within a frequency band."""
    nperseg = min(round(2 * fs), len(x))
    f, cxy = sig.coherence(x, y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return float("nan")
    return float(np.mean(cxy[mask]))


def plv_between(x: np.ndarray, y: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """Phase-locking value between two signals in a band."""
    nyq = fs / 2
    lo, hi = max(band[0], 0.5), min(band[1], nyq - 1)
    if lo >= hi or len(x) < round(4 * fs):
        return float("nan")
    sos = sig.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xa = sig.hilbert(sig.sosfiltfilt(sos, x))
    ya = sig.hilbert(sig.sosfiltfilt(sos, y))
    trim = max(round(fs / 2), 1)
    delta = np.angle(xa[trim:-trim]) - np.angle(ya[trim:-trim])
    return float(abs(np.mean(np.exp(1j * delta))))


def theta_gamma_coupling(x: np.ndarray, fs: float,
                         theta_band=(4, 8), gamma_band=(30, 40)) -> float:
    """Phase-amplitude coupling: modulation index (Tort et al. 2010).

    Theta phase modulates gamma amplitude — a marker of local circuit
    processing relevant to ASD phenotypes.
    """
    nyq = fs / 2
    if gamma_band[1] > nyq - 1 or len(x) < round(4 * fs):
        return float("nan")
    sos_theta = sig.butter(4, theta_band, btype="bandpass", fs=fs, output="sos")
    sos_gamma = sig.butter(4, gamma_band, btype="bandpass", fs=fs, output="sos")
    theta_phase = np.angle(sig.hilbert(sig.sosfiltfilt(sos_theta, x)))
    gamma_amp = np.abs(sig.hilbert(sig.sosfiltfilt(sos_gamma, x)))
    n_bins = 18
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (theta_phase >= bin_edges[i]) & (theta_phase < bin_edges[i + 1])
        if mask.any():
            mean_amp[i] = gamma_amp[mask].mean()
    total = mean_amp.sum()
    if total <= 0:
        return 0.0
    p = mean_amp / total
    p = p[p > 0]
    h = -np.sum(p * np.log(p))
    h_max = np.log(n_bins)
    return float((h_max - h) / h_max)


def extract_window_features(eeg: np.ndarray, fs: float, channel_names: list[str]) -> dict:
    """Extract all features from a multi-channel EEG window.

    Returns a flat dict of feature_name -> value.
    """
    n_ch = eeg.shape[1]
    features = {}
    for j, name in enumerate(channel_names):
        x = eeg[:, j]
        if not np.isfinite(x).all():
            continue
        bands = _welch_bands(x, fs)
        for k, v in bands.items():
            features[f"{name}__{k}"] = v
        if fs / 2 > 40:
            features[f"{name}__theta_gamma_mi"] = theta_gamma_coupling(x, fs)
    for j in range(n_ch):
        for k in range(j + 1, n_ch):
            pair = f"{channel_names[j]}_{channel_names[k]}"
            xj, xk = eeg[:, j], eeg[:, k]
            if not (np.isfinite(xj).all() and np.isfinite(xk).all()):
                continue
            for band_name, (lo, hi) in BANDS.items():
                if hi > fs / 2:
                    continue
                features[f"pair_{pair}__coh_{band_name}"] = coherence_between(xj, xk, fs, (lo, hi))
                if len(xj) >= round(4 * fs):
                    features[f"pair_{pair}__plv_{band_name}"] = plv_between(xj, xk, fs, (lo, hi))
    return features


# ---------------------------------------------------------------------------
# 3. Prediction dataset builder with temporal causality
# ---------------------------------------------------------------------------

@dataclass
class PredictionSample:
    """One feature vector for prediction, with strict temporal provenance."""
    animal_id: str
    day: str
    window_end_s: float
    horizon_s: float
    label: int  # 1 = pre-event, 0 = baseline
    features: dict = field(default_factory=dict)


def build_prediction_dataset(
    eeg: np.ndarray,
    fs: float,
    channel_names: list[str],
    events: list[BehaviorEvent],
    animal_id: str,
    day: str,
    horizons_s: list[float] = (10, 30, 60),
    window_s: float = 4.0,
    stride_s: float = 2.0,
    recording_duration_s: float | None = None,
) -> list[PredictionSample]:
    """Build labelled samples with strict no-lookahead constraint.

    For each horizon H, a sample is labelled positive if a behavioral event
    starts within [window_end, window_end + H]. Only EEG up to window_end
    is used (causal). Negative samples come from periods with no event
    within 2*max(horizons) seconds.
    """
    if recording_duration_s is None:
        recording_duration_s = len(eeg) / fs
    events = validate_events(events, recording_duration_s)
    window_n = round(window_s * fs)
    stride_n = round(stride_s * fs)
    max_horizon = max(horizons_s)
    onsets = np.array([e.onset_s for e in events])
    samples = []
    for start in range(0, len(eeg) - window_n + 1, stride_n):
        end = start + window_n
        t_end = end / fs
        chunk = eeg[start:end]
        if not np.isfinite(chunk).all():
            continue
        feats = extract_window_features(chunk, fs, channel_names)
        for h in horizons_s:
            time_to_next = onsets - t_end
            future = time_to_next[(time_to_next >= 0) & (time_to_next <= h)]
            is_pre_event = len(future) > 0
            too_close_neg = np.any((time_to_next >= 0) & (time_to_next <= 2 * max_horizon))
            in_event = np.any((onsets <= t_end) & (np.array([e.offset_s for e in events]) >= t_end))
            if is_pre_event:
                label = 1
            elif not too_close_neg and not in_event:
                label = 0
            else:
                continue  # ambiguous zone — skip
            samples.append(PredictionSample(
                animal_id=animal_id, day=day,
                window_end_s=t_end, horizon_s=h,
                label=label, features=feats,
            ))
    return samples


# ---------------------------------------------------------------------------
# 4. Cross-validated prediction with held-out animals and days
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    horizon_s: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    f1: float
    auc_roc: float
    false_alarms_per_hour: float
    n_positive: int
    n_negative: int
    held_out_type: str  # "animal" or "day"
    folds: list[dict] = field(default_factory=list)


def evaluate_prediction(
    samples: list[PredictionSample],
    horizon: float,
    hold_out: str = "animal",
) -> PredictionResult:
    """Leave-one-animal-out or leave-one-day-out evaluation.

    Reports sensitivity, specificity, AUC-ROC, and false alarms/hour.
    """
    try:
        from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                     roc_auc_score, confusion_matrix)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise ValueError("Prediction requires: pip install 'psi-qeeg-bis[ml]'") from exc

    subset = [s for s in samples if s.horizon_s == horizon]
    if not subset:
        raise ValueError(f"No samples for horizon {horizon}s")
    feature_names = sorted(subset[0].features.keys())
    X = np.array([[s.features.get(f, np.nan) for f in feature_names] for s in subset])
    y = np.array([s.label for s in subset])
    if hold_out == "animal":
        groups = np.array([s.animal_id for s in subset])
    elif hold_out == "day":
        groups = np.array([s.day for s in subset])
    else:
        raise ValueError("hold_out must be 'animal' or 'day'")
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError(f"Need at least 2 {hold_out}s for held-out evaluation, got {len(unique_groups)}")
    finite_mask = np.isfinite(X).all(axis=0)
    X = X[:, finite_mask]
    feature_names = [f for f, ok in zip(feature_names, finite_mask) if ok]
    if X.shape[1] == 0:
        raise ValueError("No finite features available")
    predictions = np.full(len(y), -1, dtype=int)
    probas = np.full(len(y), np.nan)
    folds = []
    window_durations = np.array([2.0 for _ in subset])  # stride_s
    for held in unique_groups:
        mask_test = groups == held
        mask_train = ~mask_test
        X_tr, y_tr = X[mask_train], y[mask_train]
        X_te, y_te = X[mask_test], y[mask_test]
        if len(set(y_tr)) < 2:
            continue
        from sklearn.calibration import CalibratedClassifierCV
        base = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        model = CalibratedClassifierCV(base, ensemble=False)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)
        predictions[mask_test] = pred
        if prob.shape[1] == 2:
            probas[mask_test] = prob[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel() if len(set(y_te)) == 2 else (0, 0, 0, 0)
        total_baseline_hours = window_durations[mask_test][y_te == 0].sum() / 3600
        fa_rate = fp / total_baseline_hours if total_baseline_hours > 0 else float("nan")
        folds.append({
            "held_out": held, "n_test": int(mask_test.sum()),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "false_alarms_per_hour": round(fa_rate, 2),
        })

    valid = predictions >= 0
    if valid.sum() == 0:
        raise ValueError("No valid predictions produced")
    y_v, pred_v, prob_v = y[valid], predictions[valid], probas[valid]
    tn, fp, fn, tp = confusion_matrix(y_v, pred_v, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    total_baseline_h = window_durations[valid][y_v == 0].sum() / 3600
    fa_per_h = fp / total_baseline_h if total_baseline_h > 0 else float("nan")
    try:
        auc = float(roc_auc_score(y_v, prob_v))
    except ValueError:
        auc = float("nan")

    return PredictionResult(
        horizon_s=horizon,
        accuracy=float((pred_v == y_v).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y_v, pred_v)),
        sensitivity=sens,
        specificity=spec,
        f1=float(f1_score(y_v, pred_v, zero_division=0)),
        auc_roc=auc,
        false_alarms_per_hour=round(fa_per_h, 2),
        n_positive=int((y_v == 1).sum()),
        n_negative=int((y_v == 0).sum()),
        held_out_type=hold_out,
        folds=folds,
    )


# ---------------------------------------------------------------------------
# 5. Full pipeline orchestrator
# ---------------------------------------------------------------------------

def run_behavior_prediction(
    recordings: list[dict],
    horizons_s: list[float] = (10, 30, 60),
    window_s: float = 4.0,
    stride_s: float = 2.0,
    out: Path | None = None,
) -> dict:
    """Run the full behavioral prediction pipeline.

    Each recording dict must have: eeg (ndarray), fs (float),
    channel_names (list), events (list[BehaviorEvent]),
    animal_id (str), day (str).
    """
    all_samples = []
    for rec in recordings:
        samples = build_prediction_dataset(
            eeg=rec["eeg"], fs=rec["fs"],
            channel_names=rec["channel_names"],
            events=rec["events"],
            animal_id=rec["animal_id"],
            day=rec["day"],
            horizons_s=horizons_s,
            window_s=window_s,
            stride_s=stride_s,
        )
        all_samples.extend(samples)

    animals = sorted(set(s.animal_id for s in all_samples))
    days = sorted(set(s.day for s in all_samples))

    results = {"horizons": {}, "dataset": {
        "total_samples_per_horizon": {},
        "animals": animals,
        "days": days,
        "event_definition": DEFINED_EVENTS,
        "window_s": window_s,
        "stride_s": stride_s,
    }}

    for h in horizons_s:
        h_key = f"{h}s"
        h_samples = [s for s in all_samples if s.horizon_s == h]
        results["dataset"]["total_samples_per_horizon"][h_key] = {
            "positive": sum(1 for s in h_samples if s.label == 1),
            "negative": sum(1 for s in h_samples if s.label == 0),
        }
        results["horizons"][h_key] = {}
        if len(animals) >= 2:
            try:
                r = evaluate_prediction(all_samples, h, hold_out="animal")
                results["horizons"][h_key]["leave_animal_out"] = asdict(r)
            except ValueError as exc:
                results["horizons"][h_key]["leave_animal_out"] = {"error": str(exc)}
        if len(days) >= 2:
            try:
                r = evaluate_prediction(all_samples, h, hold_out="day")
                results["horizons"][h_key]["leave_day_out"] = asdict(r)
            except ValueError as exc:
                results["horizons"][h_key]["leave_day_out"] = {"error": str(exc)}

    results["limitations"] = (
        "Synthetic demonstration data. Real BTBR behavioral prediction requires: "
        "(1) video-scored events by trained raters with inter-rater reliability, "
        "(2) multi-day recordings from multiple animals, "
        "(3) prospective validation on unseen cohorts."
    )

    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "prediction_results.json").write_text(
            json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
        )
        rows = []
        for s in all_samples:
            row = {"animal_id": s.animal_id, "day": s.day, "window_end_s": s.window_end_s,
                   "horizon_s": s.horizon_s, "label": s.label}
            row.update(s.features)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "prediction_samples.csv", index=False)

    return results


# ---------------------------------------------------------------------------
# 6. Continuous meltdown risk score (0-100) for 180-second horizon
# ---------------------------------------------------------------------------

@dataclass
class MeltdownRiskScore:
    """A single timestamped risk assessment."""
    time_s: float
    score: int  # 0-100
    risk_level: str  # low / moderate / high / critical
    contributing_features: dict


RISK_LEVELS = {(0, 25): "low", (25, 50): "moderate", (50, 75): "high", (75, 101): "critical"}


def _risk_level(score: int) -> str:
    for (lo, hi), level in RISK_LEVELS.items():
        if lo <= score < hi:
            return level
    return "unknown"


class MeltdownRiskScorer:
    """Train on labelled data, then score live EEG windows 0-100.

    The score represents calibrated probability of a behavioral event
    (repetitive grooming / meltdown) occurring within the next 180 seconds,
    scaled to 0-100.
    """

    def __init__(self, horizon_s: float = 180.0, window_s: float = 4.0):
        self.horizon_s = horizon_s
        self.window_s = window_s
        self._model = None
        self._feature_names = None
        self._finite_mask = None

    def fit(self, samples: list[PredictionSample]):
        """Train the risk scorer from labelled prediction samples."""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
        except ImportError as exc:
            raise ValueError("Risk scoring requires: pip install 'psi-qeeg-bis[ml]'") from exc

        subset = [s for s in samples if s.horizon_s == self.horizon_s]
        if len(subset) < 10:
            raise ValueError(f"Need at least 10 samples for horizon {self.horizon_s}s, got {len(subset)}")

        self._feature_names = sorted(subset[0].features.keys())
        X = np.array([[s.features.get(f, np.nan) for f in self._feature_names] for s in subset])
        y = np.array([s.label for s in subset])

        if len(set(y)) < 2:
            raise ValueError("Need both positive and negative samples")

        self._finite_mask = np.isfinite(X).all(axis=0)
        X = X[:, self._finite_mask]

        base = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        self._model = CalibratedClassifierCV(base, ensemble=False)
        self._model.fit(X, y)

    def score_window(self, eeg_window: np.ndarray, fs: float,
                     channel_names: list[str], time_s: float) -> MeltdownRiskScore:
        """Score a single EEG window. Returns 0-100 risk score."""
        if self._model is None:
            raise ValueError("Scorer not fitted; call fit() first")

        feats = extract_window_features(eeg_window, fs, channel_names)
        x = np.array([[feats.get(f, np.nan) for f in self._feature_names]])
        x = x[:, self._finite_mask]

        if not np.isfinite(x).all():
            return MeltdownRiskScore(time_s=time_s, score=0,
                                     risk_level="unknown",
                                     contributing_features={"error": "non-finite features"})

        prob = self._model.predict_proba(x)[0]
        risk_prob = float(prob[1]) if prob.shape[0] == 2 else 0.0
        score = int(round(np.clip(risk_prob * 100, 0, 100)))

        top_feats = {}
        feat_vals = {self._feature_names[i]: float(feats.get(self._feature_names[i], 0))
                     for i in range(len(self._feature_names)) if self._finite_mask[i]}
        for key in ("frontal_L__theta_power", "frontal_L__delta_alpha_ratio",
                     "frontal_L__theta_gamma_mi", "pair_frontal_L_frontal_R__coh_theta"):
            if key in feat_vals:
                top_feats[key] = feat_vals[key]

        return MeltdownRiskScore(
            time_s=time_s,
            score=score,
            risk_level=_risk_level(score),
            contributing_features=top_feats,
        )

    def score_continuous(self, eeg: np.ndarray, fs: float,
                         channel_names: list[str],
                         stride_s: float = 2.0) -> list[MeltdownRiskScore]:
        """Score an entire recording continuously."""
        window_n = round(self.window_s * fs)
        stride_n = round(stride_s * fs)
        scores = []
        for start in range(0, len(eeg) - window_n + 1, stride_n):
            end = start + window_n
            chunk = eeg[start:end]
            if not np.isfinite(chunk).all():
                continue
            t = end / fs
            scores.append(self.score_window(chunk, fs, channel_names, t))
        return scores
