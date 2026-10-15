#!/usr/bin/env python3
"""BTBR 24-hour x 14-day continuous monitoring demo.

Simulates long-term EEG monitoring of BTBR mice with:
- 4 animals, 14 consecutive days, 24 hours per day
- Circadian variation in event frequency (more grooming in dark phase)
- Streaming feature extraction (memory-efficient, one hour at a time)
- 0-100 meltdown risk scoring with 180s horizon
- Leave-animal-out and leave-day-out cross-validation

Usage:
    python examples/btbr_demo/run_longterm.py
"""

import json
import sys
import time as _time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from psi_qeeg.behavior import (
    BehaviorEvent, extract_window_features, _welch_bands,
    MeltdownRiskScorer, PredictionSample, evaluate_prediction,
    DEFINED_EVENTS,
)


def _fast_window_features(eeg: np.ndarray, fs: float, channel_names: list[str]) -> dict:
    """Lightweight feature extraction: Welch bands only (skip PLV/coherence/PAC).

    ~5x faster than full extract_window_features. Sufficient for the
    longterm demo since the synthetic pre-event signature lives in
    theta/gamma power.
    """
    features = {}
    for j, name in enumerate(channel_names):
        x = eeg[:, j]
        if not np.isfinite(x).all():
            continue
        bands = _welch_bands(x, fs)
        for k, v in bands.items():
            features[f"{name}__{k}"] = v
    return features

# --- Parameters ---
FS = 256.0
N_ANIMALS = 4
N_DAYS = 14
HOURS_PER_DAY = 24
CHUNK_DURATION_S = 3600.0  # process one hour at a time
WINDOW_S = 4.0
STRIDE_S = 8.0  # wide stride for throughput on 1344-hour simulation
HORIZONS = [10, 30, 60, 180]


def make_eeg_chunk(n_samples: int, fs: float, rng: np.random.Generator,
                   intensity: float = 0.0) -> np.ndarray:
    """Generate one chunk of 2-channel BTBR EEG.

    intensity: 0.0 = baseline, 1.0 = full pre-event signature.
    """
    t = np.arange(n_samples) / fs
    channels = []
    for _ in range(2):
        s = np.zeros(n_samples)
        for f in np.arange(1.0, 4.0, 0.5):
            s += rng.uniform(12, 25) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        theta_base = rng.uniform(6, 12)
        theta_gain = theta_base + intensity * rng.uniform(10, 20)
        for f in np.arange(4.0, 8.0, 0.5):
            s += theta_gain * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        for f in np.arange(8.0, 13.0, 1.0):
            s += rng.uniform(1.5, 4) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        for f in np.arange(13.0, 30.0, 2.0):
            s += rng.uniform(0.3, 1.5) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        if intensity > 0.3:
            gamma_mod = (1 + intensity * 0.4 * np.sin(2 * np.pi * 6 * t))
            for f in np.arange(30.0, 40.0, 2.0):
                s += intensity * rng.uniform(1, 4) * gamma_mod * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        else:
            for f in np.arange(30.0, 40.0, 3.0):
                s += rng.uniform(0.1, 0.5) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        freqs = np.fft.rfftfreq(n_samples, 1 / fs)
        freqs[0] = 1
        pink = np.fft.irfft(rng.standard_normal(len(freqs)) / np.sqrt(freqs), n=n_samples) * 3
        s += pink
        channels.append(s)
    return np.column_stack(channels)


def circadian_event_rate(hour: int) -> float:
    """Events per hour with circadian modulation.

    BTBR mice are more active (more grooming) during the dark phase (19:00-07:00).
    """
    if 19 <= hour or hour < 7:  # dark phase
        return 4.0
    elif 7 <= hour < 10 or 16 <= hour < 19:  # transition
        return 2.0
    else:  # light phase
        return 1.0


def generate_hour_events(hour_start_s: float, hour: int, rng: np.random.Generator) -> list[BehaviorEvent]:
    """Generate grooming events for one hour with circadian variation."""
    rate = circadian_event_rate(hour)
    n_events = rng.poisson(rate)
    if n_events == 0:
        return []
    onsets = sorted(rng.uniform(hour_start_s + 60, hour_start_s + CHUNK_DURATION_S - 60, size=n_events))
    # Enforce minimum spacing
    filtered = [onsets[0]]
    for o in onsets[1:]:
        if o - filtered[-1] > 300:
            filtered.append(o)
    events = []
    for onset in filtered:
        dur = rng.uniform(3.0, 15.0)
        events.append(BehaviorEvent(
            onset_s=onset, offset_s=onset + dur,
            event_type="repetitive_grooming",
            scorer="synthetic_longterm",
            confidence=rng.uniform(0.85, 1.0),
            video_frame=round(onset * 30),
        ))
    return events


def process_hour_streaming(
    animal_id: str, day: int, hour: int, rng: np.random.Generator
) -> tuple[list[PredictionSample], list[dict], int]:
    """Process one hour: generate EEG, events, extract features, return samples + risk scores.

    Memory-efficient: only holds one hour of EEG at a time.
    """
    hour_start_s = (day * HOURS_PER_DAY + hour) * CHUNK_DURATION_S
    n = round(CHUNK_DURATION_S * FS)
    events = generate_hour_events(hour_start_s, hour, rng)
    n_events = len(events)

    # Generate EEG with pre-event signatures
    eeg = make_eeg_chunk(n, FS, rng, intensity=0.0)
    onsets = [e.onset_s for e in events]
    for onset in onsets:
        local_onset = onset - hour_start_s
        pre_start = max(0, round((local_onset - 180) * FS))
        pre_end = round(local_onset * FS)
        if pre_end > pre_start:
            pre_len = pre_end - pre_start
            pre_eeg = make_eeg_chunk(pre_len, FS, rng, intensity=1.0)
            ramp = np.linspace(0, 1, pre_len) ** 2
            ramp_2d = ramp[:, None]
            eeg[pre_start:pre_end] = (1 - ramp_2d) * eeg[pre_start:pre_end] + ramp_2d * pre_eeg

    # Extract features and build samples
    window_n = round(WINDOW_S * FS)
    stride_n = round(STRIDE_S * FS)
    max_horizon = max(HORIZONS)
    onset_arr = np.array([e.onset_s for e in events]) if events else np.array([])
    offset_arr = np.array([e.offset_s for e in events]) if events else np.array([])
    channel_names = ["frontal_L", "frontal_R"]

    samples = []
    risk_snapshots = []

    for start in range(0, n - window_n + 1, stride_n):
        end = start + window_n
        t_end = hour_start_s + end / FS
        chunk = eeg[start:end]
        if not np.isfinite(chunk).all():
            continue

        feats = _fast_window_features(chunk, FS, channel_names)
        day_str = f"day-{day + 1:02d}"

        for h in HORIZONS:
            if len(onset_arr) > 0:
                time_to_next = onset_arr - t_end
                future = time_to_next[(time_to_next >= 0) & (time_to_next <= h)]
                is_pre = len(future) > 0
                too_close = np.any((time_to_next >= 0) & (time_to_next <= 2 * max_horizon))
                in_event = np.any((onset_arr <= t_end) & (offset_arr >= t_end))
            else:
                is_pre = False
                too_close = False
                in_event = False

            if is_pre:
                label = 1
            elif not too_close and not in_event:
                label = 0
            else:
                continue

            samples.append(PredictionSample(
                animal_id=animal_id, day=day_str,
                window_end_s=t_end, horizon_s=h,
                label=label, features=feats,
            ))

        # Snapshot for risk timeline (every 60s)
        local_t = end / FS
        if round(local_t) % 60 < STRIDE_S + 0.1:
            risk_snapshots.append({
                "time_s": t_end, "day": day + 1, "hour": hour,
                "theta_power": feats.get("frontal_L__theta_power", 0),
                "delta_alpha_ratio": feats.get("frontal_L__delta_alpha_ratio", 0),
                "theta_gamma_mi": feats.get("frontal_L__theta_gamma_mi", 0),
            })

    return samples, risk_snapshots, n_events


def main():
    wall_start = _time.time()
    print("=" * 70)
    print("  BTBR Long-Term Monitoring: 24 hours x 14 days")
    print("  Continuous meltdown risk scoring (0-100)")
    print("=" * 70)

    animals = [f"BTBR-{i:03d}" for i in range(1, N_ANIMALS + 1)]
    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "btbr_longterm"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Streaming feature extraction
    print(f"\n[1] Generating & processing {N_ANIMALS} animals x {N_DAYS} days x {HOURS_PER_DAY}h")
    print(f"    Total simulated: {N_ANIMALS * N_DAYS * 24} hours of EEG")
    print(f"    Processing in 1-hour chunks (memory-efficient)")
    print(f"    Stride: {STRIDE_S}s (non-overlapping epochs for throughput)")
    print()

    all_samples = []
    all_risk = []
    event_counts = {}

    for ai, animal in enumerate(animals):
        rng = np.random.default_rng(2026 * 100 + ai)
        animal_events = 0
        for day in range(N_DAYS):
            day_events = 0
            for hour in range(HOURS_PER_DAY):
                samples, risk, n_ev = process_hour_streaming(animal, day, hour, rng)
                all_samples.extend(samples)
                all_risk.extend(risk)
                day_events += n_ev
                animal_events += n_ev
            day_str = f"day-{day + 1:02d}"
            elapsed = _time.time() - wall_start
            n_samples_so_far = len(all_samples)
            print(f"    {animal} / {day_str}: {day_events} events, "
                  f"{n_samples_so_far} total samples  ({elapsed:.0f}s elapsed)")
            day_events = 0
        event_counts[animal] = animal_events

    total_events = sum(event_counts.values())
    print(f"\n    Total: {len(all_samples)} samples, {total_events} events across all animals")

    # Phase 2: Cross-validated prediction
    print(f"\n[2] Cross-validated prediction (horizons: {HORIZONS}s)")
    results = {"horizons": {}, "dataset": {
        "animals": animals, "days": N_DAYS, "hours_per_day": HOURS_PER_DAY,
        "total_events": total_events, "events_per_animal": event_counts,
        "total_samples": len(all_samples),
    }}

    # Downsample negatives to 10:1 ratio for tractable SVM training
    MAX_NEG_RATIO = 10
    ds_rng = np.random.default_rng(42)

    def _downsample(samples_list, horizon):
        """Keep all positives, subsample negatives to MAX_NEG_RATIO:1."""
        h_all = [s for s in samples_list if s.horizon_s == horizon]
        pos = [s for s in h_all if s.label == 1]
        neg = [s for s in h_all if s.label == 0]
        max_neg = len(pos) * MAX_NEG_RATIO
        if len(neg) > max_neg:
            idx = ds_rng.choice(len(neg), size=max_neg, replace=False)
            neg = [neg[i] for i in idx]
        return pos + neg

    for h in HORIZONS:
        h_key = f"{h}s"
        h_samples_full = [s for s in all_samples if s.horizon_s == h]
        n_pos = sum(1 for s in h_samples_full if s.label == 1)
        n_neg = sum(1 for s in h_samples_full if s.label == 0)
        h_samples = _downsample(all_samples, h)
        print(f"\n  --- Horizon: {h_key} ({n_pos} pos / {n_neg} neg, "
              f"downsampled to {len(h_samples)}) ---")
        results["horizons"][h_key] = {"n_positive": n_pos, "n_negative": n_neg,
                                       "downsampled_to": len(h_samples)}

        # Leave-animal-out
        try:
            r = evaluate_prediction(h_samples, h, hold_out="animal")
            results["horizons"][h_key]["leave_animal_out"] = {
                "balanced_accuracy": r.balanced_accuracy,
                "sensitivity": r.sensitivity,
                "specificity": r.specificity,
                "f1": r.f1,
                "auc_roc": r.auc_roc,
                "false_alarms_per_hour": r.false_alarms_per_hour,
                "folds": r.folds,
            }
            print(f"    Leave-animal-out:  BA={r.balanced_accuracy:.3f}  "
                  f"Sens={r.sensitivity:.3f}  Spec={r.specificity:.3f}  "
                  f"F1={r.f1:.3f}  AUC={r.auc_roc:.3f}  FA/h={r.false_alarms_per_hour:.1f}")
        except ValueError as exc:
            print(f"    Leave-animal-out: {exc}")
            results["horizons"][h_key]["leave_animal_out"] = {"error": str(exc)}

        # Leave-day-out (sample 4 representative days)
        test_days = [f"day-{d:02d}" for d in [1, 5, 10, 14]]
        day_subset = _downsample(all_samples, h)
        try:
            r = evaluate_prediction(day_subset, h, hold_out="day")
            results["horizons"][h_key]["leave_day_out"] = {
                "balanced_accuracy": r.balanced_accuracy,
                "sensitivity": r.sensitivity,
                "specificity": r.specificity,
                "f1": r.f1,
                "auc_roc": r.auc_roc,
                "false_alarms_per_hour": r.false_alarms_per_hour,
            }
            print(f"    Leave-day-out:    BA={r.balanced_accuracy:.3f}  "
                  f"Sens={r.sensitivity:.3f}  Spec={r.specificity:.3f}  "
                  f"F1={r.f1:.3f}  AUC={r.auc_roc:.3f}  FA/h={r.false_alarms_per_hour:.1f}")
        except ValueError as exc:
            print(f"    Leave-day-out: {exc}")

    # Phase 3: Continuous risk scorer on held-out animal
    print(f"\n[3] Meltdown Risk Scorer (0-100, 180s horizon)")
    print(f"    Training: {animals[:3]}, held-out: {animals[3]}")
    train_all = [s for s in all_samples if s.animal_id != animals[3] and s.horizon_s == 180]
    train_pos = [s for s in train_all if s.label == 1]
    train_neg = [s for s in train_all if s.label == 0]
    max_neg = len(train_pos) * MAX_NEG_RATIO
    if len(train_neg) > max_neg:
        idx = ds_rng.choice(len(train_neg), size=max_neg, replace=False)
        train_neg = [train_neg[i] for i in idx]
    train_samples = train_pos + train_neg
    print(f"    Training samples: {len(train_pos)} pos + {len(train_neg)} neg = {len(train_samples)}")
    scorer = MeltdownRiskScorer(horizon_s=180.0, window_s=WINDOW_S)
    scorer.fit(train_samples)

    # Score held-out animal's risk snapshots
    held_out_risk = [r for r in all_risk if r.get("_animal") == animals[3]] if any("_animal" in r for r in all_risk) else []
    # Score from the raw features in all_risk
    test_180 = [s for s in all_samples if s.animal_id == animals[3] and s.horizon_s == 180]
    score_by_time = {}
    for s in test_180:
        feat_vec = np.array([[s.features.get(f, np.nan) for f in scorer._feature_names]])
        feat_vec = feat_vec[:, scorer._finite_mask]
        if np.isfinite(feat_vec).all():
            prob = scorer._model.predict_proba(feat_vec)[0]
            score = int(round(np.clip(float(prob[1]) * 100, 0, 100)))
            score_by_time[s.window_end_s] = {"score": score, "label": s.label}

    if score_by_time:
        scores_list = list(score_by_time.values())
        all_scores = [s["score"] for s in scores_list]
        pre_event_scores = [s["score"] for s in scores_list if s["label"] == 1]
        baseline_scores = [s["score"] for s in scores_list if s["label"] == 0]
        print(f"    Scored {len(scores_list)} windows on held-out animal")
        print(f"    Overall: min={min(all_scores)}, median={sorted(all_scores)[len(all_scores)//2]}, max={max(all_scores)}")
        if pre_event_scores:
            print(f"    Pre-event: min={min(pre_event_scores)}, median={sorted(pre_event_scores)[len(pre_event_scores)//2]}, max={max(pre_event_scores)}")
        if baseline_scores:
            print(f"    Baseline:  min={min(baseline_scores)}, median={sorted(baseline_scores)[len(baseline_scores)//2]}, max={max(baseline_scores)}")
        results["risk_scorer"] = {
            "held_out_animal": animals[3],
            "windows_scored": len(scores_list),
            "overall_median": int(sorted(all_scores)[len(all_scores) // 2]),
            "pre_event_median": int(sorted(pre_event_scores)[len(pre_event_scores) // 2]) if pre_event_scores else None,
            "baseline_median": int(sorted(baseline_scores)[len(baseline_scores) // 2]) if baseline_scores else None,
        }

    # Phase 4: Save results
    results["limitations"] = (
        "Synthetic demonstration. Pre-event signature is injected, so prediction performance "
        "reflects the signal-to-noise of the simulation, not real neural dynamics. "
        "Real deployment needs: (1) multi-day implanted EEG, (2) video-scored events, "
        "(3) prospective hold-out validation on new cohorts."
    )

    (out_dir / "longterm_results.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Risk timeline summary (hourly aggregates, not full CSV)
    if all_risk:
        risk_df = pd.DataFrame(all_risk)
        risk_df.to_csv(out_dir / "risk_snapshots.csv", index=False)

    elapsed = _time.time() - wall_start
    print(f"\n[4] Complete in {elapsed:.0f}s")
    print(f"    Results: {out_dir / 'longterm_results.json'}")
    print(f"    Risk snapshots: {out_dir / 'risk_snapshots.csv'}")
    print(f"\n    Simulated: {N_ANIMALS} animals x {N_DAYS} days x 24h = {N_ANIMALS * N_DAYS * 24}h of EEG")


if __name__ == "__main__":
    main()
