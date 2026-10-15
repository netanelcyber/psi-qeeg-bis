#!/usr/bin/env python3
"""BTBR behavioral prediction demo.

Generates synthetic multi-animal, multi-day EEG with repetitive grooming
events, then runs the full prediction pipeline with held-out evaluation.

Usage:
    python examples/btbr_demo/run_prediction.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from psi_qeeg.behavior import (
    BehaviorEvent, run_behavior_prediction,
    build_prediction_dataset, MeltdownRiskScorer,
)

# --- Parameters ---
FS = 256.0
DURATION_S = 1800.0  # 30 minutes per session — enough baseline for 180s horizon
N_ANIMALS = 4
N_DAYS = 3
HORIZONS = [10, 30, 60, 180]
RNG = np.random.default_rng(2026)

# --- BTBR-like EEG synthesis ---

def make_btbr_eeg(n_samples: int, fs: float, rng: np.random.Generator,
                  pre_event: bool = False) -> np.ndarray:
    """Two-channel frontal EEG with BTBR spectral profile.

    pre_event=True increases theta power and theta-gamma coupling,
    simulating the neural buildup before a grooming bout.
    """
    t = np.arange(n_samples) / fs
    channels = []
    for _ in range(2):
        s = np.zeros(n_samples)
        delta_gain = rng.uniform(15, 30) if not pre_event else rng.uniform(20, 35)
        for f in np.arange(1.0, 4.0, 0.3):
            s += delta_gain * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        theta_gain = rng.uniform(8, 18) if not pre_event else rng.uniform(15, 28)
        for f in np.arange(4.0, 8.0, 0.5):
            s += theta_gain * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        alpha_gain = rng.uniform(2, 5)
        for f in np.arange(8.0, 13.0, 0.5):
            s += alpha_gain * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        for f in np.arange(13.0, 30.0, 1.0):
            s += rng.uniform(0.5, 2) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        # Theta-gamma coupling in pre-event windows
        if pre_event:
            theta_filt = 0
            for f in np.arange(5.0, 7.0, 0.5):
                theta_filt += 10 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
            gamma_mod = (1 + 0.5 * np.sin(2 * np.pi * 6 * t)) * rng.uniform(2, 5)
            for f in np.arange(30.0, 40.0, 2.0):
                s += gamma_mod * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        else:
            for f in np.arange(30.0, 40.0, 2.0):
                s += rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        freqs = np.fft.rfftfreq(n_samples, 1 / fs)
        freqs[0] = 1
        pink = np.fft.irfft(rng.standard_normal(len(freqs)) / np.sqrt(freqs), n=n_samples) * 5
        s += pink
        channels.append(s)
    return np.column_stack(channels)


def generate_session(animal_id: str, day: str, rng: np.random.Generator):
    """Generate one recording session with embedded grooming events."""
    n = round(DURATION_S * FS)
    n_events = rng.integers(3, 6)
    event_onsets = sorted(rng.uniform(300, DURATION_S - 60, size=n_events))
    # Enforce 5-minute minimum inter-event spacing for clear baseline
    filtered_onsets = [event_onsets[0]]
    for o in event_onsets[1:]:
        if o - filtered_onsets[-1] > 300:
            filtered_onsets.append(o)
    events = []
    eeg = make_btbr_eeg(n, FS, rng, pre_event=False)
    for onset in filtered_onsets:
        duration = rng.uniform(3.0, 12.0)
        events.append(BehaviorEvent(
            onset_s=onset,
            offset_s=onset + duration,
            event_type="repetitive_grooming",
            scorer="synthetic_demo",
            confidence=rng.uniform(0.8, 1.0),
            video_frame=round(onset * 30),  # 30 fps video
        ))
        # Inject pre-event neural signature over 180s before onset.
        # Gradual ramp: mild at -180s, strong at -10s.
        pre_start = max(0, round((onset - 180) * FS))
        pre_end = round(onset * FS)
        if pre_end > pre_start:
            pre_len = pre_end - pre_start
            pre_eeg = make_btbr_eeg(pre_len, FS, rng, pre_event=True)
            ramp = np.linspace(0, 1, pre_len) ** 2  # quadratic ramp
            ramp_2d = ramp[:, None]
            eeg[pre_start:pre_end] = (
                (1 - ramp_2d) * eeg[pre_start:pre_end] +
                ramp_2d * pre_eeg
            )

    return {
        "eeg": eeg,
        "fs": FS,
        "channel_names": ["frontal_L", "frontal_R"],
        "events": events,
        "animal_id": animal_id,
        "day": day,
    }


def main():
    print("=" * 65)
    print("  BTBR Behavioral Event Prediction Demo")
    print("  Repetitive grooming onset from frontal EEG")
    print("=" * 65)

    # Step 1: Generate synthetic multi-animal multi-day data
    print("\n[1] Generating synthetic BTBR recordings...")
    animals = [f"BTBR-{i:03d}" for i in range(1, N_ANIMALS + 1)]
    days = [f"day-{d}" for d in range(1, N_DAYS + 1)]

    recordings = []
    total_events = 0
    for animal in animals:
        for day in days:
            rec = generate_session(animal, day, RNG)
            recordings.append(rec)
            n_ev = len(rec["events"])
            total_events += n_ev
            print(f"    {animal} / {day}: {DURATION_S:.0f}s, {n_ev} grooming bouts")

    print(f"    Total: {len(recordings)} sessions, {total_events} events")
    print(f"    Event definition: {json.dumps(list(psi_qeeg_behavior_events().keys()))}")

    # Step 2: Feature extraction summary
    print("\n[2] Features extracted per window:")
    print("    - Spectral power (delta/theta/alpha/beta/gamma) per channel")
    print("    - PLV and coherence (all band pairs) between channels")
    print("    - Theta-gamma phase-amplitude coupling (MI) per channel")

    # Step 3: Run prediction at multiple horizons
    print(f"\n[3] Prediction horizons: {HORIZONS} seconds")
    print("    Constraint: only EEG recorded BEFORE window_end is used")
    print("    For 60s horizon: features stop 60s before event onset")

    out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "btbr_prediction"
    results = run_behavior_prediction(
        recordings=recordings,
        horizons_s=HORIZONS,
        window_s=4.0,
        stride_s=2.0,
        out=out_dir,
    )

    # Step 4: Report results
    print(f"\n[4] Cross-validation results:")
    print(f"    Animals: {results['dataset']['animals']}")
    print(f"    Days: {results['dataset']['days']}")
    print()
    for h_key, h_results in results["horizons"].items():
        print(f"  --- Horizon: {h_key} ---")
        for cv_type, r in h_results.items():
            if "error" in r:
                print(f"    {cv_type}: {r['error']}")
                continue
            print(f"    {cv_type}:")
            print(f"      Balanced accuracy:    {r['balanced_accuracy']:.3f}")
            print(f"      Sensitivity (recall): {r['sensitivity']:.3f}")
            print(f"      Specificity:          {r['specificity']:.3f}")
            print(f"      F1 score:             {r['f1']:.3f}")
            print(f"      AUC-ROC:              {r['auc_roc']:.3f}")
            print(f"      False alarms/hour:    {r['false_alarms_per_hour']:.1f}")
            print(f"      Samples: {r['n_positive']} pre-event, {r['n_negative']} baseline")
            print(f"      Per-fold false alarms:")
            for fold in r["folds"]:
                print(f"        {fold['held_out']}: FA/h={fold['false_alarms_per_hour']:.1f}  "
                      f"(TP={fold['tp']} FP={fold['fp']} TN={fold['tn']} FN={fold['fn']})")
        print()

    # Step 5: Continuous 0-100 meltdown risk score (180s horizon)
    print("\n[5] Meltdown Risk Scorer (0-100, horizon=180s)")
    print("    Training on all animals except BTBR-004...")

    # Build training samples for 180s horizon
    train_recs = [r for r in recordings if r["animal_id"] != "BTBR-004"]
    test_rec = [r for r in recordings if r["animal_id"] == "BTBR-004"][0]

    train_samples = []
    for rec in train_recs:
        train_samples.extend(build_prediction_dataset(
            eeg=rec["eeg"], fs=rec["fs"],
            channel_names=rec["channel_names"],
            events=rec["events"],
            animal_id=rec["animal_id"],
            day=rec["day"],
            horizons_s=[180],
            window_s=4.0, stride_s=2.0,
        ))

    scorer = MeltdownRiskScorer(horizon_s=180.0, window_s=4.0)
    scorer.fit(train_samples)

    # Score the held-out animal's first session continuously
    risk_scores = scorer.score_continuous(
        test_rec["eeg"], test_rec["fs"], test_rec["channel_names"], stride_s=2.0
    )

    print(f"    Held-out animal: {test_rec['animal_id']} / {test_rec['day']}")
    print(f"    Scored {len(risk_scores)} windows")
    print(f"    Event onsets: {[f'{e.onset_s:.0f}s' for e in test_rec['events']]}")
    print()
    print("    Time(s)  Score  Level       Top features")
    print("    -------  -----  ----------  --------------------------------")

    # Show scores around event onsets
    event_onsets = [e.onset_s for e in test_rec["events"]]
    shown = set()
    for rs in risk_scores:
        near_event = any(abs(rs.time_s - o) <= 30 or (0 < o - rs.time_s <= 180) for o in event_onsets)
        bucket = round(rs.time_s / 10) * 10
        if near_event and bucket not in shown:
            shown.add(bucket)
            feats_str = ", ".join(f"{k.split('__')[-1]}={v:.1f}" for k, v in list(rs.contributing_features.items())[:3])
            marker = " <-- EVENT" if any(abs(rs.time_s - o) < 5 for o in event_onsets) else ""
            print(f"    {rs.time_s:7.1f}  {rs.score:5d}  {rs.risk_level:10s}  {feats_str}{marker}")

    # Summary statistics
    scores_arr = [rs.score for rs in risk_scores]
    levels = [rs.risk_level for rs in risk_scores]
    print(f"\n    Score distribution: min={min(scores_arr)}, median={sorted(scores_arr)[len(scores_arr)//2]}, max={max(scores_arr)}")
    print(f"    Risk levels: low={levels.count('low')}, moderate={levels.count('moderate')}, "
          f"high={levels.count('high')}, critical={levels.count('critical')}")

    # Save risk timeline
    import pandas as pd
    risk_df = pd.DataFrame([
        {"time_s": rs.time_s, "score": rs.score, "risk_level": rs.risk_level,
         **rs.contributing_features}
        for rs in risk_scores
    ])
    risk_df.to_csv(out_dir / "risk_timeline.csv", index=False)
    print(f"\n    Risk timeline saved: {out_dir / 'risk_timeline.csv'}")

    print(f"\nResults saved to: {out_dir}/")
    print(f"  prediction_results.json — full metrics at all horizons")
    print(f"  prediction_samples.csv  — all labelled feature vectors")
    print(f"  risk_timeline.csv       — continuous 0-100 risk score")
    print(f"\nNote: {results['limitations']}")


def psi_qeeg_behavior_events():
    from psi_qeeg.behavior import DEFINED_EVENTS
    return DEFINED_EVENTS


if __name__ == "__main__":
    main()
