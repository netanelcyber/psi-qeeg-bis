"""Engineering regression tests; no clinical performance is estimated here."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from eeg_alert.config import Config
from eeg_alert.detector import Detector, LOW_QUALITY
from eeg_alert.pipeline import _windows, run
from eeg_alert.sources import CSVSource
from eeg_alert.timing import EEGChunk, session_seconds
from eeg_alert.wearable import (LIMITS, Measurement, WearableConfig, WearableProcessor,
                                WearableReplay, load_measurements)

FIXTURES = Path(__file__).parent / "fixtures"


def processor(extra=(), **kw):
    base = [Measurement(t, "heart_rate_bpm", 70 + t % 2, 0.9) for t in range(20)]
    return WearableProcessor(WearableReplay([*base, *extra]), 20, WearableConfig(**kw))


def eeg(seconds=42, cfg=None):
    cfg = cfg or Config(fs=128, baseline_sec=20, adapt_baseline=False)
    times = np.arange(int(seconds * cfg.fs)) / cfg.fs
    rng = np.random.default_rng(2)
    x = np.tile(10 * np.sin(2 * np.pi * 10 * times), (4, 1)) + rng.normal(0, 1, (4, len(times)))
    return cfg, times, x


def chunks(x, n=128):
    return [x[:, i:i+n] for i in range(0, x.shape[1], n)]


def test_multisensor_fixtures_all_metrics_and_quality():
    values = load_measurements(FIXTURES / "cardiorespiratory.csv")
    values += load_measurements(FIXTURES / "temperature_motion.csv")
    p = WearableProcessor(WearableReplay(values), 25)
    calibrated = p.update(31)
    assert all(x["status"] == "ok" for x in calibrated.metrics.values())
    assert len(calibrated.active_metrics) == 3  # temperature/motion remain visible as context
    assert calibrated.index > 0
    bad = p.update(41)
    assert bad.metrics["heart_rate_bpm"]["status"] == "low_quality"
    assert bad.metrics["respiration_bpm"]["status"] == "invalid_value"
    assert bad.metrics["skin_temperature_c"]["status"] == "out_of_range"
    assert bad.metrics["movement_g"]["status"] == "ok"
    assert p.update(59).active_metrics == ("hrv_rmssd_ms",)
    assert p.update(60.01).active_metrics == ("heart_rate_bpm",)


def test_future_measurements_never_enter_calibration_or_index():
    p = processor([Measurement(25, "heart_rate_bpm", 140, 0.9)])
    early = p.update(20)
    assert early.metrics["heart_rate_bpm"]["baseline_samples"] == 20
    assert early.metrics["heart_rate_bpm"]["sample_t"] == 19
    assert p.update(24).metrics["heart_rate_bpm"]["sample_t"] == 19
    late = p.update(25)
    assert late.metrics["heart_rate_bpm"]["baseline_samples"] == 20
    assert late.index > early.index


def test_reusing_one_sample_does_not_create_a_baseline():
    p = WearableProcessor(WearableReplay([Measurement(0, "heart_rate_bpm", 70)]), 20)
    for t in range(21):
        frame = p.update(t)
    assert frame.metrics["heart_rate_bpm"]["baseline_samples"] == 1
    assert frame.index is None
    assert frame.metrics["heart_rate_bpm"]["status"] == "stale"


@pytest.mark.parametrize("value,quality,status", [
    (float("nan"), 1, "invalid_value"), (float("inf"), 1, "invalid_value"),
    (400, 1, "out_of_range"), (0, 1, "out_of_range"), (150, 0.1, "low_quality"),
])
def test_bad_newest_sample_never_falls_back_to_old_good_value(value, quality, status):
    p = processor([Measurement(21, "heart_rate_bpm", value, quality)])
    p.update(20)
    frame = p.update(21)
    assert frame.metrics["heart_rate_bpm"]["status"] == status
    assert frame.index is None and frame.fusion_weight == 0
    json.dumps(asdict(frame), allow_nan=False)


def test_stale_and_missing_channels_are_explicit_and_not_zero_filled():
    p = processor()
    frame = p.update(34)  # age == 15 is allowed
    assert frame.metrics["heart_rate_bpm"]["status"] == "ok"
    frame = p.update(34.01)
    assert frame.metrics["heart_rate_bpm"]["status"] == "stale"
    assert frame.metrics["hrv_rmssd_ms"]["status"] == "missing"
    assert frame.metrics["hrv_rmssd_ms"]["z"] is None
    assert frame.index is None


def test_unknown_quality_is_preserved_as_unknown():
    vals = [Measurement(t, "heart_rate_bpm", 70) for t in range(21)]
    frame = WearableProcessor(WearableReplay(vals), 20).update(20)
    row = frame.metrics["heart_rate_bpm"]
    assert row["status"] == "ok" and row["quality"] is None
    assert row["quality_supplied"] is False


def test_uncalibrated_late_arriving_channel_cannot_drive_index():
    p = processor([Measurement(25, "hrv_rmssd_ms", 1, 1)])
    p.update(20)
    frame = p.update(25)
    assert frame.metrics["hrv_rmssd_ms"]["status"] == "uncalibrated"
    assert "hrv_rmssd_ms" not in frame.active_metrics


def test_loss_of_channel_does_not_amplify_remaining_weight():
    vals = []
    for k, base in (("heart_rate_bpm", 70), ("respiration_bpm", 14)):
        vals.extend(Measurement(t, k, base) for t in range(20))
        vals.append(Measurement(20, k, base * 1.1))
    vals.append(Measurement(21, "respiration_bpm", 0))
    p = WearableProcessor(WearableReplay(vals), 20)
    full = p.update(20)
    partial = p.update(21)
    assert partial.index == pytest.approx(full.index / 2)
    assert partial.coverage == 0.5


def test_temperature_and_motion_can_be_explicitly_weighted():
    cfg = WearableConfig()
    cfg.weights["skin_temperature_c"] = 0.2
    cfg.weights["movement_g"] = 0.3
    vals = load_measurements(FIXTURES / "temperature_motion.csv")
    frame = WearableProcessor(WearableReplay(vals), 25, cfg).update(31)
    assert frame.active_metrics == ("skin_temperature_c", "movement_g")
    assert frame.index > 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_supplied_quality_rejected(value):
    with pytest.raises(ValueError, match="quality"):
        Measurement(0, "heart_rate_bpm", 70, value)


@pytest.mark.parametrize("times", [[0, 0], [2, 1]])
def test_duplicate_or_out_of_order_metric_samples_rejected(times):
    with pytest.raises(ValueError, match="strictly increasing"):
        WearableReplay(Measurement(t, "heart_rate_bpm", 70) for t in times)


def test_two_sensors_for_one_metric_are_not_silently_mixed():
    with pytest.raises(ValueError, match="Multiple sensors"):
        WearableReplay([Measurement(0, "heart_rate_bpm", 70, sensor="A"),
                        Measurement(1, "heart_rate_bpm", 80, sensor="B")])


@pytest.mark.parametrize("text", [
    "heart_rate_bpm\n70\n", "timestamp,Heart Rate\n0,70\n", "timestamp,metric,value\n0,HR,70\n",
    "timestamp,heart_rate_bpm\n0,nope\n", "timestamp,heart_rate_bpm\n0,70,extra\n",
    "timestamp,heart_rate_bpm,heart_rate_bpm\n0,70,71\n", "timestamp,heart_rate_bpm\n",
])
def test_malformed_or_unmapped_vendor_csv_fails_clearly(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_measurements(path)


def test_shared_origin_timezones_and_explicit_clock_correction(tmp_path):
    origin = "2026-10-19T12:00:00+03:00"
    assert session_seconds("2026-10-19T09:00:05Z", origin) == 5
    path = tmp_path / "absolute.csv"
    path.write_text("timestamp,heart_rate_bpm\n2026-10-19T09:00:05Z,70\n")
    assert load_measurements(path, session_start=origin, offset_sec=-2)[0].t == 3
    with pytest.raises(ValueError, match="session-start"):
        load_measurements(path)
    with pytest.raises(ValueError, match="timezone"):
        session_seconds("2026-10-19T09:00:05", origin)
    with pytest.raises(ValueError, match="finite"):
        session_seconds("nan")


def test_wide_channel_quality_and_blank_sample(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("timestamp,heart_rate_bpm,hrv_rmssd_ms,quality,hrv_rmssd_ms_quality\n"
                    "0,70,,0.9,0.1\n")
    vals = load_measurements(path)
    assert vals[0].quality == 0.9 and vals[1].quality == 0.1
    assert np.isnan(vals[1].value)


@pytest.mark.parametrize("kw", [{"fusion_weight": 1.1}, {"fusion_weight": float("nan")},
                                {"min_quality": -1}, {"min_baseline_samples": 1},
                                {"max_age_sec": {}}, {"weights": {}}])
def test_invalid_wearable_configuration_rejected(kw):
    with pytest.raises(ValueError):
        WearableConfig(**kw)


def test_source_contract_rejects_future_observations():
    class BadSource:
        metrics = ("heart_rate_bpm",)

        def through(self, t):
            return {"heart_rate_bpm": [Measurement(t + 1, "heart_rate_bpm", 70)]}

    with pytest.raises(ValueError, match="future"):
        WearableProcessor(BadSource(), 20).update(0)


@pytest.mark.parametrize("model", ["index", "kalman"])
def test_missing_wearable_input_preserves_eeg_only_results_exactly(model):
    cfg, _, x = eeg()
    cfg.model = model
    plain = list(run(chunks(x), cfg))
    absent = list(run(chunks(x), cfg, wearable=WearableProcessor(WearableReplay([]), 20)))
    for a, b in zip(plain, absent):
        assert a.t == b.t and a.state == b.state and a.raw_index == b.raw_index
        assert a.index == b.index and a.cusum == b.cusum and a.alert == b.alert
        assert a.wearable is None and b.wearable["fusion_weight"] == 0


@pytest.mark.parametrize("model", ["index", "kalman"])
def test_zero_fusion_weight_preserves_eeg_even_during_artifacts_and_dropouts(model):
    cfg, _, x = eeg()
    cfg.model = model
    x[:, 30*128:31*128] += 500
    plain = list(run(chunks(x), cfg))
    disabled = list(run(chunks(x), cfg, wearable=processor(fusion_weight=0)))
    assert [s.raw_index for s in plain] == [s.raw_index for s in disabled]
    assert [s.index for s in plain] == [s.index for s in disabled]
    assert [s.cusum for s in plain] == [s.cusum for s in disabled]


def test_fusion_changes_index_but_bad_quality_cannot_drive_alerts():
    cfg, _, x = eeg()
    good = list(run(chunks(x), cfg, wearable=processor([Measurement(21, "heart_rate_bpm", 140, 1)])))
    plain = list(run(chunks(x), cfg))
    assert good[22].raw_index > plain[22].raw_index
    bad_vals = [Measurement(t, "heart_rate_bpm", 240, 0.1) for t in range(42)]
    bad = list(run(chunks(x), cfg, wearable=WearableProcessor(WearableReplay(bad_vals), 20)))
    assert [s.raw_index for s in bad] == [s.raw_index for s in plain]
    assert [s.alert for s in bad] == [s.alert for s in plain]


def test_wearable_cannot_bypass_eeg_quality_gate():
    cfg = Config(fs=128, baseline_sec=20)
    bad_eeg = np.zeros((4, 40 * 128))
    steps = list(run(chunks(bad_eeg), cfg, wearable=processor([Measurement(25, "heart_rate_bpm", 240, 1)])))
    assert not any(s.alert for s in steps)
    assert all(s.raw_index is None and s.wearable["fusion_weight"] == 0 for s in steps)


def test_availability_change_breaks_cusum_and_sustained_level_timers():
    cfg, _, x = eeg()
    p = processor([Measurement(21, "heart_rate_bpm", 140, 1),
                   Measurement(30, "heart_rate_bpm", 140, 0.1)])
    steps = list(run(chunks(x), cfg, wearable=p))
    before = next(s for s in steps if s.t == 29)
    after = next(s for s in steps if s.t == 30)
    assert before.cusum > 1
    assert after.cusum < before.cusum
    assert after.wearable["active_metrics"] == []


def test_reset_evidence_preserves_refractory_and_calibration():
    det = Detector(Config())
    det.level, det.cusum, det.early_since, det.alert_since = 4, 200, 10, 20
    det.last_alert_t["IMMINENT"] = 30
    det.baseline.ready = True
    det.reset_evidence()
    assert det.level is None and det.cusum == 0
    assert det.early_since is det.alert_since is None
    assert det.last_alert_t["IMMINENT"] == 30 and det.baseline.ready


def test_gap_reset_discards_kalman_state_without_losing_calibrated_coefficients():
    det = Detector(Config(model="kalman"))
    det.kf.a, det.kf.P = 4.0, 0.2
    det.kf.mu = {"muscle_tension": 1.0}
    det.kf.ready = True
    det.reset_evidence()
    assert det.kf.a == 0 and det.kf.P == 1
    assert det.kf.mu == {"muscle_tension": 1.0} and det.kf.ready


def test_regular_timestamped_eeg_matches_sample_clock():
    cfg, times, x = eeg()
    plain = list(run(chunks(x), cfg))
    timed = list(run([EEGChunk(x, times)], cfg))
    assert [s.raw_index for s in plain] == [s.raw_index for s in timed]
    assert [s.t for s in plain] == [s.t for s in timed]


def test_eeg_gap_never_creates_windows_across_missing_time_or_compresses_clock():
    cfg, times, x = eeg(10)
    mask = (times < 4) | (times >= 7)
    ts, data = times[mask], x[:, mask]
    windows = list(_windows([EEGChunk(data[:, :600], ts[:600]),
                             EEGChunk(data[:, 600:], ts[600:])], cfg))
    assert [t for t, _, _ in windows] == [2, 3, 4, 9, 10]
    assert [t for t, _, gap in windows if gap] == [9]


@pytest.mark.parametrize("times", [np.array([0, 0]), np.array([1, 0]), np.array([0, np.nan])])
def test_invalid_eeg_timestamps_rejected(times):
    with pytest.raises(ValueError, match="timestamps"):
        list(_windows([EEGChunk(np.ones((4, 2)), times)], Config(fs=128)))


def test_timestamped_csv_requires_all_eeg_channels_and_timestamp(tmp_path):
    cfg = Config(fs=128)
    path = tmp_path / "eeg.csv"
    path.write_text("timestamp,TP9,AF7,AF8\n0,1,2,3\n")
    with pytest.raises(ValueError, match="every configured"):
        CSVSource(cfg, str(path), timestamped=True)
    path.write_text("TP9,AF7,AF8,TP10\n1,2,3,4\n")
    with pytest.raises(ValueError, match="timestamp"):
        CSVSource(cfg, str(path), timestamped=True)


@pytest.mark.parametrize("text", ["", "timestamp,TP9,AF7,AF8,TP10\n",
                                "timestamp,TP9,AF7,AF8,TP10\n0,1,2\n"])
def test_empty_or_truncated_eeg_csv_fails_clearly(tmp_path, text):
    path = tmp_path / "eeg.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match="EEG CSV"):
        CSVSource(Config(), str(path), timestamped=True)


def test_fused_artifact_window_is_low_quality_and_does_not_accumulate_evidence():
    cfg, _, x = eeg()
    x[:, 30*128:31*128] += 500
    steps = list(run(chunks(x), cfg, wearable=processor([Measurement(21, "heart_rate_bpm", 140, 1)])))
    rejected = next(s for s in steps if s.t == 31)
    assert rejected.state == LOW_QUALITY
    assert rejected.alert is None and rejected.cusum == 0
