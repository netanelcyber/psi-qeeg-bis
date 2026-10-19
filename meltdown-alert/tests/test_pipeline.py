import numpy as np

from eeg_alert.config import Config
from eeg_alert.evaluate import evaluate
from eeg_alert.features import band_powers, extract, quality_ok
from eeg_alert.pipeline import run
from eeg_alert.sources import SyntheticEEG


def test_alpha_peak_recovered():
    cfg = Config()
    t = np.arange(int(2 * cfg.fs)) / cfg.fs
    x = np.tile(20 * np.sin(2 * np.pi * 10 * t), (4, 1)) + np.random.default_rng(0).normal(0, 1, (4, t.size))
    bp = band_powers(x, cfg)
    assert np.all(bp["alpha"] > 5 * bp["beta"])


def test_artifact_rejected():
    cfg = Config()
    x = np.random.default_rng(1).normal(0, 5, (4, int(2 * cfg.fs)))
    assert quality_ok(x, cfg)
    x[:, 100:150] += 500
    assert not quality_ok(x, cfg)
    assert extract(x, cfg) is None


def test_flatline_rejected():
    cfg = Config()
    assert not quality_ok(np.zeros((4, int(2 * cfg.fs))), cfg)


def _session(**kw):
    cfg = Config()
    src = SyntheticEEG(cfg, **kw)
    return cfg, list(run(src.stream(), cfg))


def test_two_level_warning_on_synthetic():
    cfg, steps = _session(seed=7)
    res = evaluate(steps, [22 * 60, 42 * 60], cfg.horizon_min)
    assert res["detected_early"] == 2
    assert res["detected_imminent"] == 2
    for ev in res["events"]:
        assert ev["early_warning_lead_min"] <= cfg.horizon_min + 1
        assert ev["early_warning_lead_min"] > ev["imminent_lead_min"] > 0
    # the brief distractor spike at minute 30 must not trigger an alert
    assert res["false_alarms"] == 0


def test_no_alerts_without_events():
    cfg, steps = _session(minutes=30.0, events_min=(), distractors_min=(12.0, 22.0), seed=3)
    assert not any(s.alert for s in steps)


# --- profiles and clinical context -------------------------------------------
from eeg_alert.clinical import BVCObservation, ClinicalContext, bvc_band  # noqa: E402
from eeg_alert.detector import CLINICAL_RISK, EARLY_WARNING  # noqa: E402
from eeg_alert.profiles import apply_profile  # noqa: E402


def test_bvc_bands():
    assert bvc_band(0) == "low"
    assert bvc_band(1) == bvc_band(2) == "moderate"
    assert bvc_band(3) == "very_high"
    o = BVCObservation(0.0, {"irritable": True, "boisterous": True, "verbal_threats": True})
    assert o.score == 3


def test_bvc_expires_after_24h():
    ctx = ClinicalContext([BVCObservation(0.0, {"irritable": True, "boisterous": True, "confused": True})])
    assert ctx.modifiers(3600)[2] == "very_high"
    assert ctx.modifiers(25 * 3600)[2] == "none"


def test_psychosis_profile_settings():
    cfg = apply_profile(Config(), "psychosis")
    assert cfg.horizon_min == 20.0
    assert cfg.weights["muscle_tension"] == 1.0
    assert "de-escalation" in cfg.early_msg


def _first_early(context):
    cfg = apply_profile(Config(), "psychosis")
    src = SyntheticEEG(cfg, seed=7)
    for s in run(src.stream(), cfg, context=context):
        if s.alert and s.alert.level == EARLY_WARNING:
            return s.t, s.alert
    return None, None


def test_high_bvc_gives_earlier_warning_and_notice():
    t_plain, _ = _first_early(None)
    ctx = ClinicalContext([BVCObservation(4 * 60.0, {"irritable": True, "boisterous": True,
                                                    "verbal_threats": True})])
    t_bvc, alert = _first_early(ctx)
    assert t_plain is not None and t_bvc is not None
    assert t_bvc < t_plain
    assert "BVC 3" in alert.message
    # one-time clinical notice is emitted
    cfg = apply_profile(Config(), "psychosis")
    steps = list(run(SyntheticEEG(cfg, seed=7, minutes=8, events_min=(), distractors_min=()).stream(), cfg, context=ctx))
    assert sum(1 for s in steps if s.alert and s.alert.level == CLINICAL_RISK) == 1


def test_bvc_zero_changes_nothing():
    t_plain, _ = _first_early(None)
    t_zero, _ = _first_early(ClinicalContext([BVCObservation(60.0, {})]))
    assert t_plain == t_zero


# --- sensor-level onset map ---------------------------------------------------
from eeg_alert.onset import describe, region_of  # noqa: E402


def test_onset_identifies_leading_region():
    cfg = Config()
    src = SyntheticEEG(cfg, seed=11, lead_channel="TP9", lead_min=4.0)
    first = next(s.alert for s in run(src.stream(), cfg) if s.alert and s.alert.onset)
    assert first.onset["first_channel"] == "TP9"
    assert first.onset["first_region"] == "left temporoparietal"
    assert "not a brain focus" in describe(first.onset)


def test_region_lookup_and_unknown():
    assert region_of("AF8") == ("right frontal", "R")
    assert region_of("X1") == ("X1", "?")
    assert "no channel-level onset" in describe(None)
