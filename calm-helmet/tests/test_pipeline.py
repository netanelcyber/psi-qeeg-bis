import dataclasses

import numpy as np

from calm_helmet.config import Config
from calm_helmet.detector import detect
from calm_helmet.evaluate import score
from calm_helmet.fnirs import short_separation_clean
from calm_helmet.fusion import fuse
from calm_helmet.pipeline import prepare_streams, run
from calm_helmet.simulate import simulate_session


def test_short_separation_removes_scalp_component():
    rng = np.random.default_rng(1)
    T = 1200
    scalp = np.cumsum(rng.normal(0, 0.05, T)) + np.sin(np.arange(T) / 9)
    brain = np.zeros(T)
    brain[800:] = np.linspace(0, 2, 400)
    long_ch = brain + 1.3 * scalp + rng.normal(0, 0.05, T)
    short_ch = scalp + rng.normal(0, 0.02, T)
    clean, beta = short_separation_clean(long_ch, short_ch, fit_sec=300)
    assert abs(beta - 1.3) < 0.1
    assert np.corrcoef(clean - clean.mean(), brain)[0, 1] > 0.9


def test_fusion_survives_lost_fnirs_contact():
    s = simulate_session(n_events=1, fnirs_dropout=0.2, seed=7)
    streams = prepare_streams(s)
    index, n_mod, _ = fuse(streams)
    after = index[Config().baseline_sec:]
    assert np.isfinite(after).all()
    assert n_mod.min() >= 2


def test_motion_gates_autonomic_features():
    s = simulate_session(n_events=0, n_spikes=0, n_exercise=1, seed=3)
    _, _, z = fuse(prepare_streams(s))
    a, b = s.exercise[0]
    mid = int((a + b) / 2)
    assert np.isnan(z["hr"][mid]) and np.isnan(z["eda"][mid])


def test_warns_before_events():
    cfg = Config()
    n_events = n_early = missed = 0
    for seed in range(1000, 1006):
        s = simulate_session(n_events=2, seed=seed)
        alerts, *_ = run(s, cfg)
        r = score(alerts, s.events, cfg.horizon_min)
        n_events += len(s.events)
        n_early += len(r["early_lead_min"])
        missed += r["missed"]
        assert all(lead > 3 for lead in r["early_lead_min"])
    assert missed <= 1
    assert n_early >= 0.75 * n_events


def test_quiet_session_without_events_is_quiet():
    s = simulate_session(n_events=0, n_spikes=0, n_exercise=1, seed=11)
    alerts, *_ = run(s)
    assert alerts == []


def test_no_alerts_with_too_few_modalities():
    cfg = dataclasses.replace(Config(), min_modalities=3)
    s = simulate_session(n_events=1, seed=1002)
    streams = prepare_streams(s, cfg)
    index, n_mod, _ = fuse(streams, cfg, use_modalities={"eeg", "fnirs"})
    alerts, _ = detect(index, n_mod, cfg)
    assert alerts == []
