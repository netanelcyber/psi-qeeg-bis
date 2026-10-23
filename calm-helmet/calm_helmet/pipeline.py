"""Glue: session -> fNIRS cleaning -> fusion -> alerts."""
from .config import Config
from .detector import detect
from .fnirs import short_separation_clean
from .fusion import fuse


def prepare_streams(session, cfg=None, clean_fnirs=True):
    cfg = cfg or Config()
    streams = dict(session.raw)
    if clean_fnirs:
        streams["fnirs_pfc_hbo"], _ = short_separation_clean(
            session.raw["fnirs_pfc_hbo"], session.short_sep_hbo, cfg.ss_fit_sec)
        streams["fnirs_pfc_hbr"], _ = short_separation_clean(
            session.raw["fnirs_pfc_hbr"], session.short_sep_hbr, cfg.ss_fit_sec)
    return streams


def run(session, cfg=None, use_modalities=None, clean_fnirs=True):
    cfg = cfg or Config()
    streams = prepare_streams(session, cfg, clean_fnirs)
    index, n_mod, z = fuse(streams, cfg, use_modalities)
    alerts, smooth = detect(index, n_mod, cfg)
    return alerts, index, smooth, n_mod
