"""Event profiles: what the alert is about, how far ahead it aims, and what it tells the carer.

Both profiles track the same thing - a rise in arousal / agitation above the person's own
baseline - because physiological arousal is the best-supported common pathway before both
autistic meltdowns and aggressive incidents on psychiatric wards. They differ in feature
weighting, horizon and guidance. Neither profile distinguishes *which* state is developing;
that is the job of the PSI indices in the parent repository and is not validated.
"""
from __future__ import annotations

from typing import Dict

from .config import Config

PROFILES: Dict[str, dict] = {
    "meltdown": {
        "description": "Autistic meltdown early warning",
        "horizon_min": 10.0,
        "weights": {
            "beta_alpha_ratio": 1.0, "alpha_rel": -1.0, "theta_rel": -0.3,
            "frontal_asymmetry": -0.5, "muscle_tension": 0.7,
        },
        "early_msg": ("Early warning: sustained rise in arousal; {when}. Consider a calm break, "
                      "lowering sensory load, or the person's own regulation strategy now."),
        "imminent_msg": ("Imminent: arousal far above personal baseline. Reduce demands and sensory "
                         "input, offer a quiet space, keep language minimal and calm."),
    },
    "psychosis": {
        "description": "Agitation / aggression risk early warning in acute psychosis (inpatient research use)",
        # Inpatient wristband data show HR and skin conductance rising over ~20 min before
        # aggressive incidents (de Looff et al., Psychophysiology 2019), so the horizon is longer.
        "horizon_min": 20.0,
        "weights": {
            "beta_alpha_ratio": 1.0, "alpha_rel": -0.8, "theta_rel": -0.2,
            "frontal_asymmetry": -0.3, "muscle_tension": 1.0,   # motor agitation weighs more
        },
        "early_msg": ("Early warning: sustained rise in arousal/agitation; {when}. Consider early, "
                      "voluntary de-escalation: lower stimulation, approach calmly, listen, offer "
                      "choices and PRN discussion, and inform the clinical team."),
        "imminent_msg": ("Imminent: very high arousal/agitation. Prioritise everyone's safety using "
                         "trained de-escalation; involve the clinical team. This alert is not a reason "
                         "for coercive measures on its own."),
    },
}


def apply_profile(cfg: Config, name: str) -> Config:
    if name not in PROFILES:
        raise SystemExit(f"Unknown profile {name!r}; choose from {', '.join(PROFILES)}")
    p = PROFILES[name]
    cfg.profile = name
    cfg.horizon_min = p["horizon_min"]
    cfg.weights = dict(p["weights"])
    cfg.early_msg = p["early_msg"]
    cfg.imminent_msg = p["imminent_msg"]
    return cfg
