"""fNIRS / HD-DOT helpers.

Short-separation regression: a source-detector pair ~8 mm apart samples only scalp
and skull; a long channel (~30 mm) samples scalp + cortex. Arousal raises scalp blood
flow, which a long channel would otherwise report as "brain activity". We fit the
scalp coupling on the calm calibration period and subtract it afterwards.

Fitting on calm data only matters: during a build-up the cortical and scalp signals
are correlated (both follow arousal), and a running regression would subtract the
cortical signal too.
"""
import numpy as np


def short_separation_clean(long_ch, short_ch, fit_sec):
    """Return long_ch minus beta * short_ch, with beta fitted on the first fit_sec samples."""
    y = np.asarray(long_ch, float)
    x = np.asarray(short_ch, float)
    seg = slice(0, int(fit_sec))
    ok = np.isfinite(y[seg]) & np.isfinite(x[seg])
    if ok.sum() < 30:
        raise ValueError("not enough clean calibration samples for short-separation fit")
    xs, ys = x[seg][ok], y[seg][ok]
    xc, yc = xs - xs.mean(), ys - ys.mean()
    beta = float((xc @ yc) / (xc @ xc))
    return y - beta * x, beta
