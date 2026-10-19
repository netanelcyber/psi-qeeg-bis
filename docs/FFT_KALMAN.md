# FFT → measurements → covariance → Kalman → CUSUM

This additive research workflow is available through `psi-qeeg track`. It uses
the project's recording reader, quality checks, Welch PSD helper and band
integrator. Existing `analyze`, SOI/RDI definitions and the separate
`meltdown-alert` detector retain their behavior. No notification is sent.

## Reproduce a complete synthetic example

```bash
python -m pip install -e .
python scripts/demo_fft_tracking.py --out outputs/fft_tracking_demo
```

The demo generates four channels at 256 Hz, a 60-second baseline and a later
smooth change of alpha/beta amplitudes. Everything is synthetic. The model
parameters in `examples/fft_tracking/model.json` are illustrative; they are
not fitted psychiatric biomarkers. Output includes `recording.csv`,
`metadata.json`, `model.json`, `tracking.csv`, `tracking.json` and `tracking.png`.

Replay those same files through the ordinary CLI:

```bash
psi-qeeg track outputs/fft_tracking_demo/recording.csv \
  --metadata outputs/fft_tracking_demo/metadata.json \
  --model outputs/fft_tracking_demo/model.json \
  --baseline-sec 60 --frontal F3 F4 --bootstrap-draws 1024 --seed 5 \
  --high-z-limit 6 --out outputs/fft_tracking_replay
```

For real EEG, provide a model whose `subject_id` and `medication_profile_id`
match the recording. If medication context is unknown, explicitly use
`not_recorded`. `loadings` maps selected measurement names to coefficients
in **normalized measurement units**. These coefficients define the research
state's scale and sign; the software does not infer psychiatric meaning.
Estimate/validate them with appropriate labeled data or specify an explicitly
illustrative model. A baseline by itself cannot identify those loadings.

## Spectra and log measurements

Defaults are a two-second analysis window, one-second hop, one-second
periodic Hann segments, 50% Welch segment overlap and no zero padding.
At 256 Hz, these are N=512, L=256, M=3, with 1-Hz bin spacing. The outer
two-second window does not give a 0.5-Hz grid when Welch uses one-second
segments. Segment means are removed before tapering. For real signals the
one-sided density doubles positive-frequency interior bins, leaving DC
and Nyquist undoubled. Units are µV²/Hz.

Band powers use `signal.band_integral`'s exact-endpoint trapezoidal integration,
preserving the existing project's partition convention. This is a documented
quadrature choice instead of a rectangular sum over bin centers. Bands are
delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30 and high 30–45 Hz. The existing
`analyze` gamma band remains 30–40 Hz. The short segments limit low-frequency
resolution; a log transform does not guarantee Gaussian errors.

Measurements are selected by name:

| Name | Calculation |
| --- | --- |
| `F3:log_alpha` (and other channel/band combinations) | ln(max(band power, 10⁻¹² µV²)) |
| `F3:log_beta_alpha` | ln(beta power) − ln(alpha power) |
| `FAA` | ln(right alpha power) − ln(left alpha power) |

The Python extractor defaults to per-channel log(beta/alpha), plus FAA if a
frontal pair is supplied. Other log powers remain accessible. The CLI uses
exactly the measurements named by `model.loadings`. Duplicate or linearly
redundant selections, such as alpha, beta **and** their log ratio together,
are rejected. The replay requires verified reference/scalp metadata for FAA;
unknown BIS channel numbers never imply a left/right frontal pair.

The high band is available as a spectral measurement and an optional quality
gate. It is excluded from the default latent measurement set. Elevated power
can reflect muscle, motion or other sources, so it is not labeled a specific
muscle-tension or psychiatric measurement. `--high-z-limit 6` is an optional
research heuristic, not a clinical cutoff. Supplied SQI and EMG numerics are
also respected by the project's QC, including mandatory SQI for BIS input.

## Computing R from the FFT and a baseline

The approximation Var(ln B) ≈ 1/(K_b M) requires approximately independent,
equally informative bins and independent segments. A Hann taper correlates
bins, overlapping segments reuse samples, spectral shape affects information,
and channels/derived ratios can share substantial noise. Zero padding adds
bins without adding independent data. Consequently, **FFT parameters alone
do not determine a valid full R**. A broader band is not automatically a
more informative measurement of the latent state.

`fit_fft_noise` instead implements a parametric bootstrap conditional on an
estimated multivariate stationary Gaussian baseline process:

1. Fit a Hermitian positive-semidefinite cross-spectrum with long Hann FFT
   segments of the clean contiguous baseline. The synthesis length is at
   least twice the length of a pair of adjacent analysis windows. At least
   `max(4, channels+1)` long segments are required; short/gapped baselines fail.
2. Draw independent complex Gaussian Fourier vectors with that cross-spectrum.
   DC/Nyquist draws are real. A shared multivariate spectral square root retains
   interchannel cross-spectra. Apply inverse FFT and crop each realization to
   an adjacent pair of analysis windows.
3. Run the **same** Welch, band integration, logarithms and feature mapping
   on each pair. This includes tapering, Welch overlap, outer-window overlap,
   unequal spectral shape and derived-feature covariance.
4. Estimate the full covariance Γ₀ (`R_fft`) from the two marginal ensembles
   and Γ₁ = Cov(y[t], y[t−1]) from their pairs. Seed and draw count are saved.
5. Normalize with a fixed personal median and 1.4826 × MAD from actual baseline
   windows. For D = diag(baseline scales), R_normalized = D⁻¹ Γ₀ D⁻¹.

The report also exposes moment-matched effective band degrees of freedom,
ν_eff = 2 E[B]²/Var(B), and the **approximate diagnostic** trigamma(ν_eff/2)
for log variance. It is the log-feature bootstrap covariance, rather than this
gamma approximation, that enters the tracker. A separate overlap-only M_eff
diagnostic uses squared Hann overlap correlations; it is not a complete DOF.

R_fft is conditional on the estimated baseline spectrum and stationary-Gaussian
assumptions. It excludes uncertainty in estimating that spectrum, normalization
parameters and any extra state-model error. Nonstationary oscillations and
artifacts can invalidate it. A caller-declared clean baseline is not evidence
that the person is clinically calm. A changing latent state inside calibration
can also contaminate this estimator-noise interpretation. Validate on separate
sessions; do not add total empirical feature covariance to R_fft, which would
count spectral variability twice. `residual_covariance`, when supplied, must
describe **additional** residual error in normalized units.

## Kalman model and overlap

The per-hop latent model is:

```math
a_t = \phi a_{t-1} + d_{t-1} + \eta_{a,t},\qquad
d_t = d_{t-1} + \eta_{d,t}.
```

Both process variances and φ are explicit model parameters. They are not
inferred from FFT settings. `drift_per_min` reports the additive drift input
60 d_t / hop_s; with φ < 1 it is not the complete derivative of a_t.

Using Γ₀ as independent measurement noise every second would count overlapping
data as independent evidence. We use a stable **AR(1) approximation** for it:

```math
e_t=Ae_{t-1}+u_t,\qquad
A\approx\Gamma_1\Gamma_0^{-1},\qquad
Q_e=\Gamma_0-A\Gamma_0 A^T.
```

The implementation works in normalized coordinates, floors eigenvalues at
10⁻⁸ and caps whitened lag singular values at 0.98. This produces a stationary
PSD noise model despite finite bootstrap error. These are numerical
regularization choices, not physiological noise estimates.

The augmented state is `[a, d, e...]`, observed as y = λ a + e + v. Here the
augmented filter's ordinary measurement covariance is R_residual + 10⁻⁹ I;
the FFT-derived Γ₀ enters through the **noise state**, its initial covariance
and Q_e. This avoids also adding Γ₀ a second time as white measurement noise.
If Γ₁=0 the latent-state update equals the ordinary white-noise Kalman filter
with measurement covariance Γ₀ + R_residual. Cholesky solves and the Joseph
covariance update avoid explicit inversion and preserve numerical stability.

AR(1) does not exactly represent all overlap or physiological autocorrelation.
Inspect held-out normalized innovations, their variance and autocorrelation
before interpreting model intervals or choosing change thresholds. Missing,
flat, amplitude-contaminated or optionally high-band-contaminated windows use
prediction only. CUSUM resets at such gaps. Baseline statistics are frozen;
adapting them would require updating the noise model consistently.

## Innovation CUSUM

Use the **pre-update prediction innovation**, v_t = y_t − H x̂[t|t−1], with
S_t = H P[t|t−1] Hᵀ + R_residual. The observation-noise numerical floor is
included in S_t. A posterior difference between consecutive state estimates
is not that innovation. For an unexpected positive latent-unit shift δ:

```math
\ell_t = \delta\lambda^T S_t^{-1}v_t
          -\frac{\delta^2}{2}\lambda^T S_t^{-1}\lambda,\qquad
C_t=\max(0,C_{t-1}+\ell_t).
```

Outputs include the projected innovation z-score, NIS, CUSUM and `change_flag`.
The flag means the illustrative statistical threshold was crossed; repeated
flagged windows are not counts of distinct events. This monitors unexpected
change relative to the adaptive model, and is not a guaranteed detector of
the modeled slow drift. Thresholds need calibration of false alarms on held-out
sessions and testing on independently labeled events. No lead time, clinical
diagnosis or ability to predict a meltdown/psychotic episode is established.

## Python API and spectrogram access

```python
from psi_qeeg.spectral_tracking import (
    SpectralConfig, SpectralExtractor, fit_fft_noise, KalmanCUSUM,
)

cfg = SpectralConfig(fs=256, channels=("F3", "F4"))
extractor = SpectralExtractor(cfg)
cal = fit_fft_noise(clean_baseline_uv, extractor, draws=1024, seed=5)
tracker = KalmanCUSUM(cal, loadings=[1.0, 1.0])  # Explicit illustrative loadings.
measurement = extractor.extract(window_uv)      # (channels, 512 samples)
result = tracker.step(cal.normalize(measurement.values))
R = cal.R_normalized
psd, frequencies = measurement.psd, extractor.frequencies
```

Stacking successive `measurement.psd` arrays produces a time-indexed **Welch
spectrogram**. It is not exactly a single-window STFT magnitude-squared
spectrogram: every outer window averages multiple internal tapered FFTs.
Available times are window ends; there is no future EEG in monitoring features.
Calibration uses only the declared initial prefix, which is excluded from the
subsequent monitoring trace.

Default spectral extraction is O(channels × M × L log L) per window, plus band
integration. Calibration adds FFT synthesis across bootstrap draws, and the
augmented dense Kalman solve scales cubically in the selected feature count.
Phone performance has not been benchmarked by this implementation.

## Verification and sources

Tests check known sine power and FAA sign, parity-sensitive one-sided density,
band partitions, independent Monte Carlo covariance, interchannel correlation,
overlap dependence, zero-padding invariance, white-noise Kalman equivalence,
known-AR(1) innovation statistics, quality gaps, baseline leakage and CLI output.
These are numerical tests, not clinical validation.

- [SciPy Welch documentation and Welch (1967) reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
- [SciPy spectral analysis guide](https://docs.scipy.org/doc/scipy/tutorial/signal.html#spectral-analysis).
- [SciPy polygamma (including trigamma) definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.polygamma.html).
- [Welch & Bishop, An Introduction to the Kalman Filter](https://techreports.cs.unc.edu/papers/95-041.pdf).
- [NIST CUSUM control charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc313.htm).

The specific multivariate FFT bootstrap, AR(1) approximation and projected
innovation CUSUM combination are the implementation's research design, not
clinical recommendations made by those sources.
