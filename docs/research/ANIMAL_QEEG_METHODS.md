# Quantitative EEG in autism- and psychosis-related animal models

**13 October 2026.** qEEG is quantitative analysis of recorded EEG: spectral density, band powers, phase measures and other explicitly defined descriptors. It is an analysis method, not a separate sensor. Skull EEG, epidural EEG and depth LFP can be analyzed quantitatively, with their acquisition/reference/bandwidth documented separately.

The present target is a **future independently annotated behavioral event** in a model animal. Genotype discrimination, a group-mean EEG difference and a change after drug administration do not themselves establish this forecasting task. No animal behavioral-onset model has been trained or evaluated on real animal events in PSI; fitting tests use synthetic fixtures only.

## Primary studies and model-specific measurements

| Model and primary source | Quantitative measurement | Interpretation boundary |
| --- | --- | --- |
| Fmr1 KO, a fragile-X/autism-related genetic model; [2022 phase-locking study](https://pubmed.ncbi.nlm.nih.gov/35588990/) | Trial-wise phase locking to modulated sounds and non-phase-locked power; abnormalities resemble the corresponding FXS human EEG phenotype | A sensory-processing phenotype, not a demonstrated pre-behavioral-event warning |
| Shank3B KO, [Dhamne et al., 2017](https://doi.org/10.1186/s13229-017-0142-z) | Elevated relative gamma: 30–80 Hz power divided by 1–80 Hz power, replicated in two cohorts | Genotype-associated baseline phenotype. The study also measured seizure susceptibility; neither endpoint supplies the selected behavioral-onset target |
| MK-801 pharmacological model, [Cui et al., 2022](https://doi.org/10.3389/fnins.2022.1001869) | Depth LFP in mPFC/CA1: region/dose-dependent changes including high gamma 60–100 Hz and HFO 150–200 Hz | Post-dose oscillatory effects. An intervention time is not independently measured onset of a behavioral event |
| MK-801 model, [2023 theta/gamma study](https://pubmed.ncbi.nlm.nih.gov/36990270/) | Changes in theta/gamma co-modulation and its relationship to spatial working memory | Co-modulation is not automatically the same estimator as phase-amplitude coupling, PLV or bicoherence; no acute psychiatric-event threshold is inferred |

These models represent selected genetic or pharmacological mechanisms. They do not represent all autism or reproduce the full subjective state of human psychosis. A higher gamma value is not a universal label for sensory overload or psychosis.

## Band power from FFT-based PSD

Supply complete, calibrated samples `x` in microvolts and a verified sampling rate. If native samples are in volts, multiply amplitudes by 1e6 once; power then scales by 1e12. No conversion factor is guessed from waveform size.

The current PSI spectral convention uses Welch: two-second Hann segments, 50% overlap, constant detrending and a one-sided density spectrum. In a four-second epoch this gives three overlapping PSD segments. The frequency grid is 0.5 Hz when the two-second segment contains exactly `2*fs` samples; this grid spacing is not a claim of perfect spectral separation. PSD has units microvolt-squared per Hz.

The estimator and density units are documented in [SciPy's primary Welch reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).

For a band `B=[f1,f2]`, absolute power is:

`P_B = integral from f1 to f2 of PSD(f) df`

`band_integral` inserts exact band edges and uses trapezoidal integration. Output is microvolt-squared. Relative power uses an explicitly named denominator `D`:

`RP_B = P_B / integral over D of PSD(f) df`

For the Shank3 study's reported band definition, `RP_gamma = P_(30–80) / P_(1–80)`. This cannot be compared directly with PSI's human `P_(30–40) / P_(0.5–40)` as if they were the same quantity. For a zero denominator, relative power is unavailable.

The [standalone Python calculation example](../../examples/animal_qeeg_calculation.py) can calculate custom band powers from supplied arrays. It uses PSI's Welch convention. Cui et al. used one-second epochs and 2,048-point FFTs with smoothing; the example does not claim to replicate that entire analysis or its baseline normalization. Its MK-801 band set returns absolute powers only, avoiding an invented relative-power convention.

High-gamma/HFO analysis needs acquisition and anti-alias filters that preserve the requested frequencies. Covering 200 Hz requires `fs > 400 Hz` by Nyquist, with an appropriate acquisition margin; the cited LFP study used 1 kHz. Line-noise harmonics, movement and electrode/reference effects require their own artifact controls. The standalone example checks numerical inputs but does not supply acquisition quality control.

## Baseline change and phase measures

For an explicitly defined pre-event animal baseline, a possible research descriptor is `z_B(t) = (P_B(t)-mean_baseline)/sd_baseline`, or a power change in decibels `10*log10(P_B(t)/mean_baseline)`. These are proposed descriptors, not the article's complete SOI/RDI components or exact reproductions of every source normalization. A zero SD/nonpositive log denominator is invalid. Fit baseline statistics using eligible past/development data only, with state, genotype, treatment and reference documented. The human 24-hour calibration rule is not transplanted to mice.

Across `N` individual stimulus trials, inter-trial phase coherence is:

`ITPC(t,f) = abs((1/N) * sum_j exp(i*phase_j(t,f)))`

It ranges from zero to one and needs trial-resolved complex phases. It is distinct from current PSI's within-epoch alpha PLV between electrode pairs. Completed auditory trials can contribute descriptors before a later behavioral cutoff; responses from the future event cannot be used as predictors. Current PSI does not implement an animal auditory-ITPC pipeline.

The existing mouse analysis instead reproduces processed paired-tone ERP gating: `G = peak_to_peak(S2) / peak_to_peak(S1)` in source-defined windows. This is a dimensionless response ratio. An already averaged ERP cannot reconstruct single-trial induced power or ITPC: `PSD(mean(x_trials))` is generally different from `mean(PSD(x_trials))`. The current processed ERP dataset therefore cannot be relabelled as continuous pre-event qEEG.

## Executed calculation illustration and current code scope

```bash
python -m pip install -e .
python examples/animal_qeeg_calculation.py
```

The example uses **synthetic** 70 Hz and 175 Hz sinusoids at 1 kHz, amplitudes 4 and 3 microvolts. Their analytic powers are `A^2/2`: 8 and 4.5 microvolt-squared. The computed high-gamma and HFO integrals agree within numerical precision; the Shank3 relative gamma is 1 in its 1–80 Hz denominator. The 175 Hz component lies outside that denominator. These numbers verify arithmetic, not a finding in an animal or an event forecast.

The integrated `analyze` pipeline still uses the 0.5–40 Hz profile. The separate [animal workflow](AUTISM_ANIMAL_MODEL.md) now supplies input metadata/QC, power epochs and annotation-dependent behavioral SVM commands. Its behavioral fitting evidence is synthetic only; the real Fmr1 benchmark predicts genotype. No sequence model, automatic clinical warning or clinical threshold is supplied. See [ANIMAL_IEEG_ONSET.md](ANIMAL_IEEG_ONSET.md) for the required behavioral annotations, causal windows, held-out-animal evaluation and separate human transfer validation.

[animal_qeeg_calculation_verification.json](animal_qeeg_calculation_verification.json) records the executed synthetic example, analytical checks, invalid-input checks and actual dependency versions. It is engineering evidence only.
