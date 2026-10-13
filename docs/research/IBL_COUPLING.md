# Executed PLV, coherence and theta–gamma coupling analysis

**13 October 2026; corrected and rerun, exploratory, one mouse.** This extends the [executed public video/LFP example](OPEN_VIDEO_LFP.md), using its unchanged SH023 signal and annotation hashes. The surrogate null now strictly excludes both one-second boundary shifts. PAC estimates and all empirical p/BH-adjusted q values were recomputed. It measures connectivity/coupling around published task wheel movements. It does not establish an abnormal behavioral event, train an event predictor or validate human transfer.

## Data and analysis

The retained signal contains 240 s of calibrated intracranial LFP at 2,500 Hz, original channel indices 32, 96, 128 and 192. The source is [IBL Autism 2025](https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_autism_noel.html), DOI [10.6084/m9.figshare.30024880](https://doi.org/10.6084/m9.figshare.30024880), CC-BY-4.0. Individual genotype and anatomical reference location remain unverified. Cite IBL and Noel et al., [Nature Neuroscience 2025](https://doi.org/10.1038/s41593-025-01965-8).

Sixteen-second windows advance by two seconds. Additional windows end at the five video onsets, start at those onsets, end 60 s before them, or sample an earlier reference. The full-session encoder table groups bouts separated by less than 16 s for descriptive context: four of the five video bouts belong to one cluster. They are not independent repetitions. Three disjoint 16 s controls contain no encoder bout within 2 s on either side; this does not establish absence of body movement or artifact.

There are 113 regular windows, 151 window requests and **143 unique signal windows**. Identical sample intervals reuse an estimate, avoiding duplicated statistical tests. CSV annotations are matched back to the exact deposited encoder timestamps to prevent floating-point rounding from counting the target onset as a preceding bout. Signal samples at or after a window end are excluded. All resampling/filtering uses that supplied window alone; a window ending at onset never uses later signal.

## Calculation

Theta is declared as **4–12 Hz**, gamma as **30–80 Hz**. Each block is antialias-resampled to 500 Hz. Fourth-order Butterworth bandpass filters and Hilbert transforms extract phase and gamma envelope; one second at each edge is discarded, leaving approximately 14 s. These are exploratory analysis choices, not clinically validated bands/windows.

- **PLV:** `abs(mean(exp(1j*(phase_x-phase_y))))`; calculated for theta and gamma in all original channel pairs.
- **Magnitude-squared coherence:** `abs(Pxy)**2/(Pxx*Pyy)`, using two-second Hann Welch segments, 50% overlap and constant detrending. Exact-edge integration divided by band width gives the band mean. [SciPy definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html).
- **Imaginary coherency control:** band mean of `abs(imag(Pxy/sqrt(Pxx*Pyy)))`. It helps assess the zero-lag contribution without proving neural origin; see [Nolte et al., 2004](https://doi.org/10.1016/j.clinph.2004.04.029).
- **Theta–gamma PAC:** [Tort et al., 2010](https://doi.org/10.1152/jn.00106.2010) modulation index: mean gamma amplitude in 18 theta-phase bins is normalized to probabilities `P_j`; `MI=sum(P_j*log(P_j/(1/18)))/log(18)`. Normalized amplitude-weighted mean vector length is supplementary. Local channels and both ordered first/last phase–amplitude combinations are assessed; ordering does not show causal direction.
- **PAC controls:** 199 distinct circular gamma-envelope shifts, excluding circular distances **≤1 s, including both signed boundaries**. At 500 Hz candidates are `501 ... N-501` after edge trimming. Plus-one empirical `p=(1+count(null_MI>=observed_MI))/(199+1)` has resolution 0.005. Benjamini–Hochberg adjustment covers **2,288 unique window/mode/PAC relationships**. Overlapping tests are dependent; BH adjustment is exploratory and does not guarantee FDR under arbitrary dependence. These are coupling-timing tests, not tests of event prediction. A periodic coupling can remain after circular shifting; non-rejection is not proof of absent PAC. See [Dvorak/Fenton, 2014](https://doi.org/10.1016/j.jneumeth.2014.01.002).

Three measurement variants are computed: original reference; notch at both 50/60 Hz (Q=30); and the disjoint bipolar differences **32−96** and **128−192**. Bipolar values describe different spatial derivations and fewer channel pairs; their comparison is a sensitivity check, not an equivalent electrode layout or proof that all removed signal was artifact. The existing scalp alpha-PLV estimator and SOI/RDI scoring are unchanged. These new estimators are a separate research module, not validated PSI index components.

## Results

Medians over the 113 regular windows (pair medians within each window):

| Measure | Original | Notch 50/60 | Bipolar |
| --- | ---: | ---: | ---: |
| Theta PLV | 0.808554 | 0.808559 | 0.168654 |
| Theta coherence | 0.862005 | 0.862005 | 0.107299 |
| Gamma PLV | 0.925273 | 0.703139 | 0.088423 |
| Gamma coherence | 0.665918 | 0.656883 | 0.097523 |
| Local PAC MI, channel median | 0.000262 | 0.000638 | 0.000399 |

The high original synchrony is sensitive to derivation; gamma PLV is also sensitive to filtering around 50/60 Hz. This is compatible with a substantial shared/zero-lag component and frequency-specific contamination, but does not identify their source. Raw theta mean absolute imaginary coherency is only 0.061791, versus magnitude-squared coherence 0.862005; these are different estimators, not an algebraic decomposition.

For the first video movement at **200.382316 s**, the bipolar estimates were:

| Measure | Median of 3 wheel-quiet controls | 16 s before onset | 16 s after onset |
| --- | ---: | ---: | ---: |
| Theta PLV | 0.173514 | 0.171094 | 0.181348 |
| Theta coherence | 0.105278 | 0.081092 | 0.094733 |
| Local PAC MI | 0.000445 | 0.000574 | 0.000593 |

No local PAC relationship passed adjusted `q<0.05` in this first-onset pre/post comparison (corrected smallest bipolar q: **0.241350 before, 0.138069 after**). This is not an event-level significance test or evidence that coupling is universally absent.

The original-reference local MI rises to approximately **0.00404** and **0.00712** in windows ending at the later 222.274316/223.247316 s onsets, with corrected local adjusted q≈**0.027173**. Those windows already contain prior measured movements. Their bipolar local PAC does not pass q<0.05. Interpreting them as an isolated impending-event marker would confound an ongoing behavioral cluster with its later bouts.

Across all windows/relationships, adjusted q<0.05 occurs in **273/858 original**, **190/858 notch**, and **8/572 bipolar** PAC tests. The bipolar eight include **one local** test and **seven cross-channel** tests; several overlap in time. Four cross-channel tests occur in windows around 232–256 s, before a source movement cluster beginning at 260.247316 s. This is a candidate descriptive observation, also lying in wheel-quiet time; it has not been reproduced across independent episodes/animals or tested as a predictor. Surrogate rejection cannot exclude nonsinusoidal waveforms, shared transients, movement or a common reference; see [Aru et al., 2015](https://doi.org/10.1016/j.conb.2014.08.002).

## Scope and reproducibility

One animal, three controls and correlated/overlapping windows do not establish a warning signature, a clinical threshold or a 60-second prediction lead. In the first video's minute-prior analysis there is a measured movement inside the signal window and two further onsets before the selected target. Selecting that target would not demonstrate predicting the next movement 60 s ahead. The original target still requires independently adjudicated abnormal-event labels, artifact review and animal/session-separated validation. No behavioral model, LSTM or human warning system was trained.

```bash
python scripts/analyze_ibl_coupling.py --source outputs/ibl_video_lfp --movement-intervals data/model_organism/ibl_sh023/alf/_ibl_wheelMoves.intervals.d8a0c780-a2b9-4b5b-9c41-d4665f0333db.npy --out outputs/ibl_coupling
python scripts/verify_methodology_corrections.py --coupling outputs/ibl_coupling --human outputs/repod_subject_balanced
python -m unittest discover -s tests -v
```

The runner verifies pinned signal/annotation/encoder hashes and saves every pair, PAC relationship, requested window and quiet comparison, plus the actual surrogate offsets/seeds, phase distributions, time-course figures and a text report. `--surrogates N` sets the count consistently in calculations, the protocol, saved PAC rows and report; the default is 199. The two reviewed alignment-receipt hashes differ only in the source-validator fingerprint; the signal and annotations are identical.

Ten estimator regressions cover uniform/known-modulated amplitude, identity connectivity, shared-reference sensitivity, invalid inputs, BH adjustment, strict boundary exclusion, seeded-null reproducibility and comparison with independent SciPy coherence. The methodology audit also recomputes one real PAC/null using independent binned means and relative entropy and checks all saved PAC/BH values. See [the corrections and evidence](METHODOLOGY_CORRECTIONS.md). This is numerical/engineering validation only. The corrected rejection counts remain 273/190/8; individual p/q values changed and the former inclusive-shift receipt is retained as explicitly historical evidence.
