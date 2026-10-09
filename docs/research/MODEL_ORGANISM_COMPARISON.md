# Model-organism comparison, 9 October 2026

## What was actually analyzed

The public [Fmr1-KO2 sensory-processing dataset](https://doi.org/10.17605/OSF.IO/CVEFK) supplies processed EEG event-related potentials and animal-level gating measurements. Its [institutional description](https://research.rug.nl/en/datasets/unaffected-sensory-processing-in-fmr1-ko-mice/) reports largely unaffected sensory responses and no robust replicated sensory-processing deficit. This is useful counter-evidence to an assumption that an autism-associated genotype must always show impaired gating.

The source README says continuous raw EDFs must be requested from the authors. This work downloaded and verified the public `PairedTone_SenGat_GLM.csv`, 22 processed five-channel ERP arrays, analysis MATLAB script and description file; it did not obtain raw EDFs or independently reproduce their preprocessing. Both the OSF-reported SHA-256 and a pinned CSV SHA-256 were checked. Complete source fingerprints and results are in [model_organism_comparison.json](model_organism_comparison.json).

The CSV has **22 animals**, 10 `wt` and 12 `hem`, each measured at seven stimulus intervals (154 rows). The source code groups WT and hemizygous Fmr1-KO2 animals, averages ERP channels 2 and 3, and extracts a second/first stimulus peak-to-peak response ratio. Larger ratios indicate less suppression. Genotype and paired-tone conditions are preserved; no human psychiatric labels are assigned.

| Interstimulus interval | WT mean S2/S1 | Fmr1-KO2 (`hem`) mean S2/S1 | Difference (`hem` minus WT) |
| --- | --- | --- | --- |
| 100 ms | 0.3322 | 0.3490 | +0.0168 |
| 200 ms | 0.3280 | 0.3782 | +0.0502 |
| 300 ms | 0.4388 | 0.4506 | +0.0118 |
| 400 ms | 0.5058 | 0.5809 | +0.0751 |
| 500 ms | 0.5635 | 0.6280 | +0.0645 |
| 750 ms | 0.7132 | 0.7292 | +0.0160 |
| 1,000 ms | 0.7687 | 0.8277 | +0.0589 |

For an exploratory summary across all seven intervals, the mean of each animal's mean ratio is **0.5214** (WT; between-animal SD 0.0679) and **0.5634** (`hem`; SD 0.0737). This equal-interval average is an additional descriptive summary, not a prespecified endpoint of the source study. No significance test or claim of a reliable genotype effect is made here. The 154 repeated measurements do not constitute 154 independent animals. No meltdown or psychosis-onset event is annotated in these files.

## How this compares with the human data

| Dimension | Human RepOD cohort | Fmr1-KO2 mouse measurements | Consequence |
| --- | --- | --- | --- |
| Label | Schizophrenia diagnosis / healthy control | `wt` / `hem` genotype | Labels answer different questions |
| Acquisition | Resting scalp EEG, 19 channels | Processed stimulus-locked ERPs and gating ratios | Acquisition and preprocessing differ |
| Spectral benchmark | Four-second continuous epochs, 0.5–40 Hz | Already averaged evoked responses | Averaged ERPs cannot replace single-trial EEG spectra |
| Auditory gating | S1/S2 events unavailable | Per-animal S2/S1 ratio available | Direct cross-species gating difference cannot be computed |
| Event onset | No prospective episode event labels | No meltdown / psychosis-onset labels | Event sensitivity, lead time and false alerts cannot be estimated |
| PSI/BIS | Regional scalp coverage approximations; no validated PSI | No BIS electrode correspondence or human calibration | No transferable SOI/RDI score or clinical BIS interpretation |

The implemented result is a measurement-and-evidence comparison plus real mouse-data re-analysis and a within-mouse monitoring-coverage benchmark. It is **not** a cross-species classifier or numerical validation of autistic meltdown versus psychosis onset.

## Expanded monitoring within the model organism

The source provides five recorded ERP channels per animal. The expanded comparison preserves all five channels and uses seven S2/S1 ratios for each channel (35 features). The restricted comparison follows the original analysis by averaging channels 2 and 3 before peak extraction (seven features). Source MATLAB peak windows are reproduced with explicit conversion of one-based inclusive indices to Python slices. The restricted values must reproduce all 154 published CSV ratios within 5e-9 before evaluation can proceed.

Both sets use exactly the same 22 animals and seven stimulus intervals. A fixed linear SVC predicts documented `wt`/`hem` genotype with leave-one-animal-out evaluation; scaling is fitted only to each training fold. There is no tuning, channel selection or pooling of humans and mice. Results are point estimates, not evidence that expanded monitoring improves event prediction. The comparisons vary both channel coverage and feature dimension/averaging; they do not isolate channel count as the only factor. No BIS correspondence is asserted for these animal electrodes.

| Mouse monitoring coverage | Features | Animal balanced accuracy | Confusion matrix (`hem`, `wt`; rows true, columns predicted) |
| --- | --- | --- | --- |
| All five recorded ERP channels | 35 | 40.00% | `[[6, 6], [7, 3]]` |
| Original mean of two channels | 7 | 45.83% | `[[5, 7], [5, 5]]` |

The maximum discrepancy between reproduced and published gating ratios is **4.972e-10**, within the rounding precision of the nine-decimal CSV. Expanded coverage did not improve the point estimate in this fixed exploratory experiment; both estimates are below the binary balanced-accuracy chance reference of 50%. No reliable genotype discrimination or crisis/onset inference follows from these results. This does not establish that more channels are generally worse, and no formal comparison test was performed.

The machine-readable report includes both confusion matrices, held-out animal predictions, source-file hashes and the maximum ratio-reproduction error. This comparison covers **additional EEG/ERP channels only**. Synchronized video, motion, EMG, autonomic measures, raw continuous spectral analysis and independently annotated behavioral crisis/onset events are unavailable in the files used here. No claim of a complete multimodal enhanced-monitoring comparison is made.

## Psychosis-related model evidence

An [awake mouse NMDAR-antagonist study](https://doi.org/10.3389/fnins.2022.1001869) measured mPFC/CA1 intracranial LFP and schizophrenia-related behavioral effects. Its data statement points to article/supplementary results and author inquiries rather than a public raw-LFP archive. No raw data from this study were analyzed here.

The study includes 60–100 Hz high-gamma and 150–200 Hz HFO measurements. They lie outside this project's 0.5–40 Hz feature definition, and above the coverage of 128 Hz VitalDB BIS EEG (64 Hz Nyquist). Injection timing is an experimentally known intervention, not a verified spontaneous human psychosis-onset event. These findings support a mechanism comparison, not a calibrated onset detector.

## Requirements for a direct event comparison

An appropriate existing animal dataset would need continuous EEG/LFP, verified channel/reference metadata, animal IDs, stimulus/intervention timing and an independently defined behavioral transition. A matched human dataset would need independently annotated episode transitions and comparable sensory tasks. Analyses should keep species, genotype, drug context and acquisition separate; aggregate at the individual level and report effect sizes within each species. Shared physical frequency bands can be compared after harmonization without assuming the same functional rhythm, scalp location or psychiatric meaning across species. The existing human baseline and full ten-component SOI/RDI protocol cannot be silently transplanted to mice.

No new animal procedures or author messages were performed by this work.
