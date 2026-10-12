# Autism-model mouse qEEG: implemented workflows and empirical limits

**13 October 2026.** PSI now has an animal-specific continuous-qEEG extractor and a separate annotation-dependent behavioral SVM. A real public-data **Fmr1-KO2/WT genotype benchmark** was also run. The behavioral-onset implementation was tested with explicitly synthetic fixtures; **no behavioral model was trained on real animal events**.

## Real qEEG model actually fitted

Source: [Neuronal Oscillation Dynamics Fmr1 KO2 mice](https://osf.io/as7cz/), associated with [Kat et al., 2024](https://doi.org/10.1093/cercor/bhae201). The published experiment uses wireless skull EEG and inactive-wakefulness segments. It is not an intracranial recording cohort. The public source root supplied two processed CSVs containing `Pow, DFA, fEI, AnID, GT, Loc, Freq`, with treatment conditions additionally present in experiment 2.

The fixed classifier uses deposited **Pow only**, at two locations (`Front`, `Temp`) and all 99 source frequency labels: **198 features per animal**. Power is log10-transformed; StandardScaler is fitted only on the training experiment, followed by linear SVC (`C=1`, balanced class weights). All animals must have finite positive complete vectors; there is no imputation or outcome-based selection. The CSV's native units/frequency-label mapping are not silently converted to microvolt-squared/Hz. No raw EEG power was recomputed for this benchmark.

Experiment 1 has 20 animals (11 KO/9 WT). Experiment 2 has 23 animals (11 KO/12 WT); only its `Sal` condition is used. Its drug conditions are not additional animals. IDs are disjoint between the two experiments: **43 unique animals**, not 22,176 independent spectral rows. Both source SHA-256s and byte counts were verified, and no animal was excluded from these selected power vectors.

| Evaluation | Training animals | Independent test animals | Animal balanced accuracy | Confusion matrix: KO/WT, rows true / columns predicted |
| --- | ---: | ---: | ---: | --- |
| Primary: experiment 1 → experiment 2 saline | 20 | 23 | 57.95% | `[[10,1],[9,3]]` |
| Supplementary reverse direction | 23 | 20 | 71.72% | `[[6,5],[1,8]]` |

Neither direction is selected as a winning result. These small, single-laboratory cross-experiment point estimates have no uncertainty interval or significance test. **The target is source WT/KO genotype**, not autism diagnosis, a behavioral event or human clinical risk. The real genotype bundles cannot be used by the animal behavioral inference command.

```bash
python -m pip install -e '.[ml]'
python scripts/validate_public_fmr1_qeeg.py
```

The runner pins source file fingerprints, saves independent test predictions, both fitted genotype models and a report. [fmr1_qeeg_genotype_summary.json](fmr1_qeeg_genotype_summary.json) records the aggregate experiment; source and derived-data artifacts accompany the delivered results.

## Continuous animal qEEG implementation

`analyze-animal-qeeg` accepts CSV, or EDF/BDF with the optional MNE reader. Use the [metadata template](../../examples/animal_event_schema/metadata_template.json), replacing its placeholders, nulls and unverified reference. It requires mouse animal/session IDs, model family (`Fmr1`, `Shank3`, `Cntnap2`), genotype as context, signal kind, documented channel locations/reference, explicit units, sample clock, source/license and prespecified numerical quality limits.

Profiles are `shank3_2017_gamma` (30–80 Hz gamma / 1–80 Hz denominator) and `autism_research_1_100` (1–100 Hz delta/theta/alpha/beta/low-gamma/high-gamma power). The latter is a declared exploratory profile, not a published validated event signature; its gamma partitions are 30–80 and 80–100 Hz. All bands must be below Nyquist. Welch uses two-second Hann segments and 50% overlap. Default epochs are four seconds without overlap. Missing, flat and amplitude-limit epochs are rejected without interpolation. Movement, muscle, reference and line-noise artifacts still require independent review. ITPC, complete SOI/RDI, fEI and LRTC are not estimated by this extractor.

```bash
python -m psi_qeeg analyze-animal-qeeg data/mouse.csv \
  --metadata data/mouse_metadata.json --out outputs/mouse
```

Output is `animal_epoch_features.csv` and `animal_qeeg_analysis.json`. This command does not create event labels. It accepts continuous animal signals, not averaged ERPs. Raw-sample features stay within each epoch, so future samples do not change prior epochs. The [calculation guide](ANIMAL_QEEG_METHODS.md) explains the physical powers and denominators.

## Behavioral-onset SVM implementation

For the selected behavioral domain, supply an observable `behavior-name` and operational definition plus independent confirmed onset/offset annotations and fully reviewed follow-up. The [events](../../examples/animal_event_schema/events.csv) and [coverage](../../examples/animal_event_schema/coverage.csv) files are empty schemas, not training data. Labels use `animal_behavioral_event_onset`; their allowed sources are expert/consensus video review or an independent synchronized behavioral measurement. EEG guesses, genotype, injection time, human meltdown and psychosis labels are not substitute outcomes.

At cutoff `t`, a history uses only complete accepted past qEEG epochs. Per-feature mean, sample SD and time slope are inputs. The positive outcome is a confirmed future onset in `[t+min_lead,t+horizon]`. Current/recent events, insufficient lead, signal/history gaps and unreviewed future time are censored. The defaults (60-second history, 30-second horizon, four-second minimum lead) are **illustrative settings**, not demonstrated warning times.

The model is StandardScaler + linear balanced SVC with `C=1`. Evaluation holds out whole animals; scaling is fitted within each training fold. At least three animals and both outcome classes in every training fold are computational prerequisites, not evidence of an adequate clinical or biological sample size. Use one mouse model family/profile per model. Human and animal model bundles are kept distinct; a synthetic-fixture behavioral model is rejected on real animal input.

Example after genuine annotations are supplied (grooming is illustrative; defining abnormal/repetitive grooming requires an independent criterion):

```bash
python -m psi_qeeg prepare-animal-events data/animal_epochs.csv \
  --events data/behavior_events.csv --coverage data/behavior_coverage.csv \
  --features C1__gamma_low_power_uv2 C1__gamma_high_power_uv2 \
  --behavior-name grooming \
  --behavior-definition 'Independently reviewed grooming episode; onset/offset on synchronized video clock' \
  --out outputs/animal_dataset
python -m psi_qeeg train-animal-events outputs/animal_dataset/animal_windows.csv \
  --manifest outputs/animal_dataset/animal_dataset_manifest.json --out outputs/animal_model
python -m psi_qeeg predict-animal-events data/new_animal_epochs.csv \
  --model outputs/animal_model/animal_event_model.joblib --out outputs/animal_inference
```

The existing real Fmr1 CSVs contain genotype/condition features, with no continuous pre-event histories or independently timed behavioral outcomes. They cannot run this behavioral training path. Passing synthetic timing, QC, leakage, model-reload and CLI tests establishes engineering behavior only. Current metrics are correlated-window metrics; event sensitivity, false alerts/hour, lead time and prospective animal validation remain necessary. Inference cannot establish whether an event is already active.

Human transfer and clinical warning thresholds remain unvalidated. LSTM/Transformer is still unimplemented. See [the preclinical protocol](ANIMAL_IEEG_ONSET.md) for interpretation boundaries.
