# Article-to-code alignment

Historical implementation source: Netanel Stern, *Toward a Psychiatric State Index: Differentiating Autistic Meltdown from Psychotic Episode Onset Using Real-Time EEG and AI*, 2026, supplied PDF `PSI_Stern_2026(1).pdf`. Its retained SHA-256 and page count are in [article_source.json](article_source.json); the historical fingerprint is not replaced by that of a later document.

On **12 October 2026**, the code at `23936944186ed92bf7f123ada3b6fb50036585df` was also checked against the [current Google Docs article](https://docs.google.com/document/d/1sH4rkEbuDHridgQbBHvTpPzudxVAtggyEUS-7IO91ao/edit). The source register separately records the retrieved text fingerprint and provider modification time. The current article's section 6.5 identifies its numerical illustrations as assumed inputs, and section 7.2 explicitly describes both stages as proposals, without a trained PSI classifier or evaluated sequence model. The corrections below distinguish the historical PDF wording from the current article. No publication status or clinical validation is inferred.

| Article section | Implementation | Boundary |
| --- | --- | --- |
| 2: BIS analogy and raw EEG | BIS raw waveform importer, separate recorded monitor numerics, electrode documentation | No proprietary BIS recreation or inference of psychiatric state from BIS |
| 6.1: electrodes | Quatro contact specification, unverified VitalDB mapping, explicit expanded scalp specification | Ground/reference are contacts, not signal channels |
| 6.2: SOI/RDI components | Ten named component schema and strict missing-component handling | Event gating, posterior features, CSD and autonomic inputs are not invented from BIS |
| 6.3: personal baseline | Individual mean/SD fitting from labelled calm epochs; 24-hour duration requirement | No normative population diagnosis thresholds |
| 6.4: clinical alert states | Documented as unvalidated proposals | No real-data diagnosis or medication/sedation recommendations are emitted |
| 6.5: normalization | Exact directional clipped SOI mapping and symmetric inverted RDI mapping | Missing values are not filled or silently reweighted |
| 7.1: feature extraction | Welch PSD, experimental squared auto-bicoherence and verified-site alpha PLV | Generic descriptors do not supply all ten index components; microstates, event gating, CSD and autonomic fusion remain unspecified/unimplemented |
| 7.2: stage 1 | Optional linear SVC workflows for supplied state labels (`train`) and diagnosis groups (`train-groups`); a later extension adds independently annotated future-event windows (`train-events`) | No real event-trained model, on-chip implementation or SVM inference in `analyze`; group labels are not episode labels |
| 7.2: stage 2 | Research requirements documented only | No LSTM/Transformer architecture, sequence dataset, learned weights, training loop or inference path in this checkout |
| 8–10: dataset and validation gap | VitalDB engineering example, public diagnosis-group benchmark and validation protocol | Surgical EEG, resting-state diagnosis labels and mouse genotype data do not validate meltdown/psychosis-onset prediction |

## Accurate implementation description

The implemented workbench performs **offline FFT-based qEEG feature extraction with optional linear SVM research training/evaluation**. Describing it as an algorithm currently using `LSTM+FFT` would imply a sequence model that is absent. FFT estimates spectral/bispectral descriptors; it is not itself a psychiatric classifier. The SVM trainers export models separately, while raw-EEG analysis always reports `classification = indeterminate` with unavailable SOI/RDI composites.

The article's nominal four-second state-detection epochs do not make every estimator instantaneous: the analysis pipeline uses 60 seconds of accepted past EEG for auto-bicoherence. No on-device execution, streaming latency, prospective alert performance or long-term trajectory prediction has been measured. A Transformer is a proposed sequence-model alternative, not an implemented recurrent network.

The full-recording extension supplies [human diagnosis-group results](research/EXPANDED_HUMAN_COHORT.md) and an [annotation-dependent future-event workflow](research/EVENT_PREDICTION.md). Its synthetic engineering tests are not real onset-prediction evidence. Literature findings about alpha burst duration/occupancy, visual gamma, source CSD or MMN are [research candidates with estimator gaps](research/PRE_EVENT_SIGNAL_EVIDENCE.md), not proof of the article's alpha-variability hypothesis or imminent-attack thresholds. A [short updated technical account](research/TECHNICAL_UPDATE.md) includes a clarification for correspondence.

The intended preclinical route was clarified on 13 October: [continuous model-animal iEEG/LFP onset prediction](research/ANIMAL_IEEG_ONSET.md), requiring animal-specific events and an adapter. This is not the completed processed mouse ERP/genotype comparison or the current human-target event SVM implementation. No animal onset predictor has been fitted, and human clinical transfer remains unevaluated.

## Exact normalization

For an SOI component whose increase is hypothesized to represent overload:

`s_i = clip(100 * (x_i - mean_i) / (3 * sd_i), 0, 100)`

The direction is reversed for gating efficiency and frontoparietal PLV, matching the worked table: `mean_i - x_i`. RDI uses the article's written formula:

`r_i = clip(100 * (1 - abs(x_i - mean_i) / (3 * sd_i)), 0, 100)`

This makes deviations in **either direction** lower the RDI component. The paper's prose about “deviations below baseline” is narrower than the formula. The implementation follows the formula and documents the distinction.

Both composites use five weights, default 0.20, summing to one. Positive, finite personal calibration SDs are required. Zero SD is an error; no epsilon creates invented sensitivity. Missing components return a missing composite and explicit coverage. A numeric result is a mathematical research composite, not a calibrated probability or proof of reality contact.

## Historical worked-example inconsistencies and current corrections

The historical PDF's five SOI component values all clip at 100, giving SOI 100. It also lists a trajectory beginning at SOI 18 for `t=0`. Those cannot be the same epoch under the written formula. The current Google Docs section 6.5 now explicitly separates the **SOI 100** calculation from the independent 18-to-91 score sketch. The repository arithmetic agrees with that correction; it does not reconstruct the sketch from unavailable measurements.

The historical PDF specifies ORANGE for RDI 40–60 but describes an ORANGE transition near 70 and at 64. The current article identifies 71.6 and 64 as outside that stated ORANGE range and corrects the illustrative interval-rate arithmetic. Several threshold gaps and the meaning of a slow decline remain unspecified. Automatic alerts are still deferred pending a complete research specification and clinical validation; the numerical examples are not model outputs or evidence of clinical benefit.

## Inputs requiring scientific specification

The article gives feature concepts, but it does not completely specify a burst threshold, event-locking protocol, CSD geometry/estimator, bicoherence convention, posterior channel aggregation, autonomic combination or calibration-quality protocol. Generic available band powers are not relabelled as those complete components. The `score` command accepts independently produced, defined values; their validity and unit matching are part of the research protocol.

Medication profile identifiers are checked between current features and baseline. This is context matching, not a pharmacological correction model. Learning medication effects and treatment-related EEG changes requires labelled data and external validation.

Default index weights are 0.20 each; caller-supplied weights must satisfy the schema. No ridge-regression weight fitting is implemented. Alpha PLV is not automatically relabelled as the article's broader alpha/beta PLV or frontotemporal coherence component, and single-channel auto-bicoherence is not the calibrated bifrontal component.

See [TECHNICAL_BRIEF.md](TECHNICAL_BRIEF.md) for the implemented FFT/SVM details and [VALIDATION.md](VALIDATION.md) for the verified tests and prerequisites for future LSTM work.

